"""Is this box alive, and is it earning its bill? — the boot/zombie lattice.

Why this module exists
---------------------
Two instruments that answer the same question at two time scales, and one
vocabulary shared by three presentations:

* **The rate instrument** (`BootThroughputSampler`, `parse_pull_progress`,
  `boot_health_watch`, `build_throughput_observer`) needs a poll SERIES. It
  folds docker's per-layer pull counters out of `status_msg` into a
  high-water-mark byte total and condemns a host that cannot feed us
  `BOOT_MIN_MBPS` over a full `BOOT_MBPS_WINDOW_S` window of downloading.
* **The age instrument** (`classify_box_health`, `gather_fleet_health`) is a
  single-SNAPSHOT classifier. It exists because the rate instrument was
  session-tied and its session died while box 45373337 sat ~10 h in `loading`;
  the durable fix is that `ls` — the mandated session-start/end habit — screams
  on its own, with no daemon.
* **`GuardVerdict`** is the vocabulary both produce. Its eight string values are
  a wire contract (below), and until this port the three things that *read*
  them — `herdd ls`, `herdd guard`, `fleetd`'s alarm — each re-derived
  "is this one a zombie / an advisory / what is its short tag" from two
  frozensets and a dict living in a fourth place.

Grouping them here is not tidiness: `classify_box_health` calls
`parse_pull_progress` for its middle loading band (the one question age cannot
answer — is the pull still moving?), so the two instruments already share the
parser, and splitting them would duplicate it.

The GuardVerdict unification — exactly what it absorbs, and what it does not
------------------------------------------------------------------------------
`GuardVerdict` absorbs **set membership and short-tag rendering**, nothing else:
`herdd._GUARD_ZOMBIE_VERDICTS` -> `.is_zombie`,
`herdd._GUARD_ADVISORY_VERDICTS` -> `.is_advisory`,
`herdd._GUARD_VERDICT_SHORT` -> `.short`. Those three had no fleetd twin —
fleetd read the same two frozensets, it never re-derived membership.

It deliberately does NOT absorb either renderer:

* `fleetd.Fleet._health_alarm_msg`'s four-arm phase-aware remedy string stays in
  fleetd (later `fleet/`) and consumes `.is_zombie` / `.is_advisory`. Its
  wording is interpolated with the instance id and is the documented proximate
  fix for box 46682313's destruction; hoisting it into the enum would be a
  behavior change dressed as a refactor.
* the `herdd ls` zombie-scream banner stays in `cli/`. It special-cases
  `BOOTING` and `LOADING_SLOW` by name for a *banner*, which is presentation,
  and its content is disjoint from fleetd's — neither is derivable from the
  other.

FROZEN CONTRACTS this module owes its consumers (all pinned in
`test_vastlib_boxes_health.py`):

1. **The eight verdict STRINGS.** They are serialized into `guard --json`,
   fleetd's `state.json` and its journal (`health_alarm verdict=`), and are
   compared with `==` and `in` in three modules. The module-level `GUARD_*`
   names therefore stay **bare `str`**, exactly as they were, and each is the
   very object the matching `GuardVerdict` member carries as its `.value`. The
   enum is a str subclass so a member `==` its bare string, but nothing in this
   module ever *puts* a member where a string was: `BoxHealth.verdict` is still
   a plain `str`. (A `str`-mixin Enum's `__format__` / `__str__` changed
   between 3.10 and 3.13, and this package must run on both — see
   `GuardVerdict`'s own docstring.)
2. **`BoxHealth`'s field ORDER** `iid verdict reason age_s machine_id evidence`
   and its `._asdict()`, which is the `guard --json` row.
3. **The evidence dict's keys** — read by `fleetd.workload_evidence`,
   `_guard_evidence_bits`, `_zombie_confirm_map` and the dashboard. Every
   classification path returns all of them; absent facts are `None`, never
   missing keys. The set is APPEND-ONLY and its membership is pinned by
   `test_vastlib_boxes_health.py`, not by a count in this sentence — the count
   here read "twelve" against a thirteen-key dict for long enough that nobody
   noticed, which is what a hand-maintained number in prose is worth.

What is deliberately NOT here
-----------------------------
* **No remedy/presentation strings beyond the reasons the classifier itself
  writes.** `_guard_evidence_bits`, `_guard_fix_plan`, `cmd_guard`, the ls
  scream and fleetd's alarm text are all consumers; see above.
* **No policy action.** Nothing here parks, destroys or bids. `guard --fix`'s
  plan (and `parked_lifecycle.zombie_action`, which answers ALARM for
  `ZOMBIE_PYHALF` *by name* even though it is a member of the zombie set) live
  in `boxes/reap.py`. This module says what a box IS, never what to do to it.
* **No second JOBD_STATUS reader.** `fleetd.Hooks.jobd_status_line` is a
  near-identical `rclone cat` of the same object and stays where it is: its own
  docstring argues the independence three ways (unconditional vs fold-bounded
  read, cache staleness, distrust of this module's inference pipeline).
  Deduping it would be a real behavior change on the fail-closed path. One
  reader HERE, three pure parsers on top of it (`_jobd_status_hb_epoch`,
  `jobd_status_pyhalf`, and the epoch/pyhalf soft wrappers) — that is the
  no-drift shape *within* this module, and it is not an argument for merging
  across the module boundary.
* **No `_ts_to_epoch`.** It parses the colon-free runmeta/jobmeta stamp and
  lives in `core.fmt`; `_iso_ftz_to_epoch` below parses the colon-BEARING
  `%FT%TZ` heartbeat stamp jobd writes. Two formats, two parsers, never merged
  (`herdd.py` carries the same warning at `_iso_ftz_to_epoch`).
* **No `_ckpt_watchdog_alarm`** (jobs/risk.py) and no `_revoke_box_keys`
  (boxes/lifecycle.py) — integrator rulings, 2026-08-16.

Provenance: moved from `tools/vast/herdd.py` (plan §8 step 3, 2026-08-16),
behavior-preserving. Bodies are verbatim; the changes are annotations, the
`GuardVerdict` presentation-only unification described above, and the single
sanctioned deviation recorded on `boot_health_watch` (its `get_instance`
default arg). `fleetd.pyhalf_broken` still delegates to `jobd_status_pyhalf`
and is repointed here at step 6, not moved.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import enum
import json
import os
import re
import time
from typing import Any, Callable, Mapping, NamedTuple, Sequence

# Absorbed sibling (plan §3), not Zone S: still a flat file for the add-only
# phase, so it is imported bare-name exactly the way `core/models.py` imports
# `ladder_core`. Becomes `vastlib.images` at step 7 and this line becomes a
# normal package import; `_fleet_image_states` is the only symbol affected.
import imageref

from vastlib.core import api, config, fmt, models
from vastlib.storage import b2

Payload = models.Payload

# --------------------------------------------------------------------------- #
# boot throughput health-check (BOOT_HEALTHCHECK_DESIGN.md, phase P0)
#
# A cheap host shapes transfers per-TCP-flow to ~1-16 MB/s, so a cold multilayer
# image pull can crawl for 40+ min. GPU compute is NOT billed during `loading`
# (invoice-verified 2026-07-20 for two bid boxes and re-verified 2026-08-02 on
# the ON-DEMAND box 46633685: GPU hours 0.000 for a 31-min all-`loading` life;
# only storage + transfer bill) — so condemning a starved box early saves
# SCHEDULE, storage, and bid exposure, not GPU dollars. The GPU bill starts at
# the loading→running flip; see the phase-split note on _BOOT_LOADING_STATES.
# Our registry/B2 sources are multi-gigabit; the variable is the host NIC.
# Sustained aggregate download < BOOT_MIN_MBPS over a FULL BOOT_MBPS_WINDOW_S
# window, during a download phase, ⇒ the host is condemned: destroy + relaunch
# on a different machine. Composes with, never replaces, the fixed
# BOOT_DEADLINE_S backstop.
# --------------------------------------------------------------------------- #
# moved-from: herdd._LAYER_PROG_RE
_LAYER_PROG_RE = re.compile(
    r"^(?P<layer>[0-9a-f]{6,}): (?P<verb>Downloading|Extracting|Verifying"
    r" Checksum|Download complete|Pull complete|Already exists)"
    r"(?:.*?\[.*?\]\s*(?P<cur>[\d.]+)\s*(?P<cu>[kKMG]?B)\s*/\s*"
    r"(?P<tot>[\d.]+)\s*(?P<tu>[kKMG]?B))?")

# Docker renders progress in DECIMAL SI (kB/MB/GB = 1e3/1e6/1e9), matching the
# design's `min_mbps * 1e6` byte floor. Prefix letter -> multiplier; bare "B"=1.
# moved-from: herdd._BOOT_UNIT_MULT
_BOOT_UNIT_MULT: dict[str, float] = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}

# Boot-health knob defaults (_BOOT_KNOB_DEFAULTS) and the CLI > env >
# herdd.yaml > constant resolver (_boot_knob) live in `core.config` (they
# moved out of herdd.py to vastconf.py as I1, 2026-07-30, and vastconf is
# absorbed into core.config by this plan). Every read below is at CALL time.

# actual_status values for the PRE-CONTAINER phase: scheduling, the docker
# image pull, and vast-side container standup. Any other value that isn't
# "running" (missing record, exited, stopped) is 'gone'.
#
# Boot-phase split (owner direction 2026-08-02) — two phases, OPPOSITE cost
# profiles, and the billing boundary is exactly the loading→running flip:
#
#   "loading"   — actual_status loading/created. Vast-side work (pull + box
#                 standup). GPU-UNBILLED: invoice-verified 2026-07-20 (bid
#                 boxes 45064080/45373337) and 2026-08-02 (ON-DEMAND 46633685:
#                 GPU hours 0.000 over a 31-min all-loading life; storage
#                 0.572 h billed). Slow here burns schedule, not GPU dollars —
#                 be patient (progress-gated).
#   "env-setup" — actual_status "running", onstart/jobd bootstrap still
#                 provisioning (jobd never stamped JOBD_STATUS; a serve box's
#                 weight pull). This is the phase WE own and it bills FULL GPU
#                 rate (46636056 reconciliation 2026-08-02: invoiced GPU hours
#                 == observed time-in-running to the minute) — be aggressive
#                 (GUARD_ENVSETUP_DEADLINE_S < GUARD_LOADING_DEADLINE_S).
#
# Measurement limits, established 2026-08-02: the instance record does NOT
# expose the loading→running timestamp or billed runtime (`client_run_time`
# does not track it — 1.1 vs 0.085 invoiced hours on 46636056; `uptime_mins`
# can go negative), so env-setup age from one snapshot is bounded by boot age
# (start_date). status_msg does flip shape at the boundary: docker layer lines
# while pulling → "success, running <image>" once the container is up.
# moved-from: herdd._BOOT_LOADING_STATES
_BOOT_LOADING_STATES = {"loading", "created"}

# >=50% of the window's samples must be in a downloading phase for the
# starvation vote to count — extract-only samples (CPU/disk-bound, not network)
# are progress, not starvation, and must never get a box killed.
# moved-from: herdd._BOOT_DL_MIN_FRAC
_BOOT_DL_MIN_FRAC = 0.5


# moved-from: herdd._to_pull_bytes
def _to_pull_bytes(num: object, unit: str | None) -> int:
    """(number, unit-string e.g. 'MB'/'kB'/'B') -> integer bytes (decimal SI)."""
    try:
        v = float(num)  # type: ignore[arg-type]  # the TypeError is the guard
    except (TypeError, ValueError):
        return 0
    prefix = unit[:-1].upper() if unit and unit.endswith("B") else ""
    return int(v * _BOOT_UNIT_MULT.get(prefix, 1))


# moved-from: herdd.parse_pull_progress
def parse_pull_progress(status_msg: str | None,
                        prev: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fold one `status_msg` snapshot into a per-layer high-water-mark state.

    Returns {"layers": {layer_id: bytes_hwm}, "downloading": bool,
             "extracting": bool, "total_bytes": int}.
    - Downloading lines contribute cur-bytes; 'Download complete'/'Pull
      complete'/'Extracting' freeze the layer at its last known total (or hwm).
    - Byte totals are per-layer MONOTONIC (max(prev, seen)): vast truncates
      status_msg to a tail window, so completed layers scroll out of view — the
      fold carries prev['layers'] forward and never lets total_bytes decrease.
    - 'Already exists' layers contribute 0 (cached — free, not slow).

    `prev` is typed Mapping-or-None rather than as a TypedDict on purpose:
    `boxstate.py` calls this with a bare `{}` and the sampler calls it with its
    own last return, and `prev = prev or {}` has always made the two the same
    thing. A TypedDict here would make the `{}` call site a type error for no
    behavioral gain."""
    prev = prev or {}
    layers: dict[str, int] = dict(prev.get("layers") or {})
    downloading = False
    extracting = False
    for raw in (status_msg or "").splitlines():
        m = _LAYER_PROG_RE.match(raw.strip())
        if not m:
            continue
        layer, verb = m.group("layer"), m.group("verb")
        cur, tot = m.group("cur"), m.group("tot")
        if verb == "Already exists":
            layers.setdefault(layer, 0)                 # cached: 0 contribution
        elif verb == "Downloading":
            downloading = True
            if cur is not None:
                layers[layer] = max(layers.get(layer, 0),
                                    _to_pull_bytes(cur, m.group("cu")))
        elif verb == "Extracting":
            extracting = True
            if tot is not None:                         # freeze at layer total
                layers[layer] = max(layers.get(layer, 0),
                                    _to_pull_bytes(tot, m.group("tu")))
        elif verb in ("Download complete", "Pull complete"):
            if tot is not None:
                layers[layer] = max(layers.get(layer, 0),
                                    _to_pull_bytes(tot, m.group("tu")))
            else:
                layers.setdefault(layer, layers.get(layer, 0))
        # 'Verifying Checksum' carries no bytes and no phase — ignore.
    total = sum(layers.values())
    total = max(int(total), int(prev.get("total_bytes") or 0))   # never decrease
    return {"layers": layers, "downloading": downloading,
            "extracting": extracting, "total_bytes": int(total)}


# moved-from: herdd._get_instance_soft
def _get_instance_soft(iid: object) -> dict[str, Any] | None:
    """Like _get_instance but NEVER raises/exits: returns the instance record
    dict, or None on any API error / gone (404) / empty payload. The boot
    watcher/observers need a poll that degrades to 'no sample' instead of
    sys.exiting mid-loop."""
    ok, d, _ = api.request_soft("GET", f"v0/instances/{iid}/", retries=2)
    if not ok:
        return None
    inst = d.get("instances", d) if isinstance(d, dict) else d
    # The payload is whatever vast sent; the declared dict-or-None is the
    # contract every caller reads, and coercing here would change behavior on a
    # malformed record rather than preserve it.
    return inst or None


# moved-from: herdd.BootThroughputSampler
class BootThroughputSampler:
    """Stateful per-instance boot-pull throughput sampler (BOOT_HEALTHCHECK
    phase P0). Feed it successive vast instance records via `.feed(inst, t)`;
    it folds status_msg byte counters into a per-layer high-water mark, keeps
    (t, total_bytes, downloading) samples, and returns a verdict string or None
    (inconclusive — keep polling). ONE place for the starvation math, shared by
    `boot_health_watch` (blocking poll loop) and the per-tick workflow/babysit/
    supervise observers.

    Verdicts (`.feed` return):
      "running"  — box reached running (healthy exit)
      "slow"     — CONDEMNED: window fully elapsed since first Downloading
                   evidence, AND >=50% of window samples downloading, AND
                   (bytes(t) - bytes(t-window)) / window < min_mbps * 1e6
      "deadline" — deadline_s exceeded without running (fixed backstop)
      "gone"     — instance disappeared / API says terminal
      None       — inconclusive (no downloading evidence yet, window not full,
                   or an extract-only phase)

    `.last_mbps` / `.phase` / `.total_bytes` expose the latest measurement for
    a condemnation message / runmeta event. `start_t` is the boot-clock origin
    for the deadline backstop; the throughput clock starts LATER, at first
    Downloading evidence (`first_dl_t`)."""

    def __init__(self, *, min_mbps: float, window_s: float,
                 deadline_s: float, start_t: float) -> None:
        self.min_mbps = float(min_mbps)
        self.window_s = float(window_s)
        self.deadline_s = float(deadline_s)
        self.start_t = float(start_t)
        self.prog: dict[str, Any] = {"layers": {}, "downloading": False,
                                     "extracting": False, "total_bytes": 0}
        self.samples: list[tuple[float, int, bool]] = []  # (t, total_bytes, dl)
        self.first_dl_t: float | None = None  # first Downloading -> clock start
        self.last_mbps: float | None = None
        self.total_bytes = 0
        self.phase = "waiting"       # waiting|downloading|extracting

    def feed(self, inst: Payload | None, t: float) -> str | None:
        status = (inst.get("actual_status") or "").lower() if inst else None
        if status == "running":
            return "running"
        if status not in _BOOT_LOADING_STATES:
            return "gone"            # missing record or terminal (exited/stopped)
        # Unreachable with a falsy/None `inst`: that case sets status=None,
        # which is not in _BOOT_LOADING_STATES, so the 'gone' return above has
        # already fired. Narrowed with an ignore rather than an `assert` so the
        # body stays byte-identical to the original.
        self.prog = parse_pull_progress(
            inst.get("status_msg") or "", self.prog)  # type: ignore[union-attr]
        dl = bool(self.prog.get("downloading"))
        self.total_bytes = int(self.prog.get("total_bytes") or 0)
        if dl:
            self.phase = "downloading"
            if self.first_dl_t is None:
                self.first_dl_t = t
        elif self.prog.get("extracting"):
            self.phase = "extracting"
        self.samples.append((t, self.total_bytes, dl))
        return self._verdict(t)

    def _verdict(self, now_t: float) -> str | None:
        # Fixed-deadline backstop always applies, even before any download
        # evidence (pull never starts / unparseable status_msg).
        if now_t - self.start_t > self.deadline_s:
            return "deadline"
        # Throughput clock starts at first Downloading evidence, not at launch:
        # scheduling / registry auth / manifest negotiation produce no byte
        # counters and must not count as 0 MB/s.
        if self.first_dl_t is None:
            return None
        # Window must be FULL: no verdict before window_s of downloading-phase.
        if now_t - self.first_dl_t < self.window_s:
            return None
        lo = now_t - self.window_s
        win = [s for s in self.samples if s[0] >= lo]
        if not win:
            return None
        # Extract-only snapshots are progress, not starvation: require >=50% of
        # in-window samples to be downloading before the floor can condemn.
        if sum(1 for s in win if s[2]) < _BOOT_DL_MIN_FRAC * len(win):
            return None
        before = [s for s in self.samples if s[0] <= lo]
        base = before[-1] if before else win[0]         # bytes at window start
        rate = (win[-1][1] - base[1]) / self.window_s    # bytes/s over the window
        self.last_mbps = rate / 1e6
        if rate < self.min_mbps * 1e6:
            return "slow"
        return None


# moved-from: herdd.boot_health_watch
def boot_health_watch(iid: object, *, min_mbps: float, window_s: float,
                      poll_s: float, deadline_s: float,
                      get_instance: Callable[[Any], Payload | None] | None = None,
                      now: Callable[[], float] = time.time,
                      sleep: Callable[[float], object] = time.sleep,
                      sampler: BootThroughputSampler | None = None) -> str:
    """Poll the instance record while actual_status is loading/created and
    return one verdict: "running" | "slow" | "deadline" | "gone" (see
    BootThroughputSampler). Never raises on API errors: a failed poll (the
    `get_instance` call raising, or returning None) contributes NO sample —
    missing samples stretch the window rather than counting as zero throughput
    — while the fixed `deadline_s` still bounds the loop.

    `get_instance`/`now`/`sleep` are injectable for hermetic tests (no network,
    no real sleep). Pass your own `sampler` to read `.last_mbps`/`.phase`/
    `.total_bytes` back after the verdict.

    SANCTIONED DEVIATION (the one in this module). In `herdd.py` the default
    was `get_instance=_get_instance`, a DEF-TIME binding to what is now
    `boxes.lifecycle._get_instance`. Plan §8(b) requires cross-module calls in
    module-attribute form so the `monkeypatch.setattr(module, name, ...)` idiom
    survives the port, and a default argument cannot do that — it freezes the
    function object at import. The default is therefore None, resolved HERE at
    call time via the module attribute. Behavior-preserving: no caller in the
    tree or the suite passes `get_instance=None` explicitly (verified against
    `test_boot_health.py`, `workflowctl.py`, `bid_echo_probe.py`), so the only
    reachable difference is that patching `lifecycle._get_instance` now works,
    which is the point. The import is function-local because `boxes.lifecycle`
    is a same-ring sibling and a module-scope edge would order the two files'
    imports against each other for no benefit."""
    from vastlib.boxes import lifecycle
    get_instance = get_instance or lifecycle._get_instance
    start = now()
    s = sampler if sampler is not None else BootThroughputSampler(
        min_mbps=min_mbps, window_s=window_s, deadline_s=deadline_s, start_t=start)
    while True:
        t = now()
        try:
            inst = get_instance(iid)
        except Exception:
            inst = None
        if inst is None:
            # Failed poll OR a genuinely-gone instance are indistinguishable
            # from a single soft read; treat as 'no sample' but still honor the
            # fixed deadline so the loop can't hang forever on a dead box.
            if t - start > deadline_s:
                return "deadline"
        else:
            verdict = s.feed(inst, t)
            if verdict is not None:
                return verdict
        sleep(poll_s)


# moved-from: herdd.build_throughput_observer
def build_throughput_observer(
        *, get_instance: Callable[[Any], Payload | None] | None = None,
        min_mbps: float | None = None, window_s: float | None = None,
        deadline_s: float | None = None,
        now: Callable[[], float] = time.time,
) -> Callable[[object], dict[str, Any] | None]:
    """Per-tick boot-throughput observer for the workflow/supervise lanes:
    `observer(instance_id) -> None | {"verdict": "slow", "mbps": X,
    "window_s": W, "phase": P, "machine_id": M}`. Holds one BootThroughputSampler
    per instance_id across ticks (the sampler state lives here, not in the
    stateless reconcile_tick). Only ever surfaces the CONDEMN ('slow') verdict —
    'running'/'gone'/'deadline' are already handled by box_observer + the fixed
    boot-deadline action; any other verdict returns None (keep polling). A
    failed poll contributes no sample and returns None."""
    # Each of these three parameters is declared optional (None = "not passed")
    # and rebound to a non-None value immediately; the closure below reads the
    # rebound values, so the bodies stay verbatim and no cast is needed.
    get_instance = get_instance or _get_instance_soft
    min_mbps = (config._boot_knob("BOOT_MIN_MBPS", cli=min_mbps)
                if min_mbps is None else float(min_mbps))
    window_s = (config._boot_knob("BOOT_MBPS_WINDOW_S", cli=window_s, cast=int)
                if window_s is None else int(window_s))
    # deadline backstop for the sampler's own clock: default to the workflow's
    # fixed boot deadline so the per-tick sampler never fires 'deadline' before
    # the workflow's own _boot_deadline_action does.
    dl = float(deadline_s) if deadline_s is not None else 10 ** 9
    samplers: dict[str, BootThroughputSampler] = {}

    def observer(instance_id: object) -> dict[str, Any] | None:
        key = str(instance_id)
        t = now()
        s = samplers.get(key)
        if s is None:
            s = BootThroughputSampler(min_mbps=min_mbps, window_s=window_s,
                                      deadline_s=dl, start_t=t)
            samplers[key] = s
        try:
            inst = get_instance(instance_id)
        except Exception:
            inst = None
        if inst is None:
            return None
        verdict = s.feed(inst, t)
        if verdict != "slow":
            return None
        return {"verdict": "slow", "mbps": s.last_mbps, "window_s": int(window_s),
                "phase": s.phase, "machine_id": inst.get("machine_id")}

    return observer


# --------------------------------------------------------------------------- #
# Durable zombie-sweep — the `herdd ls` scream + `herdd guard` subcommand.
#
# Motivating incident: box 45373337 sat ~10h in vast-side `loading` (jobd never
# came up) because the only watcher was session-tied and its session died. The
# durable fix is that `ls` — the mandated session-start/end habit — itself
# screams when a box is in a zombie shape, so ANY session catches it in minutes
# with no daemon. (Billing, verified 2026-07-20 against the invoice and
# re-verified 2026-08-02 for ON-DEMAND — box 46633685, GPU hours 0.000 over a
# 31-min all-`loading` life: `loading` bills storage only, but a box at
# `running` bills full GPU rate whether or not jobd registers — so
# running-but-dead / env-setup-stuck is the EXPENSIVE shape and loading-stall
# is the SCHEDULE killer.) This is a pure, single-snapshot AGE classifier (a rate needs a poll
# series — that is the P0 BootThroughputSampler above); it reuses the Track-B
# _boot_knob machinery for its deadlines. Condemn = DESTROY, never park: a
# boot-stuck box is bare (park-boxes-not-destroy carves out bare boxes).
# --------------------------------------------------------------------------- #
# `cpu_util` above this = the box is doing something. Lives here, not beside
# either consumer, because BOTH `fleet.rows.workload_evidence` (is this box
# working?) and `boxes.reap._zombie_confirm_map` (is this box making progress?)
# must ask it with the same number, and `boxes.health` is the node below both.
#
# Chosen to be TRUE UNDER EITHER READING of a field whose units vast does not
# document. Measured 2026-08-21 (`cpu_util_calibrate.py`, `calibration/`, 36
# paired samples over 3 boxes): `cpu_util` tracks
# `busy_cores / cpu_cores_effective * 100` — ratios 3.119 and 3.221 against a
# percent-of-slice prediction of 3.125, a 0.2% match on the better box. That is
# PROVISIONAL: the rival hypothesis (cores-busy) differs by exactly the core
# count, and every box sampled had `cpu_cores_effective=32`, so this fleet
# cannot separate percent-of-32 from cores-times-3.125.
#
# So 5.0, which means real work either way — "5% of the slice" (1.6 cores of 32,
# 12 of 246) or "5 cores". It clears the measured CPU-IDLE floor with margin:
# two GPU-serving boxes, whose CPUs were doing 0.4-0.56 cores of nothing much,
# sat at 1.31 and 1.74. An earlier 1.0 here predated the measurement and would
# have fired on both.
#
# The asymmetry says which way to err when in doubt: a false BUSY leaves a box
# unparked and wastes cents until the next signal, a false IDLE parks work
# mid-run and destroys hours. But a floor low enough to fire on everything
# disables the safety net, which is its own failure — hence above the measured
# floor, not below it.
CPU_BUSY_UTIL = 5.0

# moved-from: herdd.GUARD_OK
GUARD_OK = "OK"
# moved-from: herdd.GUARD_BOOTING
GUARD_BOOTING = "BOOTING"
# moved-from: herdd.GUARD_ZOMBIE_LOADING_STALL
GUARD_ZOMBIE_LOADING_STALL = "ZOMBIE_LOADING_STALL"
# moved-from: herdd.GUARD_ZOMBIE_NO_JOBD
GUARD_ZOMBIE_NO_JOBD = "ZOMBIE_NO_JOBD"
# moved-from: herdd.GUARD_ZOMBIE_TICKET_UNCLAIMED
GUARD_ZOMBIE_TICKET_UNCLAIMED = "ZOMBIE_TICKET_UNCLAIMED"
# moved-from: herdd.GUARD_ZOMBIE_PYHALF
GUARD_ZOMBIE_PYHALF = "ZOMBIE_PYHALF"

# moved-from: herdd.GUARD_STALE_IMAGE
GUARD_STALE_IMAGE = "STALE_IMAGE"
# moved-from: herdd.GUARD_LOADING_SLOW
GUARD_LOADING_SLOW = "LOADING_SLOW"


# ZOMBIE_PYHALF is in the zombie set so it SCREAMS everywhere a zombie does (the
# loud `ls` banner, the guard table, exit 2, the reap zombie lane) — but it
# licenses no automatic action: `parked_lifecycle.zombie_action` answers ALARM
# for it by name. That is not timidity, it is the correct remedy. The condition
# is a BUNDLE fault, not a host fault, so the destroy-and-reschedule remedy the
# other zombie verdicts carry would reproduce it on the next host and burn
# BOOT_MAX_HOST_RETRIES doing so. Enforcement already exists and is gentler and
# faster: the box self-parks at JOBD_PY_BROKEN_PARK_S (300 s) and fleetd's
# _pyhalf_tick parks it at FLEETD_PYHALF_CONFIRM_S (600 s).
#
# ADVISORY verdicts (STALE_IMAGE, LOADING_SLOW) are surfaced everywhere a
# verdict is, but deliberately NOT in the zombie set — that set is what `guard
# --fix` acts on, and neither member is a dead box. STALE_IMAGE is a healthy box
# running old code; destroying it would throw away a warm disk over a fixable
# condition (velvet plan P1: measure before enforcing; the refusals are P3).
# LOADING_SLOW (2026-08-03) is a box past the nominal loading deadline whose
# image pull is STILL ADVANCING in this very snapshot — a slow boot, not a dead
# one. It is an advisory precisely so it keeps ALARMING (fleetd alarms on
# advisory verdicts, with a non-destructive remedy) without ever licensing an
# action: the failure it closes is a false ZOMBIE_LOADING_STALL at 27m on a box
# that came up healthy at 40m (46682177), whose twin was destroyed on the same
# evidence.
# moved-from: herdd._GUARD_ZOMBIE_VERDICTS
# moved-from: herdd._GUARD_ADVISORY_VERDICTS
# moved-from: herdd._GUARD_VERDICT_SHORT
class GuardVerdict(str, enum.Enum):
    """The eight box-health verdicts, and the three questions asked about them.

    PRESENTATION-ONLY UNIFICATION (plan §5, integrator ruling 2026-08-16). What
    this absorbs is set membership and the short-tag table — the two frozensets
    and the dict that `herdd` and `fleetd` both reached into. What it does NOT
    absorb is either module's rendering: fleetd's four-arm remedy string stays
    in fleetd (it consumes `.is_zombie` / `.is_advisory`), and the `ls` scream
    banner stays in the CLI. See this module's docstring for why.

    THE STRING VALUES ARE A WIRE CONTRACT and are frozen: they are written into
    `guard --json`, fleetd's `state.json` and its journal, and are compared with
    `==` and `in` in three modules. Each member's `.value` is the *same object*
    as the matching module-level `GUARD_*` constant, so the two can never drift.

    `str` mixin, deliberately, so a member `==` and hashes like its bare string
    and a mixed frozenset cannot silently miss. But nothing in this module puts
    a MEMBER where a string used to be — `BoxHealth.verdict` stays a plain
    `str`, and `classify_box_health` still returns the constants. The reason is
    narrow and measured: a `str`-mixin Enum's `__str__`/`__format__` differ
    between Python 3.10 (the repo floor) and 3.13 (fleetd's release venv), so an
    f-string interpolation of a member is not version-stable, while an `==` and
    a dict lookup are. Consumers holding a bare verdict string go through
    `GuardVerdict.of()`.
    """

    OK = GUARD_OK
    BOOTING = GUARD_BOOTING
    ZOMBIE_LOADING_STALL = GUARD_ZOMBIE_LOADING_STALL
    ZOMBIE_NO_JOBD = GUARD_ZOMBIE_NO_JOBD
    ZOMBIE_TICKET_UNCLAIMED = GUARD_ZOMBIE_TICKET_UNCLAIMED
    ZOMBIE_PYHALF = GUARD_ZOMBIE_PYHALF
    STALE_IMAGE = GUARD_STALE_IMAGE
    LOADING_SLOW = GUARD_LOADING_SLOW

    @classmethod
    def of(cls, value: object) -> GuardVerdict | None:
        """A bare verdict string (as read back out of a `BoxHealth._asdict()`,
        `guard --json` or fleetd's `state.json`) -> the member, or None for a
        value this build does not know.

        None rather than a raise: every caller is reading persisted or
        cross-process data, and an unknown verdict from a newer/older peer must
        degrade to "no opinion", never crash a health tick."""
        try:
            return cls(value)
        except ValueError:
            return None

    @property
    def is_zombie(self) -> bool:
        """Membership of the set `guard --fix` acts on. Note ZOMBIE_PYHALF is a
        member and still licenses no destroy — see the comment above the class;
        the policy lives in `parked_lifecycle.zombie_action`, not here."""
        return self in _ZOMBIE_VERDICTS

    @property
    def is_advisory(self) -> bool:
        """Alarm, never license an action. Disjoint from `is_zombie` by
        construction — `test_vastlib_boxes_health.py` pins the disjointness,
        and `test_fleetd.py` pins the STALE_IMAGE half of it cross-module."""
        return self in _ADVISORY_VERDICTS

    @property
    def short(self) -> str:
        """The six-tag scan-at-a-glance label for the loud `ls` line and the
        guard table. OK and BOOTING have no tag and never did: the original
        dict had no key for them and every caller wrote `.get(v, v)`, so this
        returns the verdict string itself for those two."""
        tag = _VERDICT_SHORT.get(self)
        return tag if tag is not None else str(self.value)


_ZOMBIE_VERDICTS = frozenset({
    GuardVerdict.ZOMBIE_LOADING_STALL, GuardVerdict.ZOMBIE_NO_JOBD,
    GuardVerdict.ZOMBIE_TICKET_UNCLAIMED, GuardVerdict.ZOMBIE_PYHALF})

_ADVISORY_VERDICTS = frozenset({GuardVerdict.STALE_IMAGE,
                                GuardVerdict.LOADING_SLOW})

# short scan-at-a-glance tag per verdict for the loud ls line / guard table.
_VERDICT_SHORT: dict[GuardVerdict, str] = {
    GuardVerdict.ZOMBIE_LOADING_STALL: "loading-stall",
    GuardVerdict.ZOMBIE_NO_JOBD: "jobd-dead",
    GuardVerdict.ZOMBIE_TICKET_UNCLAIMED: "ticket-unclaimed",
    GuardVerdict.ZOMBIE_PYHALF: "pyhalf-broken",
    GuardVerdict.STALE_IMAGE: "stale-image",
    GuardVerdict.LOADING_SLOW: "loading-slow",
}



# The string-side view of the lattice. Every consumer of a verdict reads it back
# out of a `BoxHealth._asdict()`, `guard --json`, fleetd's `state.json` or its
# journal — i.e. holds a bare `str`, never a member — and the three questions it
# asks are exactly the three the absorbed constants answered. These are the
# one-line replacement for `v in _GUARD_ZOMBIE_VERDICTS`, `v in
# _GUARD_ADVISORY_VERDICTS` and `_GUARD_VERDICT_SHORT.get(v, v)`; membership
# still lives in exactly one place (the enum), which is the point of the
# unification. An unknown/None verdict is not a zombie and not an advisory, and
# renders as itself — matching the `.get(v, v)` fallback every caller wrote.

def verdict_is_zombie(value: object) -> bool:
    """Is this verdict string one `guard --fix` acts on?"""
    v = GuardVerdict.of(value)
    return v is not None and v.is_zombie


def verdict_is_advisory(value: object) -> bool:
    """Is this verdict string an alarm that licenses no action?"""
    v = GuardVerdict.of(value)
    return v is not None and v.is_advisory


def verdict_short(value: object) -> str:
    """Scan-at-a-glance tag for a verdict string; the verdict itself when it has
    no tag (OK, BOOTING) or is not one we know."""
    v = GuardVerdict.of(value)
    return v.short if v is not None else str(value)


# A classified box: verdict + human reason + the SALIENT age (seconds) the
# verdict turned on, plus evidence for the guard table / --json. NamedTuple =
# free equality/repr for hermetic tests + JSON-friendly via ._asdict().
#
# FROZEN: the field ORDER is the `guard --json` row order and `._asdict()` is
# the row itself. `typing.NamedTuple` rather than the original
# `collections.namedtuple` only so the fields carry types; both are tuple
# subclasses with the same `_fields`, `_asdict()`, equality against a bare
# tuple and positional construction, which is what every consumer uses.
# moved-from: herdd.BoxHealth
class BoxHealth(NamedTuple):
    #: vast instance id, as the API sent it (int in practice, never coerced).
    iid: Any
    #: One of the eight GUARD_* strings. A bare `str`, not a GuardVerdict —
    #: see the enum's docstring.
    verdict: str
    #: Operator-facing prose. Formatted with `core.fmt._age_str`.
    reason: str
    #: The age the verdict turned on, rounded to whole seconds; None when the
    #: instance record carried no usable `start_date`.
    age_s: int | None
    machine_id: Any
    #: The evidence dict — see this module's docstring, contract 3.
    evidence: dict[str, Any]


# moved-from: herdd._round_age
def _round_age(x: object) -> int | None:
    return int(round(x)) if isinstance(x, (int, float)) else None


# moved-from: herdd._iso_ftz_to_epoch
def _iso_ftz_to_epoch(ts: object) -> float | None:
    """PURE. '%Y-%m-%dT%H:%M:%SZ' (the JOBD_STATUS heartbeat stamp, box-side
    `date -u +%FT%TZ`) -> UTC epoch float, or None. Distinct from
    `core.fmt._ts_to_epoch`, which parses the colon-free runmeta/jobmeta
    `now_ts` form. Two formats, two parsers — never merge them."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return None


# moved-from: herdd._jobd_status_hb_epoch
def _jobd_status_hb_epoch(line: str | None) -> float | None:
    """PURE. Epoch of a JOBD_STATUS line's heartbeat stamp. jobd writes
    '<STATE> <%FT%TZ>[ extra]' (jobd.sh:jobd_status), but the STATE field is
    itself sometimes two tokens ('RUNNING <njobs>') and 'staging=/mbps=' extra
    trails after — so the ts is NOT at a fixed index. Scan for the first token
    that parses as the %FT%TZ stamp. None when no token does."""
    for tok in (line or "").strip().split():
        ep = _iso_ftz_to_epoch(tok)
        if ep is not None:
            return ep
    return None


# moved-from: herdd.jobd_status_pyhalf
def jobd_status_pyhalf(line: str | None) -> bool | None:
    """PURE. Tri-state read of the `pyhalf=` field jobd stamps on the tail of
    every JOBD_STATUS line (jobd.sh:jobd_status, FAILCLOSED_DESIGN §5):
    True = the box SELF-REPORTS its python half dead, False = self-reports ok,
    None = no such field.

    None is the answer for EVERY box running a bundle older than the field, and
    it must never be read as broken — an old bundle is not a sick one. That
    tri-state is the whole reason a `pyhalf` consumer is safe to deploy ahead
    of the boxes: nothing engages until a box is new enough to confess.

    Canonical, on purpose: `fleetd.pyhalf_broken` delegates here, so the alarm
    (this module's health classifier) and the teeth (fleetd's `_pyhalf_tick`)
    can never read the same marker differently. The READERS stay separate — see
    `fleetd.Hooks.jobd_status_line` — because independence is worth having on
    the I/O path; a divergent PARSE is worth nothing."""
    for tok in (line or "").split():
        if tok.startswith("pyhalf="):
            v = tok.split("=", 1)[1].strip().lower()
            if v == "broken":
                return True
            if v == "ok":
                return False
    return None


# moved-from: herdd._jobd_ever_stamped
def _jobd_ever_stamped(iid: object) -> bool | None:
    """Tri-state: has jobd EVER heartbeat on this box? Reads the parent prefix
    of jobs/nodes/<iid>/JOBD_STATUS with `lsf`, because only a LISTING can
    distinguish provable absence from an unreadable read: rc==0 with the marker
    absent proves no jobd session ever ran against this disk (the marker
    persists across park/resume and is stamped even idle); rc!=0 is None —
    unreadable, which the stall sweep must treat as NOT sweepable (I3)."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or iid is None:
        return None
    rc, out, _ = b2._rclone_soft(["lsf", f"b2:{bucket}/jobs/nodes/{iid}/"])
    if rc != 0:
        return None
    return "JOBD_STATUS" in {line.strip() for line in (out or "").splitlines()}


# moved-from: herdd._jobd_status_line_soft
def _jobd_status_line_soft(iid: object) -> str | None:
    """RAW body of jobs/nodes/<iid>/JOBD_STATUS, or None (no bucket / absent
    marker / read failure / empty). Soft: never raises.

    The single B2 read behind every JOBD_STATUS-derived fact in this module —
    `_jobd_status_soft` (the coarse state token), `_jobd_heartbeat_epoch_soft`
    (the heartbeat stamp) and `_jobd_status_pyhalf_soft` (the fail-closed
    confession) all parse THIS. Keeping one reader and three pure parsers is
    what stops the three from drifting onto different notions of what the
    marker says."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or iid is None:
        return None
    rc, out, _ = b2._rclone_soft(
        ["cat", f"b2:{bucket}/jobs/nodes/{iid}/JOBD_STATUS"])
    if rc != 0 or not (out or "").strip():
        return None
    return out


# moved-from: herdd._jobd_status_pyhalf_soft
def _jobd_status_pyhalf_soft(iid: object) -> bool | None:
    """Tri-state `pyhalf` for one box straight off B2: True (self-reported
    broken) / False (self-reported ok) / None (old bundle, absent marker, or an
    unreadable read). Soft.

    The two None arms are deliberately INDISTINGUISHABLE here and both mean
    "no evidence": an unreadable marker is not a bad one (FAILCLOSED_DESIGN §3),
    and a bundle predating the field is not a sick box. Every consumer must
    treat None as "not broken"."""
    return jobd_status_pyhalf(_jobd_status_line_soft(iid))


# moved-from: herdd._jobd_heartbeat_epoch_soft
def _jobd_heartbeat_epoch_soft(iid: object) -> float | None:
    """B2 read of jobs/nodes/<iid>/JOBD_STATUS -> marker epoch (float) or
    None (no bucket / absent marker / read failure / unparseable). Soft: never
    raises — a read-only sweep must not hard-fail.

    READ THE AGE OF THIS WITH CARE. It used to be documented here as "stamped
    every JOBD_HEARTBEAT_S even when the box is IDLE"; that is FALSE and was the
    false half of two ZOMBIE_NO_JOBD false alarms on 2026-08-07. jobd.sh calls
    `status_marker` only on TRANSITIONS — daemon boot, job spawn, job reap, and
    the end of an asset-staging window — so on a box running one long job the
    marker freezes at the spawn stamp. Measured on 47045282 that night: stamped
    04:51:46 at spawn, not touched again until 05:19:16, a 27-minute gap while
    the trainer held 100% GPU and emitted a job heartbeat every 60 s.
    GUARD_JOBD_STALE_S is 600 s, so the gap alone crosses the deadline.

    So this is a WEAK, LAGGING signal and never the strong one. The strong one
    is the periodic jobd-written job event (`_job_liveness_epoch`), which
    `_fleet_jobd_hb_epoch` prefers; this marker is the fallback for a box with
    no folded job at all.

    AMENDED 2026-08-14 (FAILCLOSED_DESIGN §5). Two of the sentences above are
    now dated rather than wrong, and the difference matters:

      * `beacon_tick` re-stamps the marker every JOBD_STATUS_EVERY_S (120 s) on
        a bundle new enough to have it, so on THOSE boxes the transition-only
        lag is gone and staleness means something again. It is still weak
        against an OLD bundle, and the guard cannot tell the two apart, so
        nothing here got teeth.
      * the old closing justification — "its worst case is a park, and jobd
        self-parks an idle box at JOBD_IDLE_PARK_S anyway, so the verdict and
        the daemon's own intent agree" — was FALSIFIED by the idle-park
        inversion of 2026-08-13 (§1.3): a box that could not parse its tickets
        reset LAST_BUSY_TS on every poll and could not park at any deadline.
        The inversion is fixed; the reasoning is not restored, because it was
        never sound. A false zombie is tolerated here because the verdict is
        ADVISORY on this path, not because the box would have parked itself."""
    return _jobd_status_hb_epoch(_jobd_status_line_soft(iid))


# moved-from: herdd._scratch_probe_soft
def _scratch_probe_soft(iid: object) -> dict[str, Any] | None:
    """Latest `scratch_probe` from jobs/nodes/<iid>/events/ (jobd emits one at
    boot), or None. Soft: never raises, and every failure path — no bucket, no
    listing, no probe, an unreadable body — returns None, which
    `disksize.plan_scratch_placement` treats as "no measured facts" and keeps
    scratch on disk. That is the whole safety story: a box we cannot interrogate
    must never end up with a SMALLER allocation than one we can."""
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or iid is None:
        return None
    pfx = f"b2:{bucket}/jobs/nodes/{iid}/events/"
    rc, out, _ = b2._rclone_soft(["lsf", pfx])
    if rc != 0:
        return None
    names = sorted(line.strip() for line in (out or "").splitlines() if line.strip())
    for name in reversed(names):                  # newest first; stop at the first
        rc, body, _ = b2._rclone_soft(["cat", pfx + name])
        if rc != 0 or not (body or "").strip():
            continue
        try:
            ev = json.loads(body)
        except (ValueError, TypeError):
            continue
        if isinstance(ev, dict) and ev.get("event") == "scratch_probe":
            return ev
    return None


# moved-from: herdd._is_jobs_box
def _is_jobs_box(instance: Payload, jobs: Sequence[Payload] | None) -> bool:
    """Is this a jobs-lane (jobd) box, i.e. one that OWES a JOBD_STATUS
    heartbeat? True when it launched with CRED_ROLE=jobs (manual `launch
    --jobs` / workflow box_resolver) OR has any job ticket folded onto it (an
    ssh-`job attach` box carries no extra_env marker but does get tickets)."""
    if models._instance_env(instance).get("CRED_ROLE") == "jobs":
        return True
    return bool(jobs)


def _guard_oldest_submitted_epoch(jobs: Sequence[Payload] | None) -> float | None:
    """Moment the OLDEST still-submitted (queued, never claimed) ticket was
    queued, or None. A submitted-only ticket's last event IS its `submitted`
    event, so last_event_ts dates the queueing."""
    eps = [ep for ep in (fmt._ts_to_epoch(v.get("last_event_ts"))
                         for v in jobs or ()
                         if v.get("display_status") == "submitted")
           if ep is not None]
    return min(eps) if eps else None


# moved-from: herdd._guard_unclaimed_ticket_age
def _guard_unclaimed_ticket_age(jobs: Sequence[Payload] | None,
                                now: float) -> float | None:
    """Oldest still-submitted (queued, never claimed) ticket's age in seconds,
    or None. Raw queue age — it does NOT ask whether the wait is legitimate;
    `_guard_ticket_fifo_blocked` is what decides that, and rule (5) of
    `classify_box_health` needs both."""
    ep = _guard_oldest_submitted_epoch(jobs)
    return None if ep is None else now - ep


def _guard_newest_running_claim_epoch(jobs: Sequence[Payload] | None) -> float | None:
    """Newest moment jobd CLAIMED a job that is STILL RUNNING on this box.

    `started_at` is the fold's min(claimed, started), so it survives a
    preempt/resume relaunch and dates the original claim — which is the one
    FIFO ordered. Returns None when nothing is running; returns None ALSO when
    something is running but carries no datable claim (a fold that old cannot
    place the claim in the order, and the caller treats that as unprovable
    rather than as evidence)."""
    eps = [ep for ep in (fmt._ts_to_epoch(v.get("started_at"))
                         for v in jobs or ()
                         if v.get("display_status") == "running")
           if ep is not None]
    return max(eps) if eps else None


def _guard_newest_running_submit_epoch(jobs: Sequence[Payload] | None) -> float | None:
    """Newest moment a job still RUNNING on this box was SUBMITTED (queued).

    FIFO orders by submit time, so this — not the claim time — is the epoch
    comparable against a queued ticket's. The JOB_ID carries it: `job submit`
    mints `<YYYYmmddTHHMMSS>-<slug>-<hash>` and `_ts_to_epoch` reads the first
    15 chars. `JOB_ID_RE` does not *require* that prefix, so an id we cannot
    date drops out and an all-undatable running set reads as unprovable."""
    eps = [ep for ep in (fmt._ts_to_epoch(v.get("job_id"))
                         for v in jobs or ()
                         if v.get("display_status") == "running")
           if ep is not None]
    return max(eps) if eps else None


def _guard_ticket_fifo_blocked(jobs: Sequence[Payload] | None) -> bool:
    """Is the oldest queued ticket waiting LEGITIMATELY, behind running work?

    jobd's scheduler is strict FIFO and a `needs.gpus: "all"` arm blocks the
    whole queue behind it (JOBS_DESIGN §2 "GPU scheduling") — which is the
    SHAPE WE RECOMMEND for same-model A/B arms, submitted back to back onto one
    box. So "a ticket is unclaimed" is not on its own evidence of anything: on
    the documented happy path it is what a correctly queued second arm looks
    like, and rule (5) fired on it at GUARD_TICKET_DEADLINE_S every time
    (live: box 47976929, 2026-08-17, alarm re-raised every reaper pass for
    ~84 min while the box ran `v13-chain-full-9b-r64-train` at 52%).

    What IS evidence is a claim that jumped the queue: jobd is running a job
    that was SUBMITTED AFTER this ticket, and left the ticket sitting. Under
    strict FIFO that cannot happen — the oldest ticket that does not fit blocks
    younger ones — so running younger work means the ticket was skipped.

    **Both epochs must be SUBMIT times.** The first fix here compared the
    running job's `started_at` (a CLAIM time) against the queued ticket's
    submit time, which is a different clock: on the batch-submit-then-execute-
    serially shape — again the recommended one — every claim after the first
    necessarily post-dates every submit, so the verdict fired from job #2
    onward, forever. Live recurrence: box 47999495, 2026-08-18, ten arms queued
    inside 60 s and executed in perfect FIFO order, flagged at 1 h. The running
    job's submit time comes from its JOB_ID prefix
    (`_guard_newest_running_submit_epoch`).

    Hence: blocked (return True) when something is running and no RUNNING job
    was submitted after the oldest ticket. Both edges deliberately favour silence,
    because this verdict's ONLY remedy is an operator alarm
    (`parked_lifecycle.zombie_action` refuses to touch it — "jobd is ALIVE,
    never auto-touch a functioning box over a claiming bug"), and an alarm that
    cries wolf on the recommended workflow is worth less than no alarm.

    KNOWN BLIND SPOT, accepted: a partly idle multi-card box — one 1-GPU job
    running, six cards free, a 1-GPU ticket queued behind it — reads as blocked
    and is no longer flagged. Fixing that needs `needs.gpus` per ticket, and
    the fold has no such field: `read_job` folds EVENTS, and the `submitted`
    event carries name/entrypoint/timeout/box, never the resolved `needs`. The
    alternative is a permanent false positive on the documented serial shape,
    which is what we had."""
    newest = _guard_newest_running_submit_epoch(jobs)
    if newest is None:
        # Nothing running, or nothing datable. "Nothing running" is the genuine
        # zombie shape (jobd idle AND not claiming) -> not blocked. "Running but
        # undatable" means we cannot place the running work in the queue order,
        # which is unprovable rather than evidence -> blocked (silent).
        return any(v.get("display_status") == "running" for v in jobs or ())
    oldest = _guard_oldest_submitted_epoch(jobs)
    return oldest is None or oldest >= newest


# moved-from: herdd.classify_box_health
def classify_box_health(instance: Payload | None, *,
                        jobs: Sequence[Payload] = (),
                        jobd_hb_epoch: float | None = None,
                        now: float | None = None,
                        jobd_hb_src: str | None = None,
                        image_state: str | None = None,
                        image_reason: str | None = None,
                        jobd_pyhalf: bool | None = None) -> BoxHealth:
    """PURE. Classify ONE vast instance record into a BoxHealth verdict. No I/O:
    the caller supplies the folded `jobs` for this box (from the ls jobs fold),
    for a running jobs box the freshest jobd heartbeat epoch it measured
    (`jobd_hb_epoch`, None = absent/unread), and optionally the image-staleness
    state from `imageref.classify_image_staleness` (velvet P1). Verdicts:

      OK        — running (jobd fresh, or not a jobs box), or a non-live box.
      BOOTING   — a healthy boot in progress, in one of TWO phases with
                  opposite cost profiles (evidence["phase"], 2026-08-02 split):
                  "loading" = loading/created within GUARD_LOADING_DEADLINE_S —
                  the docker pull / vast-side standup, GPU-UNBILLED
                  (invoice-verified, storage only); "env-setup" = running +
                  jobs box whose jobd has not stamped yet, within
                  GUARD_ENVSETUP_DEADLINE_S — the onstart/bootstrap phase WE
                  own, billing FULL GPU rate since the loading→running flip.
      LOADING_SLOW — ADVISORY (never acted on): loading past
                  GUARD_LOADING_DEADLINE_S but `status_msg` shows the docker
                  pull STILL ADVANCING in this snapshot, and the box is inside
                  GUARD_LOADING_HARD_S. A slow host, not a dead boot. Added
                  2026-08-03 after the deadline produced a proven false
                  positive: 46682177 was flagged ZOMBIE_LOADING_STALL at 27m
                  and cleared to OK at 40m, while its co-resident twin on the
                  same image, 46682313, was destroyed on the same evidence 90 s
                  before that. Age cannot tell slow from dead; the pull output
                  can.
      ZOMBIE_LOADING_STALL   — loading/created past the deadline with NO pull
                  activity visible, or past GUARD_LOADING_HARD_S regardless:
                  the schedule-killer shape. GPU-unbilled while stuck, so its
                  licensed remedy is PARK — `parked_lifecycle.zombie_action`
                  never destroys in this phase (2026-08-03 amendment).
      ZOMBIE_PYHALF — running jobs box whose OWN JOBD_STATUS beacon carries
                  `pyhalf=broken` (`jobd_pyhalf=True`): jobd's offline
                  capability selftest failed, so the box can claim no ticket
                  and emit no event (FAILCLOSED_DESIGN §4/§5). Checked BEFORE
                  the two inference rules below because it is evidence of a
                  different kind — a confession, not a deduction from silence —
                  and because the beacon is periodic, so a confessed-broken box
                  keeps its marker FRESH and rule 4 would stay silent while
                  rule 5 named the symptom (an unclaimed ticket) instead of the
                  cause. `jobd_pyhalf` is TRI-STATE and only True acts: None is
                  what every bundle older than the field reports, and an
                  unreadable marker reports it too.
      ZOMBIE_NO_JOBD         — running + jobs-lane box whose JOBD_STATUS
                  heartbeat is stale past GUARD_JOBD_STALE_S, OR (absent) whose
                  box age is past GUARD_ENVSETUP_DEADLINE_S (jobd never
                  stamped — env-setup dead or overlong).
                  The EXPENSIVE shape — full GPU burn with a dead daemon.
      ZOMBIE_TICKET_UNCLAIMED — running jobs box, jobd heartbeat fine, but a
                  submitted ticket has sat unclaimed past GUARD_TICKET_DEADLINE_S
                  (jobd up yet not claiming — a different bug). NOT raised when
                  the wait is the ordinary FIFO one: jobd runs a `gpus: "all"`
                  arm to completion before the next ticket, so a queue behind
                  running work is the recommended shape, not a fault
                  (`_guard_ticket_fifo_blocked`, which carries the live false
                  positive that motivated the rule). `evidence.ticket_age_s`
                  is recorded either way.
      STALE_IMAGE — ADVISORY (never destroyed by `guard --fix`): the box is
                  otherwise healthy but is running an image whose registry tag
                  has since moved, i.e. OLD env, and a park/resume will NOT
                  refresh it. Only ever replaces an OK verdict — a zombie
                  verdict always wins, because that one is destroy-relevant and
                  must never be masked by an advisory. The state lands in
                  `evidence.image_state` REGARDLESS of which verdict won, so a
                  stale zombie is still legible as stale.

    Never flags OK/BOOTING as a zombie: the running-just-now case (jobd not yet
    stamped) is gated on box age past the env-setup deadline, and boot age uses
    the vast `start_date`. NOTE the measurement limit (2026-08-02): the API
    exposes no loading→running timestamp, so env-setup age is bounded by boot
    age FROM LAUNCH — a slow pull consumes the env-setup grace. That is
    deliberate for the manual/alarm layer (the phase bills GPU, so flag early);
    the AUTOMATIC sweep's confirm lane must supply the patience via progress
    evidence (_zombie_confirm_map: download/disk movement resets the clock).
    Phase for non-jobs running boxes is None — with no workload contract the
    API alone cannot distinguish env-setup from up; the run/serve lanes carry
    their own B2 markers (boxstate.py / SERVE_STATUS). `now` is injectable for
    hermetic tests."""
    now = time.time() if now is None else now
    i: Payload = instance or {}
    iid = i.get("id")
    mid = i.get("machine_id")
    status = (i.get("actual_status") or "").lower()
    jobs = list(jobs or [])
    is_jobs = _is_jobs_box(i, jobs)
    try:
        sd = i.get("start_date")
        boot_age = (now - float(sd)) if sd is not None else None
    except (TypeError, ValueError):
        boot_age = None
    ev: dict[str, Any] = {
          "status": status or None, "boot_age_s": _round_age(boot_age),
          "is_jobs_box": is_jobs, "jobd_hb_age_s": None,
          "jobd_hb_src": jobd_hb_src,   # "jobs" (strong) | "jobd-status" (weak)
          # tri-state, from the box's own beacon: True broken / False ok /
          # None unknown (old bundle, unreadable marker, or never read).
          "pyhalf": jobd_pyhalf,
          "ticket_age_s": None,
          # tri-state: True  = the queue age above is the ordinary FIFO wait
          #                    behind running work (rule 5 declined to flag),
          #            False = evaluated and NOT the ordinary wait,
          #            None  = rule 5 was never reached (non-jobs box, or an
          #                    earlier verdict won). Never collapse None to
          #                    False — "not blocked" is a claim about a box we
          #                    actually looked at.
          "ticket_fifo_blocked": None,
          "phase": None,   # "loading" | "env-setup" | "up" | None (unknowable)
          "pull_active": None,   # loading only: status_msg shows live pull
          "pull_bytes": None,    # loading only: bytes seen in this snapshot
          "image_state": image_state, "image_reason": image_reason,
          # Measured CPU work, straight off the instance payload. The fleet's
          # other liveness signals are all GPU- or lane-shaped, so a dedicated
          # CPU box looked idle to every one of them (`workload_evidence`).
          # POSITIVE EVIDENCE ONLY: vast does not always populate `cpu_util`,
          # so None/0.0 means "this signal says nothing", never "idle".
          "cpu_util": i.get("cpu_util"),
          "cpu_cores_effective": i.get("cpu_cores_effective")}

    def mk(verdict: str, reason: str, age: float | None) -> BoxHealth:
        # Single construction point for every return, so the advisory overlay
        # applies uniformly — and only over OK, so a destroy-relevant zombie
        # verdict is never masked by it.
        if verdict == GUARD_OK and image_state == imageref.IMG_STALE:
            return BoxHealth(iid, GUARD_STALE_IMAGE,
                             f"healthy, but running a STALE image — "
                             f"{image_reason or 'registry tag moved since launch'}"
                             f"; destroy + relaunch to pick up the new env "
                             f"(a park/resume will not)",
                             _round_age(age), mid, ev)
        return BoxHealth(iid, verdict, reason, _round_age(age), mid, ev)

    load_dl = config._boot_knob("GUARD_LOADING_DEADLINE_S", cast=int)
    load_hard = config._boot_knob("GUARD_LOADING_HARD_S", cast=int)

    # (1) pre-container: loading/created — the GPU-UNBILLED phase (image pull /
    # vast-side standup; invoice-verified, storage only). A stall here blocks
    # the schedule (the ~10h 45373337 incident). THREE bands, not two
    # (2026-08-03): age alone cannot tell "slow" from "dead", so the middle band
    # asks the one question age cannot — is the pull still moving? `status_msg`
    # carries docker's live per-layer pull output during `loading`, so ONE
    # snapshot answers it: a Downloading/Extracting line means the host is
    # actively feeding us bytes right now. That is the evidence 46682177 had at
    # 27m when it was (falsely) called dead; it came up healthy at 40m.
    if status in _BOOT_LOADING_STATES:
        ev["phase"] = "loading"
        pull = parse_pull_progress(i.get("status_msg") or "", None)
        ev["pull_active"] = bool(pull.get("downloading")
                                 or pull.get("extracting"))
        ev["pull_bytes"] = int(pull.get("total_bytes") or 0)
        if boot_age is not None and boot_age > load_dl:
            if ev["pull_active"] and boot_age <= load_hard:
                # ADVISORY: over the nominal deadline but demonstrably alive.
                # Alarms (so nobody loses sight of it), licenses nothing.
                return mk(GUARD_LOADING_SLOW,
                          f"{status} {fmt._age_str(boot_age)} — PAST the {load_dl}s "
                          f"nominal deadline, but the image pull is STILL "
                          f"ADVANCING ("
                          + ("downloading" if pull.get("downloading")
                             else "extracting")
                          + f", {ev['pull_bytes'] / 1e9:.2f} GB seen). Slow "
                          f"host, not a dead boot; GPU unbilled — let it "
                          f"finish, or park it if you need the schedule "
                          f"(`herdd stop <id>`, recoverable). Hard bound "
                          f"{load_hard}s", boot_age)
            why = ("no pull activity in status_msg"
                   if not ev["pull_active"]
                   else f"past the {load_hard}s hard bound even while pulling")
            return mk(GUARD_ZOMBIE_LOADING_STALL,
                      f"stuck in {status} for {fmt._age_str(boot_age)} "
                      f"(> {load_dl}s deadline; {why}) — boot looks dead. GPU "
                      f"UNBILLED here (storage only), so the remedy is PARK, "
                      f"not destroy: `herdd stop <id>` is recoverable and the "
                      f"idle reaper finishes it in 2h",
                      boot_age)
        seen = fmt._age_str(boot_age) if boot_age is not None else "?"
        return mk(GUARD_BOOTING,
                  f"{status} {seen} (within the {load_dl}s boot deadline; "
                  f"GPU unbilled — image pull / box standup)",
                  boot_age)

    # (2) not live (stopped/exited/gone): not a boot-health concern.
    if status != "running":
        return mk(GUARD_OK, f"{status or 'unknown'} (not a live boot)", boot_age)

    # (3) running but not a jobs-lane box: no jobd expectation -> OK. (Train/
    # serve/manual boxes carry their own supervise watchdog; guard never touches
    # them.)
    if not is_jobs:
        return mk(GUARD_OK, "running (not a jobs-lane box)", boot_age)

    # (3b) THE BOX'S OWN CONFESSION outranks every inference below it. jobd
    # stamps `pyhalf=broken` on JOBD_STATUS only after `jobd.py selftest` — a
    # pure, offline, network-blind import check — failed, so this is not a
    # deduction from silence like rules 4 and 5; it is an admission, with the
    # false-positive rate of a local import.
    #
    # It must be checked FIRST for a mechanical reason as well as a moral one:
    # the beacon is periodic now (FAILCLOSED_DESIGN §5), so a confessed-broken
    # box keeps its JOBD_STATUS marker perfectly fresh. Rule 4 therefore stays
    # silent on it, and the box would be classified either OK or — once its
    # untouched ticket aged out — ZOMBIE_TICKET_UNCLAIMED, which names the
    # symptom and points the next reader at a claiming bug that does not exist.
    #
    # ONLY True acts. `jobd_pyhalf` is None for every box on a bundle older
    # than the field and for every marker we could not read, and both of those
    # mean "no evidence" (FAILCLOSED_DESIGN §3, §8). Reading absence as broken
    # would have flagged the entire in-flight fleet the day the field shipped.
    if jobd_pyhalf is True:
        ev["phase"] = "up"           # it stamped, so the box exists and beacons
        return mk(GUARD_ZOMBIE_PYHALF,
                  "the box's own JOBD_STATUS beacon says pyhalf=broken — "
                  "`jobd.py` cannot import its own modules, so this box can "
                  "claim no ticket and emit no event while it bills. This is a "
                  "BUNDLE fault, not a host fault: do NOT destroy + relaunch, "
                  "it will reproduce on the next host. jobd self-parks at 300s "
                  "and fleetd parks it at 600s; `herdd ssh <id>` and read the "
                  "pyreason= field, then fix the bundle and re-ship",
                  boot_age)

    # (4) running jobs box: jobd heartbeat freshness is the affirmative proof.
    stale = config._boot_knob("GUARD_JOBD_STALE_S", cast=int)
    hb_age = (now - jobd_hb_epoch) if jobd_hb_epoch is not None else None
    ev["jobd_hb_age_s"] = _round_age(hb_age)
    if jobd_hb_epoch is None:
        # jobd never once stamped JOBD_STATUS: the box is in ENV-SETUP — past
        # the loading→running flip, so it bills FULL GPU rate while onstart /
        # the jobd bootstrap provisions. Its own (tighter) deadline: this phase
        # burns money, where a loading stall only burns schedule. Age is from
        # LAUNCH (no loading→running timestamp in the API — see the docstring),
        # so the automatic sweep's progress-gated confirm supplies the patience
        # for slow-pull boxes; a box that reached `running` seconds ago after a
        # fast pull is still NOT flagged.
        ev["phase"] = "env-setup"
        env_dl = config._boot_knob("GUARD_ENVSETUP_DEADLINE_S", cast=int)
        if boot_age is not None and boot_age > env_dl:
            return mk(GUARD_ZOMBIE_NO_JOBD,
                      f"running {fmt._age_str(boot_age)} but jobd never stamped "
                      f"JOBD_STATUS (> {env_dl}s env-setup deadline) — setup "
                      f"dead or overlong while the box bills full GPU; "
                      f"destroy + relaunch", boot_age)
        seen = fmt._age_str(boot_age) if boot_age is not None else "?"
        return mk(GUARD_BOOTING,
                  f"running {seen}, env-setup (onstart/jobd bootstrap) in "
                  f"progress — BILLED at full GPU rate; jobd stamp expected "
                  f"within {env_dl}s of launch", boot_age)
    ev["phase"] = "up"                    # jobd stamped at least once
    if hb_age is not None and hb_age > stale:
        # Name the evidence. "jobs" = a jobd-written job event went quiet, which
        # is the strong reading. "jobd-status" = only the transition-driven
        # JOBD_STATUS marker was available, which lags by design on a box with
        # no folded job — say so rather than assert a death we cannot see.
        if jobd_hb_src == "jobd-status":
            return mk(GUARD_ZOMBIE_NO_JOBD,
                      f"no jobd-written job event for this box; its JOBD_STATUS "
                      f"marker is {fmt._age_str(hb_age)} old (> {stale}s) and that "
                      f"marker only stamps on transitions, so this is the WEAK "
                      f"signal — confirm with `herdd job ls` before "
                      f"destroying", hb_age)
        return mk(GUARD_ZOMBIE_NO_JOBD,
                  f"jobd heartbeat {fmt._age_str(hb_age)} stale (> {stale}s) — "
                  f"daemon dead while the box bills full GPU; destroy + "
                  f"relaunch", hb_age)

    # (5) jobd alive but a ticket sits unclaimed past the deadline — AND the
    # wait is not the ordinary FIFO one. Both facts are recorded either way, so
    # `guard --json` still shows a long queue age on a box we declined to flag.
    tdl = config._boot_knob("GUARD_TICKET_DEADLINE_S", cast=int)
    tage = _guard_unclaimed_ticket_age(jobs, now)
    blocked = _guard_ticket_fifo_blocked(jobs)
    ev["ticket_age_s"] = _round_age(tage)
    ev["ticket_fifo_blocked"] = blocked
    if tage is not None and tage > tdl and not blocked:
        # Name WHICH shape fired: an idle daemon and a jumped queue are
        # different bugs, and the operator's next command differs.
        why = ("jobd claimed a NEWER job after it (FIFO skip)"
               if _guard_newest_running_claim_epoch(jobs) is not None
               else "jobd is running nothing and not claiming")
        return mk(GUARD_ZOMBIE_TICKET_UNCLAIMED,
                  f"a submitted ticket has sat unclaimed {fmt._age_str(tage)} "
                  f"(> {tdl}s) while the box is up — {why}", tage)

    return mk(GUARD_OK, "running; jobd heartbeat fresh", boot_age)


# moved-from: herdd._job_liveness_epoch
def _job_liveness_epoch(view: Payload | None) -> float | None:
    """Newest moment jobd DEMONSTRABLY wrote something for this job, or None.

    Every one of these is a B2 object jobd emitted itself, so any of them is
    affirmative proof the daemon was alive at that instant. Take all three
    rather than `last_heartbeat_ts` alone: the heartbeat loop and the checkpoint
    loop tick independently, `last_event_ts` covers whatever landed most
    recently (including a `started` right after a preempt-resume, when the
    heartbeat clock has just been reset), and a zombie verdict must be reached
    only when NOTHING jobd writes has moved."""
    cand = [e for e in (fmt._ts_to_epoch((view or {}).get(k))
                        for k in ("last_heartbeat_ts", "last_checkpoint_ts",
                                  "last_event_ts")) if e is not None]
    return max(cand) if cand else None


# moved-from: herdd._fleet_jobd_hb_epoch
def _fleet_jobd_hb_epoch(
        iid: object, jobs: Sequence[Payload] | None,
        now: float) -> tuple[float | None, str | None, bool | None]:
    """(epoch, evidence, pyhalf) for a running jobs box: the freshest
    jobd-alive epoch, the EVIDENCE it came from ("jobs" | "jobd-status" | None),
    and the box's tri-state `pyhalf` confession (True broken / False ok / None
    unknown). Bounds new B2 reads — a job event jobd wrote for this box already
    proves the daemon is alive, so when the fold is fresh we SKIP the
    JOBD_STATUS read entirely; only the IDLE-box case (no fresh folded job)
    pays a soft read.

    The asymmetry this encodes: a missed zombie costs dollars per hour and is
    recoverable, a FALSE zombie parks a run that may not be. So the fold takes
    the MAXIMUM over every proof-of-life available and never the minimum, and
    the weak JOBD_STATUS marker (see `_jobd_heartbeat_epoch_soft`) can only
    ever move the epoch FORWARD.

    `pyhalf` inherits that read budget, which makes it a DELIBERATELY partial
    signal on this path: a box whose fold is fresh returns None (unknown), even
    if its beacon is at that moment confessing. Bounded and cheap to reason
    about — the fold can only stay fresh for GUARD_JOBD_STALE_S (600 s) once
    jobd stops writing job events, which a broken python half cannot help but
    do, so the confession surfaces here within that window. It is also exactly
    why fleetd's `_pyhalf_tick` keeps its OWN unconditional reader instead of
    consuming this: the alarm may be lagged and sampled, the thing that ends a
    box's right to bill may not be.

    Two B2 reads of the same object on the slow path (epoch, then pyhalf) — the
    price of leaving `_jobd_heartbeat_epoch_soft` as the stable, widely-stubbed
    seam it already is. Both are soft `rclone cat`s on a path that only an IDLE
    jobs box reaches."""
    stale = config._boot_knob("GUARD_JOBD_STALE_S", cast=int)
    fold = [e for e in (_job_liveness_epoch(v) for v in (jobs or ()))
            if e is not None]
    best = max(fold) if fold else None
    if best is not None and (now - best) <= stale:
        return best, "jobs", None        # jobd provably alive -> no extra read
    hb = _jobd_heartbeat_epoch_soft(iid)
    pyh = _jobd_status_pyhalf_soft(iid)
    cand = [e for e in (best, hb) if e is not None]
    if not cand:
        return None, None, pyh
    # The SOURCE names what evidence existed, not which number won: "jobs" means
    # a folded job carried jobd-written events for this box (so a stale epoch is
    # the strong "the daemon went quiet" reading), "jobd-status" means the weak
    # transition-driven marker was all we had.
    return max(cand), ("jobs" if best is not None else "jobd-status"), pyh


# moved-from: herdd.gather_fleet_health
def gather_fleet_health(instances: Sequence[Payload] | None,
                        jobs_by_box: Mapping[str, Sequence[Payload]] | None,
                        *, now: float | None = None) -> dict[str, dict[str, Any]]:
    """Classify every instance -> {str(iid): BoxHealth._asdict()}. Bounds new B2
    reads to running jobs-lane boxes lacking a fresh in-fold heartbeat; with no
    such box it does ZERO extra network (latency ~0 on a healthy/empty fleet).
    Read-only + soft throughout — never raises.

    The values are DICTS, not BoxHealth tuples: every consumer (fleetd's
    workload rows, the guard table, `ls`, the dashboard) does
    `h.get("verdict")`, and the `_asdict()` here is the `guard --json` row."""
    now = time.time() if now is None else now
    out: dict[str, dict[str, Any]] = {}
    img_states = _fleet_image_states(instances)
    for i in instances or ():
        iid = i.get("id")
        jobs = (jobs_by_box or {}).get(str(iid), [])
        status = (i.get("actual_status") or "").lower()
        hb: float | None = None
        hb_src: str | None = None
        pyh: bool | None = None
        if status == "running" and _is_jobs_box(i, jobs):
            try:
                hb, hb_src, pyh = _fleet_jobd_hb_epoch(iid, jobs, now)
            except Exception:
                # An exception is NOT evidence of a sick box: pyhalf stays None
                # so the fold degrades to "unknown", never to "broken".
                hb, hb_src, pyh = None, None, None
        st, why = img_states.get(str(iid), (None, None))
        h = classify_box_health(i, jobs=jobs, jobd_hb_epoch=hb, now=now,
                                jobd_hb_src=hb_src, jobd_pyhalf=pyh,
                                image_state=st, image_reason=why)
        out[str(iid)] = h._asdict()
    return out


# moved-from: herdd._fleet_image_states
def _fleet_image_states(
        instances: Sequence[Payload] | None,
) -> dict[str, tuple[str | None, str | None]]:
    """{str(iid): (state, reason)} — velvet P1 staleness for every box.

    Resolution goes through `imageref.resolve_tag_digest_ttl` (time-bounded),
    NOT the per-process `_digest_cache`: `gather_fleet_health` runs inside
    fleetd's long-lived health tick, where an unexpiring cache would either
    never notice a real image push or pin a transient failure forever.

    "Which registries are ours" is `imageref.is_our_registry`, never a local
    registry-host read: a local copy is what made this path skip every
    `registry.example.com` box — i.e. the whole fleet — after the R2 cutover.

    Resolves once per DISTINCT image, in parallel, and only for refs that could
    possibly be stale — a box with no digest stamp or on a foreign registry is
    classified with `current_digest=None` and never costs a lookup, because
    `classify_image_staleness` short-circuits both to `not_applicable` before
    the digest is consulted. Soft: any failure leaves the box unclassified
    (None), which reads as no advisory rather than a false alarm."""
    out: dict[str, tuple[str | None, str | None]] = {}
    ins = list(instances or ())
    if not ins:
        return out

    def needs_lookup(i: Payload) -> bool:
        img = models._instance_image(i) or ""
        if not img or "@sha256:" in img:
            return False
        if not models._instance_env(i).get(imageref.IMAGE_DIGEST_ENV):
            return False
        host, _p, _t = imageref._split_image(img)
        return bool(imageref.is_our_registry(host))   # imageref is untyped

    imgs = sorted({models._instance_image(i) for i in ins if needs_lookup(i)})
    digs: dict[str, Any] = {}
    if imgs:
        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(6, len(imgs))) as ex:
                digs = dict(zip(imgs, ex.map(
                    lambda m: imageref.resolve_tag_digest_ttl(m)[0], imgs)))
        except Exception:
            digs = {}
    for i in ins:
        img = models._instance_image(i)
        try:
            out[str(i.get("id"))] = imageref.classify_image_staleness(
                image=img,
                stamped_digest=models._instance_env(i).get(imageref.IMAGE_DIGEST_ENV),
                current_digest=digs.get(img))
        except Exception:
            out[str(i.get("id"))] = (None, None)
    return out
