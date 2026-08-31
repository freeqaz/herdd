"""fleetd's engine — the reconcile tick, the unix-socket `Server`, and `main`.

ONE always-running process owns ALL vast box babysitting. Agents never spawn or
pkill a supervisor again: they register intent (`herdd fleet watch`), suspend
it (`fleet pause`), and request actions (`fleet park/resume/destroy`) over a
local unix socket. Every tick the daemon reconciles the WHOLE fleet, so a box
nobody is watching is an alarm condition with a safety net, not a silent $/hr
leak.

Design split: FLEETD_DESIGN.md owns the process model + control surface;
SUPERVISE_DESIGN.md remains the policy spec. The per-tick policy code is
IMPORTED from the supervise ring (`supervise_tick` / `job_supervise_tick` — the
same functions the legacy inline loops run), never re-implemented here.

Safety invariants (all tested in test_fleetd.py):
  * the safety net is EVIDENCE-GATED (review B1): an unwatched box that shows
    workload liveness (booting, fresh jobd heartbeat, run:/serve: label) is
    AUTO-ADOPTED + alarmed, never parked; only an unwatched box with NO
    evidence, past the grace window, is PARKED — never destroyed;
  * operator intent wins (review B2): a `herdd stop` on a watched box makes
    the watch DORMANT, so the jobs ladder's "bid box stopped -> OUTBID ->
    rescue" default can never resurrect a box a human just parked;
  * a budget breach PARKS + alarms; fleetd never destroys to enforce a cap;
  * the CEILING is durable and outlives the watch that armed it: it survives
    stop/resume/preempt/lapse, INHERITS to successor boxes (eviction
    replacement, handoff understudy) carrying spend-to-date, and what is
    enforced is REMAINING HEADROOM — never the original figure again. An
    auto-adoption with nothing to inherit gets a conservative PROVISIONAL
    default cap, never None; an unreadable ceiling means that default too.
    See the ceiling-ledger block below for the three-path defect it closes;
  * every fleetd-originated park re-labels the box `:keep` so `herdd reap`
    (which auto-destroys idle STOPPED boxes past 2h) honors the park (B4);
  * destroy happens ONLY on an explicit `fleet destroy --yes`, is journaled
    with requester + reason + condition snapshot, needs its deferred condition
    to hold on two consecutive ticks AND for DESTROY_CONFIRM_S, and executes
    at most once;
  * a pause suspends ACTIONS only — observation and budget accrual continue,
    a breach during a pause alarms immediately and parks at expiry — and is
    always bounded, so a crashed agent leaves a box that rejoins by itself.

fleetd is LAYER 2. The box-side watchdogs (MAX_HOURS self-park, jobd self-park
on drain, ephemeral-key TTLs) remain mandatory: this workstation can be asleep.

Subcommands (the argparse tree lives in `main`, at the bottom of this file):
  serve         run the daemon (socket thread + reconcile thread).
                FLEETD_DRY_RUN=1 makes every mutating action a logged no-op.
  install-unit  generate ~/.config/systemd/user/vast-fleetd.service AT RUNTIME
                (absolute paths never enter git), enable it, and enable-linger.
                Points at THIS checkout — bootstrap/soak only; see `deploy`.
  deploy        THE deploy path — release checkout, re-point, restart, verify.
                Its 441 lines live in `vastlib.fleet.deploy`; `main` only
                dispatches to them.
  status        one-shot dump of the persisted state (no daemon needed).

What is deliberately NOT here
-----------------------------
* **Persistence.** `state.json`'s schema, the journal writer and the
  single-instance lock are `vastlib.fleet.state`'s — this module owns WHEN to
  save, never the wire shape. (The four persistence methods below are still
  written out here because the split landed in the same wave; they read every
  filename and version constant from `state`, so the literals have exactly one
  home. Collapsing the bodies into `state` is an integration step, not a
  behavior change.)
* **Row building.** Every pure fold over the state document — `stray_rows`,
  `reconcile_rows`, `retention_rows`/`retention_alarms`, `ceiling_rows`,
  `watch_box_iid`, `workload_evidence`, `normalize_ceiling` — is
  `vastlib.fleet.rows`'.
* **The I/O seam.** `Hooks` is `vastlib.fleet.hooks`'. This module touches the
  vast API, B2 and the supervise lanes ONLY through `self.hooks`, which is what
  makes the tick testable with a fake transport.
* **The client half of the protocol.** `fleet_request`, the socket path and the
  wire version constant are `vastlib.fleet.client`'s, so the daemon and the CLI
  read ONE version literal instead of two that can silently diverge.
* **Policy.** What a health verdict means is `boxes.health`; what a box costs
  is `core.models`; what to do about an eviction is `supervise/`. fleetd
  decides only *when to look* and *what to persist*.
* **Deploy.** `vastlib.fleet.deploy` — the release checkout, its audit, and the
  systemd unit text.

Provenance: moved from `tools/vast/fleetd.py` (rev ea8360dc) by plan §8 step 5.
Behavior-preserving: the swallowed-exception boundaries, the alarm latch-vs-
derive decisions, the ~50 journal event names and every `state.json` key are
consumed schema and are reproduced verbatim.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
from typing import Any, cast

import notify

from vastlib.boxes import health
from vastlib.core import acctfault, api, config, machine_ledger, models
from vastlib.fleet import client, deploy, rows
from vastlib.fleet import hooks as hooks_mod
from vastlib.fleet import state as fleet_state
from vastlib.supervise import journal as sup_journal
from vastlib.supervise import serve_ident

import bidpolicy

# --------------------------------------------------------------------------- #
# daemon tunables
# --------------------------------------------------------------------------- #
# `VERSION`, `STATE_NAME`, `JOURNAL_NAME`, `JOURNAL_MAX_BYTES` and `LOCK_NAME`
# are NOT re-declared here: they are the on-disk contract and live in
# `vastlib.fleet.state`. `UNIT_NAME` is the systemd contract and lives (as
# `FLEET_UNIT_NAME`) in `vastlib.fleet.client`, which the CLI reads too — the
# daemon used to hold a second literal of each, which is how two copies of one
# contract drift.
TICK_S = 45.0                     # FLEETD_DESIGN §3: reconcile every 45s ± jitter
TICK_JITTER_S = 7.0
# Jitter is a FRACTION of whatever interval the unit passes, not an absolute:
# ±7s is 16% of the 45s default and 47% of a 15s tick, and at that width the
# jitter, not the interval, decides when eviction is noticed. Pinned so the
# default is bit-for-bit 7.0.
TICK_JITTER_FRAC = TICK_JITTER_S / TICK_S
UNWATCHED_GRACE_S = 1800.0        # 30 min of OBSERVED life, then park a no-evidence stray
#                                   (CHEAP tier — see unwatched_grace_for_dph)
UNWATCHED_GRACE_EXPENSIVE_S = 300.0  # EXPENSIVE tier: 5 min (owner ruling 2026-07-29)
EXPENSIVE_DPH_USD = 2.0            # >= this $/hr is "expensive" (owner ruling 2026-07-29)
MAX_PAUSE_S = 6 * 3600            # a pause is ALWAYS bounded (fail-safe)
MAX_OBS_DT_S = 300.0              # N7: clocks advance on observation, capped
HEALTH_EVERY_S = 180.0            # gather_fleet_health cadence (it does B2 reads).
#                                   A DURATION, not a tick count: the B2 read
#                                   rate is a property of wall-clock, so it must
#                                   not triple when the tick shortens. 180s = the
#                                   4-tick cadence this replaced, at TICK_S=45.
# `BOOT_EVIDENCE_S`, `JOBD_FRESH_S` and `EXEMPT_LABEL_TOKENS` are NOT here: they
# are read only by `rows.workload_evidence` / `rows.label_exempt` and live with
# them. A second copy beside their only consumer is how a threshold drifts.
PYHALF_CONFIRM_S = 600.0          # a self-reported broken python half must persist
#                                   this long before fleetd parks (FAILCLOSED_DESIGN §8)
DESTROY_CONFIRM_OBS = 2           # a deferred condition must hold on two
#                                   consecutive observations (S3)...
DESTROY_CONFIRM_S = 90.0          # ...AND for this long. The observation count
#                                   alone is a debounce against a racy reading;
#                                   the duration is the one that has to survive a
#                                   tick-rate change, since what makes a parked
#                                   box safe to destroy is that it STAYED parked.
DESTROY_TTL_S = 24 * 3600         # un-executed destroy requests expire (S3)
GONE_CONFIRM_TICKS = 2            # an IID watch dies after this many missing ticks
REPLACEMENT_WEDGE_MIN_S = 150.0   # ...AND this long. 5 refusals span ~180s at
#                                   the 45s default (152s at max jitter), so
#                                   this is the same alarm in wall-clock terms
#                                   at any tick rate.
REPLACEMENT_WEDGE_REFUSALS = 5    # consecutive autonomous-replacement refusals
#                                   before the lane is called WEDGED and alarms.
#                                   ~5 ticks / ~4 min: past a transient market
#                                   blip, well inside the 26 min it took a human
#                                   to notice on 2026-08-24. The existing rails
#                                   cannot see this shape — `BOOT_MAX_HOST_RETRIES`
#                                   counts SUCCESSFUL relaunches, so a lane that
#                                   can never launch never trips it, and the
#                                   `rescue_stalled` alarm needs `unrecoverable`,
#                                   which a pull-condemned box never reaches.
# 2026-07-30: jobs verdicts that are terminal for the INLINE CLI but NOT for a
# daemon watch. `job supervise` exits on `queue_empty` ("submit first, then
# supervise") because a human ran it by hand; a fleetd `jobs` watch is
# registered at LAUNCH time, minutes before the wave submits, so honoring that
# exit dropped the watch — and the stray sweep re-adopted the box as unbudgeted
# `bare` in the same tick, silently disarming the outbid rescue ladder.
JOBS_TRANSIENT_VERDICTS = ("queue_empty",)
# 2026-08-14: the verdicts a STANDING jobs watch (`fleet watch --standing`)
# SURVIVES instead of ending — the two shapes of "this queue drained", told
# apart only by who noticed the park first (fleetd's own ladder, or jobd's
# self-park on drain). Nothing else is standing: `budget` is the enforcement
# seam, `operator_park` is a human saying stop, and `unrecoverable` /
# `instance_gone` already have their own keep-or-reap paths. See
# `_standing_drain` and FLEETD_DESIGN §4a.
STANDING_KEEP_VERDICTS = ("drained", "self_parked")
PROFILES = ("run", "jobs", "serve", "bare")
# The profiles that drive a ladder and move money (bid defend/rescue, relaunch,
# eviction replacement). `bare` is observation + cap only.
POLICY_PROFILES = ("run", "jobs", "serve")
DESTROY_WHEN = ("now", "drained", "parked")
PARKED_STATES = {"stopped", "exited", "offline"}
CEILING_HISTORY_MAX = 24          # per-ceiling epoch log, bounded
ADOPT_CAP_CACHE_S = 30.0          # config re-read cadence for the adopt default
# A ceiling whose `cap_usd` is missing/None/garbage/non-positive/NaN is NOT
# "unlimited" — it resolves to the auto-adopt provisional default and says so
# in the journal. See `rows.normalize_ceiling`.
CEILING_SOURCES = ("explicit", "default", "degraded")

# argparse defaults the profile ticks read via plain attribute access. A missing
# key would otherwise surface as None through _Policy.__getattr__ and silently
# disable a policy (getattr(a, "handoff", True) never sees its default when
# __getattr__ answers), so every non-None default is seeded explicitly. S6: the
# received policy dict is OVERLAID on these — never trusted to be complete.
_COMMON_DEFAULTS: dict[str, Any] = {"dry_run": False, "handoff": True,
                                    "strict_ceiling": False, "max_bid": None,
                                    "budget": None, "no_fleet": True}
RUN_POLICY_DEFAULTS: dict[str, Any] = dict(
    _COMMON_DEFAULTS, interval=45, max_relaunch=3,
    wall_budget=48 * 3600, disk=config.DISK_DEFAULT_FLEETD_GB,
    runtype="ssh_direct",
    num_gpus=1, rescue_wait=900, boot_health=False,
    defend_at=None, price=None, image=None, onstart=None,
    env=None)
JOBS_POLICY_DEFAULTS: dict[str, Any] = dict(
    _COMMON_DEFAULTS, keep=False, rescue_wait=None,
    wall_budget=None,
    # SAFE-OFF (2026-08-08 22:17Z incident; the switch, its reasoning and its
    # override live in ONE place, the jobs-handoff config key — do not restate
    # the policy here). `make_policy` resolves it per watch so a config edit
    # takes effect on the next daemon start without a code change.
    handoff=False,
    # ...and when it IS turned back on, fleetd CAN now carry a migration to
    # `complete`: `_tick_watch` no longer ends a jobs watch while a handoff
    # holds a live understudy, which is the tick `complete` needed and never
    # got (defect #61). This is the driver's assertion of that, and the pure
    # core refuses to arm without it.
    handoff_can_complete=True,
    handoff_unsafe_ignore_preconditions=False,
    # automatic eviction replacement (owner directive 2026-08-05). None = "not
    # set by this watch", which the ladder reads as the bidpolicy default via
    # `supervise.replacement._job_replacement_knob` — NEVER as 0, which would
    # silently disarm replacement entirely.
    max_replacements=None, replace_ceiling_mult=None,
    # verified-hosts-only for a replacement rental (2026-08-16). None = unset ->
    # True via `_job_replacement_verified`, which also consults
    # JOB_REPLACEMENT_VERIFIED in the env and herdd.yaml. "0" is the widening
    # choice.
    replacement_verified=None,
    # the re-bid ladder on outbid (autobid audit 2026-08-08). Same None
    # convention: unset -> the bidpolicy default via `_rebid_knob`, which also
    # consults JOB_REBID_* env and herdd.yaml. 0 rungs disables the ladder and
    # restores the single-rescue-then-replace behavior.
    rebid_max_rungs=None, rebid_step=None,
    rebid_wait_s=None, rebid_ceiling_mult=None,
    # retention of the EVICTED box (owner directive 2026-08-05). None = unset ->
    # bidpolicy's 3h; 0 is a real choice (destroy immediately) and must not be
    # confused with it.
    replacement_retention_hours=None,
    # instance->instance disk salvage of the EVICTED box (owner directive
    # 2026-08-05). True by default: it enters NO GPU contract on either side, so
    # the only cost is bandwidth on ~1 GB. `salvage=None` here would read as
    # False through the store_false flag's default, so the ON default is seeded
    # EXPLICITLY — the same trap the comment above `handoff` names.
    salvage=True, salvage_keep_n=None, salvage_max_gb=None,
    # WHAT a `serve` watch is supposed to be serving (P3, 2026-08-24). Both
    # None on every watch that did not ask: the identity tick returns before
    # any B2 read when `expect_ident` is falsy, so a legacy watch — and every
    # `jobs` watch, which has no marker to read — is untouched.
    # `artifact` is the registry slug and is telemetry only; `expect_ident` is
    # the grade-A sha12 the check is actually made against, verified against
    # the committed registry at REGISTRATION time (cli/fleet/watch.py) so the
    # daemon can never be handed a pin nobody could satisfy.
    artifact=None, expect_ident=None)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
# moved-from: fleetd.dry_run_enabled
def dry_run_enabled() -> bool:
    """`FLEETD_DRY_RUN=1` makes every mutating action a logged no-op.

    Read here and in `fleet.deploy._dry_run_enabled` (which bakes the
    `Environment=` line into the unit) and in `fleet.hooks`, which is the seam
    that actually obeys it. `test_vastlib_fleet_deploy.py` pins this predicate
    and the deploy-side one EQUAL in both states, so the unit can never claim a
    dry run the daemon disagrees with."""
    return os.environ.get("FLEETD_DRY_RUN") == "1"


# moved-from: fleetd._env_float
def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


# moved-from: fleetd.unwatched_grace_s
def unwatched_grace_s() -> float:
    """FLEETD_UNWATCHED_GRACE_S overrides; <= 0 = alarm only, never auto-park
    (the escape hatch for a workstation whose fleet is deliberately hand-run).
    This is the CHEAP-tier grace — see `unwatched_grace_for_dph` for the
    price-aware fuse (owner ruling 2026-07-29)."""
    return _env_float("FLEETD_UNWATCHED_GRACE_S", UNWATCHED_GRACE_S)


# moved-from: fleetd.unwatched_grace_expensive_s
def unwatched_grace_expensive_s() -> float:
    """FLEETD_UNWATCHED_GRACE_EXPENSIVE_S overrides; <= 0 = alarm only, never
    auto-park, same escape-hatch semantics as the cheap tier."""
    return _env_float("FLEETD_UNWATCHED_GRACE_EXPENSIVE_S",
                      UNWATCHED_GRACE_EXPENSIVE_S)


# moved-from: fleetd.expensive_dph_threshold
def expensive_dph_threshold() -> float:
    """FLEETD_EXPENSIVE_DPH overrides — the owner's "more than that really
    needs to be managed properly" line (2026-07-29)."""
    return _env_float("FLEETD_EXPENSIVE_DPH", EXPENSIVE_DPH_USD)


# moved-from: fleetd.unwatched_grace_for_dph
def unwatched_grace_for_dph(dph: float | None) -> tuple[float, str]:
    """Price-aware unwatched-grace fuse (owner ruling 2026-07-29, verbatim:
    "Anything <$2/hour is pretty cheap. Anything more than that really needs
    to be managed properly."): more grace for cheap boxes, less for expensive
    ones. `dph` is the SAME field the budget-accrual path reads
    (`models._num_dph(inst.get("dph_total"))`) — never a second price source.

    Missing/unparseable dph fails toward the EXPENSIVE (short) fuse: the
    evidence gate (`rows.workload_evidence`) already protects a genuinely busy
    box regardless of tier, so the safe default for an unreadable price is the
    strict one, not the lenient one.

    Boundary: dph == threshold reads EXPENSIVE — "more than that" needs
    management, and `>=` keeps the strict side of the line safe.

    Returns (grace_s, tier) with tier in {"cheap", "expensive"} for the
    journal."""
    if dph is None or dph >= expensive_dph_threshold():
        return unwatched_grace_expensive_s(), "expensive"
    return unwatched_grace_s(), "cheap"


def keep_stamp_needed(drained: bool | None,
                      results_present: bool | None) -> bool:
    """PURE. Should a GRADED park stamp the reap keep-token? (item 1,
    `FLEET_REVIEW_2026-08-20.md`.) The token is unconditional — it exempts the
    box from the 2h idle reaper forever, at $2.13–$4.62/day of allocated disk —
    so it is worth skipping only for a box that demonstrably holds nothing:
    every ticket terminal AND every finished ticket's results on B2.

    FAIL-OPEN TO KEEP: either evidence unknown (None) stamps. An unknown must
    never be the thing that leads to a later destroy, and the pair is asymmetric
    by design — `drained` is True on an empty queue where `results_present` is
    None, so a box that never ran anything still keeps."""
    return not (drained is True and results_present is True)


# moved-from: fleetd.notify_enabled
def notify_enabled() -> bool:
    """`FLEETD_NOTIFY=0` turns the per-tick notification poll off entirely.

    The inbox is a HIDDEN endpoint (NOTIFY_DESIGN §1.3) whose behavior we do not
    control, and the rows are evidence only (D2) — so there is a switch that
    stops the extra GET without a redeploy, and turning it off can never change
    a reconcile outcome. Default ON."""
    return os.environ.get("FLEETD_NOTIFY", "1") not in ("0", "false", "no")


# moved-from: fleetd.notify_policy_enabled
def notify_policy_enabled() -> bool:
    """`FLEETD_NOTIFY_POLICY=1` lets matched outbid rows reach the eviction
    classifier and the rescue quote (NOTIFY_DESIGN S2b). **Default OFF.**

    Off is not caution about the code; it is the deploy gate written down. §6.6
    makes a three-lane adversarial review — precedence, races, money-path rails
    — a BLOCKER on S2b, and this daemon runs from a git checkout, so a merge
    plus any restart would otherwise arm a money-path change that the review has
    not seen. With the switch off, `notify_rows` never reaches the ladder, the
    classifier's `notify` argument is None on every path, and fleetd is byte-for-
    byte its S2a self — which is also exactly what happens if the hidden inbox
    is retired (D2), so the OFF state is a state we already have to be correct
    in.

    Turning it on is one env line in the unit, after the review, with `fleet
    report`'s `notify_outbid_matched` rows as the calibration read (§6.3)."""
    return os.environ.get("FLEETD_NOTIFY_POLICY", "0") in ("1", "true", "yes")


# moved-from: fleetd.pyhalf_confirm_s
def pyhalf_confirm_s() -> float:
    """FLEETD_PYHALF_CONFIRM_S overrides; <= 0 disables the enforcement and
    leaves the condition as an alarm only (the escape hatch).

    Sized as a BACKSTOP, not a first responder. The box declares its own python
    half broken within a second of boot and parks itself at
    JOBD_PY_BROKEN_PARK_S (300 s, see FAILCLOSED_DESIGN §4); fleetd only has to
    act for a box too wedged to do that — a dead curl to the vast API, a daemon
    that died after stamping. So this window sits BEYOND the box's own, and the
    health cache's up-to-4-tick (~3 min) lag is slack inside it rather than a
    race against it."""
    return _env_float("FLEETD_PYHALF_CONFIRM_S", PYHALF_CONFIRM_S)


# moved-from: fleetd.pyhalf_broken
def pyhalf_broken(line: str | None) -> bool | None:
    """PURE. Tri-state read of the `pyhalf=` field jobd stamps on JOBD_STATUS:
    True (self-reported broken), False (self-reported ok), None (no such field).

    None is the answer for EVERY box running a bundle older than the field, and
    it must never be read as broken — an old bundle is not a sick one. That is
    what makes this safe to deploy ahead of the boxes: the teeth simply do not
    engage until a box is new enough to confess.

    Delegates to `boxes.health.jobd_status_pyhalf`, which is the canonical parse
    (2026-08-14). That module grew its own reader of this field when
    `classify_box_health` learned the ZOMBIE_PYHALF verdict, and two
    hand-written parses of one marker is how the ALARM and the TEETH end up
    disagreeing about what a box said — the exact class of defect this campaign
    exists to close. The READERS stay separate on purpose; see
    `hooks.Hooks.jobd_status_line`."""
    return health.jobd_status_pyhalf(line)


# moved-from: fleetd.global_budget_usd
def global_budget_usd() -> float | None:
    v = os.environ.get("FLEETD_GLOBAL_BUDGET_USD")
    try:
        return float(v) if v else None
    except ValueError:
        return None


# `_HERE` was `tools/vast` in the flat module, and `repo_root` was two dirnames
# above it. THIS file sits three levels deeper (`tools/vast/vastlib/fleet/`), so
# the same repo root is FIVE dirnames up — recomputed, not copy-pasted, and
# hoisted to a module constant for the same reason `core.config._HERE` and
# `boxes.ssh._REPO_ROOT` are constants: the depth is a property of the module's
# location, not of the function.
#
# Getting it wrong is SILENT, and it has already happened once here: on
# 2026-08-09 `repo_root` was one dirname too few, which put the root at
# `<repo>/tools` and cost two things without a single error — the generated unit
# got `WorkingDirectory=<repo>/tools` (and a doubled
# `Documentation=<repo>/tools/tools/...`), and `_env_stat()` stat'ed
# `<repo>/tools/.env`, a path that never exists, so `_maybe_reload_env` compared
# None to None forever and the N5 hot-reload never fired once.
#
# `test_vastlib_fleet_daemon.py::test_repo_root_matches_flat_fleetd_computation`
# pins this against the flat file's own expression, and
# `::test_repo_root_contains_fleetd_script` pins the property the deploy path
# actually needs.
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(_TOOLS_VAST_DIR))
# The Zone E launcher, which is what `ExecStart=` must name — `__file__` was
# `tools/vast/fleetd.py` in the flat module and `cmd_install_unit` baked it
# straight into the unit. Baking THIS file instead would install a unit that
# execs a package module and crash-loops on RestartSec=5.
_FLEETD_SCRIPT = os.path.join(_TOOLS_VAST_DIR, "fleetd.py")


# moved-from: fleetd.repo_root
def repo_root() -> str:
    """The REPO ROOT of the checkout this daemon is running from."""
    return _REPO_ROOT


# moved-from: fleetd.git_rev
def git_rev() -> str | None:
    """Short rev of the checkout the daemon is running FROM (S6 skew warning)."""
    try:
        p = subprocess.run(["git", "-C", repo_root(), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() or None
    except Exception:
        return None


# moved-from: fleetd._Policy
class _Policy(argparse.Namespace):
    """Namespace whose MISSING attributes read as None instead of raising, so a
    partial policy dict (old client, or a state.json watch written before a new
    flag existed) can never AttributeError a daemon tick (S6).

    The compat device this implements is the other half of `fleet.state`'s
    frozen schema: a watch record persisted by an older rev carries an older
    policy dict, and it must still tick."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401  (argparse namespace)
        if name.startswith("__"):
            raise AttributeError(name)
        return None


# moved-from: fleetd._redirect_policy
def _redirect_policy(existing: dict[str, Any] | None,
                     incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Policy for a REDIRECTED `Fleet.watch` (see its docstring for why this is
    a merge and the same-key upsert is a wholesale replace).

    Base = the policy the watch is already running; the caller's keys win,
    except where the caller's value is `None` and the base has a real one —
    argparse sends `None` for "not given", and reading that as "clear it" is how
    raising a budget would silently disarm `salvage` or `max_replacements` on a
    live ladder. `False` wins, so `--no-handoff` / `--no-salvage` still work."""
    base = dict(existing or {})
    for k, v in (incoming or {}).items():
        if v is None and base.get(k) is not None:
            continue
        base[k] = v
    return base


# moved-from: fleetd.make_policy
def make_policy(profile: str, policy: dict[str, Any] | None, target: str,
                budget_usd: float | None = None, iid: object = None) -> _Policy:
    """Rebuild the argparse namespace the tick functions expect: OUR defaults
    first, the received dict overlaid on top.

    `iid` (non-run profiles) is the box the ladder is CURRENTLY on — see
    `rows.watch_box_iid`. It outranks both `policy["id"]` and the watch key,
    because both of those record where the watch started, and a rebuilt ladder
    pointed at a replaced box supervises a box that no longer exists."""
    base = dict(RUN_POLICY_DEFAULTS if profile == "run" else JOBS_POLICY_DEFAULTS)
    base.update({k: v for k, v in (policy or {}).items()
                 if k not in ("func", "jobfunc")})
    base["no_fleet"] = True                      # never re-delegate to ourselves
    if profile != "run" and base.get("handoff"):
        # A stored watch policy from before 2026-08-08 carries `handoff: True`
        # (it was the default), and a daemon restart would resurrect it from
        # state.json — so the SAFE-OFF switch is re-applied here, on every
        # rebuild, rather than only at the defaults. The config key is the one
        # place that decides; a watch cannot opt itself back in.
        base["handoff"] = config.jobs_handoff_enabled()
    # serve = the jobs ladder minus queue semantics (serve_mode): same
    # defend/rescue bid policy, no drained/queue_empty exits, no jobd reattach.
    base["serve_mode"] = profile == "serve"
    # 2026-07-30: ONE cap, not two. `a.budget` is what the ladder spends against
    # (and what the handoff decision reads as `budget_usd`); the watch's
    # `budget_usd` is what fleetd enforces. A caller that set only the latter
    # must not leave the ladder uncapped.
    if base.get("budget") is None and budget_usd is not None:
        base["budget"] = budget_usd
    if profile == "run":
        base["run_id"] = base.get("run_id") or target.split(":", 1)[-1]
    else:
        chosen = iid or base.get("id") or target
        try:
            base["id"] = int(chosen)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            base["id"] = chosen
    return _Policy(**base)


# --------------------------------------------------------------------------- #
# serve identity alarms (P3, 2026-08-24)
#
# DERIVED, never latched: every one of them is recomputed from
# `w["serve_identity"]`, which the serve ladder rewrites on every tick, so each
# retracts itself the moment the condition stops being true — a relaunch onto
# the right artifact, or an operator dropping the pin. A latched copy would
# still be lit after the fix, which is the failure the derived/latched split
# exists to prevent (see `alarm_records`).
#
# EVERY MESSAGE ENDS IN A COMMAND. A refusal that does not say what to run next
# is a refusal an operator cannot act on, and the mismatch alarm in particular
# has a clock on it: a parked box is a 2h hand-off to the idle reaper unless it
# is `keep`-labelled, so the alarm has to say so while there is still a box to
# look at.
# --------------------------------------------------------------------------- #
def _serve_identity_alarms(target: str, w: dict[str, Any], iid: object,
                           now: float) -> list[tuple[str, str]]:
    """Identity alarms for one watch, from its persisted verdict alone. Pure."""
    rec = w.get("serve_identity")
    if not isinstance(rec, dict):
        return []
    state = rec.get("state")
    if state not in serve_ident.ALARM_STATES:
        return []
    art = rec.get("artifact")
    art_s = f" (artifact {art})" if art else ""
    age = int(now - (rec.get("since") or now))
    if state == "mismatch":
        park = ("PARKED" if rec.get("parked")
                else "and the PARK FAILED — stop it by hand "
                     f"(`herdd stop {iid}`)")
        return [(f"watch:{target}:serve_identity_mismatch",
                 f"{iid}: SERVE IDENTITY MISMATCH{art_s} for {age}s — the box "
                 f"VERIFIED ident {rec.get('observed')} on its own weights, "
                 f"this watch expects {rec.get('expected')}. {park}; withdrawn "
                 f"from the serve ladder (no rescue, no relaunch, NOT "
                 f"destroyed). Every eval scored against this endpoint carries "
                 f"the wrong label — re-check them. Next: "
                 f"`herdd fleet log --iid {iid}` for the verdict, then "
                 f"relaunch on the right artifact "
                 f"(`tools/vast/launch_serve.sh --model-artifact <slug>`) and "
                 f"`herdd fleet destroy {iid}` this one. It is parked, so "
                 f"the idle reaper takes it in 2h unless you keep-label it")]
    if state == "unarmed":
        return [(f"watch:{target}:serve_identity_unarmed",
                 f"{iid}: serve watch pins ident {rec.get('expected')}{art_s} "
                 f"but the box's READY marker carries NO ident= field "
                 f"({age}s) — this box never gated its own weights, so the "
                 f"pin is unenforced and 'no claim' is not a passing claim. "
                 f"Either it was launched without --model-artifact or it "
                 f"predates the on-box gate: relaunch with "
                 f"`launch_serve.sh --model-artifact {art or '<slug>'}`, or "
                 f"drop the pin (`herdd fleet watch {iid} --profile serve "
                 f"--budget <USD>` with no --expect-ident) and say out loud "
                 f"that this endpoint is ungated")]
    reason = rec.get("reason") or "identity_unknown"
    remedy = serve_ident.FAILED_REMEDY.get(
        reason, "unrecognised identity failure — read the box's serve log")
    return [(f"watch:{target}:serve_identity_failed",
             f"{iid}: the box wrote `FAILED {reason}`{art_s} {age}s ago — it "
             f"is NOT serving and is still billing. {remedy}. The box is not "
             f"parked (fleetd does not park a box that already refused to "
             f"serve — you may want its logs): `herdd ssh {iid}` to look, "
             f"then `herdd fleet destroy {iid}`")]


# --------------------------------------------------------------------------- #
# the daemon
# --------------------------------------------------------------------------- #
# moved-from: fleetd.Fleet
class Fleet:
    """Durable watch table + the reconcile tick. All I/O goes through `hooks`,
    all time through `hooks.now()`.

    Locking (review B3): `self.lock` guards STRUCTURAL state mutations and is
    held only for short critical sections, never across hook I/O — a slow tick
    (a _relaunch can take minutes) must not block a client command. `tick_lock`
    serializes reconcile passes with each other."""

    # moved-from: fleetd.Fleet.__init__
    def __init__(self, dirpath: str | None = None,
                 hooks: hooks_mod.FleetHooks | None = None) -> None:
        self.hooks: hooks_mod.FleetHooks = hooks or hooks_mod.Hooks()
        self.lock = threading.RLock()
        # Persistence is `fleet.state.Store`'s: the schema, the atomic save and
        # the journal record shape are one module's, and this daemon owns one
        # Store. BOTH injections matter — the Store takes the daemon's OWN
        # structural lock (a private lock of its own would stop excluding these
        # mutations from a save) and the daemon's clock seam `hooks.now` (a
        # `time.time` default would step around a test's frozen clock and out of
        # `meta.saved_ts` and every journal `ts`).
        self._store = fleet_state.Store(dirpath, now=self.hooks.now,
                                        lock=self.lock)
        self.dir = self._store.dir
        self.state_path = self._store.state_path
        self.journal_path = self._store.journal_path
        self.tick_lock = threading.RLock()
        # The SAME object the Store holds, not a copy: every mutation below is a
        # mutation of the document `save()` writes.
        self.state: dict[str, Any] = self._store.state
        self.runtime: dict[str, dict[str, Any]] = {}   # target -> live objects
        self.last_tick_ts = 0.0           # last SUCCESSFUL reconcile (not attempt)
        self.started_ts: float = self.hooks.now()
        self._alarm_since: dict[str, float] = {}   # derived alarm key -> first seen
        self._alarm_logged: set[str] = set()       # derived keys already journaled
        self.rev = git_rev()
        self._ticks = 0
        self._health: dict[str, dict[str, Any]] = {}
        self._health_ts: float | None = None    # last gather_fleet_health (wall)
        # iid -> last journaled verdict. Values come off a health row, so the
        # value type is whatever `gather_fleet_health` put there (a verdict
        # string today) — kept `Any` rather than narrowed, because the compare
        # below is `!=` against that same field and must not start coercing.
        self._health_alarmed: dict[str, Any] = {}
        self._env_mtime = self._env_stat()
        self._adopt_cap_cache: tuple[float, float] | None = None

    # ------------------------------------------------------------ persistence #
    # Four thin delegations to the Store. The daemon decides WHEN to save and
    # what to journal; `fleet.state` owns the schema, the atomic write and the
    # frozen record shape. They stay methods because that is the call shape the
    # whole tick loop (and every test that drives it) uses.
    # moved-from: fleetd.Fleet._load
    def _load(self) -> dict[str, Any]:
        return fleet_state.load_state(self.state_path)

    # moved-from: fleetd.Fleet.save
    def save(self) -> None:
        """Atomic temp+rename — a killed daemon never leaves a half-written
        state file (restart must lose nothing, S2)."""
        self._store.save()

    # moved-from: fleetd.Fleet._rotate_journal
    def _rotate_journal(self) -> None:
        self._store._rotate_journal()

    # moved-from: fleetd.Fleet.journal
    def journal(self, event: str, iid: object = None,
                **fields: Any) -> dict[str, Any]:  # noqa: ANN401 — arbitrary event body
        return self._store.journal(event, iid, **fields)

    # --------------------------------------------------------------- alarms #
    # An alarm still lit after you fixed it is worse than no alarm: it teaches
    # the operator to skip the one block that also carries the real conditions
    # (budget breach -> PARK, eviction, the stray fuse). So this channel has
    # exactly TWO shapes and no third:
    #
    #   DERIVED (the default) — a pure function of the PERSISTED state,
    #     recomputed on every READ (`alarm_records`). It lights the moment its
    #     condition is true and goes out the moment it is false, with no tick in
    #     between: `fleet watch`-ing an auto-adopted box clears its "AUTO-ADOPTED
    #     (no budget cap)" line on the very next `fleet status`. Nothing is
    #     appended anywhere, so nothing can be left behind.
    #   LATCHED (`latch_alarm`) — only for conditions whose evidence is CONSUMED
    #     by the tick that noticed them (a destroy request that expired and was
    #     popped; a park whose pending_action was already taken). Nothing in the
    #     state can re-derive those, so they persist in state.json across ticks
    #     AND restarts until the owning code retracts them or an operator acks
    #     (`herdd fleet ack <key>`). Persistence is the POINT there, and they
    #     render as `[LATCHED]` so they never read as a live measurement.
    #
    # WHICH SITE IS WHICH IS A PER-SITE DECISION WITH MONEY BEHIND IT, not a
    # style: latched here are `destroy:<iid>:expired`, `action:<iid>:failed`,
    # `pyhalf:<iid>` and the deferred/refused `handoff:<iid>`; derived (never
    # appended) are the budget-park alarm off `w['state']`, N2/serve-not-live,
    # the unbudgeted-adoption alarm and the stray alarms off `live_ts`. A port
    # that uniformly latches or uniformly derives is a behavior change even
    # though every test may pass.
    #
    # The defect this replaced (2026-07-31, box 46347213): alarms were a plain
    # per-tick list that every producer appended to. A resolved condition stayed
    # lit for a whole tick interval — unbounded whenever `hooks.instances()`
    # kept failing, because that path returned BEFORE the rebuild while still
    # bumping the tick clock, so `tick_age_s` looked fresh over a frozen list —
    # and, the mirror-image bug, a genuinely one-shot condition (destroy
    # EXPIRED, park FAILED) flashed for a single tick and was gone forever.
    @property
    def alarms(self) -> list[str]:
        """Human-readable alarm strings for the state RIGHT NOW."""
        return [r["msg"] for r in self.alarm_records()]

    # moved-from: fleetd.Fleet.alarm_records
    def alarm_records(self, now: float | None = None) -> list[dict[str, Any]]:
        """Alarms as records: key, message, sticky flag, age. Read-only — a
        status call must never mutate the fleet."""
        now = self.hooks.now() if now is None else now
        with self.lock:
            derived = list(self._derive_alarms(now))
            latched = {k: dict(v) for k, v in
                       (self.state.get("alarms") or {}).items()}
        recs: list[dict[str, Any]] = []
        for key, msg in derived:
            since = self._alarm_since.get(key)
            if since is None:
                since = self._alarm_since[key] = now
            recs.append({"key": key, "msg": msg, "sticky": False,
                         "since_ts": since, "age_s": round(now - since, 1)})
        for key, rec in sorted(latched.items()):
            first = rec.get("first_ts") or now
            recs.append({"key": key, "msg": rec.get("msg"), "sticky": True,
                         "iid": rec.get("iid"), "since_ts": first,
                         "age_s": round(now - first, 1),
                         "count": rec.get("count", 1)})
        return recs

    # moved-from: fleetd.Fleet.latch_alarm
    def latch_alarm(self, key: str, msg: str, iid: object = None) -> dict[str, Any]:
        """Raise a STICKY alarm. Use ONLY where the condition's evidence is gone
        by the time it is noticed — everything a later tick can re-derive from
        the state belongs in `_derive_alarms`, which retracts itself."""
        now = self.hooks.now()
        with self.lock:
            table = self.state.setdefault("alarms", {})
            prev = table.get(key) or {}
            table[key] = {"msg": msg, "iid": None if iid is None else str(iid),
                          "first_ts": prev.get("first_ts") or now,
                          "last_ts": now, "count": prev.get("count", 0) + 1}
        if not prev:
            self.journal("alarm_latched", iid=iid, key=key, note=msg)
        # `cast`, not `dict(...)`: the flat daemon returns the LIVE record, and a
        # defensive copy here would be the one behavior change in this method
        # (strict typing wants a narrowing, not a new object — §6).
        return cast("dict[str, Any]", table[key])

    # moved-from: fleetd.Fleet.clear_alarm
    def clear_alarm(self, key: object, reason: str | None = None,
                    requester: str | None = None) -> dict[str, Any] | None:
        """Retract a latched alarm (the owning code's `resolved` path, or an
        operator ack). Derived alarms have no clear path by construction."""
        rec: dict[str, Any] | None
        with self.lock:
            rec = (self.state.get("alarms") or {}).pop(str(key), None)
        if rec is None:
            return None
        self.save()
        self.journal("alarm_cleared", iid=rec.get("iid"), key=str(key),
                     reason=reason, requester=requester,
                     age_s=round(self.hooks.now() - (rec.get("first_ts") or 0), 1),
                     note=rec.get("msg"))
        return rec

    # moved-from: fleetd.Fleet.ack_alarm
    def ack_alarm(self, key: object = None, all_keys: bool = False,
                  requester: str | None = None) -> dict[str, Any]:
        """`fleet ack`: clear latched alarms an operator has seen. A DERIVED key
        is refused with the reason — you clear those by fixing the condition,
        and they clear themselves the instant you do."""
        with self.lock:
            keys = sorted((self.state.get("alarms") or {}).keys())
        if all_keys:
            cleared = [k for k in keys
                       if self.clear_alarm(k, "ack --all", requester) is not None]
            return {"cleared": cleared}
        key = str(key)
        rec = self.clear_alarm(key, "ack", requester)
        if rec is None:
            if any(r["key"] == key for r in self.alarm_records()):
                raise ValueError(
                    f"{key} is a DERIVED alarm — it reports a condition that is "
                    f"true right now and clears itself the moment you fix it; "
                    f"there is nothing to acknowledge")
            raise KeyError(f"no latched alarm {key!r} (latched: {keys or 'none'})")
        return {"cleared": [key]}

    # moved-from: fleetd.Fleet._journal_alarm_transitions
    def _journal_alarm_transitions(self, now: float) -> None:
        """Durable record of the derived channel: a `fleet status` reader sees
        alarms live, but nobody was watching at 03:00. Journal each derived
        alarm ONCE when it lights and once when it goes out (the health path
        keeps its own, older `health_alarm` events — skipped here, not doubled).
        Also prunes `_alarm_since` so a cleared alarm cannot keep a stale age.
        The journal bookkeeping is deliberately its OWN set: `_alarm_since` is
        touched by the read path, and a `fleet status` that happened to land
        first must not swallow the journal record."""
        with self.lock:
            live = {k: m for k, m in self._derive_alarms(now)
                    if not k.startswith("health:")}
        for key, msg in sorted(live.items()):
            self._alarm_since.setdefault(key, now)
            if key not in self._alarm_logged:
                self._alarm_logged.add(key)
                self.journal("alarm_raised", key=key, note=msg)
        for key in [k for k in self._alarm_logged if k not in live]:
            self._alarm_logged.discard(key)
            since = self._alarm_since.pop(key, now)
            self.journal("alarm_resolved", key=key,
                         age_s=round(now - since, 1),
                         note="condition no longer true in the fleet state")
        for key in [k for k in self._alarm_since
                    if k not in live and not k.startswith("health:")]:
            self._alarm_since.pop(key, None)

    # moved-from: fleetd.Fleet._derive_alarms
    def _derive_alarms(self, now: float) -> list[tuple[str, str]]:
        """EVERY alarm whose condition survives in the persisted state is
        recomputed here, from that state alone: no I/O, no mutation, callable
        from a status read or from an offline `fleetd status` dump. Returns
        [(key, message)]. Call under `self.lock`."""
        out: list[tuple[str, str]] = []
        down_since = (self.state.get("meta") or {}).get("api_unavailable_since")
        if down_since:
            out.append(("fleet:api_unavailable",
                        f"vast API unreadable for {int(now - down_since)}s — the "
                        f"fleet is NOT being reconciled (no ladders, no parks, no "
                        f"accrual); every reading below is from before that"))
        for target, w in sorted(self.state.get("watches", {}).items()):
            out.extend(self._derive_watch_alarms(target, w, now))
        for iid, s in sorted(self.state.get("strays", {}).items()):
            out.extend(self._derive_stray_alarms(iid, s, now))
        out.extend(rows.retention_alarms(self.state, now))
        for iid, req in sorted(self.state.get("destroys", {}).items()):
            if req.get("executed"):
                continue
            if req.get("last_error"):
                out.append((f"destroy:{iid}:failed",
                            f"{iid}: destroy FAILED ({req['last_error']}) — retrying"))
            if req.get("held_reason"):
                out.append((f"destroy:{iid}:held",
                            f"{iid}: destroy held — {req['held_reason']}"))
        cap = global_budget_usd()
        if cap is not None:
            total = sum(self.state["spend_by_box"].values())
            if total >= cap:
                out.append(("fleet:budget",
                            f"FLEET budget ${total:.2f} >= ${cap:.2f} — new "
                            f"spend-capable watches REFUSED and money moves "
                            f"suspended; park boxes by hand (`fleet park <IID>`) "
                            f"to stop the bill"))
        return out

    # moved-from: fleetd.Fleet._derive_watch_alarms
    def _derive_watch_alarms(self, target: str, w: dict[str, Any],
                             now: float) -> list[tuple[str, str]]:
        iid = w.get("iid") or target
        st = w.get("state")
        # The CEILING's cumulative spend, which is what the cap is measured
        # against — reporting the watch's own counter next to an inherited cap
        # would print "$0.00 of $5.00" on a box with $4.90 already spent.
        spend = self._ceiling_spend(w)
        # A budget park is the loudest thing about a watch and it OUTRANKS the
        # S8 dormancy silence: fleetd parked this box itself, and it stays
        # parked (and alarming) until a human raises the cap or lets it go. The
        # persistence is real — the box IS capped right now — not a leftover.
        if st == "budget_parked":
            return [(f"watch:{target}:budget",
                     f"{iid}: BUDGET ${spend:.2f} >= ${w.get('budget_usd')} — "
                     f"PARKED by fleetd; raise the cap (`fleet watch {iid} "
                     f"--budget ...`) or let it go")]
        if st == "budget_park_failed":
            return [(f"watch:{target}:budget",
                     f"{iid}: BUDGET ${spend:.2f} >= ${w.get('budget_usd')} and the "
                     f"PARK FAILED ({w.get('last_error')}) — park it by hand")]
        ident = _serve_identity_alarms(target, w, iid, now)
        if ident and (w.get("serve_identity") or {}).get("state") == "mismatch":
            # OUTRANKS the S8 dormancy silence, exactly as a budget park does
            # and for the mirror-image reason: the box is withdrawn RIGHT NOW
            # and stays withdrawn, so this is a standing condition and not a
            # leftover. Everything else about the watch is noise next to "the
            # weights are wrong" — hence the early return.
            return ident
        if w.get("dormant") or st in ("gone", "instance_gone"):
            # S8 silence has ONE exception: a standing watch whose box is still
            # LIVE (i.e. `--keep`, or a box resumed by hand) is money leaving
            # the wallet with the ladder deliberately idle. A PARKED standing
            # box — the normal shape — stays silent, which is the point of the
            # mode. Derived, so it retracts itself the tick a ticket lands.
            if w.get("standing_dormant") and w.get("standing_live_since"):
                scap = w.get("budget_usd")
                sleft = (f", ${scap - spend:.2f} of ${scap:.2f} left"
                         if scap is not None else "")
                return [(f"watch:{target}:standing_idle",
                         f"{iid}: STANDING jobs watch is DORMANT on a box that "
                         f"is still LIVE "
                         f"({int(now - w['standing_live_since'])}s since its "
                         f"queue drained{sleft}) — submit the next wave (the "
                         f"watch re-arms itself, no `fleet watch` needed) or "
                         f"`fleet park {iid}` to stop the bill")]
            return []                               # S8: dormant boxes do not alarm
        out: list[tuple[str, str]] = []
        if w.get("last_tick_error"):
            out.append((f"watch:{target}:tick_error",
                        f"{target}: tick error {w['last_tick_error']}"))
        if st == "init_error":
            out.append((f"watch:{target}:init_error",
                        f"{target}: init failed ({w.get('init_error')}) — retrying"))
        if w.get("paused_until") and now < w["paused_until"] \
                and w.get("budget_breach_pending"):
            out.append((f"watch:{target}:pause_breach",
                        f"{iid}: BUDGET ${spend:.2f} >= ${w.get('budget_usd')} "
                        f"DURING A PAUSE — parking at expiry "
                        f"({int(w['paused_until'] - now)}s)"))
        if w.get("adopted"):
            src = w.get("ceiling_source")
            cap, left = w.get("budget_usd"), None
            if cap is not None:
                left = round(cap - spend, 2)
            if cap is None:
                # Unreachable through `watch()` since the ceiling ledger landed
                # (2026-08-09): every adoption inherits a cap or gets the
                # provisional default. Kept as a fail-closed assertion — if a
                # future path ever produces one again, it must not be silent,
                # because silence here is the whole 2026-08-03 defect.
                out.append((f"watch:{target}:adopted",
                            f"{iid}: AUTO-ADOPTED with NO BUDGET CAP — this "
                            f"should be unreachable; the box is spending "
                            f"unbounded. `fleet watch {iid} --budget <USD>` now, "
                            f"then file it"))
            elif src == "inherited":
                out.append((f"watch:{target}:adopted",
                            f"{iid}: its armed watch LAPSED and the safety net "
                            f"re-adopted it — the ceiling SURVIVED "
                            f"(${spend:.2f} of ${cap:.2f}, ${left:.2f} left) but "
                            f"the ladder did not; re-register a real "
                            f"`fleet watch` to re-arm bid rescue/replacement"))
            else:
                out.append((f"watch:{target}:adopted",
                            f"{iid}: AUTO-ADOPTED by the safety net under the "
                            f"PROVISIONAL default cap ${cap:.2f} "
                            f"(${left:.2f} left) — nobody chose that figure; "
                            f"register a real `fleet watch {iid} --budget <USD>`"))
        if w.get("profile") == "serve" and w.get("budget_usd") is None \
                and not w.get("adopted"):
            # legacy state.json watch from before serve joined the policy tier
            # (2026-08-02): it now runs the bid ladder with NO spend cap.
            out.append((f"watch:{target}:serve_unbudgeted",
                        f"{iid}: serve watch predates the bid ladder (no budget "
                        f"cap) — re-register: `herdd fleet watch {iid} "
                        f"--profile serve --budget <USD>`"))
        # A WEDGED replacement lane: refusing, tick after tick, with nothing in
        # the system that escalates. Derived from the durable streak counter so
        # it retracts itself the moment a rental succeeds, and so a daemon
        # restart cannot reset it back to silence. It names the BOUND and the
        # MARKET GAP, because "no qualifying replacement offer" sent the operator
        # hunting for an empty market on 2026-08-24 when the market held exactly
        # one offer, 3.4% over the ceiling.
        _repl = w.get("replacement") or {}
        _refusals = _repl.get("replacement_refusals") or 0
        _since = _repl.get("replacement_refusals_since")
        # The count alone is tick-relative, and "past a transient market blip"
        # is a wall-clock claim: at a 15s tick five refusals is 75s. A missing
        # `_since` (state written before the field) falls back to the count.
        _wedged = (isinstance(_refusals, int)
                   and _refusals >= REPLACEMENT_WEDGE_REFUSALS
                   and (not _since or now - float(_since) >= REPLACEMENT_WEDGE_MIN_S))
        if _wedged:
            _ceil = _repl.get("replacement_refusal_ceiling")
            _mkt = _repl.get("replacement_market_floor")
            _gap = (f", cheapest qualifying offer seen at ${float(_mkt):.4f}/hr"
                    if _mkt else ", no qualifying offer seen at any price")
            _reason = _repl.get("replacement_refusal_reason")
            _acct = self._account_fault_note(_reason, w)
            _head = (f"{iid}: AUTONOMOUS REPLACEMENT WEDGED — {_refusals} "
                     f"consecutive refusals"
                     + (f" over {int(now - float(_since))}s" if _since else ""))
            if _acct:
                # The ceiling/market framing below is ACTIVELY WRONG here: the
                # refusals never reached the market. Naming a bound that was
                # never tested is what sent 2026-08-25 hunting for offers while
                # the balance sat at -$1.50.
                out.append((f"watch:{target}:replacement_wedged",
                            f"{_head} — {_acct}. Nothing about this watch, its "
                            f"ceiling or the market is the problem"))
            else:
                out.append((f"watch:{target}:replacement_wedged",
                            _head
                            + f" ({_reason}), ceiling "
                            f"${_ceil}{_gap}. The ceiling already re-prices against "
                            f"live market evidence, so this is a bound that HELD: "
                            f"raise it (`fleet watch {iid} --replace-ceiling-mult "
                            f"<N>`), raise `--budget`, or accept the queue is not "
                            f"affordable in this market"))
        if w.get("queue_empty_since"):
            out.append((f"watch:{target}:queue_empty",
                        f"{iid}: jobs watch armed but its QUEUE IS EMPTY for "
                        f"{int(now - w['queue_empty_since'])}s — submit the wave "
                        f"(watch kept, cap ${w.get('budget_usd')})"))
        if w.get("unrecoverable_since"):
            fix = ("relaunch the serve (launch_serve.sh)"
                   if w.get("profile") == "serve"
                   else "`job retarget` the tickets")
            # Name WHY the automatic replacement did not happen. The old text
            # ("raise the bid") was the operator instruction on 2026-08-05 for a
            # box no bid could win — an alarm that prescribes the move the ladder
            # already proved impossible is how two hand-rescues got spent.
            refused = w.get("replacement_refused")
            auto = (f"; AUTO-REPLACEMENT REFUSED: {refused}" if refused
                    else "" if w.get("profile") == "serve"
                    else "; auto-replacement did not run (see the journal)")
            # An ACCOUNT-level refusal replaces the three remedies entirely.
            # None of them can work — measured 2026-08-25, where every one of
            # `job retarget` / raise the cap / `fleet destroy` was offered
            # against a $0.00 balance and none of them was the answer.
            _acct = self._account_fault_note(refused, w)
            tail = (_acct if _acct
                    else f"{fix}, raise the budget/cap, or `fleet destroy` it")
            out.append((f"watch:{target}:rescue_stalled",
                        f"{iid}: {w.get('profile')} RESCUE STALLED for "
                        f"{int(now - w['unrecoverable_since'])}s — box still "
                        f"exists, watch kept (cap ${w.get('budget_usd')})"
                        f"{auto}; {tail}"))
        out.extend(ident)
        m = self._health_alarm_msg(iid)
        if m:
            out.append((f"health:{iid}", m))
        return out

    @staticmethod
    def _account_fault_note(refused: object, w: dict[str, Any]) -> str:
        """The remedy line for an ACCOUNT-caused refusal, or "" if the failure
        is one the market/host could plausibly answer.

        Reads the stored reason CODE first (set by the ladder at the moment it
        saw the API error) and falls back to string-matching the refusal text,
        because the pull lane's refusal reaches this dict only as prose.
        """
        code = acctfault.classify(refused)
        if code is None:
            reason = (w.get("replacement")
                      or {}).get("replacement_refusal_reason")
            if str(reason or "") == acctfault.REASON:
                code = acctfault.REASON
        return acctfault.describe(code) if code else ""

    # moved-from: fleetd.Fleet._derive_stray_alarms
    def _derive_stray_alarms(self, iid: str, s: dict[str, Any],
                             now: float) -> list[tuple[str, str]]:
        # Only a stray SEEN LIVE by the most recent successful reconcile can
        # alarm. Without this gate a record left behind by a parked box (the
        # sweep skips non-live instances before it ever touches the record)
        # would alarm forever off a stale reading — the exact latching shape
        # this design exists to prevent.
        last_ok = (self.state.get("meta") or {}).get("last_ok_tick_ts")
        if not s.get("live_ts") or (last_ok and s["live_ts"] < last_ok):
            return []
        if s.get("paused_until") and now < s["paused_until"]:
            return []
        out: list[tuple[str, str]] = []
        m = self._health_alarm_msg(iid)
        if m:
            out.append((f"health:{iid}", m))
        if s.get("adopt_error"):
            out.append((f"stray:{iid}",
                        f"{iid}: UNWATCHED and auto-adopt FAILED "
                        f"({s['adopt_error']}) — `fleet watch` it"))
        elif s.get("park_error"):
            out.append((f"stray:{iid}",
                        f"{iid}: unwatched past {int(s.get('grace_s') or 0)}s "
                        f"[{s.get('tier')}, ${s.get('dph_disp')}/hr] — park FAILED "
                        f"({s['park_error']}) — park it by hand"))
        elif s.get("parked_ts"):
            out.append((f"stray:{iid}", f"{iid}: UNWATCHED — PARKED by the safety "
                                        f"net (park already requested)"))
        else:
            out.append((f"stray:{iid}",
                        f"{iid}: UNWATCHED + IDLE-LOOKING for "
                        f"{int(s.get('observed_s') or 0)}s at "
                        f"${s.get('dph_disp')}/hr [{s.get('tier')}] "
                        f"(label={s.get('label')!r}) — `fleet watch` it"))
        return out

    # moved-from: fleetd.Fleet._env_stat
    def _env_stat(self) -> float | None:
        try:
            return os.stat(os.path.join(repo_root(), ".env")).st_mtime
        except OSError:
            return None

    # moved-from: fleetd.Fleet._maybe_reload_env
    def _maybe_reload_env(self) -> None:
        """N5: keys get re-minted; pick up a changed .env without a restart."""
        m = self._env_stat()
        if m != self._env_mtime:
            self._env_mtime = m
            try:
                config.load_env()
                self.journal("env_reloaded")
            except Exception as e:
                self.journal("env_reload_failed", error=str(e)[:200])

    def _record_machine_ids(self, instances: Any, now: float) -> None:  # noqa: ANN401
        """Write down `instance_id -> machine_id` while the box still exists.

        vast's API is the only source of that mapping and drops a row the moment
        the box is destroyed, which is why `hostfacts.py ingest` resolved 3 of
        202 records on 2026-08-24. fleetd reads every instance every tick and
        already has both halves in hand — this is the write-down, not a new
        query.

        Never fatal and never a gate: a full disk must not stop a reconcile.
        The ledger is an index that can always be rebuilt going forward, and
        losing a tick's worth of it costs nothing a later tick will not redo.
        """
        try:
            pairs = [(i.get("id"), i.get("machine_id"))
                     for i in (instances or []) if isinstance(i, dict)]
            n = machine_ledger.record(pairs, now=now)
        except Exception as e:                            # noqa: BLE001
            self.journal("machine_ledger_write_failed", error=str(e)[:200])
            return
        if n:
            self.journal("machine_ledger_updated", entries_changed=n)

    # --------------------------------------------------------- ceiling ledger #
    # See the block comment above `rows.handoff_predecessor` for the defect, the
    # three paths and the invariants. Everything here is called under, or takes,
    # `self.lock` for the structural writes; the resolution itself is cheap.
    # moved-from: fleetd.Fleet.adopt_default_cap
    def adopt_default_cap(self) -> float:
        """The provisional cap for an adoption with nothing to inherit. Never
        None, never <= 0 — `core.config` resolves it fail-closed.

        Cached for ADOPT_CAP_CACHE_S because `_ceiling_peek` calls this once per
        watch per tick AND on every status read, and the resolver reads up to
        three config files. The cache is short enough that an edit to
        `herdd.yaml` takes effect within a tick, and a resolver that raises
        anyway falls back to the constant — a config read that failed is still
        not permission to run uncapped."""
        now = self.hooks.now()
        cached = self._adopt_cap_cache
        if cached is not None and now - cached[0] < ADOPT_CAP_CACHE_S:
            return cached[1]
        try:
            v = config.fleetd_adopt_default_budget_usd()
        except Exception:
            v = config.ADOPT_DEFAULT_BUDGET_USD
        self._adopt_cap_cache = (now, v)
        return v

    # moved-from: fleetd.Fleet._ceilings
    def _ceilings(self) -> dict[str, Any]:
        c = self.state.get("ceilings")
        if not isinstance(c, dict):
            c = self.state["ceilings"] = {}
        return c

    # moved-from: fleetd.Fleet._ceiling_index
    def _ceiling_index(self) -> dict[str, Any]:
        ix = self.state.get("ceiling_by_box")
        if not isinstance(ix, dict):
            ix = self.state["ceiling_by_box"] = {}
        return ix

    # moved-from: fleetd.Fleet._ceiling_peek
    def _ceiling_peek(
        self, cid: object,
    ) -> tuple[dict[str, Any], float, float, str | None] | None:
        """PURE. `(record, cap_usd, spend_usd, degraded_reason)` or None.

        Mutates nothing and journals nothing, so the DERIVED-alarm path and
        `status()` can call it — those are read paths, and a status call that
        mutated the fleet is the latching bug this alarm channel exists to
        avoid. `_ceiling_read` is the tick-path twin that also repairs."""
        if not cid:
            return None
        rec = self._ceilings().get(str(cid))
        if rec is None:
            return None
        cap, spend, degraded = rows.normalize_ceiling(rec, self.adopt_default_cap())
        return rec, cap, spend, degraded

    # moved-from: fleetd.Fleet._ceiling_read
    def _ceiling_read(
        self, cid: object,
    ) -> tuple[dict[str, Any], float, float, str | None] | None:
        """`(record, cap_usd, spend_usd, degraded_reason)` or None if no such
        ceiling. The record is REPAIRED in place when it reads degraded, so a
        garbage cap becomes the provisional default durably instead of being
        re-derived (and re-journaled) on every tick. TICK PATH ONLY — read
        paths use `_ceiling_peek`."""
        peeked = self._ceiling_peek(cid)
        if peeked is None:
            return None
        rec, cap, spend, degraded = peeked
        if degraded:
            if not isinstance(rec, dict):
                rec = {}
                self._ceilings()[str(cid)] = rec
            rec["cap_usd"], rec["spend_usd"] = cap, spend
            rec["source"] = "degraded"
            rec["degraded_reason"] = degraded
            self.journal("ceiling_degraded", ceiling_id=str(cid),
                         cap_usd=round(cap, 4), reason=degraded,
                         note="an unreadable ceiling means the CONSERVATIVE "
                              "DEFAULT, never unlimited — re-arm with "
                              "`fleet watch <IID> --budget <USD>`")
        return rec, cap, spend, degraded

    # moved-from: fleetd.Fleet._resolve_ceiling_id
    def _resolve_ceiling_id(self, target: str, iid: object = None,
                            label: str | None = None) -> str | None:
        """Which durable ceiling does this box/watch draw on? Order matters:

        1. the box index — set when a ladder replaced the box or a handoff
           named its understudy, so a SUCCESSOR resolves to its predecessor;
        2. the watch key — the id the ceiling was armed under;
        3. the `job:<pred>:handoff` label — the fallback for an understudy the
           ladder never reported to us (restart mid-migration).

        Returns a ceiling id that EXISTS in the ledger, or None."""
        cs, ix = self._ceilings(), self._ceiling_index()
        for cand in (ix.get(str(iid)) if iid else None,
                     ix.get(str(target)),
                     str(iid) if iid else None,
                     str(target)):
            if cand and str(cand) in cs:
                return str(cand)
        pred = rows.handoff_predecessor(label)
        if pred:
            for cand in (ix.get(pred), pred):
                if cand and str(cand) in cs:
                    return str(cand)
        return None

    # moved-from: fleetd.Fleet._ceiling_bind_box
    def _ceiling_bind_box(self, cid: object, iid: object) -> None:
        """Point a box id at a ceiling — the successor-inheritance seam. Called
        for every box a ladder moves a watch onto (eviction replacement, SLA
        relaunch, completed handoff) and for a handoff understudy the moment the
        ladder names it, which is BEFORE the stray sweep can see it. That
        ordering is the whole of the path-2 fix: by the time the safety net
        adopts the understudy, the understudy already has a ceiling."""
        if not cid or iid in (None, "", "None"):
            return
        cid, iid = str(cid), str(iid)
        if str(cid) not in self._ceilings():
            return
        ix = self._ceiling_index()
        if ix.get(iid) == cid:
            return
        prev = ix.get(iid)
        ix[iid] = cid
        rec = self._ceilings()[cid]
        members = rec.setdefault("members", [])
        if iid not in members:
            members.append(iid)
        self.journal("ceiling_box_bound", iid=iid, ceiling_id=cid,
                     previous_ceiling=prev,
                     cap_usd=rows._num(rec.get("cap_usd")),
                     spend_usd=round(rows._num(rec.get("spend_usd")) or 0.0, 4),
                     note="successor box bound to the ceiling it inherits — it "
                          "draws down the SAME headroom, it does not open a "
                          "second cap")

    # moved-from: fleetd.Fleet._ceiling_arm
    def _ceiling_arm(self, cid: object, cap_usd: float, target: str,
                     requester: str | None, source: str,
                     reset_spend: bool = False) -> dict[str, Any]:
        """Create or re-arm a ceiling. Re-arming REPLACES the cap and KEEPS the
        spend: that is the whole re-arm arithmetic. Arming $5 on a ceiling that
        has already spent $2 leaves $3 of headroom, not $5 — anything else is
        the N x cap preempt-loop the filing describes.

        `reset_spend` is the deliberate escape hatch (`fleet watch
        --reset-spend`): a genuinely new campaign on the same box. It is loud,
        journaled with the figure it discarded, and never automatic."""
        cid = str(cid)
        now = self.hooks.now()
        rec = self._ceilings().get(cid)
        if not isinstance(rec, dict):
            rec = {"created_ts": now, "spend_usd": 0.0, "members": [],
                   "history": [], "epochs": 0}
            self._ceilings()[cid] = rec
        prev_cap = rows._num(rec.get("cap_usd"))
        prev_spend = rows._num(rec.get("spend_usd")) or 0.0
        if reset_spend:
            rec["spend_usd"] = 0.0
        rec.update({"cap_usd": float(cap_usd), "source": source,
                    "origin_target": rec.get("origin_target") or str(target),
                    "requester": requester or rec.get("requester"),
                    "updated_ts": now})
        rec.pop("degraded_reason", None)
        hist = rec.setdefault("history", [])
        hist.append({"ts": round(now, 3), "event": "armed", "target": str(target),
                     "cap_usd": float(cap_usd), "source": source,
                     "spend_usd": round(rec.get("spend_usd", 0.0), 4),
                     "reset_spend": bool(reset_spend), "requester": requester})
        del hist[:-CEILING_HISTORY_MAX]
        self.journal("ceiling_armed", ceiling_id=cid, target=str(target),
                     cap_usd=round(float(cap_usd), 4), source=source,
                     previous_cap_usd=(round(prev_cap, 4)
                                       if prev_cap is not None else None),
                     spend_usd=round(rec.get("spend_usd", 0.0), 4),
                     remaining_usd=round(float(cap_usd)
                                         - rec.get("spend_usd", 0.0), 4),
                     spend_discarded_usd=(round(prev_spend, 4)
                                          if reset_spend and prev_spend else None),
                     requester=requester,
                     note=("SPEND RESET by explicit operator request — the "
                           "ceiling starts over" if reset_spend else
                           "the cap is CUMULATIVE over this ceiling's whole "
                           "lineage; what is enforced is remaining headroom"))
        return rec

    # moved-from: fleetd.Fleet._resolve_watch_ceiling
    def _resolve_watch_ceiling(self, target: str, profile: str,
                               budget_usd: float | None, adopted: bool,
                               requester: str | None, label: str | None,
                               reset_spend: bool,
                               ) -> tuple[str | None, float | None, float, str]:
        """`(ceiling_id, effective_budget_usd, carried_spend_usd, note)` for a
        watch that is about to be written. Call under `self.lock`.

        Three cases, and none of them can produce an automatic None:
          * explicit + a figure -> arm/re-arm; the figure is the cap, the
            ledger's spend is carried onto the watch record;
          * explicit + no figure (`--profile bare` with no `--budget`) -> if a
            ceiling exists, INHERIT it (the ceiling belongs to the box, not to
            whichever watch is holding it); if none exists, this is a human
            deliberately asking for observation-only and stays uncapped;
          * adopted -> inherit if anything is inheritable, else arm a NEW
            ceiling at the conservative provisional default.
        """
        cid = self._resolve_ceiling_id(target, iid=target, label=label)
        found = self._ceiling_read(cid)
        if budget_usd is not None and not adopted:
            cid = cid or str(target)
            rec = self._ceiling_arm(cid, budget_usd, target, requester,
                                    "explicit", reset_spend=reset_spend)
            carried = rows._num(rec.get("spend_usd")) or 0.0
            return cid, float(budget_usd), carried, "explicit"
        if found is not None:
            _rec, cap, spend, _deg = found
            return cid, cap, spend, "inherited"
        if adopted:
            cid = str(target)
            cap = self.adopt_default_cap()
            self._ceiling_arm(cid, cap, target, requester, "default")
            return cid, cap, 0.0, "default"
        return None, budget_usd, 0.0, "uncapped"

    # moved-from: fleetd.Fleet._ceiling_spend
    def _ceiling_spend(self, w: dict[str, Any]) -> float:
        """Cumulative spend charged to this watch's ceiling. PURE (peek, not
        read) — the alarm derivation and `status()` call it. Falls back to the
        watch's own counter when it has no ceiling (an explicit uncapped `bare`
        watch), so the breach test is unchanged for that one case."""
        own = rows._num(w.get("spend_usd")) or 0.0
        found = self._ceiling_peek(w.get("ceiling_id"))
        if found is None:
            return own
        return max(found[2], own)

    # moved-from: fleetd.Fleet._charge_ceiling
    def _charge_ceiling(self, w: dict[str, Any]) -> None:
        """Charge this watch's spend INCREMENT to its ceiling. Deltas, not a
        max(), so two boxes drawing on one ceiling at once (the handoff overlap:
        primary and understudy both live for a few minutes) ADD rather than
        shadow each other. Idempotent: charging twice in a tick charges zero the
        second time."""
        cid = w.get("ceiling_id")
        found = self._ceiling_read(cid)
        if found is None:
            return
        rec = found[0]
        own = rows._num(w.get("spend_usd")) or 0.0
        charged = rows._num(w.get("_ceiling_charged_usd"))
        if charged is None:
            charged = own                # first tick of this watch: nothing new
        delta = own - charged
        if delta > 0:
            rec["spend_usd"] = (rows._num(rec.get("spend_usd")) or 0.0) + delta
            rec["updated_ts"] = self.hooks.now()
        w["_ceiling_charged_usd"] = own

    # moved-from: fleetd.Fleet._ceiling_tick_field
    def _ceiling_tick_field(self, w: dict[str, Any]) -> float | None:
        """`ceiling_spend_usd` for a tick line — or None when it is the same
        number as `spend_usd`, so the ~18k tick lines in a week of journal only
        grow a field when the ceiling total actually DIVERGES from this watch's
        own counter (a lapse carried forward, a shared successor). `journal()`
        drops None-valued fields, so None costs nothing."""
        cid = w.get("ceiling_id")
        if not cid:
            return None
        total = round(self._ceiling_spend(w), 4)
        return None if total == round(rows._num(w.get("spend_usd")) or 0.0, 4) else total

    # ------------------------------------------------------------ watch table #
    # moved-from: fleetd.Fleet._explicit_owner
    def _explicit_owner(self, target: str,
                        ) -> tuple[str | None, dict[str, Any] | None]:
        """The EXPLICIT (human-registered, `adopted=False`) watch that owns
        `target` — matched by watch key OR by resolved iid. Call under the lock."""
        for t, w in self.state["watches"].items():
            if (t == target or str(w.get("iid") or t) == target) \
                    and not w.get("adopted"):
                return t, w
        return None, None

    # moved-from: fleetd.Fleet.watch
    def watch(self, target: object, profile: str = "bare",
              budget_usd: float | None = None, policy: dict[str, Any] | None = None,
              requester: str | None = None, adopted: bool = False,
              label: str | None = None, reset_spend: bool = False,
              standing: bool | None = None) -> dict[str, Any]:
        """Register or upsert a watch.

        Addressing: a watch is FILED under the key it was registered with, but
        its ladder may since have moved it onto another box (`rows.watch_box_iid`)
        — and that replacement id is what `herdd ls` and `fleet status` show,
        so it is the id an operator naturally types. Registering the replacement
        id therefore REDIRECTS to the owning key instead of colliding with it.

        `policy` semantics, deliberately different on the two paths:

        - Addressing a watch by its OWN key REPLACES `policy` wholesale (an
          explicit re-registration states the whole policy; unchanged since v1).
        - A REDIRECT MERGES (`_redirect_policy`): the existing policy is the
          base and the caller's keys win except where the caller's value is
          `None`. The operator addressed a box, not the watch record, usually to
          change one thing (the cap); blanking `salvage` / `max_replacements` /
          `max_bid` on a LIVE ladder as a side effect of raising a budget is a
          spend-control change nobody asked for. `False` still wins, so
          `--no-handoff` / `--no-salvage` remain expressible; resetting a flag
          to "unset" needs the owning key, and the refusal message says so.

        `label` (auto-adopt only) lets a handoff understudy resolve its
        predecessor's ceiling from its `job:<pred>:handoff` label. `reset_spend`
        is the explicit escape hatch that starts a ceiling's spend over — see
        `_ceiling_arm`.

        `standing` (jobs only, 2026-08-14) opts the watch into surviving its
        queue draining — see `_standing_drain`. Tri-state on purpose: `True` /
        `False` STATE it (the CLI always does, so `fleet watch` without
        `--standing` turns it off, the same wholesale semantics `policy` has on
        the own-key path), and `None` means "leave whatever the record says", so
        an internal re-registration cannot silently disarm a standing watch.

        THE CAP IS NOT THE ARGUMENT. `budget_usd` is what the caller ASKED for;
        what lands on the record is `_resolve_watch_ceiling`'s answer, which
        inherits a durable ceiling when one exists and substitutes the
        conservative provisional default for an adoption that has nothing to
        inherit. No automatic path can produce an uncapped watch.
        """
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r} (want {PROFILES})")
        target = str(target)
        # 2026-07-30: the safety net ADOPTS, it never DOWNGRADES. An explicit
        # watch carries the profile and the hard budget cap a human chose, so it
        # outranks auto-adoption — otherwise an adoption racing a `fleet watch`
        # (accept thread vs reconcile thread) blanks profile=jobs/budget=5 back
        # to bare/None. Refuse quietly, never raise: the caller here is a tick.
        if adopted:
            with self.lock:
                owner_t, owner_w = self._explicit_owner(target)
                snap = dict(owner_w) if owner_w is not None else None
            if owner_w is not None and snap is not None:
                self.journal("auto_adopt_refused", iid=target, target=owner_t,
                             profile=snap.get("profile"),
                             budget_usd=snap.get("budget_usd"),
                             note="an explicit fleet watch already owns this box "
                                  "— auto-adopt never overwrites a chosen "
                                  "profile/budget")
                return owner_w
        if standing and profile != "jobs":
            # `standing` is a QUEUE concept. `serve` has no queue (serve_mode
            # strips the drain exits outright), `run` ends on a terminal event,
            # and `bare` has no ladder to keep armed — for all three the flag
            # would name a transition that cannot happen.
            raise ValueError("--standing applies to the `jobs` profile only "
                             "(it survives a QUEUE DRAIN; serve has no queue, "
                             "run ends on a terminal event, bare has no ladder)")
        if profile in ("run", "jobs", "serve") and budget_usd is None:
            raise ValueError("budget_usd is required for the run/jobs/serve "
                             "profiles (hard spend cap — same rule as "
                             "`supervise --budget`; serve gained the bid ladder "
                             "2026-08-02, so it moves money too)")
        if global_budget_usd() is not None and profile in ("run", "jobs", "serve") \
                and self._global_breached():
            # N3: over the fleet ceiling, refuse NEW spend-capable watches.
            raise ValueError("fleet budget ceiling reached "
                             f"(FLEETD_GLOBAL_BUDGET_USD) — refusing a new "
                             f"{profile} watch until spend is reconciled")
        redirected_from = None
        with self.lock:
            # S4: one watch per instance. A colliding target is one of two very
            # different things, and telling them apart is a spend-control matter.
            for other, ow in list(self.state["watches"].items()):
                if other == target or str(ow.get("iid") or other) != target:
                    continue
                if adopted or rows.watch_box_iid(ow) != target:
                    # (b) genuine ambiguity (two distinct watches, or a `run`
                    # watch whose label currently resolves here). Refused — but
                    # NOT with "unwatch it first": unwatch drops supervision of
                    # a box that may be running a live job and discards its
                    # accrued spend, which is a far worse outcome than the
                    # registration the operator was attempting.
                    raise ValueError(
                        f"instance {target} is already supervised by watch "
                        f"{other!r} (profile={ow.get('profile')}, budget="
                        f"{ow.get('budget_usd')}). Re-issue this command "
                        f"against {other!r} to change its profile or cap — the "
                        f"supervision and the accrued spend carry over. Do NOT "
                        f"`fleet unwatch` to get past this: that DROPS "
                        f"supervision of a LIVE box and resets its spend "
                        f"accounting; use it only if you want it unsupervised.")
                if ow.get("profile") in POLICY_PROFILES and profile == "bare":
                    # A redirect must never DOWNGRADE the ladder that is holding
                    # a spot box (same rule the safety net follows for adoption).
                    raise ValueError(
                        f"instance {target} is the CURRENT box of "
                        f"{ow.get('profile')} watch {other!r} (its original box "
                        f"was replaced). Refusing to apply profile='bare' here: "
                        f"that would disarm the outbid rescue / replacement "
                        f"ladder. Re-issue with --profile {ow.get('profile')} "
                        f"to change the cap, or address {other!r} directly if "
                        f"you really mean to downgrade it.")
                # (a) `target` IS this watch's current box — the ladder rented
                # or handed off to it. Resolve to the operator's intent: apply
                # the registration to the owning key, keeping spend_usd,
                # created_ts and the replacement linkage.
                redirected_from, target = target, other
                policy = _redirect_policy(ow.get("policy"), policy)
                break
            w = self.state["watches"].get(target) or {}
            # Resolve the DURABLE ceiling before the record is written, against
            # the post-redirect target. `budget_usd` from here on is the
            # ENFORCED cap, not the requested one, and `carried` is the
            # spend-to-date the new watch inherits — the two halves of
            # "remaining headroom, never the original figure".
            cid, budget_usd, carried, ceiling_note = self._resolve_watch_ceiling(
                str(target), profile, budget_usd, adopted, requester, label,
                reset_spend)
            _seed_spend = (carried if (reset_spend and ceiling_note == "explicit")
                           else max(rows._num(w.get("spend_usd")) or 0.0, carried))
            w.update({"target": target, "profile": profile,
                      "budget_usd": budget_usd, "policy": policy or {},
                      "ceiling_id": cid, "ceiling_source": ceiling_note,
                      "requester": requester, "state": "watched",
                      "created_ts": w.get("created_ts") or self.hooks.now(),
                      # The counter starts at the ledger's spend-to-date, not at
                      # zero. Before the ledger existed this read
                      # `w.get("spend_usd", 0.0)` off a record `_end_watch` had
                      # just popped — i.e. 0.0 — so a re-arm at the same figure
                      # granted a whole fresh cap (box 46916278: $10 armed six
                      # times, $60 of real ceiling).
                      # `--reset-spend` clears the WATCH counter too; anything
                      # else would leave the box parked against a ceiling that
                      # says $0 spent.
                      "spend_usd": _seed_spend,
                      "_ceiling_charged_usd": _seed_spend,
                      "paused_until": w.get("paused_until"),
                      "pause_reason": w.get("pause_reason"),
                      # NEVER re-point a watch at its key: for a watch whose
                      # ladder already replaced its box, the key is a destroyed
                      # instance id and writing it here kills the watch two
                      # ticks later (`instance_gone`) — see `rows.watch_box_iid`.
                      "iid": w.get("iid") if profile == "run"
                      else (rows.watch_box_iid(w) or target),
                      "adopted": bool(adopted),
                      # An explicit (re-)registration always lands ARMED: a
                      # standing watch re-armed by hand mid-dormancy starts
                      # supervising on the next tick rather than waiting for a
                      # ticket. `standing_cycles` is history and is deliberately
                      # NOT reset — it counts drains this watch has survived.
                      "standing": (bool(w.get("standing")) if standing is None
                                   else bool(standing)),
                      "standing_dormant": False, "standing_since": None,
                      "standing_wake_pending": False,
                      "dormant": False, "dormant_reason": None,
                      "budget_breach_pending": False, "missing_ticks": 0,
                      "last_action": w.get("last_action"),
                      "last_tick_ts": w.get("last_tick_ts")})
            self.state["watches"][target] = w
            self._ceiling_bind_box(cid, w.get("iid"))
            self.runtime.pop(str(target), None)    # rebuild against the new policy
            self.state["strays"].pop(str(w.get("iid") or target), None)
            self.state["intents"].pop(str(w.get("iid") or target), None)
        self.save()
        if redirected_from is not None:
            # Both ids, always — an autonomous rental that changed which box a
            # cap lands on is exactly the thing the journal exists to record.
            self.journal("watch_redirected", iid=w.get("iid"), target=target,
                         requested=redirected_from, profile=profile,
                         budget_usd=budget_usd, requester=requester,
                         spend_usd=round(w.get("spend_usd", 0.0), 4),
                         note=f"{redirected_from} is watch {target}'s current "
                              f"box (auto-rented replacement / handoff); the "
                              f"registration was applied to the owning watch, "
                              f"preserving its accrued spend and merging its "
                              f"policy")
        self.journal("watch_auto_adopted" if adopted else "watch_registered",
                     iid=w.get("iid"), target=target, profile=profile,
                     budget_usd=budget_usd, requester=requester,
                     standing=w.get("standing") or None,
                     # The cap on the record, WHERE IT CAME FROM, and the
                     # headroom left. `budget_usd` alone was never enough to
                     # tell a $5 the operator typed from a $5 that was silently
                     # a sixth fresh $5.
                     ceiling_id=cid, ceiling_source=ceiling_note,
                     spend_carried_usd=round(carried, 4) or None,
                     remaining_usd=(round(budget_usd - carried, 4)
                                    if budget_usd is not None else None))
        return w if redirected_from is None else dict(w, redirected_from=redirected_from)

    # moved-from: fleetd.Fleet.unwatch
    def unwatch(self, target: object, requester: str | None = None) -> dict[str, Any]:
        w: dict[str, Any] | None
        with self.lock:
            w = self.state["watches"].pop(str(target), None)
            self.runtime.pop(str(target), None)
        if w is None:
            raise KeyError(f"no watch for {target}")
        self.save()
        self.journal("watch_removed", iid=w.get("iid"), target=str(target),
                     requester=requester,
                     note="box left RUNNING; the unwatched safety net re-applies "
                          "after the grace window")
        return w

    # moved-from: fleetd.Fleet.pause
    def pause(self, target: object, seconds: float | None, reason: str | None = None,
              requester: str | None = None) -> dict[str, Any]:
        """S1: suspend ACTIONS. Observation + budget accrual continue; a breach
        during the pause alarms now and parks at expiry. Works on an unwatched
        target too (a pause-only entry the safety net skips)."""
        target = str(target)
        now = self.hooks.now()
        until = None if (seconds is None or seconds <= 0) else \
            now + min(float(seconds), MAX_PAUSE_S)
        with self.lock:
            ent = self.state["watches"].get(target)
            kind = "watch"
            if ent is None:
                ent = self.state["strays"].setdefault(
                    target, {"first_seen_ts": now, "observed_s": 0.0})
                kind = "stray"
            ent["paused_until"] = until
            ent["pause_reason"] = reason
        self.save()
        self.journal("paused" if until else "pause_cleared", iid=target,
                     target=target, kind=kind,
                     until=fleet_state.iso(until) if until else None, reason=reason,
                     requester=requester)
        return {"target": target, "until": until, "kind": kind,
                "until_iso": fleet_state.iso(until) if until else None}

    # moved-from: fleetd.Fleet.request_action
    def request_action(self, target: object, action: str, reason: str | None = None,
                       requester: str | None = None) -> dict[str, Any]:
        """Queue an explicit park/resume for the next tick (executed + journaled
        by the daemon, never by the client)."""
        if action not in ("park", "resume"):
            raise ValueError(f"unknown action {action!r}")
        target = str(target)
        pa = {"action": action, "reason": reason, "requester": requester,
              "ts": self.hooks.now()}
        with self.lock:
            w = self.state["watches"].get(target)
            if w is not None:
                w["pending_action"] = pa
            else:
                s = self.state["strays"].setdefault(
                    target, {"first_seen_ts": self.hooks.now(), "observed_s": 0.0})
                s["pending_action"] = pa
        self.save()
        self.journal(f"{action}_requested", iid=target, reason=reason,
                     requester=requester)
        return {"target": target, "action": action}

    # moved-from: fleetd.Fleet.operator_intent
    def operator_intent(self, target: object, kind: object,
                        requester: str | None = None,
                        reason: str | None = None) -> dict[str, Any]:
        """B2: the workstation CLI tells the daemon what a human is about to do
        (`herdd stop|start|destroy`), BEFORE the vast PUT/DELETE. An
        intent-stopped box is an operator park no matter what the bid ladder
        would infer, so the daemon can never resurrect it."""
        if kind not in ("stop", "start", "destroy"):
            raise ValueError(f"unknown intent {kind!r}")
        target = str(target)
        with self.lock:
            if kind in ("start",):
                self.state["intents"].pop(target, None)
            else:
                self.state["intents"][target] = {
                    "kind": kind, "ts": self.hooks.now(),
                    "requester": requester, "reason": reason}
            w = next((x for t, x in self.state["watches"].items()
                      if str(x.get("iid") or t) == target), None)
            if w is not None and kind == "start" and not w.get("standing_dormant"):
                # NOT for a standing watch: its dormancy is released by a TICKET
                # (`_standing_tick`), and clearing `dormant` here left
                # `standing_dormant` set, so the resume gate stopped running
                # altogether — the watch ticked the ladder while every readout
                # still called it dormant. `start the box, THEN submit` stays
                # safe because the ticket, not the start, is what re-arms.
                w["dormant"] = False
                w["dormant_reason"] = None
        self.save()
        self.journal(f"operator_intent_{kind}", iid=target, requester=requester,
                     reason=reason,
                     watched=None if w is None else w.get("profile"))
        return {"target": target, "kind": kind,
                "watched": bool(w), "profile": (w or {}).get("profile"),
                "note": (f"box is fleet-watched (profile={w['profile']}) — "
                         f"supervision goes DORMANT; `fleet resume` re-arms it"
                         if w is not None and kind == "stop" else None)}

    def ticket_placed(self, target: object, job_id: object = None,
                      source: str | None = None,
                      requester: str | None = None) -> dict[str, Any]:
        """A CLI just wrote a NON-TERMINAL job ticket into this box's queue.

        The wake half of the standing-watch contract. `_standing_tick` infers the
        same fact by polling the queue, but only while the box is LIVE and only
        as well as the B2 listing lets it — and the poll answers `unknown`
        exactly when it matters (N7). `job submit|retarget|requeue` know it
        first-hand at the moment of the write, so they say so.

        Measured 2026-08-27, the night this closes: `jobs_watch_standing_resumed`
        had fired 0 times against 84 drains. Tickets were retargeted onto a
        drained standing box, the box was evicted minutes later, and the dormant
        watch journaled nothing — no bid rescue, no replacement, work stranded.

        NEVER a refusal and never a money move: it sets a flag the next tick
        consumes, and `_standing_tick` still refuses to re-arm a box that is not
        live. A box with no watch, or a watch that is already armed, is a
        no-op that reports itself as one."""
        target = str(target)
        with self.lock:
            w = next((x for t, x in self.state["watches"].items()
                      if str(x.get("iid") or t) == target
                      or str(t) == target), None)
            woken = bool(w is not None and w.get("standing_dormant"))
            if woken:
                assert w is not None                # narrowing; `woken` implies it
                w["standing_wake_pending"] = True
                w["standing_wake_source"] = source or "ticket_placed"
                w["standing_wake_job_id"] = None if job_id is None else str(job_id)
        if woken:
            self.save()
            self.journal("jobs_watch_standing_woken", iid=(w or {}).get("iid"),
                         target=target, job_id=None if job_id is None else str(job_id),
                         source=source, requester=requester,
                         note="a non-terminal ticket was placed on a standing "
                              "watch's box — the ladder re-arms on the next tick "
                              "the box reads LIVE, without waiting on a queue "
                              "poll that an unreadable listing can never answer")
        return {"target": target, "watched": bool(w),
                "profile": (w or {}).get("profile"),
                "standing": bool((w or {}).get("standing")),
                "standing_dormant": bool((w or {}).get("standing_dormant")),
                "woken": woken,
                "note": ("standing watch is dormant — it re-arms on the next "
                         "tick this box reads live" if woken else None)}

    # moved-from: fleetd.Fleet.request_destroy
    def request_destroy(self, target: object, when: str = "now",
                        reason: str | None = None, requester: str | None = None,
                        yes: bool = False,
                        results_check: bool = True) -> dict[str, Any]:
        """The ONLY path to a destroy. Explicit (`--yes`), journaled with
        requester + reason, executed by the daemon, at most once, and (S4) it
        auto-unwatches an actively-watched box first."""
        if not yes:
            raise ValueError("destroy requires yes=True (explicit confirmation)")
        if when not in DESTROY_WHEN:
            raise ValueError(f"unknown when {when!r} (want {DESTROY_WHEN})")
        target = str(target)
        with self.lock:
            self.state["destroys"][target] = {
                "when": when, "reason": reason, "requester": requester,
                "ts": self.hooks.now(), "executed": False, "cond_streak": 0,
                "results_check": bool(results_check)}
            dropped = [t for t, w in self.state["watches"].items()
                       if str(w.get("iid") or t) == target]
            for t in dropped:
                self.state["watches"].pop(t, None)
                self.runtime.pop(t, None)
        self.save()
        self.journal("destroy_requested", iid=target, when=when, reason=reason,
                     requester=requester, unwatched=", ".join(dropped) or None)
        return {"target": target, "when": when, "unwatched": dropped}

    # moved-from: fleetd.Fleet._require_watch
    def _require_watch(self, target: object) -> dict[str, Any]:
        w: dict[str, Any] | None = self.state["watches"].get(str(target))
        if w is None:
            raise KeyError(f"no watch for {target}")
        return w

    # -------------------------------------------------------------- the tick #
    # moved-from: fleetd.Fleet.tick
    def tick(self) -> None:
        """ONE reconcile pass over the whole fleet (FLEETD_DESIGN §3).

        The ORDER below is policy, not sequence: notify -> destroys -> per-watch
        -> strays -> global budget -> `last_ok_tick_ts` -> alarm transitions ->
        save. Moving the stray sweep before the per-watch pass would let it
        adopt a box a watch is about to claim; moving `last_ok_tick_ts` earlier
        would let the stray alarms fire off a reading this tick has not made."""
        now = self.hooks.now()
        self._maybe_reload_env()
        instances = self.hooks.instances()
        if instances is None:                       # API blip: change NOTHING
            # `last_tick_ts` is the last SUCCESSFUL reconcile, never the last
            # attempt: bumping it here made `tick_age_s` read fresh while every
            # reading beside it was frozen at the last good tick — a status
            # block that lies about its own freshness. The outage is now its own
            # (self-retracting) alarm instead.
            self.state["meta"].setdefault("api_unavailable_since", now)
            self.journal("api_unavailable",
                         note="skipping this tick; an unreadable API is not an "
                              "empty fleet (N7: clocks do not advance)")
            return
        self.state["meta"].pop("api_unavailable_since", None)
        self._record_machine_ids(instances, now)
        self.last_tick_ts = now
        with self.tick_lock:
            self._ticks += 1
            last_ok = self.state["meta"].get("last_ok_tick_ts")
            # N7: age/grace clocks advance on OBSERVED time only, capped, so an
            # outage or a sleeping workstation can never fast-forward a park.
            obs_dt = 0.0 if not last_ok else min(max(0.0, now - last_ok),
                                                 MAX_OBS_DT_S)
            by_iid = {str(i.get("id")): i for i in instances}
            if (self._health_ts is None
                    or now - self._health_ts >= HEALTH_EVERY_S):
                # Stamped whether or not the fold raised: a failing B2 read must
                # not turn into a retry every tick.
                self._health_ts = now
                try:
                    self._health = self.hooks.health(instances) or {}
                except Exception:
                    self._health = {}
            self._tick_notify(now)
            self._tick_destroys(by_iid, now)
            for target in list(self.state["watches"].keys()):
                try:
                    self._tick_watch(target, by_iid, now, obs_dt)
                except Exception as e:              # one bad watch never stops the fleet
                    self.journal("watch_error", target=target,
                                 error=f"{type(e).__name__}: {e}")
                    w = self.state["watches"].get(target)
                    if w is not None:               # derived: cleared by the next
                        w["last_tick_error"] = f"{type(e).__name__}: {e}"
            self._tick_strays(by_iid, now, obs_dt)
            self._tick_global_budget()
            self.state["meta"]["last_ok_tick_ts"] = now
            self._journal_alarm_transitions(now)
            self.save()

    # ------------------------------------------------- notifications (S2a) --
    # ONE authed GET per tick against vast's notification inbox, journaled and
    # nothing else. NOTIFY_DESIGN D2 is the boundary and it is absolute here:
    # these rows are EVIDENCE, they feed no classifier, no bid, no park and no
    # rental. Wiring them into the eviction classifier and the rescue ladder is
    # S2b, which re-enters the adversarial review before deploy. Until then the
    # only thing that changes when this poll fails — or when the hidden endpoint
    # is retired — is that a `notify_seen` line stops appearing.
    # moved-from: fleetd.Fleet._notify_state
    def _notify_state(self) -> dict[str, Any]:
        ns = self.state.get("notify")
        if not isinstance(ns, dict):
            ns = {}
            self.state["notify"] = ns
        return ns

    # moved-from: fleetd.Fleet._notify_health
    def _notify_health(self, ok: bool, err: object = None) -> None:
        """Journal poll health on TRANSITION only (ok -> failing, failing -> ok).

        A retired endpoint answers 404 on every tick forever, and a per-tick
        announcement of one unchanging fact is the exact defect FLEET_REVIEW
        item 6 found twice (158 events for 2 facts; 79 identical refusals in
        66 min). The condition still shows continuously in `fleet status`
        state — what is rate-limited is the CLAIM, not the knowledge."""
        ns = self._notify_state()
        was_ok = bool(ns.get("poll_ok", True))
        now = self.hooks.now()
        if ok:
            fails = int(ns.get("consecutive_failures") or 0)
            since = ns.get("failing_since")
            ns["poll_ok"] = True
            ns["last_ok_ts"] = now
            ns.pop("consecutive_failures", None)
            ns.pop("failing_since", None)
            ns.pop("fail_error", None)
            if not was_ok:
                self.journal(notify.POLL_ERROR_EVENT, state="ok",
                             failures=fails or None,
                             failing_s=(round(now - since, 1)
                                        if since is not None else None),
                             note="notification poll recovered")
            return
        gone = notify.is_gone(err)  # type: ignore[no-untyped-call]
        ns["poll_ok"] = False
        ns["consecutive_failures"] = int(ns.get("consecutive_failures") or 0) + 1
        ns["fail_error"] = str(err)
        ns.setdefault("failing_since", now)
        if was_ok:
            self.journal(
                notify.POLL_ERROR_EVENT, state="failing", error=str(err),
                gone=(True if gone else None),
                note=("the HIDDEN inbox endpoint is GONE (NOTIFY_DESIGN §1.3 "
                      "always said it could be) — fleetd degrades to exactly "
                      "its pre-notify behavior"
                      if gone else
                      "notification poll failing; reconcile is unaffected (D2)"))

    # moved-from: fleetd.Fleet._tick_notify
    def _tick_notify(self, now: float) -> None:
        """Poll the inbox, journal what we have never seen (D4). Never raises.

        Failure of ANY kind here — hook missing, transport dead, endpoint
        retired, payload a shape we have never met — must cost the tick
        nothing, so the whole body is defensive and the caller's contract is
        "returns"."""
        if not notify_enabled():
            return
        hook = getattr(self.hooks, "notifications", None)
        if hook is None:                    # an older Hooks/fake: no poll, no noise
            return
        try:
            payload, err = hook()
        except Exception as e:              # a hook that raises is a poll failure
            payload, err = None, f"error {type(e).__name__}: {e}"
        if payload is None:
            self._notify_health(False, err)
            return
        ns = self._notify_state()
        try:
            res = notify.poll(payload,  # type: ignore[no-untyped-call]
                              ns.get("cursor"))
        except Exception as e:              # `poll` is total by construction, so
            self._notify_health(              # this is a bug, not a bad row — and
                False, f"unreadable inbox payload: {type(e).__name__}: {e}")
            return                            # it announces ONCE, like any other
        # A poll that answered AND parsed is the only definition of healthy: a
        # payload we cannot read is a failing poll, not a recovered one, or the
        # health flag would flap ok/failing every tick and announce both.
        self._notify_health(True)
        ns["cursor"] = res.cursor
        ns["last_poll_ts"] = now
        ns["rows_seen"] = res.rows_seen
        ns["seen_total"] = int(ns.get("seen_total") or 0) + len(res.new)
        if res.gap:
            ns["gaps"] = int(ns.get("gaps") or 0) + 1
            self.journal(notify.GAP_EVENT, rows=res.rows_seen,
                         window=notify.WINDOW,
                         note="a FULL window and not one row we had already "
                              "seen: notifications may have been missed. The "
                              "LABELING has a hole; reconcile is unaffected")
        for row in res.new:
            self.journal(notify.SEEN_EVENT,
                         **notify.journal_fields(row))  # type: ignore[no-untyped-call]
        # The OUTBID lookaside (S2b, §6.3): the small, bounded, aged-out set of
        # displacement records an eviction observed in the next few ticks may be
        # matched against. Retained unconditionally — retention is not policy,
        # and a switch flipped between the row and the stop it explains should
        # not lose the row — while whether anything READS it is
        # `notify_policy_enabled()`. Journaled rows stay the record of truth
        # (D4); this is a working set, not a second copy of the feed.
        ns["outbid"] = notify.retain_outbid(  # type: ignore[no-untyped-call]
            ns.get("outbid"),
            notify.outbid_evidence_rows(res.new),  # type: ignore[no-untyped-call]
            now)

    # moved-from: fleetd.Fleet._notify_feed
    def _notify_feed(self, jc: dict[str, Any]) -> None:
        """Hand this tick's outbid lookaside to the jobs ladder — or take it
        away (S2b, §6.3).

        THE ONE SEAM. Everything S2b added downstream — the classifier's
        `notify` argument, the rescue quote, all four journal events — is
        reachable only through `jc["notify_rows"]`, and this is the only line
        that writes it. So the deploy gate is one predicate in one place, and
        "S2b off" is not a promise about a dozen call sites but a fact about
        one: with the switch off the key is REMOVED (not emptied — a ladder
        restored from a state file must not inherit a stale feed either), and
        every downstream path takes its pre-S2b branch.

        Matching itself stays in the ladder (`supervise.replacement`'s
        `_job_notify_match`), which is where the eviction cycle, its latch and
        its clock live; what fleetd owns is the poll and the lookaside. The
        whole set is handed over, not a per-box slice: the ladder may swap the
        box under us mid-tick, and a list filtered against the box we THOUGHT we
        were supervising is the exact shape of a row matched to the wrong
        instance.

        **OFF DISARMS THE LATCH, not only the feed** (review round 1, 2-1/M1).
        The first cut popped `notify_rows` alone, and that was not the switch it
        claimed to be: `notify_matched` is DURABLE state, so a box that latched
        a row while armed kept pricing its rescue off that row with the switch
        off — in-process after an `.env` hot-reload, and again after a restart,
        for as long as the box stayed down (the latch clears only on
        return-to-live or a box swap). Demonstrated twice, independently, with
        the gate off throughout and no rows fed: a real PUT at $1.212. That is
        precisely the emergency-off path — turn it on, see something wrong, set
        the env to 0 — so it is the one state the switch has to be right in.

        `notify_consumed_ids` is deliberately NOT popped. It is dedup MEMORY,
        not evidence: forgetting it across a gate flap would let re-arming
        re-match and re-price a row an earlier cycle already spent, which is the
        2-2 defect arriving through the switch instead of through the window.
        Keeping it can only ever refuse a match."""
        if not (notify_enabled() and notify_policy_enabled()):
            jc.pop("notify_rows", None)
            jc.pop("notify_matched", None)
            jc.pop("notify_quote_said", None)
            return
        jc["notify_rows"] = list(self._notify_state().get("outbid") or [])

    # --- destroys (explicit only; deferred conditions re-checked every tick) --
    # moved-from: fleetd.Fleet._tick_destroys
    def _tick_destroys(self, by_iid: dict[str, dict[str, Any]], now: float) -> None:
        for iid, req in list(self.state["destroys"].items()):
            if req.get("executed"):
                continue
            if now - req.get("ts", now) > DESTROY_TTL_S:          # S3 TTL
                self.state["destroys"].pop(str(iid), None)
                self.journal("destroy_expired", iid=iid, when=req.get("when"),
                             requester=req.get("requester"),
                             note="condition never held inside the TTL")
                # LATCHED: the request is gone, so no later tick can re-derive
                # this. A destroy an operator asked for and never got is exactly
                # the thing that must not scroll past once and vanish.
                self.latch_alarm(f"destroy:{iid}:expired",
                                 f"{iid}: destroy request EXPIRED unexecuted after "
                                 f"{int(DESTROY_TTL_S / 3600)}h "
                                 f"(when={req.get('when')}, "
                                 f"by={req.get('requester')}) — the box is STILL "
                                 f"RUNNING; re-request or ack", iid=iid)
                continue
            inst = by_iid.get(str(iid))
            if inst is None:
                self.state["destroys"].pop(str(iid), None)
                self.journal("destroy_skipped", iid=iid, reason="already gone")
                continue
            when = req.get("when", "now")
            req["held_reason"] = None               # derived alarms: re-earned
            req["last_error"] = None                # every tick, never latched
            cond_ok, snapshot = self._destroy_condition(iid, inst, when)
            if not cond_ok:
                req["cond_streak"] = 0
                req.pop("cond_since", None)
                continue
            req["cond_streak"] = req.get("cond_streak", 0) + 1
            held_s = now - float(req.setdefault("cond_since", now))
            # S3: a DEFERRED condition must hold on two consecutive observations
            # AND for DESTROY_CONFIRM_S (an explicit `--when now` is the
            # operator's own confirmation). At a 15s tick the count alone would
            # buy 15s of confidence for a destroy.
            if when != "now" and (req["cond_streak"] < DESTROY_CONFIRM_OBS
                                  or held_s < DESTROY_CONFIRM_S):
                self.journal("destroy_condition_pending", iid=iid, when=when,
                             streak=req["cond_streak"], held_s=round(held_s, 1),
                             snapshot=snapshot)
                continue
            if req.get("results_check", True) and when != "now":
                rp = self.hooks.results_present(iid)
                if rp is False:
                    req["cond_streak"] = 0
                    req.pop("cond_since", None)
                    self.journal("destroy_deferred_no_results", iid=iid,
                                 note="job results not published to B2 — pass "
                                      "results_check=false to override")
                    req["held_reason"] = ("results missing on B2 (pass "
                                          "results_check=false to override)")
                    continue
            fresh = self.hooks.instance(iid)                      # S3 re-stat
            if fresh is None:
                self.state["destroys"].pop(str(iid), None)
                self.journal("destroy_skipped", iid=iid, reason="gone at re-stat")
                continue
            ok, err = self.hooks.destroy(iid)
            req["executed"] = bool(ok)              # at most once; retry if it failed
            self.journal("destroyed" if ok else "destroy_failed", iid=iid,
                         when=when, reason=req.get("reason"),
                         requester=req.get("requester"), snapshot=snapshot,
                         error=None if ok else err)
            if ok:
                with self.lock:
                    self.state["destroys"].pop(str(iid), None)
                    for target, w in list(self.state["watches"].items()):
                        if str(w.get("iid") or target) == str(iid):
                            self.state["watches"].pop(target, None)
                            self.runtime.pop(target, None)
                    self.state["strays"].pop(str(iid), None)
            else:
                req["last_error"] = err             # derived while the request lives

    # moved-from: fleetd.Fleet._destroy_condition
    def _destroy_condition(self, iid: object, inst: dict[str, Any],
                           when: str) -> tuple[bool, dict[str, Any]]:
        status = (inst.get("actual_status") or "").lower()
        if when == "now":
            return True, {"status": status}
        if when == "parked":
            return status in PARKED_STATES, {"status": status}
        drained = self.hooks.drained(iid)
        return drained is True, {"status": status, "drained": drained}

    # --- one watch ------------------------------------------------------------
    # moved-from: fleetd.Fleet._tick_watch
    def _tick_watch(self, target: str, by_iid: dict[str, dict[str, Any]],
                    now: float, obs_dt: float) -> None:
        w = self.state["watches"].get(target)
        if w is None:
            return
        iid = self._resolve_iid(w, by_iid)
        inst = by_iid.get(str(iid)) if iid else None
        w["last_tick_ts"] = now
        w["last_tick_error"] = None                 # a tick that runs retracts it

        pa = w.pop("pending_action", None)          # explicit request outranks policy
        if pa:
            self._exec_action(iid or target, pa, inst, w)
            w["last_action"] = pa["action"]
            return

        # S4: an IID watch dies with its instance (a run watch keeps looking —
        # its box comes back under the same label after a relaunch).
        if inst is None and w["profile"] != "run":
            # ...UNLESS a handoff is mid-flight with a live understudy (defect
            # #61, 2026-08-08). `handoff_poll` returns `complete` on the tick
            # AFTER `drain_primary` destroys the primary — and this early return
            # is what that tick used to hit instead, so the watch ended
            # `instance_gone` at GONE_CONFIRM_TICKS and the understudy was left
            # with no watch and no budget cap for the stray sweep to adopt as an
            # uncapped `bare` box. Deterministic, not a race: the ladder could
            # never reach its own promotion step under the daemon.
            #
            # So while the ladder holds a LIVE understudy, tick it anyway. The
            # jobs ladder already handles a missing primary (the fence gates skip
            # the stop-classify and queue exits), `complete` promotes the
            # understudy into `jc["iid"]`, and the code below copies that onto
            # `w["iid"]` — the understudy inherits THIS watch, with its budget,
            # its spend to date and its replacement counters. Bounded by the
            # handoff's own deadlines: an understudy that dies or never produces
            # aborts, the phase returns to IDLE, and the next tick lands here
            # again with nothing to carry.
            if not self._handoff_in_flight(target, by_iid):
                w["missing_ticks"] = w.get("missing_ticks", 0) + 1
                if w["missing_ticks"] >= GONE_CONFIRM_TICKS:
                    self._end_watch(target, w, "instance_gone", None)
                return
            self.journal("jobs_handoff_carryover", iid=w.get("iid"), target=target,
                         note="primary is gone and a handoff understudy is live "
                              "— ticking the ladder so the migration can complete "
                              "and the understudy inherits this watch's budget")
        w["missing_ticks"] = 0

        # B2: operator intent beats every inferred classification.
        intent = self.state["intents"].get(str(iid))
        if intent and intent.get("kind") in ("stop", "destroy"):
            if not w.get("dormant"):
                w["dormant"] = True
                w["dormant_reason"] = f"operator_{intent['kind']}"
                w["state"] = "dormant"
                self.journal("watch_dormant", iid=iid, target=target,
                             reason=w["dormant_reason"],
                             requester=intent.get("requester"),
                             note="operator intent — the bid ladder will NOT "
                                  "rescue this box; `fleet resume` re-arms")
            return
        if w.get("standing_dormant") and not w.get("dormant"):
            # REPAIR, for records already on disk: standing dormancy is released
            # by a TICKET (`_standing_tick`), never by liveness, so anything that
            # cleared `dormant` alone stranded the standing flags — the gate
            # below never fired, `jobs_watch_standing_resumed` never journaled
            # (0 of 84 drains, measured 2026-08-27) and `fleet status` reported
            # `state=standing` for a watch that was in fact ticking the ladder.
            # `operator_intent(start)` was that path; it no longer clears it.
            w["dormant"] = True
            w["dormant_reason"] = w.get("dormant_reason") or "standing_drained"
            w["state"] = "standing"
        if w.get("dormant"):
            live = inst is not None and \
                (inst.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES
            if w.get("standing_dormant"):
                # The dormant-but-ARMED phase of a standing watch. It owns its
                # own re-arm rule (a TICKET, not mere liveness) and keeps
                # accruing meanwhile; returning False means "still dormant".
                if not self._standing_tick(target, w, iid, inst, live, intent, now):
                    return
            elif live and not intent:               # somebody resumed it: re-arm
                w["dormant"] = False
                w["dormant_reason"] = None
                w["state"] = "watched"
                self.journal("watch_rearmed", iid=iid, target=target)
            else:
                return                              # S8: dormant boxes do not alarm

        if w.get("paused_until"):
            if now >= w["paused_until"]:
                w["paused_until"] = None
                w["pause_reason"] = None
                w["state"] = "watched"
                self.journal("pause_expired", iid=iid, target=target,
                             note="supervision resumed automatically")
                if w.pop("budget_breach_pending", False):
                    self._park_on_budget(w, target, inst)   # S1: park at expiry
                    return
            else:
                w["state"] = "paused"
                self._accrue(w, inst, now)          # S1: observation + accrual go on
                if self._budget_breached(w):
                    w["budget_breach_pending"] = True    # derived: pause + pending
                self.journal("tick_paused", iid=iid, target=target,
                             left_s=round(w["paused_until"] - now, 1),
                             spend_usd=round(w.get("spend_usd", 0.0), 4),
                             reason=w.get("pause_reason"))
                return

        self._health_alarm(iid)                     # N1: advisory only

        # ...and the ONE health-shaped condition that is not advisory. Ordered
        # beside the budget cap on purpose: both are "this box may no longer
        # bill" rules, both park, neither destroys (FLEETD_DESIGN §3/§8).
        if self._pyhalf_tick(target, w, iid, inst, now):
            return

        if self._budget_breached(w):
            self._park_on_budget(w, target, inst)
            return

        if self._global_breached():
            # N3: over the fleet ceiling nothing new is spent, nothing is parked.
            # The alarm is the ONE fleet-level line (`fleet:budget`, derived from
            # spend_by_box vs the cap) — it already says money moves are
            # suspended, and repeating it per watch just buries the rest.
            self._accrue(w, inst, now)
            self.journal("tick_suspended_global_budget", iid=iid, target=target)
            return

        if w["profile"] in ("run", "jobs", "serve"):
            # serve joined the policy tier 2026-08-02: same jobs ladder in
            # serve_mode (bid defend/rescue for a spot endpoint), replacing the
            # old observe-and-alarm-only serve watch.
            self._tick_policy_watch(target, w, now, inst)
        else:
            self._tick_simple_watch(target, w, inst, now)

    # moved-from: fleetd.Fleet._handoff_in_flight
    def _handoff_in_flight(self, target: str,
                           by_iid: dict[str, dict[str, Any]]) -> bool:
        """Is this watch's ladder holding a LIVE handoff understudy right now?

        The predicate that keeps a jobs watch alive across its primary's destroy
        (defect #61). Deliberately narrow: a runtime that exists, a non-IDLE
        handoff phase, an understudy id, and that box PRESENT and live in this
        tick's listing. Anything softer would keep a dead watch on the books
        forever, which is the failure mode `instance_gone` exists to prevent."""
        rt = self.runtime.get(target)
        hf = (rt or {}).get("hf")
        if not hf or hf.get("phase") in (None, "IDLE"):
            return False
        u = hf.get("understudy_iid")
        if u is None:
            return False
        inst = by_iid.get(str(u))
        return inst is not None and \
            (inst.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES

    # moved-from: fleetd.Fleet._resolve_iid
    def _resolve_iid(self, w: dict[str, Any],
                     by_iid: dict[str, dict[str, Any]]) -> str | None:
        if w["profile"] == "run":
            run_id = w["target"].split(":", 1)[-1]
            for iid, inst in by_iid.items():
                if models._instance_run_label(inst) == run_id:
                    w["iid"] = iid
                    return iid
            return w.get("iid")
        current: str | None = w.get("iid") or w["target"]
        return current

    # moved-from: fleetd.Fleet._exec_action
    def _exec_action(self, iid: object, pa: dict[str, Any],
                     inst: dict[str, Any] | None = None,
                     w: dict[str, Any] | None = None) -> None:
        if pa["action"] == "park":
            ok, err = self._park(iid, inst, why=pa.get("reason") or "requested",
                                 requester=pa.get("requester"))
            if w is not None and ok:
                w["dormant"] = True                 # S8: our own park is not an
                w["dormant_reason"] = "fleet_park"  # eviction to be rescued
                w["state"] = "parked"
        else:
            ok, err = self.hooks.resume(iid)
            self.journal("resumed" if ok else "resume_failed", iid=iid,
                         reason=pa.get("reason"), requester=pa.get("requester"),
                         error=None if ok else err)
            if w is not None and ok:
                w["dormant"] = False
                w["dormant_reason"] = None
                w["state"] = "watched"
                self.state["intents"].pop(str(iid), None)
        # LATCHED: `pending_action` was consumed by this tick, so a failed
        # park/resume leaves nothing behind to re-derive — and an operator's
        # explicit park that silently did NOT happen is the last thing that may
        # blink once and disappear. A later action on the same box that DOES
        # succeed retracts it.
        key = f"action:{iid}:failed"
        if ok:
            self.clear_alarm(key, reason=f"{pa['action']} succeeded")
        else:
            self.latch_alarm(key, f"{iid}: {pa['action']} FAILED ({err}) — "
                                  f"requested by {pa.get('requester') or 'fleetd'}"
                                  f"; the box did NOT change state", iid=iid)

    # moved-from: fleetd.Fleet._park
    def _park(self, iid: object, inst: dict[str, Any] | None, why: str,
              requester: str | None = None, *,
              graded_keep: bool = False) -> tuple[bool, str | None]:
        """A fleetd park is a RESUMABILITY PROMISE, so stamp the reap keep-token
        first (B4) — `herdd reap` destroys stopped boxes idle > 2h without it.
        Best-effort on spot: the host can re-rent the GPUs
        (parked-spot-box-reclaimed), so the durable artifact is the checkpoint,
        not the box.

        `graded_keep=True` says NOBODY promised to bring this box back (the
        pyhalf and unwatched-safety-net parks), so the stamp is graded by
        `keep_stamp_needed` and skipped for a box holding nothing — item 1 of
        `FLEET_REVIEW_2026-08-20.md`, since an unconditional token bills
        allocated disk until a human destroys it. DEFAULT OFF: a budget cap or
        an operator's own park always stamps, and so does any future caller."""
        drained: bool | None = None
        results: bool | None = None
        if graded_keep:
            try:
                drained, results = (self.hooks.drained(iid),
                                    self.hooks.results_present(iid))
            except Exception:               # an unreadable queue keeps the box
                drained = results = None
        if not graded_keep or keep_stamp_needed(drained, results):
            changed, label = self.hooks.keep_label(iid, inst)
            if changed:
                self.journal("keep_label_stamped", iid=iid, label=label,
                             note="reap (destroys idle stopped boxes > 2h) now "
                                  "honors this park")
        else:
            self.journal("keep_label_skipped", iid=iid, reason=why,
                         drained=drained, results_present=results,
                         label=(inst or {}).get("label"),
                         note="queue all-terminal and its results published to "
                              "B2 — nobody promised this box back, so it is "
                              "left to the 2h idle reaper instead of billing "
                              "allocated disk on a permanent keep token")
        ok, err = self.hooks.park(iid)
        self.journal("parked" if ok else "park_failed", iid=iid, reason=why,
                     requester=requester, error=None if ok else err)
        return ok, err

    # --- serve/bare: budget accounting + park on breach, no bid moves ---------
    # moved-from: fleetd.Fleet._accrue
    def _accrue(self, w: dict[str, Any], inst: dict[str, Any] | None,
                now: float) -> None:
        """Billing-model-faithful accrual, persisted every tick (S2):
        dph x elapsed while `running`; STORAGE rate only while loading/created.

        Vast does NOT bill GPU during `loading` — invoice-verified 2026-07-20
        (bid) and 2026-08-02 (on-demand 46633685: GPU hours 0.000 for a 31-min
        all-`loading` life, invoiced $0.041 storage while THIS accrual said
        $0.155 at full dph). That overstatement was quoted as a billing
        measurement, so the accrual now mirrors the invoice: the GPU rate
        starts at the loading→running flip; before it only the disk bills."""
        prev = w.get("_spend_ts")
        w["_spend_ts"] = now
        if inst is None:
            w["_was_live"] = False
            return
        status = (inst.get("actual_status") or "").lower()
        live = status in bidpolicy.LIVE_STATES
        dph = models._num_dph(inst.get("dph_total"))
        if dph:
            w["_last_dph"] = dph
        if live and prev:
            if status == "running":
                rate = dph
            else:                         # loading/created: storage only
                sd = models._storage_day(inst)          # $/day or None
                rate = (sd / 24.0) if sd is not None else None
            if rate:
                w["spend_usd"] = w.get("spend_usd", 0.0) \
                    + rate * (now - prev) / 3600.0
        w["_was_live"] = live
        self._charge_ceiling(w)
        self.state["spend_by_box"][str(w.get("iid") or w["target"])] = \
            round(w.get("spend_usd", 0.0), 4)

    # moved-from: fleetd.Fleet._tick_simple_watch
    def _tick_simple_watch(self, target: str, w: dict[str, Any],
                           inst: dict[str, Any] | None, now: float) -> None:
        if inst is None:
            w["state"] = "gone"
            self.journal("tick", iid=w.get("iid"), target=target,
                         profile=w["profile"], state="gone")
            return
        self._accrue(w, inst, now)
        live = (inst.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES
        w["state"] = "live" if live else "not_live"
        w["last_status"] = inst.get("actual_status")
        # N2 (serve not live) and the unbudgeted-adoption alarm are DERIVED from
        # these fields — see _derive_watch_alarms. They are never appended here,
        # so `fleet watch`-ing an adopted box retracts its alarm immediately
        # rather than at the next tick.
        if self._budget_breached(w):
            self._park_on_budget(w, target, inst)
            return
        # Disk allocated-vs-used rides every tick. Two reasons it belongs here
        # rather than in a one-shot audit: (1) storage bills on the ALLOCATED
        # size, so `disk_gb` is the number that costs money and `disk_used_gb`
        # is the only evidence it was the right one — both already ride the
        # instances payload, so this is free; (2) sampling at the tick cadence
        # captures the HIGH-WATER MARK. A box that unpacks a multi-GB tarball
        # and then deletes it (fetch_eval_env.sh, eval_sidecar.sh) peaks well
        # above its steady state, and any post-hoc single reading of
        # `disk_usage` misses that peak — which would under-size every box the
        # sizing estimator later provisions. Take max() over a run's ticks.
        d_alloc, d_used = models._disk_gb(inst)
        self.journal("tick", iid=w.get("iid"), target=target,
                     profile=w["profile"], state=w["state"],
                     spend_usd=round(w.get("spend_usd", 0.0), 4),
                     ceiling_spend_usd=self._ceiling_tick_field(w),
                     disk_gb=d_alloc, disk_used_gb=d_used)

    # moved-from: fleetd.Fleet._budget_breached
    def _budget_breached(self, w: dict[str, Any]) -> bool:
        """Breach is measured against the CEILING's cumulative spend, not this
        watch's own counter. Those differ exactly when a ceiling has outlived a
        watch (a lapse, a re-arm) or is shared with a successor box — which is
        every case the durable ceiling exists for."""
        b = w.get("budget_usd")
        return b is not None and self._ceiling_spend(w) >= b

    # moved-from: fleetd.Fleet._global_breached
    def _global_breached(self) -> bool:
        cap = global_budget_usd()
        if cap is None:
            return False
        return bool(sum(self.state["spend_by_box"].values()) >= cap)

    # moved-from: fleetd.Fleet._pyhalf_tick
    def _pyhalf_tick(self, target: str, w: dict[str, Any], iid: object,
                     inst: dict[str, Any] | None, now: float) -> bool:
        """TEETH, narrowly. Park a jobs box that has SELF-REPORTED its python
        half dead for longer than the confirm window. Returns True when it
        parked (the caller stops ticking the watch).

        Why this trigger and not the zombie verdict (FAILCLOSED_DESIGN §8): the
        obvious move after 47737955 is to give fleetd's existing
        ZOMBIE_NO_JOBD alarm teeth, since it fired correctly at T+20min. It was
        correct BY ACCIDENT. That verdict is reached from JOBD_STATUS staleness,
        and JOBD_STATUS used to be stamped only on transitions — so it goes
        stale on EVERY healthy idle jobs box too. Enforcing on it would have
        parked healthy rented boxes, which is the same bug pointing the other
        way. `boxes.health._jobd_heartbeat_epoch_soft` says as much in its own
        docstring, and its justification for tolerating that false positive
        ("jobd self-parks an idle box anyway") was itself falsified by the
        idle-park inversion this campaign fixed.

        `pyhalf=broken` is a different kind of evidence: not an inference from
        silence but a CONFESSION, written by the box after a deterministic,
        offline capability check failed. Its false-positive rate is the
        selftest's, which is a local import that no network condition can
        redden. So the teeth bite on the confession, and the inference stays an
        alarm.

        Three fail-open guards, all deliberate:
          * unreadable marker (None) NEVER accumulates — a B2 outage makes every
            box unreadable at once, and that must not be a fleet-wide park.
          * `pyhalf=ok` or a bundle too old to carry the field clears the clock.
          * a confirm window of <= 0 disables enforcement entirely.
        """
        if w.get("profile") != "jobs" or not iid:
            return False
        confirm = pyhalf_confirm_s()
        if confirm <= 0:
            return False
        # getattr, not a bare call: the suite's FakeHooks is a scripted
        # stand-in that does not inherit from Hooks, so an older or narrower
        # double simply leaves the teeth disarmed rather than raising through
        # the whole tick. Absent hook == no evidence == no enforcement.
        reader = getattr(self.hooks, "jobd_status_line", None)
        if reader is None:
            return False
        broken = pyhalf_broken(reader(iid))
        if broken is not True:
            # ok, or old bundle, or unreadable: forget any partial confirmation.
            if w.pop("_pyhalf_since", None) is not None:
                self.clear_alarm(f"pyhalf:{iid}")
            return False
        since = w.get("_pyhalf_since")
        if since is None:
            w["_pyhalf_since"] = now
            self.journal("pyhalf_broken_seen", iid=iid, target=target,
                         confirm_s=confirm,
                         note="box self-reports its python half dead; it can "
                              "emit no lifecycle events. Parking if it persists")
            return False
        held = now - since
        self.latch_alarm(
            f"pyhalf:{iid}",
            f"{iid}: PYHALF BROKEN {int(held)}s — the box reports it cannot run "
            f"jobd.py at all (no events can be emitted). Parking at "
            f"{int(confirm)}s; `herdd ssh {iid}` to read the reason field")
        if held < confirm:
            return False
        # graded_keep: a box wedged mid-work has non-terminal tickets and keeps
        # its disk (the reason + the bundle); one wedged after everything
        # shipped holds nothing worth billing for.
        ok, err = self._park(iid, inst, why="pyhalf_broken", graded_keep=True)
        w["state"] = "pyhalf_parked" if ok else "pyhalf_park_failed"
        w["last_action"] = "pyhalf_park"
        w["last_error"] = None if ok else err
        w["dormant"] = bool(ok)
        w["dormant_reason"] = "fleet_park" if ok else None
        self.journal("pyhalf_parked" if ok else "pyhalf_park_failed", iid=iid,
                     target=target, held_s=round(held, 1), confirm_s=confirm,
                     error=None if ok else err,
                     note="a box that can neither claim work nor report on it "
                          "was billing; parked, NOT destroyed — the disk holds "
                          "the reason and the bundle that caused it")
        # True even when the park FAILED, matching _park_on_budget's caller: once
        # a "may no longer bill" rule fires, the watch stops ticking either way.
        # Returning False here let _tick_policy_watch run on and overwrite
        # w["state"] with "watched", erasing the only durable record that the
        # box is wedged. The latched alarm keeps burning and the next tick
        # retries the park, which is what a transient refusal needs.
        return True

    # moved-from: fleetd.Fleet._park_on_budget
    def _park_on_budget(self, w: dict[str, Any], target: str,
                        inst: dict[str, Any] | None = None) -> None:
        """FLEETD_DESIGN §3: a budget breach PARKS and alarms — resumable
        (best-effort on spot), never a destroy, never a `failed` terminal."""
        iid = w.get("iid") or target
        ok, err = self._park(iid, inst, why="budget_cap")
        w["state"] = "budget_parked" if ok else "budget_park_failed"
        w["last_action"] = "budget_park"
        w["last_error"] = None if ok else err
        w["dormant"] = bool(ok)                     # S8
        w["dormant_reason"] = "fleet_park" if ok else None
        self.journal("budget_parked" if ok else "budget_park_failed", iid=iid,
                     target=target, spend_usd=round(w.get("spend_usd", 0.0), 4),
                     ceiling_id=w.get("ceiling_id"),
                     ceiling_source=w.get("ceiling_source"),
                     ceiling_spend_usd=round(self._ceiling_spend(w), 4),
                     budget_usd=w.get("budget_usd"), error=None if ok else err)
        # The alarm is DERIVED from `state` (budget_parked / budget_park_failed)
        # and so keeps burning for as long as the box is actually capped —
        # INTENTIONALLY outliving the S8 dormancy silence, because "fleetd
        # parked your box at its cap" is a standing condition, not an event. It
        # goes out when a human raises the cap, resumes, or drops the watch.

    # moved-from: fleetd.Fleet._health_alarm_msg
    def _health_alarm_msg(self, iid: object) -> str | None:
        """The health alarm line for `iid`, or None. Pure: reads the cached
        `gather_fleet_health` verdicts (refreshed every HEALTH_EVERY_S — this
        alarm is therefore only as fresh as that cache, which is the one alarm
        here that can lag its condition by that long) and returns the message
        the derivation renders.

        This four-arm remedy string is deliberately NOT part of
        `boxes.health.GuardVerdict`: the enum absorbs membership and short-tag
        rendering, and hoisting an operator instruction into it would be a
        behavior change dressed as a refactor.

        ADVISORY verdicts (velvet P1's STALE_IMAGE, and 2026-08-03's
        LOADING_SLOW) alarm like a zombie but carry a DIFFERENT remedy:
        `guard --fix` must not be offered as a destroy, because a stale-image
        box is healthy and a still-pulling box is merely slow.

        The remedy string is PHASE-AWARE (2026-08-03). It used to read
        `fix: herdd guard --fix` for every zombie verdict, including
        ZOMBIE_LOADING_STALL — a GPU-UNBILLED phase where destroy is
        irreversible and saves ~$0.01/hr. That alarm text is the proximate
        cause of 46682313's destruction: fleetd raised the (false) verdict at
        27m with `destroy + relaunch` in the reason and `guard --fix` as the
        offered fix, a session followed it at 38m, and the co-resident twin on
        the same image cleared the same verdict to OK 90 s later. An alarm that
        prescribes an irreversible remedy for a recoverable condition is part
        of the failure, not a bystander."""
        h = (self._health or {}).get(str(iid)) or {}
        v = h.get("verdict")
        zombie = health.verdict_is_zombie(v)
        advisory = health.verdict_is_advisory(v)
        if not (zombie or advisory):
            return None
        if v == health.GUARD_LOADING_SLOW:
            fix = ("advisory only — the pull is still advancing; let it finish "
                   "or park it (`herdd stop`), never destroy: GPU unbilled")
        elif v == health.GUARD_ZOMBIE_LOADING_STALL:
            fix = (f"GPU UNBILLED here — park, do not destroy: "
                   f"`herdd stop {iid}` (recoverable; the idle reaper "
                   f"finishes it in 2h). `herdd guard --fix` does this")
        elif zombie:
            fix = "fix: herdd guard --fix"
        else:
            fix = ("advisory only — destroy+relaunch to refresh the env; "
                   "a park/resume will not")
        return f"{iid}: HEALTH {v} — {h.get('reason')} ({fix})"

    # moved-from: fleetd.Fleet._health_alarm
    def _health_alarm(self, iid: object) -> None:
        """N1: gather_fleet_health verdicts surface as ALARMS (a running-but-dead
        box bills full GPU and is invisible to supervise until the budget cap).
        `herdd guard --fix` stays the destroy path.

        The alarm itself is derived (`_health_alarm_msg`); what happens HERE is
        the durable half — a zombie verdict is journaled once per verdict
        transition, because a `fleet status` alarm is only seen by a caller who
        happens to look, which is how a ZOMBIE_LOADING_STALL sat unseen for 3 h
        on 2026-07-30 (box 46256890)."""
        key = str(iid)
        h = (self._health or {}).get(key) or {}
        v = h.get("verdict")
        zombie = health.verdict_is_zombie(v)
        advisory = health.verdict_is_advisory(v)
        if zombie or advisory:
            if self._health_alarmed.get(key) != v:
                self._health_alarmed[key] = v
                self.journal("health_alarm", iid=key, verdict=v,
                             advisory=advisory, reason=h.get("reason"))
        elif key in self._health_alarmed and v is not None:
            self._health_alarmed.pop(key, None)
            self.journal("health_alarm_cleared", iid=key, verdict=v)

    # --- run/jobs: the imported supervise ladders ----------------------------
    # moved-from: fleetd.Fleet._tick_policy_watch
    def _tick_policy_watch(self, target: str, w: dict[str, Any], now: float,
                           inst: dict[str, Any] | None = None) -> None:
        # keep the accrual clock/rate fresh even while the policy tick owns the
        # spend: a later pause (S1) and the restart backfill (S2) both read them,
        # and a stale `_spend_ts` would double-charge the gap.
        w["_spend_ts"] = now
        if inst is not None:
            dph = models._num_dph(inst.get("dph_total"))
            if dph:
                w["_last_dph"] = dph
            w["_was_live"] = (inst.get("actual_status") or "").lower() \
                in bidpolicy.LIVE_STATES
        rt = self.runtime.get(target)
        if rt is None:
            rt = self._init_runtime(target, w)
            if rt is None:
                return
        a = rt["a"]
        if w["profile"] == "run":
            st, hf = rt["st"], rt["hf"]
            st["spend_usd"] = max(st.get("spend_usd", 0.0), w.get("spend_usd", 0.0))
            act = self.hooks.run_tick(st, a, hf, rt["handoff_on"])
            fleet_state._run_lane_state_persist(st, w)
            w["spend_usd"] = st.get("spend_usd", 0.0)
            w["iid"] = str(st.get("instance_id") or w.get("iid") or "") or None
            w["state"] = st.get("actual_status") or "unknown"
            self._charge_ceiling(w)
            # A `run` relaunch puts the label — and the ceiling — on a NEW box.
            self._ceiling_bind_box(w.get("ceiling_id"), w.get("iid"))
            self.state["spend_by_box"][str(w.get("iid"))] = round(w["spend_usd"], 4)
            self.journal("tick", iid=w.get("iid"), target=target, profile="run",
                         state=w["state"], spend_usd=round(w["spend_usd"], 4),
                         ceiling_spend_usd=self._ceiling_tick_field(w),
                         relaunches=st.get("relaunch_count"))
            if act is not None:
                self.hooks.run_finalize(st, a, act, hf, rt["handoff_on"])
                self._end_watch(target, w, act.kind, act.reason)
        else:
            # jobs AND serve: one ladder (serve_mode strips the queue
            # semantics; make_policy set it from the profile).
            jc, hf = rt["jc"], rt["hf"]
            jc["spend_usd"] = max(jc.get("spend_usd", 0.0), w.get("spend_usd", 0.0))
            self._notify_feed(jc)
            iid_before = str(jc.get("iid"))
            repl_before = int(jc.get("replacements", 0) or 0)
            ret_before = rows._retention_status_map(jc)
            verdict = self.hooks.jobs_tick(jc, hf)
            w["spend_usd"] = jc.get("spend_usd", 0.0)
            w["iid"] = str(jc.get("iid"))
            self._charge_ceiling(w)
            # SUCCESSOR INHERITANCE, both shapes, before anything else can see
            # the new box. `w["iid"]` follows an eviction replacement / SLA
            # relaunch / completed handoff; `hf["understudy_iid"]` names the
            # handoff understudy while the migration is still in flight — which
            # is minutes BEFORE the stray sweep would otherwise adopt it as an
            # uncapped `bare` box (path 2, understudy 47215526).
            self._ceiling_bind_box(w.get("ceiling_id"), w["iid"])
            self._ceiling_bind_box(w.get("ceiling_id"),
                                   (hf or {}).get("understudy_iid"))
            # The ladder may have swapped the box under us (eviction replacement,
            # pull/SLA reschedule, completed handoff). `w["iid"]` above already
            # follows it — this journals WHY, because an autonomous rental that
            # only shows up as a changed id in a tick line is not an audit trail.
            fleet_state._replacement_state_persist(jc, w)
            # The serve lane's identity verdict, same rule and same reason: the
            # alarms are DERIVED from the watch record, so a verdict that is
            # not written back is a verdict nobody ever sees.
            fleet_state._serve_identity_persist(jc, w)
            # Every money-moving rung the ladder took this tick, under its own
            # name (task #78). Drained BEFORE `jobs_replaced` so the log reads in
            # causal order: evicted -> decision -> launched -> retargeted ->
            # destroyed -> replaced. `iid` in the fields names the box the event
            # is ABOUT (the condemned box, the box destroyed); `target` is always
            # the watch, which is the identity that survives a box swap and the
            # thing `fleet log --iid` could never previously be pointed at.
            for _ev, _fields in jc.pop("ladder_journal", None) or []:
                _fields = dict(_fields)
                self.journal(_ev, iid=str(_fields.pop("iid", None) or w["iid"]),
                             target=target, **_fields)
            if int(jc.get("replacements", 0) or 0) > repl_before:
                last = (jc.get("replacement_history") or [{}])[-1]
                # SAY WHAT ACTUALLY HAPPENED TO THE OLD BOX. This note read "and
                # the old box destroyed" unconditionally, which has been false
                # on every default-configuration replacement since retention
                # shipped 2026-08-05 (the box is RETAINED for 3h and still
                # bills). On 2026-08-16 that sentence is why a retained box that
                # came back to life read, in the only log an operator checks, as
                # a box that no longer existed.
                old_fate = rows._retention_fate(rows._retention_status_map(jc),
                                                iid_before)
                self.journal("jobs_replaced", iid=w["iid"], target=target,
                             from_box=iid_before, to_box=w["iid"],
                             eviction_class=last.get("class"),
                             rental=last.get("rental"), dph=last.get("dph"),
                             replacements_used=jc.get("replacements"),
                             old_box_fate=old_fate[0],
                             budget_usd=w.get("budget_usd"),
                             spend_usd=round(w["spend_usd"], 4),
                             note="EVICTED box auto-replaced; queue retargeted "
                                  f"and {old_fate[1]} — the watch now "
                                  "supervises the replacement under the SAME "
                                  "budget cap")
                w.pop("unrecoverable_since", None)   # recovered by replacement
            # The ECONOMIC handoff is the other way the ladder spends money on
            # its own initiative, and until defect #63 it journalled nothing: on
            # 2026-08-08 it rented a second box and destroyed a healthy primary
            # with `jobs_replaced` (eviction-only) as the sole nearby line, so
            # the incident first read as a spot eviction. Drain what the ladder
            # queued — one entry per PHASE, because the handoff `complete`
            # transition is unreachable under fleetd (defect #61: a non-`run`
            # watch ends at `inst is None` before the ladder can tick again), so
            # hanging visibility on the end of the migration would show nothing.
            for kind, fields in jc.pop("handoff_journal", None) or []:
                self.journal(f"jobs_handoff_{kind}", iid=w["iid"], target=target,
                             **fields)
                if kind == "work_warning":
                    continue          # advisory; never a standing alarm state
                if kind in ("deferred", "refused"):
                    # Sticky, not derived: the arithmetic that produced the
                    # refusal is gone by the next poll (the market and the
                    # horizon both move), so `_derive_alarms` could not
                    # reconstruct it. Keyed per box — one standing "this box is
                    # over its ceiling and here is why we are not moving it".
                    self.latch_alarm(f"handoff:{w['iid']}",
                                     fields.get("note") or "handoff deferred",
                                     iid=w["iid"])
                else:
                    # The deferral's own text promises "re-testing each poll", so
                    # once the ladder DOES move, leaving it latched would have
                    # `fleet status` contradicting `fleet log` until an operator
                    # acked a condition that had resolved itself.
                    self.clear_alarm(f"handoff:{w['iid']}",
                                     reason=f"handoff {kind}")
            # A retained box is money leaving the wallet for a box nobody
            # rented on purpose, so every status TRANSITION is journaled with
            # its deadline and estimated cost — the same standard the rental
            # decisions are held to.
            for iid_r, rec in rows._retention_status_map(jc).items():
                if ret_before.get(iid_r, {}).get("status") == rec.get("status"):
                    continue
                self.journal("jobs_box_retention", iid=iid_r, target=target,
                             status=rec.get("status"),
                             eviction_class=rec.get("class"),
                             deadline=sup_journal._iso_z(rec.get("deadline_ts")),
                             retention_h=rec.get("retention_h"),
                             est_cost_usd=rec.get("cost_usd"),
                             est_cost_hi_usd=rec.get("cost_hi_usd"),
                             keep_labeled=rec.get("keep_labeled"),
                             note=rows.RETENTION_NOTES.get(rec.get("status"), ""))
            w["state"] = "watched"
            self.state["spend_by_box"][str(w["iid"])] = round(w["spend_usd"], 4)
            self.journal("tick", iid=w["iid"], target=target, profile=w["profile"],
                         spend_usd=round(w["spend_usd"], 4),
                         ceiling_spend_usd=self._ceiling_tick_field(w),
                         verdict=verdict)
            if verdict in JOBS_TRANSIENT_VERDICTS:
                self._jobs_watch_idle(target, w, verdict)
                return
            if verdict == "identity_mismatch":
                self._serve_watch_identity_mismatch(target, w)
                return
            if w.pop("queue_empty_since", None) is not None:
                self.journal("jobs_queue_filled", iid=w["iid"], target=target,
                             note="tickets present — the jobs ladder (defend / "
                                  "outbid rescue / drain park) is driving again")
            if verdict == "unrecoverable" and inst is not None:
                self._jobs_watch_unrecoverable(
                    target, w, refused=jc.get("replacement_refused"))
                return
            w["replacement_refused"] = None
            if w.pop("unrecoverable_since", None) is not None:
                self.journal("jobs_rescue_recovered", iid=w["iid"], target=target,
                             note="box back under its jobs watch — budget cap + "
                                  "bid ladder + reattach-on-resume re-armed")
            if verdict is not None:
                if verdict in ("budget", "drained", "self_parked"):
                    w["dormant"] = True             # S8: our own park, not an eviction
                if w.get("standing") and verdict in STANDING_KEEP_VERDICTS:
                    self._standing_drain(target, w, verdict)
                    return
                self._end_watch(target, w, verdict, None)

    # moved-from: fleetd.Fleet._jobs_watch_idle
    def _jobs_watch_idle(self, target: str, w: dict[str, Any],
                         verdict: str) -> None:
        """A jobs watch whose queue is EMPTY is pre-submission, not finished
        (`drained` is finished). Keep the watch: dropping it cost box 46240842
        its rescue ladder on 2026-07-30 — the explicit jobs+$5 watch registered
        at 00:34 exited `queue_empty` at 00:35, the stray sweep re-adopted the
        still-booting box as unbudgeted `bare` in the SAME tick, and the ~02:0x
        spot preemption found no ladder to engage. The budget cap keeps being
        enforced meanwhile: the ladder accrues spend and parks on breach every
        tick, and it resumes full policy the moment a ticket appears."""
        w["state"] = verdict
        first = w.get("queue_empty_since") is None
        if first:
            w["queue_empty_since"] = self.hooks.now()
            self.journal("jobs_queue_empty", iid=w.get("iid"), target=target,
                         profile="jobs", budget_usd=w.get("budget_usd"),
                         note="nothing submitted yet — watch KEPT with its "
                              "profile+budget so the outbid rescue ladder stays "
                              "armed (`queue_empty` is terminal for the inline "
                              "CLI only, never for the daemon)")
        # the alarm is derived from `queue_empty_since` and retracts itself the
        # tick a ticket appears (which pops that key — jobs_queue_filled).

    def _serve_watch_identity_mismatch(self, target: str,
                                       w: dict[str, Any]) -> None:
        """The serve ladder withdrew this box: it verified an identity that is
        not the one the watch was registered for.

        The watch is KEPT, and that is the whole design of the response. Ending
        it would pop the record `_derive_watch_alarms` reads, the alarm would
        vanish with it, and the stray sweep would re-adopt a parked box as an
        anonymous `bare` one — an operator would be left with a stopped box, no
        alarm, and no way to find out why. `unrecoverable` reached exactly this
        conclusion on 2026-07-31 for the same structural reason.

        NOT dormant either, unlike a budget park. Dormancy is the S8 silence
        for parks WE chose as policy; this one is a defect report, and it has
        to keep burning until an operator retires the watch or relaunches the
        serve. The alarm derives off `w["serve_identity"]` and OUTRANKS the
        dormancy check, so it survives whatever else marks this box down."""
        w["state"] = "identity_mismatch"
        if w.get("identity_mismatch_since") is None:
            w["identity_mismatch_since"] = self.hooks.now()
            rec = w.get("serve_identity") or {}
            self.journal("serve_identity_withdrawn", iid=w.get("iid"),
                         target=target, profile=w.get("profile"),
                         artifact=rec.get("artifact"),
                         expected_ident=rec.get("expected"),
                         observed_ident=rec.get("observed"),
                         parked=rec.get("parked"),
                         budget_usd=w.get("budget_usd"),
                         spend_usd=round(w.get("spend_usd", 0.0), 4),
                         note="box PARKED and withdrawn from the serve ladder "
                              "(no rescue, no relaunch, never a destroy); the "
                              "watch is KEPT so the alarm stays where an "
                              "operator can act on it")

    # moved-from: fleetd.Fleet._standing_drain
    def _standing_drain(self, target: str, w: dict[str, Any],
                        verdict: str) -> None:
        """A STANDING watch survives its queue draining (item 2 of
        `FLEET_REVIEW_2026-08-14.md`; the "park-and-keep" option §4a filed as
        open). The box still parks exactly as before — the ladder did that
        inside `jobs_tick`, before this verdict came back, and `--keep` still
        suppresses it — but the WATCH does not end.

        What that buys, in the order the 47694876 cycle lost them: no
        `watch_finished`, so no stray re-adoption as unbudgeted `bare`, so no
        "armed watch LAPSED" alarm, so no re-arm for an operator to forget. The
        ladder, its budget, its replacement counters and its policy stay on the
        record, and the next ticket resumes them (`_standing_tick`).

        Budget: a drain does NOT reset the cap. One ceiling spans every cycle of
        a standing watch, so cycle two enforces the REMAINING headroom — the
        conservative reading, and the same arithmetic a lapse-then-inherit
        already produced. The ceiling's `epochs` counter is deliberately NOT
        bumped: no watch ended, so no epoch ended.

        `watch_finished` was how an operator learned the work was done, so the
        replacement event carries the same figures under its own name."""
        self._charge_ceiling(w)
        cid = w.get("ceiling_id")
        found = self._ceiling_read(cid)
        cap = spend = None
        if found is not None:
            rec, cap, spend, _deg = found
            rec["last_verdict"] = f"standing_{verdict}"
            rec["last_target"] = str(target)
            rec["updated_ts"] = self.hooks.now()
            hist = rec.setdefault("history", [])
            hist.append({"ts": round(self.hooks.now(), 3),
                         "event": "standing_drained", "target": str(target),
                         "verdict": verdict, "cap_usd": round(cap, 4),
                         "spend_usd": round(spend, 4)})
            del hist[:-CEILING_HISTORY_MAX]
        w["standing_dormant"] = True
        w["dormant"] = True                         # S8: our own park, not an
        w["dormant_reason"] = f"standing_{verdict}"  # eviction to be rescued
        w["state"] = "standing"
        w["standing_since"] = self.hooks.now()
        w["standing_cycles"] = int(w.get("standing_cycles") or 0) + 1
        w.pop("standing_live_since", None)
        w.pop("standing_queue_unknown", None)
        # A wake this drain OVERTOOK: the ladder has just read the whole queue as
        # terminal, which is better evidence than the placement that preceded it.
        w.pop("standing_wake_pending", None)
        w.pop("standing_wake_source", None)
        w.pop("standing_wake_job_id", None)
        w.pop("queue_empty_since", None)            # drained is not pre-submission
        self.journal("jobs_watch_standing_drained", iid=w.get("iid"),
                     target=target, profile=w["profile"], verdict=verdict,
                     cycles=w["standing_cycles"],
                     spend_usd=round(w.get("spend_usd", 0.0), 4),
                     budget_usd=w.get("budget_usd"), ceiling_id=cid,
                     cap_usd=round(cap, 4) if cap is not None else None,
                     ceiling_spend_usd=(round(spend, 4)
                                        if spend is not None else None),
                     # `cap` and `spend` are assigned together from ONE tuple, so
                     # the second test is a type narrowing, not a second case.
                     remaining_usd=(round(cap - spend, 4)
                                    if cap is not None and spend is not None
                                    else None),
                     note="queue drained and the box parked per policy — the "
                          "WATCH is KEPT, dormant but ARMED (ladder, cap and "
                          "replacement policy intact). The next ticket resumes "
                          "it with no re-arm; the cap is NOT reset, so the next "
                          "cycle spends the remaining headroom")

    # moved-from: fleetd.Fleet._standing_tick
    def _standing_tick(self, target: str, w: dict[str, Any], iid: object,
                       inst: dict[str, Any] | None, live: bool,
                       intent: dict[str, Any] | None, now: float) -> bool:
        """One tick of the dormant-but-armed phase. True == supervision RESUMES
        this tick (the caller falls through to the ladder), False == still
        dormant.

        Two rules, both learned the expensive way:

        * ACCRUAL NEVER PAUSES. A `--keep` standing box stays live and billing
          through its dormancy, and the cap is cumulative across cycles — so the
          clock keeps running and a breach still parks, dormant or not.
        * ONLY A TICKET RE-ARMS, never mere liveness. Re-entering the ladder
          against an all-terminal queue drain-parks the box seconds later (box
          46648873, 2026-08-03, is that exact failure from the other direction:
          a watch armed over a stale queue parked the box in 4 s). So a live
          standing box with nothing pending stays quiet, which is also what
          makes resuming a box by hand *in order to* submit safe.

        An unreadable queue is NOT evidence of work (N7): it holds the dormancy
        and journals once per episode rather than guessing in either direction.
        The queue read costs one B2 listing per tick and only while the box is
        LIVE — a parked standing box reads nothing at all.

        A ticket-placement WAKE (`Fleet.ticket_placed`) short-circuits the poll:
        it is the same fact told first-hand instead of inferred, so it re-arms
        even where the listing is unreadable. It never re-arms a box that is not
        live — that would let a ticket written onto a parked box rent against an
        eviction classification nobody asked for."""
        self._accrue(w, inst, now)
        if self._budget_breached(w):
            # ONCE, not per tick. The non-standing path gets this for free (its
            # budget park sets `dormant`, and the next tick returns at the
            # dormancy check); a standing watch is ALREADY dormant, so without
            # the state guard every tick would re-issue the park API call and
            # re-journal it — the 158-events-for-2-facts shape the 2026-08-14
            # review called out. A FAILED park leaves another state and does
            # retry, which is the behaviour it should keep.
            if w.get("state") != "budget_parked":
                self._park_on_budget(w, target, inst)
            return False
        if not live or intent:
            w.pop("standing_live_since", None)
            return False
        w.setdefault("standing_live_since", now)
        # A WAKE outranks the queue read. `job submit|retarget|requeue` tell the
        # daemon the moment they write a non-terminal ticket (`ticket_placed`),
        # which is first-hand evidence the poll can only infer — and cannot infer
        # at all when the B2 listing is unreadable (N7), the shape that leaves a
        # box riding out its bid with the ladder idle. Consumed exactly once.
        trigger = "queue"
        if w.pop("standing_wake_pending", False):
            trigger = str(w.pop("standing_wake_source", None) or "ticket_placed")
        else:
            drained = self.hooks.drained(iid)
            if drained is not False:                # True (all terminal) or None
                if drained is None and not w.get("standing_queue_unknown"):
                    w["standing_queue_unknown"] = True
                    self.journal("jobs_watch_standing_queue_unknown",
                                 iid=iid, target=target,
                                 note="standing watch is dormant on a LIVE box "
                                      "and the queue could not be read — an "
                                      "unreadable queue is not evidence of work, "
                                      "so the watch stays dormant (cap still "
                                      "accruing). Journaled once per dormancy, "
                                      "not per tick")
                return False
        w.pop("standing_wake_source", None)
        w.pop("standing_wake_job_id", None)
        w.pop("standing_queue_unknown", None)
        w["standing_dormant"] = False
        w["dormant"] = False
        w["dormant_reason"] = None
        w["state"] = "watched"
        dormant_s = round(now - (w.get("standing_since") or now), 1)
        w["standing_since"] = None
        w.pop("standing_live_since", None)
        self.journal("jobs_watch_standing_resumed", iid=iid, target=target,
                     profile=w.get("profile"), dormant_s=dormant_s,
                     trigger=trigger, cycles=w.get("standing_cycles"),
                     budget_usd=w.get("budget_usd"),
                     spend_usd=round(w.get("spend_usd", 0.0), 4),
                     ceiling_spend_usd=round(self._ceiling_spend(w), 4),
                     remaining_usd=(round(w["budget_usd"] - self._ceiling_spend(w), 4)
                                    if w.get("budget_usd") is not None else None),
                     note="a ticket appeared on a standing watch's box — the "
                          "SAME watch resumes supervising (bid defend/rescue, "
                          "eviction replacement, reattach) with no re-arm, no "
                          "bare adoption and no fresh cap")
        return True

    # moved-from: fleetd.Fleet._jobs_watch_unrecoverable
    def _jobs_watch_unrecoverable(self, target: str, w: dict[str, Any],
                                  refused: str | None = None) -> None:
        """The jobs ladder gave up (rescue stalled past deadline / no market
        read) but the instance still EXISTS in the API listing — for the daemon
        that is an ALARM, not an exit. Dropping the watch here cost box 46347213
        on 2026-07-31: the explicit jobs+$5 watch died `unrecoverable` at 04:33
        mid-preemption, the box auto-resumed at 04:42 when its OWN standing bid
        regained priority, onstart re-pulled the stale launch-pinned jobd
        bundle, and — with the watch gone — the was_live->live reattach that
        would have pushed the current jobd never fired. The stray sweep
        re-adopted the box as unbudgeted `bare` and it billed full GPU, idle,
        until an operator noticed. Keep the watch: budget accrual, the bid
        ladder, and reattach-on-resume all re-arm the moment the box returns
        (jobs_rescue_recovered); a box that is truly GONE leaves the listing
        and dies through the instance_gone path instead. `unrecoverable` stays
        terminal for the INLINE `job supervise` CLI (exit 3 + retarget
        instructions), same split as `queue_empty`.

        `refused` (2026-08-05) is the automatic-replacement ladder's own refusal
        reason, carrying its arithmetic. Reaching here at all now means BOTH the
        bid rescue and the replacement declined, so the alarm has to say which
        bound stopped the spend — budget remainder, replacement cap, price
        ceiling — or the operator is back to guessing, which is the state this
        directive was written to end."""
        w["state"] = "unrecoverable"
        w["replacement_refused"] = refused
        if w.get("unrecoverable_since") is None:
            w["unrecoverable_since"] = self.hooks.now()
            self.journal("jobs_rescue_stalled", iid=w.get("iid"), target=target,
                         profile=w.get("profile"), budget_usd=w.get("budget_usd"),
                         replacement_refused=refused,
                         note="rescue gave up AND the automatic replacement was "
                              "refused (see replacement_refused) — watch KEPT "
                              "with its profile+budget; an auto-resume re-arms "
                              "the ladder, a truly dead box is reaped via "
                              "instance_gone")

    # moved-from: fleetd.Fleet._init_runtime
    def _init_runtime(self, target: str,
                      w: dict[str, Any]) -> dict[str, Any] | None:
        # `rows.watch_box_iid` — NOT the key — decides which box the rebuilt
        # ladder supervises. A watch whose ladder already replaced its box is
        # rebuilt on every daemon restart and on every re-`watch`, and pointing
        # it back at the key resurrects a destroyed instance id.
        a = make_policy(w["profile"], w.get("policy"), target,
                        budget_usd=w.get("budget_usd"), iid=rows.watch_box_iid(w))
        self._spend_backfill(w)
        try:
            if w["profile"] == "run":
                st, hf, handoff_on = self.hooks.run_init(a)
                st["spend_usd"] = max(st.get("spend_usd", 0.0),
                                      w.get("spend_usd", 0.0))
                fleet_state._run_lane_state_restore(st, w)
                rt = {"a": a, "st": st, "hf": hf, "handoff_on": handoff_on}
            else:
                jc, hf = self.hooks.jobs_init(a)
                jc["spend_usd"] = w.get("spend_usd", 0.0)
                fleet_state._replacement_state_restore(jc, w)
                # The condemn latch has to come back with it: a daemon restart
                # that forgot a mismatch would hand the ladder a fresh licence
                # to rescue the box it withdrew.
                fleet_state._serve_identity_restore(jc, w)
                rt = {"a": a, "jc": jc, "hf": hf,
                      "handoff_on": jc.get("handoff_on", True)}
        except Exception as e:                       # B2/API blip at adoption
            w["state"] = "init_error"
            w["init_error"] = f"{type(e).__name__}"  # derived until init succeeds
            self.journal("watch_init_failed", target=target,
                         error=f"{type(e).__name__}: {e}",
                         note="retrying next tick")
            return None
        w["init_error"] = None
        self.runtime[target] = rt
        self.journal("watch_adopted", iid=w.get("iid"), target=target,
                     profile=w["profile"], spend_usd=round(w.get("spend_usd", 0.0), 4))
        return rt

    # moved-from: fleetd.Fleet._spend_backfill
    def _spend_backfill(self, w: dict[str, Any]) -> None:
        """S2: a box that was LIVE when the daemon died kept billing. Charge the
        downtime at its last known rate so a restart can never bypass the cap."""
        if not w.get("_was_live") or not w.get("_last_dph") or not w.get("_spend_ts"):
            return
        gap = self.hooks.now() - w["_spend_ts"]
        if gap <= 0:
            return
        add = w["_last_dph"] * gap / 3600.0
        w["spend_usd"] = w.get("spend_usd", 0.0) + add
        w["_spend_ts"] = self.hooks.now()
        self.journal("spend_backfilled", iid=w.get("iid"), target=w["target"],
                     downtime_s=round(gap, 1), added_usd=round(add, 4),
                     spend_usd=round(w["spend_usd"], 4))

    # moved-from: fleetd.Fleet._end_watch
    def _end_watch(self, target: str, w: dict[str, Any], verdict: str | None,
                   reason: str | None) -> None:
        """A watch ends; its CEILING does not.

        This is the seam path 3 fell through: the record was popped whole, the
        stray sweep re-adopted the still-live box as uncapped `bare` in the same
        reconcile pass (6ms on box 46687567), and the spend counter restarted at
        zero. The watch still ends — `drained` really does mean this ladder is
        finished — but the cap and the spend-to-date survive in the ledger, so
        whatever picks the box up next inherits remaining headroom."""
        self._charge_ceiling(w)
        cid = w.get("ceiling_id")
        found = self._ceiling_read(cid)
        cap = spend = None
        if found is not None:
            rec, cap, spend, _deg = found
            rec["last_verdict"] = verdict
            rec["last_target"] = str(target)
            rec["epochs"] = int(rec.get("epochs") or 0) + 1
            rec["updated_ts"] = self.hooks.now()
            hist = rec.setdefault("history", [])
            hist.append({"ts": round(self.hooks.now(), 3), "event": "watch_ended",
                         "target": str(target), "verdict": verdict,
                         "cap_usd": round(cap, 4), "spend_usd": round(spend, 4)})
            del hist[:-CEILING_HISTORY_MAX]
        with self.lock:
            self.state["watches"].pop(target, None)
            self.runtime.pop(target, None)
        self.journal("watch_finished", iid=w.get("iid"), target=target,
                     profile=w["profile"], verdict=verdict, reason=reason,
                     spend_usd=round(w.get("spend_usd", 0.0), 4),
                     ceiling_id=cid,
                     cap_usd=round(cap, 4) if cap is not None else None,
                     ceiling_spend_usd=round(spend, 4) if spend is not None else None,
                     # `cap` and `spend` are assigned together from ONE tuple, so
                     # the second test is a type narrowing, not a second case.
                     remaining_usd=(round(cap - spend, 4)
                                    if cap is not None and spend is not None
                                    else None),
                     note=("the watch ended; its CEILING SURVIVES — a re-arm or "
                           "an auto-adoption inherits this cap and this "
                           "spend-to-date, so what is enforced next is the "
                           "remaining headroom" if cid else None))

    # --- strays: evidence-gated safety net (review B1) ------------------------
    # moved-from: fleetd.Fleet._tick_strays
    def _tick_strays(self, by_iid: dict[str, dict[str, Any]], now: float,
                     obs_dt: float) -> None:
        watched = {str(w.get("iid") or t) for t, w in self.state["watches"].items()}

        # PRUNE first (item D): the loop below only visits boxes the API still
        # lists, so a record for a box that has LEFT the listing is never touched
        # again and lives in state.json forever. `fleet status` rendered those as
        # `UNWATCHED $0.000` rows indefinitely. Bounded on the same `live_ts`
        # evidence `rows.stray_rows` filters on, and only for boxes that are gone
        # — a PARKED box is still listed, and its record is deliberately kept.
        def _prunable(iid: str, s: object) -> bool:
            if str(iid) in by_iid or not isinstance(s, dict):
                return False                     # still listed, or not a record
            seen = s.get("live_ts")
            if not isinstance(seen, (int, float)):
                return True                      # never seen live by this daemon
            return bool((now - seen) > rows.UNWATCHED_STALE_S)
        with self.lock:
            for iid in [i for i, s in list(self.state["strays"].items())
                        if _prunable(i, s)]:
                self.state["strays"].pop(iid, None)
                self.journal("stray_record_pruned", iid=iid,
                             note="box has left the instance listing and its "
                                  "last live sighting is stale — dropping the "
                                  "record so it stops rendering as UNWATCHED")
        for iid, inst in by_iid.items():
            if iid in watched:
                with self.lock:                  # structural: status() may iterate
                    self.state["strays"].pop(iid, None)
                continue
            if (inst.get("actual_status") or "").lower() not in bidpolicy.LIVE_STATES:
                continue                             # parked/stopped: storage only
            if rows.label_exempt(inst.get("label")):
                continue                             # explicit opt-out (workflowctl)
            with self.lock:                      # structural: status() may iterate
                s = self.state["strays"].setdefault(
                    iid, {"first_seen_ts": now, "observed_s": 0.0})
            # `live_ts` is the alarm-derivation gate: a stray record is only
            # allowed to alarm if THIS reconcile saw the box live. Records
            # outlive the boxes they describe (a parked box is skipped above,
            # record and all), and an alarm derived from a stale record is the
            # latching bug in another costume.
            s["live_ts"] = now
            s["label"] = inst.get("label")
            pa = s.pop("pending_action", None)
            if pa:
                self._exec_action(iid, pa, inst)
                continue
            if s.get("paused_until") and now < s["paused_until"]:
                self.journal("tick_paused", iid=iid, kind="stray",
                             left_s=round(s["paused_until"] - now, 1))
                continue
            # N7: the grace clock counts OBSERVED seconds, never wall time.
            s["observed_s"] = s.get("observed_s", 0.0) + obs_dt
            self._health_alarm(iid)
            evidence = rows.workload_evidence(inst, (self._health or {}).get(iid))
            if evidence:
                # B1: never park a box that shows live work — ADOPT it (`bare`:
                # observation + cap, no bid moves) and keep alarming. A busy box
                # is exempt regardless of price tier — the evidence gate
                # outranks the fuse (owner ruling 2026-07-29 only shortens the
                # NO-evidence grace, it never overrides B1).
                #
                # The adoption is no longer UNCAPPED. `watch()` resolves a
                # durable ceiling: the predecessor's cap and spend-to-date when
                # this box can inherit one (a lapsed watch on the same id — path
                # 3; a `job:<pred>:handoff` understudy via `label` — path 2),
                # otherwise a conservative provisional default (path 1). Passing
                # `None` here means "we are not naming a figure", never "no
                # ceiling" — that reading is what cost 121 boxes their cap.
                with self.lock:
                    self.state["strays"].pop(iid, None)
                try:
                    aw = self.watch(iid, "bare", None,
                                    {"adopted_from": inst.get("label"),
                                     "evidence": evidence},
                                    requester="fleetd:auto-adopt", adopted=True,
                                    label=inst.get("label"))
                except Exception as e:      # never let one box abort the sweep
                    self.journal("auto_adopt_failed", iid=iid,
                                 error=f"{type(e).__name__}: {e}")
                    with self.lock:         # put the record back: the failure is
                        s["adopt_error"] = f"{type(e).__name__}"   # retried, and
                        self.state["strays"][iid] = s   # its grace clock is real
                    continue
                if not aw.get("adopted"):
                    continue                # an explicit watch owns it: it stays
                self.journal("unwatched_adopted", iid=iid, evidence=evidence,
                             label=inst.get("label"),
                             budget_usd=aw.get("budget_usd"),
                             ceiling_id=aw.get("ceiling_id"),
                             ceiling_source=aw.get("ceiling_source"),
                             spend_carried_usd=(round(aw.get("spend_usd", 0.0), 4)
                                                or None),
                             note="workload evidence — adopted as `bare`, NOT "
                                  "parked; see ceiling_source for whether the "
                                  "cap was INHERITED from a durable ceiling or "
                                  "is the provisional auto-adopt DEFAULT")
                # No alarm here: the box now has a watch, and `watch:<t>:adopted`
                # (unbudgeted adoption) is derived from it — one durable line
                # that a real `fleet watch` retracts, instead of an event.
                continue
            s["adopt_error"] = None
            if s.get("parked_ts"):
                continue                    # derived: "PARKED by the safety net"
            # Owner ruling 2026-07-29: the unwatched-grace fuse is price-aware —
            # more grace for cheap boxes, less for expensive ones. `raw_dph` is
            # the SAME field the budget-accrual path reads (`_accrue` above);
            # missing/unparseable dph fails toward the short (expensive) fuse.
            raw_dph = models._num_dph(inst.get("dph_total"))
            grace, tier = unwatched_grace_for_dph(raw_dph)
            dph = raw_dph if raw_dph is not None else 0.0
            dph_disp = f"{raw_dph:.3f}" if raw_dph is not None else "?"
            s["dph_disp"], s["tier"], s["grace_s"] = dph_disp, tier, grace
            self.journal("unwatched", iid=iid, observed_s=round(s["observed_s"], 1),
                         dph=dph, dph_known=raw_dph is not None, tier=tier,
                         grace_s=grace, label=inst.get("label"),
                         note="no fleet watch owns this box and it shows no "
                              "workload evidence — fuse armed at "
                              f"tier={tier} dph={dph_disp}")
            if grace > 0 and s["observed_s"] >= grace:
                # graded_keep: no watch and no operator asked for this box, so
                # nothing here is a resumability promise to anyone.
                ok, err = self._park(iid, inst, why="unwatched_safety_net",
                                     graded_keep=True)
                s["parked_ts"] = now if ok else None
                s["park_error"] = None if ok else err
                self.journal("unwatched_parked" if ok else "unwatched_park_failed",
                             iid=iid, observed_s=round(s["observed_s"], 1),
                             dph=dph, dph_known=raw_dph is not None, tier=tier,
                             grace_s=grace, error=None if ok else err,
                             note="safety net: an unwatched box with no workload "
                                  f"evidence is PARKED, NEVER destroyed — fuse "
                                  f"fired at tier={tier} dph={dph_disp}")
                # both outcomes derive from the record (`parked_ts` / `park_error`)

    # moved-from: fleetd.Fleet._tick_global_budget
    def _tick_global_budget(self) -> None:
        """N3: the fleet ceiling ALARMS and freezes new spend — it parks nothing
        automatically (mass-parking a fleet on an accounting estimate is worse
        than the overspend it prevents)."""
        cap = global_budget_usd()
        if cap is None:
            return
        total = sum(self.state["spend_by_box"].values())
        if total >= cap:                    # the alarm derives from the same two
            self.journal("global_budget_breached",   # numbers (`fleet:budget`)
                         total_usd=round(total, 4), cap_usd=cap)

    # ------------------------------------------------------------- status ----#
    # moved-from: fleetd.Fleet.status
    def status(self) -> dict[str, Any]:
        # NOTE: the local list was called `rows` in the flat module; here that
        # name belongs to the `vastlib.fleet.rows` MODULE, whose row builders
        # this method calls. Renamed to `srows` — the payload key is still
        # `"rows"`, which is the wire contract `fleet status` renders.
        with self.lock:
            now = self.hooks.now()
            srows: list[dict[str, Any]] = []
            for target, w in sorted(self.state["watches"].items()):
                srows.append({
                    "target": target, "iid": w.get("iid"),
                    "profile": w.get("profile"), "state": w.get("state"),
                    "spend_usd": round(w.get("spend_usd", 0.0), 4),
                    "budget_usd": w.get("budget_usd"),
                    # What the cap is actually measured against, where it came
                    # from, and what is left. `spend_usd` above is this watch's
                    # own counter and understates a ceiling carried across a
                    # lapse or shared with a successor box.
                    "ceiling_id": w.get("ceiling_id"),
                    "ceiling_source": w.get("ceiling_source"),
                    "ceiling_spend_usd": round(self._ceiling_spend(w), 4),
                    "remaining_usd": (
                        round(w["budget_usd"] - self._ceiling_spend(w), 4)
                        if w.get("budget_usd") is not None else None),
                    "paused": bool(w.get("paused_until")),
                    "pause_left_s": (round(w["paused_until"] - now, 1)
                                     if w.get("paused_until") else None),
                    "pause_reason": w.get("pause_reason"),
                    "dormant": bool(w.get("dormant")),
                    # A standing watch reads `dormant` between waves; these two
                    # say WHY, so `fleet status` can tell "armed and waiting for
                    # the next submit" from "parked and finished".
                    "standing": bool(w.get("standing")),
                    "standing_dormant": bool(w.get("standing_dormant")),
                    "adopted": bool(w.get("adopted")),
                    "last_action": w.get("last_action"),
                    "requester": w.get("requester")})
            # Only strays seen LIVE recently, and never a box already queued for
            # destruction — `_tick_strays` cannot prune a record whose box has
            # left the listing, so the raw dict accumulates long-dead boxes that
            # rendered as `UNWATCHED  $0.000` forever (item D).
            # `cast`, not a copy: `fleet status` renders the SAME row objects
            # the flat daemon handed it; a rebuild here would be a second
            # representation of a wire payload for a typing convenience.
            srows.extend(cast("list[dict[str, Any]]",
                              rows.stray_rows(self.state, now)))
        # OUTSIDE the lock (alarm_records takes it — RLock, but keep the read
        # path honest): alarms are DERIVED here, at read time, so a condition
        # the operator just fixed is gone from this payload immediately rather
        # than at the next tick.
        records = self.alarm_records(now)
        with self.lock:
            return {"version": client.FLEET_PROTO_VERSION, "rev": self.rev,
                    "rows": srows,
                    "alarms": [r["msg"] for r in records],
                    "alarm_records": records,
                    "api_ok": not (self.state.get("meta") or {}).get(
                        "api_unavailable_since"),
                    "dry_run": getattr(self.hooks, "dry_run", None),
                    "tick_age_s": (round(now - self.last_tick_ts, 1)
                                   if self.last_tick_ts else None),
                    "spend_total_usd": round(
                        sum(self.state["spend_by_box"].values()), 4),
                    "intents": self.state["intents"],
                    "retained": rows.retention_rows(self.state, now),
                    # Durable ceilings, including ones no live watch holds —
                    # that is the headroom the next re-arm or auto-adoption
                    # inherits, and it was invisible (silently $0) before.
                    "ceilings": rows.ceiling_rows(self.state, now),
                    "adopt_default_budget_usd": self.adopt_default_cap(),
                    # What the v0/bundles/ gate has cost this daemon. The only
                    # place the pacer is observable: it journals nothing.
                    "api_pace": api.bundles_pace_stats(),
                    "destroys": self.state["destroys"]}

    # moved-from: fleetd.Fleet.spend
    def spend(self, since: object = None, reconcile: bool = False) -> dict[str, Any]:
        with self.lock:
            by_box = dict(self.state["spend_by_box"])
        out: dict[str, Any] = {"since": since, "by_box": by_box,
                               "total_usd": round(sum(by_box.values()), 4)}
        if not reconcile:
            return out
        # ONE listing read, and a failed one degrades to "no reconciliation"
        # rather than to an empty fleet — the N7 rule: an unreadable API is not
        # evidence of anything.
        try:
            instances = self.hooks.instances()
        except Exception:
            instances = None
        if instances is None:
            out["reconcile"] = None
            out["reconcile_error"] = ("instance listing unavailable — an "
                                      "unreadable API is not an empty fleet")
            return out
        with self.lock:
            rrows = rows.reconcile_rows(self.state, instances, self.hooks.now())
        out["reconcile"] = rrows
        out["reconcile_basis"] = (
            "upper_bound_usd = dph_total x (now - start_date). The vast API "
            "exposes NO per-instance invoice, so this is an independent estimate "
            "from the box's own billing anchor, not the bill. It OVER-states any "
            "box that spent time `loading` (no GPU billing there, and the API "
            "has no loading->running timestamp) or parked, so read a divergence "
            "as 'this much of the box's billed life fleetd never watched'.")
        return out


# --------------------------------------------------------------------------- #
# socket server — JSON lines, one request/response per connection. Its accept
# loop runs on its OWN thread (review B3): a slow reconcile tick must never make
# a client command time out.
# --------------------------------------------------------------------------- #
# moved-from: fleetd.Server
class Server:
    # moved-from: fleetd.Server.__init__
    def __init__(self, fleet: Fleet, sock_path: str | None = None) -> None:
        self.fleet = fleet
        # The socket path resolution is `vastlib.fleet.client`'s, called as a
        # MODULE ATTRIBUTE: conftest's autouse fixture redirects `FLEETD_SOCK`
        # so the suite can never reach the live daemon, and that guard bites
        # only if this resolution stays where the fixture can see it.
        self.sock_path = sock_path or client.fleet_sock_path()
        self.sock: socket.socket | None = None
        self.stop = threading.Event()

    # moved-from: fleetd.Server.bind
    def bind(self) -> socket.socket:
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)                # N4: stale socket
        os.makedirs(os.path.dirname(self.sock_path), exist_ok=True)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(self.sock_path)
        os.chmod(self.sock_path, 0o600)              # owner only (§2)
        s.listen(16)
        s.settimeout(1.0)
        self.sock = s
        return s

    # moved-from: fleetd.Server.handle
    def handle(self, req: object) -> tuple[bool, Any, str | None]:
        """(ok, data, error) for one decoded request. Pure w.r.t. the socket, so
        the protocol is golden-testable without a real connection.

        The wire version is ONE constant, `client.FLEET_PROTO_VERSION`, read as
        a module attribute — the daemon and the CLI used to hold two literals
        that could silently diverge."""
        if not isinstance(req, dict):
            return False, None, "malformed request"
        op = req.get("op")
        args = req.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return False, None, "malformed args"
        v = req.get("v")
        if v is not None and int(v) != client.FLEET_PROTO_VERSION:
            return False, None, (f"protocol version {v} != "
                                 f"{client.FLEET_PROTO_VERSION}")
        f = self.fleet
        try:
            if op == "ping":
                return True, {"version": client.FLEET_PROTO_VERSION, "rev": f.rev,
                              "pid": os.getpid(),
                              "tick_age_s": (round(f.hooks.now() - f.last_tick_ts, 1)
                                             if f.last_tick_ts else None),
                              "watches": len(f.state["watches"]),
                              "dry_run": getattr(f.hooks, "dry_run", None)}, None
            if op == "status":
                return True, f.status(), None
            if op == "spend":
                return True, f.spend(args.get("since"),
                                     reconcile=bool(args.get("reconcile"))), None
            if op == "ack":
                return True, f.ack_alarm(args.get("key"),
                                         bool(args.get("all")),
                                         args.get("requester")), None
            if op == "watch":
                # `standing` absent (an older client, or an internal caller)
                # means UNCHANGED, not False — see Fleet.watch.
                w = f.watch(args.get("target"), args.get("profile") or "bare",
                            args.get("budget_usd"), args.get("policy"),
                            args.get("requester"),
                            reset_spend=bool(args.get("reset_spend")),
                            standing=args.get("standing"))
                # `standing` rides the response ONLY when it is set, so a
                # non-standing watch's payload is byte-identical to what every
                # client before 2026-08-14 received.
                return True, {"target": w["target"], "profile": w["profile"],
                              "iid": w.get("iid"),
                              **({"standing": True} if w.get("standing") else {}),
                              "spend_usd": round(w.get("spend_usd", 0.0), 4),
                              # The cap that LANDED (an inherited ceiling can
                              # differ from the figure asked for) and the
                              # headroom left under it — the client prints both,
                              # so "re-armed at $5" can never again mean "and
                              # got a sixth fresh $5".
                              "budget_usd": w.get("budget_usd"),
                              "ceiling_id": w.get("ceiling_id"),
                              "ceiling_source": w.get("ceiling_source"),
                              "remaining_usd": (
                                  round(w["budget_usd"] - f._ceiling_spend(w), 4)
                                  if w.get("budget_usd") is not None else None),
                              # set when the operator addressed the box by the
                              # id its ladder moved to (Fleet.watch): the client
                              # prints WHICH watch the cap actually landed on.
                              "redirected_from": w.get("redirected_from")}, None
            if op == "unwatch":
                w = f.unwatch(args.get("target"), args.get("requester"))
                return True, {"target": w["target"]}, None
            if op == "pause":
                return True, f.pause(args.get("target"), args.get("seconds"),
                                     args.get("reason"), args.get("requester")), None
            if op in ("park", "resume"):
                return True, f.request_action(args.get("target"), op,
                                              args.get("reason"),
                                              args.get("requester")), None
            if op == "operator_intent":
                return True, f.operator_intent(args.get("target"),
                                               args.get("kind"),
                                               args.get("requester"),
                                               args.get("reason")), None
            if op == "ticket_placed":
                # ADDITIVE: an older daemon answers `unknown op` and every
                # caller is best-effort, so a new CLI against a running old
                # daemon degrades to the queue poll it already had.
                return True, f.ticket_placed(args.get("target"),
                                             args.get("job_id"),
                                             args.get("source"),
                                             args.get("requester")), None
            if op == "destroy":
                return True, f.request_destroy(
                    args.get("target"), args.get("when") or "now",
                    args.get("reason"), args.get("requester"),
                    bool(args.get("yes")),
                    results_check=args.get("results_check", True)), None
            if op == "tick":                       # test/debug: force one pass
                f.tick()
                return True, {"ticked": True}, None
            return False, None, f"unknown op {op!r}"
        except (KeyError, ValueError) as e:
            return False, None, str(e)
        except Exception as e:                     # never take the daemon down
            return False, None, f"{type(e).__name__}: {e}"

    # moved-from: fleetd.Server.serve_forever
    def serve_forever(self) -> None:
        while not self.stop.is_set():
            try:
                assert self.sock is not None       # bind() precedes serve_forever
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError as e:
                if e.errno == errno.EINVAL or self.stop.is_set():
                    break
                continue
            with conn:
                conn.settimeout(10)
                try:
                    buf = b""
                    while b"\n" not in buf and len(buf) < 1 << 20:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                    try:
                        req = json.loads(buf.split(b"\n", 1)[0].decode())
                    except (ValueError, UnicodeDecodeError):
                        resp: dict[str, Any] = {"ok": False, "error": "malformed json"}
                    else:
                        ok, data, err = self.handle(req)
                        resp = {"ok": ok, "data": data} if ok else \
                               {"ok": False, "error": err, "data": data}
                    conn.sendall((json.dumps(resp, default=str) + "\n").encode())
                except OSError:
                    pass

    # moved-from: fleetd.Server.close
    def close(self) -> None:
        self.stop.set()
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
# moved-from: fleetd._reconcile_loop
def _reconcile_loop(fleet: Fleet, stop: threading.Event,
                    interval: float | None = None) -> None:
    while not stop.is_set():
        try:
            fleet.tick()
        except Exception as e:                      # a tick must never kill the daemon
            fleet.journal("tick_error", error=f"{type(e).__name__}: {e}")
        base = TICK_S if interval is None else interval
        jitter = base * TICK_JITTER_FRAC          # 7.0s at the 45s default
        stop.wait(max(1.0, base + random.uniform(-jitter, jitter)))


# moved-from: fleetd.cmd_serve
def cmd_serve(args: argparse.Namespace) -> int:
    config.load_env()
    d = fleet_state.state_dir()
    # H10 / the worst bug available here: this is an OPEN FILE HANDLE, and the
    # flock lives exactly as long as the binding does. `lock` is never read
    # again on purpose — dropping the name (a "tidy-up" of an unused local)
    # closes the fd, releases the lock and admits a SECOND reconciler.
    lock = fleet_state.acquire_single_instance_lock(d)
    if lock is None:
        sys.exit(f"error: another fleetd already holds "
                 f"{os.path.join(d, fleet_state.LOCK_NAME)} "
                 f"— refusing to start a second reconciler")
    fleet = Fleet(d)
    server = Server(fleet)
    server.bind()
    fleet.journal("fleetd_started", version=client.FLEET_PROTO_VERSION,
                  rev=fleet.rev,
                  pid=os.getpid(), dry_run=fleet.hooks.dry_run,
                  sock=server.sock_path, watches=len(fleet.state["watches"]))
    if args.once:
        fleet.tick()
        server.close()
        return 0
    stop = threading.Event()
    tick_thread = threading.Thread(target=_reconcile_loop,
                                   args=(fleet, stop, args.interval), daemon=True)
    sock_thread = threading.Thread(target=server.serve_forever, daemon=True)
    tick_thread.start()
    sock_thread.start()

    # The handler SETS A FLAG; it does not raise. That is what makes the
    # `finally` below run on a systemd stop/restart — `finally` does NOT run on
    # an unhandled SIGTERM, so an exception-based shutdown (or a wait loop
    # hidden in a helper that swallows the signal) silently loses `fleet.save()`
    # and the `fleetd_stopped` line on every deploy.
    def _sig(_signum: int, _frame: object) -> None:
        stop.set()
        server.stop.set()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    try:
        while not stop.is_set():
            stop.wait(1.0)
    finally:
        stop.set()
        server.stop.set()
        fleet.save()                                # clean save on SIGTERM
        server.close()
        fleet.journal("fleetd_stopped", pid=os.getpid())
    return 0


# moved-from: fleetd.cmd_install_unit
def cmd_install_unit(args: argparse.Namespace) -> int:
    """Point the unit at THIS checkout. Kept for the bootstrap/soak case, but it
    is no longer the routine path — `deploy` is, because this one can only ever
    deploy the tree it happens to be running from. The audit below is the guard:
    it refuses a linked worktree, a scratch path, a non-`main` branch or a dirty
    tree, which is every shape that has silently mis-deployed the daemon.

    `enable --now` here is the DELIBERATE opposite of `cmd_deploy`'s bare
    `enable` + `restart` (H8): `--now` no-ops on an already-active unit and
    reports success while the old config keeps running, which is fine for a
    first install and wrong for every re-deploy. Do not unify the two."""
    # NOT `__file__`: the unit must exec the Zone E launcher, whose path is a
    # frozen contract (the reaper unit, `herdd`'s subprocess wrappers and 550
    # doc references all bind `tools/vast/fleetd.py`).
    script = _FLEETD_SCRIPT
    repo = repo_root()
    bad = deploy.checkout_audit(repo)
    if bad and not args.force:
        print(f"!! refusing to install from {repo} ({len(bad)} reason(s)):")
        for b in bad:
            print(f"   - {b}")
        print(f"   use `fleetd.py deploy` (release checkout: "
              f"{deploy.deploy_checkout_path()}), or --force to override")
        return 2
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    path = os.path.join(unit_dir, client.FLEET_UNIT_NAME)
    with open(path, "w") as f:
        f.write(deploy.render_unit(sys.executable, script, repo,
                                   dry_run=args.dry_run or dry_run_enabled()))
    print(f"wrote {path}")
    if args.no_enable:
        print("next steps:")
        print("  systemctl --user daemon-reload")
        print(f"  systemctl --user enable --now {client.FLEET_UNIT_NAME}")
        print("  loginctl enable-linger $USER   # survive logout — REQUIRED")
        return 0
    subprocess.call(["systemctl", "--user", "daemon-reload"])
    rc = subprocess.call(["systemctl", "--user", "enable", "--now",
                          client.FLEET_UNIT_NAME])
    print(f"systemctl --user enable --now {client.FLEET_UNIT_NAME} -> rc={rc}")
    # S5: without lingering the daemon dies at logout and the fleet goes dark.
    lrc = subprocess.call(["loginctl", "enable-linger",
                           os.environ.get("USER") or ""])
    print(f"loginctl enable-linger -> rc={lrc}")
    print("  herdd fleet ping             # verify")
    return rc


# moved-from: fleetd.cmd_status
def cmd_status(args: argparse.Namespace) -> int:
    f = Fleet()
    print(json.dumps(f.status(), indent=2, sort_keys=True, default=str))
    return 0


# moved-from: fleetd.main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="fleetd — persistent fleet-supervision daemon "
                    "(tools/vast/FLEETD_DESIGN.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the daemon (socket + reconcile tick)")
    s.add_argument("--once", action="store_true",
                   help="run exactly one reconcile tick and exit (soak/debug)")
    s.add_argument("--interval", type=float, default=None,
                   help=f"reconcile interval seconds (default {TICK_S:g} +- jitter)")
    s.set_defaults(fn=cmd_serve)
    u = sub.add_parser("install-unit", help="generate + enable the systemd user "
                                            "unit pointing at THIS checkout")
    u.add_argument("--no-enable", action="store_true")
    u.add_argument("--force", action="store_true",
                   help="install even if this checkout fails the release audit")
    u.add_argument("--dry-run", action="store_true",
                   help="bake FLEETD_DRY_RUN=1 into the unit (read-only soak)")
    u.set_defaults(fn=cmd_install_unit)
    d = sub.add_parser("deploy", help="update the RELEASE checkout to a known "
                                      "revision, re-point the unit, restart, "
                                      "and verify the live rev")
    d.add_argument("--checkout", default=None,
                   help=f"release checkout (default ${deploy.DEPLOY_CHECKOUT_ENV} "
                        f"or {deploy.DEPLOY_CHECKOUT_DEFAULT}); cloned if absent")
    d.add_argument("--ref", default=None,
                   help=f"revision to deploy (default: {deploy.DEPLOY_LOCAL_REF} — "
                        f"the LOCAL repo's main, which is where landed work "
                        f"lives; {deploy.DEPLOY_REF_DEFAULT} when the local "
                        f"fetch fails)")
    d.add_argument("--source", default=None,
                   help="repo to bootstrap-clone from and to link .env from "
                        "(default: the checkout this script runs from)")
    d.add_argument("--python", default=None,
                   help=f"interpreter to bake (default "
                        f"{deploy.DEPLOY_PYTHON_DEFAULT} if present, else the "
                        f"running one)")
    d.add_argument("--no-restart", action="store_true",
                   help="write the unit but do not daemon-reload/restart")
    d.add_argument("--force", action="store_true",
                   help="install even if the release audit fails")
    d.add_argument("--dry-run", action="store_true",
                   help="bake FLEETD_DRY_RUN=1 into the unit (read-only soak)")
    d.add_argument("--verify-timeout", type=float, default=60.0,
                   help="seconds to wait for the daemon to report the new rev")
    d.set_defaults(fn=deploy.cmd_deploy)
    st = sub.add_parser("status", help="dump the persisted state (no daemon needed)")
    st.set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    rc: int = args.fn(args)
    return rc
