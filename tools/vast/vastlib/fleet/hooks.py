"""vastlib.fleet.hooks — the daemon's ONE I/O seam, as a Protocol plus its
production binding.

Why this exists
---------------
`fleetd` does no I/O of its own: every read of the vast API, every B2 look, every
mutating PUT/DELETE and every clock read goes through one object, and the tests
inject a fake one (the fake-transport discipline `test_supervise.py` and
`test_jobd.py` established). That object — `fleetd.Hooks` — was also where the
daemon's coupling to the CLI concentrated: 23 `herdd.<attr>` references across
16 distinct names inside 185 lines (measured at ea8360dc; the plan's inherited
"21 of 49" is the older count).

Two things happen here. The seam becomes a **typed Protocol**, so the shape the
daemon depends on is stated once and machine-checked — and, per plan §7's
signature-only-coverage finding, this is the FIRST machine-checked statement of
the lane signatures at all (`test_fleetd.py` and `test_standing_watch.py` only
arity-check their `FakeHooks` against them). And the default implementation
rebinds from `herdd.*` to the `vastlib` modules that now own each primitive,
in module-attribute form so the existing patch idiom survives (plan §8b).

Where each production binding now points
----------------------------------------
    request_soft            -> `vastlib.core.api`            (landed)
    _get_instance_soft      -> `vastlib.boxes.health`        (landed)
    _rclone_soft            -> `vastlib.storage.b2`          (landed)
    _put_state_soft         -> `vastlib.boxes.lifecycle`     (landed)
    _destroy_soft           -> `vastlib.boxes.lifecycle`     (landed)
    _reap_kept              -> `vastlib.core.labels`         (landed)
    _fold_fleet_jobs        -> `vastlib.jobs.view`           (landed, step 5)
    gather_fleet_health     -> `vastlib.boxes.health`        (landed)
    supervise_*             -> `vastlib.supervise.run_lane`  (landed)
    job_supervise_*         -> `vastlib.supervise.job_lane`  (landed)
    LIVE_STATES             -> `bidpolicy` (Zone S, bare name — same object
                               `herdd` re-exported)
    jobmeta.*               -> `jobmeta`  (Zone S, bare name; see below)
    notify.INBOX_PATH       -> `notify`   (flat pure leaf, bare name)

`Hooks.drained` / `Hooks.results_present` reached `jobmeta` THROUGH the CLI
(`herdd.jobmeta.list_queue`), which is a Zone S shipped leaf arriving via a
re-export. They import it bare-name here. Mechanically rewriting that prefix to
`vastlib.jobmeta` would produce an import that cannot resolve AND a Zone S
violation in the same line.

Two shapes that look like bugs and are not
------------------------------------------
* **`drained` returns True on an empty queue; `results_present` returns None.**
  Drain is provable from absence — no tickets, nothing running — while
  publication is not: an empty queue is no evidence that anything was ever
  published. A typing pass that "normalizes" the pair breaks the S3 destroy
  gate, so the asymmetry is stated in both docstrings and pinned in
  `test_vastlib_fleet_hooks.py`.
* **`jobd_status_line` is not consolidated with `boxes.health`'s pyhalf read.**
  Its 40-line docstring argues why, and that reasoning is ported verbatim.

What is deliberately NOT here
-----------------------------
* **No policy.** Every method is a read, a write, or a delegation. What a
  verdict MEANS is `boxes.health`; what to do about an eviction is `supervise/`.
* **No `Fleet`/`Server`/tick loop** — `fleet/daemon.py`.
* **No merge with `ladder_core.LaneHooks`.** Same suffix, unrelated purpose (it
  is the bid ladder's observation seam). A reader grepping `Hooks` in this tree
  gets three unrelated hits; this is one of them.

Provenance: moved from `fleetd.py` (`class Hooks`), plan §8 step 5.
Behavior-preserving: the only changes are the binding targets and the types.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import notify

from vastlib.boxes import health, lifecycle
from vastlib.core import api, labels
from vastlib.jobs import view
from vastlib.storage import b2
from vastlib.supervise import job_lane, run_lane

import bidpolicy
import jobmeta

if TYPE_CHECKING:                                   # pragma: no cover - typing only
    import argparse
    from collections.abc import Mapping, MutableMapping, Sequence


# --------------------------------------------------------------------------- #
# the seam, as a type
# --------------------------------------------------------------------------- #
@runtime_checkable
class FleetHooks(Protocol):
    """Everything the daemon is allowed to touch outside its own state.

    Implemented by `Hooks` (production) and by every `FakeHooks` in the test
    suite. The tri-state returns are the load-bearing part of this contract:

      `instances()`         None  = the API read FAILED. It is NOT an empty
                                    fleet — reading it that way would alarm and
                                    park every watched box at once.
      `jobd_status_line()`  None  = unknown (no bucket, no marker, B2 error).
                                    `_pyhalf_tick` must not accumulate confirm
                                    time on a read it could not make.
      `drained()`           None  = unknown; True on an EMPTY queue.
      `results_present()`   None  = unknown; None on an EMPTY queue too — see
                                    the module docstring for why the pair is
                                    deliberately asymmetric.
      `health()`            `{}`  = the fold failed. Alarms only, never an
                                    action, which is exactly why
                                    `jobd_status_line` stays independent of it.

    `park` / `resume` / `destroy` return `(ok, note_or_err)` where slot 2 on the
    dry-run OK path is the human-readable NOTE `"dry-run"`, not an error.
    """

    #: True when every mutating call is a no-op (env `FLEETD_DRY_RUN`).
    dry_run: bool

    def now(self) -> float: ...

    def instances(self) -> list[Any] | None: ...

    def notifications(self) -> tuple[Any, str | None]: ...

    def instance(self, iid: object) -> dict[str, Any] | None: ...

    def jobd_status_line(self, iid: object) -> str | None: ...

    def health(self, instances: Sequence[Any]) -> dict[str, Any]: ...

    def park(self, iid: object) -> tuple[bool, str | None]: ...

    def resume(self, iid: object) -> tuple[bool, str | None]: ...

    def destroy(self, iid: object) -> tuple[bool, str | None]: ...

    def keep_label(self, iid: object,
                   inst: Mapping[str, Any] | None) -> tuple[bool, str | None]: ...

    def drained(self, iid: object) -> bool | None: ...

    def results_present(self, iid: object) -> bool | None: ...

    def run_init(self, a: argparse.Namespace) -> tuple[dict[str, Any],
                                                       MutableMapping[str, Any],
                                                       bool]: ...

    def run_tick(self, st: MutableMapping[str, Any], a: argparse.Namespace,
                 hf: MutableMapping[str, Any],
                 handoff_on: bool) -> Any: ...      # noqa: ANN401 — bidpolicy.Action|None

    def run_finalize(self, st: MutableMapping[str, Any], a: argparse.Namespace,
                     act: Any,                      # noqa: ANN401 — bidpolicy.Action
                     hf: MutableMapping[str, Any], handoff_on: bool) -> None: ...

    def jobs_init(self, a: argparse.Namespace) -> tuple[dict[str, Any],
                                                        dict[str, Any]]: ...

    def jobs_tick(self, jc: MutableMapping[str, Any],
                  hf: MutableMapping[str, Any]) -> str | None: ...


# COLLAPSE CANDIDATE, flagged not fixed: `fleet/daemon.py` and `fleet/deploy.py`
# each carry their own copy of this one-line env read (as `fleetd.py` did not —
# there it was defined once and imported by all three consumers). It is defined
# here because `Hooks(dry_run=None)` defaults from it and `hooks` must not
# import `daemon` (daemon imports hooks). Whoever integrates the fleet package
# should point all three at ONE definition; three copies of a kill-switch is the
# same shape of hazard as the two protocol-version literals this port collapsed.
#
# DELIBERATELY MARKER-LESS (ruled 2026-08-16, wave 6a; fleetd-reexports H4):
# `fleet/daemon.py::dry_run_enabled` owns the `fleetd.dry_run_enabled` mapping —
# the flat name was the DAEMON's kill-switch read, and one flat name cannot have
# two rename targets. This copy exists only because `Hooks(dry_run=None)`
# defaults from it and `hooks` must not import `daemon`; the collapse above is
# still owed, and it is what will delete this definition.
def dry_run_enabled() -> bool:
    return os.environ.get("FLEETD_DRY_RUN") == "1"


# --------------------------------------------------------------------------- #
# the production binding
# --------------------------------------------------------------------------- #
# moved-from: fleetd.Hooks
class Hooks:
    """Production bindings. Every mutating call is a no-op under FLEETD_DRY_RUN."""

    def __init__(self, dry_run: bool | None = None) -> None:
        self.dry_run = dry_run_enabled() if dry_run is None else dry_run

    def now(self) -> float:
        return time.time()

    def instances(self) -> list[Any] | None:
        """Live fleet snapshot, or None when the API read failed (a failed read
        must NEVER be read as 'the fleet is empty' — that would alarm/park
        every watched box)."""
        ok, data, _err = api.request_soft("GET", "v1/instances/")
        if not ok:
            return None
        inst = data.get("instances", data) if isinstance(data, dict) else data
        return list(inst or [])

    def notifications(self) -> tuple[Any, str | None]:
        """vast's notification inbox as `(payload, err)` — NOTIFY_DESIGN S2a.

        `retries=0` on purpose. `request_soft`'s default retry ladder sleeps up
        to 30 s per attempt, and this read is the one poll in the tick that
        NOTHING depends on (D2: evidence only) — making the whole fleet's
        reconcile wait on a hidden endpoint's blip would invert that. The next
        tick is 45 s away and the feed's window is ~3 days deep, so a skipped
        poll loses nothing.

        Never raises: a transport surprise degrades to `(None, err)`, and the
        caller journals only when the poll's HEALTH changes."""
        try:
            ok, data, err = api.request_soft("GET", notify.INBOX_PATH,
                                             retries=0)
        except Exception as e:                      # pragma: no cover - paranoia
            return None, f"error {type(e).__name__}: {e}"
        return (data if ok else None), (None if ok else (err or "unknown error"))

    def instance(self, iid: object) -> dict[str, Any] | None:
        """Fresh single-instance read (S3: re-stat immediately before a DELETE)."""
        try:
            return health._get_instance_soft(iid)
        except Exception:
            return None

    def jobd_status_line(self, iid: object) -> str | None:
        """Raw body of jobs/nodes/<iid>/JOBD_STATUS, or None when it cannot be
        read (no bucket, absent marker, B2 error). Tri-state ON PURPOSE: None
        means "we do not know", and _pyhalf_tick must never accumulate confirm
        time on a read it could not make — otherwise a B2 outage would park the
        whole fleet at once.

        Stays inside the Hooks seam (the daemon itself does no I/O) and reuses
        the soft rclone primitive, so it inherits the never-raises contract
        every other read-only sweep here relies on.

        KEPT DELIBERATELY, reviewed 2026-08-14 when `herdd` grew its own
        pyhalf read (the ZOMBIE_PYHALF verdict). Consuming that instead — i.e.
        taking the field off `self.hooks.health()`'s evidence — would be one
        less B2 read and is the wrong trade three times over:

          * `gather_fleet_health` reads JOBD_STATUS only when a box has no
            FRESH folded job event, so its `pyhalf` is None (unknown) for up to
            GUARD_JOBD_STALE_S on exactly the box that just broke. This tick
            reads unconditionally.
          * the health fold is cached and up to ~4 ticks (~3 min) stale, and
            `Hooks.health` swallows its own failures to `{}`. An alarm may lag
            and may be sampled; a rule that ends a box's right to BILL may not.
          * the health path is an INFERENCE pipeline — the same one whose
            documented reasoning about idle boxes was falsified by the
            2026-08-13 incident (FAILCLOSED_DESIGN §1.3). A confession should
            not have to travel through it to be acted on, for the same reason
            §5 puts the beacon strictly below the half it reports on.

        The duplication that WAS removed is the parse: `pyhalf_broken`
        delegates to `boxes.health.jobd_status_pyhalf`. One reading of the
        bytes, two independent paths to them."""
        bucket = os.environ.get("B2_BUCKET")
        if not bucket or iid is None:
            return None
        try:
            rc, out, _ = b2._rclone_soft(
                ["cat", f"b2:{bucket}/jobs/nodes/{iid}/JOBD_STATUS"])
        except Exception:
            return None
        if rc != 0 or not (out or "").strip():
            return None
        return out

    def health(self, instances: Sequence[Any]) -> dict[str, Any]:
        """gather_fleet_health verdicts (N1) — ALARMS only, never an action."""
        try:
            live = {i.get("id") for i in instances
                    if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES}
            jobs_by_box = view._fold_fleet_jobs(live)
        except Exception:
            jobs_by_box = {}
        try:
            return health.gather_fleet_health(instances, jobs_by_box)
        except Exception:
            return {}

    def park(self, iid: object) -> tuple[bool, str | None]:
        if self.dry_run:
            return True, "dry-run"
        return lifecycle._put_state_soft(iid, "stopped")

    def resume(self, iid: object) -> tuple[bool, str | None]:
        if self.dry_run:
            return True, "dry-run"
        return lifecycle._put_state_soft(iid, "running")

    def destroy(self, iid: object) -> tuple[bool, str | None]:
        if self.dry_run:
            return True, "dry-run"
        return lifecycle._destroy_soft(iid)

    def keep_label(self, iid: object,
                   inst: Mapping[str, Any] | None) -> tuple[bool, str | None]:
        """B4: `herdd reap` DESTROYS stopped boxes idle > 2h unless the label
        carries a `keep` token. Every fleetd park is a resumability promise, so
        stamp the token before parking. Returns (changed, label|None)."""
        label = (inst or {}).get("label") or ""
        if labels._reap_kept(label):
            return False, label
        new = (label + ":keep") if label else "keep:fleetd-park"
        if self.dry_run:
            return True, new
        ok, _d, _e = api.request_soft("PUT", f"v0/instances/{iid}/",
                                      {"label": new})
        return bool(ok), (new if ok else label)

    def drained(self, iid: object) -> bool | None:
        """True when every ticket on the box is terminal; None when unknown.

        True on an EMPTY queue — and `results_present` returns None on the same
        input. See the module docstring: the asymmetry is the contract."""
        try:
            jids = jobmeta.list_queue(str(iid))
        except Exception:
            return None
        if not jids:
            return True
        try:
            views = [jobmeta.read_job(j) for j in jids]
        except Exception:
            return None
        return all(v["status"] in jobmeta.TERMINAL for v in views)

    def results_present(self, iid: object) -> bool | None:
        """S3: True/False when we can tell whether this box's jobs published
        results to B2, None when unknown (never blocks on unknown).

        None — not True — on an EMPTY queue: no ticket is no evidence that
        anything was published, while it IS evidence that nothing is running."""
        try:
            jids = jobmeta.list_queue(str(iid))
        except Exception:
            return None
        if not jids:
            return None
        try:
            views = [jobmeta.read_job(j) for j in jids]
        except Exception:
            return None
        done = [v for v in views if v.get("status") in ("done", "succeeded")]
        if not done:
            return None
        return all(bool(v.get("results") or v.get("results_key")
                        or v.get("published")) for v in done)

    # --- profile ticks: the EXISTING supervise policy, called from the daemon --
    def run_init(self, a: argparse.Namespace) -> tuple[dict[str, Any],
                                                       MutableMapping[str, Any],
                                                       bool]:
        return run_lane.supervise_init(a)

    def run_tick(self, st: MutableMapping[str, Any], a: argparse.Namespace,
                 hf: MutableMapping[str, Any],
                 handoff_on: bool) -> Any:          # noqa: ANN401 — bidpolicy.Action|None
        return run_lane.supervise_tick(st, a, hf, handoff_on)

    def run_finalize(self, st: MutableMapping[str, Any], a: argparse.Namespace,
                     act: Any,                      # noqa: ANN401 — bidpolicy.Action
                     hf: MutableMapping[str, Any], handoff_on: bool) -> None:
        # destroy_on_park_failure=False: FLEETD_DESIGN §3/§8 — the daemon parks
        # and alarms on a cap, and NEVER originates a destroy. A policy
        # constant, not a default.
        return run_lane.supervise_finalize(st, a, act, hf, handoff_on,
                                           destroy_on_park_failure=False)

    def jobs_init(self, a: argparse.Namespace) -> tuple[dict[str, Any],
                                                        dict[str, Any]]:
        return job_lane.job_supervise_init(a)

    def jobs_tick(self, jc: MutableMapping[str, Any],
                  hf: MutableMapping[str, Any]) -> str | None:
        return job_lane.job_supervise_tick(jc, hf)


if TYPE_CHECKING:                                   # pragma: no cover - typing only
    # The machine-checked statement that the production binding still satisfies
    # the seam. mypy runs over the package, not the tests, so this assignment is
    # what catches a signature drift between `Hooks` and `FleetHooks` — the test
    # file's `isinstance` check only sees method NAMES.
    def _assert_hooks_implements_the_protocol(h: Hooks) -> FleetHooks:
        return h
