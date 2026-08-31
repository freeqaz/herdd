"""`vastlib.fleet.hooks` — the daemon's I/O seam, and the shapes it must keep.

Why this file exists
--------------------
`Hooks` is the whole reason `fleetd` is testable, and the port changes 16 of its
binding targets at once (`herdd.<attr>` -> the `vastlib` module that now owns
each primitive). Three classes of breakage would be silent:

1. **A tri-state collapsed to a bool.** `instances()` returning `[]` instead of
   `None` on a failed API read tells the daemon the fleet is EMPTY, which alarms
   and parks every watched box. `jobd_status_line()` returning `""` instead of
   `None` accumulates confirm time toward a park on a read that never happened —
   a B2 outage would park the fleet. `drained()`/`results_present()` are the
   pair a typing pass most wants to "normalize": drained is True on an empty
   queue, results_present is None on the same input, and that asymmetry gates a
   destroy.
2. **A binding that resolves to the wrong module** — or, worse, a
   `from x import fn` that makes every existing patch site vacuous. Each seam is
   exercised here by patching the MODULE ATTRIBUTE and asserting the patch was
   taken (plan §8b).
3. **A signature drift against the flat class.** `fleetd.Hooks` WAS a second
   class and the ported one was compared to it method-for-method,
   signature-for-signature. Plan §8 step 6d emptied `fleetd.py`: `fleetd.Hooks`
   is now an identity re-export of `hooks.Hooks`, so those comparisons compare
   a class with itself and are deleted. `test_fleetd.py`'s `FakeHooks` is still
   arity-checked against `fleetd.Hooks` — that is the arm that still has teeth,
   and it now lands on this class. What stays here is the same shape check
   against the roster (`_METHODS`) and the `Protocol`, which is what the flat
   comparison was a proxy for.

The `Protocol` itself is checked by mypy inside the package (see the
`_assert_hooks_implements_the_protocol` stub at the bottom of `hooks.py` — an
`isinstance` check can only see method NAMES, so it is not a substitute).

What is deliberately NOT here
-----------------------------
* No daemon policy. Whether a park should happen is `fleet/daemon.py`'s; this
  file only proves the hook does what it says when asked.
* No live API, B2 or socket. Every seam is stubbed at its module attribute; the
  one PUT (`keep_label`) is asserted against a stub, and conftest's mutation
  guard is what answers if a future edit forgets to stub it.
* No repoint of `test_fleetd.py` / `test_fleetd_pyhalf.py` /
  `test_fleetd_notify.py`. They still drive `fleetd.Hooks` and still pass; they
  migrate with their caller at plan steps 6-7.

Provenance: created 2026-08-16 alongside `vastlib/fleet/hooks.py`, plan §8
step 5.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import bidpolicy                                       # noqa: E402  Zone S
import fleetd                                          # noqa: E402  twin, still live

from vastlib.boxes import health, lifecycle            # noqa: E402
from vastlib.core import api, result                   # noqa: E402
from vastlib.fleet import hooks                        # noqa: E402
from vastlib.jobs import view                          # noqa: E402
from vastlib.storage import b2                         # noqa: E402
from vastlib.supervise import job_lane, run_lane       # noqa: E402

_METHODS = ("now", "instances", "notifications", "instance", "jobd_status_line",
            "health", "park", "resume", "destroy", "keep_label", "drained",
            "results_present", "run_init", "run_tick", "run_finalize",
            "jobs_init", "jobs_tick")


@pytest.fixture
def h():                                                           # noqa: ANN201
    return hooks.Hooks(dry_run=False)


# --------------------------------------------------------------------------- #
# 1 — the seam's shape, against the Protocol and against the flat twin
# --------------------------------------------------------------------------- #
def test_default_impl_satisfies_the_protocol(h) -> None:           # noqa: ANN001
    assert isinstance(h, hooks.FleetHooks)


def test_protocol_and_impl_cover_exactly_the_roster() -> None:
    """Post-6d form of `…_cover_exactly_the_flat_classes_methods`.

    `fleetd.Hooks` is the launcher's re-export of THIS class, so the flat set
    and the ported set were the same set. The roster (`_METHODS`) and the
    `Protocol` are the two independent statements that survive, and the
    identity assertion below keeps the deleted arm's real content: a second
    `Hooks` class in `fleetd.py` would give `test_fleetd.py`'s arity-checked
    `FakeHooks` a different contract from the daemon's.
    """
    assert fleetd.Hooks is hooks.Hooks
    ported = {n for n, _ in inspect.getmembers(hooks.Hooks, inspect.isfunction)
              if not n.startswith("__")}
    assert ported == set(_METHODS)
    proto = {n for n in dir(hooks.FleetHooks)
             if not n.startswith("_") and callable(getattr(hooks.FleetHooks, n))}
    assert proto == set(_METHODS)


@pytest.mark.parametrize("name", _METHODS)
def test_signatures_match_the_protocol(name) -> None:              # noqa: ANN001
    """Parameter NAMES and order are the contract — the daemon calls these
    positionally and `FakeHooks` mirrors them.

    The flat-class arm of this test (`inspect.signature(fleetd.Hooks.<name>)`)
    went at step 6d with the flat class; the `Protocol` is the remaining
    independent declaration of the same shape.
    """
    ported = inspect.signature(getattr(hooks.Hooks, name))
    proto = inspect.signature(getattr(hooks.FleetHooks, name))
    assert list(proto.parameters) == list(ported.parameters), name


def test_dry_run_defaults_from_the_env_per_construction(monkeypatch) -> None:
    monkeypatch.delenv("FLEETD_DRY_RUN", raising=False)
    assert hooks.Hooks().dry_run is False
    monkeypatch.setenv("FLEETD_DRY_RUN", "1")
    assert hooks.Hooks().dry_run is True
    # NOT a flat-twin comparison and NOT tautological post-6d: `fleetd.dry_run_
    # enabled` re-exports `fleet.daemon`'s predicate, which is a SECOND body
    # (`fleet.deploy` has a third, and bakes the unit's `Environment=` line
    # from it). Three copies of one env read, pinned equal here and in
    # `test_vastlib_fleet_deploy.py`.
    assert hooks.dry_run_enabled() == fleetd.dry_run_enabled()
    monkeypatch.setenv("FLEETD_DRY_RUN", "yes")     # only "1" arms it
    assert hooks.Hooks().dry_run is False
    assert hooks.Hooks(dry_run=True).dry_run is True   # explicit beats env


def test_now_is_the_daemons_only_clock(h, monkeypatch) -> None:    # noqa: ANN001
    monkeypatch.setattr(hooks.time, "time", lambda: 1234.5)
    assert h.now() == 1234.5


# --------------------------------------------------------------------------- #
# 2 — the tri-states (each None is load-bearing)
# --------------------------------------------------------------------------- #
def test_instances_none_on_a_failed_read_is_not_an_empty_fleet(h,  # noqa: ANN001
                                                               monkeypatch) -> None:
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(False, None, "network"))
    assert h.instances() is None, "[] here alarms and parks every watched box"


def test_instances_unwraps_and_normalises(h, monkeypatch) -> None:  # noqa: ANN001
    seen: list[tuple] = []

    def _req(method, path, *a, **k):
        seen.append((method, path))
        return result.Soft(True, {"instances": [{"id": 41}]}, None)

    monkeypatch.setattr(api, "request_soft", _req)
    assert h.instances() == [{"id": 41}]
    assert seen == [("GET", "v1/instances/")]
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(True, [{"id": 7}], None))
    assert h.instances() == [{"id": 7}]
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(True, None, None))
    assert h.instances() == [], "a SUCCESSFUL empty read is a real empty fleet"


def test_notifications_never_waits_on_the_retry_ladder(h,          # noqa: ANN001
                                                       monkeypatch) -> None:
    import notify
    seen: dict = {}

    def _req(method, path, **kw):
        seen.update({"method": method, "path": path, **kw})
        return result.Soft(True, {"rows": []}, None)

    monkeypatch.setattr(api, "request_soft", _req)
    assert h.notifications() == ({"rows": []}, None)
    assert seen["retries"] == 0, "the retry ladder sleeps up to 30s per attempt"
    assert seen["path"] == notify.INBOX_PATH


def test_notifications_never_raises(h, monkeypatch) -> None:       # noqa: ANN001
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(False, None, "HTTP 503"))
    assert h.notifications() == (None, "HTTP 503")
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(False, None, None))
    assert h.notifications() == (None, "unknown error")

    def _boom(*a, **k):
        raise RuntimeError("transport surprise")

    monkeypatch.setattr(api, "request_soft", _boom)
    payload, err = h.notifications()
    assert payload is None and "RuntimeError" in str(err)


def test_instance_swallows_to_none(h, monkeypatch) -> None:        # noqa: ANN001
    monkeypatch.setattr(health, "_get_instance_soft", lambda iid: {"id": iid})
    assert h.instance(41) == {"id": 41}

    def _boom(_iid):
        raise RuntimeError("api gone")

    monkeypatch.setattr(health, "_get_instance_soft", _boom)
    assert h.instance(41) is None


# --------------------------------------------------------------------------- #
# 3 — jobd_status_line: the B2 read whose None must not park a fleet
# --------------------------------------------------------------------------- #
def test_jobd_status_line_reads_the_marker(h, monkeypatch) -> None:  # noqa: ANN001
    seen: list[list[str]] = []

    def _rclone(args):
        seen.append(list(args))
        return result.ProcResult(0, "state=running pyhalf=broken\n", "")

    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft", _rclone)
    assert h.jobd_status_line(41) == "state=running pyhalf=broken\n"
    assert seen == [["cat", "b2:bkt/jobs/nodes/41/JOBD_STATUS"]]


@pytest.mark.parametrize("outcome", ["no-bucket", "no-iid", "rc", "empty",
                                     "raise"])
def test_jobd_status_line_is_none_for_every_unknown(h, monkeypatch,  # noqa: ANN001
                                                    outcome) -> None:
    monkeypatch.setenv("B2_BUCKET", "bkt")
    iid: object = 41
    if outcome == "no-bucket":
        monkeypatch.delenv("B2_BUCKET")
    elif outcome == "no-iid":
        iid = None

    def _rclone(args):
        if outcome == "raise":
            raise RuntimeError("b2 down")
        if outcome == "rc":
            return result.ProcResult(1, "", "not found")
        return result.ProcResult(0, "   \n", "")

    monkeypatch.setattr(b2, "_rclone_soft", _rclone)
    assert h.jobd_status_line(iid) is None, outcome


# --------------------------------------------------------------------------- #
# 4 — health: alarms only, so both failure modes collapse to {}
# --------------------------------------------------------------------------- #
def test_health_folds_live_ids_and_delegates(h, monkeypatch) -> None:  # noqa: ANN001
    seen: dict = {}

    def _fold(live, prog=None):
        seen["live"] = live
        return {"41": ["job"]}

    def _gather(instances, jobs_by_box, **kw):
        seen["jobs"] = jobs_by_box
        return {"41": {"verdict": "OK"}}

    monkeypatch.setattr(view, "_fold_fleet_jobs", _fold)
    monkeypatch.setattr(health, "gather_fleet_health", _gather)
    inst = [{"id": 41, "actual_status": "RUNNING"},
            {"id": 42, "actual_status": "exited"}]
    assert h.health(inst) == {"41": {"verdict": "OK"}}
    assert seen["live"] == {41}, "the filter is bidpolicy.LIVE_STATES, lowercased"
    assert "running" in bidpolicy.LIVE_STATES
    assert seen["jobs"] == {"41": ["job"]}


def test_health_swallows_both_failure_modes(h, monkeypatch) -> None:  # noqa: ANN001
    def _boom(*a, **k):
        raise RuntimeError("b2 down")

    monkeypatch.setattr(view, "_fold_fleet_jobs", _boom)
    monkeypatch.setattr(health, "gather_fleet_health",
                        lambda inst, jobs, **k: {"folded": jobs})
    assert h.health([]) == {"folded": {}}, "a dead fold still gathers"
    monkeypatch.setattr(health, "gather_fleet_health", _boom)
    assert h.health([]) == {}


# --------------------------------------------------------------------------- #
# 5 — the money paths: dry-run is a no-op, and slot 2 is a NOTE there
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("method", "state"), [("park", "stopped"),
                                               ("resume", "running")])
def test_park_and_resume_put_the_state(h, monkeypatch, method,     # noqa: ANN001
                                       state) -> None:
    seen: list[tuple] = []
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: seen.append((iid, st))
                        or result.OkErr(True, None))
    assert tuple(getattr(h, method)(41)) == (True, None)
    assert seen == [(41, state)]


def test_destroy_delegates_to_the_soft_primitive(h, monkeypatch) -> None:  # noqa: ANN001
    seen: list[object] = []
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid: seen.append(iid) or result.OkErr(False, "409"))
    assert tuple(h.destroy(41)) == (False, "409")
    assert seen == [41]


@pytest.mark.parametrize("method", ["park", "resume", "destroy"])
def test_dry_run_short_circuits_before_any_call(monkeypatch, method) -> None:
    """`(True, "dry-run")`: slot 2 on the OK path is a human-readable NOTE, not
    an error — the reap/lifecycle manifest pins the same shape."""
    def _boom(*a, **k):
        raise AssertionError("a dry run must not reach the API")

    monkeypatch.setattr(lifecycle, "_put_state_soft", _boom)
    monkeypatch.setattr(lifecycle, "_destroy_soft", _boom)
    assert getattr(hooks.Hooks(dry_run=True), method)(41) == (True, "dry-run")


def test_keep_label_stamps_the_token_before_a_park(h, monkeypatch) -> None:  # noqa: ANN001
    seen: list[tuple] = []

    def _req(method, path, body=None, **kw):
        seen.append((method, path, body))
        return result.Soft(True, {}, None)

    monkeypatch.setattr(api, "request_soft", _req)
    assert h.keep_label(41, {"label": "run:R1"}) == (True, "run:R1:keep")
    assert seen == [("PUT", "v0/instances/41/", {"label": "run:R1:keep"})]
    seen.clear()
    assert h.keep_label(41, None) == (True, "keep:fleetd-park")
    assert h.keep_label(41, {}) == (True, "keep:fleetd-park")


def test_keep_label_leaves_an_already_kept_label_alone(h,          # noqa: ANN001
                                                       monkeypatch) -> None:
    def _boom(*a, **k):
        raise AssertionError("no PUT for a label that already keeps")

    monkeypatch.setattr(api, "request_soft", _boom)
    assert h.keep_label(41, {"label": "keep:fleetd-park"}) == \
        (False, "keep:fleetd-park")


def test_keep_label_dry_run_computes_without_putting(monkeypatch) -> None:
    def _boom(*a, **k):
        raise AssertionError("a dry run must not PUT")

    monkeypatch.setattr(api, "request_soft", _boom)
    assert hooks.Hooks(dry_run=True).keep_label(41, {"label": "x"}) == \
        (True, "x:keep")


def test_keep_label_reports_the_old_label_on_failure(h,            # noqa: ANN001
                                                     monkeypatch) -> None:
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(False, None, "HTTP 500"))
    assert h.keep_label(41, {"label": "run:R1"}) == (False, "run:R1")


# --------------------------------------------------------------------------- #
# 6 — drained vs results_present: the asymmetry is the contract
# --------------------------------------------------------------------------- #
def _queue(monkeypatch, jids, views=None) -> None:                 # noqa: ANN001
    monkeypatch.setattr(hooks.jobmeta, "list_queue", lambda iid: list(jids))
    by_id = dict(zip(jids, views or []))
    monkeypatch.setattr(hooks.jobmeta, "read_job", lambda j: by_id[j])


def test_empty_queue_is_drained_but_not_evidence_of_results(h,     # noqa: ANN001
                                                            monkeypatch) -> None:
    _queue(monkeypatch, [])
    assert h.drained(41) is True, "drain IS provable from absence"
    assert h.results_present(41) is None, "publication is NOT"


def test_drained_is_all_terminal(h, monkeypatch) -> None:          # noqa: ANN001
    term = sorted(hooks.jobmeta.TERMINAL)[0]
    _queue(monkeypatch, ["j1", "j2"], [{"status": term}, {"status": term}])
    assert h.drained(41) is True
    _queue(monkeypatch, ["j1", "j2"], [{"status": term}, {"status": "running"}])
    assert h.drained(41) is False


@pytest.mark.parametrize("method", ["drained", "results_present"])
def test_unknown_is_none_for_both(h, monkeypatch, method) -> None:  # noqa: ANN001
    def _boom(*a, **k):
        raise RuntimeError("b2 down")

    monkeypatch.setattr(hooks.jobmeta, "list_queue", _boom)
    assert getattr(h, method)(41) is None
    monkeypatch.setattr(hooks.jobmeta, "list_queue", lambda iid: ["j1"])
    monkeypatch.setattr(hooks.jobmeta, "read_job", _boom)
    assert getattr(h, method)(41) is None


def test_results_present_reads_the_three_publication_keys(h,       # noqa: ANN001
                                                          monkeypatch) -> None:
    _queue(monkeypatch, ["j1"], [{"status": "done", "results_key": "k"}])
    assert h.results_present(41) is True
    _queue(monkeypatch, ["j1"], [{"status": "succeeded", "published": True}])
    assert h.results_present(41) is True
    _queue(monkeypatch, ["j1"], [{"status": "done"}])
    assert h.results_present(41) is False
    # nothing DONE yet is not the same as nothing published
    _queue(monkeypatch, ["j1"], [{"status": "running"}])
    assert h.results_present(41) is None


def test_jobmeta_is_reached_bare_name_not_through_a_package() -> None:
    """Zone S: `herdd.jobmeta.*` became `jobmeta.*`, not `vastlib.jobmeta.*`
    (which resolves nowhere and would be a zone violation as well)."""
    import jobmeta
    assert hooks.jobmeta is jobmeta
    assert not getattr(jobmeta, "__package__", "").startswith("vastlib")


# --------------------------------------------------------------------------- #
# 7 — the lane delegations (the step-5 rebind those manifests deferred)
# --------------------------------------------------------------------------- #
def test_run_lane_delegations(h, monkeypatch) -> None:             # noqa: ANN001
    seen: list[tuple] = []
    monkeypatch.setattr(run_lane, "supervise_init",
                        lambda a: seen.append(("init", a)) or ({}, {}, True))
    monkeypatch.setattr(run_lane, "supervise_tick",
                        lambda st, a, hf, on: seen.append(("tick", on)) or "act")
    assert h.run_init("A") == ({}, {}, True)
    assert h.run_tick({}, "A", {}, True) == "act"
    assert [s[0] for s in seen] == ["init", "tick"]


def test_run_finalize_pins_destroy_on_park_failure_false(h,        # noqa: ANN001
                                                         monkeypatch) -> None:
    """FLEETD_DESIGN §3/§8: the daemon parks and alarms, and NEVER originates a
    destroy. A policy constant, not a default — `supervise_finalize` defaults it
    to True for the inline CLI."""
    real = inspect.signature(run_lane.supervise_finalize)
    assert real.parameters["destroy_on_park_failure"].default is True
    seen: dict = {}
    monkeypatch.setattr(run_lane, "supervise_finalize",
                        lambda *a, **kw: seen.update(kw))
    h.run_finalize({}, "A", "act", {}, True)
    assert seen == {"destroy_on_park_failure": False}


def test_job_lane_delegations(h, monkeypatch) -> None:             # noqa: ANN001
    monkeypatch.setattr(job_lane, "job_supervise_init",
                        lambda a: ({"jc": a}, {"hf": 1}))
    monkeypatch.setattr(job_lane, "job_supervise_tick",
                        lambda jc, hf: "queue_empty")
    assert h.jobs_init("A") == ({"jc": "A"}, {"hf": 1})
    assert h.jobs_tick({}, {}) == "queue_empty"


def test_every_seam_is_bound_as_a_module_attribute(h, monkeypatch) -> None:  # noqa: ANN001
    """A `from x import fn` anywhere in the chain makes every patch site above
    (and every existing one, once they migrate) vacuous — green tests steering
    nothing. Each seam is re-patched here through its owning module and the
    hook must take the patch."""
    sentinels = {
        (api, "request_soft"): lambda *a, **k: result.Soft(
            True, {"instances": [{"S": 1}]}, None),
        (health, "_get_instance_soft"): lambda iid: {"S": 1},
        (b2, "_rclone_soft"): lambda args: result.ProcResult(0, "S", ""),
        (lifecycle, "_put_state_soft"): lambda i, s: result.OkErr(True, "S"),
        (lifecycle, "_destroy_soft"): lambda i: result.OkErr(True, "S"),
        (health, "gather_fleet_health"): lambda i, j, **k: {"S": 1},
        (view, "_fold_fleet_jobs"): lambda live, prog=None: {"S": []},
        (run_lane, "supervise_init"): lambda a: "S",
        (run_lane, "supervise_tick"): lambda *a: "S",
        (run_lane, "supervise_finalize"): lambda *a, **k: "S",
        (job_lane, "job_supervise_init"): lambda a: "S",
        (job_lane, "job_supervise_tick"): lambda *a: "S",
    }
    for (mod, name), fn in sentinels.items():
        assert hasattr(mod, name), f"{mod.__name__}.{name} — patch target gone"
        monkeypatch.setattr(mod, name, fn)
    monkeypatch.setenv("B2_BUCKET", "bkt")
    assert h.instances() == [{"S": 1}]
    assert h.instance(41) == {"S": 1}
    assert h.jobd_status_line(41) == "S"
    assert h.health([]) == {"S": 1}
    assert tuple(h.park(41)) == (True, "S")
    assert tuple(h.destroy(41)) == (True, "S")
    assert h.run_init("A") == "S"
    assert h.jobs_tick({}, {}) == "S"
