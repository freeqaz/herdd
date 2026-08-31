"""Portable tests for the come-online boot SLA + pick-time inet-down floor
(owner directive 2026-08-03: "longer than 10 minutes to come online is
unacceptable" — serve box 46682177 spent 39 minutes in its image pull on an
805 Mb/s host while the session sat waiting).

Two layers, both covered here:

  * PREVENTION — the auto-pickers apply a default advertised-download floor
    (LAUNCH_INET_DOWN_MBPS, 1000), bypassed by explicit --inet-down / pins /
    --any-inet, with an unfloored fallback pass on the soft pickers.
  * ENFORCEMENT — the OWNING lifecycle (fleetd watch / supervise /
    job supervise ticks) arms a boot deadline per profile milestone (jobs:
    JOBD_STATUS stamp; serve: box-written SERVE_STATUS token; run: the
    loading->running flip) and on breach destroys + excludes the machine +
    relaunches, with escalating tolerance (_boot_deadline_backoff) and the
    BOOT_MAX_HOST_RETRIES hard disarm.

The 2026-08-03 guard ruling is NOT weakened: the passive sweep still never
destroys a loading box (regression-pinned in test_guard.py /
test_parked_lifecycle.py); SLA kills belong exclusively to the lifecycle that
owns the box and re-attaches its workload.

Toolchain-free lane: no vast API, no B2 — every seam monkeypatched.
"""
import argparse
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy  # noqa: E402
import jobmeta  # noqa: E402
import subprocess  # noqa: E402
from vastlib.boxes import health, lifecycle  # noqa: E402
from vastlib.core import api  # noqa: E402
from vastlib.market import offers as market_offers  # noqa: E402
from vastlib.supervise import handoff, job_lane, journal, replacement  # noqa: E402

NOW = 2_000_000.0
SLA = 600            # BOOT_SLA_S default
MAX_RETRIES = 3      # BOOT_MAX_HOST_RETRIES default


# --------------------------------------------------------------------------- #
# pick-time floor
# --------------------------------------------------------------------------- #
def _search_ns(**kw):
    base = dict(limit=20, type="bid", num_gpus=1, unverified=False, gpu=None,
                gpu_ram=0, max_dph=None, host_disk=0, reliability=0, cuda=0,
                inet_down=None, machine=None, host=None, geo=None,
                exclude_machines=None, any_inet=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_inet_floor_applies_to_auto_pick():
    q = market_offers.build_search_query(_search_ns())
    assert q["inet_down"] == {"gte": 1000.0}


def test_explicit_inet_down_wins_and_zero_disables():
    assert market_offers.build_search_query(_search_ns(inet_down=800))["inet_down"] == \
        {"gte": 800.0}
    assert "inet_down" not in market_offers.build_search_query(_search_ns(inet_down=0))


def test_machine_or_host_pin_bypasses_the_default_floor():
    assert "inet_down" not in market_offers.build_search_query(_search_ns(machine=[5]))
    assert "inet_down" not in market_offers.build_search_query(_search_ns(host=[9]))


def test_any_inet_escape_hatch_bypasses_the_default_floor():
    assert "inet_down" not in market_offers.build_search_query(_search_ns(any_inet=True))


def test_floor_knob_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("LAUNCH_INET_DOWN_MBPS", "3000")
    assert market_offers.build_search_query(_search_ns())["inet_down"] == {"gte": 3000.0}
    monkeypatch.setenv("LAUNCH_INET_DOWN_MBPS", "0")     # 0 = floor off globally
    assert "inet_down" not in market_offers.build_search_query(_search_ns())


def test_pick_cheapest_offer_floors_by_default_then_falls_back(monkeypatch):
    """The soft picker (relaunch/understudy/workflow lanes) must apply the
    floor first but NEVER return empty-handed because of it: when no offer
    clears the default floor, an unfloored pass runs — a slow host under the
    boot SLA beats no host at all."""
    queries = []

    def fake_request_soft(method, path, q=None, **kw):
        queries.append(q)
        if q and "inet_down" in q:
            return True, {"offers": []}, None
        return True, {"offers": [{"id": 1, "min_bid": 0.2}]}, None

    monkeypatch.setattr(api, "request_soft", fake_request_soft)
    offer = market_offers.pick_cheapest_offer(gpu=("RTX 5090",))
    assert offer == {"id": 1, "min_bid": 0.2}
    assert queries[0]["inet_down"] == {"gte": 1000.0}
    assert "inet_down" not in queries[-1]


def test_pick_cheapest_offer_explicit_floor_stays_hard(monkeypatch):
    """An explicit inet_down is a real constraint: no unfloored fallback."""
    queries = []
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, q=None, **kw: (queries.append(q),
                                                    (True, {"offers": []}, None))[1])
    assert market_offers.pick_cheapest_offer(gpu=("RTX 5090",), inet_down=400) is None
    assert queries and all(q["inet_down"] == {"gte": 400.0} for q in queries)


def test_pick_cheapest_offer_any_inet_never_floors(monkeypatch):
    queries = []
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, q=None, **kw: (queries.append(q),
                                                    (True, {"offers": []}, None))[1])
    market_offers.pick_cheapest_offer(gpu=("RTX 5090",), any_inet=True)
    assert queries and all("inet_down" not in q for q in queries)


def test_search_offers_soft_falls_back_unfloored(monkeypatch, capsys):
    def fake_request_soft(method, path, q=None, **kw):
        if q and "inet_down" in q:
            return True, {"offers": []}, None
        return True, {"offers": [{"id": 7}]}, None

    monkeypatch.setattr(api, "request_soft", fake_request_soft)
    a = _search_ns(gpu=["RTX 5090"])
    assert market_offers._search_offers_soft(a) == [{"id": 7}]
    assert "inet-down floor" in capsys.readouterr().out
    assert a.inet_down is None            # the caller's namespace is untouched


# --------------------------------------------------------------------------- #
# escalating tolerance
#
# MIGRATED (was MIGRATION-BLOCKED, step 6e batch B3): `_boot_deadline_backoff`
# landed in `vastlib.supervise.replacement` — with the four SLA subjects that
# call it on the breach path, not in `core.config` as the raising stub guessed —
# so the deadline-crossing tests repoint with the milestone/disarm ones and the
# file has a single namespace again. Seam placement is by RESOLUTION: the SLA
# bodies read `health._jobd_status_line_soft`, `journal._sup_emit`,
# `lifecycle._destroy_soft`/`_destroy_and_revoke`, and their own module globals
# for `_serve_status_line_soft` / `_serve_sla_emit` / `_relaunch` /
# `_job_pull_condemn` / `_confirm_gone` (the last a forwarder over
# `lifecycle._confirm_gone`, patchable at either end by design).
# --------------------------------------------------------------------------- #
def test_boot_deadline_backoff_widens_after_max_kills():
    assert replacement._boot_deadline_backoff(600, 0) == 600.0
    assert replacement._boot_deadline_backoff(600, 1) == 600.0   # < BOOT_SLA_MAX_KILLS(2)
    assert replacement._boot_deadline_backoff(600, 2) == 1200.0  # widened, not flapping
    assert replacement._boot_deadline_backoff(600, 3) == 2400.0


# --------------------------------------------------------------------------- #
# jobs lane: env-setup SLA (running box, jobd never stamped)
# --------------------------------------------------------------------------- #
def _args(**kw):
    base = dict(id=41, dry_run=False, budget=None, max_bid=None,
                handoff=True, strict_ceiling=False, keep=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _jc(**kw):
    jc, _hf = job_lane.job_supervise_init(_args(**kw.pop("args", {})))
    jc.update(kw)
    return jc


def _inst(iid=41, status="running", *, age=100, machine=7, status_msg=""):
    return {"id": iid, "actual_status": status, "machine_id": machine,
            "start_date": NOW - age, "status_msg": status_msg,
            "num_gpus": 1, "gpu_name": "RTX 4090", "label": "jobs-wave"}


def test_jobs_sla_silent_under_deadline(monkeypatch):
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: None)
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA - 100), NOW) is None


def test_jobs_sla_condemns_past_deadline(monkeypatch):
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: None)
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 60), NOW) == "sla"


def test_jobs_sla_milestone_disarms_permanently(monkeypatch):
    """jobd stamping a NON-CONFESSING JOBD_STATUS is the came-online proof: the
    SLA is met and never re-checked (no more B2 reads) for this box."""
    reads = {"n": 0}
    monkeypatch.setattr(health, "_jobd_status_line_soft",
                        lambda iid: (reads.__setitem__("n", reads["n"] + 1),
                                     "RUNNING 1 pyhalf=ok")[1])
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None
    assert jc["boot_online_iid"] == "41"
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 9999), NOW) is None
    assert reads["n"] == 1


def test_jobs_sla_not_armed_without_observed_boot(monkeypatch):
    """A watch attached to an already-running (or resumed) box never saw it in
    `loading` — a stale JOBD_STATUS from the previous session would fake the
    milestone, so the SLA must not arm at all."""
    monkeypatch.setattr(health, "_jobd_status_line_soft",
        lambda iid: pytest.fail("unarmed SLA must not read JOBD_STATUS"))
    jc = _jc()
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None


def test_jobs_sla_backoff_uses_the_kill_counter(monkeypatch):
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: None)
    jc = _jc(boot_loading_iid="41", pull_relaunches=2)     # widened to 1200s
    assert replacement._job_boot_sla_tick(jc, _inst(age=700), NOW) is None
    assert replacement._job_boot_sla_tick(jc, _inst(age=1300), NOW) == "sla"


def test_jobs_sla_disabled_by_knob(monkeypatch):
    monkeypatch.setenv("BOOT_SLA_S", "0")
    monkeypatch.setattr(health, "_jobd_status_line_soft",
        lambda iid: pytest.fail("disabled SLA must not read JOBD_STATUS"))
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None


def test_tick_routes_a_jobs_sla_breach_into_the_reschedule(monkeypatch):
    """End-to-end through the real tick: a box observed booting, now `running`
    past the SLA with jobd never stamped, reaches _job_pull_condemn with the
    'sla' verdict (destroy + exclude + relaunch + retarget live there, pinned
    by test_pull_watchdog.py)."""
    boxes = {"cur": _inst(status="loading", age=60)}
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [boxes["cur"]])
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: None)
    monkeypatch.setattr(jobmeta, "list_queue", lambda box: ["j1"])
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda j, live_iids=None: {"job_id": j, "status": "submitted",
                                                   "display_status": "submitted"})
    hit = {}
    monkeypatch.setattr(replacement, "_job_pull_condemn",
                        lambda jc, inst, verdict: hit.update(verdict=verdict) or None)
    jc, hf = job_lane.job_supervise_init(_args())
    assert job_lane.job_supervise_tick(jc, hf) is None            # loading: arms the SLA
    assert jc["boot_loading_iid"] == "41"
    boxes["cur"] = _inst(status="running", age=SLA + 120)
    hit.clear()                                            # drop the loading-phase verdict
    assert job_lane.job_supervise_tick(jc, hf) is None
    # The env-setup budget is clocked from the RUNNING transition, so a box that
    # has merely EXISTED past the SLA is not yet in breach — it has just arrived.
    assert hit == {}, "a freshly-running box must get its own env-setup budget"
    assert jc["boot_running_iid"] == "41"
    # Age the env-setup phase itself past the SLA; now it breaches.
    jc["boot_running_since"] -= SLA + 1
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert hit == {"verdict": "sla"}


def test_a_slow_pull_does_not_eat_the_env_setup_budget(monkeypatch):
    """BOOT_PULL_TIMEOUT_S and BOOT_SLA_S are both 600s and were both clocked on
    start_date, so they shared ONE budget instead of granting one each: a box
    whose pull legally took 9 minutes had 60 seconds to bootstrap jobd. Box
    47166718 (2026-08-08) pulled the 7 GB t212 image in 8m59s -- inside the pull
    timeout -- and was condemned 82 seconds later for "running 10m"."""
    boxes = {"cur": _inst(status="loading", age=60)}
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [boxes["cur"]])
    monkeypatch.setattr(handoff, "_job_handoff_reconcile", lambda jc, hf: None)
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: None)
    monkeypatch.setattr(jobmeta, "list_queue", lambda box: ["j1"])
    monkeypatch.setattr(jobmeta, "read_job",
                        lambda j, live_iids=None: {"job_id": j, "status": "submitted",
                                                   "display_status": "submitted"})
    hit = {}
    monkeypatch.setattr(replacement, "_job_pull_condemn",
                        lambda jc, inst, verdict: hit.update(verdict=verdict) or None)
    jc, hf = job_lane.job_supervise_init(_args())
    job_lane.job_supervise_tick(jc, hf)                           # loading: arm
    # Pull finished at 9m -- legal -- so the box is running with 1m of box age
    # left. Under the old start_date anchor this fired 'sla' immediately.
    boxes["cur"] = _inst(status="running", age=SLA - 60)
    hit.clear()
    job_lane.job_supervise_tick(jc, hf)
    assert hit == {}, "a 9-minute pull must not consume the env-setup budget"


def test_condemn_sla_verdict_reschedules_like_a_pull_verdict(monkeypatch):
    jc = _jc(instances=[_inst()])
    calls = []
    monkeypatch.setattr(replacement, "_launch_job_replacement",
                        lambda jctx, excl, **kw: (
                            calls.append(("launch", excl, kw.get("max_dph"))),
                            (77, 0.5, None))[1])
    monkeypatch.setattr(replacement, "_retarget_pending_tickets",
                        lambda old, new: (calls.append(("retarget", old, new)),
                                          (["j1"], []))[1])
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
                        lambda ids, ins, intent, noun="": (
                            calls.append(("destroy", list(ids))), [])[1])
    emitted = []
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: emitted.append((event, kw)))
    assert replacement._job_pull_condemn(jc, _inst(age=SLA + 60), "sla") is None
    kinds = [c[0] for c in calls]
    assert kinds.index("retarget") < kinds.index("destroy")
    assert jc["iid"] == "77" and 7 in jc["pull_bad_machines"]
    assert ("pull_condemned" in [e for e, _ in emitted]
            and next(kw for e, kw in emitted
                     if e == "pull_condemned")["verdict"] == "sla")


# --------------------------------------------------------------------------- #
# jobs lane: the milestone is a stamp that is NOT CONFESSING (2026-08-14).
#
# Box 47737955 met the old milestone -- "a JOBD_STATUS stamp exists" -- at
# T+9m26s while `jobd.py` could not import its own modules, so it claimed
# nothing, emitted nothing, and billed $1.742 over 52 minutes. The bash half
# wrote the marker; THE SLA MEASURED THE HALF THAT WORKED.
# --------------------------------------------------------------------------- #
def _pyhalf_line(state="IDLE", field=""):
    return f"{state} 2026-08-13T04:00:00Z{field}"


def test_jobs_sla_milestone_withheld_from_a_confessing_stamp(monkeypatch):
    """Same reader, same box, same instant — the ONLY difference is what the
    line says about the python half, and that has to decide the milestone."""
    line = {"v": _pyhalf_line(field=" pyhalf=ok")}
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: line["v"])
    ok = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(ok, _inst(age=SLA - 100), NOW) is None
    assert ok["boot_online_iid"] == "41"

    line["v"] = _pyhalf_line(field=" pyhalf=broken")
    bad = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(bad, _inst(age=SLA - 100), NOW) is None
    assert "boot_online_iid" not in bad, \
        "a confessing stamp must not be recorded as coming online"


def test_jobs_sla_never_condemns_a_confessed_bundle_fault(monkeypatch, capsys):
    """PAST the deadline and still no condemn. The SLA's remedy is destroy +
    exclude the machine + relaunch elsewhere; `pyhalf=broken` is a shipped-
    bundle fault, host-independent by construction, so every replacement
    reproduces it. It says so once, journals once, and leaves the box to the
    two park paths that own this shape (jobd at 300s, fleetd at 600s)."""
    monkeypatch.setattr(health, "_jobd_status_line_soft",
                        lambda iid: _pyhalf_line(field=" pyhalf=broken"))
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 9999), NOW) is None
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 9999), NOW) is None
    out = capsys.readouterr().out
    assert out.count("BOOT-SLA HELD") == 1                 # said once, not per tick
    assert "bundle fault" in out and "relaunch reproduces it" in out
    evs = [e for e, _f in (jc.get("ladder_journal") or [])]
    assert evs == ["boot_sla_held_pyhalf_broken"]


def test_jobs_sla_milestone_accepts_a_bundle_with_no_pyhalf_field(monkeypatch):
    """BACK-COMPAT, and the reason the check is `is not True` and not `== ok`.
    Every box in the fleet on the day the field shipped emits no `pyhalf=` at
    all; a strict milestone would hold each of them to a signal they cannot
    send and then DESTROY AND RELAUNCH them at the deadline."""
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: _pyhalf_line())
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 9999), NOW) is None
    assert jc["boot_online_iid"] == "41"


def test_jobs_sla_milestone_accepts_pyhalf_ok(monkeypatch):
    monkeypatch.setattr(health, "_jobd_status_line_soft",
                        lambda iid: _pyhalf_line(state="RUNNING 1",
                                                 field=" pyhalf=ok"))
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 9999), NOW) is None
    assert jc["boot_online_iid"] == "41"


def test_jobs_sla_milestone_latches_when_a_broken_box_recovers(monkeypatch):
    """Withholding the milestone is not condemning it: if the box comes back
    with a healthy beacon the SLA is met THEN, on evidence."""
    line = {"v": _pyhalf_line(field=" pyhalf=broken")}
    monkeypatch.setattr(health, "_jobd_status_line_soft", lambda iid: line["v"])
    jc = _jc(boot_loading_iid="41")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 60), NOW) is None
    assert "boot_online_iid" not in jc
    line["v"] = _pyhalf_line(field=" pyhalf=ok")
    assert replacement._job_boot_sla_tick(jc, _inst(age=SLA + 120), NOW) is None
    assert jc["boot_online_iid"] == "41"


def test_fleet_watch_jobs_profile_runs_the_same_sla_ladder(monkeypatch):
    """THE LANE. FAILCLOSED_DESIGN §1 records the boot SLA as "not armed on
    this lane anyway -- BOOT_SLA_S is armed by `job supervise` / `supervise` /
    workflow, not by `fleet watch`, which is what the launcher uses". That is
    WRONG, measured here: fleetd's `jobs` profile ticks through
    `Hooks.jobs_tick` -> `herdd.job_supervise_tick`, the same function whose
    running branch calls `_job_boot_sla_tick` (pinned by
    test_tick_routes_a_jobs_sla_breach_into_the_reschedule above). One ladder,
    both drivers. launch_jobs_box.sh ends by registering exactly this watch, so
    an unattended box IS under the SLA.

    The residual gap is narrower and is NOT closed here: `jc` lives in fleetd's
    non-persisted `runtime` map, so a daemon restart between `loading` and the
    milestone loses `boot_loading_iid` and disarms the SLA for that box."""
    import inspect
    from vastlib.fleet import daemon as fleet_daemon
    from vastlib.fleet import hooks as fleet_hooks
    seen = {}
    monkeypatch.setattr(job_lane, "job_supervise_tick",
                        lambda jc, hf: seen.update(args=(jc, hf)))
    jc, hf = job_lane.job_supervise_init(_args())
    fleet_hooks.Hooks().jobs_tick(jc, hf)
    assert seen["args"] == (jc, hf)
    # ...and that a `jobs` watch reaches that hook at all.
    watch_src = inspect.getsource(fleet_daemon.Fleet._tick_watch)
    assert 'w["profile"] in ("run", "jobs", "serve")' in watch_src
    assert "self._tick_policy_watch(target, w, now, inst)" in watch_src
    assert "self.hooks.jobs_tick(jc, hf)" in \
        inspect.getsource(fleet_daemon.Fleet._tick_policy_watch)


# --------------------------------------------------------------------------- #
# serve lane: SLA + relaunch-spec re-fire
# --------------------------------------------------------------------------- #
def _serve_spec(tmp_path, monkeypatch, iid="41", **over):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    spec = {"script": "launch_serve.sh", "serve_id": "sv-1",
            "argv": ["--model", "b2:base-models/m", "--gpu", "5090"],
            "exclude_machines": [3], "sla_kills": 0}
    spec.update(over)
    d = tmp_path / "herdd" / "serve-relaunch"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{iid}.json").write_text(json.dumps(spec))
    return spec


def _serve_jc(**kw):
    jc, _hf = job_lane.job_supervise_init(_args(**dict({"id": 41, "budget": 5.0,
                                                 "serve_mode": True},
                                                **kw.pop("args", {}))))
    jc.update(kw)
    return jc


def test_serve_sla_condemns_a_box_stuck_on_the_launched_marker(
        tmp_path, monkeypatch):
    """The 46682177 shape: SERVE_STATUS still reads the workstation's LAUNCHED
    token past the deadline — the box never started onstart. This is the
    image-pull / container-standup phase: HOST-side, the one phase whose
    remedy is a different host."""
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("LAUNCHED", None, ""))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(status="loading", age=SLA + 60),
                                  NOW) == "sla"
    assert jc["boot_sla_phase"] == "image-pull"


def test_serve_sla_running_but_no_onstart_names_the_phase(
        tmp_path, monkeypatch):
    """46682177 showed `running` while its image was still pulling — astat
    alone cannot phase a serve boot, the marker can: LAUNCHED + running =
    pre-onstart, still host-side, still a kill."""
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("LAUNCHED", None, ""))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(status="running", age=SLA + 60),
                                  NOW) == "sla"
    assert jc["boot_sla_phase"].startswith("pre-onstart")


def test_serve_sla_ready_marker_is_the_milestone(tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("READY", NOW - 5, "m1"))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None
    assert jc["boot_online_iid"] == "41"


def test_serve_sla_b2_pull_stall_alarms_our_transfer_path_no_kill(
        tmp_path, monkeypatch, capsys):
    """A stalled `PULLING base` is a B2 transfer stall: could be OUR transfer
    path (rclone stream clamp vs b2x), so host rotation fixes nothing — the
    SLA must alarm loudly naming the phase + suspect and must NOT kill."""
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("PULLING", NOW - (SLA + 120), "base"))
    emitted = []
    monkeypatch.setattr(replacement, "_serve_sla_emit",
                        lambda sid, ev, **kw: emitted.append((ev, kw)))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None
    assert [e for e, _ in emitted] == ["boot_sla_phase_stall"]
    ev = emitted[0][1]
    assert ev["phase"] == "b2-pull (base)" and "not the host" in ev["suspect"]
    assert ev["elapsed_s"] >= SLA
    assert ev["status_msg_available"] is False        # null-status_msg telemetry
    assert "NOT killing" in capsys.readouterr().out
    # one-shot per sub-phase: a second tick does not re-alarm
    assert replacement._serve_boot_sla_tick(jc, _inst(age=SLA + 1999), NOW + 60) is None
    assert len(emitted) == 1


def test_serve_sla_onstart_provisioning_stall_blames_our_code(
        tmp_path, monkeypatch):
    """`PULLING boot` = onstart running, pre-B2-transfer: the 25-minute
    pre-pull window in the 46682177 timeline. Our code — alarm, no kill."""
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("PULLING", NOW - (SLA + 120), "boot"))
    emitted = []
    monkeypatch.setattr(replacement, "_serve_sla_emit",
                        lambda sid, ev, **kw: emitted.append((ev, kw)))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None
    assert emitted and emitted[0][1]["phase"] == "onstart-provisioning"


def test_serve_sla_pulling_under_deadline_is_silent(tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("PULLING", NOW - 60, "base"))
    monkeypatch.setattr(replacement, "_serve_sla_emit",
                        lambda sid, ev, **kw: pytest.fail("no stall, no alarm"))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None


def test_serve_sla_unreadable_marker_gives_no_verdict(tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: (None, None, None))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(age=SLA + 999), NOW) is None


def test_serve_sla_no_spec_disables_enforcement(tmp_path, monkeypatch):
    """No relaunch spec (pre-SLA launch, pinned launch, --on-box attach): the
    SLA cannot re-fire the serve, so it must never destroy it — disabled for
    this watch, marker never read."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))     # empty state dir
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
        lambda sid: pytest.fail("spec-less SLA must not read the marker"))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(status="loading", age=SLA + 999),
                                  NOW) is None
    assert jc["boot_sla_disabled"] is True


def test_serve_sla_backoff_reads_kills_from_the_spec(tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch, sla_kills=2)         # widened to 1200s
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("LAUNCHED", None, ""))
    jc = _serve_jc()
    assert replacement._serve_boot_sla_tick(jc, _inst(age=700), NOW) is None
    assert replacement._serve_boot_sla_tick(jc, _inst(age=1300), NOW) == "sla"


def _wire_serve_condemn(monkeypatch, *, destroy_fail=None, rc=0):
    calls = []
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
                        lambda ids, ins, intent, noun="": (
                            calls.append(("destroy", list(ids), intent)),
                            destroy_fail or [])[1])

    class _R:
        returncode = rc

    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: (calls.append(("run", list(cmd))),
                                           _R())[1])
    monkeypatch.setattr(replacement, "_serve_sla_emit",
                        lambda sid, event, **kw: calls.append(("emit", sid,
                                                               event, kw)))
    return calls


def test_serve_condemn_destroys_then_refires_with_exclusion(
        tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch)
    calls = _wire_serve_condemn(monkeypatch)
    jc = _serve_jc(instances=[_inst()])
    assert replacement._serve_boot_sla_condemn(jc, _inst(machine=7)) == "sla_relaunched"
    kinds = [c[0] for c in calls]
    assert kinds.index("destroy") < kinds.index("run")
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]
    cmd = [c for c in calls if c[0] == "run"][0][1]
    assert cmd[0] == "bash" and cmd[1].endswith("launch_serve.sh")
    assert "--model" in cmd and "b2:base-models/m" in cmd  # the saved argv
    assert cmd[cmd.index("--serve-id") + 1] == "sv-1"      # SAME serve identity
    assert cmd[cmd.index("--sla-kills") + 1] == "1"        # kill count carried
    ex = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--exclude-machine"]
    assert ex == ["3", "7"]           # prior exclusions + the machine just killed
    events = [c[2] for c in calls if c[0] == "emit"]
    assert "boot_sla_condemned" in events and "boot_sla_relaunched" in events
    cond = next(c[3] for c in calls if c[0] == "emit"
                and c[2] == "boot_sla_condemned")
    assert cond["phase"] and cond["suspect"] == "host"      # breach names its cause
    assert cond["status_msg_available"] is False    # m=127653: null the whole pull


def test_serve_condemn_kill_cap_disarms_and_keeps_the_box(
        tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch, sla_kills=MAX_RETRIES)
    calls = _wire_serve_condemn(monkeypatch)
    jc = _serve_jc(instances=[_inst()])
    assert replacement._serve_boot_sla_condemn(jc, _inst()) is None
    assert jc["boot_sla_disabled"] is True
    assert [c[0] for c in calls if c[0] in ("destroy", "run")] == []
    assert "boot_sla_exhausted" in [c[2] for c in calls if c[0] == "emit"]


def test_serve_condemn_budget_rail_blocks_the_relaunch(tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch)
    calls = _wire_serve_condemn(monkeypatch)
    jc = _serve_jc(instances=[_inst()], spend_usd=9.0)      # budget 5.0
    assert replacement._serve_boot_sla_condemn(jc, _inst()) is None
    assert [c[0] for c in calls if c[0] in ("destroy", "run")] == []


def test_serve_condemn_failed_destroy_never_relaunches(tmp_path, monkeypatch):
    """A live twin under the same serve:<ID> label would collide with the
    replacement's dup preflight — no destroy confirmation, no re-fire."""
    _serve_spec(tmp_path, monkeypatch)
    calls = _wire_serve_condemn(monkeypatch, destroy_fail=["41"])
    jc = _serve_jc(instances=[_inst()])
    assert replacement._serve_boot_sla_condemn(jc, _inst()) is None
    assert [c[0] for c in calls if c[0] == "run"] == []


def test_serve_condemn_failed_relaunch_is_loud_unrecoverable(
        tmp_path, monkeypatch):
    _serve_spec(tmp_path, monkeypatch)
    calls = _wire_serve_condemn(monkeypatch, rc=1)
    jc = _serve_jc(instances=[_inst()])
    assert replacement._serve_boot_sla_condemn(jc, _inst()) == "unrecoverable"
    assert "boot_sla_relaunch_failed" in [c[2] for c in calls if c[0] == "emit"]


def test_serve_condemn_dry_run_touches_nothing(tmp_path, monkeypatch, capsys):
    _serve_spec(tmp_path, monkeypatch)
    calls = _wire_serve_condemn(monkeypatch)
    jc = _serve_jc(instances=[_inst()], args={"dry_run": True})
    assert replacement._serve_boot_sla_condemn(jc, _inst()) is None
    assert [c[0] for c in calls if c[0] in ("destroy", "run")] == []
    assert "[dry-run]" in capsys.readouterr().out


def test_tick_routes_a_serve_sla_breach(tmp_path, monkeypatch):
    """serve_mode tick wiring: a loading serve box past the SLA reaches
    _serve_boot_sla_condemn (previously the serve lane was exempt from every
    boot watchdog because its launch shape was not reconstructible)."""
    _serve_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(replacement, "_serve_status_line_soft",
                        lambda sid: ("LAUNCHED", None, ""))
    box = _inst(status="loading", age=SLA + 120)
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [box])
    hit = {}
    monkeypatch.setattr(replacement, "_serve_boot_sla_condemn",
                        lambda jc, inst: hit.update(iid=jc["iid"]) or
                        "sla_relaunched")
    jc, hf = job_lane.job_supervise_init(_args(id=41, budget=5.0, serve_mode=True))
    assert job_lane.job_supervise_tick(jc, hf) == "sla_relaunched"
    assert hit == {"iid": "41"}


# --------------------------------------------------------------------------- #
# run lane: supervise boot SLA (milestone = the loading->running flip)
# --------------------------------------------------------------------------- #
def _run_args(**kw):
    d = dict(boot_sla=True, dry_run=False, machine=None, exclude_machines=None)
    d.update(kw)
    return argparse.Namespace(**d)


def _run_state(**kw):
    st = bidpolicy.mk_poll_state(present=True, actual_status="loading",
                         relaunch_count=0, max_relaunch=3)
    st.update({"run_id": "run-sla", "instance_id": "inst-1",
               "machine_id": 140087, "boot_sampler": None,
               "boot_sampler_iid": None, "excluded_machines": []})
    st.update(kw)
    return st


def _run_wire(monkeypatch):
    calls = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **f: calls.append(("emit", ev, f)))
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, dry_run=False: (calls.append(("destroy", iid)),
                                                    (True, None))[1])
    monkeypatch.setattr(replacement, "_confirm_gone", lambda iid: True)
    monkeypatch.setattr(replacement, "_relaunch",
                        lambda st, a: (calls.append(
                            ("relaunch", list(getattr(a, "exclude_machines",
                                                      None) or []))),
                            "relaunched")[1])
    return calls


def test_run_sla_silent_under_deadline(monkeypatch):
    calls = _run_wire(monkeypatch)
    st, a = _run_state(), _run_args()
    inst = {"start_date": NOW - 100, "machine_id": 140087}
    assert replacement._supervise_boot_sla(st, a, get_instance=lambda i: inst,
                                 now=lambda: NOW) is None
    assert calls == []
    assert st["boot_sla_armed_iid"] == "inst-1"     # armed while pre-running


def test_run_sla_breach_destroys_excludes_and_relaunches(monkeypatch):
    calls = _run_wire(monkeypatch)
    st, a = _run_state(), _run_args()
    inst = {"start_date": NOW - (SLA + 60), "machine_id": 140087}
    st["boot_sla_armed_iid"] = "inst-1"
    res = replacement._supervise_boot_sla(st, a, get_instance=lambda i: inst,
                                now=lambda: NOW)
    assert res == "condemned"
    kinds = [c[0] for c in calls]
    assert kinds == ["emit", "destroy", "relaunch"]
    assert calls[0][1] == "boot_sla_condemned"
    assert 140087 in st["excluded_machines"]
    assert calls[2][1] == [140087]                  # exclusion reached relaunch
    assert st["boot_sla_kills"] == 1


def test_run_sla_milestone_resets_the_kill_counter(monkeypatch):
    calls = _run_wire(monkeypatch)
    st = _run_state(actual_status="running", boot_sla_armed_iid="inst-1",
                    boot_sla_kills=1)
    assert replacement._supervise_boot_sla(st, _run_args()) is None
    assert st["boot_sla_armed_iid"] is None and st["boot_sla_kills"] == 0
    assert calls == []


def test_run_sla_opt_out_flag_and_knob(monkeypatch):
    monkeypatch.setattr(health, "_get_instance_soft",
        lambda iid: pytest.fail("disabled SLA must not poll"))
    st = _run_state()
    assert replacement._supervise_boot_sla(st, _run_args(boot_sla=False)) is None
    monkeypatch.setenv("BOOT_SLA_S", "0")
    assert replacement._supervise_boot_sla(st, _run_args()) is None


def test_run_sla_max_relaunch_guard(monkeypatch):
    calls = _run_wire(monkeypatch)
    st, a = _run_state(relaunch_count=3, max_relaunch=3), _run_args()
    inst = {"start_date": NOW - (SLA + 60), "machine_id": 140087}
    res = replacement._supervise_boot_sla(st, a, get_instance=lambda i: inst,
                                now=lambda: NOW)
    assert res == "stop_fatal"
    assert [c[0] for c in calls if c[0] in ("destroy", "relaunch")] == []


def test_run_sla_no_start_date_no_verdict(monkeypatch):
    calls = _run_wire(monkeypatch)
    st, a = _run_state(), _run_args()
    assert replacement._supervise_boot_sla(st, a, get_instance=lambda i: {"machine_id": 1},
                                 now=lambda: NOW) is None
    assert calls == []
