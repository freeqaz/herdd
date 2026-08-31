"""Portable tests for herdd.py's `supervise` machinery — see SUPERVISE_DESIGN.md
§8 ("Test plan") for the taxonomy these are built from.

Runs in the toolchain-free lane (`pytest -m "not integration"`): NO real vast
API, NO B2/rclone, NO network. Three layers:

  1. Pure `poll(state) -> Action` table tests — one case per §2 ladder row plus
     precedence proofs. Hand-built states via `mk_poll_state`.
  2. `_classify_http` unit tests, plus a couple of tests that drive the REAL
     `request_soft()` against a fake `urllib.request.urlopen` to prove the
     transient-retry-backoff / fatal-no-retry behavior actually implemented
     there (not just re-asserted at the driver level).
  3. `cmd_supervise` driver tests: `request_soft`, `_read_run_soft`,
     `_last_stopping_actor`, `_status_marker_soft`, `_ensure_b2_remote`, and
     `runmeta.emit_event` are all replaced with deterministic fakes; `time.time`/
     `time.sleep` are replaced with a fake clock so the loop never actually
     blocks. `cli.main.main()` is driven via `sys.argv` so the real argparse
     wiring (including the `--budget` requirement) is exercised end to end.
     Every row drives `cli.main.main()` since step 6d — the flat arm went with
     the fat `herdd.py` (see the Harness).
"""
import argparse
import base64
import copy
import io
import json
import os
import socket
import sys
import time
import types
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bidpolicy  # noqa: E402 (Zone S: the poll/handoff decision core)
import imageref  # noqa: E402 (Zone S: image digest/login helpers)
import jobmeta  # noqa: E402 (Zone S: the shared module object vastlib imports)
import runmeta as rm  # noqa: E402 (same module object vastlib imports)
import herdd  # noqa: E402,F401 (the thin launcher: imported so this file
                # still proves it IMPORTS — the re-export surface every other
                # flat consumer reaches is only as alive as that)
from vastlib.boxes import health as boxes_health  # noqa: E402
from vastlib.boxes import lifecycle  # noqa: E402
from vastlib.boxes import ssh as ssh_mod  # noqa: E402
from vastlib.cli import main as cli_main  # noqa: E402
from vastlib.cli import supervise as cli_supervise  # noqa: E402
from vastlib.cli import train as cli_train  # noqa: E402
from vastlib.cli.job import supervise as cli_job_supervise  # noqa: E402
from vastlib.core import api, config, fmt, labels, models  # noqa: E402
from vastlib.fleet import client as fleet_client  # noqa: E402
from vastlib.jobs import control as jobs_control  # noqa: E402
from vastlib.jobs import risk as jobs_risk  # noqa: E402
from vastlib.jobs import view as jobs_view  # noqa: E402
from vastlib.launch import launch as launch_mod  # noqa: E402
from vastlib.launch import spec as launch_spec  # noqa: E402
from vastlib.market import offers, pricing  # noqa: E402
from vastlib.storage import b2  # noqa: E402
from vastlib.supervise import handoff, job_lane, journal  # noqa: E402
from vastlib.supervise import replacement, retention, run_lane  # noqa: E402

mk = bidpolicy.mk_poll_state
poll = bidpolicy.poll
Action = bidpolicy.Action


# =============================================================================
# 1. Pure poll() ladder — one test per §2 row, plus precedence proofs
# =============================================================================
def test_poll_running_live_is_noop():
    s = mk(view={"status": "running"}, present=True, actual_status="running")
    assert poll(s) == Action("noop", "live")


def test_poll_terminal_done_event_stops():
    s = mk(view={"status": "done"}, present=True, actual_status="running")
    assert poll(s) == Action("stop_terminal", "terminal:done")


def test_poll_terminal_failed_event_stops():
    s = mk(view={"status": "failed"}, present=True, actual_status="running")
    assert poll(s) == Action("stop_terminal", "terminal:failed")


def test_poll_terminal_fires_even_while_box_still_ssh_alive():
    # crash debug-hold: box is live but the event log says failed -> terminal wins
    s = mk(view={"status": "failed"}, present=True, actual_status="running")
    a = poll(s)
    assert a.kind == "stop_terminal" and a.reason == "terminal:failed"


def test_poll_done_inferred_I4_when_gone_and_status_marker_done():
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           status_marker="DONE 20260709T000000000Z")
    assert poll(s) == Action("stop_terminal", "terminal:done")


def test_poll_stale_done_marker_ignored_while_box_is_live():
    # I4 only infers a terminal when the instance is GONE; a live box's stale
    # STATUS=DONE must never short-circuit an in-progress run
    s = mk(view={"status": "running"}, present=True, actual_status="running",
           status_marker="DONE 20260709T000000000Z")
    assert poll(s) == Action("noop", "live")


def test_poll_operator_stop_intended_status():
    s = mk(view={"status": "running"}, present=True, actual_status="running",
           intended_status="stopped")
    assert poll(s) == Action("stop_terminal", "operator_stop")


def test_poll_operator_stop_suppressed_while_handoff_fenced():
    # D1 (live canary handoff-canary-2, 2026-07-15): the handoff fence parks the
    # primary itself — intended_status=stopped during an open fence is OUR park,
    # not operator intent. Row 2a must not exit the supervisor mid-cutover.
    s = mk(view={"status": "running"}, present=True, actual_status="stopped",
           intended_status="stopped", handoff_fenced=True)
    a = poll(s)
    assert a != Action("stop_terminal", "operator_stop")
    assert a == Action("noop", "debounce_not_live")    # falls to the debounce row


def test_poll_operator_destroy_suppressed_while_handoff_fenced():
    # DRAINING destroys the primary via this same CLI; row 2b reading that as
    # operator_destroy would exit on the drain->complete tick.
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           stopping_actor="cli:laptop", handoff_fenced=True)
    a = poll(s)
    assert a != Action("stop_terminal", "operator_destroy")


def test_poll_operator_stop_still_fires_when_not_fenced():
    # the suppression is fence-scoped only: default state keeps rows 2a/2b.
    s = mk(view={"status": "running"}, present=True, actual_status="running",
           intended_status="stopped", handoff_fenced=False)
    assert poll(s) == Action("stop_terminal", "operator_stop")


def test_poll_operator_destroy_gone_plus_cli_actor():
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           stopping_actor="cli:laptop")
    assert poll(s) == Action("stop_terminal", "operator_destroy")


def test_poll_operator_destroy_not_triggered_by_non_cli_actor():
    # a `stopping` event from the box itself (or supervisor) is NOT operator intent
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           stopping_actor="box:123", not_live_streak=2)
    a = poll(s)
    assert a.kind == "emit_evicted"          # falls through to normal eviction


def test_poll_not_live_first_poll_is_debounced():
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           not_live_streak=1)
    assert poll(s) == Action("noop", "debounce_not_live")


def test_poll_debounce_precedes_guardrail():
    # a single not-live blip must noop even if a guardrail is already breached
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           not_live_streak=1, relaunch_count=5, max_relaunch=3)
    assert poll(s) == Action("noop", "debounce_not_live")


def test_poll_outbid_confirmed_present_but_not_live():
    s = mk(view={"status": "running"}, present=True, actual_status="stopped",
           not_live_streak=2)
    assert poll(s) == Action("emit_evicted", "outbid")


def test_poll_host_death_gone_and_confirmed():
    s = mk(view={"status": "running"}, present=False, actual_status=None,
           not_live_streak=2)
    assert poll(s) == Action("emit_evicted", "host_death")


def test_poll_evicted_reason_prefers_box_reported_reason():
    s = mk(view={"status": "running"}, present=True, actual_status="stopped",
           not_live_streak=2, stopping_reason="max_hours_exceeded")
    assert poll(s) == Action("emit_evicted", "max_hours_exceeded")


def test_poll_evicted_recorded_backoff_pending_is_noop():
    s = mk(view={"status": "evicted"}, present=False, actual_status=None,
           not_live_streak=2, backoff_ready=False)
    assert poll(s) == Action("noop", "backoff")


def test_poll_evicted_recorded_backoff_ready_relaunches():
    s = mk(view={"status": "evicted"}, present=False, actual_status=None,
           not_live_streak=2, backoff_ready=True)
    assert poll(s) == Action("relaunch", "resume_after_evicted")


def test_poll_guardrail_max_relaunch():
    s = mk(view={"status": "evicted"}, present=False, actual_status=None,
           not_live_streak=2, backoff_ready=True,
           relaunch_count=3, max_relaunch=3)
    assert poll(s) == Action("stop_budget", "max_relaunch")


def test_poll_guardrail_budget():
    s = mk(view={"status": "evicted"}, present=False, actual_status=None,
           not_live_streak=2, backoff_ready=True,
           spend_usd=10.5, budget_usd=10.0)
    assert poll(s) == Action("stop_budget", "budget")


def test_poll_guardrail_wall_budget():
    s = mk(view={"status": "evicted"}, present=False, actual_status=None,
           not_live_streak=2, backoff_ready=True,
           wall_clock_s=200_000, wall_budget_s=172_800)
    assert poll(s) == Action("stop_budget", "wall_budget")


def test_poll_precedence_terminal_beats_operator_stop():
    s = mk(view={"status": "done"}, present=True, actual_status="running",
           intended_status="stopped")
    assert poll(s) == Action("stop_terminal", "terminal:done")


def test_poll_precedence_operator_intent_beats_live():
    s = mk(view={"status": "running"}, present=True, actual_status="running",
           intended_status="stopped")
    assert poll(s) == Action("stop_terminal", "operator_stop")


def test_poll_precedence_guardrail_beats_relaunch():
    s = mk(view={"status": "evicted"}, present=False, actual_status=None,
           not_live_streak=2, backoff_ready=True,
           relaunch_count=3, max_relaunch=3)
    a = poll(s)
    assert a.kind == "stop_budget" and a.reason == "max_relaunch"


def test_poll_is_pure_no_mutation():
    s = mk(view={"status": "running", "nested": {"x": 1}}, present=True,
           actual_status="running", not_live_streak=0)
    before = copy.deepcopy(s)
    poll(s)
    assert s == before


# =============================================================================
# 1b. Pure poll() bid-movement ladder (SPOT_DESIGN §3.2): raise_bid / rescue_bid
# =============================================================================
def _live_bid(**over):
    base = dict(view={"status": "running"}, present=True, actual_status="running",
                last_bid=0.60, max_bid=0.75, market_min_bid=None,
                last_bid_put_ts=0.0, now=1_000_000.0)
    base.update(over)
    return mk(**base)


def _outbid(**over):
    base = dict(view={"status": "running"}, present=True, actual_status="stopped",
                not_live_streak=2, last_bid=0.10, max_bid=0.75,
                market_min_bid=0.20, last_bid_put_ts=0.0, now=1_000_000.0)
    base.update(over)
    return mk(**base)


def test_poll_pressure_raises_bid_on_live_box():
    # market climbed to within 10% of our standing bid -> proactive defend
    s = _live_bid(market_min_bid=0.55)               # 0.55 >= 0.9*0.60 = 0.54
    a = poll(s)
    assert a.kind == "raise_bid"
    assert a.reason == "defend:0.66"                 # min(1.2*0.55=0.66, max 0.75)


def test_poll_no_pressure_is_noop_live():
    assert poll(_live_bid(market_min_bid=0.30)) == Action("noop", "live")


def test_poll_raise_skipped_at_max_bid():
    # already at the cap -> the raise would be < 1 cent -> skip (never exceed max_bid)
    s = _live_bid(last_bid=0.75, max_bid=0.75, market_min_bid=0.70)
    assert poll(s) == Action("noop", "live")


def test_poll_raise_rate_limited_skips():
    # last bid PUT < 60s ago -> the rate-limit guard holds even under pressure
    s = _live_bid(market_min_bid=0.55, now=1000.0, last_bid_put_ts=970.0)
    assert poll(s) == Action("noop", "live")


def test_poll_raise_disabled_when_market_read_failed():
    assert poll(_live_bid(market_min_bid=None)) == Action("noop", "live")


def test_poll_raise_disabled_when_standing_bid_unknown():
    assert poll(_live_bid(market_min_bid=0.55, last_bid=None)) == Action("noop", "live")


def test_poll_outbid_rescues_bid_before_relaunch():
    a = poll(_outbid())
    assert a.kind == "rescue_bid"
    assert a.reason == "rescue:0.24"                 # min(1.2*0.20=0.24, 0.75)


def test_poll_rescue_uncapped_legacy_caps_at_1p2x_market():
    # max_bid None (legacy) still yields a finite target: 1.2x the market min
    a = poll(_outbid(max_bid=None))
    assert a == Action("rescue_bid", "rescue:0.24")


def test_poll_rescue_suppressed_after_attempt_falls_to_evicted():
    # a rescue already fired this cycle -> no re-rescue; normal eviction resumes
    assert poll(_outbid(rescue_attempted=True)) == Action("emit_evicted", "outbid")


def test_poll_rescue_disabled_when_no_market_read_falls_to_evicted():
    # offers-read failure -> market_min_bid None -> bid actions disabled, eviction
    # logic unaffected
    assert poll(_outbid(market_min_bid=None)) == Action("emit_evicted", "outbid")


def test_poll_rescue_not_affordable_under_max_bid_falls_to_evicted():
    # target min(1.2*mmb, max_bid) not above our bid -> relaunch instead of rescue
    s = _outbid(last_bid=0.55, max_bid=0.55, market_min_bid=0.40)   # target 0.48
    assert poll(s) == Action("emit_evicted", "outbid")


def test_poll_rescue_rate_limited_skips_to_evicted():
    s = _outbid(now=1000.0, last_bid_put_ts=980.0)
    assert poll(s) == Action("emit_evicted", "outbid")


def test_poll_spend_cap_beats_raise_bid():
    # money-moving bid actions run AFTER the spend/wall caps (invariant §5.3)
    s = _live_bid(market_min_bid=0.55, spend_usd=10.0, budget_usd=5.0)
    assert poll(s) == Action("stop_budget", "budget")


def test_poll_guardrail_beats_rescue_bid():
    s = _outbid(relaunch_count=3, max_relaunch=3)
    assert poll(s) == Action("stop_budget", "max_relaunch")


def test_poll_bid_path_is_pure_no_mutation():
    for s in (_live_bid(market_min_bid=0.55), _outbid()):
        before = copy.deepcopy(s)
        poll(s)
        assert s == before


# =============================================================================
# 2a. _classify_http unit tests
# =============================================================================
@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 599])
def test_classify_http_transient_int_codes(code):
    assert api._classify_http(code) == "transient"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_classify_http_fatal_int_codes(code):
    assert api._classify_http(code) == "fatal"


def test_classify_http_httperror_instance():
    transient = urllib.error.HTTPError("u", 503, "unavail", None, io.BytesIO(b""))
    fatal = urllib.error.HTTPError("u", 404, "missing", None, io.BytesIO(b""))
    assert api._classify_http(transient) == "transient"
    assert api._classify_http(fatal) == "fatal"


def test_classify_http_network_and_timeout_are_transient():
    assert api._classify_http(urllib.error.URLError("boom")) == "transient"
    assert api._classify_http(socket.timeout()) == "transient"
    assert api._classify_http(TimeoutError()) == "transient"


def test_classify_http_config_missing_key_is_fatal():
    assert api._classify_http("config: VASTAI_API_KEY not set") == "fatal"


def test_classify_http_string_forms():
    assert api._classify_http("HTTP 429 on GET x: rate limited") == "transient"
    assert api._classify_http("HTTP 401 on GET x: nope") == "fatal"
    assert api._classify_http("network ConnectionResetError on GET x") == "transient"
    assert api._classify_http("error timed out on GET x") == "transient"


def test_classify_http_bool_never_treated_as_an_http_code():
    # bool is an int subclass in Python; the guard must catch it explicitly
    assert api._classify_http(True) == "transient"
    assert api._classify_http(False) == "transient"


def test_classify_http_unknown_defaults_transient():
    # "safest: retry-then-degrade, never a false fatal exit"
    assert api._classify_http("something weird") == "transient"
    assert api._classify_http(None) == "transient"
    assert api._classify_http(object()) == "transient"


# =============================================================================
# 2b. request_soft() itself — real retry/backoff/fatal-stop against a fake
#     urlopen (no network).
# =============================================================================
class _FakeResp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_request_soft_transient_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = {"n": 0}
    sleeps = []

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.HTTPError(req.full_url, 503, "unavailable",
                                         None, io.BytesIO(b""))
        return _FakeResp(b'{"instances": []}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, data, err = api.request_soft("GET", "v1/instances/", retries=5,
                                         _sleep=lambda s: sleeps.append(s))
    assert ok is True
    assert data == {"instances": []}
    assert err is None
    assert calls["n"] == 3           # 2 failures + 1 success
    assert len(sleeps) == 2          # backed off before each retry, not after success
    assert all(s >= 0 for s in sleeps)


def test_request_soft_fatal_stops_immediately_no_retry(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized",
                                     None, io.BytesIO(b'{"msg":"bad key"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def must_not_sleep(s):
        pytest.fail("request_soft must not back off on a fatal (401)")

    ok, data, err = api.request_soft("GET", "v1/instances/", retries=5,
                                         _sleep=must_not_sleep)
    assert ok is False
    assert data is None
    assert "HTTP 401" in err
    assert calls["n"] == 1            # no retries for a fatal


def test_request_soft_404_is_also_fatal(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "not found",
                                     None, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, data, err = api.request_soft("GET", "v1/instances/", retries=5,
                                         _sleep=lambda s: pytest.fail("no retry on 404"))
    assert ok is False and "HTTP 404" in err


def test_request_soft_missing_key_is_fatal_and_never_touches_network(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    called = {"n": 0}

    def fake_urlopen(req, timeout=None):
        called["n"] += 1
        raise AssertionError("must never hit the network without an API key")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, data, err = api.request_soft("GET", "v1/instances/")
    assert ok is False
    assert called["n"] == 0
    assert err.startswith("config:")
    assert api._classify_http(err) == "fatal"


# =============================================================================
# 3. cmd_supervise driver tests — fully faked I/O, fake clock, real argparse
#
# MIGRATED (was MIGRATION-BLOCKED, step 6e), whole section through §3b-bis: the
# five run-lane drivers (`_observe`, `_accrue_cost`, `_emit_cost`,
# `_do_bid_move`, `_supervise_boot_health`) are no longer raising stubs — the
# integrator ruling landed their bodies in `vastlib.supervise.replacement` and
# left `run_lane` one CALL-TIME forwarder each, so both patch surfaces steer and
# the tests below drive the REAL `_observe` beneath the API fake, exactly as they
# did flat. Subject and seams moved together.
# =============================================================================
class FakeAPI:
    """Scripted stand-in for `api.request_soft`, dispatched by (method, path)."""

    def __init__(self):
        self.instances_queue = []                    # list of (ok, data, err)
        self.instances_default = (True, {"instances": []}, None)
        self.offers = (True, {"offers": [{"id": 9001, "min_bid": 0.10}]}, None)
        self.bid = (True, {"success": True}, None)    # PUT bid_price; override to 429
        self.launch = (True, {"success": True, "new_contract": 42424242}, None)
        self.confirm_gone = (True, {"instances": None}, None)
        self.destroy = (True, {}, None)
        self.stop_put = (True, {}, None)             # override to fail the park
        self.stopped_iids = set()                    # iids parked via PUT stopped
        self.destroyed_iids = set()                  # iids DELETEd (report gone after)
        self.relabels = {}                           # iid(str) -> label from a PUT {label}
        self.calls = []

    def _process_v1(self, resp):
        """Reflect two-instance lifecycle in the v1 listing so the driver observes
        the world it actually mutated: a DELETEd instance is gone, a relabelled
        instance (the handoff cutover PUTs run:<ID>:handoff -> run:<ID>) carries
        its new label, and a PARKED instance reports actual/intended stopped —
        exactly what the real API shows after the fence's own park. That last
        reflection is load-bearing: without it the harness hid D1 (the fence park
        tripping poll()'s operator_stop row and exiting the supervisor 49s after
        its own fence — live canary handoff-canary-2, 2026-07-15); with it, EVERY
        fence scenario in this file is a D1 regression test."""
        if not (self.destroyed_iids or self.relabels or self.stopped_iids):
            return resp
        ok, data, err = resp
        if not (ok and isinstance(data, dict) and isinstance(data.get("instances"), list)):
            return resp
        out = []
        for inst in data["instances"]:
            iid = str(inst.get("id"))
            if iid in self.destroyed_iids:
                continue
            if iid in self.relabels:
                inst = {**inst, "label": self.relabels[iid]}
            if iid in self.stopped_iids:
                inst = {**inst, "actual_status": "stopped",
                        "intended_status": "stopped"}
            out.append(inst)
        return (ok, {**data, "instances": out}, err)

    def __call__(self, method, path, body=None, timeout=60, retries=5, _sleep=None):
        self.calls.append((method, path))
        if method == "GET" and path == "v1/instances/":
            resp = self.instances_queue.pop(0) if self.instances_queue \
                else self.instances_default
            return self._process_v1(resp)
        if method == "POST" and path == "v0/bundles/":
            return self.offers
        if method == "PUT" and "bid_price" in path:
            return self.bid
        if method == "PUT" and path.startswith("v0/asks/"):
            return self.launch
        if method == "PUT" and path.startswith("v0/instances/") \
                and isinstance(body, dict) and body.get("state") == "stopped":
            # park request: on success, later GETs on this iid report stopped
            ok, d, err = self.stop_put
            if ok:
                self.stopped_iids.add(path.rstrip("/").rsplit("/", 1)[-1])
            return self.stop_put
        if method == "PUT" and path.startswith("v0/instances/") \
                and isinstance(body, dict) and "label" in body:
            # handoff cutover relabel run:<ID>:handoff -> run:<ID> (reflected in v1)
            self.relabels[path.rstrip("/").rsplit("/", 1)[-1]] = body["label"]
            return (True, {}, None)
        if method == "DELETE" and path.startswith("v0/instances/"):
            self.destroyed_iids.add(path.rstrip("/").rsplit("/", 1)[-1])
            return self.destroy
        if method == "GET" and path.startswith("v0/instances/"):
            iid = path.rstrip("/").rsplit("/", 1)[-1]
            if iid in self.destroyed_iids:            # DELETE wins over a prior park
                return self.confirm_gone
            if iid in self.stopped_iids:
                return (True, {"instances": {"actual_status": "stopped"}}, None)
            return self.confirm_gone
        return (True, {}, None)


class FakeClock:
    """Deterministic time.time()/time.sleep() so the loop never actually blocks,
    plus a hard cap so a mis-scripted test fails fast instead of hanging."""

    def __init__(self, start=1_700_000_000.0, max_sleeps=500):
        self.t = start
        self.sleeps = []
        self.max_sleeps = max_sleeps

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        if len(self.sleeps) > self.max_sleeps:
            raise RuntimeError(
                f"supervise loop exceeded {self.max_sleeps} sleep() calls — "
                "a scripted scenario never reached a stop condition")
        self.t += s


class Harness:
    """Wires the `supervise` subcommand to fully faked I/O: no
    vast API, no B2/rclone, no real clock. `run_id` gets a fresh isolated
    XDG_CACHE_HOME per test (tmp_path) so launch-spec capture never touches a
    real ~/.cache.

    MIGRATED (step 6e): subject is `vastlib.cli.main.main()` and every seam is
    stubbed at the module the run lane RESOLVES it through — `api.request_soft`,
    `b2._ensure_b2_remote`, `spec._status_marker_soft` / `_last_stopping_actor` /
    `_read_run_soft`, and `config.load_env` (read inside `cli.main`'s parser
    builder, which is where the flat prologue's `.env` load went).

    What unblocked it: `run_lane._observe/_accrue_cost/_emit_cost/_do_bid_move/
    _supervise_boot_health` were raising stubs, and these tests exercise the REAL
    `_observe` beneath the API fake — they cannot stub it without changing what
    they test. The bodies landed in `supervise.replacement` and `run_lane` now
    carries a call-time forwarder for each, so the tick reaches the real code.

    THE `flat=True` ARM IS GONE (step 6d). Seven scenarios — the ones that cross
    the RELAUNCH path — used to drive `herdd.main()` with this seam set
    resolved through `herdd.__dict__`, because `replacement._relaunch` calls
    `_reset_run_markers` and that name was a raising seam with no vastlib home.
    Two things ended that: the body was MOVED into
    `supervise.replacement._reset_run_markers` (home ruling on the def there —
    `storage.b2` refuses path policy, and its twin `_handoff_b2_write` lives in
    `supervise.handoff` for the same reason), and `herdd.py` became a thin
    launcher whose `main` IS `cli.main.main`. Keeping the flat arm after that
    would have been worse than useless: every `setattr(herdd, …)` above became
    a rebind of a re-export nothing reads, so the harness would have driven the
    REAL api/rclone/clock while looking green. One arm now, on the owners."""

    def __init__(self, monkeypatch, tmp_path, run_id, max_sleeps=500):
        self.run_id = run_id
        self.api = FakeAPI()
        self.clock = FakeClock(max_sleeps=max_sleeps)
        self.events = []
        self.view_queue = []
        self.view_default = {"status": "unknown", "n_events": 0}
        self.stopping_actor = None

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.delenv("B2_BUCKET", raising=False)
        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        monkeypatch.setattr(config, "load_env", lambda: None)
        monkeypatch.setattr(api, "request_soft", self.api)
        monkeypatch.setattr(b2, "_ensure_b2_remote", lambda: None)
        monkeypatch.setattr(launch_spec, "_status_marker_soft", lambda run_id: None)
        monkeypatch.setattr(launch_spec, "_last_stopping_actor",
                            lambda run_id: self.stopping_actor)
        monkeypatch.setattr(launch_spec, "_read_run_soft", self._fake_read_run_soft)
        monkeypatch.setattr(rm, "emit_event", self._fake_emit)
        monkeypatch.setattr(time, "time", self.clock.time)
        monkeypatch.setattr(time, "sleep", self.clock.sleep)
        self._monkeypatch = monkeypatch

    def _fake_read_run_soft(self, run_id, live_iids=()):
        if self.view_queue:
            return dict(self.view_queue.pop(0))
        return dict(self.view_default)

    def _fake_emit(self, run_id, event, *, actor=None, runner=None,
                   bucket=None, **fields):
        rec = {"run_id": run_id, "event": event, "actor": actor, **fields}
        self.events.append(rec)
        return {**rec, "_emitted": True}

    def event_kinds(self):
        return [e["event"] for e in self.events]

    def run(self, extra_argv):
        # handoff is the parser DEFAULT (flipped 2026-07-15) and runs a startup
        # _handoff_reconcile GET that would consume one scripted instances_queue
        # entry. These core mechanics tests isolate the bid/poll/relaunch layer,
        # so default them to --no-handoff unless the test explicitly opts into a
        # ceiling flag (the handoff scenario tests pass --handoff and script the
        # extra reconcile GET themselves).
        ceiling = {"--handoff", "--no-handoff", "--strict-ceiling"}
        pre = [] if ceiling.intersection(extra_argv) else ["--no-handoff"]
        argv = ["herdd", "supervise", self.run_id, *pre, *extra_argv]
        self._monkeypatch.setattr(sys, "argv", argv)
        cli_main.main()


GONE = (True, {"instances": []}, None)


def _listed(run_id, iid, **overrides):
    inst = {"id": iid, "label": f"run:{run_id}", "actual_status": "running",
            "dph_total": 0.3}
    inst.update(overrides)
    return (True, {"instances": [inst]}, None)


def test_supervise_operator_stop_exits_immediately_no_relaunch(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-opstop")
    h.api.instances_queue = [_listed("run-opstop", 501, intended_status="stopped")]
    h.view_queue = [{"status": "running"}]
    h.run(["--budget", "5"])

    kinds = h.event_kinds()
    assert kinds[0] == "supervisor_started"
    assert kinds[1] == "supervised"
    assert "heartbeat" in kinds
    assert kinds[-1] == "supervisor_exiting"
    assert h.events[-1]["reason"] == "operator_stop"
    assert "evicted" not in kinds and "relaunched" not in kinds


def test_supervise_operator_destroy_exits_no_relaunch(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-opdestroy")
    h.stopping_actor = "cli:laptop"
    h.api.instances_queue = [GONE]
    h.view_queue = [{"status": "running"}]
    h.run(["--budget", "5"])

    assert h.events[-1]["reason"] == "operator_destroy"
    assert "relaunched" not in h.event_kinds()
    assert "evicted" not in h.event_kinds()


def test_supervise_terminal_done_exits_while_box_still_live(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-done")
    h.api.instances_queue = [_listed("run-done", 601)]
    h.view_queue = [{"status": "done"}]
    h.run(["--budget", "5"])

    assert h.events[-1]["reason"] == "terminal:done"
    assert "relaunched" not in h.event_kinds()


def test_supervise_running_is_noop_until_terminal(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-noop")
    live = _listed("run-noop", 701)
    h.api.instances_queue = [live, live, live]
    h.view_queue = [{"status": "running"}, {"status": "running"}, {"status": "done"}]
    h.run(["--budget", "5", "--interval", "1"])

    heartbeats = [e for e in h.events if e["event"] == "heartbeat"]
    assert len(heartbeats) == 3                       # 2 noop ticks + terminal tick
    assert all(e["actual_status"] == "running" for e in heartbeats)
    assert "evicted" not in h.event_kinds()
    assert "relaunched" not in h.event_kinds()
    assert h.events[-1]["reason"] == "terminal:done"


def test_supervise_host_death_evicts_then_relaunches_then_terminal(monkeypatch, tmp_path):
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this
    # scenario reaches `_relaunch` -> `_reset_run_markers`, which is no longer
    # a raising seam — the body landed in `supervise.replacement`.
    h = Harness(monkeypatch, tmp_path, "run-hostdeath")
    h.api.instances_queue = [GONE, GONE, GONE]
    h.view_queue = [
        {"status": "running"},   # tick1: streak=1 -> debounce noop
        {"status": "running"},   # tick2: streak=2 -> emit_evicted(host_death)
        {"status": "evicted"},   # tick3: backoff elapsed -> relaunch
        {"status": "done"},      # tick4: training finished -> terminal exit
    ]
    h.run(["--budget", "50", "--interval", "200"])

    kinds = h.event_kinds()
    assert kinds[0] == "supervisor_started" and kinds[1] == "supervised"
    assert kinds.count("evicted") == 1
    assert kinds.count("relaunched") == 1
    assert kinds[-1] == "supervisor_exiting"

    evicted_ev = next(e for e in h.events if e["event"] == "evicted")
    assert evicted_ev["reason"] == "host_death"

    relaunched_ev = next(e for e in h.events if e["event"] == "relaunched")
    assert relaunched_ev["instance_id"] == 42424242
    assert relaunched_ev["relaunch_count"] == 1

    assert h.events[-1]["reason"] == "terminal:done"
    assert ("PUT", "v0/asks/9001/") in h.api.calls    # relaunch actually issued a PUT


def test_supervise_outbid_stopped_but_listed_detected_via_actual_status(
        monkeypatch, tmp_path):
    """The design's core Q1 fact: an outbid box stays LISTED (HTTP 200, present)
    with actual_status=stopped — eviction must come from that field, never from
    'show failed'."""
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this
    # scenario reaches `_relaunch` -> `_reset_run_markers`, which is no longer
    # a raising seam — the body landed in `supervise.replacement`.
    h = Harness(monkeypatch, tmp_path, "run-outbid")
    listed_stopped = _listed("run-outbid", 801, actual_status="stopped")
    h.api.instances_queue = [listed_stopped, listed_stopped, listed_stopped]
    h.view_queue = [
        {"status": "running"},
        {"status": "running"},
        {"status": "evicted"},
        {"status": "done"},
    ]
    h.run(["--budget", "50", "--interval", "200"])

    evicted_ev = next(e for e in h.events if e["event"] == "evicted")
    assert evicted_ev["reason"] == "outbid"
    assert evicted_ev["instance_id"] == 801           # husk captured while present

    relaunched_ev = next(e for e in h.events if e["event"] == "relaunched")
    assert relaunched_ev["relaunch_count"] == 1

    # husk destroyed (and confirmed gone) strictly before the new PUT
    assert ("DELETE", "v0/instances/801/") in h.api.calls
    assert h.api.calls.index(("DELETE", "v0/instances/801/")) < \
        h.api.calls.index(("PUT", "v0/asks/9001/"))


def test_supervise_transient_burst_survives_no_relaunch_then_fatal_stop(
        monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-transient")
    transient_a = (False, None, "HTTP 503 on GET v1/instances/: unavailable")
    transient_b = (False, None, "network URLError(timed out) on GET v1/instances/")
    transient_c = (False, None, "HTTP 429 on GET v1/instances/: rate limited")
    fatal = (False, None, "HTTP 401 on GET v1/instances/: bad key")
    h.api.instances_queue = [transient_a, transient_b, transient_c, fatal]
    h.run(["--budget", "50", "--interval", "5"])

    kinds = h.event_kinds()
    assert "evicted" not in kinds
    assert "relaunched" not in kinds
    hb = [e for e in h.events if e["event"] == "heartbeat"]
    assert len(hb) == 3                                # one per transient tick
    assert all(e["actual_status"] == "unknown" for e in hb)
    assert hb[0]["last_error"] == transient_a[2]
    assert hb[2]["last_error"] == transient_c[2]
    assert h.events[-1]["reason"].startswith("observe_fatal:")
    assert "HTTP 401" in h.events[-1]["reason"]


def test_supervise_fatal_404_stops_cleanly_no_relaunch(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-fatal404")
    h.api.instances_queue = [(False, None, "HTTP 404 on GET v1/instances/: not found")]
    h.run(["--budget", "5"])

    assert h.event_kinds() == ["supervisor_started", "supervised", "cost",
                               "supervisor_exiting"]
    assert "HTTP 404" in h.events[-1]["reason"]


def test_supervise_max_relaunch_guardrail_stops_the_loop(monkeypatch, tmp_path):
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this
    # scenario reaches `_relaunch` -> `_reset_run_markers`, which is no longer
    # a raising seam — the body landed in `supervise.replacement`.
    h = Harness(monkeypatch, tmp_path, "run-maxrelaunch")
    h.api.instances_queue = [GONE, GONE, GONE]
    h.view_queue = [
        {"status": "running"},
        {"status": "running"},
        {"status": "evicted"},
    ]
    # queue exhausted after the 1 relaunch -> instances/view fall to defaults
    # (still "gone" / "unknown"); the cap must trip before a 2nd relaunch.
    h.run(["--budget", "50", "--interval", "200", "--max-relaunch", "1"])

    kinds = h.event_kinds()
    assert kinds.count("relaunched") == 1
    assert kinds.count("evicted") == 1
    assert h.events[-1]["reason"] == "max_relaunch"


def test_supervise_budget_guardrail_stops_relaunch_after_eviction(monkeypatch, tmp_path):
    # The spend/time caps are enforced at step 2c (before the live short-circuit)
    # AND max_relaunch at step 5 (before re-issue). This test exercises the path
    # where the box is live long enough to accrue cost past budget, THEN goes
    # away — the cap trips before any relaunch (and, per the fix, would also trip
    # while still live; see the two continuously-live cap tests below).
    h = Harness(monkeypatch, tmp_path, "run-budget")
    live = _listed("run-budget", 901, dph_total=36.0)   # $36/hr -> $0.01/s
    h.api.instances_queue = [live, live, GONE, GONE]
    h.view_queue = [{"status": "running"}] * 4
    h.run(["--budget", "1.0", "--interval", "200"])
    # tick1: live, dt=0 -> no cost yet -> noop(live)
    # tick2: live, dt=200 -> spend_usd = 36/3600*200 = 2.0 -> noop(live)
    # tick3: gone, streak=1 -> debounce noop (guardrail not reached: debounce
    #        precedes it in the ladder, same as test_poll_debounce_precedes_guardrail)
    # tick4: gone, streak=2 -> spend_usd(2.0) >= budget(1.0) -> stop_budget

    assert h.events[-1]["reason"] == "budget"
    assert "relaunched" not in h.event_kinds()


def test_supervise_wall_budget_guardrail_stops_relaunch_after_eviction(monkeypatch, tmp_path):
    # same reachability caveat as above: wall_budget is checked in the same
    # not-live-only guardrail step, so exercise it via an explicit eviction.
    h = Harness(monkeypatch, tmp_path, "run-wallbudget")
    h.api.instances_queue = [GONE, GONE, GONE]
    h.view_queue = [
        {"status": "running"},
        {"status": "running"},
        {"status": "evicted"},
    ]
    h.run(["--budget", "1000", "--interval", "200", "--wall-budget", "250"])

    assert h.events[-1]["reason"] == "wall_budget"
    assert "relaunched" not in h.event_kinds()


# poll() checks _spend_time_exceeded BEFORE the live short-circuit, and the
# driver PARKS the live box on stop_budget (2026-07-10 suspend-by-default:
# GPU billing — the dominant cost — stops; disk kept for diagnosis), with a
# destroy fallback if the stop doesn't take (tested separately below).
def test_supervise_budget_guardrail_should_hard_stop_a_continuously_live_overbudget_box(
        monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-budget-live", max_sleeps=40)
    live = _listed("run-budget-live", 901, dph_total=36.0)   # $36/hr -> $0.01/s
    h.api.instances_queue = [live] * 40
    h.view_queue = [{"status": "running"}] * 40
    h.run(["--budget", "0.05", "--interval", "2"])            # 5c cap, 2s/tick

    assert h.events[-1]["reason"] == "budget"
    assert "evicted" not in h.event_kinds()
    assert "relaunched" not in h.event_kinds()
    # the live over-budget box must be PARKED (GPU bill stopped), not destroyed
    assert ("PUT", "v0/instances/901/") in h.api.calls
    assert ("DELETE", "v0/instances/901/") not in h.api.calls


# the cap must still GUARANTEE the bill stops: if the park PUT fails, the
# driver falls back to destroy.
def test_supervise_budget_park_failure_falls_back_to_destroy(
        monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-budget-parkfail", max_sleeps=40)
    live = _listed("run-budget-parkfail", 903, dph_total=36.0)
    h.api.instances_queue = [live] * 40
    h.view_queue = [{"status": "running"}] * 40
    h.api.stop_put = (False, None, "HTTP 500 on PUT")         # park refused
    h.run(["--budget", "0.05", "--interval", "2"])

    assert h.events[-1]["reason"] == "budget"
    assert ("DELETE", "v0/instances/903/") in h.api.calls


# same contract as the budget case: --wall-budget hard-stops + PARKS a
# continuously-live box (destroy only as fallback).
def test_supervise_wall_budget_guardrail_should_hard_stop_a_continuously_live_box(
        monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-wallbudget-live", max_sleeps=40)
    live = _listed("run-wallbudget-live", 902)
    h.api.instances_queue = [live] * 40
    h.view_queue = [{"status": "running"}] * 40
    h.run(["--budget", "1000", "--interval", "100", "--wall-budget", "250"])

    assert h.events[-1]["reason"] == "wall_budget"
    assert "evicted" not in h.event_kinds()
    assert "relaunched" not in h.event_kinds()
    assert ("PUT", "v0/instances/902/") in h.api.calls
    assert ("DELETE", "v0/instances/902/") not in h.api.calls


def test_supervise_wall_budget_guardrail_stops_during_a_transient_outage(
        monkeypatch, tmp_path):
    # the wall-clock HARD stop must still be honored even while obs_status is
    # transient (design §3: "HARD stop honored" inside the transient branch)
    h = Harness(monkeypatch, tmp_path, "run-wallbudget-transient")
    transient = (False, None, "HTTP 503 on GET v1/instances/: unavailable")
    h.api.instances_queue = [transient] * 10
    h.run(["--budget", "1000", "--interval", "100", "--wall-budget", "250"])

    assert h.events[-1]["reason"] == "wall_budget"
    assert "evicted" not in h.event_kinds()
    assert "relaunched" not in h.event_kinds()


# =============================================================================
# 3b. Bid movement driver (SPOT_DESIGN §3.2): defend / rescue / resume-wait.
#     The market read is a POST v0/bundles/ filtered to machine_id — the fake
#     _listed instance carries a machine_id so _observe issues it; the bid PUT is
#     PUT v0/instances/bid_price/<id>/.
# =============================================================================
def test_supervise_market_pressure_raises_bid_on_live_box(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-defend")
    live = _listed("run-defend", 701, machine_id=555)
    h.api.instances_queue = [live, live]
    h.api.offers = (True, {"offers": [{"id": 9001, "min_bid": 0.55}]}, None)
    h.view_queue = [{"status": "running"}, {"status": "done"}]
    h.run(["--budget", "50", "--interval", "1", "--price", "0.60", "--max-bid", "0.75"])

    kinds = h.event_kinds()
    assert "bid_raised" in kinds
    br = next(e for e in h.events if e["event"] == "bid_raised")
    assert br["phase"] == "defend"
    assert br["old"] == 0.60 and br["new"] == 0.66 and br["market_min_bid"] == 0.55
    # BID_TARGET_MULT history: 1.2 -> 2.00 (2026-08-08 displacement audit) ->
    # 1.20 (2026-08-09 owner ruling). At 1.20x the MULTIPLE prices the raise:
    # 1.2 x 0.55 = $0.66, under the 0.65 x $5.00 = $3.25 cost cap AND under
    # --max-bid $0.75. (During the 2.00 era the target was $1.10 and --max-bid
    # $0.75 was what bound.)
    assert ("PUT", "v0/instances/bid_price/701/") in h.api.calls
    assert "evicted" not in kinds and "relaunched" not in kinds
    assert h.events[-1]["reason"] == "terminal:done"


def test_supervise_outbid_rescued_in_place_no_relaunch(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-rescue")
    stopped = _listed("run-rescue", 801, actual_status="stopped", machine_id=555)
    live = _listed("run-rescue", 801, machine_id=555)
    h.api.instances_queue = [stopped, stopped, live, live]
    h.api.offers = (True, {"offers": [{"id": 9001, "min_bid": 0.20}]}, None)
    h.view_queue = [{"status": "running"}, {"status": "running"},
                    {"status": "running"}, {"status": "done"}]
    h.run(["--budget", "50", "--interval", "200", "--price", "0.10",
           "--max-bid", "0.75", "--rescue-wait", "900"])

    kinds = h.event_kinds()
    br = next(e for e in h.events if e["event"] == "bid_raised")
    assert br["phase"] == "rescue" and br["new"] == 0.24   # 1.2 x floor 0.20
    # (mult history: 1.2 -> 2.00 on 2026-08-08, when this was 0.40 -> back to
    # 1.20 by owner ruling 2026-08-09)
    assert "rescued" in kinds
    assert "evicted" not in kinds and "relaunched" not in kinds
    assert ("PUT", "v0/instances/bid_price/801/") in h.api.calls
    # rescued in place -> the stopped box is NEVER destroyed
    assert ("DELETE", "v0/instances/801/") not in h.api.calls
    assert h.events[-1]["reason"] == "terminal:done"


def test_supervise_rescue_wait_timeout_falls_through_to_relaunch(monkeypatch, tmp_path):
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this
    # scenario reaches `_relaunch` -> `_reset_run_markers`, which is no longer
    # a raising seam — the body landed in `supervise.replacement`.
    h = Harness(monkeypatch, tmp_path, "run-rescuetimeout")
    stopped = _listed("run-rescuetimeout", 801, actual_status="stopped", machine_id=555)
    h.api.instances_queue = [stopped, stopped, stopped, stopped]
    h.api.offers = (True, {"offers": [{"id": 9001, "min_bid": 0.20}]}, None)
    h.view_queue = [{"status": "running"}, {"status": "running"},
                    {"status": "running"}, {"status": "evicted"}, {"status": "done"}]
    # interval 200 > rescue-wait 100 -> the wait elapses within one poll gap
    h.run(["--budget", "50", "--interval", "200", "--price", "0.10",
           "--max-bid", "0.75", "--rescue-wait", "100"])

    kinds = h.event_kinds()
    assert "bid_raised" in kinds
    assert kinds.count("rescued") == 0
    assert kinds.count("evicted") == 1
    assert kinds.count("relaunched") == 1
    # timed-out rescue-wait -> destroy husk (confirm gone) BEFORE the relaunch PUT
    assert ("DELETE", "v0/instances/801/") in h.api.calls
    assert h.api.calls.index(("DELETE", "v0/instances/801/")) < \
        h.api.calls.index(("PUT", "v0/asks/9001/"))
    assert h.events[-1]["reason"] == "terminal:done"


def test_supervise_bid_put_429_is_skipped_not_fatal(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path, "run-bid429")
    live = _listed("run-bid429", 701, machine_id=555)
    h.api.instances_queue = [live, live]
    h.api.offers = (True, {"offers": [{"id": 9001, "min_bid": 0.55}]}, None)
    h.api.bid = (False, None, "HTTP 429 on PUT v0/instances/bid_price/701/: rate limited")
    h.view_queue = [{"status": "running"}, {"status": "done"}]
    h.run(["--budget", "50", "--interval", "1", "--price", "0.60", "--max-bid", "0.75"])

    kinds = h.event_kinds()
    assert "bid_raised" not in kinds                  # PUT failed -> no event
    assert ("PUT", "v0/instances/bid_price/701/") in h.api.calls   # but attempted
    assert h.events[-1]["reason"] == "terminal:done"  # a 429 is never fatal


def test_do_bid_move_failed_put_still_advances_rate_limit_clock(monkeypatch):
    # a failed PUT (429/transient) must advance last_bid_put_ts anyway, so poll's
    # 60s guard holds and a 429 storm can't re-issue every --interval (<60s).
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: (False, "HTTP 429"))
    monkeypatch.setattr(time, "time", lambda: 5000.0)
    st = {"instance_id": 701, "run_id": "r", "market_min_bid": 0.55,
          "max_bid": 0.75, "last_bid": 0.60, "last_bid_put_ts": 0.0}
    a = types.SimpleNamespace(dry_run=False, rescue_wait=900)
    run_lane._do_bid_move(st, a, Action("raise_bid", "pressure"))
    assert st["last_bid_put_ts"] == 5000.0          # clock started despite failure
    assert st["last_bid"] == 0.60                    # bid unchanged (PUT failed)
    assert st["last_error"] == "HTTP 429"


def test_supervise_offers_read_failure_disables_bid_but_not_eviction(monkeypatch, tmp_path):
    # a failed market read -> market_min_bid None -> no bid action, and the outbid
    # box still evicts+relaunches on the unchanged path (invariant §5.1). The
    # market read is isolated from the relaunch offer search here.
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this
    # scenario reaches `_relaunch` -> `_reset_run_markers`, which is no longer
    # a raising seam — the body landed in `supervise.replacement`.
    h = Harness(monkeypatch, tmp_path, "run-offersfail")
    # The market read is stubbed at its OWNER: the run lane resolves it as
    # `pricing._market_min_bid_read`, so a patch of the `herdd` re-export
    # would leave the real read live and the scenario would never terminate.
    monkeypatch.setattr(pricing, "_market_min_bid_soft", lambda mid, g=None: None)
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda mid, g=None: models.MarketRead(False, False, None))
    stopped = _listed("run-offersfail", 801, actual_status="stopped", machine_id=555)
    h.api.instances_queue = [stopped, stopped, stopped]
    h.view_queue = [{"status": "running"}, {"status": "running"},
                    {"status": "evicted"}, {"status": "done"}]
    h.run(["--budget", "50", "--interval", "200", "--price", "0.10", "--max-bid", "0.75"])

    kinds = h.event_kinds()
    assert "bid_raised" not in kinds
    assert kinds.count("evicted") == 1
    assert kinds.count("relaunched") == 1


# =============================================================================
# 3b-bis. #73 in the RUN lane — the self-referential floor.
#
# The jobs ladder got `bidpolicy.market_floor_is_self` on 2026-08-08
# (test_eviction_blindspot.py §3). The run lane shares the offers read, the
# `_bid_target` rails and `_refresh_default_ceiling`, so it shared the ratchet
# and NOT the guard: `poll()` on a self-referential floor of $2.697 answered
# `raise_bid defend:3.236`. Numbers are box 47218938's and box 47214941's, from
# FLEETD_INCIDENT_2026-08-08.md.
# =============================================================================
SELF_FLOOR_BID = 1.338            # 47218938: what launch printed AND PUT
SELF_FLOOR_DPH_TOTAL = 1.4752222222222222     # that bid + diskHour
SELF_FLOOR_ONDEMAND = 4.258


def _run_inst(iid=701, **over):
    """A run-lane instance body as `_observe` reads it: a BID box we are the
    live tenant of, priced with dph_base (the standing bid) distinct from
    dph_total (bid + storage)."""
    inst = {"id": iid, "label": "run:run-selffloor", "actual_status": "running",
            "intended_status": "running", "machine_id": 37586, "num_gpus": 1,
            "is_bid": True, "dph_base": SELF_FLOOR_BID,
            "dph_total": SELF_FLOOR_DPH_TOTAL}
    inst.update(over)
    return inst


def _run_observe_env(monkeypatch, inst, *, market, on_demand=SELF_FLOOR_ONDEMAND):
    """Stub every seam `_observe` touches. Returns the emitted supervisor
    events as (event, fields) pairs."""
    monkeypatch.setattr(
        api, "request_soft",
        lambda m, p, *a, **k: (True, {"instances": [inst] if inst else []}, None))
    monkeypatch.setattr(pricing, "_market_min_bid_soft", lambda mid, g=None: market)
    # _observe reads the evidence-preserving MarketRead (row-level self-floor
    # guard, review 2026-08-10 F3); a scalar `market` models one listed chunk
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda mid, g=None: models.MarketRead(
                            market is not None, market is not None, market,
                            floors=((market,) if market is not None else ()),
                            scaled=False))
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, g=None: on_demand)
    monkeypatch.setattr(launch_spec, "_read_run_soft",
                        lambda rid, live_iids=(): {"status": "running"})
    monkeypatch.setattr(launch_spec, "_last_stopping_actor", lambda rid: None)
    monkeypatch.setattr(launch_spec, "_status_marker_soft", lambda rid: None)
    events = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **f: events.append((ev, f)))
    return events


def _run_st(**over):
    # `_t0` is NOW, not 0: the wall-budget guardrail outranks every bid row in
    # poll()'s precedence, so an epoch-zero start would answer `stop_budget`
    # for every case below and prove nothing about the ladder.
    t0 = time.time()
    st = bidpolicy.mk_poll_state(max_bid=None, last_bid=None, now=t0)
    st.update({"run_id": "run-selffloor", "_t0": t0, "_last_obs_t": t0,
               "machine_id": None, "num_gpus": None, "is_bid": False,
               "self_floor_at": None, "floor_samples": [], "on_demand": None,
               "first_seen_dph": None, "explicit_max_bid": False,
               "strict_ceiling": False, "dph_total": None, "dt": 0.0,
               "husk_id": None, "instance_id": None, "obs_status": "ok",
               "last_error": None})
    st.update(over)
    return st


def test_run_lane_observe_seeds_last_bid_from_the_standing_bid(monkeypatch):
    """`dph_total` is the bid PLUS storage. Seeded from it, `last_bid` sits one
    storage sliver ($0.137 here) ABOVE the number vast reports back as the
    chunk's `min_bid` — and `market_floor_is_self` is an exact-equality test by
    design, so the guard below could not recognise our own bid."""
    _run_observe_env(monkeypatch, _run_inst(), market=None)
    st = run_lane._observe(_run_st(), None)
    assert st["is_bid"] is True
    assert st["last_bid"] == SELF_FLOOR_BID          # dph_base, NOT dph_total
    assert st["dph_total"] == SELF_FLOOR_DPH_TOTAL   # the cost basis is unchanged
    assert st["first_seen_dph"] == SELF_FLOOR_BID


def test_run_lane_reconciles_last_bid_from_the_box(monkeypatch):
    """Review 2026-08-10 (F2/M3): the run lane seeded `last_bid` from the
    ORIGINAL launch bid and updated it only on its own successful PUTs — after
    a fleetd restart (state rebuilt from the launch spec), an out-of-band
    `herdd bid --price`, or a PUT vast applied but answered 5xx, the guard's
    standing arm compared the echo against a price we no longer hold, and the
    covering history entry aged out of the echo window while the price still
    stood. The belief now follows the observed dph_base when no PUT of ours
    is in flight."""
    _run_observe_env(monkeypatch, _run_inst(), market=None)
    st = run_lane._observe(_run_st(last_bid=0.10), None)    # stale launch belief
    assert st["last_bid"] == SELF_FLOOR_BID, \
        "the lane's belief must follow the box's dph_base"
    # ...but a price we JUST PUT is not clobbered by a body fetched pre-PUT
    st2 = _run_st(last_bid=0.10)
    st2["last_bid_put_ts"] = time.time()
    st2 = run_lane._observe(st2, None)
    assert st2["last_bid"] == 0.10, "an in-flight PUT wins over a stale body"


def test_run_lane_self_referential_floor_is_suppressed(monkeypatch):
    """THE money bug in the run lane. Live bid box, standing bid $1.338, and the
    offers read hands back $1.338 — the price to displace OURSELVES. Without the
    guard `_bid_target` answers 2.00 x 1.338 and `poll()` returns `raise_bid`."""
    events = _run_observe_env(monkeypatch, _run_inst(), market=SELF_FLOOR_BID)
    st = run_lane._observe(_run_st(), None)

    assert st["market_min_bid"] is None, "our own bid was read back as the market"
    bidpolicy._refresh_default_ceiling(st)
    assert st["floor_samples"] == [], \
        "a self-read must not enter the median-floor fallback for max_bid"
    act = poll(st)
    assert act.kind == "noop", f"the run ladder chased its own bid: {act}"

    selfy = [f for ev, f in events if ev == "bid_self_floor"]
    assert len(selfy) == 1, "say it once, with both numbers"
    assert selfy[0]["market_min_bid"] == SELF_FLOOR_BID
    assert selfy[0]["standing_bid"] == SELF_FLOOR_BID


def test_run_lane_self_floor_guard_is_tenant_gated(monkeypatch):
    """On a STOPPED box the same equality means the opposite thing: somebody
    else holds the chunk now, at a price that happens to match what we were
    paying. That is a real market read and the rescue ladder's whole input."""
    inst = _run_inst(actual_status="stopped", intended_status="stopped")
    events = _run_observe_env(monkeypatch, inst, market=SELF_FLOOR_BID)
    st = run_lane._observe(_run_st(last_bid=SELF_FLOOR_BID), None)

    assert st["market_min_bid"] == SELF_FLOOR_BID
    bidpolicy._refresh_default_ceiling(st)
    assert st["floor_samples"] == [SELF_FLOOR_BID]
    assert not [1 for ev, _f in events if ev == "bid_self_floor"]


def test_run_lane_a_real_floor_still_gets_defended(monkeypatch):
    """The must-not-regress half, mirroring the jobs lane's. The suppression is
    EXACT-equality and tenant-gated, so a genuine market move on the same box
    still moves the bid — otherwise the fix is a silent disarming of defend."""
    _run_observe_env(monkeypatch, _run_inst(), market=1.60)
    st = run_lane._observe(_run_st(max_bid=3.0, explicit_max_bid=True), None)

    assert st["market_min_bid"] == 1.60
    bidpolicy._refresh_default_ceiling(st)
    assert st["floor_samples"] == [1.60]
    act = poll(st)
    assert act.kind == "raise_bid", \
        f"a real floor above our bid must still be defended, got {act}"
    assert bidpolicy._bid_target(st["market_min_bid"], st["max_bid"],
                               st["on_demand"]) > SELF_FLOOR_BID


def test_run_lane_self_floor_never_reaches_a_bid_put(monkeypatch, tmp_path):
    """End-to-end through the driver: no PUT bid_price is ever issued against a
    floor that is our own standing bid."""
    h = Harness(monkeypatch, tmp_path, "run-selffloor")
    monkeypatch.setattr(pricing, "_market_min_bid_soft",
                        lambda mid, g=None: SELF_FLOOR_BID)
    monkeypatch.setattr(pricing, "_market_min_bid_read",
                        lambda mid, g=None: models.MarketRead(
                            True, True, SELF_FLOOR_BID,
                            floors=(SELF_FLOOR_BID,), scaled=False))
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, g=None: SELF_FLOOR_ONDEMAND)
    live = (True, {"instances": [_run_inst(iid=901)]}, None)
    h.api.instances_queue = [live, live]
    h.view_queue = [{"status": "running"}, {"status": "done"}]
    h.run(["--budget", "50", "--interval", "1"])

    assert "bid_raised" not in h.event_kinds()
    assert ("PUT", "v0/instances/bid_price/901/") not in h.api.calls
    assert "bid_self_floor" in h.event_kinds()


# =============================================================================
# 3c. `herdd bid` thin command (SPOT_DESIGN §3.2): PUT bid_price in place.
# =============================================================================
def test_cmd_bid_dry_run_prints_and_never_puts(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: calls.append((a, k)) or (True, {}, None))
    monkeypatch.setattr(sys, "argv",
                        ["herdd", "bid", "701", "--price", "0.5", "--dry-run"])
    cli_main.main()
    out = capsys.readouterr().out
    assert "dry-run" in out and "701" in out
    assert calls == []                                # dry-run never PUTs


def test_cmd_bid_puts_bid_price_body(monkeypatch, capsys):
    seen = {}

    def fake(method, path, body=None, **k):
        seen["call"] = (method, path, body)
        return True, {"success": True}, None

    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(api, "request_soft", fake)
    monkeypatch.setattr(sys, "argv", ["herdd", "bid", "701", "--price", "0.5"])
    cli_main.main()
    assert seen["call"] == ("PUT", "v0/instances/bid_price/701/",
                            {"client_id": "me", "price": 0.5})


def test_cmd_bid_rejects_out_of_range_price(monkeypatch):
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["herdd", "bid", "701", "--price", "99"])
    with pytest.raises(SystemExit):
        cli_main.main()


# =============================================================================
# 4. P0 run-spec object (SPOT_DESIGN §3.1): spec build / capture / relaunch body.
#    No network — spec.json reads are faked; secrets come from the local env.
# =============================================================================
def _spec_ns(**over):
    base = dict(price=None, image=None, disk=None, onstart=None, runtype=None,
                env=None, dry_run=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_build_launch_spec_never_carries_secret_values():
    # every secret VALUE must be dropped; only the NAME survives (invariant §5.8)
    env_list = ["RUNSET=s", "B2_BUCKET=bkt", "BASE_MODEL_B2=m/base",
                "B2_KEY_ID=SUPERSECRETKID", "B2_APPLICATION_KEY=SUPERSECRETAPP",
                "HF_TOKEN=SECRETHF", "OPENROUTER_API_KEY=SECRETOR"]
    spec = launch_spec._build_launch_spec(
        run_id="r1", runset="s", image="img:1", image_login_ref=None, disk=40,
        runtype="ssh_direct", gpu=[], gpu_ram=0.0, num_gpus=1,
        env_list=env_list, onstart="#!/bin/sh\necho hi\n",
        orig_bid=None, max_bid=None)
    blob = json.dumps(spec)
    for leaked in ("SUPERSECRETKID", "SUPERSECRETAPP", "SECRETHF", "SECRETOR"):
        assert leaked not in blob
    assert set(spec["secret_env_keys"]) == {
        "B2_KEY_ID", "B2_APPLICATION_KEY", "HF_TOKEN", "OPENROUTER_API_KEY"}
    assert spec["env"] == {"RUNSET": "s", "B2_BUCKET": "bkt",
                           "BASE_MODEL_B2": "m/base"}
    assert base64.b64decode(spec["onstart_b64"]).decode() == "#!/bin/sh\necho hi\n"


def test_split_env_secrets_oddly_named_and_url_creds_never_leak():
    # a --env passthrough whose name lacks the classic TOKEN|KEY|SECRET|PASS
    # families, or whose value embeds a URL credential, must still be withheld
    # from the durable B2 spec (invariant §5.8) — name-only in secret_env_keys.
    env_list = [
        "RUNSET=s", "LLM_BASE_URL=https://api.example.com/v1",  # safe URL, kept
        "DATABASE_URL=postgres://u:pw@h:5432/db",               # URL cred -> secret
        "DOCKER_AUTH=abc123",                                   # AUTH family
        "hf_token=lowersecret",                                 # case-insensitive
        "MY_PASSWORD=hunter2",                                  # PASS/PWD family
    ]
    env, secret_keys = launch_spec._split_env_secrets(env_list)
    assert env == {"RUNSET": "s", "LLM_BASE_URL": "https://api.example.com/v1"}
    assert set(secret_keys) == {"DATABASE_URL", "DOCKER_AUTH", "hf_token",
                                "MY_PASSWORD"}
    # and nothing secret-shaped survives in a serialized spec
    spec = launch_spec._build_launch_spec(
        run_id="r1", runset="s", image="img:1", image_login_ref=None, disk=40,
        runtype="ssh_direct", gpu=[], gpu_ram=0.0, num_gpus=1,
        env_list=env_list, onstart="#!/bin/sh\n", orig_bid=None, max_bid=None)
    blob = json.dumps(spec)
    for leaked in ("postgres://u:pw@h", "abc123", "lowersecret", "hunter2"):
        assert leaked not in blob


# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_relaunch_body` landed in
# `vastlib.supervise.replacement`, so subject and seams move together —
# `_build_launch_spec` / `_read_spec_soft` / `_raw_events_soft` /
# `image_login_arg` / `_mask_image_login` at `launch.spec`, `_capture_launch_spec`
# at `supervise.run_lane`, `pub_key_text` at `boxes.ssh` (the module
# `_relaunch_body` reaches the injector through).
def test_spec_roundtrip_write_capture_relaunch_body(monkeypatch):
    # cmd_train's spec.json -> _capture -> _relaunch_body reproduces the box, with
    # secret VALUES re-injected from the LOCAL env (never read back from B2).
    env_list = ["RUN_ID=r1", "RUNSET=myset", "B2_BUCKET=bkt",
                "BASE_MODEL_B2=models/base",
                "B2_KEY_ID=orig-kid", "B2_APPLICATION_KEY=orig-key",
                "HF_TOKEN=orig-hf"]
    spec = launch_spec._build_launch_spec(
        run_id="r1", runset="myset", image="img:1", image_login_ref=None, disk=64,
        runtype="ssh_direct", gpu=["H100 PCIE"], gpu_ram=0.0, num_gpus=2,
        env_list=env_list, onstart="#!/bin/bash\necho hi\n",
        orig_bid=0.6, max_bid=0.75)
    monkeypatch.setattr(launch_spec, "_read_spec_soft", lambda rid: spec)
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: [])
    # box-scoped B2 creds are preferred at relaunch, mirroring cmd_train
    monkeypatch.setenv("B2_BOX_KEY_ID", "local-box-kid")
    monkeypatch.setenv("B2_BOX_APPLICATION_KEY", "local-box-key")
    monkeypatch.setenv("HF_TOKEN", "local-hf")
    monkeypatch.setattr(ssh_mod, "pub_key_text",
                        lambda *a, **k: "ssh-ed25519 AAAAtest u@h")

    a = _spec_ns()
    captured, orig_bid = run_lane._capture_launch_spec("r1", a)
    assert orig_bid == 0.6
    body, missing = replacement._relaunch_body({"run_id": "r1",
                                            "launch_spec": captured}, a, 0.5)
    assert missing == []
    assert body["image"] == "img:1" and body["disk"] == 64
    assert body["runtype"] == "ssh_direct"
    # The spec's wire replays verbatim — but PREFIXED with the ssh key
    # install/repair. cmd_train snapshots the PRE-inject wire (the injection
    # happens later, inside _do_launch), so before 2026-07-31 every relaunched
    # box was born un-ssh-able; box 46449950 is the live case.
    assert body["onstart"].endswith("#!/bin/bash\necho hi\n")
    assert models.SSH_INJECT_MARKER in body["onstart"]
    assert body["env"]["RUNSET"] == "myset"
    assert body["env"]["BASE_MODEL_B2"] == "models/base"
    assert body["env"]["B2_KEY_ID"] == "local-box-kid"          # box-scoped source
    assert body["env"]["B2_APPLICATION_KEY"] == "local-box-key"
    assert body["env"]["HF_TOKEN"] == "local-hf"


# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_relaunch_body` landed in
# `vastlib.supervise.replacement`, so subject and seams move together —
# `_build_launch_spec` / `_read_spec_soft` / `_raw_events_soft` /
# `image_login_arg` / `_mask_image_login` at `launch.spec`, `_capture_launch_spec`
# at `supervise.run_lane`, `pub_key_text` at `boxes.ssh` (the module
# `_relaunch_body` reaches the injector through).
def test_relaunch_body_image_login_rederived_token_never_in_spec(monkeypatch):
    # image_login is handed to vast's API and to a stranger's docker daemon, so
    # the spec stores only a REDACTED marker and the real credential is minted
    # fresh at relaunch. A relaunch that cannot mint one must REFUSE, not
    # silently relaunch unauthenticated onto a box we are already paying for.
    monkeypatch.setenv("REGISTRY_AUTH_SECRET", "s" * 32)
    image = "registry.example.com/train:t215-latest"
    login = launch_spec.image_login_arg(image, None)
    assert login.startswith("-u vast -p ")
    tok = login.split()[3]
    ref = launch_spec._mask_image_login(login)
    assert tok not in ref                               # spec marker is redacted
    spec = {"image": image, "image_login": ref,
            "env": {}, "secret_env_keys": []}
    a = _spec_ns()
    body, missing = replacement._relaunch_body({"run_id": "r1", "launch_spec": spec},
                                           a, 0.5)
    assert missing == []
    # re-derived, not stored: a fresh mint, same shape, never the marker
    assert body["image_login"].startswith("-u vast -p ")
    assert body["image_login"] != ref
    monkeypatch.delenv("REGISTRY_AUTH_SECRET")          # secret gone -> refuse
    _, missing2 = replacement._relaunch_body({"run_id": "r1", "launch_spec": spec},
                                         a, 0.5)
    assert missing2 == ["REGISTRY_AUTH_SECRET"]
    assert launch_spec.image_login_arg(image, None) is None


# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_relaunch_body` landed in
# `vastlib.supervise.replacement`, so subject and seams move together —
# `_build_launch_spec` / `_read_spec_soft` / `_raw_events_soft` /
# `image_login_arg` / `_mask_image_login` at `launch.spec`, `_capture_launch_spec`
# at `supervise.run_lane`, `pub_key_text` at `boxes.ssh` (the module
# `_relaunch_body` reaches the injector through).
def test_capture_falls_back_to_event_scrape_when_no_spec(monkeypatch):
    # pre-spec runs keep working at the old fidelity; RUNSET still reaches the box
    monkeypatch.setattr(launch_spec, "_read_spec_soft", lambda rid: {})
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: [
        {"event": "launched", "image": "legacy:img", "disk": 80,
         "runtype": "ssh_direct", "runset": "oldset", "dph": 0.42}])
    a = _spec_ns()
    spec, orig_bid = run_lane._capture_launch_spec("r1", a)
    assert spec["image"] == "legacy:img" and spec["runset"] == "oldset"
    assert not spec.get("secret_env_keys")             # legacy has no secret list
    assert orig_bid == 0.42
    body, missing = replacement._relaunch_body({"run_id": "r1", "launch_spec": spec},
                                           a, 0.5)
    assert missing == [] and body["env"]["RUNSET"] == "oldset"


def test_read_spec_soft_parses_and_degrades(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (0, '{"v":1,"run_id":"r1"}', ""))
    assert launch_spec._read_spec_soft("r1") == {"v": 1, "run_id": "r1"}
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "boom"))
    assert launch_spec._read_spec_soft("r1") == {}          # read failure -> degrade
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (0, "not json", ""))
    assert launch_spec._read_spec_soft("r1") == {}          # malformed -> degrade
    monkeypatch.delenv("B2_BUCKET")
    assert launch_spec._read_spec_soft("r1") == {}          # no bucket -> degrade


# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_relaunch_body` is ported, so
# `replacement._relaunch` reaches a real body. This row REFUSES on the missing
# secret before it launches, so it never reaches `_reset_run_markers` — the one
# seam on that path that is still unported (see the two tests at the end of this
# group, which stay flat for it).
def test_relaunch_refuses_and_keeps_husk_when_secret_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    events, destroyed = [], []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: events.append((ev, kw)))
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"instances": []}, None))
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, dry_run=False: destroyed.append(iid) or (True, None))
    st = {"run_id": "r1", "husk_id": 555, "max_bid": None,
          "launch_spec": {"image": "img", "env": {}, "secret_env_keys": ["HF_TOKEN"]}}
    verdict = replacement._relaunch(st, _spec_ns())
    assert verdict == "stop_fatal"
    assert destroyed == []                              # husk NOT destroyed on refusal
    assert any(ev == "relaunch_refused" for ev, _ in events)
    assert st["last_error"] == "missing_secret_env:HF_TOKEN"


# MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d). These two tests
# ARE `_reset_run_markers`' behaviour spec, so they follow the body: it moved
# from `herdd.py` into `supervise.replacement` (home ruling on the def there —
# `storage.b2` takes no path policy). Subject is the owner and the transport
# seams are patched at `storage.b2`, because that is what the body resolves;
# patching the `herdd` re-exports would steer nothing at all.
def test_reset_run_markers_stamps_relaunched_and_clears_holds(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    rcats, rclones = [], []
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True: rcats.append((path, body)))
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: rclones.append(args) or (0, "", ""))
    replacement._reset_run_markers("r1", dry_run=False)
    assert rcats[0][0] == "b2:bkt/checkpoints/r1/STATUS"
    assert rcats[0][1].startswith("RELAUNCHED ")
    assert ["deletefile", "b2:bkt/checkpoints/r1/STOP"] in rclones
    assert ["deletefile", "b2:bkt/checkpoints/r1/EXTEND"] in rclones


# MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d). These two tests
# ARE `_reset_run_markers`' behaviour spec, so they follow the body: it moved
# from `herdd.py` into `supervise.replacement` (home ruling on the def there —
# `storage.b2` takes no path policy). Subject is the owner and the transport
# seams are patched at `storage.b2`, because that is what the body resolves;
# patching the `herdd` re-exports would steer nothing at all.
def test_reset_run_markers_skips_without_bucket_or_on_dry_run(monkeypatch):
    called = []
    monkeypatch.setattr(b2, "_b2_rcat", lambda *a, **k: called.append(1))
    monkeypatch.delenv("B2_BUCKET", raising=False)
    replacement._reset_run_markers("r1")                # no bucket -> skip
    monkeypatch.setenv("B2_BUCKET", "bkt")
    replacement._reset_run_markers("r1", dry_run=True)  # dry-run -> skip
    assert called == []


# --------------------------------------------------------------------------- #
# CPU compile-farm advisory status fold (cmd_runs FARM column) — pure + injected
# rclone runner, no network. See jobs_view._farm_status_by_run / _parse_farm_status.
# --------------------------------------------------------------------------- #
def test_parse_farm_status_word_and_empty():
    assert jobs_view._parse_farm_status("RUNNING 2026-07-09T00:00:00Z") == "RUNNING"
    assert jobs_view._parse_farm_status("done 2026-07-09T00:00:00Z") == "DONE"
    assert jobs_view._parse_farm_status("") is None
    assert jobs_view._parse_farm_status(None) is None
    assert jobs_view._parse_farm_status("   ") is None


def _fake_farm_runner(dirs, status):
    """rclone-shaped runner: lsf farm/ lists `dirs`; cat farm/<rid>/FARM_STATUS
    returns status[rid] (rc=1 == absent marker)."""
    def runner(args):
        if args[:2] == ["lsf", "--dirs-only"] and args[-1].endswith("/farm/"):
            return 0, "".join(f"{d}/\n" for d in dirs), ""
        if args and args[0] == "cat" and "/farm/" in args[-1]:
            rid = args[-1].split("/farm/")[1].split("/")[0]
            if rid in status:
                return 0, status[rid], ""
            return 1, "", "not found"
        return 1, "", "unexpected"
    return runner


def test_farm_fold_maps_only_matching_runs():
    # farm/ has dirs for r1 (RUNNING) and rX (a custom --farm-run that is not a
    # training run row) — only the displayed run r1 gets annotated; r2 has no
    # farm namespace -> absent from the map (row falls back to '-').
    runner = _fake_farm_runner(
        ["r1", "rX"],
        {"r1": "RUNNING 2026-07-09T00:00:00Z", "rX": "DONE 2026-07-09T01:00:00Z"})
    got = jobs_view._farm_status_by_run("b2:bkt", ["r1", "r2"], runner=runner)
    assert got == {"r1": "RUNNING"}


def test_farm_fold_zero_cost_when_no_farm_namespace():
    # lsf returns empty -> no per-run cats, empty map. (One rclone call total.)
    calls = []
    def runner(args):
        calls.append(args)
        return 0, "", ""          # lsf farm/ empty
    got = jobs_view._farm_status_by_run("b2:bkt", ["r1", "r2"], runner=runner)
    assert got == {}
    assert len(calls) == 1        # exactly the gating lsf, no FARM_STATUS cats


def test_ckpt_steps_by_run_folds_every_layout():
    """cmd_runs STEP fallback: max checkpoint-<n> per run from ONE recursive lsf.

    Covers all THREE on-disk layouts — bare, adapter/, and the arms/<arm>/ one
    every ladder run uses (which lives at depth 4, the reason the listing is not
    capped at 3) — collects train_summary.json paths out of the same pass,
    ignores non-checkpoint dirs, mints no slot for a run with neither, and never
    raises on an rclone failure (advisory column). See _ckpt_steps_by_run."""
    listing = ("r1/checkpoint-25/\n"
               "r1/checkpoint-200/\n"
               "r2/adapter/checkpoint-4/\n"
               "r2/adapter/checkpoint-217/\n"
               "r4/arms/DSP/checkpoint-20/\n"
               "r4/arms/DSP/checkpoint-196/\n"
               "r4/arms/B/checkpoint-62/\n"
               "r5/train_summary.json\n"
               "r3/artifacts/\n"
               "r3/\n")
    calls = []

    def runner(args):
        calls.append(args)
        return 0, listing, ""

    got = jobs_view._ckpt_steps_by_run("b2:bkt", runner=runner)
    assert {k: v["step"] for k, v in got.items()} == {
        "r1": 200, "r2": 217, "r4": 196, "r5": None}
    # max is across ARMS, not per-arm: DSP's 196 beats B's 62
    assert got["r5"]["summaries"] == ["b2:bkt/checkpoints/r5/train_summary.json"]
    assert "r3" not in got              # no checkpoints, no summary, no slot
    assert len(calls) == 1              # exactly one listing for the whole tree
    assert "4" in calls[0]              # depth 4 — arms/<arm>/checkpoint-N

    def dead(args):
        return 127, "", "rclone not found on PATH"
    assert jobs_view._ckpt_steps_by_run("b2:bkt", runner=dead) == {}


def test_train_summary_step_is_the_last_step_source():
    """A finished run whose checkpoint dirs were pruned still records
    global_steps in train_summary.json. json.loads must survive the bare NaN
    literals the trainer writes for a run with no recorded loss points, and any
    unreadable/garbage summary is skipped rather than raised."""
    bodies = {
        "b2:bkt/checkpoints/r1/train_summary.json":
            '{"global_steps": 300, "wall_seconds": 10555.8, "loss_last": NaN}',
        "b2:bkt/checkpoints/r2/arms/A/train_summary.json": '{"global_steps": 62}',
        "b2:bkt/checkpoints/r2/arms/B/train_summary.json": '{"global_steps": 196}',
        "b2:bkt/checkpoints/r3/train_summary.json": 'not json at all',
    }

    def runner(args):
        body = bodies.get(args[1])
        return (0, body, "") if body is not None else (1, "", "not found")

    assert jobs_view._train_summary_step(
        ["b2:bkt/checkpoints/r1/train_summary.json"], runner=runner) == 300
    # multi-arm: max across arms, same as the checkpoint-dir scan
    assert jobs_view._train_summary_step(
        ["b2:bkt/checkpoints/r2/arms/A/train_summary.json",
         "b2:bkt/checkpoints/r2/arms/B/train_summary.json"], runner=runner) == 196
    assert jobs_view._train_summary_step(
        ["b2:bkt/checkpoints/r3/train_summary.json"], runner=runner) is None
    assert jobs_view._train_summary_step(["b2:bkt/checkpoints/gone/x.json"],
                                       runner=runner) is None
    assert jobs_view._train_summary_step([], runner=runner) is None


def test_emit_launched_soft_records_boxes_train_never_saw(monkeypatch):
    """RECORDING fix: every run:-labelled launch now writes a `launched` event.

    Only `cmd_train` used to emit one, so plain `herdd launch --label run:<id>`
    and the jobs/workflow arm launcher recorded nothing at all — no gpu, no dph,
    no offer, no start time — and no fold change can recover a fact that was
    never written. Asserts the event carries the price the cost estimate needs,
    that a non-run label is a no-op, that cmd_train's own richer event is not
    doubled, and that a transport failure cannot fail a launch that succeeded."""
    import argparse
    emitted = []
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(rm, "emit_event",
                        lambda run_id, event, **f: emitted.append(
                            (run_id, event, f)))
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (True, {"gpu_name": "H100",
                                                "machine_id": 77}, None))
    a = argparse.Namespace(gpu=["h100"])
    body = {"label": "run:myrun", "image": "img:1", "disk": 40,
            "runtype": "ssh_direct"}

    lifecycle._emit_launched_soft(a, body, 4242, offer_id=99, dph="1.25")
    assert len(emitted) == 1
    run_id, event, f = emitted[0]
    assert (run_id, event) == ("myrun", "launched")
    assert f["instance_id"] == 4242 and f["offer_id"] == 99
    assert f["dph"] == 1.25                # numeric: the cost estimate reads it
    assert f["gpu"] == "H100"              # the ACTUAL card, not the selector
    assert f["machine_id"] == 77

    # a label that names no run, and the fleetd `:keep` suffix which must still
    # resolve to the bare run id (labels are appendable)
    lifecycle._emit_launched_soft(a, {"label": "upstream-monorepo"}, 1, None, None)
    assert len(emitted) == 1
    lifecycle._emit_launched_soft(a, {"label": "run:myrun:keep"}, 5, None, None)
    assert emitted[-1][0] == "myrun"

    # cmd_train suppresses it — its own step-13 event is richer, and two
    # `launched` events in one epoch make newest-launch-wins ambiguous
    a.__dict__["_runmeta_launched"] = True
    before = len(emitted)
    lifecycle._emit_launched_soft(a, body, 7, 1, 1.0)
    assert len(emitted) == before


def test_emit_launched_soft_never_fails_a_successful_launch(monkeypatch):
    import argparse

    def boom(*a, **k):
        raise RuntimeError("B2 is down")

    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(rm, "emit_event", boom)
    monkeypatch.setattr(api, "request_soft", boom)
    # the box is already running; a metadata write must not raise past here
    lifecycle._emit_launched_soft(argparse.Namespace(gpu=[]),
                                {"label": "run:r1"}, 1, 2, 0.5)


def test_farm_fold_never_raises_on_rclone_failure():
    def runner(args):
        return 127, "", "rclone not found on PATH"
    # advisory column must degrade, not crash the whole `runs` listing
    assert jobs_view._farm_status_by_run("b2:bkt", ["r1"], runner=runner) == {}


# =============================================================================
# job-box stop classification (SPOT_DESIGN §3.5) — the three-way replacement for
# the stale mine-vs-operator binary that misread an OUTBID as an operator park
# (2026-07-11 bakeoff-05). Pure function, tested like poll().
# =============================================================================
classify = job_lane.classify_job_box_stop


def test_classify_self_parked_is_success():
    # (a) jobd self-parked on drain -> parked_self box-event -> SUCCESS exit,
    # regardless of bid-ness / intended_status.
    assert classify(present=True, live=False, is_bid=True, intended_status="stopped",
                    box_parked=True, box_drained=False) == "self_parked"
    assert classify(present=True, live=False, is_bid=False, intended_status="running",
                    box_parked=True, box_drained=False) == "self_parked"
    # a `drained` (couldn't self-park; asked the laptop to) also counts as (a)
    assert classify(present=True, live=False, is_bid=False, intended_status="stopped",
                    box_parked=False, box_drained=True) == "self_parked"


def test_classify_bid_outbid_falls_through_to_rescue_not_operator_park():
    # (b) THE REGRESSION: a bid box shows stopped with NO self-park event. It must
    # NOT be read as an operator park — return None so supervise rescues it.
    assert classify(present=True, live=False, is_bid=True, intended_status="stopped",
                    box_parked=False, box_drained=False) is None
    # even if intended_status is running (frozen), still not an operator park
    assert classify(present=True, live=False, is_bid=True, intended_status="running",
                    box_parked=False, box_drained=False) is None


def test_classify_ondemand_operator_park_is_clean_exit():
    # (c) an ON-DEMAND box (never outbid) with intended_status=stopped and no
    # self-park event == a genuine operator `herdd stop` -> clean exit.
    assert classify(present=True, live=False, is_bid=False, intended_status="stopped",
                    box_parked=False, box_drained=False) == "operator_park"


def test_classify_self_park_wins_over_operator_intent():
    # box-event stream is consulted FIRST: parked_self beats intended_status.
    assert classify(present=True, live=False, is_bid=False, intended_status="stopped",
                    box_parked=True, box_drained=False) == "self_parked"


# --- integration: emit a parked_self box-event, fold it back via read_box, and
# feed the fold to the classifier (fake-B2 in-memory harness) -----------------
class _FakeBoxB2:
    """rcat/copy over an in-memory key->body store (the subset read_box +
    emit_box_event touch). Mirrors test_jobmeta.FakeB2 minimally."""
    def __init__(self, bucket="bkt"):
        self.bucket = bucket
        self.store = {}

    def _key(self, remote):
        p = f"b2:{self.bucket}/"
        assert remote.startswith(p), remote
        return remote[len(p):]

    def __call__(self, args, input=None):
        op = args[0]
        if op == "rcat":
            self.store[self._key(args[1])] = input
            return 0, "", ""
        if op == "copy":
            src, dst = self._key(args[1]), args[2]
            for k, body in self.store.items():
                if k.startswith(src):
                    fp = os.path.join(dst, k[len(src):])
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    with open(fp, "w") as fh:
                        fh.write(body)
            return 0, "", ""
        return 0, "", ""


def test_supervise_reads_parked_self_then_classifies_success(tmp_path):
    jm = jobmeta
    fake = _FakeBoxB2()
    iid = "44482324"
    jm.emit_box_event(iid, "parked_self", runner=fake, bucket="bkt",
                      reason="drained", idle_s=700, n_done=2, n_failed=0)
    bx = jm.read_box(iid, runner=fake, bucket="bkt", cache_dir=str(tmp_path / "c"))
    assert bx["parked"] and bx["park_reason"] == "drained"
    # the fold feeds the classifier -> success, even though the box "looks
    # stopped" like an eviction would
    assert classify(present=True, live=False, is_bid=True,
                    intended_status="stopped",
                    box_parked=bx["parked"], box_drained=bx["drained_pending"]) \
        == "self_parked"


def test_supervise_no_box_event_bid_outbid_is_rescue(tmp_path):
    # empty box-event stream (never self-parked) + a stopped bid box -> None
    # (rescue), NOT operator_park.
    jm = jobmeta
    fake = _FakeBoxB2()
    bx = jm.read_box("999", runner=fake, bucket="bkt", cache_dir=str(tmp_path / "c"))
    assert not bx["parked"] and not bx["drained_pending"]
    assert classify(present=True, live=False, is_bid=True,
                    intended_status="stopped",
                    box_parked=bx["parked"], box_drained=bx["drained_pending"]) is None


# =============================================================================
# 4. N2 supervise cost + liveness fixes (SPOT_DESIGN §3.2 bid-decay, §3.7 watchdog)
# =============================================================================
import datetime as _dt  # noqa: E402

_bid_action = bidpolicy._bid_action
_decay_candidate = bidpolicy._decay_candidate
_next_decay_streak = bidpolicy._next_decay_streak
_default_max_bid = bidpolicy._default_max_bid
_bid_target = bidpolicy._bid_target
_ckpt_watchdog_alarm = jobs_risk._ckpt_watchdog_alarm
_ts_to_epoch = fmt._ts_to_epoch

mk_hs = bidpolicy.mk_handoff_state
handoff_poll = bidpolicy.handoff_poll
HandoffAction = bidpolicy.HandoffAction
_handoff_candidate_ok = bidpolicy._handoff_candidate_ok
_handoff_headroom_ok = bidpolicy._handoff_headroom_ok


def _ts(epoch):
    """A jobmeta-format ts ('YYYYMMDDTHHMMSSmmmZ') for a given UTC epoch second."""
    return _dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc) \
        .strftime("%Y%m%dT%H%M%S") + "000Z"


# --- 4a. bid decay: the floor receded, LOWER the standing bid ----------------
def _decaying(**over):
    # live box, our standing bid 0.90, floor collapsed to 0.20 -> target 0.24
    # (1.2x), which is > 1 step below 0.90: a decay candidate.
    base = dict(view={"status": "running"}, present=True, actual_status="running",
                last_bid=0.90, max_bid=2.0, market_min_bid=0.20,
                last_bid_put_ts=0.0, now=1_000_000.0, decay_streak=0)
    base.update(over)
    return mk(**base)


def test_decay_candidate_true_when_live_floor_receded():
    assert _decay_candidate(_decaying()) is True


def test_decay_candidate_false_when_not_live():
    assert _decay_candidate(_decaying(actual_status="stopped", present=True)) is False


def test_decay_candidate_false_when_bid_near_floor():
    # floor high enough that target ~ our bid: not a decay candidate
    assert _decay_candidate(_decaying(market_min_bid=0.75)) is False  # 1.2*0.75=0.90 == bid


def test_next_decay_streak_increments_then_resets():
    s = _decaying(decay_streak=2)
    assert _next_decay_streak(s) == 3
    s2 = _decaying(decay_streak=2, market_min_bid=0.75)   # 1.2*0.75=0.90 == bid, not a candidate
    assert _next_decay_streak(s2) == 0


def test_bid_action_lowers_only_after_streak_threshold():
    # below the streak threshold: no move yet (avoid reacting to a brief dip)
    assert _bid_action(_decaying(decay_streak=2)) is None
    a = _bid_action(_decaying(decay_streak=3))
    assert a == Action("lower_bid", "decay:0.24")


def test_bid_action_decay_respects_rate_limit():
    s = _decaying(decay_streak=5, now=1000.0, last_bid_put_ts=970.0)  # 30s < 60s
    assert _bid_action(s) is None


def test_poll_lowers_bid_on_live_box_after_decay():
    a = poll(_decaying(decay_streak=3))
    assert a == Action("lower_bid", "decay:0.24")


def test_poll_decay_never_fires_when_a_raise_is_due():
    # market climbed to within 10% of our bid AND streak is high: raise wins,
    # the two conditions are mutually exclusive (target above vs below our bid).
    s = _decaying(market_min_bid=0.85, last_bid=0.90, decay_streak=9, max_bid=2.0)
    a = poll(s)
    assert a.kind == "raise_bid"


def test_bid_action_pure_no_mutation_on_decay():
    s = _decaying(decay_streak=3)
    before = copy.deepcopy(s)
    _bid_action(s)
    assert s == before


# --- 4a-bis. decay hysteresis: a rung owns the price it just paid ------------
_recent_raise_hold = bidpolicy._recent_raise_hold


def _hist(*entries, machine="m1"):
    """`ladder_core.record_bid` entries: [ts_first, price, machine_id, ts_last]."""
    return [[ts, price, machine, ts] for ts, price in entries]


def _with_hist(s, hist, machine="m1"):
    """`bid_history`/`machine_id` are lane state, not `mk_poll_state` keys (the
    RunState TypedDict pins that factory's shape) — the lanes assign them on."""
    return dict(s, bid_history=hist, machine_id=machine)


def _rung_then_decay(age_s, **over):
    """A live decay candidate whose standing bid was RAISED `age_s` ago by a
    rung (0.24 -> 0.30), which is the shape of the 2026-08-26 oscillation."""
    now = 1_000_000.0
    hist = over.pop("bid_history",
                    _hist((now - age_s - 600.0, 0.24), (now - age_s, 0.30)))
    machine = over.pop("machine_id", "m1")
    base = dict(last_bid=0.30, market_min_bid=0.20, decay_streak=3, now=now)
    base.update(over)
    return _with_hist(_decaying(**base), hist, machine)


def test_decay_refused_inside_the_rung_hysteresis_window():
    """The rung paid $0.30 to keep the warm box; giving it back four minutes
    later is the controller fighting itself, not the market."""
    s = _rung_then_decay(240.0)
    assert _recent_raise_hold(s) == 0.30
    assert _bid_action(s) is None
    assert poll(s).kind == "noop"


def test_decay_allowed_once_the_hysteresis_window_expires():
    s = _rung_then_decay(bidpolicy.REBID_WAIT_S + 1.0)
    assert _recent_raise_hold(s) is None
    assert _bid_action(s) == Action("lower_bid", "decay:0.24")


def test_hysteresis_window_boundary_is_rebid_wait_s_exactly():
    """The window is REBID_WAIT_S and the boundary is explicit, because the field
    case sits 4 s inside it: the 2026-08-26 03:38:12 rung was decayed away at
    03:43:08, 296 s later. Blocked at 295 s, allowed at 305 s. Whether 300 s is
    the right width is an owner question; that it is the width is this test."""
    assert bidpolicy.REBID_WAIT_S == 300
    assert _bid_action(_rung_then_decay(295.0)) is None
    assert _bid_action(_rung_then_decay(296.0)) is None          # the field case
    assert _bid_action(_rung_then_decay(305.0)) == Action("lower_bid", "decay:0.24")


def test_hysteresis_holds_only_the_price_the_rung_paid():
    """One-directional and bounded: a decay that still lands AT or ABOVE the
    rung's price is not giving anything back, so it runs."""
    s = _rung_then_decay(60.0, market_min_bid=0.30, last_bid=0.45)   # target 0.36
    assert _recent_raise_hold(s) == 0.30
    assert _bid_action(s) == Action("lower_bid", "decay:0.36")


def test_the_recorder_and_the_hysteresis_agree_on_the_entry_shape():
    """The seam, not a mock of it: `ladder_core.note_standing_bid` is what writes
    `bid_history`, and `_recent_raise_hold` is what reads it. A rung that raises
    the standing bid has to be visible to the guard through the REAL recorder —
    the run lane hands its persistent state straight to `_bid_action`, so this is
    that lane end to end."""
    import ladder_core
    now = 1_000_000.0
    st = _decaying(last_bid=0.30, market_min_bid=0.20, decay_streak=3, now=now)
    st["machine_id"] = 147654
    ladder_core.note_standing_bid(st, 0.24, 147654, now - 600.0)
    ladder_core.note_standing_bid(st, 0.30, 147654, now - 120.0)   # the rung
    assert _recent_raise_hold(st) == 0.30
    assert _bid_action(st) is None                       # decay refused
    # ...and once the rung's window has passed, the ladder decays again
    assert _bid_action(dict(st, now=now + bidpolicy.REBID_WAIT_S)) \
        == Action("lower_bid", "decay:0.24")


def test_hysteresis_never_blocks_a_raise_or_a_rescue():
    """The guard may only suppress a decay. A defend raise and a rescue on the
    same history are untouched."""
    now = 1_000_000.0
    hist = _hist((now - 600.0, 0.24), (now - 30.0, 0.30))
    raise_s = _with_hist(                                    # floor at our bid
        _decaying(last_bid=0.30, market_min_bid=0.29, now=now), hist)
    assert _bid_action(raise_s).kind == "raise_bid"
    rescue_s = _with_hist(
        _decaying(present=True, actual_status="exited", last_bid=0.30,
                  market_min_bid=0.40, now=now), hist)
    assert _bid_action(rescue_s).kind == "rescue_bid"


def test_hysteresis_is_inert_without_the_state_the_daemon_may_not_carry():
    """Backward compatibility with a state file written before the key existed:
    no `bid_history` (or a history from another machine) => the pre-hysteresis
    decay, never a crash."""
    assert _recent_raise_hold(_decaying()) is None
    assert _bid_action(_decaying(decay_streak=3)) == Action("lower_bid", "decay:0.24")
    other = _rung_then_decay(60.0, machine_id="m2")     # replacement box: no echoes
    assert _recent_raise_hold(other) is None
    assert _bid_action(other) == Action("lower_bid", "decay:0.24")
    junk = _rung_then_decay(60.0, bid_history=[[], ["x", "y"], [None, None, None]])
    assert _recent_raise_hold(junk) is None


def test_hysteresis_ignores_a_decay_or_a_flat_price_in_the_history():
    """Only a RAISE arms the hold — a history that only ever fell (or stood
    still) leaves the decay ladder exactly as it was."""
    now = 1_000_000.0
    fell = _rung_then_decay(
        60.0, bid_history=_hist((now - 600.0, 0.40), (now - 60.0, 0.30)))
    assert _recent_raise_hold(fell) is None
    flat = _rung_then_decay(60.0, bid_history=_hist((now - 60.0, 0.30)))
    assert _recent_raise_hold(flat) is None


# --- 4a-ter. the dwells are DURATIONS, not poll counts -----------------------
_decay_dwell_satisfied = bidpolicy._decay_dwell_satisfied
_next_decay_state = bidpolicy.next_decay_state


def _run_decay_streak(tick_s, ticks, t0=1_000_000.0):
    """Drive the streak the way a lane does — advance `next_decay_state` at the
    top of each tick, THEN let `_bid_action` read it on that same tick — and
    return the state as the `ticks`-th consecutive candidate poll sees it. N
    consecutive polls therefore span N-1 intervals, which is what the poll COUNT
    always meant."""
    s = _decaying(now=t0, decay_streak=0)
    s["decay_streak_since"] = None
    for i in range(ticks):
        s = dict(s, now=t0 + i * tick_s)
        streak, since = _next_decay_state(s)
        s["decay_streak"], s["decay_streak_since"] = streak, since
    return s


def test_decay_dwell_is_unchanged_at_the_45s_tick():
    """The dwell the count bought: the third consecutive candidate poll, 90 s
    after the first. Same poll, same tick, before and after."""
    assert bidpolicy.BID_DECAY_S == 90.0
    assert _bid_action(_run_decay_streak(45.0, 2)) is None            # 45 s
    assert _bid_action(_run_decay_streak(45.0, 3)) == Action("lower_bid", "decay:0.24")


def test_decay_dwell_does_not_get_3x_more_aggressive_at_a_15s_tick():
    """The defect a COUNT carries: at a 15 s tick three polls span 30 s, so the
    ratified 90 s dwell would have become 30 s the day the tick shortened. As a
    duration it still takes 90 s — seven polls, not three."""
    assert _bid_action(_run_decay_streak(15.0, 3)) is None            # 30 s
    assert _bid_action(_run_decay_streak(15.0, 6)) is None            # 75 s
    assert _bid_action(_run_decay_streak(15.0, 7)) == Action("lower_bid", "decay:0.24")


def test_one_non_candidate_poll_still_clears_the_run():
    """J1's anti-ratchet: the RESET is the load-bearing half of the dwell, and
    it survives the move to timestamps — the clock restarts, it does not
    accumulate."""
    s = _run_decay_streak(45.0, 2)
    quiet = dict(s, market_min_bid=0.75)          # 1.2*0.75 = 0.90 == our bid
    assert _next_decay_state(quiet) == (0, None)
    resumed = dict(quiet, decay_streak=0, decay_streak_since=None,
                   market_min_bid=0.20, now=s["now"] + 45.0)
    streak, since = _next_decay_state(resumed)
    assert streak == 1 and since == resumed["now"]      # a fresh 90 s, not 45 s
    assert _bid_action(dict(resumed, decay_streak=streak,
                            decay_streak_since=since)) is None


def test_decay_dwell_falls_back_to_the_poll_count_without_a_timestamp():
    """Backward compatibility with the running daemon's state files, which carry
    `decay_streak` and no `decay_streak_since`."""
    assert _decay_dwell_satisfied(_decaying(decay_streak=2)) is False
    assert _decay_dwell_satisfied(_decaying(decay_streak=3)) is True
    assert _bid_action(_decaying(decay_streak=3)) == Action("lower_bid", "decay:0.24")


def test_handoff_dwell_is_a_duration_too():
    """Same defect, same fix, on the gate that rents a SECOND box: five 45 s
    polls put the ARM 180 s after the first over-ceiling read."""
    assert bidpolicy.HANDOFF_DWELL_S == 180.0
    armable = dict(_hs(), over_ceiling_streak=1)   # `_hs` is section 5's fixture
    assert handoff_poll(dict(armable, over_ceiling_since=armable["now"] - 179.0)
                        ).kind == "noop"
    assert handoff_poll(dict(armable, over_ceiling_since=armable["now"] - 180.0)
                        ).kind == "arm"
    # ...and a state with no timestamp keeps the legacy 5-poll count
    assert handoff_poll(dict(armable, over_ceiling_streak=4)).kind == "noop"
    assert handoff_poll(dict(armable, over_ceiling_streak=5)).kind == "arm"


# --- 4b. --max-bid default derives from the observed floor, not a spike ------
def test_default_max_bid_from_rolling_median_floor():
    # 3x median([0.10, 0.11, 0.12]) = 3 * 0.11 = 0.33
    assert _default_max_bid([0.10, 0.11, 0.12], 5.0) == 0.33


def test_default_max_bid_median_ignores_a_transient_spike():
    # a single spiked read cannot anchor the cap high: median stays low
    # (median([0.10, 0.11, 9.99]) = 0.11 -> 3x = 0.33; a mean would balloon to ~10)
    assert _default_max_bid([0.10, 0.11, 9.99], None) == 0.33


def test_default_max_bid_falls_back_to_first_seen_before_any_floor_read():
    assert _default_max_bid([], 0.40) == 0.5           # 1.25 * 0.40
    assert _default_max_bid(None, 0.40) == 0.5


def test_default_max_bid_none_when_no_floor_and_no_dph():
    assert _default_max_bid([], None) is None


def test_default_max_bid_prefers_floor_over_first_seen_dph():
    # once a floor read exists it governs, NOT the (possibly spiked) first-seen dph
    assert _default_max_bid([0.10], 9.99) == 0.3       # 3 * 0.10, not 1.25*9.99


# --- 4b'. on-demand-anchored ceiling (AUTOBID_DESIGN) -----------------------
def test_default_max_bid_get_and_hold_is_just_under_on_demand():
    # on-demand known -> default (get-and-hold) hard cap = on_demand - 1 rounding unit
    # (0.001, NOT a full cent — a cheap GPU's floor sits <1c under on-demand), and it
    # WINS over the floor-median fallback
    assert _default_max_bid([], None, on_demand=1.00) == 0.999
    assert _default_max_bid([0.10, 0.11, 0.12], 5.0, on_demand=1.00) == 0.999


def test_default_max_bid_strict_ceiling_is_half_on_demand():
    # BID_CEILING_ONDEMAND_FRAC moved 0.50 -> 0.75 (2026-08-08): it must stay
    # strictly ABOVE the standing-bid target (0.65 x on-demand) or every fresh
    # launch breaches it and handoff dead-arms.
    assert _default_max_bid([], None, on_demand=1.00, strict_ceiling=True) == 0.75
    assert _default_max_bid([0.40], 5.0, on_demand=1.00, strict_ceiling=True) == 0.75


def test_default_max_bid_falls_back_to_median_when_on_demand_unknown():
    # on_demand None -> the J1 anti-ratchet median path is intact
    assert _default_max_bid([0.10, 0.11, 9.99], None, on_demand=None) == 0.33


def test_bid_target_uses_1p2x_and_clamps_below_on_demand():
    # The name is literal again: BID_TARGET_MULT went 1.2 -> 2.00 (2026-08-08
    # displacement audit) -> 1.20 (owner ruling 2026-08-09, "pay near the
    # market"). BID_TARGET_MULT_UNPRICED (the no-on-demand preference below)
    # stayed 1.20 throughout — an aggressive multiple is only safe under the
    # on-demand cost cap, and with no on-demand read there is no cap.
    assert _bid_target(0.50, 1.00) == 0.60              # 1.2 * 0.50, under max_bid
    assert _bid_target(0.50, None) == 0.60              # uncapped legacy still finite
    # The on-demand rails bind even when max_bid is higher than on-demand.
    # 2026-08-09: the binding rail on a floor this close to on-demand (0.28/0.30 =
    # 93%) is no longer the `on_demand - 0.001` clamp — it is the HARD ceiling
    # (0.75 x 0.30 = $0.225, under the floor), so the bid is REFUSED rather than
    # placed at $0.299. Paying 99.7% of on-demand for a preemptible box was the
    # dominated outcome the recalibration removed.
    assert _bid_target(0.28, 5.0, on_demand=0.30) is None
    # a floor with real headroom is priced by the multiple, under the cost cap
    # and never by max_bid (2.00 era: 0.56)
    assert _bid_target(0.28, 5.0, on_demand=1.00) == 0.336   # 1.2 x floor, under the cap
    assert _bid_target(None, 1.0) is None


def test_preferred_ceiling_is_half_on_demand():
    assert bidpolicy._preferred_ceiling(1.00) == 0.75      # was 0.50 pre-2026-08-08
    assert bidpolicy._preferred_ceiling(None) is None


def test_preferred_ceiling_alarm_fires_only_when_live_bid_over_half_on_demand():
    over, pref = bidpolicy._preferred_ceiling_alarm(mk(
        present=True, actual_status="running", last_bid=0.85, on_demand=1.00))
    assert over is True and pref == 0.75
    # under the preferred line -> no alarm
    under, _ = bidpolicy._preferred_ceiling_alarm(mk(
        present=True, actual_status="running", last_bid=0.60, on_demand=1.00))
    assert under is False
    # strict-ceiling never alarms (the preferred line IS the hard cap there)
    strict, _ = bidpolicy._preferred_ceiling_alarm({
        "present": True, "actual_status": "running", "last_bid": 0.60,
        "on_demand": 1.00, "strict_ceiling": True})
    assert strict is False


def test_poll_hard_on_demand_clamp_holds_in_full_decision():
    # thin/hot market: 1.2*floor and max_bid both exceed on-demand -> target clamps
    # to just under on-demand; a live box at that clamp is stable (no raise)
    s = mk(view={"status": "running"}, present=True, actual_status="running",
           last_bid=0.299, max_bid=1.0, market_min_bid=0.28, on_demand=0.30,
           last_bid_put_ts=0.0, now=1_000_000.0)
    # target = min(1.2*0.28=0.336, 1.0, 0.30-0.001=0.299) = 0.299 == last_bid -> noop
    assert poll(s) == Action("noop", "live")


# --- 4c. missed-checkpoint watchdog -----------------------------------------
def _running_view(**over):
    now = 1_000_000.0
    v = {"job_id": "job-x", "display_status": "running", "status": "started",
         "checkpoint_s": 300, "n_checkpoints": 3,
         "started_at": _ts(now - 10_000), "last_resumed_ts": None,
         "last_checkpoint_ts": _ts(now - 200), "last_event": "checkpoint"}
    v.update(over)
    return v


def test_watchdog_silent_when_checkpoints_are_fresh():
    assert _ckpt_watchdog_alarm(_running_view(), 1_000_000.0) is None


def test_watchdog_fires_on_stale_checkpoint_silence():
    # last checkpoint 1200s ago, checkpoint_s=300 -> > 3x -> alarm
    v = _running_view(last_checkpoint_ts=_ts(1_000_000.0 - 1200))
    alarm = _ckpt_watchdog_alarm(v, 1_000_000.0)
    assert alarm is not None and "NO checkpoint" in alarm and "job-x" in alarm


def test_watchdog_uses_latest_of_resume_or_checkpoint_reference():
    # a fresh resume (200s ago) means the box is alive again -> NOT stale, even
    # though the last checkpoint is ancient.
    v = _running_view(last_checkpoint_ts=_ts(1_000_000.0 - 9000),
                      last_resumed_ts=_ts(1_000_000.0 - 200))
    assert _ckpt_watchdog_alarm(v, 1_000_000.0) is None


def test_watchdog_silent_for_non_running_jobs():
    v = _running_view(display_status="interrupted",
                      last_checkpoint_ts=_ts(1_000_000.0 - 9000))
    assert _ckpt_watchdog_alarm(v, 1_000_000.0) is None


def test_watchdog_silence_path_off_without_checkpoint_s():
    # a job that opted OUT of checkpointing never triggers the silence alarm
    v = _running_view(checkpoint_s=None,
                      last_checkpoint_ts=_ts(1_000_000.0 - 9000))
    assert _ckpt_watchdog_alarm(v, 1_000_000.0) is None


def test_watchdog_fires_on_explicit_sync_failed_event_regardless_of_ckpt_s():
    # the box-side checkpoint_sync_failed signal fires even without checkpoint_s
    v = _running_view(checkpoint_s=None, last_event="checkpoint_sync_failed")
    alarm = _ckpt_watchdog_alarm(v, 1_000_000.0)
    assert alarm is not None and "checkpoint_sync_failed" in alarm


def test_watchdog_fires_on_pure_silence_before_first_checkpoint():
    # dead key from the start: no checkpoint event ever, only started_at, long ago
    v = _running_view(n_checkpoints=0, last_checkpoint_ts=None,
                      started_at=_ts(1_000_000.0 - 5000), last_event="started")
    alarm = _ckpt_watchdog_alarm(v, 1_000_000.0)
    assert alarm is not None and "NO checkpoint" in alarm


def test_ts_to_epoch_roundtrip_and_junk():
    e = 1_700_000_000
    assert abs(_ts_to_epoch(_ts(e)) - e) < 1.0
    assert _ts_to_epoch("not-a-ts") is None
    assert _ts_to_epoch(None) is None
    assert _ts_to_epoch("") is None


# =============================================================================
# 5. Handoff pure decision core (T2; HANDOFF_DESIGN §6/§7). Layer-1 table:
#    one case per handoff_poll transition + every guardrail, plus the candidate
#    filter / headroom helpers and a no-mutation purity proof. poll() and its
#    tests above are untouched — handoff is a SEPARATE pure function.
# =============================================================================
NOW = 1_000_000.0


def _hs(**over):
    """Armable IDLE handoff state: dwell satisfied, a cheap-enough candidate, fat
    budget headroom. Each test overrides exactly the fields it exercises."""
    base = dict(
        phase="IDLE", over_ceiling_streak=bidpolicy.HANDOFF_DWELL_POLLS,
        primary_dph=1.00, primary_on_demand=1.10,
        candidate_min_bid=0.20, candidate_on_demand=1.00,
        remaining_wall_h=24.0, budget_usd=100.0, spend_usd=0.0,
        handoffs_done=0, cooldown_until=0.0, now=NOW)
    base.update(over)
    return mk_hs(**base)


# --- 5a. candidate filter (§2.3) and headroom gate (§1/§3) -------------------
def test_handoff_candidate_ok_when_cheap_and_amortizes():
    # understudy target 1.2*0.20=0.24 <= 0.5*1.10 primary pref; savings >> overhead
    assert _handoff_candidate_ok(1.00, 0.20, 1.00, 24.0, 1.10) is True


def test_handoff_candidate_rejected_when_over_primary_line():
    # candidate target = _bid_target(1.20, None, 2.00) = $1.32 (cushion-raised
    # over the 0.65 cost cap) > 0.75 x 1.10 = $0.825: the migration would not
    # land us under the line we are escaping -> reject
    assert _handoff_candidate_ok(1.00, 1.20, 2.00, 24.0, 1.10) is False
    # the pre-2026-08-09 fixture for the same rule (floor $0.80 at 80% of a $1.00
    # on-demand) now refuses one rail EARLIER — the candidate is un-priceable at
    # all under the hard ceiling — so it is kept as a second, stronger case.
    assert _handoff_candidate_ok(1.00, 0.80, 1.00, 24.0, 1.10) is False
    assert bidpolicy._bid_target(0.80, None, 1.00) is None


def test_handoff_candidate_rejected_in_the_floors_track_od_market():
    """**Reversed 2026-08-09** (recalibration item A) — and it is the D10
    correction landing in code.

    This case was written from the live 2026-07-15 observation that every idle
    offer priced `min_bid ~= 0.98 x its OWN dph_total`, and it asserted that a
    candidate floor of $0.085 against a $0.087 on-demand rate MUST pass, escaping
    a $0.244-od primary. AUTOBID_AUDIT_2026-08-08 §2 (defect D10) then established
    that the 0.98 observation was itself the doc-50-R1 defect measuring itself:
    it read the BID-view `dph_total`, which is the interruptible price, not
    on-demand. Real floors sit near HALF of on-demand.

    So this row is not a market regime — it is a corrupted measurement, and under
    the hard ceiling the pair it describes (floor at 98% of on-demand) is exactly
    the machine we must not bid on: 0.75 x $0.087 = $0.065 is under the floor, so
    no survivable bid exists. Migrating ONTO such a box would buy the same
    razor-thin eviction the whole audit was about.

    The genuine version of the same escape — a cheap candidate under the primary's
    line — is asserted alongside so the feature is not merely turned off."""
    assert _handoff_candidate_ok(0.24, 0.085, 0.087, 2.0, 0.2444) is False
    assert bidpolicy._bid_target(0.085, None, 0.087) is None
    # the honest form: the same primary, a candidate whose floor is the measured
    # ~45% of ITS own on-demand rate. Target $0.072 (1.2 x the $0.060 floor —
    # since the 2026-08-09 return to a 1.20x multiple the MULTIPLE prices it;
    # in the 2.00 era the 0.65 cost cap did, at $0.086), comfortably under the
    # $0.183 line we are escaping, and it passes.
    assert bidpolicy._bid_target(0.060, None, 0.1325) == pytest.approx(0.072)
    assert _handoff_candidate_ok(0.24, 0.060, 0.1325, 2.0, 0.2444) is True


def test_handoff_candidate_rejected_when_amortization_fails_short_run():
    # cheap enough (0.24 <= 0.55) but too little wall left to amortize the 2x window
    assert _handoff_candidate_ok(1.00, 0.20, 1.00, 0.3, 1.10) is False


def test_handoff_candidate_rejected_on_missing_market_read():
    assert _handoff_candidate_ok(1.00, None, 1.00, 24.0, 1.10) is False
    assert _handoff_candidate_ok(None, 0.20, 1.00, 24.0, 1.10) is False
    # primary on-demand unknown -> the escape line can't be anchored -> refuse
    assert _handoff_candidate_ok(1.00, 0.20, 1.00, 24.0, None) is False


# --- 5a-bis. the 2026-08-08 FABRICATED-HORIZON incident (defect #63) ---------
# The amortization inequality is only as honest as `remaining_wall_h`, and that
# number was invented. `job_supervise_tick` read a flat 24.0 h whenever
# --wall-budget was unset, and fleetd's JOBS_POLICY_DEFAULTS seed
# `wall_budget=None` — so EVERY jobs watch under fleetd priced its migration
# against a full day of runway nothing had measured. On 2026-08-08 that armed a
# VOLUNTARY handoff on a running, healthy box roughly 90 s from the end of a
# cell; the ticket carried `n_checkpoints: 0`, so the migration threw the work
# away and the job restarted from zero. The numbers below are that box's.
_INCIDENT = dict(primary_dph=0.830,           # the box we were escaping
                 candidate_min_bid=0.3333,    # cheapest qualifying offer's floor
                 candidate_on_demand=0.9333,
                 primary_on_demand=1.2667)    # -> preferred ceiling $0.633


def _pending_view(**over):
    """A folded jobs view of the shape `job_supervise_tick` reads off the queue
    (jobmeta.fold_events' field set, trimmed to what the horizon reads)."""
    v = {"job_id": "job-x", "status": "started", "display_status": "running",
         "timeout_s": 4200, "started_at": _ts(NOW - 1500), "last_resumed_ts": None}
    v.update(over)
    return v


def test_handoff_candidate_flips_on_the_horizon_alone():
    """Characterisation of the incident arithmetic — the SAME offer, the SAME
    market, decided entirely by a number no one measured. target =
    1.2 x 0.3333 = $0.400, which clears the $0.633 line it is escaping either
    way, so gate (1) is not what moves; overhead = (0.830 + 0.400) x 0.5 =
    $0.615. At the fabricated 24 h the migration projects $10.32 of savings and
    ARMS; at the ticket's real 0.75 h it projects $0.32 and must refuse."""
    assert _handoff_candidate_ok(remaining_wall_h=24.0, **_INCIDENT) is True
    assert _handoff_candidate_ok(remaining_wall_h=0.75, **_INCIDENT) is False


def test_jobs_horizon_is_bounded_by_the_ticket_timeout_not_a_default_day():
    """P0-a. With no --wall-budget the horizon is what the PENDING TICKETS can
    still run: the incident's single job declared timeout_s=4200 and was 25 min
    in, so 2700 s == 0.75 h. Fed to the same filter, the migration is refused."""
    v = _pending_view(timeout_s=4200, started_at=_ts(NOW - 1500))
    rwh = jobs_risk._jobs_remaining_wall_h([v], NOW)
    assert rwh == pytest.approx(0.75)
    assert _handoff_candidate_ok(remaining_wall_h=rwh, **_INCIDENT) is False


def test_jobs_horizon_gives_a_queued_ticket_its_whole_timeout():
    # nothing has started burning it yet, so the full declared budget is ahead
    v = _pending_view(display_status="queued", status="submitted",
                      started_at=None, timeout_s=3600)
    assert jobs_risk._jobs_remaining_wall_h([v], NOW) == pytest.approx(1.0)


def test_jobs_horizon_sums_every_pending_ticket():
    # the box keeps working until the LAST ticket is done, so the horizon is a
    # sum, not a max: 2700 s left on the running one + 3600 s queued = 1.75 h.
    views = [_pending_view(job_id="job-a", timeout_s=4200,
                           started_at=_ts(NOW - 1500)),
             _pending_view(job_id="job-b", display_status="queued",
                           status="submitted", started_at=None, timeout_s=3600)]
    assert jobs_risk._jobs_remaining_wall_h(views, NOW) == pytest.approx(1.75)


def test_jobs_horizon_prefers_the_resume_over_the_first_attempt():
    # jobd re-runs the entrypoint under a FRESH `timeout $JOB_TIMEOUT_S` on every
    # attempt (onstart/jobd.sh), and the fold's `started_at` is min(claimed,
    # started) — the FIRST attempt, which after a preemption is long stale. The
    # current attempt's clock starts at `last_resumed_ts`.
    v = _pending_view(timeout_s=4200, started_at=_ts(NOW - 40_000),
                      last_resumed_ts=_ts(NOW - 1500))
    assert jobs_risk._jobs_remaining_wall_h([v], NOW) == pytest.approx(0.75)


def test_jobs_horizon_clamps_an_over_timeout_straggler_at_zero():
    # a job past its own timeout is about to be killed; it must contribute 0,
    # never a NEGATIVE that eats another ticket's genuine runway.
    views = [_pending_view(job_id="job-a", timeout_s=600,
                           started_at=_ts(NOW - 100_000)),
             _pending_view(job_id="job-b", display_status="queued",
                           status="submitted", started_at=None, timeout_s=3600)]
    assert jobs_risk._jobs_remaining_wall_h(views, NOW) == pytest.approx(1.0)


def test_jobs_horizon_takes_the_min_of_the_wall_budget_and_the_queue():
    v = _pending_view(display_status="queued", status="submitted",
                      started_at=None, timeout_s=4 * 3600)
    assert jobs_risk._jobs_remaining_wall_h([v], NOW,
                                          wall_remaining_h=1.5) == pytest.approx(1.5)
    # ...and the queue still wins when IT is the tighter bound
    assert jobs_risk._jobs_remaining_wall_h([v], NOW,
                                          wall_remaining_h=9.0) == pytest.approx(4.0)


def test_jobs_horizon_is_none_when_nothing_yields_a_bound():
    """Fail CLOSED. A ticket with no declared timeout, a running ticket with no
    attempt timestamp, and an empty queue all leave the cost unbounded — and a
    None horizon has to make the candidate filter refuse, because a voluntary
    handoff is an optimisation: refusing costs a missed saving, never work."""
    assert jobs_risk._jobs_remaining_wall_h([], NOW) is None
    assert jobs_risk._jobs_remaining_wall_h([_pending_view(timeout_s=None)], NOW) is None
    assert jobs_risk._jobs_remaining_wall_h(
        [_pending_view(started_at=None, last_resumed_ts=None)], NOW) is None
    assert _handoff_candidate_ok(remaining_wall_h=None, **_INCIDENT) is False


def test_jobs_horizon_skips_an_unreadable_ticket_without_losing_the_others():
    # one bad row must not fabricate runway for the rest, nor erase theirs
    views = [_pending_view(job_id="job-a", timeout_s=None),
             _pending_view(job_id="job-b", timeout_s=4200,
                           started_at=_ts(NOW - 1500))]
    assert jobs_risk._jobs_remaining_wall_h(views, NOW) == pytest.approx(0.75)


def test_handoff_headroom_ok_with_fat_budget():
    # projected_2x = (1.00+0.24)*0.5 = 0.62; 100-0 > 0.62
    assert _handoff_headroom_ok(100.0, 0.0, 1.00, 0.24) is True


def test_handoff_headroom_refused_on_thin_budget():
    # only 0.10 left, projected_2x 0.62 -> refuse
    assert _handoff_headroom_ok(1.0, 0.9, 1.00, 0.24) is False


def test_handoff_headroom_refused_without_candidate_read():
    assert _handoff_headroom_ok(100.0, 0.0, 1.00, None) is False


def test_handoff_headroom_unbounded_when_no_budget_cap():
    assert _handoff_headroom_ok(None, 0.0, 1.00, 0.24) is True


# --- 5b. IDLE arming ladder: dwell, cooldown, cap, refusals -----------------
def test_handoff_idle_no_pressure_is_noop():
    assert handoff_poll(_hs(over_ceiling_streak=0)) == HandoffAction("noop", "idle")


def test_handoff_dwell_not_yet_met_is_noop():
    # 4 consecutive over-polls: still dwelling, no ARM (the 5th arms)
    assert handoff_poll(_hs(over_ceiling_streak=4)) == HandoffAction("noop", "dwell")


def test_handoff_arms_on_fifth_over_ceiling_poll():
    assert handoff_poll(_hs(over_ceiling_streak=5)) == \
        HandoffAction("arm", "dwell_satisfied")


def test_handoff_cooldown_suppresses_arm():
    assert handoff_poll(_hs(cooldown_until=NOW + 1.0)) == \
        HandoffAction("noop", "cooldown")


def test_handoff_max_exhaustion_falls_back_to_get_and_hold():
    assert handoff_poll(_hs(handoffs_done=bidpolicy.HANDOFF_MAX)) == \
        HandoffAction("noop", "max_handoffs")


def test_handoff_arm_refused_on_thin_headroom():
    assert handoff_poll(_hs(budget_usd=1.0, spend_usd=0.9)) == \
        HandoffAction("noop", "headroom")


def test_handoff_arm_refused_on_candidate_filter():
    # cheap-enough headroom passes (target computable) but the candidate itself is
    # not get-and-hold viable -> candidate_reject.
    # 2026-08-09: the old fixture (floor 0.80 / od 1.00) is now UN-PRICEABLE under
    # the hard ceiling, so it refuses one gate earlier, on `headroom` — which
    # cannot bound a 2x window it has no candidate price for. Both refusals are
    # correct; this test wants the CANDIDATE gate, so the candidate is priceable
    # (floor at 60% of its own on-demand) and simply too expensive: target $1.32
    # against the $0.825 line we are escaping.
    assert handoff_poll(_hs(candidate_min_bid=1.20, candidate_on_demand=2.00)) == \
        HandoffAction("noop", "candidate_reject")
    assert handoff_poll(_hs(candidate_min_bid=0.80, candidate_on_demand=1.00)) == \
        HandoffAction("noop", "headroom")


# --- 5c. phase advance (§2.1) -----------------------------------------------
def test_handoff_armed_launches_understudy():
    assert handoff_poll(_hs(phase="ARMED", handoff_started_ts=NOW)) == \
        HandoffAction("launch_understudy", "armed")


def test_handoff_launching_waits_to_warm_before_checkpoint():
    hs = _hs(phase="LAUNCHING", handoff_started_ts=NOW, ckpt_pulled_epoch=None)
    assert handoff_poll(hs) == HandoffAction("warm_wait", "booting")


def test_handoff_marks_synced_once_checkpoint_pulled():
    hs = _hs(phase="WARMING", handoff_started_ts=NOW, ckpt_pulled_epoch=7)
    assert handoff_poll(hs) == HandoffAction("mark_synced", "checkpoint_pulled")


def test_handoff_synced_fences_the_primary():
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=7)
    assert handoff_poll(hs) == HandoffAction("fence_primary", "synced")


def test_handoff_cutover_blocks_resume_until_final_flush_seen():
    # fence gating: final_flush_seen=False never yields resume_understudy
    hs = dict(_hs(phase="CUTOVER"))
    hs["final_flush_seen"] = False
    assert handoff_poll(hs) == HandoffAction("noop", "await_flush")


def test_handoff_cutover_resumes_understudy_after_flush():
    hs = dict(_hs(phase="CUTOVER"))
    hs["final_flush_seen"] = True
    assert handoff_poll(hs) == HandoffAction("resume_understudy", "post_flush")


# --- F1: CUTOVER flush-timeout escape hatch ---------------------------------
def test_handoff_cutover_flush_timeout_resumes_understudy():
    # fence stood HANDOFF_FENCE_TIMEOUT_S with no flush (SIGKILL park / primary
    # already terminal) -> proceed from the last SYNCED checkpoint.
    hs = dict(_hs(phase="CUTOVER"))
    hs["final_flush_seen"] = False
    hs["fence_ts"] = NOW - bidpolicy.HANDOFF_FENCE_TIMEOUT_S
    assert handoff_poll(hs) == HandoffAction("resume_understudy", "flush_timeout")


def test_handoff_cutover_before_flush_timeout_still_awaits():
    hs = dict(_hs(phase="CUTOVER"))
    hs["final_flush_seen"] = False
    hs["fence_ts"] = NOW - (bidpolicy.HANDOFF_FENCE_TIMEOUT_S - 1)
    assert handoff_poll(hs) == HandoffAction("noop", "await_flush")


def test_handoff_cutover_no_fence_ts_never_timeouts():
    # fence_ts None (defensive: reconciled into CUTOVER without a fence stamp) must
    # not crash and must keep awaiting the flush, never a spurious timeout.
    hs = dict(_hs(phase="CUTOVER"))
    hs["final_flush_seen"] = False
    hs["fence_ts"] = None
    assert handoff_poll(hs) == HandoffAction("noop", "await_flush")


# --- 5b-bis. the 2026-08-08 22:17Z incident: work-awareness rails -------------
# Box 47214941 (H200, jobs profile, one RUNNING eval ticket four minutes into
# setup). The handoff armed at 22:17:41 on `primary_bid 3.41` vs `on_demand
# 3.876`, fenced the RUNNING job at 22:21:17, parked the primary and pinned its
# standing bid to $0.001, and rented an understudy that fleetd then had no watch
# for. Every number below is that box's, read from the fleetd journal.
_FENCE_INCIDENT = dict(primary_bid=3.41, on_demand=3.876315789473685,
                       candidate_min_bid=1.3333333333333333,
                       remaining_wall_h=9.904633606142468,
                       pinned_bid=0.001)


def test_handoff_refuses_to_arm_when_the_driver_cannot_complete():
    """Defect #61. Under fleetd the jobs watch ends at `inst is None` two ticks
    after the primary is destroyed — one tick BEFORE handoff_poll can return
    `complete` — so the understudy inherited no watch and no budget cap. A
    feature that cannot finish must not be allowed to start."""
    assert handoff_poll(_hs(driver_can_complete=False)) == \
        HandoffAction("noop", "precondition:driver_cannot_complete")
    # ...and the same state with a driver that CAN finish still arms, so the
    # refusal is the new field and nothing else.
    assert handoff_poll(_hs(driver_can_complete=True)) == \
        HandoffAction("arm", "dwell_satisfied")


def test_handoff_refuses_to_arm_over_an_uncheckpointed_running_job():
    """Defect #62. Arming rents a second box to reach a fence we already know we
    must not open — refuse before the money moves, not just at the fence."""
    assert handoff_poll(_hs(running_unresumable=1)) == \
        HandoffAction("noop", "precondition:unresumable_running_job")


def test_handoff_unsafe_override_restores_the_old_arming_but_is_named():
    """The escape hatch buys back the OLD arm economics and nothing else."""
    hs = _hs(driver_can_complete=False, running_unresumable=1,
             unsafe_override=True)
    assert handoff_poll(hs) == HandoffAction("arm", "dwell_satisfied")


def test_handoff_fence_hold_refuses_to_park_an_uncheckpointed_running_job():
    """Defect #62 at the irreversible step. `n_checkpoints: 0` on the running
    ticket is exactly what the incident fenced over, and the work was lost."""
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=1,
             running_unresumable=1)
    assert handoff_poll(hs) == \
        HandoffAction("noop", "fence_hold:no_resumable_checkpoint")


def test_handoff_fence_holds_when_the_running_job_is_nearly_done():
    """Task #67. ARM is not enough: the incident armed at 22:17:41 and fenced at
    22:21:17, and the cell became ~90 s from done inside that 216-second gap."""
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=1,
             min_running_eta_s=90.0)
    assert handoff_poll(hs) == HandoffAction("noop", "fence_hold:eta_90s")
    # comfortably far from done -> the fence opens as before
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=1,
             min_running_eta_s=bidpolicy.HANDOFF_FENCE_HOLD_ETA_S + 1)
    assert handoff_poll(hs) == HandoffAction("fence_primary", "synced")


def test_handoff_fence_hold_eta_is_tri_state_not_zero():
    """An UNKNOWN ETA is not a near-zero one. None must never hold the fence
    (the ARM-side horizon gate is what refuses on unknowns, before a second box
    exists) and must never be coerced to 0 or to infinity."""
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=1,
             min_running_eta_s=None)
    assert handoff_poll(hs) == HandoffAction("fence_primary", "synced")


def test_handoff_fence_holds_on_a_stale_checkpoint():
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=1,
             ckpt_stale=True)
    assert handoff_poll(hs) == HandoffAction("noop", "fence_hold:checkpoint_stale")


def test_handoff_fence_hold_is_not_skippable_by_the_unsafe_override():
    """The override is about ARM economics. It never licenses parking a box on
    top of work that cannot come back."""
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=1,
             running_unresumable=1, unsafe_override=True)
    assert handoff_poll(hs) == \
        HandoffAction("noop", "fence_hold:no_resumable_checkpoint")


def test_handoff_fence_hold_is_bounded_by_the_existing_deadline():
    """No new timer: SYNCED is a pre-CUTOVER phase, so a hold that never clears
    is aborted by precedence 2 and the understudy is reaped."""
    hs = _hs(phase="SYNCED", ckpt_pulled_epoch=1, running_unresumable=1,
             handoff_started_ts=NOW - bidpolicy.HANDOFF_DEADLINE_S)
    assert handoff_poll(hs) == HandoffAction("abort_reap", "deadline")


def test_open_fence_unwinds_and_never_outlives_its_window():
    """Task #62(b). CUTOVER's only exits were `resume_understudy` and a
    `retarget_incomplete` latch that returns and stays put, so a cutover whose
    old-ticket delete failed left the primary parked and pinned at $0.001 —
    the incident's exact configuration — with no path back. Past
    HANDOFF_FENCE_UNWIND_S the fence unwinds: tickets back, bid restored, box
    resumed."""
    hs = dict(_hs(phase="CUTOVER"))
    hs["final_flush_seen"] = True                     # flush seen; cutover wedged
    hs["fence_ts"] = NOW - bidpolicy.HANDOFF_FENCE_UNWIND_S
    assert handoff_poll(hs) == HandoffAction("abort_unfence", "fence_unwind")
    # one second earlier the normal cutover still owns the phase
    hs["fence_ts"] = NOW - (bidpolicy.HANDOFF_FENCE_UNWIND_S - 1)
    assert handoff_poll(hs) == HandoffAction("resume_understudy", "post_flush")


def test_work_at_risk_is_priced_into_the_amortization():
    """Task #67. Redoing discarded work costs primary_dph an hour on whatever
    box redoes it, so it belongs in the overhead in the same dollars as the 2x
    window. Deliberately not a hard rule: the same run with a 10 h horizon still
    migrates, because restarting an hour to save $0.60/hr for ten IS the right
    trade."""
    base = dict(primary_dph=1.00, candidate_min_bid=0.20,
                candidate_on_demand=1.00, primary_on_demand=1.10)
    # 1.5 h left, no work at risk: savings (1.00-0.24)*1.5 = $1.14 > overhead
    # (1.24*0.5) = $0.62 -> migrate.
    assert _handoff_candidate_ok(remaining_wall_h=1.5, **base) is True
    # the SAME migration with an hour of uncheckpointed work at risk adds $1.00
    # of overhead and stops paying for itself.
    assert _handoff_candidate_ok(remaining_wall_h=1.5, work_at_risk_h=1.0,
                                 **base) is False
    # ...and a long horizon still clears it.
    assert _handoff_candidate_ok(remaining_wall_h=10.0, work_at_risk_h=1.0,
                                 **base) is True


# --- 5b-ter. the trigger domain: our own policy bid must not arm a handoff ----
def _trigger_state(**over):
    """A LIVE bid box priced exactly as the 2026-08-08 22:17Z incident's was."""
    base = dict(view={"status": "running"}, present=True, actual_status="running",
                last_bid=_FENCE_INCIDENT["primary_bid"],
                market_min_bid=3.10,                  # the floor the ladder was handed
                max_bid=3.875, on_demand=_FENCE_INCIDENT["on_demand"],
                last_bid_put_ts=0.0, now=NOW)
    base.update(over)
    return mk(**base)


def test_incident_bid_is_over_the_preferred_ceiling_and_still_must_not_arm():
    """The trigger-domain ruling. $3.41 really was over the $2.907 preferred
    ceiling — the ALARM is correct and stays. Under the 2026-08-08 rails $3.41 was
    also exactly `_bid_target(3.10, 3.875, 3.876)`: the survival cushion
    (1.10 x floor) outranked the 0.65 x on-demand cost cap on a tight machine, so
    the bid the handoff called excessive was the bid the policy had just decided
    to pay, and it did not arm — reason `at_policy_target`.

    **2026-08-09 (recalibration item A) makes the same verdict stronger.** The
    preferred line is now a HARD clamp, so the policy cannot place $3.41 at all:
    `_bid_target` escalates instead. The trigger therefore refuses one rail
    earlier, on `no_policy_target` — the same fail-closed direction, for the
    better reason that there is no legal bid on this machine and the answer is the
    replacement/on-demand rung, not a second box.

    Either way the ruling holds: a bid the policy would not raise cannot by itself
    arm a migration."""
    s = _trigger_state()
    over, pref = bidpolicy._preferred_ceiling_alarm(s)
    assert over is True and pref == pytest.approx(2.907, abs=1e-3)
    dec = bidpolicy.bid_decision(3.10, 3.875,
                                         _FENCE_INCIDENT["on_demand"])
    assert dec.price is None and dec.escalate is True
    assert dec.ceiling == pytest.approx(2.907, abs=1e-3)
    armed, pref2, target, reason = bidpolicy._handoff_trigger(s)
    assert armed is False and reason == "no_policy_target"
    assert pref2 == pytest.approx(2.907, abs=1e-3)
    assert target is None


def test_trigger_arms_when_the_bid_really_is_above_the_policy_target():
    """The feature is not dead: a bid the policy would NOT put today (a stale
    ratchet the decay ladder has not yet walked back, an operator --max-bid held
    over a collapsed floor) is still a genuine handoff trigger."""
    s = _trigger_state(market_min_bid=0.40)           # floor collapsed under us
    armed, pref, target, reason = bidpolicy._handoff_trigger(s)
    assert armed is True and reason == "over_policy_target"
    assert target is not None and target < _FENCE_INCIDENT["primary_bid"]


def test_trigger_is_silent_under_the_preferred_ceiling():
    s = _trigger_state(last_bid=1.00, market_min_bid=0.40)
    assert bidpolicy._handoff_trigger(s)[0] is False
    assert bidpolicy._handoff_trigger(s)[3] == "under_pref"


def test_trigger_refuses_when_the_policy_target_is_unreadable():
    """No market read this tick means we cannot tell a policy bid from a stale
    one — fail closed, the same direction as every other handoff gate."""
    s = _trigger_state(market_min_bid=None)
    armed, _pref, target, reason = bidpolicy._handoff_trigger(s)
    assert armed is False and target is None and reason == "no_policy_target"


def test_handoff_draining_waits_for_understudy_to_produce():
    hs = dict(_hs(phase="DRAINING"))
    hs["understudy_producing"] = False
    assert handoff_poll(hs) == HandoffAction("noop", "await_understudy_ckpt")


def test_handoff_draining_destroys_primary_once_understudy_producing():
    hs = dict(_hs(phase="DRAINING"))
    hs["understudy_producing"] = True
    hs["primary_gone"] = False
    assert handoff_poll(hs) == HandoffAction("drain_primary", "understudy_producing")


def test_handoff_completes_after_primary_gone():
    hs = dict(_hs(phase="DRAINING"))
    hs["understudy_producing"] = True
    hs["primary_gone"] = True
    assert handoff_poll(hs) == HandoffAction("complete", "drained")


# --- 5d. guardrails: deadline abort + primary-evicted rows (§3/§5) ----------
@pytest.mark.parametrize("phase", ["ARMED", "LAUNCHING", "WARMING", "SYNCED"])
def test_handoff_deadline_aborts_every_pre_cutover_phase(phase):
    hs = _hs(phase=phase, handoff_started_ts=NOW - bidpolicy.HANDOFF_DEADLINE_S,
             ckpt_pulled_epoch=7)
    assert handoff_poll(hs) == HandoffAction("abort_reap", "deadline")


def test_handoff_no_deadline_abort_after_cutover():
    # past the deadline but already CUTOVER (primary fenced) -> not a pre-CUTOVER
    # abort; the machine proceeds on the fence instead
    hs = dict(_hs(phase="CUTOVER"))
    hs["handoff_started_ts"] = NOW - bidpolicy.HANDOFF_DEADLINE_S - 100
    hs["final_flush_seen"] = False
    assert handoff_poll(hs) == HandoffAction("noop", "await_flush")


def test_handoff_primary_evicted_pre_synced_aborts_and_reaps():
    hs = _hs(phase="LAUNCHING", handoff_started_ts=NOW, primary_evicted=True)
    assert handoff_poll(hs) == HandoffAction("abort_reap", "primary_evicted")


def test_handoff_primary_evicted_at_synced_fast_cutovers():
    # understudy already SYNCED + primary died -> promote it (skip relaunch),
    # and this WINS over the deadline (never abort to zero boxes)
    hs = _hs(phase="SYNCED", handoff_started_ts=NOW - bidpolicy.HANDOFF_DEADLINE_S,
             ckpt_pulled_epoch=7, primary_evicted=True)
    assert handoff_poll(hs) == HandoffAction("resume_understudy", "fast_cutover")


# --- 5e. purity -------------------------------------------------------------
def test_handoff_poll_is_pure_no_mutation():
    states = [
        _hs(over_ceiling_streak=0),
        _hs(over_ceiling_streak=5),
        _hs(phase="ARMED", handoff_started_ts=NOW),
        _hs(phase="LAUNCHING", handoff_started_ts=NOW),
        _hs(phase="SYNCED", handoff_started_ts=NOW, ckpt_pulled_epoch=7),
        _hs(phase="CUTOVER"),
        _hs(phase="CUTOVER", fence_ts=NOW - bidpolicy.HANDOFF_FENCE_TIMEOUT_S),
        _hs(phase="DRAINING", understudy_producing=True, primary_gone=True),
        _hs(phase="SYNCED", handoff_started_ts=NOW - bidpolicy.HANDOFF_DEADLINE_S,
            primary_evicted=True),
    ]
    for s in states:
        before = copy.deepcopy(s)
        handoff_poll(s)
        assert s == before


# =============================================================================
# 5f. Driver harness — the run-lane handoff scenario 1 (happy path), end-to-end
#
# MIGRATED (was MIGRATION-BLOCKED, step 6e): `Harness`-driven, and the harness
# moved — the five run-lane drivers (`_observe`, `_accrue_cost`, `_emit_cost`,
# `_do_bid_move`, `_supervise_boot_health`) are no longer raising stubs. Seams
# are stubbed where the run lane resolves them: the two policy reads as
# `bidpolicy.<name>`, the market read as `pricing.<name>`, the box-side handoff
# signals as `handoff.<name>`. The rows that RELAUNCH still pass `flat=True`
# (`_reset_run_markers`, unported — see the Harness docstring).
#     in dry-run (T4 acceptance gate; the full §5 fault matrix is T5).
#
#     arm -> launch understudy -> warm -> fence -> cutover-on-final_flush ->
#     understudy producing -> primary drained -> handoff_complete, asserting the
#     event sequence AND that both boxes' spend was counted.
# =============================================================================
def _hf_prim(run_id, iid=501, **over):
    inst = {"id": iid, "label": f"run:{run_id}", "actual_status": "running",
            "dph_total": 0.60, "machine_id": 555}
    inst.update(over)
    return inst


def _hf_twin(run_id, iid=777, **over):
    inst = {"id": iid, "label": f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}",
            "actual_status": "running", "dph_total": 0.12, "machine_id": 556}
    inst.update(over)
    return inst


def test_supervise_handoff_happy_path_end_to_end(monkeypatch, tmp_path):
    run_id = "run-handoff"
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)

    # the primary sits permanently over its 0.50x preferred ceiling (alarm forced
    # over so the dwell counter arms after HANDOFF_DWELL_POLLS consecutive polls;
    # on_demand faked so the §2.3 primary-relative escape line is anchored).
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm",
                        lambda st: (True, 0.30))
    # ...and the dwell input itself, which since 2026-08-08 is the TRIGGER (bid
    # over the policy target), not the alarm — see `_arm_over_ceiling`.
    monkeypatch.setattr(bidpolicy, "_handoff_trigger",
                        lambda st: (True, 0.30, 0.20, "over_policy_target"))
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, n=None: 0.62)
    # a genuinely cheaper offer that clears the §2.3 candidate + headroom gates.
    h.api.offers = (True, {"offers": [{"id": 999, "min_bid": 0.10,
                                       "dph_total": 0.50}]}, None)

    # box-side fence signals derived from the driver's own progress events: the
    # final flush becomes visible only AFTER the fence parked the primary, and the
    # understudy is 'producing' only AFTER the cutover relabelled it — this proves
    # the driver honours the two-writer fence ordering.
    def fake_signals(rid, cutover_ts=None):
        kinds = [e["event"] for e in h.events]
        return {"final_flush_seen": "handoff_fence" in kinds,
                "understudy_producing": "handoff_cutover" in kinds}
    monkeypatch.setattr(handoff, "_handoff_run_signals", fake_signals)
    # box-side .synced boot proof present as soon as asked (the observe gate
    # still requires the twin live in the instance snapshot).
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: 999)

    prim = (True, {"instances": [_hf_prim(run_id)]}, None)
    both = (True, {"instances": [_hf_prim(run_id), _hf_twin(run_id)]}, None)
    # [0] reconcile GET (primary only -> no twin to adopt), then ticks:
    #   t1-t6 primary only (dwell -> arm -> launch understudy),
    #   t7-t12 primary + twin (warm/sync/fence/cutover/drain/complete).
    h.api.instances_queue = [prim] + [prim] * 6 + [both] * 6
    h.api.instances_default = both
    h.view_queue = [{"status": "running"}] * 11        # t12 falls to default 'done'
    h.view_default = {"status": "done"}

    h.run(["--budget", "50", "--interval", "200", "--handoff", "--dry-run"])

    kinds = h.event_kinds()
    handoff_seq = [k for k in kinds if k.startswith("handoff_")]
    assert handoff_seq == ["handoff_armed", "handoff_launch", "handoff_synced",
                           "handoff_fence", "handoff_cutover", "handoff_complete"]

    # the understudy launched under the :handoff twin label (never the plain run:)
    launch = next(e for e in h.events if e["event"] == "handoff_launch")
    assert launch["label"] == f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}"
    assert launch["dph"] == 0.12                        # 1.2 * 0.10 min_bid
    # (0.20 during the 2026-08-08 2.00x era; 1.20 restored 2026-08-09)

    # complete fires ONLY after the fence + producing gates -> the destroy of the
    # primary happened after the understudy was confirmed producing (phase order).
    complete = next(e for e in h.events if e["event"] == "handoff_complete")
    assert complete["handoffs_done"] == 1

    # BOTH boxes' spend was counted: the 2x-box window billed the understudy into
    # the run's spend (against --budget), tracked as handoff_spend_usd (§3).
    assert complete["handoff_spend_usd"] > 0.0
    assert complete["spend_usd"] >= complete["handoff_spend_usd"]

    # a voluntary cost move, never an eviction: no evict/relaunch churn.
    assert "evicted" not in kinds and "relaunched" not in kinds
    assert h.events[-1]["event"] == "supervisor_exiting"
    assert h.events[-1]["reason"] == "terminal:done"


# =============================================================================
# 5g. Driver harness — the HANDOFF_DESIGN §5 fault matrix (T5).
#
# MIGRATED (was MIGRATION-BLOCKED, step 6e): `Harness`-driven, and the harness
# moved — the five run-lane drivers (`_observe`, `_accrue_cost`, `_emit_cost`,
# `_do_bid_move`, `_supervise_boot_health`) are no longer raising stubs. Seams
# are stubbed where the run lane resolves them: the two policy reads as
# `bidpolicy.<name>`, the market read as `pricing.<name>`, the box-side handoff
# signals as `handoff.<name>`. The rows that RELAUNCH still pass `flat=True`
# (`_reset_run_markers`, unported — see the Harness docstring).
#
#     One driver-level scenario per §5 row, built on the T4 harness. Unlike the
#     T4 happy path these run WITHOUT --dry-run, so every reap (DELETE), park
#     (PUT stopped) and relabel (PUT label) is a real, assertable API call — a
#     fault-injection suite that cannot see the reap it demands is only half a
#     net. Each scenario asserts, per the T5 brief:
#       * the TERMINAL outcome (which box survives, which is destroyed);
#       * the full handoff_ EVENT SEQUENCE (T4's real event names);
#       * SPEND accounting (both boxes billed in-window; no double-count; the
#         budget guard sees the true total);
#       * NO ORPHANED BOX (destroy/park actually issued on the understudy);
#       * FENCE ORDERING (no primary destroy / understudy write-enable before the
#         final_flush-gated cutover — except the explicit fast-cutover row).
#
#     Offer floor 0.10 / on-demand 0.50 clears the §2.3 candidate + headroom
#     gates against a 0.30-0.72 primary; _preferred_ceiling_alarm is forced over
#     so the dwell counter arms after HANDOFF_DWELL_POLLS consecutive polls.
# =============================================================================
HANDOFF_CONTRACT = 42424242          # FakeAPI.launch new_contract (understudy launch id)


def _resp(*insts):
    return (True, {"instances": list(insts)}, None)


def _arm_over_ceiling(monkeypatch, dwell_s=None):
    """Force the primary permanently over its 0.50x preferred ceiling so the dwell
    arms on the fifth consecutive poll (mirrors the happy path); fake the
    primary's on-demand read (anchors the §2.3 primary-relative escape line);
    fake the box-side .synced boot proof PRESENT (the observe gate still requires
    the twin live in the instance snapshot — rows that need a NEVER-synced twin
    re-patch _handoff_synced_epoch_soft to None themselves); return a
    cheap-enough offer that clears the §2.3 gates.

    `dwell_s` exists because the dwell is a DURATION now: these §5 rows script
    their instance queue per POLL and choose a long `--interval` to land a
    1800 s deadline on a given tick, so a row that wants the arm on poll five
    passes `4 * interval` and keeps the choreography it was written with. The
    dwell's own arithmetic is pinned by `test_handoff_dwell_is_a_duration_too`,
    not here."""
    # ONE namespace since step 6d, as the step-6e note here predicted ("drops to
    # the vastlib line alone when `_reset_run_markers` is ported"). Every row
    # drives `cli.main.main()`, which resolves the two policy reads as
    # `bidpolicy.<name>` and the rest at their vastlib owners; the second
    # `setattr(herdd, …)` each of these carried is now a rebind of a re-export
    # nothing reads — worse than redundant, because it reads like coverage.
    monkeypatch.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda st: (True, 0.30))
    # Since 2026-08-08 the DWELL reads `_handoff_trigger` (over the preferred
    # ceiling AND over what the bid policy would put right now), not the alarm —
    # see HANDOFF_DESIGN §11. These rows exercise the migration MACHINERY, so
    # they force the trigger the same way they force the alarm; the trigger's own
    # domain is pinned by the `_trigger_state` tests on the incident's numbers.
    monkeypatch.setattr(bidpolicy, "_handoff_trigger",
                        lambda st: (True, 0.30, 0.20, "over_policy_target"))
    if dwell_s is not None:
        monkeypatch.setattr(bidpolicy, "HANDOFF_DWELL_S", float(dwell_s))
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, n=None: 0.62)
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: 999)
    return (True, {"offers": [{"id": 999, "min_bid": 0.10, "dph_total": 0.50}]}, None)


def _fence_signals(h):
    """Box-side fence signals derived from the driver's OWN progress events (as in
    the happy path): the primary's final flush is visible only AFTER the fence
    parked it (or, on the fast-cutover row, once it is already gone); the
    understudy is 'producing' only AFTER the cutover relabelled it. Proves the
    driver honours the two-writer fence ordering rather than the test feeding it."""
    def sig(rid, cutover_ts=None):
        kinds = [e["event"] for e in h.events]
        return {"final_flush_seen": ("handoff_fence" in kinds
                                     or "handoff_cutover" in kinds),
                "understudy_producing": "handoff_cutover" in kinds}
    return sig


def _handoff_seq(h):
    return [k for k in h.event_kinds() if k.startswith("handoff_")]


# --- §5 row: "understudy never goes live" -> handoff_abort at the deadline ----
def test_supervise_handoff_understudy_never_live_aborts_at_deadline(monkeypatch, tmp_path):
    run_id = "run-h-neverlive"
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 1000)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    # the understudy launch returns a contract but the box NEVER appears in the
    # instance listing (never allocated / never boots) -> deadline is the reaper.
    prim = _resp(_hf_prim(run_id))
    h.api.instances_queue = [prim] * 8
    h.api.instances_default = prim
    h.view_queue = [{"status": "running"}] * 8
    h.view_default = {"status": "done"}
    # interval 1000: arm t5, launch t6, deadline (1800s from arm) trips t7.
    h.run(["--budget", "50", "--interval", "1000", "--handoff"])

    seq = _handoff_seq(h)
    assert seq == ["handoff_armed", "handoff_launch", "handoff_abort"]
    abort = next(e for e in h.events if e["event"] == "handoff_abort")
    assert abort["reason"] == "deadline"
    # no fence was ever opened (I1): the primary was never parked/relabelled.
    assert "handoff_fence" not in seq and "handoff_cutover" not in seq
    # NO ORPHAN: the understudy launch-contract was actually destroyed.
    assert ("DELETE", f"v0/instances/{HANDOFF_CONTRACT}/") in h.api.calls
    # survivor is the primary; it was never destroyed and the run continued.
    assert ("DELETE", "v0/instances/501/") not in h.api.calls
    assert "evicted" not in h.event_kinds() and "relaunched" not in h.event_kinds()
    assert h.events[-1]["reason"] == "terminal:done"


# --- §5 row: "primary evicted mid-handoff, understudy SYNCED" -> fast-cutover -
def test_supervise_handoff_primary_evicted_synced_fast_cutover(monkeypatch, tmp_path):
    run_id = "run-h-fastcut"
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 200)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    prim = _resp(_hf_prim(run_id))
    twin = _resp(_hf_twin(run_id))                    # primary GONE (host death), twin live
    # reconcile + t1-t6 primary only (dwell->arm->launch); t7 twin live + primary
    # gone -> SYNCED (streak 1); t8 primary gone streak 2 while SYNCED -> fast cut.
    h.api.instances_queue = [prim] * 7 + [twin] * 4
    h.api.instances_default = twin
    h.view_queue = [{"status": "running"}] * 9
    h.view_default = {"status": "done"}
    h.run(["--budget", "50", "--interval", "200", "--handoff"])

    seq = _handoff_seq(h)
    # fast-cutover SKIPS the fence (the primary is already gone — nothing to park).
    assert seq == ["handoff_armed", "handoff_launch", "handoff_synced",
                   "handoff_cutover", "handoff_complete"]
    assert "handoff_fence" not in seq
    cut = next(e for e in h.events if e["event"] == "handoff_cutover")
    assert cut["reason"] == "fast_cutover"
    complete = next(e for e in h.events if e["event"] == "handoff_complete")
    assert complete["understudy"] == 777 and complete["handoffs_done"] == 1
    # NO abort-to-zero-boxes (I2): the synced understudy was promoted, not reaped.
    assert "handoff_abort" not in seq
    assert ("DELETE", "v0/instances/777/") not in h.api.calls
    # correct relabel: the understudy was PUT back to the canonical run: label.
    assert ("PUT", "v0/instances/777/") in h.api.calls
    # both boxes billed in-window; the understudy is now canonical (no double-count).
    assert complete["handoff_spend_usd"] > 0.0
    assert complete["spend_usd"] >= complete["handoff_spend_usd"]
    # the primary eviction is driven by handoff, never the normal evict/relaunch.
    assert "evicted" not in h.event_kinds() and "relaunched" not in h.event_kinds()


# --- §5 row: "primary evicted mid-handoff, understudy NOT synced" -> abort ----
def test_supervise_handoff_primary_evicted_presynced_aborts_then_relaunches(
        monkeypatch, tmp_path):
    run_id = "run-h-presync"
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this row
    # relaunches the primary, so it reaches `_relaunch` ->
    # `_reset_run_markers`, whose body landed in `supervise.replacement`. Its
    # fence-signal seam moves to the owner with it.
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 200)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    prim = _resp(_hf_prim(run_id))
    gone = _resp()                                    # primary gone, understudy never appeared
    # t1-t6 primary live (arm/launch); t7 gone streak1 (still LAUNCHING); t8 gone
    # streak2 -> primary_evicted while pre-synced -> abort+reap, then relaunch.
    h.api.instances_queue = [prim] * 7 + [gone] * 6
    h.api.instances_default = gone
    h.view_queue = ([{"status": "running"}] * 8
                    + [{"status": "evicted"}] * 3 + [{"status": "done"}])
    h.view_default = {"status": "done"}
    h.run(["--budget", "50", "--interval", "200", "--handoff", "--max-relaunch", "3"])

    seq = _handoff_seq(h)
    assert seq == ["handoff_armed", "handoff_launch", "handoff_abort"]
    abort = next(e for e in h.events if e["event"] == "handoff_abort")
    assert abort["reason"] == "primary_evicted"
    # never fenced the primary (I1): abort keeps it the sole writer.
    assert "handoff_fence" not in seq and "handoff_cutover" not in seq
    # NO ORPHAN: the understudy launch-contract was reaped on abort.
    assert ("DELETE", f"v0/instances/{HANDOFF_CONTRACT}/") in h.api.calls
    # NEVER zero boxes (I2): the normal eviction ladder engaged after the abort.
    kinds = h.event_kinds()
    assert "evicted" in kinds and "relaunched" in kinds
    # the abort-reap of the understudy preceded the relaunch of the primary.
    assert kinds.index("handoff_abort") < kinds.index("relaunched")


# --- §5 row: "understudy evicted mid-handoff (during warm)" -> abort, stay ----
def test_supervise_handoff_understudy_evicted_during_warm_aborts(monkeypatch, tmp_path):
    run_id = "run-h-uwarm"
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 1000)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    prim = _resp(_hf_prim(run_id))
    # the understudy is ALLOCATED (a real contract, id 777) but crashes/evicts
    # during warm — it shows up stopped (never live -> never synced) and is then
    # reaped at the deadline; the primary stays live the whole time.
    twin_dead = _resp(_hf_prim(run_id), _hf_twin(run_id, actual_status="stopped"))
    h.api.instances_queue = [prim] * 7 + [twin_dead] * 3
    h.api.instances_default = twin_dead
    h.view_queue = [{"status": "running"}] * 8
    h.view_default = {"status": "done"}
    # interval 1000: arm t5, launch t6, deadline t7 while the twin is stopped.
    h.run(["--budget", "50", "--interval", "1000", "--handoff"])

    seq = _handoff_seq(h)
    assert seq == ["handoff_armed", "handoff_launch", "handoff_abort"]
    abort = next(e for e in h.events if e["event"] == "handoff_abort")
    assert abort["reason"] == "deadline"
    assert "handoff_fence" not in seq and "handoff_cutover" not in seq
    # the ADOPTED understudy id (777, seen live in the listing) is the one reaped.
    assert abort["instance_id"] == 777
    assert ("DELETE", "v0/instances/777/") in h.api.calls
    # stay on the primary: it is never destroyed and the run continues on it.
    assert ("DELETE", "v0/instances/501/") not in h.api.calls
    assert "evicted" not in h.event_kinds() and "relaunched" not in h.event_kinds()
    assert h.events[-1]["reason"] == "terminal:done"


# --- §5 row: "both die" -> abort-reap the understudy remnant, relaunch primary -
def test_supervise_handoff_both_boxes_die_reconciles_without_wedge(monkeypatch, tmp_path):
    run_id = "run-h-bothdie"
    # MIGRATED (was MIGRATION-BLOCKED, step 6e; closed at step 6d): this row
    # relaunches the primary, so it reaches `_relaunch` ->
    # `_reset_run_markers`, whose body landed in `supervise.replacement`. Its
    # fence-signal seam moves to the owner with it.
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 200)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    prim = _resp(_hf_prim(run_id))
    # t7: BOTH sick — primary gone AND the understudy remnant present-but-stopped
    # (adopted id 777); t8: both gone -> primary_evicted pre-synced -> abort reaps
    # the 777 remnant, then the primary's own ladder relaunches. Never wedged.
    both_sick = _resp(_hf_twin(run_id, actual_status="stopped"))
    gone = _resp()
    h.api.instances_queue = [prim] * 7 + [both_sick] + [gone] * 6
    h.api.instances_default = gone
    h.view_queue = ([{"status": "running"}] * 8
                    + [{"status": "evicted"}] * 3 + [{"status": "done"}])
    h.view_default = {"status": "done"}
    h.run(["--budget", "50", "--interval", "200", "--handoff", "--max-relaunch", "3"])

    seq = _handoff_seq(h)
    assert seq == ["handoff_armed", "handoff_launch", "handoff_abort"]
    abort = next(e for e in h.events if e["event"] == "handoff_abort")
    assert abort["reason"] == "primary_evicted"
    # reconcile reaps the understudy REMNANT (the adopted 777, not the raw contract).
    assert abort["instance_id"] == 777
    assert ("DELETE", "v0/instances/777/") in h.api.calls
    # the primary's ladder is authoritative -> it does not wedge; it relaunches.
    kinds = h.event_kinds()
    assert "evicted" in kinds and "relaunched" in kinds
    assert h.events[-1]["reason"] == "terminal:done"


# --- §5 invariant I3: "budget exhausted mid-window" -> reap window, park, stop -
def test_supervise_handoff_budget_exhausted_mid_window_reaps_and_parks(monkeypatch, tmp_path):
    run_id = "run-h-budget"
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 200)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    # cheap during dwell (fat headroom clears the ARM gate), then the primary's
    # get-and-hold bid SPIKES post-arm (§10 ruling 3: it may climb toward
    # on-demand while the understudy warms) — the double-bill blows --budget.
    cheap = _resp(_hf_prim(run_id, dph_total=0.30))
    spiked = _resp(_hf_prim(run_id, dph_total=7.2))
    live_both = _resp(_hf_prim(run_id, dph_total=7.2), _hf_twin(run_id))
    # t1-t5 cheap (arm t5); t6 launch (spiked); t7 warm (spiked, twin not yet up);
    # t8 twin live (SYNCED, pre-cutover) AND cumulative spend crosses --budget.
    h.api.instances_queue = [cheap] * 6 + [spiked] * 2 + [live_both] * 3
    h.api.instances_default = live_both
    h.view_queue = [{"status": "running"}] * 12
    h.view_default = {"status": "running"}
    h.run(["--budget", "1.0", "--interval", "200", "--handoff"])

    seq = _handoff_seq(h)
    # the window opened (armed/launched/synced) then the guard reaped it — the
    # understudy never fenced the primary (pre-cutover reap).
    assert seq == ["handoff_armed", "handoff_launch", "handoff_synced",
                   "handoff_abort"]
    abort = next(e for e in h.events if e["event"] == "handoff_abort")
    assert abort["reason"] == "supervisor_stop"
    assert "handoff_fence" not in seq and "handoff_cutover" not in seq
    # the budget guard saw the TRUE double-bill total and stopped the run.
    assert h.events[-1]["reason"] == "budget"
    # NO ORPHAN + NO RUNAWAY: understudy reaped (DELETE), primary parked (PUT
    # stopped, not destroyed), no relaunch churn.
    assert ("DELETE", "v0/instances/777/") in h.api.calls
    assert ("PUT", "v0/instances/501/") in h.api.calls
    assert ("DELETE", "v0/instances/501/") not in h.api.calls
    assert "relaunched" not in h.event_kinds()
    cost = [e for e in h.events if e["event"] == "cost"]
    assert cost and cost[-1]["cost_usd"] >= 1.0        # true burn crossed the cap


# --- §5 row: "supervisor crashes mid-handoff" -> reconcile-on-restart adopts ---
def test_supervise_handoff_reconcile_adopts_live_twin_on_restart(monkeypatch, tmp_path):
    run_id = "run-h-reconcile"
    h = Harness(monkeypatch, tmp_path, run_id, max_sleeps=60)
    h.api.offers = _arm_over_ceiling(monkeypatch, dwell_s=4 * 200)
    monkeypatch.setattr(handoff, "_handoff_run_signals", _fence_signals(h))

    # A crashed supervisor left a LIVE run:<ID>:handoff twin already staged. On
    # restart, reconcile must ADOPT it at SYNCED (not orphan it, not launch a
    # THIRD box) and drive the fence->cutover->complete from there.
    both = _resp(_hf_prim(run_id), _hf_twin(run_id))
    h.api.instances_queue = [both] * 8
    h.api.instances_default = both
    h.view_queue = [{"status": "running"}] * 8
    h.view_default = {"status": "done"}
    h.run(["--budget", "50", "--interval", "200", "--handoff"])

    seq = _handoff_seq(h)
    # adopted (reconciled), NOT armed/launched — no third box, no fresh arm.
    assert seq[0] == "handoff_reconciled"
    assert "handoff_armed" not in seq and "handoff_launch" not in seq
    recon = next(e for e in h.events if e["event"] == "handoff_reconciled")
    assert recon["phase"] == "SYNCED" and recon["live"] is True
    # never launched a third box (no understudy ask PUT).
    assert ("PUT", "v0/asks/999/") not in h.api.calls
    # from the adopted twin, the full fenced cutover completes.
    assert seq[1:] == ["handoff_fence", "handoff_cutover", "handoff_complete"]
    fence_i = h.event_kinds().index("handoff_fence")
    cut_i = h.event_kinds().index("handoff_cutover")
    assert fence_i < cut_i                             # fence (flush) precedes cutover
    complete = next(e for e in h.events if e["event"] == "handoff_complete")
    assert complete["understudy"] == 777
    # the primary was drained (destroyed) only after the understudy took over.
    assert ("DELETE", "v0/instances/501/") in h.api.calls
    assert ("DELETE", "v0/instances/777/") not in h.api.calls


# --- spend accounting: no double-count once the understudy is canonical (§3) --
def test_handoff_accrue_no_double_count_after_promotion():
    """Once the understudy becomes st's tracked box, _accrue_cost bills it and
    _handoff_accrue must NOT bill it again (the id guard, HANDOFF_DESIGN §3)."""
    st = {"instance_id": 777, "dt": 200.0, "spend_usd": 5.0}
    hf = {"phase": "DONE", "understudy_iid": 777, "understudy_dph": 0.12,
          "understudy_status": "running", "handoff_spend_usd": 1.0}
    handoff._handoff_accrue(st, hf)
    assert st["spend_usd"] == 5.0                      # no second bill for the same box
    assert hf["handoff_spend_usd"] == 1.0
    # while the understudy is still a SEPARATE live box, it DOES accrue into both.
    st2 = {"instance_id": 501, "dt": 200.0, "spend_usd": 5.0}
    hf2 = {"phase": "SYNCED", "understudy_iid": 777, "understudy_dph": 0.12,
           "understudy_status": "running", "handoff_spend_usd": 1.0}
    handoff._handoff_accrue(st2, hf2)
    assert st2["spend_usd"] > 5.0 and hf2["handoff_spend_usd"] > 1.0


# =============================================================================
# 5h. T4b — the driver PRODUCES exactly what T6's box-side guards CONSUME.
#
#     T6 (onstart/train.sh) shipped inert: it reads HANDOFF_EPOCH/HANDOFF_TTL_S
#     from the understudy launch env, maxes runs/<ID>/handoff/<epoch>.json to
#     refuse a stale writer, and stays up on runs/<ID>/handoff/promoted. These
#     tests pin the producer side to those EXACT names/paths + the fence bid-pin
#     belt the epoch guard needs for the parked primary.
# =============================================================================
def _t4b_understudy_st(**over):
    st = {
        "run_id": "r1", "dph_total": 0.60, "last_bid": 0.60,
        "on_demand": 0.65, "remaining_wall_h": 10.0,
        "launch_spec": {
            "image": "reg/img:tag", "disk": 100, "runtype": "ssh_direct",
            "env": {"RUN_ID": "r1"}, "runset": "rs1",
            "secret_env_keys": ["B2_KEY_ID", "B2_APPLICATION_KEY"],
        },
    }
    st.update(over)
    return st


_T4B_OFFER = {"id": 999, "min_bid": 0.10, "dph_total": 0.50}


# --- (a) understudy launch env carries HANDOFF_EPOCH + HANDOFF_TTL_S -----------
# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_relaunch_body` landed in
# `vastlib.supervise.replacement`, the same module as
# `_handoff_understudy_body`, so the call reaches a real body. The mint seam is
# stubbed at `launch.spec`, where `_resolve_secret` resolves it.
def test_handoff_understudy_body_sets_epoch_and_ttl_env(monkeypatch):
    monkeypatch.setattr(launch_spec, "_ship_b2_pair",
                        lambda name, hours=None, dry_run=False: ("KID", "SEC"))
    a = argparse.Namespace(dry_run=True)
    body, bid, missing = replacement._handoff_understudy_body(
        _t4b_understudy_st(), a, _T4B_OFFER, epoch=2)
    assert missing == []
    # T6 train.sh (`HANDOFF_EPOCH="${HANDOFF_EPOCH:-}"`, `HANDOFF_TTL_S:-`) reads
    # these two strings from the launch env — set them under the EXACT names.
    assert body["env"]["HANDOFF_EPOCH"] == "2"
    assert body["env"]["HANDOFF_TTL_S"] == str(bidpolicy.HANDOFF_TTL_S)
    # the dead-man deadline is the window PLUS a margin (a healthy handoff, whose
    # understudy boot eats much of the window, must not trip it early).
    assert bidpolicy.HANDOFF_TTL_S > bidpolicy.HANDOFF_DEADLINE_S
    # the pre-existing launch env survives alongside the two new contracts.
    assert body["env"]["RUN_ID"] == "r1" and body["env"]["B2_KEY_ID"] == "KID"


def test_handoff_understudy_body_epoch_defaults_to_one(monkeypatch):
    # a direct call without an explicit epoch (the first handoff) stamps epoch 1 —
    # strictly above the original primary's absent (== 0) HANDOFF_EPOCH.
    monkeypatch.setattr(launch_spec, "_ship_b2_pair",
                        lambda name, hours=None, dry_run=False: ("KID", "SEC"))
    a = argparse.Namespace(dry_run=True)
    body, _, _ = replacement._handoff_understudy_body(_t4b_understudy_st(), a,
                                                      _T4B_OFFER)
    assert body["env"]["HANDOFF_EPOCH"] == "1"


# --- (b) ARM writes runs/<ID>/handoff/<epoch>.json (bare-int marker) -----------
def test_handoff_arm_writes_epoch_marker(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    writes = []
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True: (writes.append(path) or True))
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: None)
    st = {"run_id": "r1", "instance_id": 501, "now": 1000.0, "last_bid": 0.60,
          "on_demand": 1.0}
    hf = handoff._init_handoff_state()
    a = argparse.Namespace(dry_run=False)
    handoff._do_handoff_move(st, a, hf, bidpolicy.HandoffAction("arm", "over_ceiling"))
    # epoch 1 (handoffs_done 0 + 1); the marker T6 lsf-maxes is a BARE-INT filename.
    assert hf["epoch"] == 1
    assert "b2:bkt/runs/r1/handoff/1.json" in writes
    # a SECOND handoff (handoffs_done already 1) stamps the strictly-greater epoch 2.
    hf2 = handoff._init_handoff_state(); hf2["handoffs_done"] = 1
    handoff._do_handoff_move(st, a, hf2, bidpolicy.HandoffAction("arm", "over_ceiling"))
    assert hf2["epoch"] == 2 and "b2:bkt/runs/r1/handoff/2.json" in writes


def test_handoff_arm_marker_skipped_in_dry_run(monkeypatch):
    # dry-run (and no-bucket) never touch B2 — mirrors _reset_run_markers.
    monkeypatch.setenv("B2_BUCKET", "bkt")
    writes = []
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True: (writes.append(path) or True))
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: None)
    st = {"run_id": "r1", "instance_id": 501, "now": 1000.0}
    hf = handoff._init_handoff_state()
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=True), hf,
                             bidpolicy.HandoffAction("arm", "over_ceiling"))
    assert writes == [] and hf["epoch"] == 1


# --- (c) CUTOVER writes runs/<ID>/handoff/promoted BEFORE the primary destroy --
def test_handoff_promoted_marker_precedes_primary_destroy(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    log = []      # one shared ordered log of B2 writes + destroy API calls
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True: (log.append(("B2", path)) or True))
    monkeypatch.setattr(lifecycle, "_put_label_soft", lambda iid, label: (True, None))
    monkeypatch.setattr(handoff, "_confirm_gone", lambda iid: True)
    monkeypatch.setattr(lifecycle, "_revoke_box_keys", lambda names: None)

    def _fake_destroy(iid, dry_run=False):
        log.append(("DESTROY", iid))
        return True, None
    monkeypatch.setattr(lifecycle, "_destroy_soft", _fake_destroy)
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: None)

    st = {"run_id": "r1", "instance_id": 501, "now": 2000.0}
    hf = handoff._init_handoff_state()
    hf.update({"phase": "DRAINING", "epoch": 1, "primary_iid": 501,
               "understudy_iid": 777})
    a = argparse.Namespace(dry_run=False)
    # cutover (writes `promoted`), then the LATER-tick drain (destroys the primary).
    handoff._do_handoff_move(st, a, hf, bidpolicy.HandoffAction("resume_understudy",
                                                              "post_flush"))
    handoff._do_handoff_move(st, a, hf, bidpolicy.HandoffAction("drain_primary",
                                                              "understudy_producing"))
    assert ("B2", "b2:bkt/runs/r1/handoff/promoted") in log
    assert ("DESTROY", 501) in log
    # load-bearing ORDER: promoted lands before the primary is torn down, so a
    # supervisor death mid-cutover can never self-park the true survivor.
    assert log.index(("B2", "b2:bkt/runs/r1/handoff/promoted")) \
        < log.index(("DESTROY", 501))


# --- (d) FENCE pins the parked primary's bid below floor (no auto-resume) ------
def test_handoff_fence_pins_primary_bid_below_floor(monkeypatch):
    bids = []
    monkeypatch.setattr(lifecycle, "_put_state_soft", lambda iid, state: (True, None))
    monkeypatch.setattr(lifecycle, "_wait_states_soft",
                        lambda iid, states, timeout: True)
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: (bids.append((iid, price)) or (True, None)))
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: None)
    floor = 0.10
    st = {"run_id": "r1", "instance_id": 501, "now": 3000.0, "market_min_bid": floor}
    hf = handoff._init_handoff_state()
    hf.update({"phase": "SYNCED", "primary_iid": 501, "understudy_iid": 777})
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=False), hf,
                             bidpolicy.HandoffAction("fence_primary", "synced"))
    # the primary's bid was pinned to the API-minimum, strictly BELOW the live floor
    # — vast can't auto-resume it during the fence->drain window (box-44566398 leak),
    # the belt the epoch guard needs (the parked primary has no HANDOFF_EPOCH).
    assert bids == [(501, bidpolicy.HANDOFF_PARK_BID)]
    assert bidpolicy.HANDOFF_PARK_BID < floor
    assert hf["phase"] == "CUTOVER"
    assert hf["fence_ts"] == 3000.0                    # F1: opens the flush-timeout clock


def test_handoff_fence_bid_pin_dry_run_is_no_api(monkeypatch):
    bids = []
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: pytest.fail("no park in dry-run"))
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: bids.append(price))
    monkeypatch.setattr(journal, "_sup_emit", lambda *a, **k: None)
    st = {"run_id": "r1", "instance_id": 501, "now": 3000.0, "market_min_bid": 0.10}
    hf = handoff._init_handoff_state()
    hf.update({"phase": "SYNCED", "primary_iid": 501, "understudy_iid": 777})
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=True), hf,
                             bidpolicy.HandoffAction("fence_primary", "synced"))
    assert bids == [] and hf["phase"] == "CUTOVER"
    assert hf["fence_ts"] == 3000.0                    # F1: stamped even in dry-run


# --- F1 driver: flush-timeout resume carries its reason into marker + event ----
def test_handoff_flush_timeout_marker_and_event_reason(monkeypatch):
    # resume_understudy driven with reason 'flush_timeout' lands that reason in BOTH
    # the promoted marker (telemetry) and the handoff_cutover event; the parked
    # primary is NOT marked gone (unlike fast_cutover) — it still drains normally.
    monkeypatch.setenv("B2_BUCKET", "bkt")
    writes = {}
    monkeypatch.setattr(b2, "_b2_rcat",
                        lambda path, body, hard=True: (writes.__setitem__(path, body), True)[1])
    monkeypatch.setattr(lifecycle, "_put_label_soft", lambda iid, label: (True, None))
    events = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: events.append((ev, kw)))
    st = {"run_id": "r1", "instance_id": 501, "now": 5000.0}
    hf = handoff._init_handoff_state()
    hf.update({"phase": "CUTOVER", "epoch": 1, "primary_iid": 501,
               "understudy_iid": 777,
               "fence_ts": 5000.0 - bidpolicy.HANDOFF_FENCE_TIMEOUT_S})
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=False), hf,
                             bidpolicy.HandoffAction("resume_understudy", "flush_timeout"))
    assert hf["phase"] == "DRAINING"
    assert hf.get("primary_gone") is False             # parked primary still drains
    marker = writes.get("b2:bkt/runs/r1/handoff/promoted")
    assert marker is not None and '"reason":"flush_timeout"' in marker
    cut = [kw for ev, kw in events if ev == "handoff_cutover"]
    assert cut and cut[0]["reason"] == "flush_timeout"


# --- F2: DRAINING-stall alarm — bounded observability, fires exactly once ------
def test_handoff_stall_alarm_fires_exactly_once():
    emits = []
    hf = handoff._init_handoff_state()
    hf.update({"phase": "DRAINING", "understudy_iid": 777, "fence_ts": 1000.0})
    now = 1000.0 + bidpolicy.HANDOFF_DEADLINE_S
    handoff._handoff_stall_alarm(hf, now, lambda **f: emits.append(f))
    handoff._handoff_stall_alarm(hf, now + 100, lambda **f: emits.append(f))
    assert len(emits) == 1                             # latched: no re-alarm while stalled
    assert emits[0]["understudy"] == 777 and emits[0]["phase"] == "DRAINING"
    assert emits[0]["waited_s"] >= bidpolicy.HANDOFF_DEADLINE_S
    assert hf["stall_alarmed"] is True


def test_handoff_stall_alarm_silent_before_deadline():
    emits = []
    hf = handoff._init_handoff_state()
    hf.update({"phase": "DRAINING", "understudy_iid": 777, "fence_ts": 1000.0})
    handoff._handoff_stall_alarm(hf, 1000.0 + bidpolicy.HANDOFF_DEADLINE_S - 1,
                                 lambda **f: emits.append(f))
    assert emits == [] and hf["stall_alarmed"] is False


def test_handoff_stall_alarm_reset_clears_latch():
    # _handoff_reset re-arms the alarm for the next opportunity (F2 latch hygiene).
    hf = handoff._init_handoff_state()
    hf["stall_alarmed"] = True
    handoff._handoff_reset(hf, handoffs_done=1, cooldown_until=0.0)
    assert hf["stall_alarmed"] is False and hf["fence_ts"] is None


def test_handoff_stall_alarm_covers_wedged_cutover():
    # S4 (2026-07-18 review): CUTOVER normally exits at HANDOFF_FENCE_TIMEOUT_S
    # (< the deadline), so a CUTOVER still open at the deadline is wedged (a
    # retarget_incomplete latch / an understudy dead inside the fence window) —
    # it must page, not wait for a DRAINING it may never reach.
    assert bidpolicy.HANDOFF_FENCE_TIMEOUT_S < bidpolicy.HANDOFF_DEADLINE_S
    emits = []
    hf = handoff._init_handoff_state()
    hf.update({"phase": "CUTOVER", "understudy_iid": 777, "fence_ts": 1000.0})
    handoff._handoff_stall_alarm(hf, 1000.0 + bidpolicy.HANDOFF_DEADLINE_S,
                                 lambda **f: emits.append(f))
    assert len(emits) == 1 and emits[0]["phase"] == "CUTOVER"
    assert hf["stall_alarmed"] is True


# --- F4: run-lane understudy launch runs _launch_preflight (twin-dup guard) ----
def _f4_launch_move(monkeypatch, run_id, instances):
    """Drive one launch_understudy move with a stubbed body + recorded launch/event
    I/O; `instances` is st's per-tick snapshot the preflight now consults."""
    monkeypatch.setattr(replacement, "_handoff_understudy_body",
                        lambda st, a, offer, epoch=1:
                        ({"label": f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}"}, 0.12, []))
    launches = []
    monkeypatch.setattr(launch_mod, "launch_instance",
                        lambda oid, body: (launches.append(oid), (True, 4242, None))[-1])
    events = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda rid, ev, **kw: events.append((ev, kw)))
    st = {"run_id": run_id, "instance_id": 501, "now": 1000.0, "_instances": instances}
    hf = handoff._init_handoff_state()
    hf.update(phase="ARMED", handoff_started_ts=1000.0, primary_iid=501, epoch=1,
              chosen_offer={"id": 999, "min_bid": 0.10, "dph_total": 0.50})
    handoff._do_handoff_move(st, argparse.Namespace(dry_run=False), hf,
                             bidpolicy.HandoffAction("launch_understudy", "armed"))
    return hf, launches, events


def test_handoff_run_lane_understudy_refused_by_existing_twin(monkeypatch):
    # a live run:<ID>:handoff twin already exists (a crash-orphan we did not
    # reconcile) -> the second understudy is REFUSED by _launch_preflight (its
    # dead handoff-twin allowance is now actually exercised on this path), aborting
    # as understudy_unlaunchable with NO duplicate launch.
    run_id = "r1"
    twin = {"id": 777, "label": f"run:{run_id}{labels.HANDOFF_LABEL_SUFFIX}",
            "actual_status": "running"}
    prim = {"id": 501, "label": f"run:{run_id}", "actual_status": "running"}
    hf, launches, events = _f4_launch_move(monkeypatch, run_id, [prim, twin])
    assert launches == []                                  # no duplicate box launched
    abort = [kw for ev, kw in events if ev == "handoff_abort"]
    assert abort and abort[-1]["reason"] == "understudy_unlaunchable"
    assert hf["phase"] == "IDLE"                            # abort reset the machine


def test_handoff_run_lane_understudy_launches_when_no_twin(monkeypatch):
    # F4 must not block the normal path: no existing :handoff twin -> preflight
    # passes and the understudy launches (the primary run:<ID> alone never refuses).
    run_id = "r2"
    prim = {"id": 501, "label": f"run:{run_id}", "actual_status": "running"}
    hf, launches, events = _f4_launch_move(monkeypatch, run_id, [prim])
    assert launches == [999] and hf["phase"] == "LAUNCHING"
    assert str(hf["understudy_iid"]) == "4242"


# ===========================================================================
# 8. jobs-lane handoff (T7; HANDOFF_DESIGN §2.2 / §9). The jobs-box analogue of
#    the run-lane handoff drives the SAME pure handoff_poll; these fault-inject
#    the jobs-only I/O (`launch --jobs` understudy, `job retarget` cutover,
#    per-JOB markers) at the _do_job_handoff_move / _job_handoff_tick layer and
#    assert the two §5 jobs-only rows + the happy-path ordering + reconcile.
# ===========================================================================
class _JobHandoffIO:
    """Records the jobs-lane handoff I/O and scripts the injected box signals so a
    scenario can walk the phase machine deterministically (no vast API, no B2)."""

    def __init__(self, monkeypatch, queue=("job-a",)):
        self.markers = []            # (job_id, rel) B2 handoff markers written
        self.destroyed = []          # iids DELETEd
        self.parked = []             # (iid, state) PUTs
        self.pinned = []             # (iid, bid) PUTs
        self.retargeted = []         # job_ids passed to cmd_job_retarget
        self.relabeled = []          # (iid, label) instance-label PUTs at cutover
        self.events = []             # (event, fields) box-lifecycle emits
        self.queue = list(queue)     # the PRIMARY's live queue (retarget removes)
        self.delete_fails = False    # §5: old-ticket delete fails at cutover
        self.u_live = True           # understudy liveness (observe)
        self.synced = False          # prewarm/pull complete (understudy SYNCED)
        self.final_flush = False     # primary flushed (fence satisfied)
        self.producing = False       # understudy claimed/checkpointed (drain gate)
        self.launch_iid = "9100"     # None => understudy unlaunchable

        m = monkeypatch
        m.setattr(replacement, "_launch_job_understudy",
                  lambda jctx, hf, epoch: (self.launch_iid, 0.20, None)
                  if self.launch_iid else (None, None, "understudy_unlaunchable"))
        m.setattr(replacement, "_job_understudy_offer",
                  lambda jctx, hf=None: {"id": 1, "min_bid": 0.20, "dph_total": 1.0})

        def _observe(jctx, hf):
            if not hf.get("understudy_iid"):
                return
            hf["understudy_status"] = "running" if self.u_live else "stopped"
            hf["understudy_dph"] = 0.20
            if self.u_live and hf.get("understudy_live_since") is None:
                hf["understudy_live_since"] = jctx.get("now")
            if hf.get("ckpt_pulled_epoch") is None and self.synced \
                    and hf.get("phase") in ("LAUNCHING", "WARMING"):
                hf["ckpt_pulled_epoch"] = hf.get("handoffs_done", 0) + 1
        m.setattr(handoff, "_handoff_observe_job_understudy", _observe)
        m.setattr(handoff, "_handoff_job_signals",
                  lambda running, allj, u: {"final_flush_seen": self.final_flush,
                                            "understudy_producing": self.producing})
        m.setattr(handoff, "_handoff_job_b2_write",
                  lambda jid, rel, body, dry_run=False:
                  (self.markers.append((jid, rel)), True)[1])
        m.setattr(lifecycle, "_put_state_soft",
                  lambda iid, state: (self.parked.append((iid, state)), (True, None))[1])
        m.setattr(lifecycle, "_wait_states_soft", lambda *a, **k: (True, "stopped"))
        m.setattr(lifecycle, "_put_bid_soft",
                  lambda iid, bid: (self.pinned.append((iid, bid)), (True, None))[1])
        m.setattr(lifecycle, "_put_label_soft",
                  lambda iid, label: (self.relabeled.append((iid, label)), (True, None))[1])
        m.setattr(lifecycle, "_destroy_soft",
                  lambda iid, dry_run=False, **k: (self.destroyed.append(iid), (True, None))[1])
        m.setattr(handoff, "_confirm_gone", lambda iid, **k: True)
        m.setattr(jobmeta, "emit_box_event",
                  lambda iid, ev, **kw: (self.events.append((ev, kw)),
                                         {"_emitted": True})[1])
        m.setattr(jobmeta, "list_queue", lambda box, **k: list(self.queue))

        def _retarget(ns):
            # cmd_job_retarget writes the new ticket, then DELETEs the old. The
            # delete-fail row keeps the old ticket in place (design's warn+continue).
            self.retargeted.append(ns.job_id)
            if not self.delete_fails and ns.job_id in self.queue:
                self.queue.remove(ns.job_id)
        m.setattr(jobs_control, "cmd_job_retarget", _retarget)


def _jctx(io_, **over):
    d = {"a": argparse.Namespace(dry_run=False, budget=100.0, handoff=True),
         "iid": "9000", "dry_run": False, "now": 1_000.0, "dt": 60.0,
         "instances": [], "last_bid": 1.0, "on_demand": 2.0, "dph": 1.0,
         "budget_usd": 100.0, "spend_usd": 0.0, "remaining_wall_h": 24.0,
         "primary_evicted": False, "_over_pref": True,
         # These fixtures exercise the MACHINERY (markers, retarget, relabel,
         # abort), so they stand in for a driver that can finish the migration
         # and for a queue whose running jobs are checkpointed. The 2026-08-08
         # work-awareness rails are pinned by their own tests, on the pure core
         # and on the real `job_supervise_tick`; declaring them here keeps this
         # block testing what it was written to test.
         "handoff_can_complete": True, "running_unresumable": 0,
         "min_running_eta_s": None, "ckpt_stale": False, "work_at_risk_h": 0.0,
         "pending_views": [],
         "pending_jobs": list(io_.queue), "running_jobs": list(io_.queue)}
    d.update(over)
    return d


def _drive_to_armed(io_, jctx, hf, tick_s=45.0):
    """Walk the dwell to the ARM. The dwell is a DURATION (HANDOFF_DWELL_S), so
    the clock has to move the way a real lane's does — five 45 s ticks is what
    the old five-poll count was."""
    for _ in range(bidpolicy.HANDOFF_DWELL_POLLS):
        handoff._job_handoff_tick(jctx, hf)
        jctx["now"] = jctx.get("now", 0.0) + tick_s
    return hf


def test_job_handoff_happy_path_ordering(monkeypatch):
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a", "job-b"])
    hf = handoff._init_job_handoff_state()
    jctx = _jctx(io_)
    _drive_to_armed(io_, jctx, hf)
    assert hf["phase"] == "ARMED"
    # epoch marker written per pending JOB at ARM (jobs/<JID>/handoff/1.json).
    assert sorted(io_.markers) == [("job-a", "1.json"), ("job-b", "1.json")]
    assert io_.retargeted == [] and io_.destroyed == []   # nothing moved/dropped yet

    handoff._job_handoff_tick(jctx, hf)                    # ARMED -> LAUNCHING
    assert hf["phase"] == "LAUNCHING" and hf["understudy_iid"] == "9100"

    io_.synced = True
    handoff._job_handoff_tick(jctx, hf)                    # LAUNCHING -> SYNCED
    assert hf["phase"] == "SYNCED"
    assert io_.parked == [] and io_.retargeted == []       # primary NOT touched pre-fence

    handoff._job_handoff_tick(jctx, hf)                    # SYNCED -> fence -> CUTOVER
    assert hf["phase"] == "CUTOVER"
    assert io_.parked == [("9000", "stopped")]             # primary parked at fence
    assert io_.pinned == [("9000", bidpolicy.HANDOFF_PARK_BID)]  # bid pinned below floor
    assert io_.retargeted == []                            # retarget waits for flush

    handoff._job_handoff_tick(jctx, hf)                    # CUTOVER blocks (no flush)
    assert hf["phase"] == "CUTOVER" and io_.retargeted == []

    io_.final_flush = True
    handoff._job_handoff_tick(jctx, hf)                    # CUTOVER -> retarget -> DRAINING
    assert hf["phase"] == "DRAINING"
    assert io_.retargeted == ["job-a", "job-b"]            # both tickets moved (same JID)
    assert ("job-a", "promoted") in io_.markers and ("job-b", "promoted") in io_.markers
    # BOX relabel: the promoted understudy is now canonical, so its instance label
    # drops the dead-primary handoff-twin marker (job:9000:handoff -> job:9100) —
    # NO :handoff suffix. Keyed on the understudy iid, not the run/primary.
    assert io_.relabeled == [("9100", "job:9100")]
    assert not any(lab.endswith(labels.HANDOFF_LABEL_SUFFIX) for _, lab in io_.relabeled)
    assert io_.destroyed == []                             # primary NOT destroyed yet

    handoff._job_handoff_tick(jctx, hf)                    # DRAINING blocks (not producing)
    assert io_.destroyed == []

    io_.producing = True
    handoff._job_handoff_tick(jctx, hf)                    # DRAINING -> drain primary
    assert io_.destroyed == ["9000"]                       # destroy AFTER producing (§2.2/6)

    handoff._job_handoff_tick(jctx, hf)                    # DRAINING -> complete
    assert jctx.get("_handoff_completed_iid") == "9100"    # supervise the survivor now
    assert hf["phase"] == "IDLE" and hf["handoffs_done"] == 1
    kinds = [e for e, _ in io_.events]
    assert kinds == ["handoff_armed", "handoff_launch", "handoff_synced",
                     "handoff_fence", "handoff_cutover", "handoff_complete"]


def test_job_handoff_cutover_relabels_understudy_off_handoff_suffix(monkeypatch):
    """Issue A (canary-job 2026-07-15): after a jobs-lane cutover the promoted
    understudy stayed labelled `job:<primary>:handoff` even though the primary was
    destroyed, so it read as a handoff-twin of a dead box (confusing `ls`,
    reconcile's twin scan, and a future epoch-2 handoff). The cutover must PUT a
    canonical instance label `job:<understudy>` (mirroring the run lane's relabel),
    keyed on the understudy iid, with NO :handoff suffix and NO dead-primary
    reference."""
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    io_.synced = True
    hf = handoff._init_job_handoff_state()
    jctx = _jctx(io_)
    _drive_to_armed(io_, jctx, hf)
    handoff._job_handoff_tick(jctx, hf)                    # -> LAUNCHING
    handoff._job_handoff_tick(jctx, hf)                    # -> SYNCED
    handoff._job_handoff_tick(jctx, hf)                    # -> fence -> CUTOVER
    assert io_.relabeled == []                             # nothing relabeled pre-flush
    io_.final_flush = True
    handoff._job_handoff_tick(jctx, hf)                    # CUTOVER -> retarget -> DRAINING
    assert hf["phase"] == "DRAINING"
    assert io_.relabeled == [(io_.launch_iid, f"job:{io_.launch_iid}")]
    (iid, label), = io_.relabeled
    assert label == "job:9100"                             # keyed on understudy, canonical
    assert labels.HANDOFF_LABEL_SUFFIX not in label       # no :handoff twin marker
    assert "9000" not in label                             # no dead-primary reference


def test_job_handoff_asset_pull_fail_aborts_and_keeps_primary(monkeypatch):
    # §5 'bundle/asset pull fails (jobs)': understudy jobd errors -> never SYNCED
    # -> deadline abort -> reap understudy, retarget NEVER issued, primary keeps
    # ticket. (invariants I1: no two writers, I2: primary still running.)
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    io_.synced = False                                     # prewarm/pull never completes
    hf = handoff._init_job_handoff_state()
    jctx = _jctx(io_)
    _drive_to_armed(io_, jctx, hf)
    handoff._job_handoff_tick(jctx, hf)                    # -> LAUNCHING
    assert hf["phase"] == "LAUNCHING" and hf["understudy_iid"] == "9100"

    # stays WARMING/LAUNCHING (no sync); push the clock past the 2x-box deadline.
    jctx["now"] = hf["handoff_started_ts"] + bidpolicy.HANDOFF_DEADLINE_S + 1
    handoff._job_handoff_tick(jctx, hf)

    assert io_.destroyed == ["9100"]                       # understudy reaped
    assert io_.retargeted == []                            # retarget NEVER issued
    assert io_.parked == []                                # primary NOT parked/fenced
    assert io_.queue == ["job-a"]                          # primary keeps its ticket
    assert hf["phase"] == "IDLE" and hf["handoffs_done"] == 0   # abort != a handoff
    assert ("handoff_abort", ) in [(e,) for e, _ in io_.events]


def test_job_handoff_retarget_delete_fail_no_dup_no_lost_ticket(monkeypatch):
    # §5 'retarget's old-ticket delete fails': the NEW ticket is written (job not
    # lost) but the OLD ticket lingers -> cutover is INCOMPLETE. Guarantee: do NOT
    # destroy the primary (double-claim risk); the husk stays parked + epoch-fenced.
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    io_.synced = True
    io_.delete_fails = True                                # inject the delete failure
    hf = handoff._init_job_handoff_state()
    jctx = _jctx(io_)
    _drive_to_armed(io_, jctx, hf)
    handoff._job_handoff_tick(jctx, hf)                    # -> LAUNCHING
    handoff._job_handoff_tick(jctx, hf)                    # -> SYNCED
    handoff._job_handoff_tick(jctx, hf)                    # -> fence -> CUTOVER
    io_.final_flush = True
    handoff._job_handoff_tick(jctx, hf)                    # CUTOVER -> retarget (delete fails)

    # NO LOST TICKET: the new ticket was written (retarget issued for the job)...
    assert io_.retargeted == ["job-a"]
    # ...and the OLD ticket still lingers (delete failed) -> cutover held open.
    assert hf["retarget_incomplete"] == ["job-a"]
    assert hf["phase"] == "CUTOVER"                        # NOT advanced to DRAINING
    # NO DUPLICATE EXECUTION: the primary husk is NOT destroyed even once the
    # understudy is 'producing' — the epoch marker + bid-pin fence a resumed husk.
    assert ("job-a", "promoted") not in io_.markers        # promotion withheld
    assert io_.relabeled == []                             # canonical relabel withheld too
    io_.producing = True
    handoff._job_handoff_tick(jctx, hf)
    assert io_.destroyed == []                             # primary NEVER destroyed


def test_job_handoff_reconcile_adopts_live_twin(monkeypatch):
    # §5 crash row: a live job:<primary>:handoff twin left by a crashed supervisor
    # is adopted (resumed at SYNCED), not orphaned.
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    # affirmative jobd boot proof (JOBD_STATUS stamped by its main loop) — a live
    # twin WITHOUT it resumes at LAUNCHING, tested separately below.
    monkeypatch.setattr(handoff, "_jobd_status_soft", lambda iid: "IDLE")
    hf = handoff._init_job_handoff_state()
    twin = {"id": "9100", "label": "job:9000:handoff",
            "actual_status": "running", "dph_total": 0.20}
    jctx = _jctx(io_, instances=[{"id": "9000", "label": "job:9000"}, twin])
    handoff._job_handoff_reconcile(jctx, hf)
    assert hf["phase"] == "SYNCED" and str(hf["understudy_iid"]) == "9100"
    assert hf["ckpt_pulled_epoch"] is not None
    assert ("handoff_reconciled", ) in [(e,) for e, _ in io_.events]


def test_job_handoff_reconcile_live_twin_without_boot_proof_warms(monkeypatch):
    # A twin that is API-live but whose jobd never stamped JOBD_STATUS (e.g. the
    # box is still `loading`) must resume at LAUNCHING, not SYNCED — absence of
    # a park marker is not evidence the box ever booted (live canary 2026-07-15).
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    monkeypatch.setattr(handoff, "_jobd_status_soft", lambda iid: None)
    hf = handoff._init_job_handoff_state()
    twin = {"id": "9100", "label": "job:9000:handoff",
            "actual_status": "loading", "dph_total": 0.20}
    jctx = _jctx(io_, instances=[{"id": "9000", "label": "job:9000"}, twin])
    handoff._job_handoff_reconcile(jctx, hf)
    assert hf["phase"] == "LAUNCHING"
    assert hf["ckpt_pulled_epoch"] is None


def test_job_handoff_reconcile_adopted_twin_retargets_at_cutover(monkeypatch):
    # S1 (2026-07-18 review): reconcile runs at the TOP of the supervise loop,
    # before the tick populates jctx["pending_jobs"] — the adopted twin used to
    # inherit an empty snapshot and the cutover retargeted ZERO tickets ("completing"
    # a migration that moved nothing; jobs stranded on the parked husk). The fence
    # must re-snapshot from the live queue.
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a", "job-b"])
    monkeypatch.setattr(handoff, "_jobd_status_soft", lambda iid: "IDLE")
    hf = handoff._init_job_handoff_state()
    twin = {"id": "9100", "label": "job:9000:handoff",
            "actual_status": "running", "dph_total": 0.20}
    # reconcile-at-boot reality: jctx has NO pending_jobs yet
    jctx = _jctx(io_, instances=[{"id": "9000", "label": "job:9000"}, twin])
    del jctx["pending_jobs"], jctx["running_jobs"]
    handoff._job_handoff_reconcile(jctx, hf)
    assert hf["phase"] == "SYNCED" and hf["pending_jobs"] == []   # the empty adopt
    # ...then the loop populates jctx before the handoff tick, as cmd_job_supervise does
    jctx["pending_jobs"], jctx["running_jobs"] = ["job-a", "job-b"], ["job-a"]
    handoff._job_handoff_tick(jctx, hf)                    # SYNCED -> fence -> CUTOVER
    assert hf["phase"] == "CUTOVER"
    assert hf["pending_jobs"] == ["job-a", "job-b"]        # fence re-snapshot (the fix)
    io_.final_flush = True
    handoff._job_handoff_tick(jctx, hf)                    # CUTOVER -> retarget -> DRAINING
    assert hf["phase"] == "DRAINING"
    assert io_.retargeted == ["job-a", "job-b"]            # tickets actually MOVED


def test_job_handoff_fast_cutover_falls_back_to_live_queue(monkeypatch):
    # S1 belt: fast_cutover (primary evicted at SYNCED) SKIPS fence_primary, so
    # the reconcile-adopted empty snapshot must be back-filled from the tick's
    # live queue inside resume_understudy itself.
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    hf = handoff._init_job_handoff_state()
    hf.update({"phase": "SYNCED", "understudy_iid": "9100", "primary_iid": "9000",
               "ckpt_pulled_epoch": 1, "pending_jobs": [], "running_jobs": []})
    jctx = _jctx(io_, primary_evicted=True)                # debounced verdict
    handoff._job_handoff_tick(jctx, hf)                    # SYNCED -> fast_cutover
    assert hf["phase"] == "DRAINING" and hf["primary_gone"] is True
    assert io_.retargeted == ["job-a"]                     # moved despite empty snapshot
    # S2 belt: the outbid husk's standing bid is pinned so a receding floor
    # can't auto-resume a box whose tickets just moved.
    assert ("9000", bidpolicy.HANDOFF_PARK_BID) in io_.pinned


def test_job_primary_evicted_debounced_pure():
    # S2 (2026-07-18 review): a single not-live blip (resume flap / transient
    # instances-API miss) must NOT read as an eviction — that reaped a warming
    # understudy into a 1800s cooldown, or fast-cutover'd off a live primary.
    f = job_lane._job_primary_evicted
    assert not f(True, True, 0)                            # live: never
    assert not f(True, False, 0)                           # first not-live tick: blip
    assert not f(False, False, 0)                          # first absent tick: blip
    assert f(True, False, bidpolicy.NOT_LIVE_DEBOUNCE - 1)   # persisted: evicted
    assert f(False, False, bidpolicy.NOT_LIVE_DEBOUNCE - 1)  # persisted absence: evicted


def test_job_primary_shape_falls_back_to_arm_snapshot():
    # P5 (2026-07-18 review): with the primary missing from the tick's instance
    # snapshot, understudy sizing must come from the shape cached at ARM — not
    # default to a 1-GPU/120GB/default-image box for a multi-GPU job.
    shape = {"gpu_name": "RTX 5090", "num_gpus": 4,
             "disk_space": 300, "image_uuid": "img-x"}
    jctx = {"iid": "9000", "instances": []}                # snapshot miss
    assert models._job_primary_shape(jctx, {"primary_shape": shape}) == shape
    assert models._job_primary_shape(jctx, {}) is None    # pre-ARM: no fallback
    live = {"id": "9000", "gpu_name": "H100", "num_gpus": 8}
    jctx["instances"] = [live]
    assert models._job_primary_shape(jctx, {"primary_shape": shape}) is live


def test_job_handoff_understudy_unlaunchable_aborts(monkeypatch):
    # launch fails outright -> immediate abort_reap (no understudy to reap), primary
    # untouched (I2: the primary keeps running under the normal ladder).
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    io_.launch_iid = None                                  # _launch_job_understudy -> (None,None)
    hf = handoff._init_job_handoff_state()
    jctx = _jctx(io_)
    _drive_to_armed(io_, jctx, hf)
    handoff._job_handoff_tick(jctx, hf)                    # ARMED -> launch fails -> abort
    assert hf["phase"] == "IDLE" and io_.retargeted == [] and io_.parked == []
    assert ("handoff_abort", ) in [(e,) for e, _ in io_.events]


def test_job_handoff_stall_alarm_emits_once(monkeypatch):
    # F2 wiring (jobs lane): a DRAINING box HANDOFF_DEADLINE_S past the fence with no
    # producing proof emits handoff_stall exactly once via _job_handoff_tick — and
    # NEVER force-destroys the primary on the timer (proof-of-life gate stands).
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    io_.producing = False                                  # understudy never proves producing
    hf = handoff._init_job_handoff_state()
    hf.update({"phase": "DRAINING", "understudy_iid": "9100", "primary_iid": "9000",
               "fence_ts": 1000.0, "pending_jobs": ["job-a"], "running_jobs": ["job-a"]})
    jctx = _jctx(io_, now=1000.0 + bidpolicy.HANDOFF_DEADLINE_S)
    handoff._job_handoff_tick(jctx, hf)
    jctx["now"] += 100
    handoff._job_handoff_tick(jctx, hf)
    stalls = [kw for e, kw in io_.events if e == "handoff_stall"]
    assert len(stalls) == 1 and stalls[0]["phase"] == "DRAINING"
    assert io_.destroyed == []                             # NO forced destroy on the timer


# --- F5: jobs-lane pre-launch re-validation + no_offer vs unlaunchable ---------
def test_launch_job_understudy_no_offer_reason(monkeypatch):
    # no qualifying offer -> reason 'no_offer' (distinct from unlaunchable).
    monkeypatch.setattr(replacement, "_job_understudy_offer", lambda jctx, hf=None: None)
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0}
    cid, dph, reason = replacement._launch_job_understudy(jctx, {}, epoch=1)
    assert cid is None and reason == "no_offer"


def test_launch_job_understudy_stale_offer_rechecked_before_spend(monkeypatch):
    # belt-and-suspenders: the picked offer no longer clears the §2.3 filter (its
    # floor climbed vs its on-demand between ARM and launch) -> abort as no_offer,
    # WITHOUT ever attempting the launch (no spend).
    # Fixture raised min_bid 0.50 -> 0.70 at the 2026-08-09 return to a 1.20x
    # multiple: the refusal is target > 0.75 x primary on-demand ($0.825), and
    # 1.2 x 0.50 = $0.60 now clears it (2.0 x 0.50 = $1.00 did not). 1.2 x 0.70
    # = $0.84 still refuses, exercising the same branch.
    launched = []
    monkeypatch.setattr(replacement, "_job_understudy_offer",
                        lambda jctx, hf=None: {"id": 1, "min_bid": 0.70, "dph_total": 0.80})
    monkeypatch.setattr(launch_mod, "_do_launch",
                        lambda ns: launched.append(ns) or ("X", 1, 0.5))
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0}
    cid, dph, reason = replacement._launch_job_understudy(jctx, {}, epoch=1)
    assert cid is None and reason == "no_offer"
    assert launched == []                                  # never spent on a stale offer


def test_launch_job_understudy_launch_failure_is_unlaunchable(monkeypatch):
    # a viable offer but the launch call fails -> reason 'understudy_unlaunchable'.
    monkeypatch.setattr(replacement, "_job_understudy_offer",
                        lambda jctx, hf=None: {"id": 1, "min_bid": 0.10, "dph_total": 1.0})
    def _boom(ns):
        raise SystemExit("no allocation")
    monkeypatch.setattr(launch_mod, "_do_launch", _boom)
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0}
    cid, dph, reason = replacement._launch_job_understudy(jctx, {}, epoch=1)
    assert cid is None and reason == "understudy_unlaunchable"


def test_launch_job_understudy_sets_no_job_deadline_ttl(monkeypatch):
    # F6: the understudy launch ns carries no_job_deadline == HANDOFF_TTL_S so jobd's
    # existing no-job park enforces the same TTL as the run-lane watchdog.
    seen = {}
    monkeypatch.setattr(replacement, "_job_understudy_offer",
                        lambda jctx, hf=None: {"id": 1, "min_bid": 0.10, "dph_total": 1.0})
    monkeypatch.setattr(launch_mod, "_do_launch",
                        lambda ns: (seen.update(ns=ns), ("9100", 1, 0.12))[1])
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0, "now": 1000.0}
    cid, dph, reason = replacement._launch_job_understudy(jctx, {}, epoch=1)
    assert cid == "9100" and reason is None
    assert seen["ns"].no_job_deadline == bidpolicy.HANDOFF_TTL_S == 2700


def test_launch_job_understudy_prices_from_offer_dict(monkeypatch):
    # D8 (live jobs-lane canary 2026-07-15): the understudy launch ns must carry a
    # concrete price computed from the offer dict in hand, NOT price=None (which
    # made _do_launch re-price via the structurally-dead v0/bundles id-filter ->
    # understudy_unlaunchable on every jobs-lane handoff).
    seen = {}
    monkeypatch.setattr(replacement, "_job_understudy_offer",
                        lambda jctx, hf=None: {"id": 1, "min_bid": 0.10, "dph_total": 1.0})
    monkeypatch.setattr(launch_mod, "_do_launch",
                        lambda ns: (seen.update(ns=ns), ("9100", 1, 0.12))[1])
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0, "now": 1000.0}
    cid, dph, reason = replacement._launch_job_understudy(jctx, {}, epoch=1)
    assert cid == "9100" and reason is None
    assert seen["ns"].type == "bid"
    assert seen["ns"].price == 0.12                        # 1.2x floor, under on-demand
    # (2.00-era value: 0.20; BID_TARGET_MULT back to 1.20 per 2026-08-09 ruling)
    assert seen["ns"].price is not None                    # never defer to the dead id-filter


def test_launch_job_understudy_unwinnable_offer_is_no_offer(monkeypatch):
    # D7/D8: a razor-thin market whose floor is at/over the machine's on-demand
    # yields no winnable bid price -> abort as no_offer BEFORE any launch (no
    # spend). Since the doc 50 R1 family fix the on-demand reference is the
    # LIVE MARKET read, never the bid row's own dph_total (which is the
    # interruptible price) — so the razor market is modeled by the probe.
    launched = []
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, num_gpus=None: 0.30)
    monkeypatch.setattr(replacement, "_job_understudy_offer",
                        lambda jctx, hf=None: {"id": 1, "machine_id": 7,
                                               "min_bid": 0.30, "dph_total": 0.3012})
    monkeypatch.setattr(launch_mod, "_do_launch",
                        lambda ns: launched.append(ns) or ("X", 1, 0.3))
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0, "now": 1000.0}
    cid, dph, reason = replacement._launch_job_understudy(jctx, {}, epoch=1)
    assert cid is None and reason == "no_offer"
    assert launched == []


def test_job_handoff_unlaunchable_abort_reason_propagates(monkeypatch):
    # the driver maps _launch_job_understudy's reason 1:1 onto the abort reason.
    io_ = _JobHandoffIO(monkeypatch, queue=["job-a"])
    io_.launch_iid = None                                  # stub returns unlaunchable reason
    hf = handoff._init_job_handoff_state()
    jctx = _jctx(io_)
    _drive_to_armed(io_, jctx, hf)
    handoff._job_handoff_tick(jctx, hf)                    # ARMED -> launch fails -> abort
    abort = [kw for e, kw in io_.events if e == "handoff_abort"]
    assert abort and abort[-1]["reason"] == "understudy_unlaunchable"


# --- defect #63: the horizon the DRIVER hands the filter, and the deferral ----
# The pure filter above was never wrong; `job_supervise_tick` fed it a fabricated
# 24.0 h. These drive the REAL tick (not `_job_handoff_tick`, which cannot see
# where the number comes from) with the 2026-08-08 market on the wire.
#
# PRIMARY DPH, 2026-08-08 (autobid displacement audit). The incident's own
# $0.830 was over the THEN preferred ceiling of 0.50 x $1.2667 = $0.633. That
# ceiling moved to BID_CEILING_ONDEMAND_FRAC = 0.75 (it must stay above the new
# 0.65 x on-demand standing-bid target, or every fresh box latches the breach),
# so $0.830 is now UNDER the line and this fixture would never dwell — the
# deferral these tests exist to pin would never be reached.
#
# The dph is therefore derived from the live constant rather than frozen, one
# cent over the ceiling, so the breach is structural and cannot silently lapse
# again the next time the ceiling moves. `_INCIDENT` keeps the recorded numbers.
#
# The market has to HOLD the bid over the line, not merely start it there: the
# decay ladder aims at the same standing target every tick, so a dph parked
# above the ceiling by hand is walked back under it within BID_DECAY_POLLS and
# the dwell never completes. Only a TIGHT machine keeps a bid up there now —
# one whose floor is high enough that the survival cushion (1.10 x floor)
# outranks the 0.65 x on-demand cost cap. That is exactly the regime handoff
# still exists for, and the audit says so: on a machine where the cost cap
# binds, the cap already does handoff's job without a second box.
#
# 2026-08-09 (recalibration item A) moved the fixture again, and the reason is
# worth stating because it narrows handoff's reachable domain a SECOND time. The
# preferred ceiling is now a HARD clamp on every emitted bid
# (`bidpolicy.effective_bid_ceiling`): a cushion that does not fit under it is an
# ESCALATION, not a bid. So the tight-machine regime this fixture used to sit in
# no longer produces a breaching bid — it produces no bid at all — and the policy
# can no longer place a bid above the preferred ceiling from ANY path.
#
# What survives, and what this fixture is now: a standing bid that is above the
# line but that the CURRENT policy would not have placed — a bid from before the
# clamp existed, a `herdd bid --price` by hand, or an on-demand price that has
# since fallen. That is exactly the shape `_handoff_trigger` was rewritten for
# ("bid over the ceiling AND over what the policy would put right now"), so the
# fixture keeps the same two numbers and only stops deriving the bid from a call
# that now refuses.
_HORIZON_FLOOR = round(
    bidpolicy.BID_CEILING_ONDEMAND_FRAC * _INCIDENT["primary_on_demand"]
    / bidpolicy.BID_MIN_CUSHION_MULT * 1.02, 4)
# The bid the PRE-2026-08-09 policy emitted at that floor (the survival cushion,
# which outranked the cost cap and had nothing but `on_demand - EPS` under it).
# Written longhand so it stays a legible historical number; `_bid_target` at this
# floor now returns None, and the assertion below pins that.
_HORIZON_PRIMARY_DPH = round(
    bidpolicy.BID_MIN_CUSHION_MULT * _HORIZON_FLOOR, 3)
assert _HORIZON_PRIMARY_DPH > bidpolicy._preferred_ceiling(
    _INCIDENT["primary_on_demand"]), "fixture must breach the preferred ceiling"
assert bidpolicy._bid_target(_HORIZON_FLOOR, None,
                           _INCIDENT["primary_on_demand"]) is None, (
    "the hard ceiling must refuse to EMIT this bid — a policy that can still "
    "place it has lost the item-A clamp")

_HORIZON_INST = {"id": 9000, "actual_status": "running",
                 "intended_status": "running", "machine_id": 7,
                 "dph_total": _HORIZON_PRIMARY_DPH, "num_gpus": 2,
                 "is_bid": True, "label": "job:9000", "gpu_name": "RTX 4090"}

# The candidate's standing-bid target and the two derived numbers the deferral
# must journal, written out longhand from HANDOFF_DESIGN §2.3 (overhead is the
# 2x-box window; delta is the per-hour saving) rather than read back from the
# driver, so the assertions still test arithmetic and are not a restatement of
# the code under test.
_HORIZON_CAND_TARGET = bidpolicy._bid_target(
    _INCIDENT["candidate_min_bid"], None, _INCIDENT["candidate_on_demand"])
_HORIZON_DELTA_DPH = round(_HORIZON_PRIMARY_DPH - _HORIZON_CAND_TARGET, 4)
_HORIZON_OVERHEAD = round(
    (_HORIZON_PRIMARY_DPH + _HORIZON_CAND_TARGET) * bidpolicy.HANDOFF_WINDOW_H, 4)


# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_sticky_on_demand` landed at
# `vastlib.market.pricing` — the home job_lane.py's seam comment named — so the
# tick reaches it as `pricing._sticky_on_demand` and this fixture moves with its
# drivers. Seams sit where the tick RESOLVES them: `lifecycle.<name>` for the
# instance read and the bid PUT, `pricing.<name>` for the market reads and
# `_offer_ondemand_ref`, `retention.<name>` for the sweep, `replacement.<name>`
# for the SLA tick / understudy offer / understudy launch, and bare in
# `job_lane` for `_box_lifecycle_soft` (still a raising SEAM stub there; body at
# `vastlib.jobs.view`).
def _horizon_env(monkeypatch, views, *, floor=None, bid_put_ok=True):
    """Stub every seam `job_supervise_tick` touches so the box is live, over its
    preferred ceiling, and looking at the incident's qualifying cheap offer —
    i.e. everything the 2026-08-08 handoff had EXCEPT a measured horizon. Renting
    a second box is wired to explode: at this horizon nothing may be launched.

    `floor` and `bid_put_ok` exist for the trigger-domain change (2026-08-08): at
    the DEFAULT floor the standing bid IS `_bid_target(floor, ...)`, so the
    handoff correctly never arms and the ECONOMIC gate below it is unreachable.
    A test that wants to exercise the economics has to put the box in the one
    state where a bid genuinely sits above its own policy target — a receded
    floor whose decay PUT is not landing — and say so."""
    m = monkeypatch
    _floor = _HORIZON_FLOOR if floor is None else floor
    m.setattr(lifecycle, "_instances_soft", lambda: [dict(_HORIZON_INST)])
    m.setattr(jobmeta, "list_queue", lambda box, **k: [v["job_id"] for v in views])
    m.setattr(jobmeta, "read_job",
              lambda jid, **k: next(dict(v) for v in views if v["job_id"] == jid))
    m.setattr(jobmeta, "emit_box_event",
              lambda iid, ev, **kw: {"_emitted": True})
    m.setattr(job_lane, "_box_lifecycle_soft",
              lambda iid: {"parked": False, "drained_pending": False})
    m.setattr(retention, "_job_retention_sweep", lambda jc, now: None)
    m.setattr(replacement, "_job_boot_sla_tick", lambda jc, inst, now: None)
    m.setattr(pricing, "_market_min_bid_soft", lambda mid, g=None: _floor)
    m.setattr(pricing, "_market_min_bid_read",
              lambda mid, g=None: models.MarketRead(True, True, _floor))
    m.setattr(pricing, "_market_ondemand_soft",
              lambda mid, n=None: _INCIDENT["primary_on_demand"])
    m.setattr(replacement, "_job_understudy_offer",
              lambda jc, hf=None, **k: {"id": 1, "machine_id": 8,
                                        "min_bid": _INCIDENT["candidate_min_bid"]})
    m.setattr(pricing, "_offer_ondemand_ref",
              lambda offer, num_gpus=None: _INCIDENT["candidate_on_demand"])
    m.setattr(lifecycle, "_put_bid_soft",
              lambda iid, bid: (True, None) if bid_put_ok
              else (False, "stub: bid PUT refused"))
    m.setattr(replacement, "_launch_job_understudy",
              lambda *a, **k: (_ for _ in ()).throw(
                  AssertionError("a bounded horizon must never rent a second box")))


def _horizon_ns(**over):
    a = argparse.Namespace(id=9000, dry_run=False, budget=100.0, max_bid=None,
                           handoff=True, strict_ceiling=False, rescue_wait=None,
                           keep=False, wall_budget=None,
                           # this harness drives `job_supervise_tick` in a loop,
                           # i.e. a driver that CAN finish a migration (defect
                           # #61's precondition). Without it every case below
                           # would refuse on the precondition and stop testing
                           # the economics they were written for.
                           handoff_can_complete=True)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _hms(s):
    return f"{int(s) // 3600}:{(int(s) % 3600) // 60:02d}:{int(s) % 60:02d}"


def _tqdm_tail(step, elapsed_s, *, total=100, step_delta=60.0, rate=1.0):
    """Two consecutive training bars, `step_delta` seconds apart — the ONLY
    honest step time available without SSH (`_step_delta_s`). Remaining steps x
    that delta is the ETA the handoff now prices against."""
    def bar(st, el):
        return (f"\r {int(100 * st / total):>3}%|##        | {st}/{total} "
                f"[{_hms(el)}<1:00:00, {rate:.2f}s/it]")
    return bar(step - 1, elapsed_s - step_delta) + "\n" + bar(step, elapsed_s)


def _now_epoch():
    return _dt.datetime.now(_dt.timezone.utc).timestamp()


class _TickClock:
    """The jobs lane's own `time.time()`, advanced ONE tick interval per
    `job_supervise_tick`.

    A loop that never moves the clock can never satisfy a dwell expressed as a
    DURATION, and that is the point: the tick interval is now an input to when a
    gate fires, so a test that exercises one has to state it."""

    def __init__(self, monkeypatch, tick_s, start=None):
        self.t = float(_now_epoch() if start is None else start)
        self.tick_s = float(tick_s)
        monkeypatch.setattr(time, "time", lambda: self.t)

    def tick(self, fn, *a):
        out = fn(*a)
        self.t += self.tick_s
        return out


def test_jobs_tick_ceiling_comes_from_the_queue_not_a_fabricated_day(monkeypatch):
    """P0-a at the driver, as landed for defect #63 and still true: fleetd seeds
    `wall_budget=None`, so this is the shape EVERY fleetd jobs watch runs in, and
    the timeout-derived bound must be MEASURED (2700 s left of a 4200 s ticket =
    0.75 h) and never a flat 24.0.

    What moved on 2026-08-08 (defect #67) is which bound the handoff PRICES
    against. `timeout_s` is a hang detector; it is published as
    `timeout_ceiling_h` and used as a cap. `remaining_wall_h` is now the work
    estimate, and this ticket has no progress signal at all — so it is None
    (UNKNOWN), which refuses. Assuming the ceiling was the whole defect: the
    incident's ticket declared `timeout_s: 36000` and had run 345 s, which
    published a 9.904 h 'horizon' on ~1-2 h of real work."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=4200, started_at=_ts(t0 - 1500))]
    _horizon_env(monkeypatch, views)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert jc["timeout_ceiling_h"] == pytest.approx(0.75, abs=0.01)
    assert jc["remaining_wall_h"] is None          # no progress signal -> UNKNOWN


def test_jobs_tick_horizon_is_the_progress_eta_not_the_timeout_ceiling(monkeypatch):
    """Defect #67, at the driver, on the incident's own shape: a LONG hang
    detector over a SHORT piece of remaining work. The ticket declares
    `timeout_s: 36000` (10 h) and is 345 s in — the incident's numbers, which
    published `remaining_wall_h: 9.904`. Its bar says 90 of 100 steps at 60 s a
    step, i.e. ten minutes left. The tick must price the migration against the
    ten minutes."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=36000, started_at=_ts(t0 - 345),
                           last_tail=_tqdm_tail(90, 5400, total=100,
                                                step_delta=60.0))]
    _horizon_env(monkeypatch, views)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert jc["timeout_ceiling_h"] == pytest.approx(9.904, abs=0.01)   # the old number
    assert jc["remaining_wall_h"] == pytest.approx(600 / 3600.0, abs=0.01)
    # ...and the 5x-inflated projection the old number produced is exactly why:
    assert jc["timeout_ceiling_h"] / jc["remaining_wall_h"] > 50


#: A bid genuinely ABOVE its own policy target — the one regime left where the
#: economic gate is even reachable (see `_horizon_env`): the floor receded to a
#: third and the decay PUT is not landing, so our standing bid is stale-high
#: rather than policy-high. Anything the ladder CAN fix by lowering the bid, it
#: fixes by lowering the bid; a handoff is for what it cannot.
_STALE_HIGH_FLOOR = round(_HORIZON_FLOOR / 3.0, 4)


def test_jobs_tick_defers_the_handoff_and_journals_the_arithmetic(monkeypatch):
    """P0-a + P0-b. Dwell is satisfied, the bid really is above what the policy
    would put, and the offer is genuinely cheaper — but at the real horizon the
    migration cannot pay for itself: the tick stays IDLE, rents nothing, and
    leaves ONE deferral (with its numbers) on the handoff journal fleetd drains.
    The incident was invisible in `fleet log` precisely because nothing was
    written when the ladder made a money decision."""
    t0 = _now_epoch()
    # started 1500 s before the ARMING poll, not before the first one: the dwell
    # is HANDOFF_DWELL_S of wall clock, and the ticket's timeout ceiling decays
    # with it — without the shift the ceiling (not the measured estimate) would
    # be the binding bound and this test would stop testing measurement.
    views = [_pending_view(timeout_s=4200,
                           started_at=_ts(t0 - 1500 + bidpolicy.HANDOFF_DWELL_S),
                           n_checkpoints=3,      # resumable: the work rails pass
                           last_tail=_tqdm_tail(55, 3300, total=100,
                                                step_delta=60.0))]
    _horizon_env(monkeypatch, views, floor=_STALE_HIGH_FLOOR, bid_put_ok=False)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    clock = _TickClock(monkeypatch, 45.0)     # the dwell is a duration: 5 polls
    for _ in range(bidpolicy.HANDOFF_DWELL_POLLS + 2):   # x 45 s == the old count
        assert clock.tick(job_lane.job_supervise_tick, jc, hf) is None
    assert hf["phase"] == "IDLE"                     # never armed, nothing rented
    deferrals = [(ev, f) for ev, f in jc.get("handoff_journal") or []
                 if ev == "deferred"]
    assert len(deferrals) == 1, f"expected ONE deferral, got {deferrals}"
    kind, fields = deferrals[0]
    assert "HANDOFF DEFERRED" in fields["note"] and "9000" in fields["note"]
    # the refusal has to carry its own arithmetic, the standard every other
    # money decision on this ladder is held to (_job_eviction_replace's prints).
    # 45 steps x 60 s = 2700 s of measured work left — the same number the
    # timeout ceiling happened to give here, now MEASURED instead of assumed.
    assert fields["horizon_s"] == pytest.approx(2700, abs=60)
    assert fields["overhead_usd"] == pytest.approx(_HORIZON_OVERHEAD, abs=1e-3)
    assert fields["delta_dph"] == pytest.approx(_HORIZON_DELTA_DPH, abs=1e-3)
    assert all(tok in fields["note"] for tok in
               (f"${_HORIZON_OVERHEAD:.3f}", f"${_HORIZON_DELTA_DPH:.3f}",
                "s of horizon left"))
    # and the deferral must still be a REFUSAL: the 2x-box window costs more
    # than the migration can recover in the horizon that is left.
    assert _HORIZON_OVERHEAD > _HORIZON_DELTA_DPH * (fields["horizon_s"] / 3600.0)


def test_jobs_tick_refuses_over_an_uncheckpointed_running_job_and_says_so(monkeypatch):
    """Defect #62 at the driver, on the incident's shape: a RUNNING ticket at
    `n_checkpoints: 0`. Everything else says go — the bid is stale-high, the
    horizon is measured and long, the offer is cheap — and the tick still refuses,
    rents nothing, and leaves ONE refusal on the journal fleetd drains."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=36000, started_at=_ts(t0 - 345),
                           n_checkpoints=0,
                           last_tail=_tqdm_tail(5, 300, total=100,
                                                step_delta=60.0))]
    _horizon_env(monkeypatch, views, floor=_STALE_HIGH_FLOOR, bid_put_ok=False)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    clock = _TickClock(monkeypatch, 45.0)     # the dwell is a duration: 5 polls
    for _ in range(bidpolicy.HANDOFF_DWELL_POLLS + 2):   # x 45 s == the old count
        assert clock.tick(job_lane.job_supervise_tick, jc, hf) is None
    assert hf["phase"] == "IDLE"
    refusals = [f for ev, f in jc.get("handoff_journal") or [] if ev == "refused"]
    assert len(refusals) == 1, refusals
    assert refusals[0]["reason"] == "unresumable_running_job"
    assert "NO checkpoint" in refusals[0]["note"]


def test_jobs_tick_warns_once_past_the_completion_threshold(monkeypatch):
    """HANDOFF_WARN_PCT advisory (task #67). Percent is a warning unit, never a
    blocking one — the line lands on the journal and nothing else changes."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=36000, started_at=_ts(t0 - 345),
                           n_checkpoints=0,
                           last_tail=_tqdm_tail(95, 5700, total=100,
                                                step_delta=60.0))]
    _horizon_env(monkeypatch, views, floor=_STALE_HIGH_FLOOR, bid_put_ok=False)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    for _ in range(3):
        assert job_lane.job_supervise_tick(jc, hf) is None
    warns = [f for ev, f in jc.get("handoff_journal") or [] if ev == "work_warning"]
    assert len(warns) == 1, warns                     # once per job per condition
    assert warns[0]["pct"] >= bidpolicy.HANDOFF_WARN_PCT
    assert warns[0]["n_checkpoints"] == 0
    assert "NO checkpoint" in warns[0]["note"]        # the EVICTION exposure too


def test_unfence_never_resumes_a_box_at_the_fence_pin(monkeypatch):
    """Task #62: the $0.001 pin must not outlive the fence on ANY path. With no
    pre-fence bid recorded and no policy target readable, resuming would leave a
    live box that can never win its market — so the box stays PARKED and the
    refusal is loud, and under no circumstances is $0.001 written back."""
    bids, states = [], []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, bid: (bids.append((str(iid), bid)), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (states.append((str(iid), st)), (True, None))[1])
    assert handoff._handoff_unfence_primary("9000", {}) is False
    assert bids == [] and states == []
    # ...and with a policy target it recovers rather than refusing
    assert handoff._handoff_unfence_primary("9000", {}, policy_target=2.55) is True
    assert bids == [("9000", 2.55)] and states == [("9000", "running")]


def test_jobs_tick_with_no_readable_ticket_bound_refuses_and_says_so(monkeypatch):
    """Fail CLOSED: a queue that yields no horizon at all publishes None (never a
    default day, and never the hang-detector ceiling), which the candidate filter
    refuses on."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=None, started_at=_ts(t0 - 1500))]
    _horizon_env(monkeypatch, views)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    for _ in range(bidpolicy.HANDOFF_DWELL_POLLS + 1):
        assert job_lane.job_supervise_tick(jc, hf) is None
    assert jc["remaining_wall_h"] is None
    assert hf["phase"] == "IDLE"


def test_jobs_tick_wall_budget_still_caps_the_work_horizon(monkeypatch):
    """--wall-budget keeps its old meaning: it is one of the bounds, and the
    tightest wins. 4 h of measured work behind a 0.5 h wall budget is 0.5 h."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=5 * 3600, started_at=_ts(t0 - 60),
                           last_tail=_tqdm_tail(20, 4800, total=100,
                                                step_delta=180.0))]
    _horizon_env(monkeypatch, views)
    jc, hf = job_lane.job_supervise_init(_horizon_ns(wall_budget=1800))
    assert job_lane.job_supervise_tick(jc, hf) is None
    # raw work estimate is 80 steps x 180 s = 4 h; the wall budget caps it at 0.5
    assert jc["remaining_wall_h"] == pytest.approx(0.5, abs=0.01)


# --- F3: reap-on-exit guaranteed on EVERY cmd_job_supervise exit path ----------
# MIGRATED (was MIGRATION-BLOCKED, step 6e): `_sticky_on_demand` landed at
# `vastlib.market.pricing`, so the `unrecoverable` exit path no longer dies inside
# the port and the fixture moves with its three drivers. Subject is
# `cli.job.supervise.cmd_job_supervise` — the same body, reached through the cli
# ring rather than the launcher — and each seam is stubbed where the jobs tick
# resolves it. `config.load_env` is kept (harmless here: these drive the command
# function directly, and only `cli.main.main()` calls it) so the fixture still
# refuses to read a real `.env` if the prologue ever moves.
def _job_sup_exit_env(monkeypatch, *, seed_phase, understudy, inst,
                      box_lifecycle=None, jids=(), view=None, prefence_bid=None):
    """Stub cmd_job_supervise's I/O to hit ONE exit deterministically, with a seeded
    pre/post-cutover understudy in hf (via the reconcile hook; the tick is frozen so
    the phase stays put). Returns the list the REAL _job_handoff_reap_on_exit
    appends destroyed iid(s) to — the finally-clause contract under test. `view` is
    the job status template returned for EVERY read_job (persists across ticks; None
    => terminal 'done')."""
    destroyed = []
    m = monkeypatch
    m.setattr(config, "load_env", lambda: None)
    m.setattr(b2, "_ensure_b2_remote", lambda: None)
    m.setattr(time, "sleep", lambda s: None)
    m.setattr(lifecycle, "_instances_soft", lambda: ([dict(inst)] if inst else []))
    m.setattr(job_lane, "_box_lifecycle_soft",
              lambda iid: dict(box_lifecycle or {"parked": False, "drained_pending": False}))
    _view = view or {"status": "done", "display_status": "done"}
    m.setattr(jobmeta, "list_queue", lambda box, **k: list(jids))
    m.setattr(jobmeta, "read_job", lambda j, **k: {**_view, "job_id": j})
    m.setattr(pricing, "_market_min_bid_soft", lambda mid, g=None: None)
    m.setattr(pricing, "_market_min_bid_read",
              lambda mid, g=None: models.MarketRead(False, False, None))
    m.setattr(pricing, "_market_ondemand_soft", lambda mid, n=None: None)
    m.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda s: (False, None))
    m.setattr(jobs_risk, "_ckpt_watchdog_alarm", lambda v, now: None)
    m.setattr(lifecycle, "_stop_instance_soft", lambda iid: True)
    m.setattr(lifecycle, "_destroy_soft",
              lambda iid, dry_run=False, **k: (destroyed.append(iid), (True, None))[1])
    m.setattr(lifecycle, "_confirm_gone", lambda iid, **k: True)
    m.setattr(jobmeta, "emit_box_event", lambda *a, **k: {"_emitted": True})
    m.setattr(handoff, "_job_handoff_reconcile",
              lambda jctx, hf: hf.update(phase=seed_phase, understudy_iid=understudy,
                                         primary_iid=jctx.get("iid"),
                                         prefence_bid=prefence_bid))
    m.setattr(handoff, "_job_handoff_tick", lambda jctx, hf: None)
    return destroyed


def _job_sup_ns(**over):
    a = argparse.Namespace(id=9000, dry_run=False, budget=5.0, max_bid=None,
                           handoff=True, strict_ceiling=False, rescue_wait=None,
                           keep=False)
    for k, v in over.items():
        setattr(a, k, v)
    return a


_JS_LIVE = {"id": "9000", "actual_status": "running", "dph_total": 1.0,
            "is_bid": True, "machine_id": 1, "num_gpus": 1}
_JS_STOPPED = {"id": "9000", "actual_status": "stopped", "dph_total": 1.0,
               "is_bid": True, "machine_id": 1, "intended_status": "running"}
_JS_OPPARK = {"id": "9000", "actual_status": "running", "dph_total": 1.0,
              "is_bid": False, "machine_id": 1, "intended_status": "stopped"}
_JS_RUNVIEW = {"status": "running", "display_status": "running"}
_JS_DONEVIEW = {"status": "done", "display_status": "done"}


@pytest.mark.parametrize("scenario", ["queue_empty", "queue_drained", "budget",
                                      "self_parked", "operator_park", "unrecoverable"])
def test_job_supervise_reaps_pre_cutover_on_every_exit(monkeypatch, scenario):
    cfg = {
        "queue_empty":  dict(inst=_JS_LIVE, jids=[]),
        "queue_drained": dict(inst=_JS_LIVE, jids=["job-a"], view=_JS_DONEVIEW),
        "budget":       dict(inst=_JS_LIVE, jids=["job-a"], view=_JS_RUNVIEW),
        "self_parked":  dict(inst=_JS_STOPPED, jids=[],
                             box_lifecycle={"parked": True, "drained_pending": False,
                                            "park_reason": "drain"}),
        "operator_park": dict(inst=_JS_OPPARK, jids=[]),
        "unrecoverable": dict(inst=None, jids=["job-a"], view=_JS_RUNVIEW),
    }[scenario]
    d = _job_sup_exit_env(monkeypatch, seed_phase="SYNCED", understudy="9100", **cfg)
    a = _job_sup_ns(budget=(0.0 if scenario == "budget" else 5.0))
    if scenario == "unrecoverable":
        with pytest.raises(SystemExit):
            cli_job_supervise.cmd_job_supervise(a)
    else:
        cli_job_supervise.cmd_job_supervise(a)
    # the finally reaped the PRE-cutover understudy on this exit path (no leaked box).
    assert d == ["9100"], scenario


def test_job_supervise_does_not_reap_a_draining_understudy(monkeypatch):
    # DRAINING means the cutover COMMITTED: the tickets are on the understudy,
    # which is now the canonical box. The phase-guarded reaper must leave it,
    # even as the loop exits on the budget cap.
    d = _job_sup_exit_env(monkeypatch, seed_phase="DRAINING", understudy="9100",
                          inst=_JS_LIVE, jids=["job-a"], view=_JS_RUNVIEW)
    cli_job_supervise.cmd_job_supervise(_job_sup_ns(budget=0.0))
    assert d == []


def test_job_supervise_unwinds_an_open_fence_on_exit(monkeypatch):
    """2026-08-08, task #62. CUTOVER was lumped in with DRAINING as
    'post-cutover, leave it', and that premise is wrong: at CUTOVER the retarget
    has NOT run, so the understudy owns nothing and the PRIMARY is parked with
    its bid pinned to $0.001 while still holding every ticket. A supervisor that
    exits there leaves exactly the incident's end state — an unwatched second box
    plus a primary that is off and cannot win a market. The exit must unwind the
    fence (restore the bid, resume the box) and reap the pre-writer understudy."""
    d = _job_sup_exit_env(monkeypatch, seed_phase="CUTOVER", understudy="9100",
                          inst=_JS_LIVE, jids=["job-a"], view=_JS_RUNVIEW,
                          prefence_bid=2.55)          # what `fence_primary` records
    bids, states = [], []
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, bid: (bids.append((str(iid), bid)), (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, st: (states.append((str(iid), st)), (True, None))[1])
    cli_job_supervise.cmd_job_supervise(_job_sup_ns(budget=0.0))
    assert d == ["9100"]                                  # understudy reaped
    assert ("9000", bidpolicy.HANDOFF_PARK_BID) not in bids  # never re-pinned
    assert bids and bids[-1][0] == "9000" and bids[-1][1] > bidpolicy.HANDOFF_PARK_BID
    assert ("9000", "running") in states                  # and un-parked


def _run_main(monkeypatch, argv, capture=None):
    """Drive cli_main.main() through the REAL argparse wiring with load_env/config
    stubbed. `capture` replaces cmd_job_supervise so we read the parsed namespace
    without running the loop."""
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    if capture is not None:
        monkeypatch.setattr(cli_job_supervise, "cmd_job_supervise", capture)
    monkeypatch.setattr(sys, "argv", ["herdd", *argv])
    cli_main.main()


def test_job_supervise_handoff_conflicts_with_strict_ceiling(monkeypatch):
    # the mutually-exclusive group must reject both flags together (argparse exits 2).
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["job", "supervise", "42", "--budget", "5",
                                "--handoff", "--strict-ceiling"])


def test_job_supervise_handoff_flag_parses(monkeypatch):
    got = {}
    _run_main(monkeypatch, ["job", "supervise", "42", "--budget", "5", "--handoff"],
              capture=lambda a: got.update(a=a))
    a = got["a"]
    assert a.handoff is True and getattr(a, "strict_ceiling", False) is False


# --- F7: train --wall-budget forwards to the child supervise (HOURS -> SECONDS) -
def _train_ns(**over):
    # handoff defaults True to mirror the real parser default (flipped 2026-07-15).
    a = argparse.Namespace(max_bid=None, strict_ceiling=False, handoff=True,
                           defend_at=None, rescue_wait=None, wall_budget=None)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_supervise_argv_forwards_wall_budget_as_seconds():
    argv = fleet_client._supervise_argv(_train_ns(wall_budget=6.0), "run-x", 50.0,
                                   None, None, None)
    assert "--wall-budget" in argv
    assert argv[argv.index("--wall-budget") + 1] == str(6.0 * 3600.0)   # HOURS->SECS
    # the mandatory --budget is always present.
    assert argv[argv.index("--budget") + 1] == "50.0"


def test_supervise_argv_omits_wall_budget_when_absent():
    # default handoff=True -> forwards --handoff explicitly (flip 2026-07-15).
    argv = fleet_client._supervise_argv(_train_ns(), "run-x", 50.0, None, None, None)
    assert "--wall-budget" not in argv
    assert "--handoff" in argv
    assert "--strict-ceiling" not in argv and "--no-handoff" not in argv


def test_supervise_argv_forwards_handoff_and_wall_budget_together():
    argv = fleet_client._supervise_argv(_train_ns(handoff=True, wall_budget=2.5),
                                   "run-x", 50.0, None, None, None)
    assert "--handoff" in argv
    assert argv[argv.index("--wall-budget") + 1] == str(2.5 * 3600.0)


def test_supervise_argv_forwards_no_handoff_when_disabled():
    # --no-handoff on train -> child gets --no-handoff (get-and-hold only)
    argv = fleet_client._supervise_argv(_train_ns(handoff=False), "run-x", 50.0,
                                   None, None, None)
    assert "--no-handoff" in argv
    assert "--handoff" not in argv and "--strict-ceiling" not in argv


def test_supervise_argv_forwards_strict_ceiling_over_handoff():
    # --strict-ceiling wins: forward it, never --handoff/--no-handoff
    argv = fleet_client._supervise_argv(_train_ns(strict_ceiling=True), "run-x", 50.0,
                                   None, None, None)
    assert "--strict-ceiling" in argv
    assert "--handoff" not in argv and "--no-handoff" not in argv


# --- F8: supervise --budget is required UNLESS --dry-run ----------------------
def test_supervise_dry_run_without_budget_parses(monkeypatch):
    # argparse no longer marks --budget required; a --dry-run supervise parses with
    # budget None (mirrors cmd_job_supervise).
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    got = {}
    monkeypatch.setattr(cli_supervise, "run", lambda a: got.update(a=a))
    monkeypatch.setattr(sys, "argv", ["herdd", "supervise", "run-x", "--dry-run"])
    cli_main.main()
    assert got["a"].budget is None and got["a"].dry_run is True


def test_supervise_live_without_budget_exits_post_parse():
    # a live (non-dry-run) supervise with no --budget errors post-parse (the hard
    # spend cap stays mandatory on any run that actually spends).
    a = argparse.Namespace(run_id="run-x", dry_run=False, budget=None)
    with pytest.raises(SystemExit) as e:
        cli_supervise.run(a)
    assert "budget" in str(e.value).lower()


# =============================================================================
# 8. Live-canary defect regressions (handoff-canary-2, 2026-07-15). D1 (the
#    fence park tripping operator_stop) is covered structurally: FakeAPI's
#    _process_v1 now reflects parks as intended/actual stopped, so every fence
#    scenario above re-proves it. Below: D2 (spec filter backfill), D3 (.synced
#    boot proof), D4 (producing on the no-SIGTERM flush_timeout path).
# =============================================================================
def test_handoff_run_signals_producing_via_cutover_ts_without_flush(monkeypatch):
    # O1 reality: vast delivers no SIGTERM on the fence park, so final_flush
    # NEVER lands and the flush-gated producing path is unreachable. A checkpoint
    # event after the cutover moment must prove the understudy producing.
    evs = [{"event": "handoff_cutover", "ts": "20260715T120000000Z"},
           {"event": "checkpoint", "ts": "20260715T120301000Z"}]
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: evs)
    sig = handoff._handoff_run_signals("r1", cutover_ts="20260715T120000")
    assert sig["understudy_producing"] is True
    assert sig["final_flush_seen"] is False


def test_handoff_run_signals_pre_cutover_checkpoint_not_producing(monkeypatch):
    # the primary's OWN last checkpoint (before the fence/cutover) must not
    # count as understudy proof-of-life.
    evs = [{"event": "checkpoint", "ts": "20260715T115900000Z"},
           {"event": "handoff_cutover", "ts": "20260715T120000000Z"}]
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: evs)
    sig = handoff._handoff_run_signals("r1", cutover_ts="20260715T120000")
    assert sig["understudy_producing"] is False


def test_handoff_run_signals_flush_path_still_works(monkeypatch):
    # legacy/post_flush path unchanged: flush then checkpoint => producing,
    # with no cutover_ts at all.
    evs = [{"event": "final_flush", "ts": "20260715T120000000Z"},
           {"event": "checkpoint", "ts": "20260715T120200000Z"}]
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda rid: evs)
    sig = handoff._handoff_run_signals("r1")
    assert sig["final_flush_seen"] is True
    assert sig["understudy_producing"] is True


def test_handoff_synced_epoch_soft_parses_max_and_fails_closed(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (0, "1.json\n1.synced\n2.synced\njunk.synced\npromoted\n", ""))
    assert handoff._handoff_synced_epoch_soft("r1") == 2
    # read failure / empty / no bucket => None (no proof, no SYNCED, no fence)
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "boom"))
    assert handoff._handoff_synced_epoch_soft("r1") is None
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (0, "1.json\n", ""))
    assert handoff._handoff_synced_epoch_soft("r1") is None
    monkeypatch.delenv("B2_BUCKET")
    assert handoff._handoff_synced_epoch_soft("r1") is None


def test_handoff_observe_understudy_requires_synced_marker(monkeypatch):
    # D3: a merely-live (even 'loading') twin must NOT stamp ckpt_pulled_epoch
    # until the box-side .synced proof exists for the ARMED epoch.
    st = {"run_id": "r1", "now": 100.0,
          "_instances": [{"id": 777, "label": "run:r1:handoff",
                          "actual_status": "loading", "dph_total": 0.05}]}
    hf = handoff._init_handoff_state()
    hf["phase"] = "LAUNCHING"
    hf["epoch"] = 1
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: None)
    handoff._handoff_observe_understudy(st, hf)
    assert hf["ckpt_pulled_epoch"] is None             # live but unproven
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: 1)
    handoff._handoff_observe_understudy(st, hf)
    assert hf["ckpt_pulled_epoch"] == 1                # proof present -> staged


def test_handoff_observe_understudy_stale_marker_from_prior_epoch(monkeypatch):
    # epoch-2 arm must not be satisfied by handoff-1's leftover 1.synced.
    st = {"run_id": "r1", "now": 100.0,
          "_instances": [{"id": 778, "label": "run:r1:handoff",
                          "actual_status": "running", "dph_total": 0.05}]}
    hf = handoff._init_handoff_state()
    hf["phase"] = "WARMING"
    hf["epoch"] = 2
    monkeypatch.setattr(handoff, "_handoff_synced_epoch_soft", lambda rid: 1)
    handoff._handoff_observe_understudy(st, hf)
    assert hf["ckpt_pulled_epoch"] is None


def test_jobd_status_soft_reads_first_token(monkeypatch):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (0, "IDLE 2026-07-15T12:00:00Z\n", ""))
    assert handoff._jobd_status_soft(9100) == "IDLE"
    monkeypatch.setattr(b2, "_rclone_soft",
                        lambda args: (0, "RUNNING 2 123 456\n", ""))
    assert handoff._jobd_status_soft(9100) == "RUNNING"
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (1, "", "no file"))
    assert handoff._jobd_status_soft(9100) is None
    monkeypatch.delenv("B2_BUCKET")
    assert handoff._jobd_status_soft(9100) is None


def test_init_state_backfills_search_filters_from_spec(monkeypatch):
    # D2: `train --gpu-ram 24 --cuda 12.8 --supervise` spawns a child supervise
    # with NO search filters on its argv; the captured spec must backfill them so
    # _relaunch and _handoff_pick_offer never search the whole market (the live
    # canary's understudy landed on an 8GB GTX 1080).
    spec = {"image": "img:1", "disk": 60, "runtype": "ssh_direct",
            "gpu": [], "gpu_ram": 24.0, "num_gpus": 1, "cuda": 12.8, "env": {}}
    monkeypatch.setattr(run_lane, "_capture_launch_spec",
                        lambda rid, a: (spec, 0.24))
    a = argparse.Namespace(run_id="r1", max_relaunch=3, budget=5.0,
                           wall_budget=7200.0, max_bid=None, defend_at=None,
                           strict_ceiling=False, gpu=None, gpu_ram=0,
                           num_gpus=1, cuda=0)
    st = run_lane._init_state(a)
    assert a.gpu_ram == 24.0 and a.cuda == 12.8
    assert st["launch_spec"]["gpu_ram"] == 24.0


def test_init_state_explicit_filters_beat_spec(monkeypatch):
    spec = {"image": "img:1", "gpu": ["RTX 3090"], "gpu_ram": 24.0,
            "num_gpus": 1, "cuda": 12.8, "env": {}}
    monkeypatch.setattr(run_lane, "_capture_launch_spec",
                        lambda rid, a: (spec, 0.24))
    a = argparse.Namespace(run_id="r1", max_relaunch=3, budget=5.0,
                           wall_budget=7200.0, max_bid=None, defend_at=None,
                           strict_ceiling=False, gpu=["RTX 5090"], gpu_ram=32,
                           num_gpus=2, cuda=12.9)
    run_lane._init_state(a)
    assert a.gpu == ["RTX 5090"] and a.gpu_ram == 32
    assert a.num_gpus == 2 and a.cuda == 12.9


def test_build_launch_spec_carries_cuda():
    spec = launch_spec._build_launch_spec(
        run_id="r1", runset="s", image="img:1", image_login_ref=None, disk=40,
        runtype="ssh_direct", gpu=[], gpu_ram=24.0, num_gpus=1, cuda=12.8,
        env_list=[], onstart="#!/bin/sh\n", orig_bid=None, max_bid=None)
    assert spec["cuda"] == 12.8 and spec["gpu_ram"] == 24.0


# =============================================================================
# 9. defects D5/D6 (live canary handoff-canary-3, 2026-07-15): per-chunk market
#    floor + underbid-park misclassification
# =============================================================================
def _chunk_offers_api(monkeypatch, offers):
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, body=None, retries=0:
                        (True, {"offers": offers}, None))


def test_market_min_bid_soft_matches_instance_chunk(monkeypatch):
    # the live D5 shape: machine 42830 listed a 1-GPU chunk at 0.1333 and the
    # 2-GPU chunk at 0.2667; our 2-GPU instance's floor is the 2-GPU chunk's.
    _chunk_offers_api(monkeypatch, [
        {"num_gpus": 1, "min_bid": 0.1333},
        {"num_gpus": 2, "min_bid": 0.2667},
        {"num_gpus": 4, "min_bid": 0.5333},
    ])
    assert pricing._market_min_bid_soft(42830, 2) == 0.2667


def test_market_min_bid_soft_scales_per_gpu_without_exact_chunk(monkeypatch):
    # no 3-GPU chunk listed -> best per-GPU floor (0.1333) x 3
    _chunk_offers_api(monkeypatch, [
        {"num_gpus": 1, "min_bid": 0.1333},
        {"num_gpus": 4, "min_bid": 0.5332},
    ])
    assert pricing._market_min_bid_soft(42830, 3) == round(0.1333 * 3, 4)


def test_market_min_bid_soft_no_num_gpus_keeps_min_across_chunks(monkeypatch):
    _chunk_offers_api(monkeypatch, [
        {"num_gpus": 1, "min_bid": 0.1333},
        {"num_gpus": 2, "min_bid": 0.2667},
    ])
    assert pricing._market_min_bid_soft(42830) == 0.1333
    assert pricing._market_min_bid_soft(42830, None) == 0.1333


def test_market_min_bid_soft_fail_soft(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (False, None, "HTTP 500"))
    assert pricing._market_min_bid_soft(42830, 2) is None
    assert pricing._market_min_bid_soft(None, 2) is None


def test_underbid_parked_pure():
    assert bidpolicy._underbid_parked({"last_bid": 0.16, "market_min_bid": 0.2667})
    # AT the floor is NOT underbid (a real operator park must not be masked)
    assert not bidpolicy._underbid_parked({"last_bid": 0.2667,
                                         "market_min_bid": 0.2667})
    assert not bidpolicy._underbid_parked({"last_bid": None,
                                         "market_min_bid": 0.2667})
    assert not bidpolicy._underbid_parked({"last_bid": 0.16,
                                         "market_min_bid": None})


def test_poll_underbid_park_not_operator_stop():
    # the exact live D6 state: vast parked the box (intended=stopped, actual
    # exited) 47s after our own decay PUT dropped the bid under the floor.
    # poll must fall through to the not-live rows (debounce -> rescue), never
    # exit operator_stop.
    s = bidpolicy.mk_poll_state(
        view={"status": "running"}, present=True, actual_status="exited",
        intended_status="stopped", last_bid=0.16, market_min_bid=0.2667,
        budget_usd=5.0, now=100.0)
    act = bidpolicy.poll(s)
    assert act.kind != "stop_terminal"
    assert act == bidpolicy.Action("noop", "debounce_not_live")


def test_poll_park_with_bid_at_floor_is_still_operator_stop():
    s = bidpolicy.mk_poll_state(
        view={"status": "running"}, present=True, actual_status="exited",
        intended_status="stopped", last_bid=0.2667, market_min_bid=0.2667,
        budget_usd=5.0, now=100.0)
    assert bidpolicy.poll(s) == bidpolicy.Action("stop_terminal", "operator_stop")


def test_poll_underbid_park_reaches_rescue_after_debounce():
    # past the debounce, the same underbid-park state must reach a money move
    # (rescue re-raise toward the correct floor), not an exit.
    s = bidpolicy.mk_poll_state(
        view={"status": "running"}, present=True, actual_status="exited",
        intended_status="stopped", last_bid=0.16, market_min_bid=0.2667,
        max_bid=0.334, budget_usd=5.0, now=1000.0,
        not_live_streak=bidpolicy.NOT_LIVE_DEBOUNCE)
    act = bidpolicy.poll(s)
    assert act.kind == "rescue_bid"
    # 1.2x the CORRECT (chunk-matched) floor, not the per-GPU one
    assert act.reason == "rescue:0.32"


def test_poll_underbid_park_with_cli_stopping_intent_is_operator_stop():
    # P1 (2026-07-18 review): a REAL `herdd stop` while the bid happens to sit
    # under the floor must NOT be swallowed by the D6 underbid carve-out —
    # `stopping` is non-terminal in the fold, so nothing else ends the run for a
    # present+stopped box, and rescue would re-raise the bid and resume the box
    # against the operator. A live cli stopping intent wins over the underbid read.
    s = bidpolicy.mk_poll_state(
        view={"status": "running"}, present=True, actual_status="exited",
        intended_status="stopped", last_bid=0.16, market_min_bid=0.2667,
        max_bid=0.334, budget_usd=5.0, now=1000.0,
        stopping_actor="cli:laptop",
        not_live_streak=bidpolicy.NOT_LIVE_DEBOUNCE)
    assert bidpolicy.poll(s) == bidpolicy.Action("stop_terminal", "operator_stop")
    # a non-cli actor (or none) keeps the D6 rescue behavior
    s["stopping_actor"] = "supervise:laptop"
    assert bidpolicy.poll(s).kind == "rescue_bid"


# --- D7: razor-thin floor/on-demand gap (found pre-spend 2026-07-15) --------
def test_bid_target_razor_thin_floor_is_now_refused_not_raised_to_floor():
    """**Reversed 2026-08-09** (recalibration item A), and worth reading as the
    D7-raise-to-floor branch's obituary.

    Live shape (offer 44567939): floor 0.2133, on-demand 0.2141 — a gap smaller
    than the on-demand clamp step, so the clamped target ($0.213) sat BELOW the
    floor, a known-losing bid vast answers with an underbid park. The fix then was
    to raise back to the floor ($0.214, still strictly under on-demand).

    A floor at 99.6% of on-demand now fails the hard ceiling
    (0.75 x 0.2141 = $0.161) and is refused outright. That is the better answer:
    the box would have cost within a tenth of a cent of its guaranteed on-demand
    price while carrying every eviction risk of spot.

    The raise-to-floor branch is KEPT — it is cheap and it is the correct handling
    if a caller-supplied cap ever lands a target under the floor — but it is now
    unreachable through the on-demand rails, because the survival cushion puts the
    target at 1.10 x floor before any downward clamp and every clamp that could
    push it under the floor now escalates or refuses first. Measured: 8
    `raised_to_floor` outcomes in a 400k-point random sweep of the
    (floor, max_bid, on_demand, cap) space, all of them sub-cent floors where the
    $0.001 grid rounding is what raised it."""
    dec = bidpolicy.bid_decision(0.2133, None, 0.2141)
    assert dec.price is None and dec.escalate is True
    assert dec.ceiling == pytest.approx(0.161, abs=1e-3)
    # the branch itself still works where the grid, not the market, is the cause
    assert bidpolicy._bid_target(0.0011, None, None) == 0.002


def test_bid_target_unwinnable_floor_returns_none():
    # floor at/over on-demand: no bid can ever win -> None (bid moves disable)
    assert bidpolicy._bid_target(0.30, None, on_demand=0.30) is None
    # floor over max_bid: can't afford to hold -> None (eviction ladder owns it)
    assert bidpolicy._bid_target(0.50, 0.40, on_demand=1.00) is None


def test_bid_target_wide_gap_unchanged():
    # the D7 branch must not touch the normal shapes
    # (1.2 x floor since the 2026-08-09 ruling; 0.56 during the 2.00x era)
    assert bidpolicy._bid_target(0.28, 5.0, on_demand=1.00) == 0.336  # 1.2 x floor
    assert bidpolicy._bid_target(0.50, 1.00) == 0.60         # unpriced path: 1.2 x floor


def test_auto_bid_price_razor_thin_floor():
    # 2026-08-09: a floor at 99.6% of on-demand is refused, not raised to the
    # floor — see test_bid_target_razor_thin_floor_is_now_refused_not_raised_to_floor
    assert pricing._auto_bid_price(0.2133, 0.2141) is None
    # floor == on-demand: unwinnable -> None (caller requires --price / skips)
    assert pricing._auto_bid_price(0.30, 0.30) is None
    # wide gap unchanged (1.2 x floor since 2026-08-09; 0.40 in the 2.00x era)
    assert pricing._auto_bid_price(0.20, 1.00) == 0.24     # 1.2 x floor, under the 0.65 cap


def test_auto_bid_price_matches_bid_target_grid():
    # P2 (2026-07-18 review): the launch price and the steady-state defend/decay
    # target must land on the SAME $0.001 grid point, else the first supervised
    # poll decays a fresh launch bid one step (the old +1e-4 nudge diverged on
    # ~9% of floors, e.g. 0.0029). Sweep a fine floor grid, with and without an
    # on-demand clamp in play.
    for i in range(1, 2000):
        mb = i / 10000.0                                  # 0.0001 .. 0.1999
        for od in (None, mb * 4, mb * 1.15):
            assert pricing._auto_bid_price(mb, od) == \
                bidpolicy._bid_target(mb, None, on_demand=od), (mb, od)


# =============================================================================
# 10. handoff DEFAULT flip (2026-07-15): --handoff is the default over the
#     preferred ceiling; --no-handoff / --strict-ceiling opt out.
# =============================================================================
# `target` stays spelled as the flat entry-point name the call sites already
# pass; step 6e resolves it here to the vastlib home the parser now binds
# (`cmd_supervise` became `vastlib.cli.supervise.run`).
_CAPTURE_TARGETS = {
    "cmd_supervise": (cli_supervise, "run"),
    "cmd_job_supervise": (cli_job_supervise, "cmd_job_supervise"),
    "cmd_train": (cli_train, "run"),
}


def _capture_ns(monkeypatch, argv, cmdattr, target):
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    got = {}
    _mod, _attr = _CAPTURE_TARGETS[target]
    monkeypatch.setattr(_mod, _attr, lambda a: got.update(a=a))
    monkeypatch.setattr(sys, "argv", ["herdd"] + argv)
    cli_main.main()
    return got["a"]


def test_supervise_parser_handoff_default_on(monkeypatch):
    a = _capture_ns(monkeypatch, ["supervise", "run-x", "--dry-run"],
                    "cmd", "cmd_supervise")
    assert a.handoff is True and a.strict_ceiling is False


def test_supervise_parser_no_handoff_opts_out(monkeypatch):
    a = _capture_ns(monkeypatch, ["supervise", "run-x", "--dry-run", "--no-handoff"],
                    "cmd", "cmd_supervise")
    assert a.handoff is False


def test_supervise_parser_strict_ceiling(monkeypatch):
    a = _capture_ns(monkeypatch, ["supervise", "run-x", "--dry-run", "--strict-ceiling"],
                    "cmd", "cmd_supervise")
    assert a.strict_ceiling is True


def test_supervise_parser_handoff_and_strict_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(config, "load_env", lambda: None)
    monkeypatch.setattr(config, "load_herdd_config", lambda: {})
    monkeypatch.setattr(sys, "argv",
                        ["herdd", "supervise", "run-x", "--handoff", "--strict-ceiling"])
    with pytest.raises(SystemExit):
        cli_main.main()


def test_job_supervise_parser_handoff_default_off_since_the_incident(monkeypatch):
    """SAFE-OFF on the jobs lane (2026-08-08, HANDOFF_DESIGN §11) — including the
    legacy inline driver, so the default does not depend on which driver you
    happen to reach the ladder through. `--handoff` still asks for it, and the
    run lane's default is untouched (see the supervise parser tests above)."""
    monkeypatch.delenv(config.JOBS_HANDOFF_UNSAFE_ENV, raising=False)
    a = _capture_ns(monkeypatch, ["job", "supervise", "45", "--dry-run"],
                    "jobcmd", "cmd_job_supervise")
    assert a.handoff is False and a.strict_ceiling is False
    a = _capture_ns(monkeypatch, ["job", "supervise", "45", "--dry-run", "--handoff"],
                    "jobcmd", "cmd_job_supervise")
    assert a.handoff is True


def test_job_supervise_parser_no_handoff_opts_out(monkeypatch):
    a = _capture_ns(monkeypatch, ["job", "supervise", "45", "--dry-run", "--no-handoff"],
                    "jobcmd", "cmd_job_supervise")
    assert a.handoff is False


def test_job_supervise_parser_wall_budget(monkeypatch):
    # S3 (2026-07-18 review): without the flag the amortization horizon was a
    # hardcoded 24h fiction — a job with 20 min left would still migrate. The
    # flag exists (seconds, like run supervise's) and defaults to None.
    a = _capture_ns(monkeypatch, ["job", "supervise", "45", "--dry-run",
                                  "--wall-budget", "1800"],
                    "jobcmd", "cmd_job_supervise")
    assert a.wall_budget == 1800.0
    a = _capture_ns(monkeypatch, ["job", "supervise", "45", "--dry-run"],
                    "jobcmd", "cmd_job_supervise")
    assert a.wall_budget is None


def test_train_parser_handoff_default_on(monkeypatch):
    a = _capture_ns(monkeypatch,
                    ["train", "--run", "r1", "--runset", "s", "--dry-run"],
                    "cmd", "cmd_train")
    assert a.handoff is True and a.strict_ceiling is False


def test_train_parser_no_handoff_opts_out(monkeypatch):
    a = _capture_ns(monkeypatch,
                    ["train", "--run", "r1", "--runset", "s", "--dry-run", "--no-handoff"],
                    "cmd", "cmd_train")
    assert a.handoff is False


# --------------------------------------------------------------------------- #
# boot-throughput watchdog (BOOT_HEALTHCHECK phase P0) — CLI supervise lane
#
# MIGRATED (was MIGRATION-BLOCKED, step 6e): the body landed in
# `supervise.replacement` — the home the stub named — and `run_lane` keeps one
# CALL-TIME forwarder, which is the subject here (the lane's own entry point).
# Seams follow the BODY: `health._get_instance_soft` (the `get_instance or ...`
# default), `journal._sup_emit`, `lifecycle._destroy_soft`, and
# `replacement._confirm_gone` / `replacement._relaunch`, which the body reads as
# its own module globals.
# --------------------------------------------------------------------------- #
def _boot_args(**kw):
    """A minimal supervise-arg namespace for _supervise_boot_health."""
    d = dict(boot_health=True, dry_run=False, machine=None, exclude_machines=None)
    d.update(kw)
    return argparse.Namespace(**d)


def _boot_state(**kw):
    st = bidpolicy.mk_poll_state(present=True, actual_status="loading",
                               relaunch_count=0, max_relaunch=3)
    st.update({"run_id": "run-boot", "instance_id": "inst-slow",
               "machine_id": 140087, "boot_sampler": None,
               "boot_sampler_iid": None, "excluded_machines": []})
    st.update(kw)
    return st


def _slow_inst_seq(n=18):
    return [{"actual_status": "loading",
             "status_msg": f"a1b2c3d4e5f6: Downloading [=>] {t * 0.001:.4f}MB/2000MB"}
            for t in range(0, n * 20, 20)]


def test_supervise_boot_health_opt_out_is_noop(monkeypatch):
    """Without --boot-health the watchdog is entirely inert (never polls)."""
    a = _boot_args(boot_health=False)
    st = _boot_state()
    called = {"n": 0}
    monkeypatch.setattr(boxes_health, "_get_instance_soft",
                        lambda iid: called.__setitem__("n", called["n"] + 1))
    assert run_lane._supervise_boot_health(st, a) is None
    assert called["n"] == 0


def test_supervise_boot_health_retires_sampler_when_running(monkeypatch):
    """Once the box is no longer pre-running, the sampler is dropped so a later
    box starts fresh (and no condemn can fire)."""
    a = _boot_args()
    st = _boot_state(actual_status="running", boot_sampler="stale",
                     boot_sampler_iid="inst-slow")
    assert run_lane._supervise_boot_health(st, a) is None
    assert st["boot_sampler"] is None and st["boot_sampler_iid"] is None


def test_supervise_boot_health_condemns_and_relaunches(monkeypatch):
    """A sustained-slow pull: emit boot_killed_slow, record the machine in the
    exclusion set, destroy + confirm gone, and relaunch (with the exclusion set
    passed through `a.exclude_machines`). Verdict 'condemned'."""
    a = _boot_args()
    st = _boot_state()
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    seq = _slow_inst_seq()
    idx = {"n": 0}

    def gi(iid):
        r = seq[min(idx["n"], len(seq) - 1)]
        idx["n"] += 1
        clock["t"] += 20
        return r

    emitted = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda run_id, ev, **f: emitted.append((ev, f)))
    destroyed = []
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, dry_run=False: (destroyed.append(iid), (True, None))[1])
    monkeypatch.setattr(replacement, "_confirm_gone", lambda iid: True)
    relaunched = {"seen_exclude": "UNSET"}

    def fake_relaunch(st_, a_):
        relaunched["seen_exclude"] = list(getattr(a_, "exclude_machines", None) or [])
        return "relaunched"

    monkeypatch.setattr(replacement, "_relaunch", fake_relaunch)

    res = None
    for _ in seq:
        res = run_lane._supervise_boot_health(st, a, get_instance=gi)
        if res is not None:
            break
    assert res == "condemned"
    assert any(ev == "boot_killed_slow" for ev, _ in emitted)
    ev, fields = next((ev, f) for ev, f in emitted if ev == "boot_killed_slow")
    assert fields["machine_id"] == 140087 and fields["window_s"] == 300
    assert fields["phase"] == "downloading" and fields["mbps"] < 5
    assert destroyed == ["inst-slow"]
    assert 140087 in st["excluded_machines"]
    assert relaunched["seen_exclude"] == [140087]     # exclusion reached the relaunch


def test_supervise_boot_health_max_relaunch_guard(monkeypatch):
    """When the relaunch budget is already spent, a slow condemn stops the loop
    (stop_fatal) instead of launching yet another box on yet another slow host."""
    a = _boot_args()
    st = _boot_state(relaunch_count=3, max_relaunch=3)
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    seq = _slow_inst_seq()
    idx = {"n": 0}

    def gi(iid):
        r = seq[min(idx["n"], len(seq) - 1)]
        idx["n"] += 1
        clock["t"] += 20
        return r

    monkeypatch.setattr(journal, "_sup_emit", lambda *a_, **k: None)
    monkeypatch.setattr(replacement, "_relaunch",
                        lambda *a_, **k: pytest.fail("must not relaunch past budget"))
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda *a_, **k: pytest.fail("must not destroy past budget"))
    res = None
    for _ in seq:
        res = run_lane._supervise_boot_health(st, a, get_instance=gi)
        if res is not None:
            break
    assert res == "stop_fatal"
    assert "max_relaunch" in (st.get("last_error") or "")


def test_supervise_boot_health_failed_poll_no_sample(monkeypatch):
    """A failed instance poll (None) contributes no sample and never condemns."""
    a = _boot_args()
    st = _boot_state()
    assert run_lane._supervise_boot_health(st, a, get_instance=lambda iid: None) is None


def test_build_search_query_excludes_machines():
    a = argparse.Namespace(
        type="bid", limit=5, num_gpus=1, unverified=False, gpu=None, gpu_ram=None,
        max_dph=None, host_disk=None, reliability=None, cuda=None, inet_down=None,
        machine=None, host=None, geo=None, exclude_machines=[111, 222])
    q = offers.build_search_query(a)
    assert q["machine_id"] == {"notin": [111, 222]}
    # an explicit --machine pin WINS over exclusion (never search nothing)
    a.machine = [999]
    q2 = offers.build_search_query(a)
    assert q2["machine_id"] == {"in": [999]}


def test_supervise_argv_forwards_boot_health():
    a = argparse.Namespace(strict_ceiling=False, handoff=True, defend_at=None,
                           rescue_wait=None, wall_budget=None, boot_health=True)
    argv = fleet_client._supervise_argv(a, "run-x", 5.0, None, None, None)
    assert "--boot-health" in argv
    a.boot_health = False
    assert "--boot-health" not in fleet_client._supervise_argv(a, "run-x", 5.0, None, None, None)


def test_supervise_parser_boot_health_flag(monkeypatch):
    a = _capture_ns(monkeypatch, ["supervise", "run-x", "--dry-run", "--boot-health"],
                    "cmd", "cmd_supervise")
    assert a.boot_health is True
    a2 = _capture_ns(monkeypatch, ["supervise", "run-x", "--dry-run"],
                     "cmd", "cmd_supervise")
    assert a2.boot_health is False


def test_train_parser_boot_health_flag(monkeypatch):
    a = _capture_ns(monkeypatch,
                    ["train", "--run", "r1", "--runset", "s", "--dry-run", "--boot-health"],
                    "cmd", "cmd_train")
    assert a.boot_health is True


# --- understudy image pin + disk sizing (velvet P3/P4) ---------------------- #
# A handoff MIGRATES a job that is already running, so the understudy must
# reproduce the primary's environment rather than adopt the newest one. The old
# code copied `image_uuid` (a mutable TAG) and `disk_space` (the primary's
# ALLOCATED size), so an env push mid-run moved the job onto a different env,
# and a 160G primary holding 17G minted another 160G box.

def _understudy_ns(monkeypatch, primary, *, tag_digest="sha256:new"):
    """Capture the Namespace the understudy launch would use."""
    seen = {}
    monkeypatch.setattr(replacement, "_job_understudy_offer",
                        lambda jctx, hf=None: {"id": 1, "min_bid": 0.10,
                                               "dph_total": 1.0})
    monkeypatch.setattr(models, "_job_primary_shape",
                        lambda jctx, hf=None: primary)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda _i: tag_digest)
    monkeypatch.setattr(launch_mod, "_do_launch",
                        lambda ns: (seen.update(vars(ns)), ("X", 1, 0.5))[1])
    jctx = {"iid": "9000", "last_bid": 1.0, "dph": 1.0, "on_demand": 1.10,
            "remaining_wall_h": 24.0}
    replacement._launch_job_understudy(jctx, {}, epoch=1)
    return seen


IMG = "registry.example.com/train:t215-latest"


def test_understudy_rolls_FORWARD_when_the_tag_moved(monkeypatch, capsys):
    """INVERTED 2026-08-04 by the rolling ruling ("always pinned to the latest
    image. no hash pins or anything"). This used to assert the opposite — the
    understudy pinned the primary's launch digest so a handoff reproduced the
    migrating job's env (velvet P3/P4). That digest replay is exactly what the
    ruling retires, and it happened invisibly on the eviction path.

    What must NOT be lost with it is the notice. Rolling silently would be the
    real regression: the reason line is now the only signal anyone gets that a
    job changed envs mid-flight, so it is asserted here alongside the ref."""
    ns = _understudy_ns(monkeypatch, {
        "image_uuid": IMG, "disk_space": 160, "disk_usage": 17,
        "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:orig"]]},
        tag_digest="sha256:moved")
    assert ns["image"] == IMG, "the understudy must take the tag, not a digest"
    assert "@sha256:" not in ns["image"]
    err = capsys.readouterr().err
    assert "ROLLED FORWARD" in err and "second half" in err


def test_understudy_keeps_the_tag_when_it_has_not_moved(monkeypatch):
    ns = _understudy_ns(monkeypatch, {
        "image_uuid": IMG, "disk_space": 160, "disk_usage": 17,
        "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:same"]]},
        tag_digest="sha256:same")
    assert ns["image"] == IMG


def test_understudy_replays_UNPINNED_when_the_tag_will_not_resolve(monkeypatch):
    """A digest GC'd from the registry would not pull, so an unresolvable check
    must not make the migration worse than it is today."""
    ns = _understudy_ns(monkeypatch, {
        "image_uuid": IMG, "disk_space": 160, "disk_usage": 17,
        "extra_env": [[imageref.IMAGE_DIGEST_ENV, "sha256:orig"]]},
        tag_digest=None)
    assert ns["image"] == IMG


def test_an_unstamped_primary_is_todays_behaviour_untouched(monkeypatch):
    ns = _understudy_ns(monkeypatch, {"image_uuid": IMG, "disk_space": 160,
                                      "disk_usage": 17})
    assert ns["image"] == IMG


def test_understudy_disk_comes_from_USAGE_not_the_primarys_allocation(monkeypatch):
    """The measured incident shape: 160G allocated, 17G used."""
    ns = _understudy_ns(monkeypatch, {"image_uuid": IMG, "disk_space": 160,
                                      "disk_usage": 17})
    assert ns["disk"] == 40, "1.4x17 + 12G overhead, rounded up"


def test_understudy_disk_keeps_the_allocation_when_usage_is_unreadable(monkeypatch):
    """A booting box reports disk_usage -1. A handoff is time-critical and a
    too-small replacement loses the run, so every uncertain path keeps today's
    behaviour."""
    ns = _understudy_ns(monkeypatch, {"image_uuid": IMG, "disk_space": 160,
                                      "disk_usage": -1})
    assert ns["disk"] == 160


def test_understudy_disk_never_GROWS_past_the_primarys_allocation(monkeypatch):
    ns = _understudy_ns(monkeypatch, {"image_uuid": IMG, "disk_space": 40,
                                      "disk_usage": 35})
    assert ns["disk"] == 40


# --- serve_mode: the jobs ladder minus queue semantics (fleetd `serve`) -------
# 2026-08-02: serve boxes launch spot by default (owner ruling — on-demand was
# a tooling gap, not a doctrine), so the ladder must defend/rescue an endpoint
# that has NO jobd queue. serve_mode strips exactly the queue exits and the
# jobd reattach; everything else (budget, stop-classify, defend, rescue,
# unrecoverable) is the SAME tested code path.
#
# MIGRATED (was MIGRATION-BLOCKED, step 6e): both named blockers landed —
# `_sticky_on_demand` at `vastlib.market.pricing` and `_serve_self_park_soft` at
# `vastlib.supervise.replacement` (with the rest of the serve cluster; the tick
# reaches it as `replacement._serve_self_park_soft`). Fixture and drivers move
# together. Placement is by RESOLUTION: `lifecycle.<name>` for the instance read,
# the park and the bid PUT, `pricing.<name>` for the market reads,
# `models._instance_serve_label`, `bidpolicy._preferred_ceiling_alarm`, and bare
# in `job_lane` for `_box_lifecycle_soft` / `_job_sup_reattach`.

def _serve_tick_env(monkeypatch, inst, *, self_parked=False, market=None):
    """Stub job_supervise_tick's I/O for a SERVE-mode context. Returns
    (bid_puts, reattaches) recorders."""
    bid_puts, reattaches = [], []
    m = monkeypatch
    m.setattr(lifecycle, "_instances_soft", lambda: ([dict(inst)] if inst else []))
    m.setattr(replacement, "_serve_self_park_soft",
              lambda sid, **k: bool(self_parked))
    m.setattr(models, "_instance_serve_label", lambda i: "sv-test")
    m.setattr(job_lane, "_box_lifecycle_soft",
              lambda iid: (_ for _ in ()).throw(AssertionError(
                  "serve_mode must not consult the jobd box-event stream")))
    m.setattr(jobmeta, "list_queue",
              lambda box, **k: (_ for _ in ()).throw(AssertionError(
                  "serve_mode must not read the job queue")))
    m.setattr(pricing, "_market_min_bid_soft", lambda mid, g=None: market)
    m.setattr(pricing, "_market_min_bid_read",
              lambda mid, g=None: models.MarketRead(True, True, market))
    m.setattr(pricing, "_market_ondemand_soft", lambda mid, n=None: None)
    m.setattr(bidpolicy, "_preferred_ceiling_alarm", lambda s: (False, None))
    m.setattr(lifecycle, "_stop_instance_soft", lambda iid: True)
    m.setattr(lifecycle, "_put_bid_soft",
              lambda iid, p: (bid_puts.append((iid, p)), (True, None))[1])
    m.setattr(job_lane, "_job_sup_reattach",
              lambda jc, iid: reattaches.append(iid))
    return bid_puts, reattaches


def _serve_ns(**over):
    a = argparse.Namespace(id=9000, dry_run=False, budget=5.0, max_bid=None,
                           handoff=True, strict_ceiling=False, rescue_wait=None,
                           keep=False, serve_mode=True)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_serve_mode_empty_queue_reaches_the_ladder_not_queue_empty(monkeypatch):
    """The exact gap that kept serve boxes on-demand: with no jobd queue the
    jobs tick exits `queue_empty` BEFORE the defend/rescue ladder. serve_mode
    must keep ticking (return None) — the box itself is the workload."""
    _serve_tick_env(monkeypatch, _JS_LIVE)
    jc, hf = job_lane.job_supervise_init(_serve_ns())
    assert jc["serve_mode"] is True
    assert jc["handoff_on"] is False        # jobd-shaped handoff has no serve analog
    assert job_lane.job_supervise_tick(jc, hf) is None


def test_serve_mode_outbid_box_gets_a_rescue_bid(monkeypatch):
    """A stopped BID serve box with no self-park marker is an eviction: the
    ladder must raise the bid (rescue), never abandon it or misread it as an
    operator park."""
    bid_puts, _re = _serve_tick_env(monkeypatch, _JS_STOPPED, market=1.2)
    jc, hf = job_lane.job_supervise_init(_serve_ns())
    verdict = None
    for _ in range(bidpolicy.NOT_LIVE_DEBOUNCE + 1):
        verdict = job_lane.job_supervise_tick(jc, hf)
        assert verdict is None              # rescuing, not exiting
    assert bid_puts, "no rescue bid was PUT for an outbid serve box"
    assert jc["rescue_deadline"] is not None


def test_serve_mode_watchdog_self_park_is_success_not_eviction(monkeypatch):
    """A fresh SELF_PARKED marker (MAX_HOURS watchdog) explains the stop: the
    tick exits `self_parked` and must NOT touch the bid. Without the marker
    read, the ladder would rescue-resume the box against its own watchdog
    forever."""
    bid_puts, _re = _serve_tick_env(monkeypatch, _JS_STOPPED, self_parked=True)
    jc, hf = job_lane.job_supervise_init(_serve_ns())
    assert job_lane.job_supervise_tick(jc, hf) == "self_parked"
    assert bid_puts == []


def test_serve_mode_resume_does_not_reattach_jobd(monkeypatch):
    """On a not-live -> live transition the jobs lane re-attaches jobd; a serve
    box revives via onstart instead — reattach must be skipped."""
    _bp, reattaches = _serve_tick_env(monkeypatch, _JS_STOPPED)
    jc, hf = job_lane.job_supervise_init(_serve_ns())
    assert job_lane.job_supervise_tick(jc, hf) is None
    live_env = dict(_JS_LIVE)
    monkeypatch.setattr(lifecycle, "_instances_soft", lambda: [live_env])
    assert job_lane.job_supervise_tick(jc, hf) is None
    assert reattaches == []


def test_serve_mode_budget_still_parks(monkeypatch):
    _serve_tick_env(monkeypatch, _JS_LIVE)
    jc, hf = job_lane.job_supervise_init(_serve_ns(budget=0.0))
    assert job_lane.job_supervise_tick(jc, hf) == "budget"


# --- _serve_self_park_soft: marker parse + freshness --------------------------
# DELETED at step 6d, deliberately. This group was the FLAT arm of a pair: it
# existed only so a drift between `herdd.py`'s own body and the ported one in
# `vastlib.supervise.replacement` could not pass unnoticed while both existed.
# The thin launcher ended that — `herdd._serve_self_park_soft` IS
# `replacement._serve_self_park_soft` now, one object, so the flat copy asserted
# a duplication rather than a contract and its `_mark_env` patch of
# `herdd._rclone_soft` had stopped steering anything. The surviving arm is
# `test_vastlib_supervise_replacement.py` ("_serve_self_park_soft — marker parse
# + freshness"): same six rows, `_rclone_soft` stubbed at `storage.b2`, plus a
# signature pin. `SERVE_SELF_PARK_FRESH_S` is asserted there too.


def test_handoff_understudy_confessing_pyhalf_never_syncs(monkeypatch):
    """Same defect class as the boot SLA's milestone: the bash half writes
    `IDLE` on a box whose python half is dead, so the SYNCED gate would declare
    the migration ready, retarget the queue onto a box that can neither claim a
    ticket nor emit an event, and destroy the healthy primary. Worse than the
    incident it descends from — it MOVES LIVE WORK onto a dead box."""
    monkeypatch.setattr(handoff, "_jobd_status_soft", lambda iid: "IDLE")
    monkeypatch.setattr(handoff, "_box_lifecycle_soft", lambda iid: {})
    hf = {"understudy_status": "running", "understudy_iid": "9100"}

    monkeypatch.setattr(boxes_health, "_jobd_status_pyhalf_soft", lambda iid: None)
    assert handoff._handoff_job_understudy_synced({}, hf) is True   # old bundle
    monkeypatch.setattr(boxes_health, "_jobd_status_pyhalf_soft", lambda iid: False)
    assert handoff._handoff_job_understudy_synced({}, hf) is True   # healthy
    monkeypatch.setattr(boxes_health, "_jobd_status_pyhalf_soft", lambda iid: True)
    assert handoff._handoff_job_understudy_synced({}, hf) is False  # confessing


# =============================================================================
# 9. The dwells END TO END, through the real lane, at two tick intervals
#    (2026-08-26: the deployed unit moved 45 s -> 15 s, which is exactly the
#    change a poll COUNT cannot survive).
# =============================================================================
def _decay_puts_at_tick(monkeypatch, tick_s, polls):
    """Drive the REAL `job_supervise_tick` `polls` times at `tick_s` against a
    live box whose floor has receded, and return `(poll_index, price)` for every
    bid PUT the lane issued. The PUT succeeds, so the decay's own streak reset is
    exercised too."""
    t0 = _now_epoch()
    views = [_pending_view(timeout_s=36000, started_at=_ts(t0 - 60))]
    _horizon_env(monkeypatch, views, floor=_STALE_HIGH_FLOOR, bid_put_ok=False)
    jc, hf = job_lane.job_supervise_init(_horizon_ns())
    clock = _TickClock(monkeypatch, tick_s)
    puts, at = [], {"poll": 0}
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, bid: (puts.append((at["poll"], bid)),
                                          (True, None))[1])
    for i in range(polls):
        at["poll"] = i + 1
        clock.tick(job_lane.job_supervise_tick, jc, hf)
    return jc, puts


def test_lane_decay_dwell_is_unchanged_at_the_45s_tick(monkeypatch):
    """The third consecutive candidate poll, 90 s in — the same poll the old
    BID_DECAY_POLLS count fired on."""
    jc, puts = _decay_puts_at_tick(monkeypatch, 45.0, 5)
    assert [p for p, _ in puts] == [3], puts


def test_lane_decay_dwell_survives_the_15s_tick(monkeypatch):
    """THE regression this whole change exists for. At 15 s a count of 3 fires
    after 30 s — a third of the ratified dwell, on a controller whose decays
    were already preceding evictions. As a duration it still takes 90 s, i.e.
    the SEVENTH poll, and nothing fires before it."""
    jc, puts = _decay_puts_at_tick(monkeypatch, 15.0, 9)
    assert [p for p, _ in puts] == [7], puts


def test_lane_records_and_clears_the_decay_clock(monkeypatch):
    """The wiring itself: the lane stores the run's start, and a landed decay PUT
    clears BOTH halves so the next decay pays a full dwell rather than firing on
    the next tick."""
    jc, puts = _decay_puts_at_tick(monkeypatch, 45.0, 3)
    assert puts and jc["decay_streak"] == 0 and jc["decay_streak_since"] is None
    # ...and a state file written before the key existed still decays, on the
    # count, exactly as the running daemon does today
    stale = _decaying(decay_streak=3)
    stale.pop("decay_streak_since", None)
    assert _bid_action(stale) == Action("lower_bid", "decay:0.24")


def test_lane_handoff_dwell_waits_on_time_not_on_count(monkeypatch):
    """The handoff dwell through the real jobs tick. This box's economics refuse
    the migration, so the observable past the dwell is the DEFERRAL (nothing
    downstream of the dwell runs before it): at a 15 s tick five polls satisfy
    the old COUNT and reach nothing, and only 180 s of wall clock does."""
    def _run(tick_s, polls):
        t0 = _now_epoch()
        views = [_pending_view(timeout_s=36000, started_at=_ts(t0 - 60),
                               n_checkpoints=3)]
        _horizon_env(monkeypatch, views, floor=_STALE_HIGH_FLOOR,
                     bid_put_ok=False)
        jc, hf = job_lane.job_supervise_init(_horizon_ns())
        clock = _TickClock(monkeypatch, tick_s)
        for _ in range(polls):
            clock.tick(job_lane.job_supervise_tick, jc, hf)
        gated = [ev for ev, _f in jc.get("handoff_journal") or []
                 if ev in ("deferred", "refused")]
        return hf, gated
    hf5, gated5 = _run(15.0, 5)
    assert hf5["over_ceiling_streak"] == 5 and gated5 == []   # count satisfied,
    assert hf5["over_ceiling_since"] is not None              # dwell is not
    _hf13, gated13 = _run(15.0, 13)
    assert gated13, "180 s in, the gate past the dwell finally runs"
