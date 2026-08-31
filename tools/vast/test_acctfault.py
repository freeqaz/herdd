"""Account-level failures are never charged to a host, and never mis-advised.

The incident these pin (2026-08-25): the vast balance hit $0.000 / −$1.504,
vast stopped the boxes, and the replacement ladder emitted 76
`insufficient_credit` refusals over 57 minutes. Two things had to be true and
only one of them was:

* NOTHING wrote a host strike during the outage — true then by luck, because
  every eviction that hour classified `outbid`/`host_stop` and those two are
  already strike-free. The `host_failure` class is one API behaviour away
  (`not present` -> `EVICTION_HOST_FAILURE`), and a credit outage that deletes
  instances instead of stopping them lands every box there at once.
* The alarms named a remedy that could work. They did not: `job retarget`,
  raise the budget/cap, `fleet destroy`, and — on the wedge alarm — raise the
  ceiling. All four are market/watch moves against a condition no price can
  answer.

So the classifier's negative half is as load-bearing as its positive half: a
failure a DIFFERENT HOST could have satisfied must keep charging a strike, or
the reputation ledger quietly stops learning.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vastlib.core import acctfault, api  # noqa: E402
from vastlib.fleet import daemon  # noqa: E402
from vastlib.market import hostrep  # noqa: E402
from vastlib.supervise import replacement  # noqa: E402

import bidpolicy  # noqa: E402

NOW = 1_800_000_000.0

# The refusal string fleetd journaled 76 times on 2026-08-25, verbatim.
CREDIT_ERR = ("replacement launch failed: error: HTTP 400 on PUT "
              "v0/asks/37158949/: {'error': 'insufficient_credit', 'msg': "
              "'Your account lacks credit; see the billing page.'}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    p = tmp_path / "host_reputation.json"
    monkeypatch.setenv(hostrep.PATH_ENV, str(p))
    monkeypatch.delenv(hostrep.DISABLE_ENV, raising=False)
    hostrep._cache.update({"path": None, "t": 0.0, "data": None})
    return str(p)


# ------------------------------------------------------------- the predicate

@pytest.mark.parametrize("err,code", [
    (CREDIT_ERR, "insufficient_credit"),
    ("HTTP 400: {'error': 'insufficient_credit'}", "insufficient_credit"),
    ("Your account lacks credit; see the billing page.", "insufficient_credit"),
    ("HTTP 401 on GET v0/instances/: unauthorized", "auth"),
    ("HTTP 401 on GET v0/instances/", "auth"),
    ("error: invalid api key", "auth"),
    ("HTTP 403 on PUT v0/asks/1/", "permission"),
    ("account suspended", "permission"),
    ("config: VASTAI_API_KEY not set (env or .env)", "no_api_key"),
])
def test_account_level_failures_classify(err, code):
    assert acctfault.classify(err) == code


@pytest.mark.parametrize("err", [
    None,
    "",
    # THE NEGATIVE HALF. Every one of these is a fact about a host or a market,
    # and the reputation ledger is entitled to all of them.
    "Required resources are currently unavailable, state change queued.",
    "pull not finished in 10m (> 600s timeout)",
    "host pulled at 3.1 MB/s aggregate (< 8 MB/s floor)",
    "no affordable replacement: on-demand $3.7356 over the $1.202 ceiling",
    "no_offer",
    "over_ceiling",
    "HTTP 429 on GET v0/bundles/: rate limited",   # transient, and OUR polling
    "HTTP 404 on GET v0/instances/47/",            # a deleted instance
    "HTTP 500 on GET v0/instances/",
    "network <urlopen error timed out> on GET v0/instances/",
])
def test_host_and_market_failures_do_not_classify(err):
    assert acctfault.classify(err) is None


def test_classify_reads_an_exception_as_readily_as_a_string():
    """`request`'s raising twin throws SystemExit carrying the same prose, and
    that is the shape the replacement ladder actually catches."""
    assert acctfault.classify(SystemExit(f"error: {CREDIT_ERR}")) \
        == "insufficient_credit"


# ------------------------------------------------------------------ the latch

def test_the_http_funnel_latches_an_account_refusal(monkeypatch):
    """`request_soft` is the one seam every lane shares, so no caller has to
    remember to report this — and the seams that must act on it (a strike, an
    alarm's remedy) are nowhere near the call that saw the error.

    Driven over a GET on purpose: conftest's `_block_mutating_api_calls`
    replaces the funnel outright for a PUT, so a mutating probe here would test
    the guard rather than the latch."""
    acctfault.clear()
    assert acctfault.recent() is None

    body = (b'{"error": "insufficient_credit", "msg": "Your account lacks '
            b'credit; see the billing page."}')

    def _boom(_req, timeout=None):
        raise urllib.error.HTTPError("u", 400, "Bad Request", None,  # type: ignore[arg-type]
                                     io.BytesIO(body))

    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    ok, _d, err = api.request_soft("GET", "v0/instances/", retries=0)
    assert not ok and "insufficient_credit" in err
    rec = acctfault.recent()
    assert rec and rec["code"] == "insufficient_credit"


def test_the_latch_expires_so_a_topped_up_account_is_trusted_again():
    acctfault.clear()
    acctfault.note(CREDIT_ERR, now=NOW)
    assert acctfault.recent(now=NOW + 60) is not None
    assert acctfault.recent(now=NOW + acctfault.WINDOW_S + 1) is None


# ------------------------------------------------- the strike ledger's guard

def test_an_account_fault_charges_NO_strike(store):
    """THE rule. A strike is a claim about a machine; an account that cannot
    rent is true of every machine at once, so it may not become one."""
    acctfault.note(CREDIT_ERR, now=NOW)
    assert hostrep.note_strike(7, "host_failure", now=NOW) is None
    assert hostrep.summary(NOW) == []
    assert not os.path.exists(store)


def test_every_strike_kind_is_covered_by_the_one_guard(store):
    """The guard sits at the write, not at the three callers, so a fourth
    caller inherits it. Kinds are the vocabulary those callers use."""
    acctfault.note(CREDIT_ERR, now=NOW)
    for kind in hostrep.STRIKE_WEIGHTS:
        assert hostrep.note_strike(11, kind, now=NOW) is None
    assert hostrep.summary(NOW) == []


def test_a_real_host_failure_still_earns_its_strike(store):
    """The negative control. Suppressing too much is the failure mode that
    looks like success — the ledger simply stops learning."""
    acctfault.clear()
    assert hostrep.note_strike(7, "pull_timeout", now=NOW) == pytest.approx(1.0)
    assert hostrep.penalty(7, now=NOW) == pytest.approx(1.35)


def test_the_ledger_resumes_once_the_latch_has_expired(store, monkeypatch):
    acctfault.note(CREDIT_ERR, now=NOW - acctfault.WINDOW_S - 1)
    assert hostrep.note_strike(7, "pull_timeout", now=NOW) == pytest.approx(1.0)


def test_note_ok_is_untouched_by_the_guard(store):
    """Positive evidence is not a claim that can defame anyone, and a box that
    booted during a credit outage really did boot."""
    acctfault.note(CREDIT_ERR, now=NOW)
    assert hostrep.note_ok(7, now=NOW) is True


# ------------------------------------------------- the ladder's refusal code

def test_a_credit_refusal_is_reported_as_account_blocked():
    assert replacement._launch_failure_reason(SystemExit(f"error: {CREDIT_ERR}")) \
        == "account_blocked"


def test_an_ordinary_launch_failure_is_still_unlaunchable():
    assert replacement._launch_failure_reason(
        SystemExit("error: HTTP 400 on PUT v0/asks/1/: {'error': 'ask gone'}")) \
        == "unlaunchable"


# ------------------------------------------------------------- alarm remedies

def _fleet():
    class _F(daemon.Fleet):
        def __init__(self):
            self._health = {}
            self.state = {"watches": {}, "ceilings": {}, "ceiling_by_box": {}}

        def _ceiling_spend(self, w):
            return float(w.get("spend_usd") or 0.0)

    return _F()


def _msg(alarms, key):
    return next(m for k, m in alarms if k.endswith(key))


def test_the_stalled_alarm_names_the_ONE_remedy_that_can_work():
    w = {"target": "47", "iid": "47", "profile": "jobs", "budget_usd": 10.0,
         "state": "watched", "unrecoverable_since": NOW - 31,
         "replacement_refused": f"account_blocked: {CREDIT_ERR}"}
    m = _msg(_fleet()._derive_watch_alarms("47", w, NOW), "rescue_stalled")
    assert "ACCOUNT cannot rent at any price" in m
    assert "insufficient credit" in m
    assert "console.vast.ai/billing" in m
    # ...and NONE of the three the operator was given on the day.
    assert "job retarget" not in m
    assert "raise the budget/cap" not in m
    assert "fleet destroy" not in m


def test_an_ordinary_stall_keeps_its_three_remedies():
    w = {"target": "47", "iid": "47", "profile": "jobs", "budget_usd": 10.0,
         "state": "watched", "unrecoverable_since": NOW - 31,
         "replacement_refused": ("no affordable replacement: on-demand "
                                 "$3.7356 over the $1.202 ceiling")}
    m = _msg(_fleet()._derive_watch_alarms("47", w, NOW), "rescue_stalled")
    assert "job retarget" in m and "raise the budget/cap" in m
    assert "ACCOUNT cannot rent" not in m


def test_the_wedge_alarm_stops_blaming_the_ceiling():
    """`no qualifying offer seen at any price` is TRUE and misleading here: the
    refusals never reached the market at all."""
    w = {"target": "47", "iid": "47", "profile": "jobs", "budget_usd": 10.0,
         "state": "watched",
         "replacement": {"replacement_refusals": 5,
                         "replacement_refusals_since": NOW - 319,
                         "replacement_refusal_reason": "account_blocked",
                         "replacement_refusal_ceiling": 0.773}}
    m = _msg(_fleet()._derive_watch_alarms("47", w, NOW), "replacement_wedged")
    assert "ACCOUNT cannot rent at any price" in m
    assert "--replace-ceiling-mult" not in m
    assert "not affordable in this market" not in m


def test_an_ordinary_wedge_keeps_its_ceiling_advice():
    w = {"target": "47", "iid": "47", "profile": "jobs", "budget_usd": 10.0,
         "state": "watched",
         "replacement": {"replacement_refusals": 5,
                         "replacement_refusals_since": NOW - 319,
                         "replacement_refusal_reason": "over_ceiling",
                         "replacement_refusal_ceiling": 0.773}}
    m = _msg(_fleet()._derive_watch_alarms("47", w, NOW), "replacement_wedged")
    assert "--replace-ceiling-mult" in m
    assert "ACCOUNT cannot rent" not in m


# ---------------------------------------------- the eviction class's own guard

def test_a_no_credit_eviction_charges_NO_strike(store):
    """`EVICTION_NO_CREDIT` (landed 2026-08-25) names OUR account as the cause,
    so a strike written under it contradicts its own label — and it arrives via
    a STORED signal, so it can outlive the 15-minute latch that would otherwise
    have covered it."""
    acctfault.clear()
    jc: dict = {}
    replacement._job_note_evicted_machine(jc, 91, bidpolicy.EVICTION_NO_CREDIT,
                                          NOW)
    assert hostrep.summary(NOW) == []


@pytest.mark.parametrize("cls", [bidpolicy.EVICTION_HOST_FAILURE,
                                 bidpolicy.EVICTION_ONDEMAND,
                                 bidpolicy.EVICTION_UNKNOWN])
def test_the_host_classes_still_charge(store, cls):
    """The negative control for the set above. These three are unchanged —
    arguable in their own right, but not touched by an account-fault fix."""
    acctfault.clear()
    replacement._job_note_evicted_machine({}, 92, cls, NOW)
    assert [r["machine_id"] for r in hostrep.summary(NOW)] == ["92"]


def test_the_wedge_alarm_needs_a_DWELL_not_just_a_refusal_count():
    """"Past a transient market blip" is a wall-clock claim, and the refusal
    counter is tick-relative: at a 15s tick five refusals is 75s. A false
    wedge alarm is not free — it sends an operator at a market that is fine."""
    def w(since_ago):
        return {"target": "47", "iid": "47", "profile": "jobs",
                "budget_usd": 10.0, "state": "watched",
                "replacement": {"replacement_refusals": 5,
                                "replacement_refusals_since": NOW - since_ago,
                                "replacement_refusal_reason": "over_ceiling",
                                "replacement_refusal_ceiling": 0.773}}
    f = _fleet()
    assert not [k for k, _ in f._derive_watch_alarms("47", w(75), NOW)
                if k.endswith("replacement_wedged")]
    assert _msg(f._derive_watch_alarms("47", w(200), NOW), "replacement_wedged")


def test_a_wedge_with_no_recorded_start_still_alarms_on_the_count():
    """State written before `replacement_refusals_since` existed must not go
    silent — the dwell is an added condition, never a new way to miss one."""
    w = {"target": "47", "iid": "47", "profile": "jobs", "budget_usd": 10.0,
         "state": "watched",
         "replacement": {"replacement_refusals": 5,
                         "replacement_refusal_reason": "over_ceiling",
                         "replacement_refusal_ceiling": 0.773}}
    assert _msg(_fleet()._derive_watch_alarms("47", w, NOW), "replacement_wedged")
