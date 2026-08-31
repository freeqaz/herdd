"""`vastlib.supervise.retention` — the retain-or-destroy sweep, ported.

Why this file exists
--------------------
Two flat suites drive this code today and both stay UNEDITED through plan step 5
(§8 add-only amendment): `test_salvage.py`'s "herdd wiring" half and
`test_eviction_replacement.py`'s retention + quiesce/resurrection blocks. They
steered the LIVE flat copies in `herdd.py` via `monkeypatch.setattr(v, ...)`,
so none of their ~55 assertions touched `vastlib` at all. This file is the
port-time coverage the ADD-ONLY rule required: the same assertions, re-aimed at
`vastlib.supervise.retention`, with **no expectation changed** (plan §7.4). It
outlives that rule: at step 6d `monkeypatch.setattr(herdd, …)` stopped
steering anything (a re-export is not a patch point), so this file is where the
retention assertions actually bite.

Three differences from the originals, all mechanical:

1. **Patch targets move to the new homes** — `lifecycle._destroy_and_revoke` /
   `_put_state_soft` / `_put_bid_soft` / `_put_label_soft`,
   `journal._job_handoff_emit`. Nothing is patched on `herdd`.
2. **The retain-side tests call `_job_retain_or_destroy` DIRECTLY** rather than
   reaching it through `_job_eviction_replace` (which is `replacement.py`, a
   sibling module in the same wave). Same inputs, same assertions — the
   originals' `_wire` stubs the whole replacement path precisely so that the
   retention record is what is actually under test.
3. **In-module seams are patched on `retention` itself**
   (`_job_salvage_advance`, `_job_salvage_sweep`). That is not a stylistic
   choice: `_job_retention_sweep` must keep calling them as module attributes or
   the flat suites' four patch sites go vacuously green when they migrate at
   steps 6-7. Two tests below fail loudly if a future cleanup binds them at
   import time.

What is deliberately NOT here
-----------------------------
* No policy arithmetic. `bidpolicy.retention_plan` / `retention_live_cost` /
  `RETENTION_PARK_BID` are Zone S and `test_eviction_replacement.py` owns them;
  re-asserting the cost model here would put a second copy of a number in the
  tree.
* No fleetd row builders. `retention_rows` / `retention_alarms` /
  `_retention_status_map` READ the records this module writes, and fleetd is not
  ported until step 5 — so what is pinned here is the RECORD SHAPE those
  builders key on, not the builders.
* No network, no B2, no subprocess. Every PUT seam is stubbed
  (conftest's `_block_mutating_api_calls` refuses the rest), and
  `journal._job_handoff_emit` — the only B2 writer on these paths — is stubbed
  in every test that can reach it. `_job_ladder_journal` is left REAL: it is
  pure in-memory and several assertions read the queue it appends to.

Provenance: created 2026-08-16 alongside `vastlib/supervise/retention.py`, plan
§8 step 4. Mirrors `test_salvage.py` (~lines 535-670) and
`test_eviction_replacement.py` (the sweep block and the quiesce/resurrection
block).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import bidpolicy as bp                                 # noqa: E402

from vastlib.boxes import lifecycle                    # noqa: E402
from vastlib.boxes import salvage as S                 # noqa: E402
from vastlib.supervise import journal                  # noqa: E402
from vastlib.supervise import retention as R           # noqa: E402

NOW = 3_000_000.0


@pytest.fixture(autouse=True)
def _no_job_env(monkeypatch):
    """The four knobs this module reads resolve namespace > `JOB_<NAME>` env >
    bidpolicy default. A developer with one exported would silently re-price
    every window below, so the env rung is cleared rather than trusted."""
    for k in ("JOB_REPLACEMENT_RETENTION_HOURS", "JOB_RETENTION_BACKSTOP_HOURS",
              "JOB_SALVAGE_KEEP_N", "JOB_SALVAGE_MAX_GB"):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- #
# fixtures — the shapes the flat suites use, verbatim
# --------------------------------------------------------------------------- #
def _inst(iid=41, status="exited", machine=7, dph=0.76):
    return {"id": iid, "actual_status": status, "machine_id": machine,
            "dph_total": dph, "num_gpus": 2, "gpu_name": "RTX PRO 6000",
            "label": "upstream-monorepo", "start_date": NOW - 600, "is_bid": True}


def _jc(**kw):
    jc = {"a": argparse.Namespace(), "iid": "NEW", "instances": [],
          "dry_run": False, "now": NOW}
    jc.update(kw)
    return jc


def _rec(deadline_ts, status="retained"):
    return {"iid": "41", "status": status, "class": bp.EVICTION_OUTBID,
            "retained_ts": NOW, "deadline_ts": deadline_ts, "retention_h": 3.0,
            "cost_usd": 0.27, "cost_hi_usd": 0.58, "keep_labeled": True}


def _live_rec(deadline_ts, **kw):
    r = _rec(deadline_ts)
    r["storage_day_usd"] = 0.9777777777777781
    r["quiesce"] = {"stopped": True, "bid_pinned": bp.RETENTION_PARK_BID,
                    "prior_bid": 1.2, "errors": [], "why": "retention"}
    r.update(kw)
    return r


def _emit_wire(monkeypatch):
    """Stub the only B2 writer these paths reach; return the call log."""
    calls = []
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: calls.append(("emit", event, kw)))
    return calls


def _quiesce_wire(monkeypatch):
    """Record the stop/bid-pin PUTs the quiesce path issues.

    conftest's mutating-API guard refuses them suite-wide; this is the per-test
    override that lets us assert on them."""
    seen = {"stop": [], "bid": []}
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda iid, state: (seen["stop"].append((str(iid), state)),
                                            (True, None))[1])
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: (seen["bid"].append((str(iid), price)),
                                            (True, None))[1])
    return seen


def _retain_wire(monkeypatch, *, label_fail=None):
    """Stub every seam the retain path touches — the label PUT, the destroy, the
    emitter and the two quiesce PUTs. Returns `(calls, seen)`: the flat suites'
    `(kind, ...)` call log, and `_quiesce_wire`'s stop/bid record."""
    calls = _emit_wire(monkeypatch)
    monkeypatch.setattr(
        lifecycle, "_put_label_soft",
        lambda iid, label: (calls.append(("label", str(iid), label)),
                            (False, label_fail) if label_fail else (True, None))[1])
    monkeypatch.setattr(
        lifecycle, "_destroy_and_revoke",
        lambda ids, ins, intent, noun="": (
            calls.append(("destroy", list(ids), intent)), [])[1])
    return calls, _quiesce_wire(monkeypatch)


def _retained(jc):
    return (jc.get("retained_boxes") or [{}])[-1]


def _swept(monkeypatch, rec, *, instances, now, **knobs):
    """`test_eviction_replacement.py:767`, re-aimed at the ported homes."""
    calls = []
    monkeypatch.setattr(
        lifecycle, "_destroy_and_revoke",
        lambda ids, ins, intent, noun="": (
            calls.append(("destroy", list(ids), intent)), [])[1])
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: calls.append(("emit", event, kw)))
    jc = {"a": argparse.Namespace(**dict({"replacement_retention_hours": None,
                                          "retention_backstop_hours": None},
                                         **knobs)),
          "retained_boxes": [rec], "instances": instances}
    R._job_retention_sweep(jc, now)
    return jc, calls


# --------------------------------------------------------------------------- #
# 1. the constants, and the knob that has no CLI flag
# --------------------------------------------------------------------------- #
def test_the_two_module_constants_are_unchanged():
    """Both are read by exactly one function each and both are asserted on by
    the flat suites (`RETENTION_REQUIESCE_MAX` by name in the resurrection
    block, `SALVAGE_DEFER_GRACE_S` arithmetically in `test_salvage.py`)."""
    assert R.RETENTION_REQUIESCE_MAX == 3
    assert R.SALVAGE_DEFER_GRACE_S == 900.0


def test_the_backstop_grace_resolves_from_the_env_rung_too(monkeypatch):
    """There is no `--retention-backstop-hours` flag anywhere, on purpose: the
    knob resolves from a policy-namespace attr or `$JOB_RETENTION_BACKSTOP_HOURS`
    and nothing else. `test_eviction_replacement.py::
    test_the_backstop_grace_is_configurable` pins the attr rung; this pins the
    env one, so 'fixing' the missing flag cannot pass unnoticed."""
    monkeypatch.setenv("JOB_RETENTION_BACKSTOP_HOURS", "0.05")     # 3 min
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst()], now=dl + 600)
    assert [c for c in calls if c[0] == "destroy"]


# --------------------------------------------------------------------------- #
# 2. the salvage tick drivers (test_salvage.py's "herdd wiring" half)
# --------------------------------------------------------------------------- #
def test_retain_arms_salvage_at_the_MOMENT_of_eviction(monkeypatch):
    """The race is HOST RECLAMATION (~30 min observed), not the 3h window — so
    the record has to exist before anyone reads a runbook."""
    _retain_wire(monkeypatch)
    jc = _jc()
    rec = R._job_retain_or_destroy(jc, "DEAD", {"id": "DEAD"}, "outbid", NOW,
                                   new_iid="NEW")
    assert rec["status"] == "retained"
    assert rec["salvage"]["phase"] == "pending"
    assert rec["salvage"]["dead_iid"] == "DEAD"
    assert "NEW" in rec["salvage"]["dest_candidates"]


def test_already_gone_records_dead_box_gone_rather_than_silence(monkeypatch):
    """`dead_box_gone` is the measured failure rate of the whole idea; folding it
    into 'we didn't try' hides the number that decides whether salvage is worth
    it."""
    _retain_wire(monkeypatch)
    jc = _jc()
    rec = R._job_retain_or_destroy(jc, "DEAD", None, "host_failure", NOW,
                                   new_iid="NEW")
    assert rec["status"] == "already_gone"
    assert rec["salvage"]["outcome"] == S.OUTCOME_DEAD_GONE


def test_no_salvage_flag_disarms_it(monkeypatch):
    _retain_wire(monkeypatch)
    jc = _jc(a=argparse.Namespace(salvage=False))
    rec = R._job_retain_or_destroy(jc, "DEAD", {"id": "DEAD"}, "outbid", NOW,
                                   new_iid="NEW")
    assert rec["salvage"] is None


def test_retention_sweep_advances_salvage(monkeypatch):
    seen = []
    monkeypatch.setattr(R, "_job_salvage_advance",
                        lambda jc, rec, now: seen.append(rec))
    # The flat original leaves the emitter real and rides its swallow. Stubbed
    # here because the swallow is the only thing between an unset `B2_BUCKET`
    # and a live `rclone` — an env-dependent no-subprocess guarantee is not one.
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda *a, **k: {})
    jc = _jc(retained_boxes=[{"iid": "DEAD", "status": "retained",
                              "deadline_ts": NOW + 9999,
                              "salvage": {"phase": "pending",
                                          "dead_iid": "DEAD"}}])
    R._job_retention_sweep(jc, NOW)
    assert len(seen) == 1


def test_salvage_step_error_never_kills_the_supervision_loop(monkeypatch):
    """...and a record that can NEVER advance must still let the box go. The
    try/except is what makes a permanently-stuck record possible, so the two
    properties belong in one test."""
    def boom(*a, **k):
        raise RuntimeError("api exploded")
    monkeypatch.setattr(R, "_job_salvage_advance", boom)
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda *a, **k: {})
    destroyed = []
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
                        lambda ids, inst, why, noun="":
                            destroyed.extend(ids) or [])
    sal = {"phase": "pending", "dead_iid": "DEAD", "deadline_ts": NOW - 7200,
           "started_ts": NOW - 10800}
    jc = _jc(instances=[{"id": "DEAD", "actual_status": "exited"}],
             retained_boxes=[{"iid": "DEAD", "status": "expired",
                              "deadline_ts": NOW - 10 * 3600, "salvage": sal}])
    for i in range(500):                     # 500 ticks of a stuck record
        R._job_retention_sweep(jc, NOW + i * 3600)   # must not raise
    assert sal["phase"] == "pending"          # it really never advanced
    assert destroyed == ["DEAD"]              # and the box was still reclaimed


def test_retention_backstop_DEFERS_while_a_copy_is_in_flight(monkeypatch):
    """Destroying the source mid-transfer aborts the copy — the exact data loss
    the backstop sits downstream of. Salvage has its own deadline, so this can
    only defer by a bounded amount."""
    destroyed = []
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
                        lambda ids, inst, why, noun="":
                            destroyed.extend(ids) or [])
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda *a, **k: {})
    monkeypatch.setattr(R, "_job_salvage_sweep", lambda jc, now: None)
    dl = NOW - 10 * 3600
    jc = _jc(instances=[{"id": "DEAD", "actual_status": "exited"}],
             retained_boxes=[{"iid": "DEAD", "status": "expired",
                              "deadline_ts": dl,
                              "salvage": {"phase": "copying",
                                          "dead_iid": "DEAD",
                                          "deadline_ts": NOW + 600}}])
    R._job_retention_sweep(jc, NOW)
    assert destroyed == []

    jc["retained_boxes"][0]["salvage"]["phase"] = "done"
    R._job_retention_sweep(jc, NOW)
    assert destroyed == ["DEAD"]


def test_retention_deferral_is_BOUNDED_by_the_salvage_deadline(monkeypatch):
    """A record whose advance keeps throwing never reaches a terminal phase.
    Deferring on `phase != done` alone would hold a BILLING box open forever —
    the exact waste the backstop exists to stop."""
    destroyed = []
    monkeypatch.setattr(lifecycle, "_destroy_and_revoke",
                        lambda ids, inst, why, noun="":
                            destroyed.extend(ids) or [])
    monkeypatch.setattr(journal, "_job_handoff_emit", lambda *a, **k: {})
    monkeypatch.setattr(R, "_job_salvage_sweep", lambda jc, now: None)
    sal = {"phase": "copying", "dead_iid": "DEAD",
           "deadline_ts": NOW - 2 * 3600, "started_ts": NOW - 3 * 3600}
    jc = _jc(instances=[{"id": "DEAD", "actual_status": "exited"}],
             retained_boxes=[{"iid": "DEAD", "status": "expired",
                              "deadline_ts": NOW - 10 * 3600, "salvage": sal}])
    R._job_retention_sweep(jc, NOW)
    assert destroyed == ["DEAD"]


def test_salvage_defer_until_falls_back_when_the_deadline_is_missing():
    """A record persisted by an older daemon may lack `deadline_ts`; it must
    still expire rather than defer forever."""
    assert R._salvage_defer_until({"started_ts": 100.0}) == \
        100.0 + S.SALVAGE_DEADLINE_S + R.SALVAGE_DEFER_GRACE_S
    # a record with neither field bounds at epoch 0 + the windows, i.e. long
    # past for any real clock — an absent deadline can never mean "forever"
    fallback = S.SALVAGE_DEADLINE_S + R.SALVAGE_DEFER_GRACE_S
    assert R._salvage_defer_until({}) == fallback
    assert R._salvage_defer_until({"deadline_ts": "junk",
                                   "started_ts": None}) == fallback
    assert R._salvage_defer_until({}) < NOW


def test_the_salvage_sweep_runs_FIRST_and_unconditionally(monkeypatch):
    """Ordering contract: a copy already in flight keeps making progress after
    the retention record has gone terminal, and its result is the whole point of
    holding the box. So the call sits BEFORE the empty-record early return."""
    seen = []
    monkeypatch.setattr(R, "_job_salvage_sweep",
                        lambda jc, now: seen.append(now))
    R._job_retention_sweep(_jc(retained_boxes=[]), NOW)          # no records
    R._job_retention_sweep(_jc(), NOW + 1)                       # no key at all
    R._job_retention_sweep(                                      # all terminal
        _jc(retained_boxes=[{"iid": "41", "status": "reaped"}]), NOW + 2)
    assert seen == [NOW, NOW + 1, NOW + 2]


# --------------------------------------------------------------------------- #
# 3. quiesce — the box we are HOLDING must not be able to come back
# --------------------------------------------------------------------------- #
def test_a_retained_box_is_stopped_and_its_bid_dropped(monkeypatch):
    """THE REGRESSION. Retention must leave the box unable to come back."""
    _calls, seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()])
    rec = R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                                   new_iid="88")
    assert seen["stop"] == [("41", "stopped")]
    assert seen["bid"] == [("41", bp.RETENTION_PARK_BID)]
    assert rec["quiesce"]["stopped"] is True
    assert rec["quiesce"]["bid_pinned"] == bp.RETENTION_PARK_BID


def test_an_ondemand_box_is_stopped_but_never_bid_pinned(monkeypatch):
    """An on-demand instance has no standing bid; PUT-ing a price at one is a
    move against a box that cannot be outbid. The `stop` still applies — a
    queued start is not a bid concept."""
    inst = _inst(dph=1.6)
    inst["is_bid"] = False
    _calls, seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[inst])
    rec = R._job_retain_or_destroy(jc, "41", inst, bp.EVICTION_OUTBID, NOW,
                                   new_iid="88")
    assert seen["stop"] == [("41", "stopped")]
    assert seen["bid"] == []
    assert rec["quiesce"]["bid_pinned"] is None


def test_the_prior_bid_is_recorded_so_a_manual_resume_can_restore_it(
        monkeypatch, capsys):
    """Pinning to $0.001 leaves the box unable to win its market — which is the
    point, and which a human resuming it for salvage has to know to undo."""
    _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()])
    rec = R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                                   new_iid="88")
    assert rec["quiesce"]["prior_bid"] == pytest.approx(0.76)
    assert "standing bid was $0.7600" in capsys.readouterr().out


def test_a_failed_quiesce_still_retains_the_box_and_says_so(monkeypatch, capsys):
    """Best-effort, like the label PUT: failing to defend the box is never a
    reason to throw away the disk the owner asked for. But it must be LOUD —
    an un-quiesced retained box is the incident, live."""
    _retain_wire(monkeypatch)
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (False, "HTTP 500"))
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda i, p: (False, "HTTP 429"))
    jc = _jc(instances=[_inst()])
    rec = R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                                   new_iid="88")
    assert rec["status"] == "retained"
    assert rec["quiesce"]["stopped"] is False
    assert "QUIESCE FAILED" in R._quiesce_summary(rec["quiesce"])
    assert "HTTP 500" in capsys.readouterr().out


def test_the_quiesce_is_journaled_where_fleet_log_can_see_it(monkeypatch):
    """`_job_handoff_emit` writes to B2 only. A money-relevant rung has to reach
    `fleet log`, which is the ladder journal (task #78)."""
    _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()])
    R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                             new_iid="88")
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_box_quiesced"]
    assert ev, "the quiesce never reached the ladder journal"
    assert ev[0]["stopped"] is True
    assert ev[0]["bid_pinned"] == bp.RETENTION_PARK_BID


def test_dry_run_issues_no_stop_and_no_bid_pin(monkeypatch):
    seen = _quiesce_wire(monkeypatch)
    jc = _jc(dry_run=True)
    q = R._job_quiesce_box(jc, "41", _inst(), why="retention")
    assert seen == {"stop": [], "bid": []}
    assert q["stopped"] is None and q["bid_pinned"] is None


def test_quiesce_summary_renders_the_three_shapes():
    """The clause is interpolated into a journal note AND printed; `QUIESCE
    FAILED` is asserted on by name in the flat suite."""
    assert R._quiesce_summary(None) == "not quiesced"
    assert R._quiesce_summary({"errors": ["stop: HTTP 500"], "stopped": False,
                               "bid_pinned": None}) == \
        "QUIESCE FAILED (stop: HTTP 500)"
    assert R._quiesce_summary({"stopped": True, "bid_pinned": 0.001,
                               "errors": []}) == \
        "stopped (queued start withdrawn), bid pinned $0.001 (below any floor)"
    assert R._quiesce_summary({"stopped": False, "bid_pinned": None,
                               "errors": []}) == "nothing to quiesce"


# --------------------------------------------------------------------------- #
# 4. retain or destroy — the decision, and the record it appends
# --------------------------------------------------------------------------- #
def test_retention_is_the_default_the_lost_box_is_held_not_destroyed(monkeypatch):
    calls, _seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()])
    R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                             new_iid="88")
    assert "destroy" not in [c[0] for c in calls]
    rec = _retained(jc)
    assert rec["status"] == "retained" and rec["iid"] == "41"
    assert rec["replacement_iid"] == "88"
    assert rec["deadline_ts"] == pytest.approx(
        NOW + bp.REPLACEMENT_RETENTION_H * 3600.0)


def test_retention_hours_zero_restores_the_immediate_destroy(monkeypatch):
    calls, _seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()],
             a=argparse.Namespace(replacement_retention_hours=0))
    R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                             new_iid="88")
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]
    assert "label" not in [c[0] for c in calls]
    assert _retained(jc)["status"] == "destroyed"


def test_retention_is_journaled_with_its_cost_and_deadline(monkeypatch):
    """Cost disclosure: a box nobody chose to rent must never be a surprise."""
    calls, _seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()])
    R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                             new_iid="88")
    ev = [c for c in calls if c[0] == "emit" and c[1] == "eviction_box_retained"]
    assert ev, "a box was retained with no journal event"
    f = ev[0][2]
    for k in ("box", "deadline", "retention_h", "est_cost_usd",
              "est_cost_hi_usd", "keep_labeled", "eviction_class"):
        assert k in f, f"retention event is missing {k}"
    assert f["deadline"].endswith("Z")
    assert 0 < f["est_cost_usd"] <= f["est_cost_hi_usd"] < 1.0   # ~3h of disk


def test_a_label_put_failure_keeps_the_box_and_says_the_window_is_unsafe(
        monkeypatch, capsys):
    """Failing to defend the box is not a reason to destroy it — the owner
    asked for the disk. But the operator must be told the reaper may take it at
    2h, because the label is the only thing that would have stopped it."""
    calls, _seen = _retain_wire(monkeypatch, label_fail="HTTP 500")
    jc = _jc(instances=[_inst()])
    R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                             new_iid="88")
    assert "destroy" not in [c[0] for c in calls]
    rec = _retained(jc)
    assert rec["status"] == "retained" and rec["keep_labeled"] is False
    assert "2h idle mark" in capsys.readouterr().out


def test_a_box_already_out_of_the_listing_is_already_gone_not_retained(monkeypatch):
    """Host failure / spot reclaim: nothing to retain, nothing to destroy — and
    it is journaled as its OWN class, because folding it into `retained` would
    overstate how often the disk was actually available to salvage."""
    calls, _seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[])
    R._job_retain_or_destroy(jc, "41", None, bp.EVICTION_HOST_FAILURE, NOW,
                             new_iid="88")
    assert _retained(jc)["status"] == "already_gone"
    assert [c for c in calls if c[0] == "emit"
            and c[1] == "eviction_box_already_gone"]
    assert "destroy" not in [c[0] for c in calls]


def test_the_retained_record_keys_are_the_frozen_fleetd_contract(monkeypatch):
    """fleetd reads these keys directly (`_retention_status_map`,
    `retention_rows`, `retention_alarms`, the ceiling rows) and
    `_replacement_state_persist` writes them into `state.json` — plan §4
    load-compat. Renaming one silently blanks a row rather than failing."""
    _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()])
    rec = R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                                   new_iid="88")
    assert set(rec) == {"iid", "status", "class", "retained_ts", "deadline_ts",
                        "retention_h", "cost_usd", "cost_hi_usd",
                        "storage_day_usd", "replacement_iid", "label",
                        "keep_labeled", "quiesce", "salvage"}


def test_the_retention_dry_run_skips_only_the_label_put(monkeypatch):
    """The dry-run threading is per-function and inconsistent BY DESIGN. Here it
    gates the label PUT and nothing else — the record is still appended, and the
    quiesce still returns its dry-run shape."""
    calls, seen = _retain_wire(monkeypatch)
    jc = _jc(instances=[_inst()], dry_run=True)
    rec = R._job_retain_or_destroy(jc, "41", _inst(), bp.EVICTION_OUTBID, NOW,
                                   new_iid="88")
    assert "label" not in [c[0] for c in calls]
    assert rec["keep_labeled"] is False
    assert seen == {"stop": [], "bid": []}
    assert rec["quiesce"]["stopped"] is None
    assert _retained(jc) is rec


# --------------------------------------------------------------------------- #
# 5. the sweep: every retained box reaches a terminal outcome
# --------------------------------------------------------------------------- #
def test_sweep_leaves_an_open_window_alone(monkeypatch):
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600),
                       instances=[_inst()], now=NOW + 600)
    assert jc["retained_boxes"][0]["status"] == "retained"
    assert not calls


def test_sweep_marks_the_window_expired_and_leaves_reap_to_reclaim(monkeypatch):
    """Expiry does NOT need fleetd: the label stopped keeping the box, and
    `herdd reap`'s 15-minute timer owns idle-storage reclamation. The sweep
    only records it."""
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600),
                       instances=[_inst()], now=NOW + 3 * 3600 + 60)
    assert jc["retained_boxes"][0]["status"] == "expired"
    assert "destroy" not in [c[0] for c in calls]
    assert [c for c in calls if c[0] == "emit"
            and c[1] == "eviction_retention_expired"]


def test_the_backstop_destroys_a_box_the_reaper_never_took(monkeypatch):
    """A retention that never expires recreates the orphaned-billing problem.
    Past deadline + grace with the box STILL listed, the ladder finishes it —
    the case where the reap timer is simply not installed."""
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst()],
                       now=dl + bp.RETENTION_BACKSTOP_GRACE_H * 3600 + 1)
    assert [c for c in calls if c[0] == "destroy"][0][1] == ["41"]
    assert jc["retained_boxes"][0]["status"] == "destroyed"


def test_the_backstop_never_kills_a_box_someone_resumed_to_salvage(monkeypatch):
    """A LIVE retained box is a human mid-salvage; destroying it is exactly the
    data loss retention exists to prevent."""
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst(status="running")],
                       now=dl + 10 * 3600)
    assert "destroy" not in [c[0] for c in calls]


def test_a_box_that_vanishes_before_its_deadline_is_retention_lost(monkeypatch):
    """The measured failure rate of the retention promise on SPOT boxes. It is
    never folded into the clean outcome — a host that reclaims the stopped bid
    instance took the disk with it."""
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600), instances=[],
                       now=NOW + 600)
    assert jc["retained_boxes"][0]["status"] == "retention_lost"
    ev = [c for c in calls if c[0] == "emit"
          and c[1] == "eviction_retention_ended"][0][2]
    assert ev["outcome"] == "retention_lost"


def test_a_box_gone_after_its_deadline_is_the_designed_outcome(monkeypatch):
    jc, calls = _swept(monkeypatch, _rec(NOW + 3 * 3600), instances=[],
                       now=NOW + 4 * 3600)
    assert jc["retained_boxes"][0]["status"] == "reaped"


def test_the_backstop_grace_is_configurable(monkeypatch):
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _rec(dl, status="expired"),
                       instances=[_inst()], now=dl + 600,
                       retention_backstop_hours=0.05)      # 3 min
    assert [c for c in calls if c[0] == "destroy"]


def test_the_sweep_dry_run_skips_only_the_backstop_destroy(monkeypatch):
    """Second half of the per-function dry-run threading: the record still goes
    `expired` and still journals; only the destroy is withheld."""
    dl = NOW + 3 * 3600
    calls = []
    monkeypatch.setattr(
        lifecycle, "_destroy_and_revoke",
        lambda ids, ins, intent, noun="": (
            calls.append(("destroy", list(ids), intent)), [])[1])
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jctx, event, **kw: calls.append(("emit", event, kw)))
    rec = _rec(dl, status="expired")
    jc = {"a": argparse.Namespace(replacement_retention_hours=None,
                                  retention_backstop_hours=None),
          "retained_boxes": [rec], "instances": [_inst()], "dry_run": True}
    R._job_retention_sweep(jc, dl + bp.RETENTION_BACKSTOP_GRACE_H * 3600 + 1)
    assert "destroy" not in [c[0] for c in calls]
    assert rec["status"] == "expired"


def test_the_five_terminal_status_spellings_are_the_fleetd_note_keys(monkeypatch):
    """`retained | expired | reaped | retention_lost | destroyed |
    destroy_failed | already_gone` key fleetd's RETENTION_NOTES. Any renaming
    silently blanks its journal note, so the strings are pinned by driving each
    one out of the sweep rather than by quoting a list."""
    seen = set()
    for rec, insts, now in (
            (_rec(NOW + 3 * 3600), [_inst()], NOW + 600),                 # retained
            (_rec(NOW + 3 * 3600), [_inst()], NOW + 3 * 3600 + 60),       # expired
            (_rec(NOW + 3 * 3600), [], NOW + 4 * 3600),                   # reaped
            (_rec(NOW + 3 * 3600), [], NOW + 600),                        # lost
            (_rec(NOW + 3 * 3600, status="expired"), [_inst()],
             NOW + 3 * 3600 + bp.RETENTION_BACKSTOP_GRACE_H * 3600 + 1)):  # destroyed
        jc, _calls = _swept(monkeypatch, rec, instances=insts, now=now)
        seen.add(jc["retained_boxes"][0]["status"])
    assert seen == {"retained", "expired", "reaped", "retention_lost",
                    "destroyed"}


# --------------------------------------------------------------------------- #
# 6. resurrection — a retained box that is RUNNING again (box 47833510)
# --------------------------------------------------------------------------- #
def test_a_live_retained_box_inside_its_window_is_caught_and_re_parked(
        monkeypatch, capsys):
    """THE OBSERVABILITY REGRESSION. The old sweep returned at `now < deadline`
    without ever looking at `actual_status`, so the whole 2h59m of the 47833510
    window was a blind spot by construction."""
    seen = _quiesce_wire(monkeypatch)
    jc, calls = _swept(monkeypatch, _live_rec(NOW + 3 * 3600),
                       instances=[_inst(status="running", dph=0.8407)],
                       now=NOW + 3600)
    r = jc["retained_boxes"][0]
    assert r["status"] == "retained"           # still retained, NOT destroyed
    assert r["live_since_ts"] == NOW + 3600
    assert r["resurrections"] == 1 and r["requiesces"] == 1
    assert seen["stop"] == [("41", "stopped")]
    assert seen["bid"] == [("41", bp.RETENTION_PARK_BID)]
    assert "RUNNING again" in capsys.readouterr().out


def test_the_resurrection_reaches_fleet_log_with_the_money_on_it(monkeypatch):
    _quiesce_wire(monkeypatch)
    jc, _calls = _swept(monkeypatch, _live_rec(NOW + 3 * 3600),
                        instances=[_inst(status="running", dph=0.8407)],
                        now=NOW + 3600)
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_retained_box_resurrected"]
    assert ev, "a retained box came back to life and `fleet log` said nothing"
    assert ev[0]["dph"] == pytest.approx(0.8407)
    assert ev[0]["live_multiple"] == pytest.approx(20.6, abs=0.1)


def test_a_live_retained_box_is_never_destroyed_by_the_sweep(monkeypatch):
    """Re-parking is not destroying. The disk is the whole reason the box is
    being held, and a human mid-salvage must not lose it."""
    _quiesce_wire(monkeypatch)
    dl = NOW + 3 * 3600
    jc, calls = _swept(monkeypatch, _live_rec(dl, status="expired"),
                       instances=[_inst(status="running")],
                       now=dl + 10 * 3600)
    assert "destroy" not in [c[0] for c in calls]


def test_the_re_park_is_bounded_and_then_only_alarms(monkeypatch):
    """A host that keeps re-placing the instance is not winnable by PUT-ing at
    it every 45s. Past RETENTION_REQUIESCE_MAX the ladder stops acting and the
    standing alarm owns it."""
    seen = _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600, requiesces=R.RETENTION_REQUIESCE_MAX)
    jc, _calls = _swept(monkeypatch, rec, now=NOW + 3600,
                        instances=[_inst(status="running", dph=0.8407)])
    assert seen == {"stop": [], "bid": []}
    assert jc["retained_boxes"][0]["requiesces"] == R.RETENTION_REQUIESCE_MAX
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_retained_box_resurrected"]
    assert "destroy or salvage it by hand" in ev[0]["note"]


def test_a_box_the_ladder_never_quiesced_is_alarmed_but_not_touched(monkeypatch):
    """No `quiesce` record means someone else owns this box's state — a record
    written before this change, or a hand-managed retention. Alarm, hands off."""
    seen = _quiesce_wire(monkeypatch)
    rec = _rec(NOW + 3 * 3600)                       # no `quiesce` key at all
    jc, _calls = _swept(monkeypatch, rec, now=NOW + 3600,
                        instances=[_inst(status="running")])
    assert seen == {"stop": [], "bid": []}
    assert jc["retained_boxes"][0]["live_since_ts"] == NOW + 3600
    ev = [f for name, f in (jc.get("ladder_journal") or [])
          if name == "jobs_retained_box_resurrected"]
    assert "never quiesced by the ladder" in ev[0]["note"]


def test_the_liveness_flag_retracts_itself_when_the_box_goes_back_down(
        monkeypatch):
    """Derived, not latched: the alarm has to disappear on its own when the
    re-park lands, or an operator ends up acking a condition that fixed itself."""
    _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600)
    _swept(monkeypatch, rec, instances=[_inst(status="running", dph=0.8407)],
           now=NOW + 3600)
    assert rec["live_since_ts"] == NOW + 3600
    _swept(monkeypatch, rec, instances=[_inst(status="exited")], now=NOW + 3700)
    assert "live_since_ts" not in rec
    assert rec["resurrections"] == 1               # the COUNT is not retracted


def test_one_resurrection_journals_once_not_once_per_tick(monkeypatch):
    """45s ticks over a 3h window is 240 identical lines — a log nobody reads."""
    _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600)
    lines = 0
    for i in range(5):
        jc, _c = _swept(monkeypatch, rec, now=NOW + 3600 + i * 45,
                        instances=[_inst(status="running", dph=0.8407)])
        lines += len([1 for name, _f in (jc.get("ladder_journal") or [])
                      if name == "jobs_retained_box_resurrected"])
    assert lines == 1
    assert rec["requiesces"] == 1


def test_a_failed_re_park_is_retried_up_to_the_bound(monkeypatch):
    """A `stop` that 500s must not be a one-shot — but it must not be an
    unbounded retry loop either."""
    monkeypatch.setattr(lifecycle, "_put_state_soft",
                        lambda i, s: (False, "HTTP 500"))
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda i, p: (False, "HTTP 500"))
    rec = _live_rec(NOW + 3 * 3600)
    for i in range(6):
        _swept(monkeypatch, rec, now=NOW + 3600 + i * 45,
               instances=[_inst(status="running", dph=0.8407)])
    assert rec["requiesces"] == R.RETENTION_REQUIESCE_MAX


def test_an_unreadable_rate_stays_unknown_and_is_never_zero(monkeypatch):
    """Tri-state None-for-UNKNOWN, all the way through: a box we cannot price is
    unpriced. A `0.0` here would read as "this resurrection was free", which is
    exactly the reassurance that let the real one run for an hour."""
    _quiesce_wire(monkeypatch)
    inst = _inst(status="running")
    inst["dph_total"] = None
    jc, _calls = _swept(monkeypatch, _live_rec(NOW + 3 * 3600),
                        instances=[inst], now=NOW + 3600)
    r = jc["retained_boxes"][0]
    assert r["live_dph"] is None
    assert r["live_cost_usd"] is None and r["live_multiple"] is None


def test_the_live_fields_are_the_ones_fleet_status_alarms_on(monkeypatch):
    """`fleetd.retention_alarms` keys on `live_since_ts` and prints
    `live_dph` / `live_cost_usd` / `live_multiple` / `replacement_iid`. The
    builder is not ported until step 5; the field names it reads are written
    here, so they are pinned here."""
    _quiesce_wire(monkeypatch)
    rec = _live_rec(NOW + 3 * 3600, replacement_iid="88")
    jc, _calls = _swept(monkeypatch, rec,
                        instances=[_inst(status="running", dph=0.8407)],
                        now=NOW + 3600)
    r = jc["retained_boxes"][0]
    for k in ("live_since_ts", "live_dph", "live_cost_usd", "live_multiple",
              "resurrections", "requiesces", "replacement_iid"):
        assert k in r, k
