"""`vastlib.fleet.rows` — the port, graded against the live `fleetd` copy.

Why this file exists
--------------------
Plan §8's add-only phase left TWO implementations of every row builder in the
tree: `fleetd.<fn>`, which the running daemon actually called, and
`vastlib.fleet.rows.<fn>`, which nothing called yet. A port test that only
exercised the new one would have been checking this file's beliefs about fleetd
rather than fleetd. So the spine of this file WAS a PARITY harness: same input,
both implementations, assert the two outputs are equal.

**Step 6d emptied `fleetd.py`.** Every `fleetd.<fn>` here is now an identity
re-export of the `rows` function beside it, so the thirteen parity tests in §1
compared each builder with itself; they are deleted, and this file is now what
was always its second layer — the properties a parity test could not see,
because they were true of both copies and would have survived a shared mistake:

* the explicit `None`s (`spend_usd`, `budget_usd`, `divergence_pct`) that exist
  to stop `fleet status` printing a reassuring number for a box it knows
  nothing about;
* the `retention:<iid>:live` alarm KEY, which is raise/resolve identity across a
  daemon restart and therefore schema;
* `workload_evidence`'s ORDER, where a measured zombie must beat a mere label;
* `label_exempt`'s token-not-substring grammar;
* `watch_box_iid`'s `"None"` string guard, a `state.json` round-trip defence.

`ceiling_rows` is the one impure builder (it reads the fail-closed adopt
default), so both sides' config lookups are pinned to one number rather than
left to whatever `herdd.yaml` says on this machine.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import pytest

from vastlib.boxes import health
from vastlib.core import config, models
from vastlib.fleet import rows

import fleetd
import vastconf
import herdd

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CAP = 3.0
NOW = 1_000_000.0


@pytest.fixture(autouse=True)
def _pin_the_adopt_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ceiling_rows` is the only symbol here that reads config. Pin BOTH sides
    to the same figure so a parity failure is a port bug and never a difference
    of opinion between `vastconf` and `core.config` about this box's yaml."""
    monkeypatch.setattr(vastconf, "fleetd_adopt_default_budget_usd",
                        lambda *a, **k: DEFAULT_CAP)
    monkeypatch.setattr(config, "fleetd_adopt_default_budget_usd",
                        lambda *a, **k: DEFAULT_CAP)


def _state() -> dict[str, Any]:
    """One state document exercising every branch the builders have: a watched
    box that moved (`iid` != key), a run watch, a garbage watch, two retention
    records (one live, one not), a stale stray, a fresh stray, a doomed stray,
    and a ceiling ledger with a healthy record and two unreadable ones."""
    return {
        "version": 1,
        "watches": {
            "47214941": {
                "profile": "jobs", "iid": "47219872", "ceiling_id": "c1",
                "created_ts": NOW - 3600.0, "budget_usd": 5.0,
                "unrecoverable_since": None,
                "replacement": {
                    "rebid_rungs": 2, "resume_tries": 0,
                    "retained_boxes": [
                        {"iid": "47214941", "status": "retained",
                         "class": "spot_reclaim", "deadline_ts": NOW + 900.0,
                         "cost_usd": 0.12, "cost_hi_usd": 0.30,
                         "keep_labeled": True, "replacement_iid": "47219872",
                         "live_since_ts": NOW - 300.0, "live_dph": 2.804,
                         "live_cost_usd": 0.2337, "live_multiple": 42,
                         "requiesces": 3},
                        {"iid": "44612403", "status": "retention_lost",
                         "deadline_ts": NOW - 10.0},
                    ],
                },
            },
            "run:driftr3": {"profile": "run", "iid": None, "ceiling_id": "c2"},
            "46687567": {"profile": "bare", "iid": "None",
                         "unrecoverable_since": NOW - 60.0},
        },
        "ceilings": {
            "c1": {"cap_usd": 5.0, "spend_usd": 1.25, "source": "operator",
                   "origin_target": "47214941", "requester": "operator@workstation",
                   "epochs": 1, "members": ["47214941", "47219872"],
                   "last_verdict": None},
            "c2": {"cap_usd": float("nan"), "spend_usd": -3.0},
            "c3": "not a record",
        },
        "ceiling_by_box": {"47219872": "c1"},
        "strays": {
            "47800001": {"live_ts": NOW - 60.0, "paused_until": NOW + 10.0},
            "47800002": {"live_ts": NOW - 5000.0},          # stale
            "47800003": {},                                  # no live_ts at all
            "47800004": {"live_ts": NOW - 10.0, "parked_ts": NOW - 5.0},
            "47800005": {"live_ts": NOW - 10.0},             # doomed below
            "47800006": "not a dict",
        },
        "destroys": {"47800005": {"when": "drained"}},
        "intents": {},
        "spend_by_box": {"47219872": 1.25, "47800009": 0.5},
        "meta": {}, "alarms": {}, "notify": {},
    }


def _instances() -> list[dict[str, Any]]:
    return [
        {"id": 47219872, "dph_total": 0.8, "start_date": NOW - 7200.0},
        {"id": 47800009, "dph_total": 1.5, "start_date": NOW - 3600.0},
        {"id": 47800010, "dph_total": None, "start_date": None},
        {"id": 47800011, "dph_total": 0.4, "start_date": "garbage"},
        {"id": 47800012, "dph_total": 0.4, "start_date": NOW + 500.0},  # future
        {"id": None},
    ]


# --------------------------------------------------------------------------- #
# 1. THE BINDING — what is left of the parity spine (plan §8 step 6d)
# --------------------------------------------------------------------------- #

# Thirteen parity tests stood here, grading the port against the copy the
# daemon actually ran. Deleted at step 6d, with the second copy they graded against: `ceiling_rows`,
# `retention_rows`, `retention_alarms`, `recoveries_in_flight`, `stray_rows`,
# `reconcile_rows`, `handoff_predecessor` (8 labels), `_num` (15 values),
# `normalize_ceiling` (12 records), `watch_box_iid` (9 watches), `label_exempt`
# (9 labels), `_retention_fate` (9 statuses) and `_retention_status_map`. Each
# ran one function twice. The properties they were a proxy for are asserted in
# §2-§6 below, on the surviving copy; what a parity test could never state — a
# shared mistake — is unchanged either way.


def test_the_launcher_re_exports_rather_than_redefines() -> None:
    """The residue: `fleetd.<fn>` must stay a BINDING to this module.

    The daemon executes `tools/vast/fleetd.py`, and ~30 test modules plus every
    runbook name these functions through it. A second body there would be a
    daemon rendering different rows from everything that inspects it — the
    two-implementations condition the deleted parity spine existed to police,
    with nothing left watching for it."""
    for name in ("BOOT_EVIDENCE_S", "EXEMPT_LABEL_TOKENS", "JOBD_FRESH_S",
                 "RETENTION_NOTES", "UNWATCHED_STALE_S", "_RETENTION_FATE",
                 "_num", "_retention_fate", "_retention_status_map",
                 "ceiling_rows", "handoff_predecessor", "label_exempt",
                 "normalize_ceiling", "reconcile_rows", "recoveries_in_flight",
                 "retention_alarms", "retention_rows", "stray_rows",
                 "watch_box_iid", "workload_evidence"):
        assert getattr(fleetd, name) is getattr(rows, name), (
            f"fleetd.{name} is a second body again — the launcher must "
            f"re-export vastlib.fleet.rows' object, never redefine it")


def test_the_builders_do_not_all_tolerate_a_garbage_watch_record() -> None:
    """FOUND, NOT FIXED. The four builders that walk
    `state["watches"]` guard THREE different amounts:

      * `recoveries_in_flight` guards `isinstance(w, dict)` AND
        `isinstance(repl, dict)` — tolerates everything;
      * `retention_alarms` guards the WATCH but not the `replacement` record;
      * `ceiling_rows` and `retention_rows` guard neither.

    So a `None` watch or a non-dict `replacement` raises an `AttributeError`
    inside a `fleet status` render, in three of the four. Ported as found (plan
    §7.4 forbids expectation changes) — the daemon only ever writes dicts
    there. This test stops the port from silently ACQUIRING a tolerance the
    live code did not have, which would hide the defect instead of pinning it.
    The `fleetd.` half of each pair went at step 6d — same objects."""
    null_watch: dict[str, Any] = {"watches": {"garbage": None}, "destroys": {}}
    bad_repl: dict[str, Any] = {"watches": {"t": {"replacement": "not a dict"}},
                                "destroys": {}}

    for doc in (null_watch, bad_repl):
        assert rows.recoveries_in_flight(doc) == []
    assert rows.retention_alarms(null_watch, NOW) == []

    raisers = [(rows.ceiling_rows, null_watch),
               (rows.retention_rows, null_watch),
               (rows.retention_rows, bad_repl),
               (rows.retention_alarms, bad_repl)]
    for fn_new, doc in raisers:
        with pytest.raises(AttributeError):
            fn_new(doc, NOW)


def test_retention_text_tables_are_identical() -> None:
    """These strings land in journal notes an operator reads to decide whether a
    box is still costing them money. Reformatting one is a schema edit."""
    assert "bills ALLOCATED disk" in rows.RETENTION_NOTES["retained"]
    assert "destroy it by hand" in rows.RETENTION_NOTES["destroy_failed"]
    assert "still bills allocated disk" in rows._RETENTION_FATE["retained"]


def test_constants_are_the_pinned_numbers() -> None:
    """Was `…_match_fleetd`; those four are one object each since step 6d (the
    binding is asserted above). `UNWATCHED_STALE_S` keeps its literal, which is
    the half that was never parity."""
    assert rows.UNWATCHED_STALE_S == 900.0
    assert rows.BOOT_EVIDENCE_S > 0 and rows.JOBD_FRESH_S > 0
    assert "nofleet" in rows.EXEMPT_LABEL_TOKENS


# --------------------------------------------------------------------------- #
# 2. workload_evidence — ORDER is the contract, and the lattice is read, not
#    re-derived
# --------------------------------------------------------------------------- #

# `test_workload_evidence_parity` swept `_EVIDENCE_CASES` — fifteen
# instance/row pairs covering every guard verdict, the two label shapes and the
# boot/heartbeat ages — through both copies. One copy since step 6d, so both
# the sweep and its table are deleted: the ORDER contract they were chosen to
# pin is asserted directly in the four tests below, which walk the whole
# `GuardVerdict` enum rather than a hand-picked list.


def test_a_measured_zombie_is_never_rescued_by_a_label() -> None:
    """The order (booting -> boot age -> heartbeat -> ZOMBIE -> label -> jobs
    box) is the safety-net contract. A label sits BELOW the zombie test, so a
    box whose workload has been measured dead gets no evidence and the safety
    net may park it."""
    inst = {"label": "upstream-monorepo:jobs:keep"}
    for v in health.GuardVerdict:
        row = {"verdict": v.value}
        got = rows.workload_evidence(inst, row)
        if v.is_zombie:
            assert got is None, v
        elif v is health.GuardVerdict.BOOTING:
            assert got == "booting"
        else:
            assert got == f"label {inst['label']!r}", v


def test_advisory_verdicts_are_not_zombies_here_either() -> None:
    """STALE_IMAGE / LOADING_SLOW alarm but license no action, so they must NOT
    suppress evidence — a healthy box running old code is still working."""
    inst = {"label": "upstream-monorepo:jobs"}
    for v in (health.GuardVerdict.STALE_IMAGE, health.GuardVerdict.LOADING_SLOW):
        assert rows.workload_evidence(inst, {"verdict": v.value}) is not None


def test_evidence_reads_the_lattice_rather_than_re_deriving_it() -> None:
    """Cross-check against `herdd`'s sets — NOT tautological post-6d.

    `_GUARD_ZOMBIE_VERDICTS` is the one thing the thin launcher BUILDS rather
    than re-exports: the three verdict tables it used to carry no longer exist
    upstream (plan §5 unified them into `GuardVerdict.is_zombie` /
    `.is_advisory` / `.short`), so it rebuilds them from the enum. This asserts
    that rebuild agrees with the enum it was derived from, and that the
    membership the row builder reads is the same membership. `fleetd` reads
    these through the same lattice."""
    for v in herdd._GUARD_ZOMBIE_VERDICTS:
        assert health.verdict_is_zombie(v), v
        assert rows.workload_evidence({"label": "run:x"}, {"verdict": v}) is None
    assert health.GUARD_BOOTING == herdd.GUARD_BOOTING


def test_boot_evidence_beats_a_zombie_verdict() -> None:
    """A freshly booted box is busy by construction, and the boot-age test sits
    ABOVE the zombie short-circuit. Pinned because it is the one place the
    ordering favours evidence over a measured verdict."""
    row = {"verdict": herdd.GUARD_ZOMBIE_NO_JOBD, "evidence": {"boot_age_s": 5}}
    assert rows.workload_evidence({}, row) == "booted 5s ago"


# --------------------------------------------------------------------------- #
# 3. the render contract — explicit None is an answer, not a gap
# --------------------------------------------------------------------------- #

def test_stray_rows_keep_their_literal_nones() -> None:
    """H7. `fleet status` concatenates these with watch rows. Turning
    `spend_usd` into 0.0 would print a reassuring figure for a box fleetd has
    never accrued a cent against."""
    out = rows.stray_rows(_state(), NOW)
    assert {r["iid"] for r in out} == {"47800001", "47800004"}
    for r in out:
        assert r["spend_usd"] is None
        assert r["budget_usd"] is None
        assert r["profile"] == "-"
        assert r["state"] == "UNWATCHED"
    by_iid = {r["iid"]: r for r in out}
    assert by_iid["47800001"]["paused"] is True
    assert by_iid["47800001"]["last_action"] is None
    assert by_iid["47800004"]["last_action"] == "parked"


def test_a_stray_with_a_queued_destroy_never_renders() -> None:
    """The operator has already decided that box's fate; offering it as an
    unwatched stray invites a second decision on it."""
    assert "47800005" not in {r["iid"] for r in rows.stray_rows(_state(), NOW)}


def test_a_stray_with_no_live_ts_is_treated_as_stale() -> None:
    """Conservative direction: the field postdates the record, and a record a
    stamping daemon never stamped cannot have been seen live."""
    assert "47800003" not in {r["iid"] for r in rows.stray_rows(_state(), NOW)}


def test_divergence_pct_is_none_when_the_upper_bound_is_zero() -> None:
    """H7. A percentage of nothing is not 0%, it is unknown — and `ub == 0` is
    falsy, which is why the guard is `if (ub and ub > 0)` and not `is not
    None`."""
    st = {"spend_by_box": {"1": 0.0}, "watches": {}}
    inst = [{"id": 1, "dph_total": 0.0, "start_date": NOW - 3600.0}]
    row = rows.reconcile_rows(st, inst, NOW)[0]
    assert row["upper_bound_usd"] is None
    assert row["divergence_usd"] is None
    assert row["divergence_pct"] is None


def test_an_unwatched_box_is_all_divergence() -> None:
    """The single largest line the 2026-08-08 accounting missed: a box nobody
    watched bills exactly the same as one somebody did."""
    st = {"spend_by_box": {}, "watches": {}}
    inst = [{"id": 5, "dph_total": 1.0, "start_date": NOW - 3600.0}]
    row = rows.reconcile_rows(st, inst, NOW)[0]
    assert row["watched"] is False
    assert row["accrued_usd"] == 0.0
    assert row["upper_bound_usd"] == pytest.approx(1.0)
    assert row["divergence_pct"] == 100.0


def test_unwatched_head_is_the_window_before_the_watch_existed() -> None:
    st = {"spend_by_box": {}, "watches": {"5": {"iid": "5",
                                                "created_ts": NOW - 1800.0}}}
    inst = [{"id": 5, "dph_total": 1.0, "start_date": NOW - 3600.0}]
    row = rows.reconcile_rows(st, inst, NOW)[0]
    assert row["watched"] is True
    assert row["unwatched_head_s"] == 1800.0


def test_reconcile_covers_boxes_that_have_left_the_listing() -> None:
    out = {r["iid"]: r for r in rows.reconcile_rows(_state(), _instances(), NOW)}
    assert out["47800009"]["present"] is True
    st = dict(_state())
    assert rows.reconcile_rows(st, [], NOW)[0]["present"] is False


# --------------------------------------------------------------------------- #
# 4. the ceiling ledger — fail closed, and say why in a string somebody journals
# --------------------------------------------------------------------------- #

def test_a_nan_cap_is_not_an_unbreachable_ceiling() -> None:
    """H4. A NaN compares false against every bound, so an uncoerced NaN cap
    reads as the 'unlimited' this ledger refuses to express."""
    assert rows._num(float("nan")) is None
    cap, spend, why = rows.normalize_ceiling({"cap_usd": float("nan")}, DEFAULT_CAP)
    assert cap == DEFAULT_CAP and spend == 0.0
    assert why is not None and "unreadable" in why
    assert not math.isnan(cap)


def test_num_is_not_num_dph() -> None:
    """The two coercions in this file are different functions with different
    rejections, and the row builders call them at different sites. Naming them
    apart is the point; `_num_dph` accepts values `_num` refuses."""
    assert rows._num is not models._num_dph
    assert rows._num(float("inf")) is None
    assert rows._num("1.5") == 1.5


def test_ceiling_row_reasons_are_the_journaled_text() -> None:
    _, _, why = rows.normalize_ceiling({"cap_usd": -1.0}, DEFAULT_CAP)
    assert why == "cap_usd=-1.0 is not positive"
    _, _, why = rows.normalize_ceiling("nope", DEFAULT_CAP)
    assert why == "ceiling record is not an object"


def test_ceiling_rows_show_headroom_for_a_ceiling_with_no_live_watch() -> None:
    out = {r["ceiling_id"]: r for r in rows.ceiling_rows(_state(), NOW)}
    assert out["c1"]["live_boxes"] == ["47219872"], "indexed by iid, not by key"
    assert out["c1"]["remaining_usd"] == 3.75
    assert out["c3"]["live_boxes"] == [], "no watch: the headroom a re-arm inherits"
    assert out["c3"]["cap_usd"] == DEFAULT_CAP
    assert out["c3"]["degraded"] == "ceiling record is not an object"
    assert out["c2"]["spend_usd"] == 0.0, "a negative spend reads as 0.0"


def test_ceiling_rows_accepts_and_ignores_now() -> None:
    st = _state()
    assert rows.ceiling_rows(st) == rows.ceiling_rows(st, NOW)


# --------------------------------------------------------------------------- #
# 5. retention rows + the alarm KEY
# --------------------------------------------------------------------------- #

def test_retention_rows_disclose_that_a_retained_box_is_live_again() -> None:
    """The disclosed `est_cost_usd` is ALLOCATED DISK. The `live_*` triple is
    what says that is no longer the truth."""
    out = rows.retention_rows(_state(), NOW)
    assert len(out) == 1, "only `retained` / `expired` render"
    r = out[0]
    assert r["iid"] == "47214941"
    assert r["deadline"] == fleetd.iso(NOW + 900.0)
    assert r["left_s"] == 900.0
    assert r["est_cost_usd"] == 0.12
    assert r["live_since"] is not None
    assert r["live_dph"] == 2.804
    assert r["live_cost_usd"] == 0.2337


def test_retention_alarm_key_format_is_schema() -> None:
    """`retention:<iid>:live` is the identity fleetd raises, resolves and dedups
    against ACROSS a restart. A reformat double-raises every retention alarm
    once and never resolves the old key."""
    alarms = rows.retention_alarms(_state(), NOW)
    assert len(alarms) == 1
    key, msg = alarms[0]
    assert key == "retention:47214941:live"
    assert "RETAINED box is RUNNING again" in msg
    assert "$2.8040/hr" in msg
    assert "$0.23 so far" in msg
    assert "42x the disclosed storage-only rate" in msg
    assert "re-parked it 3x" in msg
    assert "fleet destroy 47214941" in msg


def test_retention_alarm_says_unknown_rather_than_zero() -> None:
    st: dict[str, Any] = {"watches": {"t": {"replacement": {"retained_boxes": [
        {"iid": "1", "status": "retained", "live_since_ts": NOW - 60.0}]}}}}
    _, msg = rows.retention_alarms(st, NOW)[0]
    assert "an unknown rate" in msg and "cost unknown" in msg


def test_a_retained_box_that_is_not_live_does_not_alarm() -> None:
    """Self-retracting by construction: the alarm is derived from
    `live_since_ts` on every read, so it goes out the tick the box is stopped
    with nobody acking anything."""
    st: dict[str, Any] = {"watches": {"t": {"replacement": {"retained_boxes": [
        {"iid": "1", "status": "retained"}]}}}}
    assert rows.retention_alarms(st, NOW) == []


# --------------------------------------------------------------------------- #
# 6. the `fleet restart` guard
# --------------------------------------------------------------------------- #

def test_recoveries_in_flight_reads_only_durable_fields() -> None:
    out = rows.recoveries_in_flight(_state())
    kinds = {(r["kind"], str(r["iid"])) for r in out}
    assert ("rebid_ladder", "47219872") in kinds
    assert ("replacement", "47214941") in kinds
    assert ("unrecoverable", "None") in kinds
    assert ("destroy_queued", "47800005") in kinds
    assert not any(r["kind"] == "resume_in_place" for r in out), "0 rungs is quiet"


def test_recoveries_in_flight_is_empty_on_a_quiet_fleet() -> None:
    assert rows.recoveries_in_flight({"watches": {}, "destroys": {}}) == []
    assert rows.recoveries_in_flight({}) == []


def test_handoff_phase_is_deliberately_invisible_to_the_guard() -> None:
    """Stated out loud in the docstring rather than implied: handoff phase is
    runtime-only and never reaches state.json, so this fold cannot see it. A
    guard that silently does not cover a case is worse than no guard."""
    assert "deliberately NOT here" in (rows.recoveries_in_flight.__doc__ or "")


# --------------------------------------------------------------------------- #
# 7. the two predicates whose failure modes cost money
# --------------------------------------------------------------------------- #

def test_watch_box_iid_rejects_the_state_json_round_trip_garbage() -> None:
    """`str(None)` through state.json leaves the four-character string `"None"`
    where an id should be. Reading it as an id is how two spend-control
    incidents happened (2026-08-05)."""
    assert rows.watch_box_iid({"iid": "None"}) is None
    assert rows.watch_box_iid({"iid": ""}) is None
    assert rows.watch_box_iid({"iid": None}) is None
    assert rows.watch_box_iid({"iid": 47219872}) == "47219872"
    assert rows.watch_box_iid({"profile": "run", "iid": "47"}) is None


def test_label_exempt_is_a_token_match_not_a_substring() -> None:
    assert rows.label_exempt("upstream-monorepo:nofleet") is True
    assert rows.label_exempt("NOFLEET") is True
    assert rows.label_exempt("nofleetd-probe") is False
    assert rows.label_exempt("xnofleet") is False


# --------------------------------------------------------------------------- #
# 8. measured CPU work is evidence — the er3ab defect
# --------------------------------------------------------------------------- #
#
# Every signal `workload_evidence` read was GPU- or lane-shaped, so a dedicated
# CPU box produced no evidence at all and the safety net was free to park it
# mid-run. Live on 2026-08-21: box 48259065, label `er3ab-cvsb`, four hours into
# an rb3 A/B eval (mwcceppc compiles, no model endpoint, no GPU), returned None
# here. It survived only because an operator had explicitly watched it.
#
# NOT the co-tenant CPU farm (killed by owner ruling 2026-08-21, `0a9f1926`).
# That was a SIDECAR stealing cores from a GPU box; this is a DEDICATED box
# whose declared workload is CPU. `test_cpu_farm_default_off.py` still pins the
# farm off.

_ER3AB = {"label": "er3ab-cvsb"}          # matches no evidence label token


def _busy(cores: float, **ev: Any) -> dict[str, Any]:
    """A health row for a box past boot, with no jobd, burning `cores`."""
    return {"verdict": "OK",
            "evidence": {"boot_age_s": 15600, "is_jobs_box": False,
                         "jobd_hb_age_s": None, "cpu_util": cores,
                         "cpu_cores_effective": 256.0, **ev}}


def test_a_cpu_busy_box_with_an_off_lane_label_is_evidence() -> None:
    """THE regression. A box burning 20 cores is working, whatever its label
    says, and the safety net must not park it."""
    assert rows.workload_evidence(_ER3AB, _busy(19.98)) is not None


def test_an_idle_cpu_box_is_still_parkable() -> None:
    """The mirror: this must not become a blanket exemption that silently
    disables the safety net for every unlabelled box."""
    assert rows.workload_evidence(_ER3AB, _busy(0.0)) is None
    assert rows.workload_evidence(_ER3AB, _busy(0.05)) is None


def test_a_cpu_idle_serve_box_gets_no_cpu_evidence() -> None:
    """Grounded in measurement, not taste. Two live GPU-serving boxes whose
    CPUs were doing 0.4-0.56 cores of housekeeping reported cpu_util 1.31 and
    1.74 (2026-08-21, `calibration/`). Those must not read as CPU work, or the
    threshold exempts every box and the safety net stops working."""
    for idle_reading in (1.31, 1.74):
        got = rows.workload_evidence(_ER3AB, _busy(idle_reading))
        assert got is None, idle_reading


def test_absent_cpu_telemetry_says_nothing_and_never_means_idle() -> None:
    """`cpu_util` is POSITIVE EVIDENCE ONLY. vast does not always populate it,
    so a missing reading must fall through to the other signals rather than
    assert idleness — a null must never be louder than a label."""
    row = _busy(19.98)
    row["evidence"]["cpu_util"] = None
    assert rows.workload_evidence(_ER3AB, row) is None          # no other signal
    assert rows.workload_evidence({"label": "run:x"}, row) == "label 'run:x'"


def test_measured_cpu_work_outranks_a_label_but_not_a_zombie_verdict() -> None:
    """Placement in the order contract: ABOVE the label loop (a measurement
    beats a string) and BELOW the zombie check (the module's existing rule that
    a measured-dead workload is not rescued by softer evidence)."""
    for v in health.GuardVerdict:
        row = _busy(19.98)
        row["verdict"] = v.value
        got = rows.workload_evidence({"label": "run:x"}, row)
        if v.is_zombie:
            assert got is None, v
        elif v is health.GuardVerdict.BOOTING:
            assert got == "booting", v
        else:
            assert got is not None and "cpu" in got.lower(), v
