"""`vastlib.fleet.state` — the frozen state document, held against a live capture.

Why this file exists
--------------------
`state.json` is the only thing that survives a `fleet deploy`, and a daemon that
fails to LOAD it does not error — it cold-re-adopts, which looks like a working
fleet while every spend cap resets to a provisional default. So the port of the
persistence half cannot be graded by "the unit tests pass": it is graded by
replaying a document the PRE-refactor daemon actually wrote.

That document is `testfixtures/fleetd_state_snapshot_2026-08-16/` (plan §8 step
0; live rev `589f84dd`, 69,577 bytes, 237 box ids). **Do not regenerate it from
the post-refactor writer** — the whole point is that it was written by the
daemon as it existed before the port. This file is its first consumer.

How the round-trip is graded, and the trap in it
------------------------------------------------
`load_state` MUTATES the document it reads: the capture has no `notify` section,
so loading it adds one, and `Store.save()` additionally rewrites
`meta.saved_ts`. A naive byte-compare after a load therefore fails, and the
tempting "fix" is to weaken the assertion to a subset compare — which would
stop the test from noticing a dropped key, the one failure mode it exists for.

So the comparison is TWO assertions, not one:

  1. **raw vs raw, byte-for-byte** — `json.dumps(json.loads(text), indent=1,
     sort_keys=True, default=str)` must reproduce the file exactly. This pins
     the serialization flags with nothing else in the way.
  2. **loaded vs raw, as an explicit ALLOWED-DELTA SET** — every differing leaf
     path is enumerated and the set must equal exactly `{"notify",
     "meta.saved_ts"}`. Not "is a subset of". A key that vanished shows up in
     that set as surely as a key that appeared.

The other direction — an OLD reader accepting a NEW writer's document — is the
deploy contract (rollback is a redeploy of the prior revision), and it is why
the allowed delta may only ever contain ADDITIVE keys.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import shutil
from typing import Any

import pytest

from vastlib.fleet import state as fstate

import fleetd
import herdd

HERE = pathlib.Path(__file__).resolve().parent
SNAPSHOT = HERE / "testfixtures" / "fleetd_state_snapshot_2026-08-16"
SNAPSHOT_STATE = SNAPSHOT / "state.json"
SNAPSHOT_JOURNAL = SNAPSHOT / "journal_tail_200.ndjsonl"

#: Cardinalities as captured (fixture README). Quoted here so a fixture that is
#: silently swapped fails loudly rather than passing a weaker test.
SNAPSHOT_SHAPE = {"alarms": 0, "ceiling_by_box": 54, "ceilings": 35,
                  "destroys": 0, "intents": 172, "meta": 2,
                  "spend_by_box": 231, "strays": 0, "watches": 1}
SNAPSHOT_WATCH = "47842494"

#: The ONLY differences a `load_state` + `Store.save()` cycle may introduce on
#: the captured document: the `notify` section the capture predates (additive,
#: which is the safe direction for a rollback), and the save stamp.
ALLOWED_LOAD_SAVE_DELTA = {"notify", "meta.saved_ts"}


def _differences(a: Any, b: Any, path: str = "") -> set[str]:
    """Every leaf path at which two JSON documents differ — added, removed or
    changed. A set of dotted paths, so an assertion can name an exact allowed
    delta instead of a direction."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: set[str] = set()
        for k in set(a) | set(b):
            sub = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.add(sub)
            else:
                out |= _differences(a[k], b[k], sub)
        return out
    return set() if a == b else {path}


# --------------------------------------------------------------------------- #
# 1. the frozen names (plan §4) — three independent copies that must agree
# --------------------------------------------------------------------------- #

def test_frozen_names_match_fleetd() -> None:
    """Four of these five are now the launcher's re-exports of THIS module
    (plan §8 step 6d), so what they assert is the binding: `fleetd.<name>` is
    the spelling ~30 test modules and every runbook use, and a second literal
    in `fleetd.py` would name a file nobody writes.

    `VERSION` is the exception and is NOT a re-export of `fstate.VERSION`:
    `fleetd.VERSION` is `client.FLEET_PROTO_VERSION`, the socket protocol
    version, pinned by ruling in the launcher's docstring. They are different
    numbers that happen to be equal, so this stays an `==` between two
    independently-declared 1s."""
    for name in ("STATE_NAME", "JOURNAL_NAME", "LOCK_NAME", "JOURNAL_MAX_BYTES"):
        assert getattr(fleetd, name) is getattr(fstate, name), name
    assert fstate.VERSION == fleetd.VERSION == 1


def test_frozen_names_match_herdds_independent_literals() -> None:
    """NOT tautological post-6d. `herdd.fleet_state_path` /
    `fleet_journal_path` re-export `vastlib.fleet.client`'s, and `client`
    declares its OWN `STATE_NAME` / `JOURNAL_NAME` literals — deliberately, so
    the submit-path reader does not import the daemon. Two live copies of two
    filenames: rename one and an external reader silently reads a file nobody
    writes — no error, just an empty view of the fleet."""
    assert os.path.basename(herdd.fleet_state_path()) == fstate.STATE_NAME
    assert os.path.basename(herdd.fleet_journal_path()) == fstate.JOURNAL_NAME


def test_iso_matches_fleetd_and_is_not_the_supervise_one() -> None:
    """`fleet.state.iso` RAISES on None; `supervise.journal._iso_z` answers
    None. Every caller here guards first, so merging them would turn a
    would-be crash into a silent null in a journal field."""
    from vastlib.supervise import journal as sup_journal

    assert fleetd.iso is fstate.iso        # was a three-input value comparison
    for ts in (0.0, 1786859649.2886715, 1_000_000.0):
        assert fstate.iso(ts).endswith("Z")
    assert fstate.iso(0.0) == "1970-01-01T00:00:00Z"
    assert sup_journal._iso_z(None) is None
    with pytest.raises(TypeError):
        fstate.iso(None)        # type: ignore[arg-type]


def test_state_version_is_written_not_just_defined() -> None:
    st = fstate.load_state("/nonexistent/does/not/exist/state.json")
    assert st["version"] == 1


# --------------------------------------------------------------------------- #
# 2. load_state — defaults, quarantine, and the isinstance RESETS
# --------------------------------------------------------------------------- #

def test_missing_file_is_a_fresh_install(tmp_path: pathlib.Path) -> None:
    st = fstate.load_state(str(tmp_path / "state.json"))
    for key in ("watches", "strays", "destroys", "intents", "spend_by_box",
                "meta", "alarms", "notify", "ceilings", "ceiling_by_box"):
        assert st[key] == {}, key
    assert not list(tmp_path.iterdir()), "a missing file must not create one"


def test_corrupt_state_is_quarantined_not_raised(tmp_path: pathlib.Path) -> None:
    """S5. A corrupt state file must never crash-loop the daemon — and the empty
    ledger it rebuilds means 'provisional default cap for every adoption', never
    'unlimited'."""
    p = tmp_path / "state.json"
    p.write_text("{not json at all")
    st = fstate.load_state(str(p))
    assert st["ceilings"] == {} and st["ceiling_by_box"] == {}
    quarantined = list(tmp_path.glob("state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert not p.exists()


def test_a_json_scalar_is_corrupt_too(tmp_path: pathlib.Path) -> None:
    """`json.load` succeeds on `7`; `isinstance(st, dict)` is what catches it."""
    p = tmp_path / "state.json"
    p.write_text("7")
    st = fstate.load_state(str(p))
    assert st["version"] == 1
    assert list(tmp_path.glob("state.json.corrupt-*"))


@pytest.mark.parametrize("key", ["notify", "ceilings", "ceiling_by_box"])
def test_non_dict_sections_are_replaced_never_merged(
        tmp_path: pathlib.Path, key: str) -> None:
    """A garbage ceiling ledger is RESET to empty, and empty means the
    conservative default — the fail-closed direction. `setdefault` would have
    kept the garbage."""
    p = tmp_path / "state.json"
    p.write_text(json.dumps({key: ["not", "a", "dict"], "version": 1}))
    st = fstate.load_state(str(p))
    assert st[key] == {}


def test_setdefault_sections_keep_existing_content(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"watches": {"a": {"iid": "1"}}, "meta": {"x": 1}}))
    st = fstate.load_state(str(p))
    assert st["watches"] == {"a": {"iid": "1"}}
    assert st["meta"] == {"x": 1}


# --------------------------------------------------------------------------- #
# 3. save_state / Store.save — the serialization flags are wire format
# --------------------------------------------------------------------------- #

def test_save_is_atomic_and_leaves_no_temp(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "state.json"
    fstate.save_state({"version": 1, "watches": {}}, str(p))
    assert p.exists()
    assert not (tmp_path / "state.json.tmp").exists()


def test_save_uses_indent1_sortkeys_and_default_str(tmp_path: pathlib.Path) -> None:
    """`default=str` is what stops one stray `set` from turning a save into a
    `TypeError` that loses the whole document; `indent=1` + `sort_keys` are why
    the captured snapshot can be byte-compared at all."""
    p = tmp_path / "state.json"
    fstate.save_state({"b": 2, "a": {"s": {1}}}, str(p))
    text = p.read_text()
    assert text.startswith('{\n "a": {\n  "s": "{1}"\n },\n "b": 2\n}')
    assert not text.endswith("\n"), "json.dump writes no trailing newline"


def test_store_save_stamps_saved_ts_from_the_injected_clock(
        tmp_path: pathlib.Path) -> None:
    store = fstate.Store(str(tmp_path), now=lambda: 1234.5)
    store.save()
    assert json.loads((tmp_path / "state.json").read_text())["meta"]["saved_ts"] == 1234.5


def test_store_round_trips_a_watch(tmp_path: pathlib.Path) -> None:
    """S2, restart loses nothing."""
    store = fstate.Store(str(tmp_path), now=lambda: 1000.0)
    store.state["watches"]["47"] = {"iid": "47", "budget_usd": 5.0,
                                    "spend_usd": 1.25}
    store.save()
    again = fstate.Store(str(tmp_path), now=lambda: 2000.0)
    assert again.state["watches"]["47"]["spend_usd"] == 1.25
    assert again.state["watches"]["47"]["budget_usd"] == 5.0


# --------------------------------------------------------------------------- #
# 4. THE LOAD-COMPAT TEST — the captured live document
# --------------------------------------------------------------------------- #

def test_snapshot_reserializes_byte_for_byte() -> None:
    """Raw vs raw: parse and re-emit with save()'s exact flags, nothing else in
    between. This is the assertion that pins the flags; the load-vs-raw one
    below is the assertion that pins the schema."""
    text = SNAPSHOT_STATE.read_text()
    assert json.dumps(json.loads(text), indent=1, sort_keys=True,
                      default=str) == text
    assert len(text.encode()) == 69577


def test_snapshot_loads_without_quarantine(tmp_path: pathlib.Path) -> None:
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    fstate.load_state(str(tmp_path / "state.json"))
    assert not list(tmp_path.glob("state.json.corrupt-*")), (
        "the live document must LOAD; a quarantine here is a cold re-adopt, "
        "which looks like a working fleet while every cap resets")


def test_snapshot_shape_survives_the_load(tmp_path: pathlib.Path) -> None:
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    st = fstate.load_state(str(tmp_path / "state.json"))
    for key, n in SNAPSHOT_SHAPE.items():
        assert len(st[key]) == n, key
    assert st["version"] == 1
    assert set(st) == set(SNAPSHOT_SHAPE) | {"version", "notify"}


def test_snapshot_load_save_delta_is_exactly_the_allowed_set(
        tmp_path: pathlib.Path) -> None:
    """H1. `load_state` adds the `notify` section the capture predates and
    `Store.save()` rewrites `meta.saved_ts`; EVERY other leaf must be identical.

    Enumerated as a set, deliberately, rather than asserted as a subset — a
    subset compare passes when a key DISAPPEARS, which is the failure this whole
    file exists to catch."""
    raw = json.loads(SNAPSHOT_STATE.read_text())
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")

    store = fstate.Store(str(tmp_path), now=lambda: 1786900000.0)
    assert store.state["meta"]["saved_ts"] == raw["meta"]["saved_ts"], (
        "load must not stamp; only save does")
    store.save()
    after = json.loads((tmp_path / "state.json").read_text())

    assert _differences(raw, after) == ALLOWED_LOAD_SAVE_DELTA
    assert after["meta"]["saved_ts"] == 1786900000.0
    assert after["notify"] == {}
    assert set(after) - set(raw) == {"notify"}, "additive only — rollback reads this"
    assert not set(raw) - set(after), "nothing may be dropped"


def test_snapshot_reproduces_byte_for_byte_modulo_the_allowed_delta(
        tmp_path: pathlib.Path) -> None:
    """The same claim as above, stated the other way: undo exactly the two
    permitted deltas and the writer's output is the captured file, byte for
    byte."""
    original = SNAPSHOT_STATE.read_text()
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    store = fstate.Store(str(tmp_path), now=lambda: 1786900000.0)
    store.save()

    doc = json.loads((tmp_path / "state.json").read_text())
    doc.pop("notify")
    doc["meta"]["saved_ts"] = json.loads(original)["meta"]["saved_ts"]
    out = tmp_path / "again.json"
    fstate.save_state(doc, str(out))
    assert out.read_text() == original


def test_the_old_reader_accepts_the_new_writers_document(
        tmp_path: pathlib.Path) -> None:
    """THE DEPLOY CONTRACT, stated directly. Rollback is a `fleet deploy` of the
    PRIOR revision, so the document this port writes has to be readable by the
    `fleetd` that is running right now — additive keys only.

    `Fleet._load` is invoked unbound against a stand-in that carries only
    `state_path`, deliberately: constructing a real `Fleet` would run
    `git_rev()` (a subprocess) and build a `Hooks`. What is under test is the
    OLD reader, not the old daemon."""
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    store = fstate.Store(str(tmp_path), now=lambda: 1786900000.0)
    store.save()

    class _StandIn:
        state_path = str(tmp_path / "state.json")

    old = fleetd.Fleet._load(_StandIn())                 # type: ignore[arg-type]
    assert not list(tmp_path.glob("state.json.corrupt-*")), (
        "the old reader quarantined the new writer's document — that is a "
        "rollback that cold-re-adopts the whole fleet")
    assert _differences(old, store.state) == set()
    for key, n in SNAPSHOT_SHAPE.items():
        assert len(old[key]) == n, key
    assert old["watches"][SNAPSHOT_WATCH]["replacement"]["bid_history"]


def test_snapshot_keeps_the_leaked_test_fixture_intent_id(
        tmp_path: pathlib.Path) -> None:
    """The `"9"` intent is a test-fixture id that leaked into the live daemon
    (conftest.py's module docstring). A loader that choked on a non-instance box
    id would be a real regression, so it is kept and asserted."""
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    st = fstate.load_state(str(tmp_path / "state.json"))
    assert st["intents"]["9"]["kind"] == "destroy"
    assert st["intents"]["9"]["reason"] == "guard_zombie_destroy"


def test_snapshot_watch_subdocuments_survive_verbatim(
        tmp_path: pathlib.Path) -> None:
    """The single live watch is the only entry carrying full `policy` and
    `replacement` sub-documents — including `replacement.bid_history`, the
    self-floor guard's echo window."""
    raw = json.loads(SNAPSHOT_STATE.read_text())["watches"][SNAPSHOT_WATCH]
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    st = fstate.load_state(str(tmp_path / "state.json"))
    w = st["watches"][SNAPSHOT_WATCH]
    assert w == raw
    assert w["profile"] == "jobs"
    assert w["policy"] == raw["policy"]
    assert w["replacement"] == raw["replacement"]
    assert w["replacement"]["bid_history"] == raw["replacement"]["bid_history"]


def test_snapshot_watch_replacement_is_a_PARTIAL_projection(
        tmp_path: pathlib.Path) -> None:
    """Key ABSENCE is load-bearing: the live record carries 12 of the 18 durable
    keys, and restoring it must leave the other six to `job_supervise_init`'s
    zeros rather than fabricating them."""
    shutil.copy(SNAPSHOT_STATE, tmp_path / "state.json")
    st = fstate.load_state(str(tmp_path / "state.json"))
    persisted = set(st["watches"][SNAPSHOT_WATCH]["replacement"])
    assert persisted < set(fstate.REPLACEMENT_STATE_KEYS)

    jc: dict[str, Any] = {}
    fstate._replacement_state_restore(jc, st["watches"][SNAPSHOT_WATCH])
    assert set(jc) == persisted
    for absent in set(fstate.REPLACEMENT_STATE_KEYS) - persisted:
        assert absent not in jc, absent


# --------------------------------------------------------------------------- #
# 5. the durable projections — parity with the flat implementation
# --------------------------------------------------------------------------- #

def _projection_fixture() -> dict[str, Any]:
    return {
        "replacements": 2,
        "replacement_history": [{"from": "1", "to": "2"}],
        "launch_dph_anchor": 0.42,
        "launch_disk_gb": 110.0,
        # 2026-08-17: the eval-env pin, durable for the disk anchor's reason.
        "launch_env_pin": {"EVAL_ENV_VER": "20260816-1813-3c0a5f5b"},
        # 2026-08-18: the sm allowlist, same reason again — the replacement lane
        # cannot read it off an evicted box. Seeded by `job_lane` (`[]` at watch
        # init, filled from the primary's LAUNCH_CC_ALLOW), so the projection
        # carries it in production and this fixture has to as well.
        "launch_cc_allow": [80, 86, 89, 90],
        "evicted_machines": {31337, 42},
        "evicted_machine_ts": {"31337": {"ts": 900.0, "class": "spot_reclaim"}},
        "retained_boxes": [{"iid": "47694876", "status": "retained"}],
        "rebid_rungs": 3,
        "resume_tries": 1,
        "evicted_announced": "47694876",
        "evicted_class": "host_stop",
        "evicted_since": 940.0,
        "notify_matched": {"event_id": "e1"},
        "notify_consumed_ids": ["e1"],
        "rescue_deadline": 1800.0,
        "rescue_put_failures": 1,
        "entry_floor": 0.31,
        "p_alt": 0.35,
        "p_alt_ts": 950.0,
        "bid_history": [[900.0, 0.42, 31337, 940.0]],
        # 2026-08-24: the replacement-ceiling re-pricing state. Durable for the
        # anchor's reason — a restart that forgot the streak re-arms the silence
        # the wedge alarm exists to end.
        "replacement_market_floor": 0.40,
        "replacement_market_floor_ts": 950.0,
        "replacement_refusals": 7,
        "replacement_refusals_since": 900.0,
        "replacement_refusal_reason": "over_ceiling",
        "replacement_refusal_ceiling": 0.387,
        "dry_run": True,          # NOT in the projection
    }


def test_replacement_projection_is_the_documented_key_set() -> None:
    """Was `…_matches_fleetd_exactly`: both implementations, same input, same
    two dicts, so the port was graded against the live code rather than this
    file's beliefs about it. Step 6d left one implementation —
    `fleetd._replacement_state_persist` IS this function — so the `mine ==
    theirs` arms are gone and the beliefs are stated outright. They are not
    guesses: they are the properties the deleted comparison was protecting
    (the key set, the omitted `dry_run`, the set->sorted-list coercion)."""
    jc = _projection_fixture()
    mine: dict[str, Any] = {}
    fstate._replacement_state_persist(jc, mine)
    assert set(mine["replacement"]) == set(fstate.REPLACEMENT_STATE_KEYS)
    assert "dry_run" not in mine["replacement"]
    assert mine["replacement"]["evicted_machines"] == [42, 31337], "set -> sorted list"

    a: dict[str, Any] = {}
    fstate._replacement_state_restore(a, mine)
    assert a["evicted_machines"] == {31337, 42}, "list -> set, the ONE coercion"
    assert a["evicted_machine_ts"] == jc["evicted_machine_ts"], "no coercion here"


def test_run_lane_projection_writes_only_bid_history() -> None:
    """Was `…_matches_fleetd_exactly`; one implementation since step 6d."""
    st = {"bid_history": [[900.0, 0.42, 31337, 940.0]], "last_bid": 0.42}
    mine: dict[str, Any] = {}
    fstate._run_lane_state_persist(st, mine)
    assert mine == {"run_state": {"bid_history": st["bid_history"]}}
    assert "last_bid" not in mine["run_state"]


def test_empty_projection_writes_nothing() -> None:
    """`if out:` — an untouched watch keeps its prior record instead of having
    it blanked."""
    w: dict[str, Any] = {"replacement": {"replacements": 9}}
    fstate._replacement_state_persist({}, w)
    assert w == {"replacement": {"replacements": 9}}
    r: dict[str, Any] = {"run_state": {"bid_history": []}}
    fstate._run_lane_state_persist({}, r)
    assert r == {"run_state": {"bid_history": []}}


@pytest.mark.parametrize("w", [None, "not a dict", 7, []])
def test_persist_side_tolerates_a_garbage_watch(w: object) -> None:
    """The PERSIST half is `isinstance(w, dict)`-guarded, so an old or garbage
    watch record can never raise inside a tick."""
    fstate._replacement_state_persist({"replacements": 1}, w)     # type: ignore[arg-type]
    fstate._run_lane_state_persist({"bid_history": []}, w)        # type: ignore[arg-type]


@pytest.mark.parametrize("w", [None, {}, {"replacement": None}])
def test_restore_side_is_a_no_op_on_an_empty_watch(w: object) -> None:
    """ASYMMETRY, ported as found and confirmed against `fleetd` while a second
    copy existed: the restore half guards `w or {}` and the RECORD's type, not
    `w`'s. `None` and an empty dict are no-ops; a non-dict `w` raises
    `AttributeError`. Pinned rather than fixed — plan §7.4 forbids expectation
    changes, and the daemon only ever hands it a watch dict. The `fleetd.`
    arms (same calls through the launcher's names) went at step 6d; the
    launcher re-exports these two functions, so they ran this code twice."""
    jc: dict[str, Any] = {}
    fstate._replacement_state_restore(jc, w)                      # type: ignore[arg-type]
    fstate._run_lane_state_restore(jc, w)                         # type: ignore[arg-type]
    assert jc == {}

    with pytest.raises(AttributeError):
        fstate._replacement_state_restore({}, 7)                  # type: ignore[arg-type]


def test_non_dict_replacement_record_is_ignored() -> None:
    jc: dict[str, Any] = {}
    fstate._replacement_state_restore(jc, {"replacement": ["nope"]})
    assert jc == {}


# --------------------------------------------------------------------------- #
# 6. the journal — a FROZEN record schema, replayed against the capture
# --------------------------------------------------------------------------- #

def test_journal_record_shape() -> None:
    rec = fstate.journal_record("tick", 1000.4567, iid=47, kept="x", dropped=None)
    assert rec == {"ts": 1000.457, "ts_iso": "1970-01-01T00:16:40Z",
                   "event": "tick", "iid": "47", "kept": "x"}
    assert "dropped" not in rec, "None-valued fields are DROPPED, not emitted"


def test_journal_record_omits_iid_when_absent() -> None:
    assert "iid" not in fstate.journal_record("tick", 1000.0)


def test_every_captured_journal_line_reconstructs(  ) -> None:
    """Replay: each of the 200 captured lines, fed back through
    `journal_record`, must reproduce itself exactly — same keys, same `ts_iso`
    derivation, same `iid` stringification.

    Also proves the None-drop contract held in production: not one captured
    record carries a null value."""
    seen: set[str] = set()
    lines = [ln for ln in SNAPSHOT_JOURNAL.read_text().splitlines() if ln.strip()]
    assert len(lines) == 200
    for line in lines:
        rec = json.loads(line)
        seen.add(rec["event"])
        assert not any(v is None for v in rec.values()), rec["event"]
        fields = {k: v for k, v in rec.items()
                  if k not in ("ts", "ts_iso", "event", "iid")}
        assert fstate.journal_record(
            rec["event"], rec["ts"], rec.get("iid"), **fields) == rec
    assert len(seen) == 29, sorted(seen)
    assert {"fleetd_started", "fleetd_stopped", "alarm_raised",
            "alarm_resolved", "tick"} <= seen


def test_store_journal_writes_appends_and_prints(tmp_path: pathlib.Path) -> None:
    """journald is the SECOND sink — the daemon's log IS the journal — so the
    line goes to stdout as well as to the file, and the record comes back to the
    caller."""
    store = fstate.Store(str(tmp_path), now=lambda: 1000.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec = store.journal("tick", iid=47, note="hello", nothing=None)
        store.journal("tick")
    assert rec["event"] == "tick" and rec["iid"] == "47"
    lines = (tmp_path / fstate.JOURNAL_NAME).read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == rec
    assert json.loads(buf.getvalue().splitlines()[0]) == rec
    assert "nothing" not in lines[0]


def test_journal_write_failure_never_stops_a_tick(tmp_path: pathlib.Path) -> None:
    """A journal that cannot be written is not a reason to stop reconciling a
    fleet. The line still reaches stdout, and the record still returns."""
    store = fstate.Store(str(tmp_path), now=lambda: 1000.0)
    store.journal_path = str(tmp_path / "no" / "such" / "dir" / "journal.ndjsonl")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rec = store.journal("tick", note="still emitted")
    assert rec["note"] == "still emitted"
    assert "still emitted" in buf.getvalue()


def test_journal_rotates_once_past_the_threshold(tmp_path: pathlib.Path) -> None:
    """Single generation: the previous `.1` is discarded, not chained."""
    p = tmp_path / fstate.JOURNAL_NAME
    p.write_bytes(b"x" * (fstate.JOURNAL_MAX_BYTES + 1))
    fstate.rotate_journal(str(p))
    assert (tmp_path / (fstate.JOURNAL_NAME + ".1")).exists()
    assert not p.exists()


def test_rotate_swallows_a_missing_journal(tmp_path: pathlib.Path) -> None:
    fstate.rotate_journal(str(tmp_path / "nope.ndjsonl"))    # must not raise


# --------------------------------------------------------------------------- #
# 7. the single-instance lock
# --------------------------------------------------------------------------- #

def test_second_daemon_is_refused(tmp_path: pathlib.Path) -> None:
    """Two reconcilers on one fleet is the worst possible bug."""
    first = fstate.acquire_single_instance_lock(str(tmp_path))
    assert first is not None
    try:
        assert fstate.acquire_single_instance_lock(str(tmp_path)) is None
    finally:
        first.close()
    second = fstate.acquire_single_instance_lock(str(tmp_path))
    assert second is not None, "closing the handle releases the flock"
    second.close()


def test_lock_file_name_and_pid(tmp_path: pathlib.Path) -> None:
    fh = fstate.acquire_single_instance_lock(str(tmp_path))
    assert fh is not None
    fh.close()
    p = tmp_path / fstate.LOCK_NAME
    assert p.exists() and p.read_text() == str(os.getpid())
