"""`vastlib.supervise.state` — the knot's key inventory, held true against source.

Why this file exists
--------------------
`state.py` ships no code. It ships *claims*: that these 170 keys are the whole
of what the supervise knot keeps in `st` / `jc` / `hf`, that eighteen of them
are a durable wire format, that five of them are tri-state floats which must
never become `0.0`, and that a handful are load-bearing precisely when ABSENT.

A documentation-only module rots the instant `main` advances, and nothing
notices. So every claim in `state.py` is re-derived here **from the source of
record** — `herdd.py`, `fleetd.py`, `bidpolicy.py`, `ladder_core.py` — on
every run. When a peer session adds `jc["new_thing"]`, the inventory test goes
red with the key name and the file:line that wrote it. That tripwire is the
deliverable; the TypedDicts are just where the answer is written down.

How it reads the source, and why not by import
----------------------------------------------
By `ast.parse` of the file text, never by importing `herdd`. Three reasons:

1. **Add-only phase.** `herdd.py` is not edited until plan step 6, and the
   flat constructors (`_init_state`, `_init_handoff_state`,
   `_init_job_handoff_state`, `job_supervise_init`) are ruled to `run_lane.py`
   / `job_lane.py`, not here. Importing them would make this file a
   direct-import test of somebody else's module.
2. **Nothing in this file may reach the network.** `herdd` import is
   side-effect-free today, but "today" is not a contract; parsing text cannot
   regress into an API call, an rclone shell-out or a live-market probe no
   matter what a future top-level statement does.
3. Parsing sees *writes that never execute in a unit test* — the mid-tick
   late-bound keys are the majority of the inventory and no fixture reaches
   them.

`fleetd` IS imported, in exactly one test, for the persistence round-trip:
re-implementing `_replacement_state_persist` / `_restore` here would be a
predicate validating itself, and the coercion asymmetry those two encode (a
`set` out as a sorted list and back; a str-keyed sidecar with no coercion at
all) is the whole point of the test. The import is cheap (`fleetd` does not
pull `herdd` at module scope) and touches no socket — `conftest.py`'s autouse
fixture redirects `FLEETD_SOCK` regardless.

What is deliberately NOT here
-----------------------------
* **No edit to `test_supervise.py`.** Its 426 `herdd.<attr>` references steer
  the still-live flat copies through step 5 (plan §8 add-only amendment). This
  file is the port-time coverage that replaces none of it.
* **No behavior assertions.** `state.py` has no behavior. Whether
  `handoff_poll` does the right thing with `min_running_eta_s is None` is
  `test_supervise.py`'s question; whether the key can still ARRIVE as None is
  this file's.
* **No fixture that constructs a "realistic" `st`.** A hand-built dict would
  encode this file's beliefs about the shape, and then the tests would be
  checking those beliefs against themselves.
"""

from __future__ import annotations

import ast
import functools
import pathlib

import pytest

from vastlib.fleet import state as fleet_state
from vastlib.supervise import state

HERE = pathlib.Path(__file__).resolve().parent

#: `herdd.fleet_watch_supervision` binds a LOCAL `st = json.load(
#: fleet_state_path())` — fleetd's `state.json`, an unrelated schema that
#: happens to reuse the name (state.py's hazard H5). It is the only such shadow
#: in the file; excluded by name so the inventory does not absorb
#: `watches` / `ceiling_by_box`.
SHADOWED_ST_FUNCS = ("fleet_watch_supervision",)

#: The one key the inventory is allowed to find undeclared: `herdd.py:8956`
#: reads `st.get("dph")`, which no writer ever sets on the run lane. Recorded in
#: `state.py` as FOUND-NOT-FIXED (hazard H4) and deliberately left out of
#: `RunState` so that annotating `run_lane.py` surfaces it as a type error.
H4_PHANTOM = ("RunState", "dph")


@functools.lru_cache(maxsize=None)
def _tree(name: str) -> ast.Module:
    return ast.parse((HERE / name).read_text(), filename=name)


@functools.lru_cache(maxsize=None)
def _funcs(name: str) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(_tree(name))
            if isinstance(n, ast.FunctionDef)}


def _first_dict_keys(node: ast.AST) -> list[str]:
    """Keys of the first all-constant dict literal inside `node`."""
    for n in ast.walk(node):
        if isinstance(n, ast.Dict) and n.keys and all(
                isinstance(k, ast.Constant) for k in n.keys):
            return [k.value for k in n.keys]      # type: ignore[union-attr]
    raise AssertionError("no constant-keyed dict literal found")


def _update_call_keys(fn: ast.AST, method: str = "update") -> list[str]:
    """Keys of the dict literal passed to `<name>.<method>({...})` inside `fn`."""
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == method):
            return _first_dict_keys(n)
    raise AssertionError(f"no .{method}({{...}}) call found")


def _key_refs(name: str, varmap: dict[str, str],
              skip_funcs: tuple[str, ...] = ()) -> dict[str, set[tuple[str, int]]]:
    """Every `<var>["k"]` / `.get("k")` / `.pop("k")` / `.setdefault("k")` /
    `.update(k=...)` in `name`, bucketed by the TypedDict `varmap` assigns the
    variable to. Values carry the line number so a failure names the writer."""
    tree = _tree(name)
    skip = [(n.lineno, n.end_lineno or n.lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in skip_funcs]
    out: dict[str, set[tuple[str, int]]] = {}

    def add(var: str, key: str, line: int) -> None:
        if any(a <= line <= b for a, b in skip):
            return
        out.setdefault(varmap[var], set()).add((key, line))

    for n in ast.walk(tree):
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id in varmap and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)):
            add(n.value.id, n.slice.value, n.lineno)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name) and n.func.value.id in varmap:
            if n.func.attr in ("get", "pop", "setdefault") and n.args \
                    and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str):
                add(n.func.value.id, n.args[0].value, n.lineno)
            elif n.func.attr == "update":
                for kw in n.keywords:
                    if kw.arg:
                        add(n.func.value.id, kw.arg, n.lineno)
    return out


def _pop_sites(name: str, varmap: dict[str, str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for n in ast.walk(_tree(name)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "pop" and isinstance(n.func.value, ast.Name)
                and n.func.value.id in varmap and n.args
                and isinstance(n.args[0], ast.Constant)):
            out.setdefault(varmap[n.func.value.id], set()).add(n.args[0].value)
    return out


def _tuple_const(name: str, const: str) -> tuple[str, ...]:
    """A module-level `NAME = ("a", "b", ...)` assignment, read without import."""
    for n in ast.walk(_tree(name)):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == const for t in n.targets):
            assert isinstance(n.value, ast.Tuple), f"{const} is not a tuple literal"
            return tuple(e.value for e in n.value.elts
                         if isinstance(e, ast.Constant))
    raise AssertionError(f"{const} not found in {name}")


def _declared(typ: object) -> set[str]:
    return set(typ.__annotations__)          # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 1. the Zone-S factories — state.py DESCRIBES their output, never rebuilds it
# --------------------------------------------------------------------------- #

def test_run_state_declares_every_mk_poll_state_key() -> None:
    """`bidpolicy.mk_poll_state` builds 25 of `st`'s keys and lives in a SHIPPED
    flat leaf this branch may not touch (plan §3). If Zone S grows a key,
    `RunState` is silently incomplete — so the drift goes red here."""
    import bidpolicy
    assert set(bidpolicy.mk_poll_state()) <= _declared(state.RunState)


def test_handoff_snapshot_matches_mk_handoff_state_exactly() -> None:
    """`HandoffSnapshot` is not an approximation of the Zone-S factory's output —
    it IS that shape. Equality (not containment) so a key added on either side
    fails: an extra key here would be a shape this module invented."""
    import bidpolicy
    assert _declared(state.HandoffSnapshot) == set(bidpolicy.mk_handoff_state())


def test_handoff_snapshot_and_carry_differ_in_the_documented_directions() -> None:
    """Hazard H3, pinned as a number. The two `hf` shapes share a name at every
    call site and are NOT interchangeable; a single `HandoffState` over both
    would be wrong in both directions. 16 keys the carry lacks, 4 the snapshot
    lacks — measured, and re-measured here."""
    carry_declared = _declared(state.HandoffCarry) - {
        # the late-bound seven, which no constructor emits
        "drain_ts", "prefence_bid", "primary_shape", "understudy_gone",
        "defer_sig", "refuse_sig", "pct_warned",
    }
    snap = _declared(state.HandoffSnapshot)
    assert len(snap - carry_declared) == 16
    assert sorted(carry_declared - snap) == [
        "chosen_offer", "cutover_ts", "epoch",
        # the dwell clock lives on the CARRY and is assigned onto the snapshot
        # rather than declared on it — `HandoffSnapshot` is the Zone-S factory's
        # frozen 35-key shape and stays equal to it
        "over_ceiling_since", "stall_alarmed"]


# --------------------------------------------------------------------------- #
# 2. the four constructors — ruled to run_lane/job_lane, described here
# --------------------------------------------------------------------------- #

def test_run_state_declares_every_init_state_key() -> None:
    import bidpolicy
    built = set(bidpolicy.mk_poll_state()) | set(
        _update_call_keys(_funcs("vastlib/supervise/run_lane.py")["_init_state"]))
    # 50 at rev 86840142; 54 since notify S2b added `notify_min_bid`,
    # `launch_dph_anchor`, `rebid_ceiling_mult` and `defense_cap` to the Zone-S
    # `mk_poll_state` factory (2026-08-16). The count is the tripwire, so it
    # moves with a stated reason or not at all.
    assert len(built) == 54, "the run-lane constructor's key count moved"
    assert built <= _declared(state.RunState)


def test_handoff_carry_matches_init_handoff_state_exactly() -> None:
    """The carry is a closed literal at construction; its 24 keys are declared
    here and nothing else is. (The 7 late-bound keys are marked as such in
    `state.py` and excluded from this comparison by name.)

    23 until 2026-08-26, when `over_ceiling_since` joined it: the handoff dwell
    became a DURATION, and a duration needs the run's start, not a tick count."""
    built = set(_first_dict_keys(
        _funcs("vastlib/supervise/handoff.py")["_init_handoff_state"]))
    late = {"drain_ts", "prefence_bid", "primary_shape", "understudy_gone",
            "defer_sig", "refuse_sig", "pct_warned"}
    assert len(built) == 24
    assert _declared(state.HandoffCarry) - late == built


def test_job_handoff_carry_adds_exactly_the_three_mirrored_keys() -> None:
    """LANE MIRRORING IS PINNED (plan §5 NOTE). `_init_job_handoff_state` is
    `_init_handoff_state()` plus three assignments; `JobHandoffCarry` inherits
    rather than merges. If the run lane ever grows one of these three — or the
    jobs lane grows a fourth — that is a lane unification, and it is an
    owner-called change, not a side effect of a refactor."""
    fn = _funcs("vastlib/supervise/handoff.py")["_init_job_handoff_state"]
    added = {n.slice.value for n in ast.walk(fn)
             if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
             and isinstance(n.slice, ast.Constant)}
    assert added == {"pending_jobs", "running_jobs", "retarget_incomplete"}
    extra = _declared(state.JobHandoffCarry) - _declared(state.HandoffCarry)
    assert extra == added


def test_job_context_declares_every_job_supervise_init_key() -> None:
    fn = _funcs("vastlib/supervise/job_lane.py")["job_supervise_init"]
    built = set()
    for n in ast.walk(fn):
        # AnnAssign as well as Assign since step 6d: the ported constructor
        # spells it `jc: dict[str, Any] = {...}`, and an Assign-only walker
        # found nothing and asserted 0 == 32 rather than a moved key count.
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, ast.Dict):
            built = {k.value for k in n.value.keys
                     if isinstance(k, ast.Constant)}
            break
    # 33 since 2026-08-17: `launch_env_pin`, the third immutable launch anchor
    # (after the price and disk ones) — the rehost lanes cannot read the eval-env
    # pin off an evicted box, so it lives on the watch.
    # 34 since 2026-08-18: `launch_cc_allow`, the fourth — the sm architecture
    # allowlist, which a replacement likewise cannot read off an evicted box.
    # 36 since 2026-08-24: `model_artifact` + `expect_ident`, the serve lane's
    # identity pin — WHAT the box is supposed to be serving. Not launch anchors:
    # they come off the WATCH POLICY (`fleet watch --artifact`), and both being
    # None is what makes the check a no-op for every pre-P3 serve watch.
    assert len(built) == 36, "the jobs-lane constructor's key count moved"
    assert built <= _declared(state.JobContext)


# --------------------------------------------------------------------------- #
# 3. THE INVENTORY REGRESSION — the reason this file exists
# --------------------------------------------------------------------------- #

def test_every_knot_key_in_the_tree_is_declared() -> None:
    """Walk `herdd.py`, `fleetd.py` and `ladder_core.py` for every subscript,
    `.get`, `.pop`, `.setdefault` and `.update(**kw)` against `st` / `jc` /
    `jctx` / `hf` / `ctx`, and require each key to be declared.

    Two exclusions, both documented in `state.py`:
      * `fleet_watch_supervision`'s local `st` is fleetd's `state.json` (H5);
      * `st["dph"]` is the phantom read (H4), left undeclared on purpose.

    `fleetd.py`'s own `st[...]` sites are that same `state.json` schema and are
    not scanned at all — only its `jc` / `hf` references are.
    """
    found: dict[str, set[tuple[str, int]]] = {}
    for name, varmap, skip in (
        # Step 6d: `herdd.py` is a launcher, so the knot it used to hold is
        # read from the five supervise modules the port put it in. Left on the
        # launcher this leg would contribute ZERO keys — the whole inventory
        # regression would pass while checking nothing.
        ("vastlib/supervise/run_lane.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "JobHandoffCarry"}, SHADOWED_ST_FUNCS),
        ("vastlib/supervise/job_lane.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "JobHandoffCarry"}, SHADOWED_ST_FUNCS),
        ("vastlib/supervise/handoff.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "JobHandoffCarry"}, SHADOWED_ST_FUNCS),
        ("vastlib/supervise/replacement.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "JobHandoffCarry"}, SHADOWED_ST_FUNCS),
        ("vastlib/supervise/retention.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "JobHandoffCarry"}, SHADOWED_ST_FUNCS),
        # Step 6d: same repoint as `test_popped_keys_...` below — the flat
        # `fleetd.py` is a launcher, so its `jc` / `hf` references are read from
        # `vastlib/fleet/daemon.py`. Left pointing at the launcher this leg
        # would contribute zero keys and pass VACUOUSLY.
        ("vastlib/fleet/daemon.py",
         {"jc": "JobContext", "hf": "JobHandoffCarry"}, ()),
        ("ladder_core.py", {"ctx": "RunState"}, ()),
        ("ladder_core.py", {"ctx": "JobContext"}, ()),
    ):
        for typ, refs in _key_refs(name, varmap, skip).items():
            found.setdefault(typ, set()).update(
                (k, line) for k, line in refs)

    undeclared: list[str] = []
    for typ, refs in sorted(found.items()):
        declared = _declared(getattr(state, typ))
        for key, line in sorted(refs):
            if key in declared or (typ, key) == H4_PHANTOM:
                continue
            undeclared.append(f"{typ}[{key!r}] (line {line})")
    assert not undeclared, (
        "keys reached in the knot but not declared in vastlib/supervise/"
        "state.py — add them WITH a writer file:line comment, or if one is a "
        "new phantom read, diagnose it rather than declaring it:\n  "
        + "\n  ".join(undeclared))


def test_run_state_does_not_declare_the_dph_phantom() -> None:
    """H4, FOUND-NOT-FIXED. `dph` is absent from `RunState` on purpose: when
    `run_lane.py` annotates its parameter, mypy reports the read as
    `typeddict-item` and the latent bug surfaces. Declaring the key to quiet
    that error would bury it again."""
    assert "dph" not in _declared(state.RunState)
    assert "dph" in _declared(state.JobContext), (
        "the jobs lane really does carry `dph`; that asymmetry is the bug")


def test_the_dph_phantom_read_is_still_present_in_herdd() -> None:
    """Tripwire on the finding itself: the arm call passes `st.get("dph")` —
    always None — as `_prefence_bid`'s second argument.

    Read from `vastlib/supervise/handoff.py` since step 6d (it was
    `herdd.py:8956`; the thinned launcher carries no bodies, so the old
    target would have made this tripwire unfireable). The defect travelled
    verbatim with the port, which is the point of a verbatim port.

    If this goes red, the defect was fixed (or the call moved). That is good
    news: update `state.py`'s FOUND-NOT-FIXED section and delete this test.
    """
    src = (HERE / "vastlib" / "supervise" / "handoff.py").read_text()
    assert '_prefence_bid(st.get("last_bid"), st.get("dph"))' in src


def test_fleet_state_json_keys_are_not_declared_here() -> None:
    """H5. `watches` / `ceiling_by_box` belong to fleetd's `state.json` schema,
    which will be `fleet/state.py`. A `RunState` that absorbed them would make
    the inventory a union of two unrelated shapes."""
    for key in ("watches", "ceiling_by_box", "strays", "spend_by_box"):
        assert key not in _declared(state.RunState)


# --------------------------------------------------------------------------- #
# 4. the durable projections — WIRE FORMAT
# --------------------------------------------------------------------------- #

def test_replacement_state_keys_mirror_fleetd() -> None:
    """A copy that can drift is worse than no copy, so it cannot.

    TWO targets since plan §8 step 6d, not three: `vastlib/fleet/state.py` is
    THE DEFINITION and `supervise/state.py` re-exports it (asserted by IDENTITY,
    so it cannot be a copy at all). `fleetd.py`'s live literal was the third
    until 6d thinned it to a launcher that re-exports the same object — nothing
    left to AST-parse, and an import-identity check there would only re-assert
    what the launcher's `from vastlib.fleet.state import …` already guarantees."""
    assert state.REPLACEMENT_STATE_KEYS is fleet_state.REPLACEMENT_STATE_KEYS
    assert fleet_state.REPLACEMENT_STATE_KEYS == _tuple_const(
        "vastlib/fleet/state.py", "REPLACEMENT_STATE_KEYS"), (
        "the definition must stay a plain module-level tuple literal — the AST "
        "helper above is the only thing that can read it without importing")


def test_run_state_keys_mirror_fleetd() -> None:
    assert state.RUN_STATE_KEYS is fleet_state.RUN_STATE_KEYS
    assert fleet_state.RUN_STATE_KEYS == _tuple_const(
        "vastlib/fleet/state.py", "RUN_STATE_KEYS")


def test_every_persisted_key_is_declared_on_its_dict() -> None:
    """A persisted key that no TypedDict declares is durable state nobody
    documented; a renamed one silently drops that state across a daemon
    restart (ladder_core.py:50-56 says so verbatim)."""
    for key in state.REPLACEMENT_STATE_KEYS:
        assert key in _declared(state.JobContext), key
    for key in state.RUN_STATE_KEYS:
        assert key in _declared(state.RunState), key


def test_replacement_state_round_trips_through_fleetd() -> None:
    """All 20 keys survive persist -> restore, including the two coercions:
    `evicted_machines` goes out as a sorted LIST (state.json is JSON) and comes
    back as a `set`; `evicted_machine_ts` keeps its stringified machine-id keys
    with no restore-side coercion at all.

    Driven through fleetd's real functions rather than a local re-implementation
    — a hand-rolled copy would be this test validating its own predicate.
    """
    import fleetd

    retained: state.RetainedBox = {
        "iid": "47694876", "status": "retained", "class": "spot_reclaim",
        "retained_ts": 1000.0, "deadline_ts": 11800.0, "retention_h": 3.0,
        "keep_labeled": True, "replacement_iid": "47700001",
    }
    jc: dict[str, object] = {
        "replacements": 2,
        "replacement_history": [{"from": "1", "to": "2"}],
        "launch_dph_anchor": 0.42,
        "launch_disk_gb": 110.0,
        # 2026-08-17: the eval-env pin, durable for the disk anchor's reason — a
        # replacement cannot read it off the evicted box, and an unpinned eval
        # box grades on eval-env/LATEST.
        "launch_env_pin": {"EVAL_ENV_VER": "20260816-1813-3c0a5f5b"},
        # 2026-08-18: the sm allowlist, durable for the same reason — a
        # replacement cannot read it off an evicted box, and a restart that
        # forgot it rehosts onto whatever architecture is cheapest.
        "launch_cc_allow": [80, 86, 89, 90],
        "evicted_machines": {31337, 42},
        "evicted_machine_ts": {"31337": {"ts": 900.0, "class": "spot_reclaim"}},
        "retained_boxes": [retained],
        "rebid_rungs": 3,
        "resume_tries": 1,
        "evicted_announced": "47694876",
        "evicted_class": "host_stop",
        "evicted_since": 940.0,
        # notify S2b (2026-08-16): the per-cycle latch and the dedup memory are
        # durable because a daemon restart mid-eviction must not re-price a
        # spent row; the rescue one-shot's two keys joined for the mirror-image
        # reason (a restart used to RE-ARM a spent rescue).
        "notify_matched": {"event_id": "e1", "iid": "47694876",
                           "your_bid": 0.45, "new_min_bid": 1.00},
        "notify_consumed_ids": ["e1"],
        "rescue_deadline": 1800.0,
        "rescue_put_failures": 1,
        "entry_floor": 0.31,
        "p_alt": 0.35,
        "p_alt_ts": 950.0,
        "bid_history": [[900.0, 0.42, 31337, 940.0]],
        # 2026-08-24 (REPLACEMENT_CEILING_WEDGE): the ceiling re-pricing state.
        # The observed-price pair is what lets a refusal act as the market read
        # it is; the streak pair is what the `replacement_wedged` alarm derives
        # from, and a restart that reset it would re-arm the wedge's silence.
        "replacement_market_floor": 0.40,
        "replacement_market_floor_ts": 950.0,
        "replacement_refusals": 7,
        "replacement_refusals_since": 900.0,
        "replacement_refusal_reason": "over_ceiling",
        "replacement_refusal_ceiling": 0.387,
        # NOT in the projection: proves the persist side is a projection, not a
        # dump. `a` is a live Namespace in the real dict and would not survive
        # json at all.
        "dry_run": True,
    }
    assert set(state.REPLACEMENT_STATE_KEYS) <= set(jc)

    w: dict[str, object] = {}
    fleetd._replacement_state_persist(jc, w)

    persisted = w["replacement"]
    assert isinstance(persisted, dict)
    assert set(persisted) == set(state.REPLACEMENT_STATE_KEYS)
    assert "dry_run" not in persisted
    assert persisted["evicted_machines"] == [42, 31337], "set -> sorted list"

    restored: dict[str, object] = {}
    fleetd._replacement_state_restore(restored, w)
    assert restored["evicted_machines"] == {31337, 42}, "list -> set"
    assert restored["evicted_machine_ts"] == jc["evicted_machine_ts"]
    for key in state.REPLACEMENT_STATE_KEYS:
        if key == "evicted_machines":
            continue
        assert restored[key] == jc[key], key
    assert restored["retained_boxes"][0]["class"] == "spot_reclaim"   # type: ignore[index]


def test_run_lane_projection_round_trips_through_fleetd() -> None:
    """The run lane's ENTIRE durable state is one key — and it is written by
    `ladder_core`, a module that never declares it. `_init_state` contributes
    nothing that survives a restart."""
    import fleetd

    st: dict[str, object] = {"bid_history": [[900.0, 0.42, 31337, 940.0]],
                             "last_bid": 0.42}
    w: dict[str, object] = {}
    fleetd._run_lane_state_persist(st, w)
    assert w["run_state"] == {"bid_history": st["bid_history"]}
    assert "last_bid" not in w["run_state"], (   # type: ignore[operator]
        "last_bid is deliberately NOT persisted — a restart re-derives it "
        "from dph_base, which cannot go stale (review 2026-08-10, F2)")
    restored: dict[str, object] = {}
    fleetd._run_lane_state_restore(restored, w)
    assert restored == {"bid_history": st["bid_history"]}


# --------------------------------------------------------------------------- #
# 5. tri-state floats — None means UNKNOWN, never 0.0 (defect #67)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("typename", sorted(state.TRISTATE_FLOAT_KEYS))
def test_tristate_keys_are_declared_optional(typename: str) -> None:
    """Every tri-state key's annotation must admit None. A TypedDict cannot
    carry a field default at all, which is exactly why it is the safe form
    here — but the annotation can still lie, and this is what stops it."""
    ann = getattr(state, typename).__annotations__
    for key in state.TRISTATE_FLOAT_KEYS[typename]:
        assert key in ann, f"{typename}[{key!r}] is not declared"
        assert "None" in str(ann[key]), (
            f"{typename}[{key!r}] must be float | None — None is UNKNOWN and "
            f"UNKNOWN refuses; collapsing it to 0.0 re-opens defect #67")


def test_the_build_state_read_default_asymmetry_survives() -> None:
    """`_handoff_job_build_state` reads `remaining_wall_h` with a 0.0 default and
    `min_running_eta_s` BARE, in adjacent lines. Both are correct for their
    field; "harmonising" them is the defect-#67 shape.

    Scanned in EVERY home the function currently has — `herdd.py` and any
    `vastlib/supervise/*.py` the knot port has landed it in — so the pin
    survives the port rather than expiring at it. Before plan §8 step 6d that
    caught a divergence between the flat copy and the ported one while both
    were live; the thinning left `herdd.py` with no `def` at all, so it
    simply drops out of `homes` — and stays in the candidate list on purpose,
    as a tripwire for a body reappearing there.
    """
    target = "_handoff_job_build_state"
    candidates = [HERE / "herdd.py",
                  *sorted((HERE / "vastlib" / "supervise").glob("*.py"))]
    homes = [p for p in candidates if p.exists() and target in p.read_text()]
    assert homes, f"{target} not found in any known home"

    checked = 0
    for path in homes:
        fns = [n for n in ast.walk(ast.parse(path.read_text()))
               if isinstance(n, ast.FunctionDef) and n.name == target]
        if not fns:
            continue          # a caller, not the definition
        checked += 1
        fn = fns[0]
        reads = {kw.arg: kw.value for n in ast.walk(fn)
                 if isinstance(n, ast.Call) for kw in n.keywords if kw.arg}

        eta = reads["min_running_eta_s"]
        assert isinstance(eta, ast.Call) and len(eta.args) == 1, (
            f"{path.name}: min_running_eta_s must be read BARE — a second "
            f".get() argument would turn 'unknown' into a number")

        wall = reads["remaining_wall_h"]
        assert isinstance(wall, ast.Call) and len(wall.args) == 2, (
            f"{path.name}: remaining_wall_h is read with an explicit 0.0 "
            f"default; the asymmetry with min_running_eta_s is deliberate")

    assert checked, f"{target} is referenced but defined nowhere"


# --------------------------------------------------------------------------- #
# 6. absence-as-state, and ladder_core's wire format
# --------------------------------------------------------------------------- #

def test_popped_keys_match_the_source_and_are_declared() -> None:
    """For every popped key, ABSENCE is a distinct load-bearing state — re-arm a
    fresh pull sampler, retract a standing refusal, drain an empty journal
    queue. This is the single hardest blocker on ever making these dataclasses,
    so the list is kept exact."""
    found: dict[str, set[str]] = {}
    for name, varmap in (
        # Step 6d: `herdd.py` is a launcher too, so its former `st`/`jc`/`hf`
        # pop sites are the three supervise modules that now hold them —
        # `job_lane` (the notify carry, the pull sampler, the rescue counter),
        # `replacement` (the eviction re-arm) and `handoff` (the refusal
        # retraction). Same reasoning as the fleetd leg below: a leg pointed at
        # a thinned launcher contributes nothing and passes VACUOUSLY.
        ("vastlib/supervise/job_lane.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "HandoffCarry"}),
        ("vastlib/supervise/replacement.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "HandoffCarry"}),
        ("vastlib/supervise/handoff.py",
         {"st": "RunState", "jc": "JobContext",
          "jctx": "JobContext", "hf": "HandoffCarry"}),
        # Step 6d: `fleetd.py` is a launcher now; its `jc.pop(...)` sites are
        # `vastlib/fleet/daemon.py`'s (the notify carry and the two journal
        # drains). Scanning the thinned launcher would find nothing and quietly
        # shrink this list instead of failing.
        ("vastlib/fleet/daemon.py", {"jc": "JobContext", "hf": "HandoffCarry"}),
    ):
        for typ, keys in _pop_sites(name, varmap).items():
            found.setdefault(typ, set()).update(keys)
    # ladder_core pops from whichever dict it was handed as `ctx` — both lanes.
    for typ in ("RunState", "JobContext"):
        found.setdefault(typ, set()).update(
            _pop_sites("ladder_core.py", {"ctx": typ}).get(typ, set()))

    assert {k: sorted(v) for k, v in sorted(found.items())} == {
        k: sorted(v) for k, v in sorted(state.POPPED_KEYS.items())}
    for typ, keys in state.POPPED_KEYS.items():
        declared = _declared(getattr(state, typ))
        assert set(keys) <= declared, f"{typ}: undeclared popped keys"


def test_ladder_core_ctx_keys_match_source_and_both_lanes_declare_them() -> None:
    """`ladder_core.py` MUTATES `st` and `jc` in place under the parameter name
    `ctx`, writing keys no constructor declares. Its docstring pins the
    constraint verbatim: "key names here are a wire format ... Renaming one
    silently drops durable state across a daemon restart." It is stdlib-only,
    sibling-imports Zone S, and is NOT ported in this step."""
    written = {n.slice.value for n in ast.walk(_tree("ladder_core.py"))
               if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
               and n.value.id == "ctx" and isinstance(n.ctx, ast.Store)
               and isinstance(n.slice, ast.Constant)}
    assert written == set(state.LADDER_CORE_CTX_KEYS)
    for key in state.LADDER_CORE_CTX_KEYS:
        assert key in _declared(state.RunState), key
        assert key in _declared(state.JobContext), key


# --------------------------------------------------------------------------- #
# 7. the module's own boundary
# --------------------------------------------------------------------------- #

def test_state_module_imports_nothing_at_runtime() -> None:
    """`state.py` is types and documentation. Its import list is `__future__` +
    `typing` + the ONE re-export edge — `vastlib.fleet.state`, which owns the
    two durable projections since plan §8 step 5 — and nothing else. That edge
    is same-ring (`supervise : fleet` are non-independent siblings) and it is
    why the Zone-S factories are still DESCRIBED here rather than re-exported:
    every additional import is another module a documentation file can drag in.

    `vastlib.fleet.state` is safe to import from here specifically because IT
    imports nothing from `vastlib` at module scope (its `fleet.client` lookup is
    function-local, precisely to keep this edge from closing a cycle)."""
    tree = ast.parse(
        (HERE / "vastlib" / "supervise" / "state.py").read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    assert imported == {"__future__", "typing", "vastlib.fleet"}

    fleet_tree = ast.parse((HERE / "vastlib" / "fleet" / "state.py").read_text())
    module_scope = {n.module or "" for n in fleet_tree.body
                    if isinstance(n, ast.ImportFrom)}
    module_scope |= {a.name for n in fleet_tree.body
                     if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith("vastlib") for m in module_scope), (
        "fleet/state.py must import no vastlib module at MODULE scope, or this "
        "re-export edge becomes an import cycle")


def test_state_module_defines_no_functions_or_classes_with_behavior() -> None:
    """A TypedDict is erased at runtime; these dicts stay plain dicts and every
    subscript / `.get` / `.pop` / `.update` site in the ported bodies is
    byte-identical. A `def` appearing here would be the first step back toward
    a second representation of the knot's state."""
    tree = ast.parse(
        (HERE / "vastlib" / "supervise" / "state.py").read_text())
    assert not [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            assert [b.id for b in node.bases if isinstance(b, ast.Name)] in (
                ["TypedDict"], ["HandoffCarry"]), node.name
