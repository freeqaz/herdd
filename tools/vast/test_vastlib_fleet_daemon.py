"""`vastlib.fleet.daemon` — the reconcile tick, the socket `Server`, `cmd_serve`.

What this file is for
---------------------
The port out of `fleetd.py` is textually verbatim everywhere it can be, so this
file does NOT re-test the policy that `test_fleetd.py`, `test_fleetd_notify.py`,
`test_notify_policy.py` and `test_standing_watch.py` already pin through
`fleetd` (they ran unedited against the flat copy until step 6d, and reach this
module through the launcher's re-exports now). It pins the five classes of claim
that the MOVE itself put at risk:

1. **The path anchors.** `repo_root()` is the one expression that could not be
   copied: `daemon.py` sits three levels below `tools/vast/`, and the flat file's
   `dirname(dirname(_HERE))` produces a *silently* wrong root from here. One
   `dirname` too few shipped on 2026-08-09 and nothing alarmed — it only
   relocated the generated unit's `WorkingDirectory=` and the `.env` the daemon
   hot-reloads. Only a comparison against the flat file's own resolution catches
   that, so that comparison is here (manifest H4).

2. **The S2b gate** (H11). `_notify_feed` pops THREE keys when the switch is off
   and deliberately keeps `notify_consumed_ids`. Popping only `notify_rows` was
   demonstrated twice to leave a latched box pricing its rescue off a matched row
   with the gate off — a real PUT at $1.212. The gate is tested MUTATION-STYLE:
   the weaker predicate is written out in the test and asserted to be
   *distinguishable* from the shipped one, so a port that regresses to it fails
   here rather than on a bill.

3. **The tick's ORDERING and its swallowed-exception boundaries** (H12/H16). The
   order notify -> destroys -> per-watch -> strays -> global budget -> `last_ok`
   -> alarm transitions -> save is policy, and every broad `except` in it is
   load-bearing: one bad watch, one failed health fold, one raising notify hook
   must each cost the tick nothing.

4. **The protocol surface**: `Server.handle` op-by-op, reading its version from
   the ONE collapsed constant `client.FLEET_PROTO_VERSION`.

5. **`cmd_serve`'s two footguns** (H9/H10): the single-instance lock is an open
   file handle held in a local that is never read again, and the shutdown path is
   a signal handler that SETS A FLAG so the `finally` (clean save +
   `fleetd_stopped`) actually runs.

Everything here is hermetic. No socket is bound, no `systemctl`/`git`/`rclone`
runs, and no vast API is touched: `daemon.subprocess` is patched as a MODULE
ATTRIBUTE (plan §8b — the call form is what makes the patch steer) and every I/O
touch goes through the injected `FakeHooks`, which is the seam the daemon was
designed around.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import notify                                          # noqa: E402
from vastlib.fleet import client, daemon, rows         # noqa: E402
from vastlib.fleet import state as fleet_state         # noqa: E402

DAEMON_SRC = Path(daemon.__file__).read_text()


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class FakeProc:
    """Enough of `subprocess.CompletedProcess` for `git_rev`'s one call."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeSubprocess:
    """Stands in for `daemon.subprocess`. Records argv; never execs anything."""

    def __init__(self, stdout: str = "deadbeef\n"):
        self.calls: list[list[str]] = []
        self.stdout = stdout

    def run(self, argv, **kw):                          # noqa: ANN001, ANN003
        self.calls.append(list(argv))
        return FakeProc(0, self.stdout, "")

    def call(self, argv, **kw):                         # noqa: ANN001, ANN003
        self.calls.append(list(argv))
        return 0


class FakeHooks:
    """Scripted stand-in for every daemon I/O touch (the `FleetHooks` protocol).

    Deliberately the same shape as `test_fleetd.py`'s fake — the daemon's seam
    did not change in the port, and a differently-shaped double here would test
    a different daemon than the flat suite does.
    """

    def __init__(self, dry_run: bool = False):
        self.t = 1_700_000_000.0
        self.dry_run = dry_run
        self.boxes: dict[str, dict] = {}
        self.api_down = False
        self.parked: list[str] = []
        self.resumed: list[str] = []
        self.destroyed: list[str] = []
        self.kept: list[str] = []
        self.park_ok = True
        self.destroy_ok = True
        self.drained_map: dict[str, bool | None] = {}
        self.results_map: dict[str, bool | None] = {}
        self.health_map: dict[str, dict] = {}
        self.health_raises = False
        self.status_lines: dict[str, str | None] = {}
        self.jobs_result: str | None = None
        self.jobs_spend = 0.0
        self.jobs_iid: str | None = None
        self.jobs_ticks: list[str] = []
        self.run_result = None
        self.run_spend = 0.0
        self.run_ticks: list[str] = []
        self.finalized: list[tuple] = []
        # notify (S2a)
        self.notify_payload: object = {"notifications": []}
        self.notify_err: str | None = None
        self.notify_raises = False
        self.notify_calls = 0

    # clock -------------------------------------------------------------- #
    def now(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s

    # fleet -------------------------------------------------------------- #
    def box(self, iid, status: str = "running", dph: float = 0.5,
            label: str | None = None, **kw) -> dict:
        self.boxes[str(iid)] = dict(id=int(iid), actual_status=status,
                                    intended_status=kw.pop("intended", status),
                                    dph_total=dph, label=label, **kw)
        return self.boxes[str(iid)]

    def park_box(self, iid) -> None:
        b = self.boxes.get(str(iid))
        if b:
            b["actual_status"] = b["intended_status"] = "stopped"

    def instances(self):
        return None if self.api_down else list(self.boxes.values())

    def instance(self, iid):
        return self.boxes.get(str(iid))

    def notifications(self):
        self.notify_calls += 1
        if self.notify_raises:
            raise RuntimeError("inbox transport exploded")
        return self.notify_payload, self.notify_err

    def jobd_status_line(self, iid):
        return self.status_lines.get(str(iid))

    def health(self, instances):
        if self.health_raises:
            raise RuntimeError("health fold exploded")
        return dict(self.health_map)

    def park(self, iid):
        self.parked.append(str(iid))
        if self.park_ok:
            self.park_box(iid)
            return True, None
        return False, "park refused"

    def resume(self, iid):
        self.resumed.append(str(iid))
        return True, None

    def destroy(self, iid):
        self.destroyed.append(str(iid))
        if self.destroy_ok:
            self.boxes.pop(str(iid), None)
            return True, None
        return False, "destroy refused"

    def keep_label(self, iid, inst):
        label = (inst or {}).get("label") or ""
        if "keep" in [t.strip() for t in label.split(":")]:
            return False, label
        new_label = (label + ":keep") if label else "keep:fleetd-park"
        b = self.boxes.get(str(iid))
        if b is not None:
            b["label"] = new_label
        self.kept.append(str(iid))
        return True, new_label

    def drained(self, iid):
        return self.drained_map.get(str(iid))

    def results_present(self, iid):
        return self.results_map.get(str(iid))

    # the imported supervise ladders, faked ------------------------------- #
    def run_init(self, a):
        return ({"run_id": a.run_id, "spend_usd": 0.0, "instance_id": None,
                 "actual_status": "running", "relaunch_count": 0},
                {"phase": "IDLE"}, True)

    def run_tick(self, st, a, hf, handoff_on):
        self.run_ticks.append(st["run_id"])
        st["spend_usd"] = self.run_spend
        st["instance_id"] = st.get("instance_id") or 111
        return self.run_result

    def run_finalize(self, st, a, act, hf, handoff_on):
        self.finalized.append((st["run_id"], act.kind))

    def jobs_init(self, a):
        return ({"a": a, "iid": str(a.id), "spend_usd": 0.0,
                 "handoff_on": True}, {"phase": "IDLE"})

    def jobs_tick(self, jc, hf):
        self.jobs_ticks.append(jc["iid"])
        jc["spend_usd"] = self.jobs_spend
        if self.jobs_iid is not None:
            jc["iid"] = str(self.jobs_iid)
        return self.jobs_result


# --------------------------------------------------------------------------- #
# fixtures + helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_subprocess(monkeypatch):
    """`Fleet.__init__` calls `git_rev()`. Patch the MODULE ATTRIBUTE so the
    call form stays the thing under test and no `git` ever runs."""
    fake = FakeSubprocess()
    monkeypatch.setattr(daemon, "subprocess", fake)
    return fake


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FLEETD_GLOBAL_BUDGET_USD", "FLEETD_UNWATCHED_GRACE_S",
              "FLEETD_UNWATCHED_GRACE_EXPENSIVE_S", "FLEETD_EXPENSIVE_DPH",
              "FLEETD_PYHALF_CONFIRM_S", "FLEETD_NOTIFY", "FLEETD_NOTIFY_POLICY",
              "FLEETD_DRY_RUN"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def hooks() -> FakeHooks:
    return FakeHooks()


@pytest.fixture
def fleet(tmp_path, monkeypatch, hooks) -> daemon.Fleet:
    monkeypatch.setenv("FLEETD_STATE_DIR", str(tmp_path / "state"))
    return daemon.Fleet(str(tmp_path / "state"), hooks=hooks)


def journal(f: daemon.Fleet) -> list[dict]:
    if not os.path.exists(f.journal_path):
        return []
    with open(f.journal_path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def events(f: daemon.Fleet) -> list[str]:
    return [r["event"] for r in journal(f)]


def event(f: daemon.Fleet, name: str) -> list[dict]:
    return [r for r in journal(f) if r["event"] == name]


def observe(f: daemon.Fleet, seconds: float, step: float | None = None) -> None:
    """Drive real reconcile ticks across `seconds` of OBSERVED time (N7: grace
    clocks advance on successful observations, capped at MAX_OBS_DT_S)."""
    step = step or daemon.MAX_OBS_DT_S
    f.tick()
    left = seconds
    while left > 0:
        f.hooks.advance(min(step, left))            # type: ignore[attr-defined]
        f.tick()
        left -= min(step, left)


def arm_jobs(f: daemon.Fleet, iid: int = 47694876, budget: float = 5.0,
             standing: bool | None = None, label: str = "jobs:w8", **kw) -> dict:
    f.hooks.box(iid, label=label, **kw)             # type: ignore[attr-defined]
    return f.watch(str(iid), "jobs", budget_usd=budget,
                   policy={"id": iid, "budget": budget},
                   requester="tester", standing=standing)


# --------------------------------------------------------------------------- #
# 1. path anchors — the one expression that could not be copied (H4)
# --------------------------------------------------------------------------- #
def test_repo_root_matches_flat_fleetd_computation() -> None:
    """`repo_root()` must resolve to what `fleetd.py`'s own expression does.

    The flat file computes `dirname(dirname(_HERE))` with `_HERE` =
    `<repo>/tools/vast`. `daemon.py` is three levels deeper, so the port
    recomputed the depth — and getting it wrong raises NOTHING (2026-08-09: one
    `dirname` too few silently moved the unit's WorkingDirectory and the `.env`
    the daemon hot-reloads, and no alarm fired for weeks). This comparison is
    the only thing that catches it.
    """
    flat_here = os.path.dirname(os.path.abspath(str(VAST_DIR / "fleetd.py")))
    expected = os.path.dirname(os.path.dirname(flat_here))
    assert daemon.repo_root() == expected
    assert daemon._TOOLS_VAST_DIR == flat_here


def test_repo_root_contains_fleetd_script() -> None:
    """The property the deploy path actually needs: the root is a checkout whose
    `tools/vast/fleetd.py` exists, and `_FLEETD_SCRIPT` names that launcher (the
    unit's `ExecStart=`), never this package module."""
    launcher = os.path.join(daemon.repo_root(), "tools", "vast", "fleetd.py")
    assert os.path.isfile(launcher), f"repo_root() {daemon.repo_root()!r} has no launcher"
    assert daemon._FLEETD_SCRIPT == launcher
    assert not daemon._FLEETD_SCRIPT.endswith("daemon.py")


def test_git_rev_reads_the_repo_root_through_the_module_attribute(fleet, _no_subprocess):
    """`git_rev()` shells out through `daemon.subprocess` — the patchable module
    attribute — and hands it `repo_root()`, not this file's directory."""
    argv = [c for c in _no_subprocess.calls if "rev-parse" in c]
    assert argv, f"no git rev-parse call recorded: {_no_subprocess.calls}"
    assert argv[0][:3] == ["git", "-C", daemon.repo_root()]
    assert fleet.rev == "deadbeef"


# --------------------------------------------------------------------------- #
# 2. the S2b gate — ONE predicate, in ONE place (H11)
# --------------------------------------------------------------------------- #
def _weak_gate(jc: dict) -> None:
    """THE REGRESSION, written out. The first cut of the gate popped
    `notify_rows` alone; `notify_matched` is DURABLE state, so a box that
    latched a row while armed kept pricing its rescue off that row with the
    switch off — demonstrated twice, a real PUT at $1.212."""
    jc.pop("notify_rows", None)


def _latched_jc() -> dict:
    """A jobs cursor mid-eviction: fed rows, a LATCHED match, a quote already
    said, and the dedup memory of a spent row."""
    return {"notify_rows": [{"event_id": "e1"}],
            "notify_matched": {"event_id": "e1", "new_min_bid": 1.212},
            "notify_quote_said": True,
            "notify_consumed_ids": ["e0"]}


def test_notify_feed_off_pops_all_three_keys_and_keeps_consumed_ids(fleet):
    """Gate OFF (the shipped default) removes the feed, the LATCH and the quote
    flag — and deliberately keeps `notify_consumed_ids`, which is dedup MEMORY,
    not evidence: forgetting it across a gate flap would let a re-arm re-match
    and re-price a row an earlier cycle already spent."""
    jc = _latched_jc()
    fleet._notify_feed(jc)
    assert "notify_rows" not in jc
    assert "notify_matched" not in jc
    assert "notify_quote_said" not in jc
    assert jc["notify_consumed_ids"] == ["e0"]


def test_notify_feed_off_is_distinguishable_from_the_weak_gate(fleet):
    """MUTATION-STYLE (H11): run the regression and the shipped predicate over
    the same cursor and assert they DISAGREE. A port that collapses back to
    "pop notify_rows only" makes these two equal and fails here — on a test,
    not on a bill."""
    shipped, mutant = _latched_jc(), _latched_jc()
    fleet._notify_feed(shipped)
    _weak_gate(mutant)
    assert shipped != mutant, (
        "the shipped S2b gate is popping only notify_rows — a latched box will "
        "price its rescue off a matched row with the switch OFF")
    assert mutant["notify_matched"]["new_min_bid"] == 1.212   # what it leaves behind
    assert "notify_matched" not in shipped                    # what it must not


def test_notify_feed_on_hands_over_the_whole_outbid_lookaside(fleet, monkeypatch):
    """Gate ON: `notify_rows` is the WHOLE retained set, copied out of the
    notify subtree — not a per-box slice (the ladder may swap the box under us
    mid-tick, and a list filtered against the box we THOUGHT we were supervising
    is the shape of a row matched to the wrong instance)."""
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    kept = [{"event_id": "e1", "instance_id": 1}, {"event_id": "e2",
                                                   "instance_id": 2}]
    fleet.state["notify"] = {"outbid": kept}
    jc = _latched_jc()
    fleet._notify_feed(jc)
    assert jc["notify_rows"] == kept
    assert jc["notify_rows"] is not kept          # a copy: the ladder may mutate
    assert jc["notify_matched"]["new_min_bid"] == 1.212   # the latch SURVIVES on
    assert jc["notify_quote_said"] is True                # the armed path


def test_notify_feed_on_with_an_empty_lookaside_feeds_an_empty_list(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    jc: dict = {}
    fleet._notify_feed(jc)
    assert jc["notify_rows"] == []


def test_notify_policy_defaults_off_and_notify_defaults_on():
    """The default is the deploy gate written down: the daemon runs from a git
    checkout, so a merge plus any restart must not arm a money-path change the
    review has not seen."""
    assert daemon.notify_policy_enabled() is False
    assert daemon.notify_enabled() is True


def test_notify_off_also_disarms_the_policy_feed(fleet, monkeypatch):
    """The predicate is an AND: the S2a poll switch being off takes the S2b feed
    with it, even with FLEETD_NOTIFY_POLICY=1."""
    monkeypatch.setenv("FLEETD_NOTIFY", "0")
    monkeypatch.setenv("FLEETD_NOTIFY_POLICY", "1")
    fleet.state["notify"] = {"outbid": [{"event_id": "e1"}]}
    jc = _latched_jc()
    fleet._notify_feed(jc)
    assert "notify_rows" not in jc and "notify_matched" not in jc


def test_the_gate_has_exactly_one_call_site():
    """H11: one predicate, one place. Splitting it across call sites, or moving
    it into the ladder, is how "S2b off" stops being a fact about one line."""
    tree = ast.parse(DAEMON_SRC)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_notify_feed"]
    assert len(calls) == 1, f"{len(calls)} call sites of _notify_feed"
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_notify_feed"]
    assert len(defs) == 1
    stmts = defs[0].body
    if (isinstance(stmts[0], ast.Expr) and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)):
        stmts = stmts[1:]                     # the docstring EXPLAINS the key
    body = "\n".join(ast.unparse(s) for s in stmts)
    for key in ("notify_rows", "notify_matched", "notify_quote_said"):
        assert f"jc.pop('{key}', None)" in body, f"{key} is no longer popped"
    assert "notify_consumed_ids" not in body, (
        "notify_consumed_ids must NOT be touched by the gate — it is dedup "
        "memory, and keeping it can only ever refuse a match")


# --------------------------------------------------------------------------- #
# 3. the notification poll (S2a) — total by construction
# --------------------------------------------------------------------------- #
def _inbox(*rows: dict) -> dict:
    return {"notifications": list(rows)}


def _outbid_row(eid: str, iid: int = 47694876, created: float = 1_699_999_000.0,
                your_bid: float = 0.9, new_min: float = 1.212) -> dict:
    return {"event_id": eid, "notif_type": notify.OUTBID_TYPE,
            "created_at": created,
            "associated_id": {"instance_id": iid, "machine_id": 7,
                              "your_bid": your_bid, "new_min_bid": new_min}}


def test_tick_notify_journals_new_rows_and_retains_the_outbid_lookaside(fleet):
    fleet.hooks.notify_payload = _inbox(_outbid_row("e1"), _outbid_row("e2"))
    fleet.hooks.box(47694876)
    fleet.tick()
    seen = event(fleet, notify.SEEN_EVENT)
    assert len(seen) == 2
    ns = fleet.state["notify"]
    assert ns["seen_total"] == 2 and ns["rows_seen"] == 2
    assert ns["cursor"]["recent_ids"] == ["e1", "e2"]
    assert len(ns["outbid"]) == 2                 # the S2b lookaside is RETAINED
    # ...retention is not policy: the rows are kept with the gate OFF, and only
    # whether anything READS them is `notify_policy_enabled()`.
    assert daemon.notify_policy_enabled() is False
    fleet.tick()                                   # same payload, nothing new
    assert len(event(fleet, notify.SEEN_EVENT)) == 2


def test_tick_notify_swallows_a_raising_hook_and_costs_the_tick_nothing(fleet):
    """H12: the whole `_tick_notify` body is defensive. A hook that raises is a
    poll FAILURE, not a dead tick."""
    fleet.hooks.notify_raises = True
    fleet.hooks.box(47694876)
    fleet.tick()
    assert fleet.state["notify"]["poll_ok"] is False
    assert "RuntimeError" in fleet.state["notify"]["fail_error"]
    assert "tick_error" not in events(fleet)
    assert fleet.last_tick_ts == fleet.hooks.now()      # the tick still SUCCEEDED


def test_notify_health_journals_on_transition_only(fleet):
    """A retired endpoint answers 404 every tick forever; announcing one
    unchanging fact per tick is the 158-events-for-2-facts defect."""
    fleet.hooks.notify_payload = None
    fleet.hooks.notify_err = "404 Not Found"
    fleet.hooks.box(47694876)
    for _ in range(4):
        fleet.tick()
        fleet.hooks.advance(45)
    failing = event(fleet, notify.POLL_ERROR_EVENT)
    assert len(failing) == 1 and failing[0]["state"] == "failing"
    assert fleet.state["notify"]["consecutive_failures"] == 4
    fleet.hooks.notify_payload, fleet.hooks.notify_err = _inbox(), None
    fleet.tick()
    health = event(fleet, notify.POLL_ERROR_EVENT)
    assert [r["state"] for r in health] == ["failing", "ok"]
    assert fleet.state["notify"]["poll_ok"] is True


def test_notify_off_skips_the_poll_entirely(fleet, monkeypatch):
    monkeypatch.setenv("FLEETD_NOTIFY", "0")
    fleet.hooks.box(47694876)
    fleet.tick()
    assert fleet.hooks.notify_calls == 0
    assert notify.SEEN_EVENT not in events(fleet)


def test_an_unreadable_payload_is_a_failing_poll_not_a_recovered_one(fleet, monkeypatch):
    """`notify.poll` is total by construction, so a raise there is a BUG, not a
    bad row — and it must announce once, like any other failure, instead of
    flapping ok/failing every tick."""
    def _boom(payload, cursor=None):
        raise TypeError("payload shape we have never met")
    monkeypatch.setattr(daemon.notify, "poll", _boom)
    fleet.hooks.box(47694876)
    fleet.tick()
    assert fleet.state["notify"]["poll_ok"] is False
    assert "unreadable inbox payload" in fleet.state["notify"]["fail_error"]


# --------------------------------------------------------------------------- #
# 4. the tick — ORDER is policy (H16), and every broad `except` is load-bearing
# --------------------------------------------------------------------------- #
def _trace_tick(f: daemon.Fleet, monkeypatch) -> list[str]:
    """Record the order of the tick's phases without changing any of them."""
    seen: list[str] = []

    def wrap(name: str):
        orig = getattr(f, name)

        def spy(*a, **kw):
            seen.append(name)
            return orig(*a, **kw)
        monkeypatch.setattr(f, name, spy)

    for name in ("_tick_notify", "_tick_destroys", "_tick_watch", "_tick_strays",
                 "_tick_global_budget", "_journal_alarm_transitions", "save"):
        wrap(name)
    return seen


def test_tick_phase_order_is_the_policy_order(fleet, monkeypatch):
    """notify -> destroys -> per-watch -> strays -> global budget -> alarm
    transitions -> save. Moving the stray sweep before the per-watch pass would
    let it adopt a box a watch is about to claim; moving `last_ok_tick_ts`
    earlier would let the stray alarms fire off a reading this tick never made.
    """
    arm_jobs(fleet)
    seen = _trace_tick(fleet, monkeypatch)
    fleet.tick()
    assert seen == ["_tick_notify", "_tick_destroys", "_tick_watch",
                    "_tick_strays", "_tick_global_budget",
                    "_journal_alarm_transitions", "save"]
    assert fleet.state["meta"]["last_ok_tick_ts"] == fleet.hooks.now()


def test_an_unreadable_api_changes_nothing(fleet, monkeypatch):
    """N7: an unreadable API is NOT an empty fleet. Nothing advances — not the
    clocks, not `last_tick_ts` (which is the last SUCCESSFUL reconcile, so
    `tick_age_s` can never read fresh beside frozen numbers)."""
    arm_jobs(fleet)
    fleet.tick()
    first = fleet.last_tick_ts
    fleet.hooks.api_down = True
    fleet.hooks.advance(600)
    seen = _trace_tick(fleet, monkeypatch)
    fleet.tick()
    assert seen == []                              # not one phase ran
    assert fleet.last_tick_ts == first
    assert fleet.state["meta"]["api_unavailable_since"] == fleet.hooks.now()
    assert "api_unavailable" in events(fleet)
    fleet.hooks.api_down = False
    fleet.tick()
    assert "api_unavailable_since" not in fleet.state["meta"]


def test_one_bad_watch_never_stops_the_fleet(fleet, monkeypatch):
    """H12. The per-watch wrapper is the difference between one broken watch and
    a fleet nobody is reconciling."""
    arm_jobs(fleet, iid=47694876)
    arm_jobs(fleet, iid=47694877, label="jobs:w9")
    orig = fleet._tick_watch

    def boom(target, by_iid, now, obs_dt):
        if target == "47694876":
            raise RuntimeError("this watch is broken")
        return orig(target, by_iid, now, obs_dt)
    monkeypatch.setattr(fleet, "_tick_watch", boom)
    fleet.tick()
    err = event(fleet, "watch_error")
    assert len(err) == 1 and err[0]["target"] == "47694876"
    assert "RuntimeError" in err[0]["error"]
    assert fleet.state["watches"]["47694876"]["last_tick_error"].startswith("RuntimeError")
    assert fleet.hooks.jobs_ticks == ["47694877"]  # the other watch still ticked
    assert fleet.state["meta"]["last_ok_tick_ts"] == fleet.hooks.now()


def test_a_failed_health_fold_degrades_to_an_empty_map(fleet):
    """H12: `health()` is alarms-only, so its failure must cost the tick nothing
    — and must not leave a STALE map standing in for a reading we did not make."""
    fleet.hooks.box(47694876)
    fleet.hooks.health_map = {"47694876": {"verdict": "ok"}}
    fleet.tick()
    assert fleet._health == {"47694876": {"verdict": "ok"}}
    fleet.hooks.health_raises = True
    fleet.hooks.advance(daemon.HEALTH_EVERY_S)
    fleet.tick()
    assert fleet._health == {}
    assert "tick_error" not in events(fleet)


def test_tick_jitter_is_a_fraction_of_the_interval(fleet, monkeypatch):
    """+-7s is 16% of the 45s default and 47% of a 15s tick — at that width the
    jitter, not the interval, decides when an eviction is noticed. The 45s
    default must still come out bit-for-bit 7.0."""
    waits: list[float] = []
    stop = threading.Event()

    def one(_timeout):                              # one iteration per call
        waits.append(_timeout)
        stop.set()

    monkeypatch.setattr(fleet, "tick", lambda: None)
    monkeypatch.setattr(stop, "wait", one)
    monkeypatch.setattr(daemon.random, "uniform", lambda lo, hi: hi)   # max jitter
    for interval in (None, 45.0, 15.0):
        stop.clear()
        daemon._reconcile_loop(fleet, stop, interval=interval)
    assert waits == [52.0, 52.0, pytest.approx(17.333, abs=0.001)]
    assert daemon.TICK_JITTER_FRAC * daemon.TICK_S == daemon.TICK_JITTER_S


def test_reconcile_loop_swallows_a_raising_tick(fleet, monkeypatch):
    """H12: a tick must never kill the daemon; the loop journals `tick_error`
    and waits out its interval."""
    stop = threading.Event()

    def boom():
        stop.set()                                 # one iteration, then done
        raise RuntimeError("tick exploded")
    monkeypatch.setattr(fleet, "tick", boom)
    daemon._reconcile_loop(fleet, stop, interval=0.01)
    err = event(fleet, "tick_error")
    assert len(err) == 1 and "RuntimeError" in err[0]["error"]


def test_health_refresh_cadence_is_first_tick_then_every_n(fleet, monkeypatch):
    """`gather_fleet_health` does B2 reads, so it runs on tick 1 and every
    HEALTH_EVERY_S after — not every tick. A DURATION: at the historical 45s
    tick that is every 4th tick, and it stays 180s apart at any tick rate."""
    calls: list[float] = []
    fleet.hooks.box(47694876)
    orig = fleet.hooks.health
    monkeypatch.setattr(fleet.hooks, "health",
                        lambda inst: (calls.append(fleet.hooks.now()),
                                      orig(inst))[1])
    t0 = fleet.hooks.now()
    for _ in range(int(daemon.HEALTH_EVERY_S / 45 * 2) + 1):
        fleet.tick()
        fleet.hooks.advance(45)
    assert calls == [t0, t0 + daemon.HEALTH_EVERY_S,
                     t0 + 2 * daemon.HEALTH_EVERY_S]


def test_health_cadence_is_a_duration_not_a_tick_count(fleet, monkeypatch):
    """The prerequisite for shortening the tick: tripling the poll rate must
    not triple the B2 read rate."""
    calls: list[float] = []
    fleet.hooks.box(47694876)
    orig = fleet.hooks.health
    monkeypatch.setattr(fleet.hooks, "health",
                        lambda inst: (calls.append(fleet.hooks.now()),
                                      orig(inst))[1])
    t0 = fleet.hooks.now()
    for _ in range(int(daemon.HEALTH_EVERY_S / 15 * 2) + 1):      # 15s tick
        fleet.tick()
        fleet.hooks.advance(15)
    assert calls == [t0, t0 + daemon.HEALTH_EVERY_S,
                     t0 + 2 * daemon.HEALTH_EVERY_S]


# --------------------------------------------------------------------------- #
# 5. destroys — explicit only, confirmed, at most once (S3)
# --------------------------------------------------------------------------- #
def test_destroy_requires_explicit_yes(fleet):
    fleet.hooks.box(47694876)
    with pytest.raises(ValueError):
        fleet.request_destroy("47694876", "now")
    with pytest.raises(ValueError):
        fleet.request_destroy("47694876", "whenever", yes=True)
    assert fleet.hooks.destroyed == []


def test_destroy_now_executes_once_and_clears_the_request(fleet):
    fleet.hooks.box(47694876)
    fleet.request_destroy("47694876", "now", reason="done", requester="tester",
                          yes=True)
    fleet.tick()
    assert fleet.hooks.destroyed == ["47694876"]
    assert "47694876" not in fleet.state["destroys"]
    d = event(fleet, "destroyed")
    assert len(d) == 1 and d[0]["requester"] == "tester"
    fleet.tick()
    assert fleet.hooks.destroyed == ["47694876"]   # at most once


def test_a_deferred_destroy_needs_two_consecutive_ticks(fleet):
    """S3: `--when now` is the operator's own confirmation; a DEFERRED condition
    has to hold twice, because a one-tick reading is a race with the box."""
    fleet.hooks.box(47694876)
    fleet.request_destroy("47694876", "parked", requester="tester", yes=True)
    fleet.tick()
    assert fleet.hooks.destroyed == []             # still running: not parked
    fleet.hooks.park_box(47694876)
    fleet.tick()                                   # streak 1 of 2
    assert fleet.hooks.destroyed == []
    pend = event(fleet, "destroy_condition_pending")
    assert pend and pend[-1]["streak"] == 1
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.tick()                                   # streak 2, dwell met
    assert fleet.hooks.destroyed == ["47694876"]


def test_a_deferred_destroy_confirmation_is_a_dwell_not_a_tick_count(fleet):
    """The debounce is what makes a parked box safe to destroy, so it must not
    shrink with the tick rate: two fast observations are not confirmation."""
    fleet.hooks.box(47694876)
    fleet.hooks.park_box(47694876)
    fleet.request_destroy("47694876", "parked", requester="tester", yes=True)
    for _ in range(int(daemon.DESTROY_CONFIRM_S / 15)):       # 15s tick
        fleet.tick()
        assert fleet.hooks.destroyed == []
        fleet.hooks.advance(15)
    fleet.tick()
    assert fleet.hooks.destroyed == ["47694876"]


def test_a_broken_streak_resets_the_confirmation(fleet):
    fleet.hooks.box(47694876)
    fleet.request_destroy("47694876", "parked", requester="tester", yes=True)
    fleet.hooks.park_box(47694876)
    fleet.tick()
    assert fleet.state["destroys"]["47694876"]["cond_streak"] == 1
    fleet.hooks.boxes["47694876"]["actual_status"] = "running"   # it came back
    fleet.tick()
    assert fleet.state["destroys"]["47694876"]["cond_streak"] == 0
    assert "cond_since" not in fleet.state["destroys"]["47694876"]
    assert fleet.hooks.destroyed == []


def test_a_deferred_destroy_holds_when_results_are_missing(fleet):
    fleet.hooks.box(47694876)
    fleet.hooks.results_map["47694876"] = False
    fleet.request_destroy("47694876", "parked", requester="tester", yes=True)
    fleet.hooks.park_box(47694876)
    fleet.tick()
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.tick()
    assert fleet.hooks.destroyed == []
    assert "destroy_deferred_no_results" in events(fleet)
    assert "results missing on B2" in fleet.state["destroys"]["47694876"]["held_reason"]
    fleet.hooks.results_map["47694876"] = True     # published: it goes through
    fleet.tick()                                   # the hold reset the dwell
    fleet.hooks.advance(daemon.DESTROY_CONFIRM_S)
    fleet.tick()
    assert fleet.hooks.destroyed == ["47694876"]


def test_an_expired_destroy_is_dropped_and_LATCHES_an_alarm(fleet):
    """H15: the request is GONE, so no later tick can re-derive this. A destroy
    an operator asked for and never got must not scroll past once and vanish."""
    fleet.hooks.box(47694876)
    fleet.request_destroy("47694876", "drained", requester="tester", yes=True)
    fleet.hooks.advance(daemon.DESTROY_TTL_S + 1)
    fleet.tick()
    assert "47694876" not in fleet.state["destroys"]
    assert "destroy_expired" in events(fleet)
    latched = fleet.state["alarms"]["destroy:47694876:expired"]
    assert "EXPIRED unexecuted" in latched["msg"]
    assert any(r["key"] == "destroy:47694876:expired"
               for r in fleet.alarm_records(fleet.hooks.now()))


def test_a_destroy_for_a_box_that_is_already_gone_is_skipped(fleet):
    fleet.request_destroy("47694876", "now", requester="tester", yes=True)
    fleet.tick()
    assert fleet.hooks.destroyed == []
    assert event(fleet, "destroy_skipped")[0]["reason"] == "already gone"


def test_a_failed_destroy_is_retried_and_its_error_derives(fleet):
    """`executed` is set from the RESULT, so a failure retries; `last_error` is
    re-earned every tick (derived) rather than latched."""
    fleet.hooks.box(47694876)
    fleet.hooks.destroy_ok = False
    fleet.request_destroy("47694876", "now", requester="tester", yes=True)
    fleet.tick()
    assert fleet.state["destroys"]["47694876"]["executed"] is False
    assert fleet.state["destroys"]["47694876"]["last_error"] == "destroy refused"
    fleet.hooks.destroy_ok = True
    fleet.tick()
    assert fleet.hooks.destroyed == ["47694876", "47694876"]
    assert "47694876" not in fleet.state["destroys"]


# --------------------------------------------------------------------------- #
# 6. the stray sweep — EVIDENCE-GATED, and it PARKS, never destroys (B1/H17)
# --------------------------------------------------------------------------- #
def test_a_stray_with_workload_evidence_is_adopted_never_parked(fleet):
    """B1: a box that shows live work is ADOPTED (`bare`: observation + cap, no
    bid moves) and alarmed. `budget=None` on that path means "we are not naming
    a figure", never "no ceiling" — reading it the other way cost 121 boxes
    their cap, so the adoption must land a resolved ceiling."""
    fleet.hooks.box(47694876, label="jobs:w8")
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.hooks.parked == []
    assert fleet.hooks.destroyed == []
    w = fleet.state["watches"]["47694876"]
    assert (w["profile"], w["adopted"]) == ("bare", True)
    assert w["budget_usd"] is not None and w["ceiling_id"]
    assert w["ceiling_source"] in daemon.CEILING_SOURCES
    ad = event(fleet, "unwatched_adopted")
    assert ad and ad[0]["evidence"].startswith("label ")
    assert "47694876" not in fleet.state["strays"]


def test_a_stray_with_no_evidence_is_PARKED_after_its_grace_never_destroyed(fleet):
    """The safety net: park (keep-labelled first, B4), never destroy."""
    fleet.hooks.box(47694876, dph=0.5)             # cheap tier -> the long fuse
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.parked == ["47694876"]
    assert fleet.hooks.destroyed == []
    assert fleet.hooks.kept == ["47694876"]        # keep label BEFORE the park
    parked = event(fleet, "unwatched_parked")[0]
    assert parked["tier"] == "cheap" and parked["grace_s"] == daemon.UNWATCHED_GRACE_S
    assert fleet.state["strays"]["47694876"]["parked_ts"]


def test_the_expensive_tier_gets_the_short_fuse(fleet):
    fleet.hooks.box(47694876, dph=daemon.EXPENSIVE_DPH_USD + 1.0)
    observe(fleet, daemon.UNWATCHED_GRACE_EXPENSIVE_S)
    assert fleet.hooks.parked == ["47694876"]
    assert event(fleet, "unwatched_parked")[0]["tier"] == "expensive"


def test_an_auto_adopt_failure_puts_the_stray_record_BACK(fleet, monkeypatch):
    """H12: the adopt wrapper is broad AND it restores the record — dropping it
    would restart the grace clock at zero on every failure, so a box that can
    never be adopted would also never be parked."""
    fleet.hooks.box(47694876, label="jobs:w8")
    fleet.tick()
    fleet.hooks.advance(daemon.MAX_OBS_DT_S)
    fleet.state["watches"].pop("47694876", None)   # un-adopt it for the retry
    fleet.state["strays"]["47694876"] = {"first_seen_ts": fleet.hooks.now(),
                                         "observed_s": 111.0}
    monkeypatch.setattr(fleet, "watch",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("nope")))
    fleet.tick()
    assert event(fleet, "auto_adopt_failed")[-1]["error"].startswith("RuntimeError")
    s = fleet.state["strays"]["47694876"]
    assert s["adopt_error"] == "RuntimeError"
    assert s["observed_s"] >= 111.0                # the grace clock stayed real


def test_a_stray_record_for_a_box_that_left_the_listing_is_pruned(fleet):
    fleet.hooks.box(47694876)
    fleet.tick()
    assert "47694876" in fleet.state["strays"]
    fleet.hooks.boxes.pop("47694876")
    fleet.hooks.advance(rows.UNWATCHED_STALE_S + 1)
    fleet.tick()
    assert "47694876" not in fleet.state["strays"]
    assert "stray_record_pruned" in events(fleet)


def test_an_exempt_label_opts_a_box_out_of_the_sweep_entirely(fleet):
    fleet.hooks.box(47694876, label="nofleet:probe")
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.hooks.parked == []
    assert "47694876" not in fleet.state["strays"]
    assert "47694876" not in fleet.state["watches"]


def test_a_watched_box_is_never_a_stray(fleet):
    arm_jobs(fleet)
    observe(fleet, daemon.UNWATCHED_GRACE_S * 2)
    assert fleet.state["strays"] == {}
    assert fleet.hooks.parked == []


def test_the_stray_alarm_is_DERIVED_not_latched(fleet):
    """H15: `fleet watch`-ing the box clears the line on the very NEXT read, with
    no tick in between — which is only true because nothing was appended."""
    fleet.hooks.box(47694876)
    fleet.tick()
    keys = [r["key"] for r in fleet.alarm_records(fleet.hooks.now())]
    assert "stray:47694876" in keys
    assert "stray:47694876" not in fleet.state.get("alarms", {})
    fleet.watch("47694876", "bare", 5.0, requester="tester")
    keys = [r["key"] for r in fleet.alarm_records(fleet.hooks.now())]
    assert "stray:47694876" not in keys            # gone with no tick at all


# --------------------------------------------------------------------------- #
# 6b. the reap keep-token is GRADED (item 1, FLEET_REVIEW_2026-08-20) — the
#     token is permanent and bills allocated disk, so it is skipped only for a
#     box that demonstrably holds nothing, and never on an unknown.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("drained,results,stamp", [
    (True, True, False),                            # the only skip
    (True, None, True), (True, False, True),        # empty queue / unpublished
    (None, True, True), (None, None, True), (None, False, True),
    (False, True, True), (False, None, True), (False, False, True),
])
def test_keep_stamp_needed_is_fail_open_to_keep(drained, results, stamp):
    assert daemon.keep_stamp_needed(drained, results) is stamp


def test_a_safety_net_park_skips_the_token_when_the_box_holds_nothing(fleet):
    """Nobody promised this box back: no watch, no operator. Drained with its
    results on B2, it goes to the 2h reaper instead of billing disk forever."""
    fleet.hooks.box(47694876, dph=0.5, label="scratch")
    fleet.hooks.drained_map["47694876"] = True
    fleet.hooks.results_map["47694876"] = True
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.parked == ["47694876"]       # still parked, never destroyed
    assert fleet.hooks.kept == [] and fleet.hooks.destroyed == []
    skipped = event(fleet, "keep_label_skipped")
    assert len(skipped) == 1
    assert (skipped[0]["reason"], skipped[0]["drained"],
            skipped[0]["results_present"]) == ("unwatched_safety_net", True, True)
    assert "keep_label_stamped" not in events(fleet)


@pytest.mark.parametrize("drained,results", [(None, True), (True, None),
                                             (False, True), (True, False)])
def test_an_unknown_or_unpublished_safety_net_park_still_stamps(fleet, drained,
                                                                results):
    fleet.hooks.box(47694876, dph=0.5, label="scratch")
    fleet.hooks.drained_map["47694876"] = drained
    fleet.hooks.results_map["47694876"] = results
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.kept == ["47694876"]
    assert "keep_label_skipped" not in events(fleet)


def test_a_raising_queue_read_keeps_the_box(fleet, monkeypatch):
    """A B2 outage must not hand a box to the reaper."""
    monkeypatch.setattr(fleet.hooks, "drained",
                        lambda iid: (_ for _ in ()).throw(RuntimeError("B2 down")))
    fleet.hooks.box(47694876, dph=0.5, label="scratch")
    observe(fleet, daemon.UNWATCHED_GRACE_S)
    assert fleet.hooks.kept == ["47694876"]


def test_a_budget_park_stamps_even_when_the_box_holds_nothing(fleet):
    """B4 unchanged where it is load-bearing: a cap park is a resumability
    promise the operator is expected to collect on, so it is never graded."""
    arm_jobs(fleet, budget=0.01)
    fleet.hooks.drained_map["47694876"] = True
    fleet.hooks.results_map["47694876"] = True
    fleet.hooks.jobs_spend = 99.0
    fleet.tick()
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["state"] == "budget_parked"
    assert fleet.hooks.kept == ["47694876"]
    assert "keep_label_skipped" not in events(fleet)


def test_an_operator_requested_park_stamps_even_when_the_box_holds_nothing(fleet):
    arm_jobs(fleet, budget=5.0)
    fleet.hooks.drained_map["47694876"] = True
    fleet.hooks.results_map["47694876"] = True
    fleet.request_action("47694876", "park", reason="drained", requester="tester")
    fleet.tick()
    assert fleet.hooks.parked == ["47694876"]
    assert fleet.hooks.kept == ["47694876"]
    assert "keep_label_skipped" not in events(fleet)


# --------------------------------------------------------------------------- #
# 7. the STANDING jobs watch — drain keeps the watch, only a ticket re-arms it
# --------------------------------------------------------------------------- #
def _drain(f: daemon.Fleet, iid: int = 47694876, verdict: str = "drained") -> None:
    """One drain tick: the ladder parks the box and returns the drain verdict,
    exactly as the inline `job supervise` loop does."""
    f.hooks.jobs_result = verdict
    f.hooks.park_box(iid)
    f.tick()
    f.hooks.jobs_result = None


def test_a_standing_watch_survives_its_queue_draining(fleet):
    arm_jobs(fleet, standing=True)
    fleet.tick()
    _drain(fleet)
    w = fleet.state["watches"]["47694876"]
    assert w["standing_dormant"] is True and w["dormant"] is True
    assert "watch_finished" not in events(fleet)
    assert "jobs_watch_standing_drained" in events(fleet)
    assert fleet.state["watches"]["47694876"]["budget_usd"] == 5.0   # cap NOT reset


def test_only_a_TICKET_re_arms_a_standing_watch_never_mere_liveness(fleet):
    """Re-entering the ladder against an all-terminal queue drain-parks the box
    seconds later, so a live standing box with nothing pending stays quiet."""
    arm_jobs(fleet, standing=True)
    fleet.tick()
    _drain(fleet)
    fleet.hooks.boxes["47694876"]["actual_status"] = "running"       # resumed by hand
    fleet.hooks.drained_map["47694876"] = True                       # queue all terminal
    fleet.hooks.advance(60)
    fleet.tick()
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True
    assert "jobs_watch_standing_resumed" not in events(fleet)
    fleet.hooks.drained_map["47694876"] = False                      # a ticket lands
    fleet.hooks.advance(60)
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert (w["standing_dormant"], w["dormant"], w["state"]) == (False, False, "watched")
    assert "jobs_watch_standing_resumed" in events(fleet)


def test_an_unreadable_queue_holds_the_dormancy_and_journals_once(fleet):
    """N7: an unreadable queue is NOT evidence of work — and the claim is made
    once per episode, not once per tick."""
    arm_jobs(fleet, standing=True)
    fleet.tick()
    _drain(fleet)
    fleet.hooks.boxes["47694876"]["actual_status"] = "running"
    fleet.hooks.drained_map["47694876"] = None                       # unreadable
    for _ in range(4):
        fleet.hooks.advance(60)
        fleet.tick()
    assert len(event(fleet, "jobs_watch_standing_queue_unknown")) == 1
    assert fleet.state["watches"]["47694876"]["standing_dormant"] is True


def test_a_non_standing_drain_still_ends_the_watch(fleet):
    """The regression guard: without `--standing` nothing moved."""
    arm_jobs(fleet, standing=False)
    fleet.tick()
    _drain(fleet)
    assert "jobs_watch_standing_drained" not in events(fleet)
    fin = event(fleet, "watch_finished")
    assert len(fin) == 1 and fin[0]["verdict"] == "drained"


def test_standing_is_refused_on_a_profile_with_no_queue(fleet):
    fleet.hooks.box(47694876, label="serve:eval")
    with pytest.raises(ValueError) as e:
        fleet.watch("47694876", "serve", budget_usd=5.0, standing=True)
    assert "jobs" in str(e.value)


# --------------------------------------------------------------------------- #
# 8. alarms — LATCH vs DERIVE is a per-site decision with money behind it (H15)
# --------------------------------------------------------------------------- #
def test_a_failed_action_LATCHES_and_a_budget_park_DERIVES(fleet):
    """Both are visible in `fleet status`; only one is appended. The failed
    action consumed its evidence (`pending_action` was already taken), so
    nothing could re-derive it; the budget park is a pure function of
    `w['state']`, so it must go out the moment the condition does."""
    arm_jobs(fleet, budget=0.01)
    fleet.hooks.jobs_spend = 99.0
    fleet.tick()
    fleet.tick()
    w = fleet.state["watches"]["47694876"]
    assert w["state"] == "budget_parked"
    recs = {r["key"]: r for r in fleet.alarm_records(fleet.hooks.now())}
    assert recs["watch:47694876:budget"]["sticky"] is False
    assert not any("budget" in k for k in fleet.state.get("alarms", {})), (
        "the budget-park alarm must DERIVE from w['state'], never latch")
    fleet.state["watches"]["47694876"]["state"] = "watched"   # condition fixed
    assert "watch:47694876:budget" not in [
        r["key"] for r in fleet.alarm_records(fleet.hooks.now())]

    fleet.hooks.park_ok = False                    # now make an action FAIL
    fleet.hooks.box(47694877)
    fleet.watch("47694877", "bare", 5.0, requester="tester")
    fleet.request_action("47694877", "park", reason="by hand", requester="tester")
    fleet.tick()
    latched = [k for k in fleet.state["alarms"] if k.startswith("action:47694877")]
    assert latched, fleet.state["alarms"]
    assert any(r["key"] in latched and r["sticky"] is True
               for r in fleet.alarm_records(fleet.hooks.now()))


def test_acking_a_latched_alarm_clears_it(fleet):
    fleet.hooks.box(47694876)
    fleet.request_destroy("47694876", "drained", requester="tester", yes=True)
    fleet.hooks.advance(daemon.DESTROY_TTL_S + 1)
    fleet.tick()
    key = "destroy:47694876:expired"
    assert key in fleet.state["alarms"]
    fleet.ack_alarm(key, requester="tester")
    assert key not in fleet.state["alarms"]


# --------------------------------------------------------------------------- #
# 9. the journal is a CONSUMED schema (H14) — `fleet_report.py`, `fleet log`,
#    the dashboard all grep these names. Renaming one is a silent break.
# --------------------------------------------------------------------------- #
JOURNAL_EVENTS = {
    "api_unavailable", "watch_error", "tick", "tick_error", "tick_paused",
    "tick_suspended_global_budget", "destroy_expired", "destroy_skipped",
    "destroy_condition_pending", "destroy_deferred_no_results", "destroyed",
    "destroy_failed", "jobs_handoff_carryover", "watch_dormant", "watch_rearmed",
    "pause_expired", "resumed", "resume_failed", "keep_label_stamped", "parked",
    "park_failed", "pyhalf_broken_seen", "pyhalf_parked", "pyhalf_park_failed",
    "budget_parked", "budget_park_failed", "health_alarm", "health_alarm_cleared",
    "jobs_replaced", "jobs_box_retention", "jobs_queue_empty", "jobs_queue_filled",
    "jobs_rescue_stalled", "jobs_rescue_recovered", "jobs_watch_standing_drained",
    "jobs_watch_standing_resumed", "jobs_watch_standing_queue_unknown",
    "watch_init_failed", "watch_adopted", "spend_backfilled", "watch_finished",
    "stray_record_pruned", "unwatched", "unwatched_adopted", "unwatched_parked",
    "unwatched_park_failed", "auto_adopt_failed", "global_budget_breached",
    "fleetd_started", "fleetd_stopped",
}


def _journal_names(src: str) -> set[str]:
    """Every literal event name a `journal(...)` call in `src` can emit.

    The ok/failed pairs (`"destroyed" if ok else "destroy_failed"`) are one call
    with two names, so a plain first-argument read would miss half the schema.
    """
    out: set[str] = set()

    def add(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
        elif isinstance(node, ast.IfExp):
            add(node.body)
            add(node.orelse)

    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "journal" and n.args):
            add(n.args[0])
    return out


def test_every_consumed_journal_event_name_is_still_emitted():
    """The port must not rename one. Read straight out of the module's literal
    `journal(...)` calls, so a name that moved to an f-string or a constant
    shows up here as missing rather than as a silent consumer break."""
    missing = sorted(JOURNAL_EVENTS - _journal_names(DAEMON_SRC))
    assert not missing, f"journal events lost in the port: {missing}"


def test_the_notify_event_names_come_from_the_notify_module():
    """S2a's three names are `notify`'s constants, not literals here — the port
    must not have inlined them (the consumer greps the same constant)."""
    assert f"self.journal({'notify'}.POLL_ERROR_EVENT" in DAEMON_SRC \
        or "notify.POLL_ERROR_EVENT" in DAEMON_SRC
    assert "notify.GAP_EVENT" in DAEMON_SRC and "notify.SEEN_EVENT" in DAEMON_SRC


# The flat daemon's journal vocabulary, captured by this file's own
# `_journal_names` from `tools/vast/fleetd.py` at fa5ad61a — the last revision
# before step 6d thinned it to a launcher. It was EQUAL to `daemon.py`'s at
# capture time (verified, both 67 names), so freezing it here loses nothing and
# keeps the two-directional check alive past the flat file's retirement.
_FLAT_JOURNAL_VOCABULARY = {
    "alarm_cleared", "alarm_latched", "alarm_raised", "alarm_resolved",
    "api_unavailable", "auto_adopt_failed", "auto_adopt_refused",
    "budget_park_failed", "budget_parked", "ceiling_armed",
    "ceiling_box_bound", "ceiling_degraded", "destroy_condition_pending",
    "destroy_deferred_no_results", "destroy_expired", "destroy_failed",
    "destroy_requested", "destroy_skipped", "destroyed",
    "env_reload_failed", "env_reloaded", "fleetd_started", "fleetd_stopped",
    "global_budget_breached", "health_alarm", "health_alarm_cleared",
    "jobs_box_retention", "jobs_handoff_carryover", "jobs_queue_empty",
    "jobs_queue_filled", "jobs_replaced", "jobs_rescue_recovered",
    "jobs_rescue_stalled", "jobs_watch_standing_drained",
    "jobs_watch_standing_queue_unknown", "jobs_watch_standing_resumed",
    "keep_label_stamped", "park_failed", "parked", "pause_cleared",
    "pause_expired", "paused", "pyhalf_broken_seen", "pyhalf_park_failed",
    "pyhalf_parked", "resume_failed", "resumed", "spend_backfilled",
    "stray_record_pruned", "tick", "tick_error", "tick_paused",
    "tick_suspended_global_budget", "unwatched", "unwatched_adopted",
    "unwatched_park_failed", "unwatched_parked", "watch_adopted",
    "watch_auto_adopted", "watch_dormant", "watch_error", "watch_finished",
    "watch_init_failed", "watch_rearmed", "watch_redirected",
    "watch_registered", "watch_removed",
}

# Names added to the daemon SINCE that capture. Kept separate so the frozen set
# above stays what it says it is — a capture, not a running edit surface.
_EVENTS_ADDED_SINCE_FLAT = {
    "keep_label_skipped",           # graded reap keep-token, 2026-08-20
    "serve_identity_withdrawn",     # serve identity mismatch (P3), 2026-08-24
    "machine_ledger_updated",       # instance->machine written down, 2026-08-25
    "machine_ledger_write_failed",  # ...and never fatal when it cannot be
    "jobs_watch_standing_woken",    # a ticket was PLACED on a dormant standing
                                    # watch's box (`ticket_placed`), 2026-08-27
}


def test_the_journal_vocabulary_is_identical_to_the_flat_daemons():
    """Stronger than the pinned list above and cheaper to keep honest: the set
    of event names `daemon.py` can emit must EQUAL the flat daemon's, in both
    directions. A rename shows as a pair, an invention as a one-sided extra.

    Step 6d retired the flat file, so the right-hand side is the frozen capture
    above rather than a live parse. Consumers (`herdd fleet journal`, the
    dashboard) grep these strings; a new event is a deliberate act that adds a
    line to `_EVENTS_ADDED_SINCE_FLAT`, not a side effect of a refactor."""
    assert _journal_names(DAEMON_SRC) == (_FLAT_JOURNAL_VOCABULARY
                                          | _EVENTS_ADDED_SINCE_FLAT)


# --------------------------------------------------------------------------- #
# 10. `Server.handle` — the golden protocol, version-checked against the ONE
#     collapsed constant (the daemon and the CLI used to hold two literals)
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(fleet, tmp_path) -> daemon.Server:
    """A Server that never binds: `handle` is pure w.r.t. the socket, which is
    what makes the protocol golden-testable without a connection."""
    return daemon.Server(fleet, sock_path=str(tmp_path / "fleetd.sock"))


def test_the_wire_version_is_the_client_constant(server, fleet):
    ok, data, err = server.handle({"op": "ping", "v": client.FLEET_PROTO_VERSION})
    assert ok and data["version"] == client.FLEET_PROTO_VERSION
    assigned = {t.id for n in ast.parse(DAEMON_SRC).body
                if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Name)}
    assert "VERSION" not in assigned, (
        "the daemon must read ONE wire-version literal, "
        "client.FLEET_PROTO_VERSION — a second copy of a contract is how two "
        "copies of it silently diverge")
    ok, data, err = server.handle({"op": "ping", "v": client.FLEET_PROTO_VERSION + 1})
    assert (ok, data) == (False, None)
    assert err == f"protocol version {client.FLEET_PROTO_VERSION + 1} != " \
                  f"{client.FLEET_PROTO_VERSION}"
    ok, _, _ = server.handle({"op": "ping"})       # absent version = accepted
    assert ok


def test_handle_ping_and_status_shape(server, fleet):
    arm_jobs(fleet)
    fleet.tick()
    ok, data, _ = server.handle({"op": "ping"})
    assert ok and data["watches"] == 1 and data["pid"] == os.getpid()
    assert data["rev"] == fleet.rev and data["tick_age_s"] == 0.0
    ok, data, _ = server.handle({"op": "status"})
    assert ok and data["version"] == client.FLEET_PROTO_VERSION
    assert [r["target"] for r in data["rows"]] == ["47694876"]
    assert "alarm_records" in data and "ceilings" in data and "destroys" in data


def test_handle_covers_every_op_the_client_can_send(server, fleet):
    """Op-by-op, in one place: the CLI's whole vocabulary must answer."""
    fleet.hooks.box(47694876)
    assert server.handle({"op": "watch", "args": {"target": "47694876",
                                                  "profile": "bare",
                                                  "budget_usd": 5.0}})[0]
    assert server.handle({"op": "pause", "args": {"target": "47694876",
                                                  "seconds": 60}})[0]
    assert server.handle({"op": "park", "args": {"target": "47694876"}})[0]
    assert server.handle({"op": "resume", "args": {"target": "47694876"}})[0]
    assert server.handle({"op": "operator_intent",
                          "args": {"target": "47694876", "kind": "stop"}})[0]
    assert server.handle({"op": "spend", "args": {}})[0]
    assert server.handle({"op": "ack", "args": {"all": True}})[0]
    assert server.handle({"op": "tick"})[1] == {"ticked": True}
    assert server.handle({"op": "unwatch", "args": {"target": "47694876"}})[0]
    # last: `destroy` AUTO-UNWATCHES an actively-watched box (S4), so anything
    # addressed to that watch afterwards is legitimately a KeyError.
    assert server.handle({"op": "destroy", "args": {"target": "47694876",
                                                    "yes": True}})[0]


def test_handle_rejects_malformed_requests_without_raising(server):
    assert server.handle("not a dict") == (False, None, "malformed request")
    assert server.handle({"op": "ping", "args": 7}) == (False, None, "malformed args")
    ok, data, err = server.handle({"op": "nonsense"})
    assert (ok, data) == (False, None) and err == "unknown op 'nonsense'"


def test_handle_never_takes_the_daemon_down(server, fleet, monkeypatch):
    """H12: the catch-all is the difference between one bad request and a dead
    control plane. A KeyError/ValueError answers as itself; anything else is
    typed and returned, never propagated."""
    ok, _, err = server.handle({"op": "unwatch", "args": {"target": "nope"}})
    assert not ok and "no watch" in err
    ok, _, err = server.handle({"op": "destroy", "args": {"target": "1"}})
    assert not ok and "yes=True" in err            # ValueError -> its own message
    monkeypatch.setattr(fleet, "status",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    ok, data, err = server.handle({"op": "status"})
    assert (ok, data) == (False, None) and err == "RuntimeError: boom"


def test_bind_creates_an_owner_only_socket_and_close_removes_it(server):
    s = server.bind()
    try:
        assert os.path.exists(server.sock_path)
        assert s.family == socket.AF_UNIX
        assert oct(os.stat(server.sock_path).st_mode)[-3:] == "600"
    finally:
        server.close()
    assert not os.path.exists(server.sock_path)
    assert server.stop.is_set()
    s.close()


# --------------------------------------------------------------------------- #
# 11. `cmd_serve` — the single-instance lock (H10) and the flag-not-raise
#     shutdown that makes the `finally` run (H9)
# --------------------------------------------------------------------------- #
class _Args:
    def __init__(self, once: bool = True, interval: float | None = 0.01):
        self.once = once
        self.interval = interval


@pytest.fixture
def serve_env(tmp_path, monkeypatch):
    """Point every path `cmd_serve` resolves at tmp_path, and keep `load_env`
    off the real repo `.env` (H3: the live symlink is the MAIN checkout's)."""
    d = tmp_path / "state"
    monkeypatch.setenv("FLEETD_STATE_DIR", str(d))
    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "fleetd.sock"))
    monkeypatch.setattr(daemon.config, "load_env", lambda *a, **kw: None)
    return d


def test_cmd_serve_once_ticks_saves_and_closes(serve_env, monkeypatch):
    monkeypatch.setattr(daemon.hooks_mod, "Hooks", FakeHooks)
    assert daemon.cmd_serve(_Args(once=True)) == 0
    lines = [json.loads(ln) for ln in
             open(os.path.join(serve_env, fleet_state.JOURNAL_NAME))]
    assert lines[0]["event"] == "fleetd_started"
    assert lines[0]["version"] == client.FLEET_PROTO_VERSION
    assert os.path.isfile(os.path.join(serve_env, fleet_state.STATE_NAME))
    assert not os.path.exists(os.environ["FLEETD_SOCK"])     # server.close() ran


def test_cmd_serve_refuses_a_second_reconciler(serve_env, monkeypatch):
    """Two reconcilers fighting over one fleet is the worst bug available, so a
    held flock is a REFUSAL to start, not a warning."""
    monkeypatch.setattr(daemon.hooks_mod, "Hooks", FakeHooks)
    os.makedirs(serve_env, exist_ok=True)
    held = fleet_state.acquire_single_instance_lock(str(serve_env))
    assert held is not None
    try:
        with pytest.raises(SystemExit) as e:
            daemon.cmd_serve(_Args(once=True))
        assert "another fleetd already holds" in str(e.value)
        assert fleet_state.LOCK_NAME in str(e.value)
    finally:
        held.close()


def test_cmd_serve_keeps_the_lock_handle_bound_for_the_process_lifetime():
    """H10: `acquire_single_instance_lock` returns an OPEN FILE HANDLE and the
    flock lives exactly as long as the binding. `lock` is deliberately never
    read again — a linter or a refactor that drops the "unused" local closes the
    fd, releases the flock and admits a SECOND reconciler."""
    fn = next(n for n in ast.parse(DAEMON_SRC).body
              if isinstance(n, ast.FunctionDef) and n.name == "cmd_serve")
    bound = [n for n in ast.walk(fn)
             if isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id == "lock" for t in n.targets)]
    assert len(bound) == 1, "cmd_serve no longer binds the lock handle to a name"
    call = bound[0].value
    assert isinstance(call, ast.Call) and \
        ast.unparse(call.func) == "fleet_state.acquire_single_instance_lock"


def test_cmd_serve_shutdown_sets_a_flag_and_runs_the_finally(serve_env, monkeypatch):
    """H9: `finally` does NOT run on an unhandled SIGTERM. The handler SETS A
    FLAG rather than raising, which is the only reason the clean `fleet.save()`
    and the `fleetd_stopped` line survive every systemd restart — including
    every deploy.
    """
    monkeypatch.setattr(daemon.hooks_mod, "Hooks", FakeHooks)
    monkeypatch.setattr(daemon, "_reconcile_loop", lambda *a, **kw: None)
    handlers: dict[int, object] = {}

    class FakeSignal:
        SIGTERM = signal.SIGTERM
        SIGINT = signal.SIGINT

        @staticmethod
        def signal(sig, fn):                        # noqa: ANN001
            handlers[sig] = fn
            return None

    monkeypatch.setattr(daemon, "signal", FakeSignal)

    def fire() -> None:
        for _ in range(500):                        # bounded: 5s ceiling
            if signal.SIGTERM in handlers:
                handlers[signal.SIGTERM](signal.SIGTERM, None)   # type: ignore[operator]
                return
            time.sleep(0.01)
        raise AssertionError("cmd_serve never installed a SIGTERM handler")

    t = threading.Thread(target=fire, daemon=True)
    t.start()
    assert daemon.cmd_serve(_Args(once=False, interval=0.01)) == 0
    t.join(timeout=5)
    assert not t.is_alive()
    names = [json.loads(ln)["event"] for ln in
             open(os.path.join(serve_env, fleet_state.JOURNAL_NAME))]
    assert names[0] == "fleetd_started" and names[-1] == "fleetd_stopped"
    assert os.path.isfile(os.path.join(serve_env, fleet_state.STATE_NAME))
    assert not os.path.exists(os.environ["FLEETD_SOCK"])
    # and the handler itself returns rather than raising, on both signals
    assert handlers[signal.SIGTERM](signal.SIGTERM, None) is None    # type: ignore[operator]
    assert handlers[signal.SIGINT](signal.SIGINT, None) is None      # type: ignore[operator]


# --------------------------------------------------------------------------- #
# 12. the port's own invariants — call form, no bootstrap, boundaries intact
# --------------------------------------------------------------------------- #
def test_the_module_does_no_sys_path_work_and_has_no_main_guard():
    """Zone P rule (§3): the thin `tools/vast/fleetd.py` launcher owns the
    bootstrap. A `sys.path.insert` here, or a `__main__` guard, would make the
    package module a second entry point."""
    tree = ast.parse(DAEMON_SRC)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            assert ast.unparse(n.func) not in ("sys.path.insert", "sys.path.append")
    assert "__main__" not in DAEMON_SRC


def test_cross_module_calls_are_module_attribute_form():
    """§8b: every cross-module reference is `module.symbol`, so a test can steer
    it with `monkeypatch.setattr(module, 'symbol', ...)`. A `from x import y`
    binds a second name that no patch can reach — and the fleet trio
    (client/hooks, rows/state, daemon) call each other."""
    tree = ast.parse(DAEMON_SRC)
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            assert n.module in ("__future__", "typing", "vastlib.boxes",
                                "vastlib.core", "vastlib.fleet",
                                "vastlib.supervise"), ast.unparse(n)
            if n.module and n.module.startswith("vastlib"):
                for a in n.names:
                    assert a.name in ("acctfault", "api", "client", "config",
                                      "deploy", "health", "hooks", "journal",
                                      "machine_ledger", "models", "rows",
                                      "serve_ident", "state"), (
                        f"{a.name} is a SYMBOL import — the fleet trio calls "
                        f"each other and must stay patchable")


def test_the_swallowed_exception_boundaries_are_still_broad(fleet):
    """H12: strict typing tempts every one of these into a narrower `except`;
    narrow any and a transient API blip becomes a dead daemon. Counted
    structurally, per enclosing function, against the flat file."""
    def handlers(src: str, only: set[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}

        def walk(node: ast.AST, ctx: str) -> None:
            for c in ast.iter_child_nodes(node):
                if isinstance(c, (ast.FunctionDef, ast.ClassDef)):
                    walk(c, c.name)
                else:
                    if isinstance(c, ast.ExceptHandler) and ctx in only:
                        out.setdefault(ctx, []).append(
                            ast.unparse(c.type) if c.type else "BARE")
                    walk(c, ctx)
        walk(ast.parse(src), "")
        return out

    load_bearing = {"tick", "_reconcile_loop", "_tick_notify", "_tick_strays",
                    "_init_runtime", "handle"}
    # The flat file's shape, produced by this same `handlers()` over
    # `tools/vast/fleetd.py` at fa5ad61a — the last revision before step 6d
    # thinned it to a launcher. It was EQUAL to the port's at capture time, so
    # freezing it keeps the boundary pinned after the flat file is gone.
    flat = {
        "_init_runtime": ["Exception"],
        "_reconcile_loop": ["Exception"],
        "_tick_notify": ["Exception", "Exception"],
        "_tick_strays": ["Exception"],
        "handle": ["(KeyError, ValueError)", "Exception"],
        "tick": ["Exception", "Exception"],
    }
    new = handlers(DAEMON_SRC, load_bearing)
    assert new == flat, "a swallowed-exception boundary changed shape in the port"
    for site, types in new.items():
        assert all(t in ("Exception", "BARE", "socket.timeout", "OSError",
                         "(KeyError, ValueError)") for t in types), (site, types)


def test_main_dispatches_the_four_subcommands(monkeypatch):
    """H18: the launcher's contract is `serve|install-unit|deploy|status`, and
    `deploy` dispatches into `fleet.deploy` — `main` only routes."""
    seen: list[str] = []
    for name, mod in (("cmd_serve", daemon), ("cmd_install_unit", daemon),
                      ("cmd_status", daemon), ("cmd_deploy", daemon.deploy)):
        monkeypatch.setattr(mod, name,
                            lambda a, _n=name: (seen.append(_n), 0)[1])
    for argv in (["serve", "--once"], ["install-unit"], ["deploy"], ["status"]):
        assert daemon.main(argv) == 0
    assert seen == ["cmd_serve", "cmd_install_unit", "cmd_deploy", "cmd_status"]
    with pytest.raises(SystemExit):
        daemon.main([])                            # a subcommand is REQUIRED


def test_install_unit_bakes_the_LAUNCHER_not_this_package_module(tmp_path, monkeypatch,
                                                                _no_subprocess):
    """The unit's `ExecStart=` is a frozen contract (`tools/vast/fleetd.py`);
    baking `daemon.py` would exec a package module and crash-loop on
    RestartSec=5."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(daemon.deploy, "checkout_audit", lambda repo: [])
    monkeypatch.setattr(daemon.deploy, "render_unit",
                        lambda python, script, repo, dry_run=False:
                        f"ExecStart={python} {script} serve\nWorkingDirectory={repo}\n")

    class A:
        no_enable = True
        force = False
        dry_run = False

    assert daemon.cmd_install_unit(A()) == 0
    unit = (tmp_path / ".config" / "systemd" / "user" /
            client.FLEET_UNIT_NAME).read_text()
    assert unit.rstrip().endswith("serve") or "serve" in unit
    assert os.path.join("tools", "vast", "fleetd.py") in unit
    assert "daemon.py" not in unit
    assert f"WorkingDirectory={daemon.repo_root()}" in unit
    assert _no_subprocess.calls == [], "--no-enable must not reach systemctl"


def test_a_narrower_hooks_object_DISARMS_rather_than_raises(fleet):
    """The two defensive `getattr`s on the hooks seam are load-bearing: the flat
    suite's `FakeHooks` does not inherit from `Hooks`, so an older/narrower
    double must turn the notify poll and the pyhalf read OFF, not blow up a
    tick. (`Fleet` binds its hooks by duck-typed attribute access, never by
    isinstance.)"""
    class Narrow(FakeHooks):
        notifications = None                       # type: ignore[assignment]
        jobd_status_line = None                    # type: ignore[assignment]

    narrow = Narrow()
    narrow.box(47694876)
    fleet.hooks = narrow
    fleet.tick()
    assert "tick_error" not in events(fleet)
    assert "watch_error" not in events(fleet)
    assert notify.SEEN_EVENT not in events(fleet)
    assert "notify" not in fleet.state or "cursor" not in fleet.state["notify"]
    src = ast.unparse(ast.parse(DAEMON_SRC))
    assert "getattr(self.hooks, 'notifications', None)" in src
    assert "getattr(self.hooks, 'jobd_status_line', None)" in src


def test_the_default_socket_path_still_honors_the_conftest_guard(fleet, monkeypatch,
                                                                 tmp_path):
    """conftest's autouse fixture points `FLEETD_SOCK` at a path that cannot
    exist so no test can reach the LIVE daemon. That guard bites only while
    `Server.__init__` resolves its default through `client.fleet_sock_path()` —
    if the resolution moves, the guard silently stops biting."""
    monkeypatch.setenv("FLEETD_SOCK", str(tmp_path / "guarded.sock"))
    assert daemon.Server(fleet).sock_path == client.fleet_sock_path()
    assert daemon.Server(fleet).sock_path == str(tmp_path / "guarded.sock")
