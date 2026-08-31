"""Portable tests for runmeta.py — pure fold + fake in-memory runner.

Runs in the toolchain-free lane (`pytest -m "not integration"`): no rclone, no
B2, no network, no creds. Every fold invariant (I1–I7) + the clock-skew
precedence rule + legacy STATUS fallback is exercised here.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runmeta as rm  # noqa: E402


def ev(event, ts, actor="box:1", nonce=None, **fields):
    """Build a valid event dict. `ts` is any sortable string for these tests."""
    d = {"v": 1, "ts": ts, "actor": actor, "event": event, "run_id": "r",
         "nonce": nonce or (ts[-4:] + event[:2])}
    d.update(fields)
    return d


# T() timestamps: fixed-width sortable, mimicking now_ts() ordering.
def T(n):
    return f"20260709T0000{n:02d}000Z"


# --- I7 / ordering -----------------------------------------------------------
def test_fold_ordering():
    a = ev("launched", T(1), actor="cli:h", instance_id="i1", gpu="A100", dph=0.5)
    b = ev("running", T(2))
    c = ev("checkpoint", T(3), step=100)
    d = ev("checkpoint", T(4), step=200)
    v1 = rm.fold_events([a, b, c, d], live_iids={"i1"})
    v2 = rm.fold_events([d, b, a, c], live_iids={"i1"})   # shuffled
    assert v1 == v2
    assert v1["status"] == "running"
    assert v1["latest_step"] == 200
    assert v1["live"] is True and v1["display_status"] == "running"
    assert v1["gpu"] == "A100" and v1["instance_id"] == "i1"


# --- I3: terminal precedence is SEMANTIC, not last-filename-wins --------------
def test_terminal_precedence():
    done = ev("done", T(5))
    run_after = ev("running", T(9))            # sorts AFTER done (stale heartbeat)
    v = rm.fold_events([done, run_after])
    assert v["status"] == "done"               # sticky terminal, not resurrected
    failed = ev("failed", T(6), rc=1, reason="boom")
    v2 = rm.fold_events([done, failed, run_after])
    assert v2["status"] == "failed"            # failed beats done in one epoch
    assert v2["fail_reason"] == "boom" and v2["fail_rc"] == 1


# --- I2 / I6: duplicate terminals collapse; cost is max, never summed --------
def test_duplicate_terminal_and_cost_dedup():
    d1 = ev("done", T(5), actor="box:1")
    d2 = ev("done", T(6), actor="box:2")       # retry -> distinct nonce/key
    c1 = ev("cost", T(5), cost_usd=1.0)
    c2 = ev("cost", T(6), cost_usd=2.5)         # cumulative snapshot
    c2dup = ev("cost", T(6), actor="cli:h", cost_usd=2.5)  # retry of same snapshot
    v = rm.fold_events([d1, d2, c1, c2, c2dup])
    assert v["status"] == "done"                # two done -> one terminal
    assert v["cost_usd"] == 2.5                 # max, NOT 1+2.5+2.5=6.0


# --- I1: one bad object never breaks the fold --------------------------------
def test_unparseable_event_skipped():
    good = ev("running", T(2))
    bad_empty = ""
    bad_json = "{not valid json"
    bad_missing = {"v": 1, "ts": T(3), "actor": "x"}     # no event/run_id
    v = rm.fold_events([good, bad_empty, bad_json, bad_missing])
    assert v["parse_errors"] == 3
    assert v["n_events"] == 1
    assert v["status"] == "running"             # the good one still folds


# --- I4: event log wins; instance-gone + STATUS terminal => inferred terminal -
def test_instance_gone_inferred_terminal():
    # non-terminal event log, instance gone, STATUS says DONE -> inferred done
    v = rm.fold_events([ev("running", T(2), instance_id="i1")], live_iids=())
    fs = rm.final_status(v, status_marker="DONE 20260709T000010000Z",
                         instance_live=False)
    assert fs["terminal"] is True and fs["status"] == "done"
    # done event but STATUS still RUNNING -> event wins (done)
    v2 = rm.fold_events([ev("done", T(5))])
    fs2 = rm.final_status(v2, status_marker="RUNNING x", instance_live=False)
    assert fs2["terminal"] is True and fs2["status"] == "done"


# --- I5: liveness is from vast, never from event recency ---------------------
def test_final_status_none_marker_non_terminal():
    # regression: events-only run, box gone, no STATUS marker -> must NOT crash
    # (was IndexError on "".split(None,1)[0]); degrades to a non-terminal stopped.
    v = rm.fold_events([ev("running", T(2), instance_id="i1")], live_iids=())
    fs = rm.final_status(v, status_marker=None, instance_live=False)
    assert fs["terminal"] is False and fs["display"] == "stopped"
    # empty-string marker + live instance -> running, still no crash
    fs2 = rm.final_status(v, status_marker="", instance_live=True)
    assert fs2["terminal"] is False and fs2["display"] == "running"


def test_stale_running_not_live():
    evs = [ev("launched", T(1), instance_id="i1"), ev("running", T(2))]
    v_dead = rm.fold_events(evs, live_iids=())            # no live instance
    assert v_dead["live"] is False
    assert v_dead["display_status"] == "stopped"          # NOT "running"
    v_live = rm.fold_events(evs, live_iids={"i1"})
    assert v_live["display_status"] == "running"


# --- preempted (SPOT_DESIGN §3.3): display-only, `status`/CHURN untouched ----
def test_preempted_reads_evicted_not_stopped_or_running():
    # box trained then self-preempted; supervisor hasn't confirmed `evicted` yet
    # (heartbeats keep flowing but don't count as a supersede) -> not live, so
    # `status` stays "running" (heartbeat/preempted never reset the epoch), but
    # display must read "evicted", never the generic "stopped".
    evs = [
        ev("launched", T(1), instance_id="i1"),
        ev("running", T(2)),
        ev("preempted", T(3)),
        ev("heartbeat", T(4), actor="supervisor"),
    ]
    v = rm.fold_events(evs, live_iids=())
    assert v["status"] == "running"            # poll()'s emit_evicted gate: unaffected
    assert v["preempted_pending"] is True
    assert v["display_status"] == "evicted"    # NOT "stopped"
    fs = rm.final_status(v, status_marker="RUNNING x", instance_live=False)
    assert fs["terminal"] is False and fs["display"] == "evicted"  # NOT "failed"

    # once the supervisor's own `evicted` confirms it, preempted is superseded
    # but display stays "evicted" via `status` itself.
    v2 = rm.fold_events(evs + [ev("evicted", T(5), actor="supervisor")], live_iids=())
    assert v2["preempted_pending"] is False
    assert v2["status"] == "evicted"
    assert v2["display_status"] == "evicted"

    # a rescue (same box resumes -> a fresh `running`) supersedes preempted and
    # liveness wins: display is "running", not "evicted".
    v3 = rm.fold_events(evs + [ev("running", T(6), actor="box:1")], live_iids={"i1"})
    assert v3["preempted_pending"] is False
    assert v3["display_status"] == "running"


# --- relaunch epochs: only the latest epoch decides status -------------------
def test_relaunch_epoch():
    evs = [
        ev("launched", T(1), actor="cli:h", instance_id="i1"),
        ev("running", T(2)),
        ev("evicted", T(3), actor="supervisor", instance_id="i1"),
        ev("relaunched", T(4), actor="supervisor", instance_id="i2", offer_id="o2"),
        ev("running", T(5), actor="box:2"),
    ]
    v = rm.fold_events(evs, live_iids={"i2"})
    assert v["status"] == "running"            # latest epoch, not the eviction
    assert v["relaunch_count"] == 1
    assert v["instance_id"] == "i2"            # current box is the relaunch


# --- legacy STATUS fallback (runs with no event log) -------------------------
def test_legacy_status_fallback():
    assert rm.status_marker_to_view("DONE 20260709T0Z", "r")["status"] == "done"
    assert rm.status_marker_to_view("FAILED rc=1", "r")["status"] == "failed"
    running = rm.status_marker_to_view("RUNNING x", "r", live_iids=())
    assert running["status"] == "running"
    assert running["display_status"] == "stopped"          # not live -> stopped
    assert rm.status_marker_to_view(None, "r")["status"] == "launched"


# --- launched with no follow-on is never silently "running" ------------------
def test_launched_no_events_phantom():
    v = rm.fold_events([ev("launched", T(1), instance_id="i1")], live_iids=())
    assert v["status"] == "launched"
    assert v["display_status"] == "stopped"    # phantom, not "running"


# --- write path: emit_event via a fake runner (round-trips through the fold) --
def test_emit_event_roundtrip_fake_runner():
    store = {}

    def fake_runner(args, input=None):
        if args and args[0] == "rcat":
            store[args[1]] = input                # b2:bucket/key -> body
            return 0, "", ""
        return 1, "", "unexpected"

    a = rm.emit_event("myrun", "running", runner=fake_runner, bucket="bkt",
                      actor="box:7")
    b = rm.emit_event("myrun", "checkpoint", runner=fake_runner, bucket="bkt",
                      actor="box:7", step=300)
    assert a["_emitted"] and b["_emitted"]
    assert a["_key"] != b["_key"]                 # unique nonce keys, no collision
    assert a["_key"].startswith("runs/myrun/events/")
    bodies = list(store.values())
    v = rm.fold_events(bodies, live_iids={"7"})
    assert v["status"] == "running" and v["latest_step"] == 300


def test_run_id_validation():
    import pytest
    for bad in ["", "a b", "a/b", "run:x", "x" * 65]:
        with pytest.raises(rm.RunmetaError):
            rm.validate_run_id(bad)
    for ok in ["gemma4-prop-01", "run_1.2", "A" * 64]:
        assert rm.validate_run_id(ok) == ok


def test_now_ts_and_key_shape():
    ts = rm.now_ts()
    assert len(ts) == len("20260709T000000000Z") and ts.endswith("Z") and ":" not in ts
    e = rm.make_event("r", "running", "box:1")
    k = rm.event_key(e)
    assert k.endswith(".json") and "box_1" in k and e["nonce"] in k


def test_derive_status_stays_in_babysit_vocabulary():
    # non-terminal must NOT match babysit's DONE*/FAILED*/STAGED* globs
    for st in ("running", "launched", "evicted", "relaunched", "unknown"):
        s = rm.derive_status({"status": st, "last_event_ts": "20260709T0Z"})
        assert s.startswith("RUNNING")
    assert rm.derive_status({"status": "done", "ended_at": "T"}).startswith("DONE")
    assert rm.derive_status(
        {"status": "failed", "fail_reason": None, "ended_at": "T"}).startswith("FAILED")


# =============================================================================
# Field-population repairs (2026-08-01). Every case below is taken from a real
# run in the B2 log that read null on the dashboard before the fix.
# =============================================================================

# --- string numerics: `emit --field K=V` stringifies everything --------------
def test_string_numerics_are_coerced():
    # the shape the `launched` emitter has ALWAYS written (dph as a str) — an
    # isinstance-only check dropped it and dph was null on 100% of runs
    v = rm.fold_events([ev("launched", T(1), instance_id="i1", gpu="H100",
                           dph="0.25"),
                        ev("checkpoint", T(2), step="200")])
    assert v["dph"] == 0.25
    assert v["latest_step"] == 200 and isinstance(v["latest_step"], int)
    assert rm._num("nope") is None and rm._num(True) is None
    assert rm._num("nan") is None and rm._num("inf") is None


# --- dph/gpu survive a relaunch (relaunched carries neither) -----------------
def test_price_and_gpu_survive_relaunch():
    evs = [ev("launched", T(1), instance_id="i1", gpu="5090", offer_id="o1",
              dph="0.20"),
           ev("relaunched", T(2), instance_id="i2", bid_price=0.31)]
    v = rm.fold_events(evs)
    assert v["instance_id"] == "i2"        # current box is still the relaunch
    assert v["gpu"] == "5090" and v["offer_id"] == "o1"  # from the older launch
    assert v["dph"] == 0.31                # relaunched prices it as `bid_price`


# --- instance_id for runs that never emit `launched` -------------------------
def test_instance_id_from_non_launch_events():
    # workflow generate/score arms only ever write stopping/resumed
    v = rm.fold_events([ev("resumed", T(1), instance_id="i7"),
                        ev("stopping", T(2), instance_id="i7",
                           reason="operator_destroy")])
    assert v["instance_id"] == "i7"
    assert v["started_at"] == T(1)         # falls back to the first event
    assert v["ended_at"] == T(2)           # newest event is an explicit stop


# --- spend snapshots ride on heartbeats, not just `cost` ---------------------
def test_cost_folds_heartbeat_spend_and_stays_max():
    evs = [ev("launched", T(1), instance_id="i1"),
           ev("cost", T(2), cost_usd=0.51),
           ev("heartbeat", T(3), actor="supervisor", spent_usd=1.02),
           ev("heartbeat", T(4), actor="supervisor", spent_usd=1.02)]
    v = rm.fold_events(evs)
    assert v["cost_usd"] == 1.02           # newest snapshot, NOT 0.51, NOT summed
    assert v["cost_source"] == "event"


# --- derived cost is computed from real events AND marked as derived --------
def test_derived_cost_is_marked_and_never_overrides_events():
    # g4it-01's real shape: launched at a known dph, destroyed 2h later, no cost
    # or heartbeat event ever emitted.
    launched = {"v": 1, "ts": "20260711T110741031Z", "actor": "cli:h",
                "event": "launched", "run_id": "r", "nonce": "a",
                "instance_id": "44507288", "dph": "1.0"}
    stopping = {"v": 1, "ts": "20260711T130741031Z", "actor": "cli:h",
                "event": "stopping", "run_id": "r", "nonce": "b",
                "instance_id": "44507288", "reason": "operator_destroy"}
    v = rm.fold_events([launched, stopping])
    assert v["cost_usd"] == 2.0 and v["cost_source"] == "derived"
    # an event-sourced figure always wins, and derived never fires without a dph
    v2 = rm.fold_events([launched, stopping,
                         {**launched, "nonce": "c", "event": "cost",
                          "cost_usd": 1.7}])
    assert v2["cost_usd"] == 1.7 and v2["cost_source"] == "event"
    nodph = {k: x for k, x in launched.items() if k != "dph"}
    v3 = rm.fold_events([nodph, stopping])
    assert v3["cost_usd"] is None and v3["cost_source"] is None


# --- liveness across the int(vast API)/str(CLI emitter) id split ------------
def test_live_matches_across_id_types():
    # vast returns int instance ids; the CLI emitter writes them as str. A raw
    # `in` read False for every CLI-launched run and forced display "stopped".
    v = rm.fold_events([ev("launched", T(1), instance_id="44507288")],
                       live_iids={44507288})
    assert v["live"] is True and v["display_status"] == "running"
