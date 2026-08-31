"""Portable tests for boxstate.py's jobs-v2 lane — pure parsing (parse_cancel,
parse_results_done), the gather_jobs2 probe against a fake in-memory B2, and
the VERDICT precedence between the legacy runs/<id>/ signals and jobs-v2's
jobs/<id>/ signals. Runs in the toolchain-free lane (`pytest -m "not
integration"`): no rclone, no B2, no network, no creds — boxstate.B2 is never
constructed here, only duck-typed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boxstate as bs  # noqa: E402


# --------------------------------------------------------------------------- #
# fake in-memory B2 — duck-types the .ok / .cat / .lsf / .stat surface
# gather_jobs2 (and gather_b2) actually call
# --------------------------------------------------------------------------- #
class FakeB2:
    def __init__(self, objects=None, ok=True):
        # objects: {key: bytes_or_str_content}. A "directory" prefix's presence
        # is derived from having at least one key under it (mirrors rclone lsf).
        self.objects = dict(objects or {})
        self.ok = ok
        self.reason = None if ok else "fake unreachable"

    def cat(self, path):
        if not self.ok:
            return False, None
        body = self.objects.get(path)
        if not body:
            return False, None
        return True, body

    def lsf(self, path):
        if not self.ok:
            return False, []
        prefix = path if path.endswith("/") else path + "/"
        names = []
        for k in self.objects:
            if k == path:
                names.append(os.path.basename(k))
            elif k.startswith(prefix):
                names.append(k[len(prefix):])
        return bool(names), sorted(names)

    def stat(self, path):
        if not self.ok:
            return False, None
        body = self.objects.get(path)
        if body is None:
            return False, None
        return True, len(body)


def _ev(ts, event, actor="box:1", **fields):
    import json as _json
    d = {"v": 1, "ts": ts, "actor": actor, "event": event, "job_id": "j1",
         "nonce": ts[-6:]}
    d.update(fields)
    return _json.dumps(d)


# --------------------------------------------------------------------------- #
# parse_results_done / parse_cancel — pure
# --------------------------------------------------------------------------- #
def test_parse_results_done_valid():
    d = bs.parse_results_done('{"rc": 0, "duration_s": 42, "n_results": 2}')
    assert d == {"rc": 0, "duration_s": 42, "n_results": 2,
                 "raw": {"rc": 0, "duration_s": 42, "n_results": 2}}


def test_parse_results_done_nonzero_rc():
    d = bs.parse_results_done('{"rc": 143, "duration_s": 7904, "n_results": 2}')
    assert d["rc"] == 143


def test_parse_results_done_absent_or_bad():
    assert bs.parse_results_done(None) is None
    assert bs.parse_results_done("") is None
    assert bs.parse_results_done("not json") is None


def test_parse_cancel_valid():
    body = ('{"v":1,"ts":"20260715T164524267Z","actor":"cli:example-rig",'
            '"reason":"operator: recipe switch"}')
    d = bs.parse_cancel(body)
    assert d["reason"] == "operator: recipe switch"
    assert d["actor"] == "cli:example-rig"
    assert d["age_s"] is not None and d["age_s"] > 0


def test_parse_cancel_non_json_still_a_presence_signal():
    d = bs.parse_cancel("not json but present")
    assert d is not None
    assert d["reason"] is None
    assert d["raw"] == "not json but present"


def test_parse_cancel_absent():
    assert bs.parse_cancel(None) is None
    assert bs.parse_cancel("") is None


# --------------------------------------------------------------------------- #
# gather_jobs2 — every field degrades independently
# --------------------------------------------------------------------------- #
def test_gather_jobs2_all_absent():
    b2 = FakeB2(objects={})
    out = bs.gather_jobs2(b2, "j1")
    assert out["reachable"] is True
    assert out["events_present"] is False
    assert out["results_present"] is False
    assert out["done"] is None
    assert out["cancel"] is None
    assert out["log_present"] is False
    assert bs.jobs2_has_signal(out) is False


def test_gather_jobs2_unreachable():
    b2 = FakeB2(ok=False)
    out = bs.gather_jobs2(b2, "j1")
    assert out["reachable"] is False
    assert bs.jobs2_has_signal(out) is False


def test_gather_jobs2_full_signal():
    objs = {
        "jobs/j1/events/20260715T120000000Z-box_1-aaa111.json":
            _ev("20260715T120000000Z", "started"),
        "jobs/j1/events/20260715T130000000Z-box_1-bbb222.json":
            _ev("20260715T130000000Z", "cancelled"),
        "jobs/j1/results/gens.jsonl": "data\n",
        "jobs/j1/results.DONE.json": '{"rc": 143, "duration_s": 60, "n_results": 1}',
        "jobs/j1/CANCEL": '{"v":1,"ts":"20260715T125959000Z","reason":"operator"}',
        "jobs/j1/log.txt": "hello world\n",
    }
    b2 = FakeB2(objects=objs)
    out = bs.gather_jobs2(b2, "j1")
    assert out["events_present"] is True
    assert out["events_total"] == 2
    assert [e["event"] for e in out["events"]] == ["started", "cancelled"]
    assert out["results_present"] is True
    assert out["results"] == ["gens.jsonl"]
    assert out["done"]["rc"] == 143
    assert out["cancel"]["reason"] == "operator"
    assert out["log_present"] is True
    assert out["log_size"] == len("hello world\n")
    assert bs.jobs2_has_signal(out) is True


# --------------------------------------------------------------------------- #
# VERDICT — jobs-v2 fills gaps the legacy runs/<id>/ signals leave open
# --------------------------------------------------------------------------- #
_EMPTY_B2 = {"reachable": True, "reason": None, "status": None,
             "boot_phases": [], "boot_phases_present": False,
             "onstart_tail": None, "onstart_present": False,
             "events_present": False, "events_total": 0, "events": []}


def _empty_b2(**overrides):
    d = dict(_EMPTY_B2)
    d.update(overrides)
    return d


def test_verdict_cancel_wins_box_gone():
    b2sec2 = bs.gather_jobs2(FakeB2(objects={
        "jobs/j1/CANCEL": '{"v":1,"ts":"20260715T125959000Z","reason":"operator: switch"}',
        "jobs/j1/results.DONE.json": '{"rc": 143, "duration_s": 60, "n_results": 1}',
    }), "j1")
    v = bs.verdict({"actual_status": "exited", "age_s": 100}, _empty_b2(), b2sec2)
    assert v.startswith("cancelled:")
    assert "operator: switch" in v


def test_verdict_done_clean_box_gone():
    b2sec2 = bs.gather_jobs2(FakeB2(objects={
        "jobs/j1/results.DONE.json": '{"rc": 0, "duration_s": 60, "n_results": 1}',
    }), "j1")
    v = bs.verdict({"actual_status": "exited", "age_s": 100}, _empty_b2(), b2sec2)
    assert v.startswith("finished:")
    assert "rc=0" in v


def test_verdict_done_nonzero_box_gone_is_failed():
    b2sec2 = bs.gather_jobs2(FakeB2(objects={
        "jobs/j1/results.DONE.json": '{"rc": 1, "duration_s": 60, "n_results": 0}',
    }), "j1")
    v = bs.verdict({"actual_status": "exited", "age_s": 100}, _empty_b2(), b2sec2)
    assert v.startswith("failed:")


def test_verdict_jobs2_progress_box_running_no_legacy_signal():
    # a fresh (just-now) event so it reads as non-stale regardless of wall clock
    import datetime
    ts_now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S") + "000Z"
    b2sec2 = bs.gather_jobs2(FakeB2(objects={
        "jobs/j1/events/" + ts_now + "-box_1-aaa111.json": _ev(ts_now, "heartbeat"),
    }), "j1")
    v = bs.verdict({"actual_status": "running", "age_s": 100}, _empty_b2(), b2sec2)
    assert v.startswith("running: jobs-v2 job")
    assert "heartbeat" in v


def test_verdict_legacy_status_wins_over_jobs2_when_both_present():
    """The legacy STATUS marker is more specific than jobs-v2's fallback event
    stream — it should keep deciding even when a jobs-v2 lane also has signal."""
    b2sec = _empty_b2(status={"word": "DONE", "phase": None, "ts": None,
                               "age_s": 30, "raw": "DONE"})
    b2sec2 = bs.gather_jobs2(FakeB2(objects={
        "jobs/j1/events/20260715T120000000Z-box_1-aaa111.json":
            _ev("20260715T120000000Z", "heartbeat"),
    }), "j1")
    v = bs.verdict(None, b2sec, b2sec2)
    assert v.startswith("finished: run DONE")


def test_verdict_no_signal_anywhere_is_unknown():
    v = bs.verdict(None, _empty_b2(), bs.gather_jobs2(FakeB2(objects={}), "j1"))
    assert v.startswith("UNKNOWN:")


# --------------------------------------------------------------------------- #
# jobs-box resolution — a jobs-v2 box carries NO run:<job-id> label
# (BOX_SATURATION_AUDIT_2026-07-30 §6 red flag 5: all three RUNNING jobs on the
# wave box reported "(no live vast instance carries label run:<job-id>)" and a
# verdict of "stopped:", because ONE jobd serves many job ids off
# jobs/queue/<IID>/ and its label names the operator's box, not the jobs.)
# --------------------------------------------------------------------------- #
def _now_ts():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S") + "000Z"


def _jobs2(objs):
    return bs.gather_jobs2(FakeB2(objects=objs), "j1")


def test_jobs_box_instance_id_from_the_events():
    ts = _now_ts()
    b2sec2 = _jobs2({
        f"jobs/j1/events/{ts}-cli_x-aaa000.json": _ev(ts, "submitted",
                                                      actor="cli:x", box="46245045"),
        f"jobs/j1/events/{ts}-box_1-aaa111.json": _ev(ts, "claimed",
                                                      instance_id="46245045"),
        f"jobs/j1/events/{ts}-box_1-aaa222.json": _ev(ts, "heartbeat",
                                                      instance_id="46245045"),
    })
    assert bs.jobs_box_instance_id(b2sec2) == "46245045"
    # nothing has claimed it yet -> no box, and we must not invent one
    assert bs.jobs_box_instance_id(_jobs2({
        f"jobs/j1/events/{ts}-cli_x-aaa000.json": _ev(ts, "submitted",
                                                      actor="cli:x")})) is None
    assert bs.jobs_box_instance_id(None) is None
    # fallback: a tail of event kinds that omit instance_id still names the box
    # through the actor jobd signs with (jobmeta.default_actor -> "box:<IID>")
    assert bs.jobs_box_instance_id(_jobs2({
        f"jobs/j1/events/{ts}-box_46245045-aaa333.json":
            _ev(ts, "checkpoint", actor="box:46245045")})) == "46245045"


def test_resolve_jobs_box_finds_the_live_instance():
    ts = _now_ts()
    b2sec2 = _jobs2({f"jobs/j1/events/{ts}-box_1-aaa111.json":
                     _ev(ts, "heartbeat", instance_id="46245045")})
    live = [{"id": 12345, "label": "run:other"},
            {"id": 46245045, "label": "wave-box", "actual_status": "running"}]
    inst, iid = bs.resolve_jobs_box(b2sec2, live)
    assert iid == "46245045" and inst is live[1]
    # no box named -> nothing resolved (and no API call attempted)
    assert bs.resolve_jobs_box(_jobs2({}), live) == (None, None)


def test_verdict_running_jobs_box_is_not_stopped():
    """THE red-flag-5 fix: with the box resolved from the job's events, a running
    jobs-v2 job reads as running and names its box."""
    ts = _now_ts()
    b2sec2 = _jobs2({f"jobs/j1/events/{ts}-box_1-aaa111.json":
                     _ev(ts, "heartbeat", instance_id="46245045")})
    vast = {"id": 46245045, "actual_status": "running", "age_s": 3600}
    v = bs.verdict(vast, _empty_b2(), b2sec2)
    assert v.startswith("running: jobs-v2 job on box 46245045"), v
    assert "stopped" not in v


def test_verdict_unresolvable_jobs_box_says_interrupted_or_queued():
    """When the box genuinely is not live, the honest words are INTERRUPTED (a box
    claimed it and went away — jobd resumes it on the next boot) or QUEUED
    (nothing claimed it) — not the old blanket "stopped"."""
    ts = _now_ts()
    claimed = _jobs2({f"jobs/j1/events/{ts}-box_1-aaa111.json":
                      _ev(ts, "claimed", instance_id="46245045")})
    v = bs.verdict(None, _empty_b2(), claimed)
    assert v.startswith("interrupted:") and "46245045" in v
    queued = _jobs2({f"jobs/j1/events/{ts}-cli_x-aaa000.json":
                     _ev(ts, "submitted", actor="cli:x")})
    assert bs.verdict(None, _empty_b2(), queued).startswith("queued:")
