"""test_job_logs_provenance.py — `job logs` must say WHOSE bytes it is showing.

Regression for a live incident on 2026-08-06, job
20260806T082213-v11-qwen25c7b-chat-dec-train-aff8, which produced TWO wrong
readings within an hour because the log view carried no provenance:

  * a FALSE FAILURE — `ChildFailedError` grepped out of a previous attempt's
    bytes while the current attempt was healthy; and
  * a FALSE RESUME CONFIRM — a match served from a heartbeat emitted by a box
    that had already been evicted.

The tail shown for a non-terminal job comes from the last heartbeat, and a
retarget/requeue does NOT invalidate it. So when the emitting box differs from
the box the job is now aimed at, those bytes PREDATE the move and say nothing
about the current attempt. Same class as the dead-box resume line: presence is
not provenance.

Pure-function tests over the header builder; no network, no B2.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402


def _view(**kw):
    v = {"status": "started", "display_status": "running", "instance_id": None,
         "target_box": None, "last_heartbeat_ts": None, "last_tail": ""}
    v.update(kw)
    return v


JID = "20260806T082213-v11-qwen25c7b-chat-dec-train-aff8"


def test_header_names_the_job_box_and_the_emitting_box():
    out = "\n".join(herdd._job_log_provenance(
        _view(instance_id="47011548", target_box="47011548",
              last_heartbeat_ts="20260806T194712475Z"), JID))
    assert "47011548" in out
    assert "heartbeat" in out.lower()


def test_mismatched_emitting_box_is_called_out_loudly():
    """THE bug: heartbeat from the dead box, job already moved on."""
    out = "\n".join(herdd._job_log_provenance(
        _view(instance_id="46999867", target_box="47011548",
              last_heartbeat_ts="20260806T170440975Z"), JID))
    assert "PROVENANCE" in out
    assert "46999867" in out and "47011548" in out
    assert "PRIOR attempt" in out
    # it must warn in BOTH directions, not just about failures
    assert "not evidence" in out


def test_matching_box_does_not_cry_wolf():
    out = "\n".join(herdd._job_log_provenance(
        _view(instance_id="47011548", target_box="47011548",
              last_heartbeat_ts="20260806T194712475Z"), JID))
    assert "PROVENANCE" not in out


def test_stale_heartbeat_is_flagged():
    out = "\n".join(herdd._job_log_provenance(
        _view(instance_id="46999867", target_box="46999867",
              last_heartbeat_ts="20260101T000000000Z"), JID))
    assert "STALE" in out


def test_no_heartbeat_yet_is_not_an_error():
    out = "\n".join(herdd._job_log_provenance(
        _view(instance_id=None, target_box="47011548"), JID))
    assert JID in out
    assert "STALE" not in out and "PROVENANCE" not in out


def test_hb_age_parses_runmeta_stamp_and_tolerates_junk():
    assert herdd._hb_age_s("20260806T194712475Z") is not None
    assert herdd._hb_age_s(None) is None
    assert herdd._hb_age_s("not-a-timestamp") is None


# --------------------------------------------------------------------------- #
# fold precedence: a `failed` tail must not outlive the failure it describes.
# ROOT CAUSE of the false-failure reading above -- fold_events set last_tail
# from the newest heartbeat and then unconditionally overwrote it with the last
# `failed` event's tail, so after a requeue a healthy job kept reporting the
# dead attempt's traceback while streaming fresh heartbeats.
# --------------------------------------------------------------------------- #
import json  # noqa: E402

import jobmeta as jm  # noqa: E402


def _ev(event, ts, **kw):
    d = {"event": event, "ts": ts, "job_id": JID}
    d.update(kw)
    return json.dumps(d)


def test_failed_tail_wins_while_the_failure_is_the_latest_word():
    bodies = [
        _ev("claimed", "20260806T090000000Z", instance_id="46962674"),
        _ev("heartbeat", "20260806T093000000Z", instance_id="46962674",
            tail="step 45/156"),
        _ev("failed", "20260806T093400000Z", instance_id="46962674", rc=1,
            reason="rc=1", tail="ChildFailedError: boom"),
    ]
    v = jm.fold_events(bodies, live_iids=set())
    assert v["last_tail"] == "ChildFailedError: boom"


def test_a_newer_heartbeat_supersedes_a_stale_failure_tail():
    """THE regression: requeued onto a new box, streaming healthy heartbeats."""
    bodies = [
        _ev("claimed", "20260806T090000000Z", instance_id="46962674"),
        _ev("failed", "20260806T093400000Z", instance_id="46962674", rc=1,
            reason="rc=1", tail="ChildFailedError: boom"),
        _ev("resumed", "20260806T170202417Z", instance_id="47011548"),
        _ev("heartbeat", "20260806T195507286Z", instance_id="47011548",
            tail="55/156 loss 0.2597"),
    ]
    v = jm.fold_events(bodies, live_iids={"47011548"})
    assert v["last_tail"] == "55/156 loss 0.2597", v["last_tail"]
    # `resumed` re-opens the fold, so the prior failure is no longer sticky --
    # status and fail_reason clear too. The tail was the ONE field that kept
    # pointing at the dead attempt.
    assert v["fail_reason"] is None
    assert v["status"] not in jm.TERMINAL


def test_newer_heartbeat_supersedes_the_tail_even_without_a_resume_event():
    """Isolates the tail rule from the requeue re-open: no `resumed` here, so
    the failure is still sticky, yet a LATER heartbeat still owns the tail."""
    bodies = [
        _ev("claimed", "20260806T090000000Z", instance_id="46962674"),
        _ev("failed", "20260806T093400000Z", instance_id="46962674", rc=1,
            reason="rc=1", tail="ChildFailedError: boom"),
        _ev("heartbeat", "20260806T195507286Z", instance_id="46962674",
            tail="55/156 loss 0.2597"),
    ]
    v = jm.fold_events(bodies, live_iids=set())
    assert v["last_tail"] == "55/156 loss 0.2597", v["last_tail"]
    assert v["fail_reason"] == "rc=1"      # failure metadata survives


def test_no_heartbeats_at_all_still_shows_the_failure_tail():
    bodies = [
        _ev("claimed", "20260806T090000000Z", instance_id="46962674"),
        _ev("failed", "20260806T093400000Z", instance_id="46962674", rc=1,
            reason="rc=1", tail="ChildFailedError: boom"),
    ]
    v = jm.fold_events(bodies, live_iids=set())
    assert v["last_tail"] == "ChildFailedError: boom"


# --- the inverse failure: a CORRECT tail branded as a prior attempt ----------- #
# The same banner fires backwards when the fold is the thing that is wrong.
# On 2026-08-07, 20260806T212132-v9-gemma4-dec-train-8818 was requeued twice
# (jobmeta.requeue_ticket emits `resumed` with `box=<new>`, the frozen-vocabulary
# ticket move) while `target_box` folded only `retargeted` — so the view read
# instance_id=47045282 (the live trainer) against target_box=47041615 (destroyed
# hours before), and `job logs` told the operator that the LIVE run's bytes
# "PREDATE the move and describe a PRIOR attempt". Cry-wolf in the direction
# that gets a healthy run killed.
def test_a_requeued_job_does_not_brand_its_own_live_tail_as_prior(tmp_path):
    import json
    import jobmeta as jm
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "testfixtures", "jobmeta", "v9-gemma4-requeue-chain.jsonl")
    with open(p) as fh:
        evs = [json.loads(l) for l in fh if l.strip()]
    v = jm.fold_events(evs, live_iids={"47045282"})
    assert v["instance_id"] == "47045282"
    out = "\n".join(herdd._job_log_provenance(
        v, "20260806T212132-v9-gemma4-dec-train-8818"))
    assert "PROVENANCE" not in out, out
    assert "47041615" not in out, "must not name a box the job left two moves ago"
