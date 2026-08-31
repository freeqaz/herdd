"""`defend: cheap|dear` — the lost-work hint (FLEET_REVIEW_2026-08-14 item 7).

On 2026-08-14 the ladder chased box 47694876 from $0.896 to $2.755/hr — ~2x the
replacement price — to defend a w8 bench bundle that was ~100% done and
deliberately checkpoint-free. The arithmetic was right; the CONFIG had no way to
say that the work was cheap to re-run, so the job-aware ceiling defended it like
training progress.

`defend:` is that missing sentence. `cheap` zeroes the lost-work term (`L`) in
`B_max = p_alt x (1 + (S + L) / R)`, collapsing it to the setup-only ceiling;
`dear` prices the work. Absent, it is DERIVED — `checkpoint_s` present => dear
(training-shaped), absent => cheap (bench-shaped) — and that derivation is
chosen so no existing config's numbers move.

Nothing here touches the network, the vast API, or a live watch.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bidpolicy as bp  # noqa: E402
import jobmeta as jm  # noqa: E402
import herdd as v  # noqa: E402

NOW = 1_000_000.0


# --------------------------------------------------------------------------- #
# 1. lost_work_hours — the term itself
# --------------------------------------------------------------------------- #
def test_cheap_zeroes_the_lost_work_term():
    assert bp.lost_work_hours(defend="cheap", ckpt_interval_h=0.4,
                              prior_runtime_h=3.0) == (0.0, "cheap")


def test_dear_with_checkpoints_is_half_an_interval():
    """Unchanged convention, shared with spot_breakeven."""
    assert bp.lost_work_hours(defend="dear", ckpt_interval_h=0.4) == (0.2, "dear")


def test_dear_without_checkpoints_loses_the_whole_run():
    """No sync means no "since the last checkpoint" — the replacement restarts
    the ticket from zero, so the accumulated wall time IS the lost work."""
    assert bp.lost_work_hours(defend="dear", ckpt_interval_h=0.0,
                              prior_runtime_h=1.135) == (1.135, "dear")


def test_checkpoints_outrank_prior_runtime_under_dear():
    """A job that checkpoints does not lose its whole run, whatever the box's
    uptime says."""
    h, _ = bp.lost_work_hours(defend="dear", ckpt_interval_h=0.4,
                              prior_runtime_h=9.0)
    assert h == 0.2


@pytest.mark.parametrize("ckpt,expect", [(0.4, "dear"), (0.0, "cheap"),
                                         (None, "cheap")])
def test_derivation_when_the_key_is_absent(ckpt, expect):
    assert bp.resolve_defend(None, ckpt_interval_h=ckpt) == expect


@pytest.mark.parametrize("explicit,ckpt", [("cheap", 0.4), ("dear", 0.0),
                                           ("DEAR", 0.0), (" cheap ", 0.4)])
def test_explicit_key_always_wins_over_the_derivation(explicit, ckpt):
    assert bp.resolve_defend(explicit, ckpt_interval_h=ckpt) == \
        explicit.strip().lower()


def test_an_unrecognised_hint_reads_as_absent_not_as_a_crash():
    """This runs inside a money-moving tick. jobmeta rejects a typo at SUBMIT
    time, where the author sees it; the ladder keeps deciding."""
    assert bp.resolve_defend("expensive", ckpt_interval_h=0.4) == "dear"
    assert bp.lost_work_hours(defend=object(), ckpt_interval_h=0.4) == (0.2, "dear")


def test_garbage_can_only_tighten_the_ceiling():
    assert bp.lost_work_hours(defend="dear", ckpt_interval_h="nope",
                              prior_runtime_h="nope") == (0.0, "dear")
    assert bp.lost_work_hours(defend="dear", prior_runtime_h=-5.0) == (0.0, "dear")


# --------------------------------------------------------------------------- #
# 2. defense_ceiling — cheap vs dear vs derived
# --------------------------------------------------------------------------- #
def test_cheap_collapses_the_ceiling_to_the_setup_only_term():
    """B_max = p_alt x (1 + S/R) — a replacement still eats the boot, but the
    wall time it discards is not billed to the defense."""
    p_alt, R = 0.896, 0.64
    cheap, basis = bp.defense_ceiling(p_alt=p_alt, remaining_h=R,
                                      defend="cheap", prior_runtime_h=1.135)
    assert basis == "remaining"
    assert cheap == round(p_alt * (1.0 + bp.SPOT_SETUP_H / R), 3)


def test_dear_prices_the_accumulated_wall_time_and_is_strictly_dearer():
    """The motivating shape: an un-checkpointed job ~1.1h in. `dear` licenses
    roughly the $2.755 the ladder actually paid; `cheap` does not."""
    p_alt, R, T = 0.896, 0.64, 1.135
    dear, _ = bp.defense_ceiling(p_alt=p_alt, remaining_h=R, defend="dear",
                                 prior_runtime_h=T)
    cheap, _ = bp.defense_ceiling(p_alt=p_alt, remaining_h=R, defend="cheap",
                                  prior_runtime_h=T)
    assert dear == round(p_alt * (1.0 + (bp.SPOT_SETUP_H + T) / R), 3)
    assert dear > 2.7 and cheap < 1.2
    assert cheap < dear


def test_the_derived_default_leaves_every_existing_config_untouched():
    """No `defend`, no `checkpoint_s` -> cheap -> L=0, which is EXACTLY what an
    un-checkpointed job priced before this landed. Back-compat is the reason the
    derivation points this way and not the other."""
    before = 0.45 * (1.0 + bp.SPOT_SETUP_H / 0.5)
    now, _ = bp.defense_ceiling(p_alt=0.45, remaining_h=0.5)
    assert now == round(before, 3)


def test_a_checkpointing_job_derives_dear_and_keeps_its_old_arithmetic():
    """The design-doc worked example, unchanged: R=0.5h, ckpt 360s -> $0.669."""
    assert bp.defense_ceiling(p_alt=0.45, remaining_h=0.5,
                              ckpt_interval_h=0.1) == (0.669, "remaining")
    assert bp.defense_ceiling(p_alt=0.45, remaining_h=0.5, ckpt_interval_h=0.1,
                              defend="dear") == (0.669, "remaining")


def test_cheap_on_a_checkpointing_job_drops_its_interval_term():
    cheap, _ = bp.defense_ceiling(p_alt=0.45, remaining_h=0.5,
                                  ckpt_interval_h=0.1, defend="cheap")
    assert cheap == round(0.45 * (1.0 + bp.SPOT_SETUP_H / 0.5), 3)
    assert cheap < 0.669


def test_an_unreadable_market_is_still_never_a_licence_to_defend():
    assert bp.defense_ceiling(p_alt=None, defend="dear",
                              prior_runtime_h=5.0) == (None, None)


# --------------------------------------------------------------------------- #
# 3. rebid_ladder — the hint is an ARGUMENT, and its effect is journaled
# --------------------------------------------------------------------------- #
def _ladder(**kw):
    """The 47694876 shape: p_alt $0.896, ~0.64h of work left, ~1.135h of
    un-checkpointed wall time behind it. cheap ceiling $1.166, dear $2.755."""
    base = dict(last_bid=1.00, market_min_bid=0.95, on_demand=6.0,
                max_bid=None, rungs_used=0, launch_dph_anchor=4.0,
                p_alt=0.896, remaining_h=0.64, prior_runtime_h=1.135)
    base.update(kw)
    return bp.rebid_ladder(**base)


def test_the_two_ceilings_are_the_motivating_numbers():
    assert _ladder(defend="cheap").ceiling == 1.166
    assert _ladder(defend="dear").ceiling == 2.755


def test_cheap_caps_a_rung_that_dear_would_have_allowed():
    """$1.166 vs $2.755 on the same market: the ladder still defends, at a
    disposable price."""
    cheap = _ladder(defend="cheap")
    dear = _ladder(defend="dear")
    assert cheap.action == dear.action == "rebid"
    assert cheap.ceiling < dear.ceiling
    assert cheap.price < dear.price


def test_the_reason_string_names_the_cheap_cap_and_both_prices():
    """Journal the hint's EFFECT: a cheap defend that changed nothing is noise;
    one that capped a ceiling dear would have allowed is the whole feature."""
    dec = _ladder(defend="cheap")
    assert "defend=cheap" in dec.reason
    assert "1.14h" in dec.reason                   # the wall time NOT defended
    assert "defend=dear would have allowed" in dec.reason
    assert f"${dec.ceiling:.3f}" in dec.reason


def test_a_derived_cheap_says_it_was_derived():
    dec = _ladder(defend=None)
    assert "defend=cheap (derived: no checkpoint_s)" in dec.reason


def test_dear_carries_no_cheap_note():
    assert "defend=cheap" not in _ladder(defend="dear").reason


def test_no_note_when_the_hint_changes_nothing():
    """Cheap with no un-checkpointed wall time to drop is not news."""
    dec = _ladder(defend="cheap", prior_runtime_h=None)
    assert "defend=cheap" not in dec.reason


def test_a_ceiling_stop_also_names_the_cheap_cap():
    """The refusal line is the audit trail, so the bound that produced it has
    to be legible there too."""
    dec = _ladder(defend="cheap", last_bid=1.16)
    assert dec.action == "stop"
    assert "JOB-AWARE defense ceiling" in dec.reason
    assert "lost work/cheap" in dec.reason
    assert "defend=cheap" in dec.reason


def test_no_palt_means_no_defense_ceiling_and_no_note():
    """`p_alt=None` still preserves the pre-2026-08-09 ladder exactly."""
    dec = _ladder(p_alt=None, defend="cheap")
    assert "defend" not in dec.reason


def test_bidpolicy_reads_no_config():
    """Purity: the hint arrives as an argument. bidpolicy must not learn to
    open job-config.yaml."""
    src = open(os.path.join(_HERE, "bidpolicy.py")).read()
    assert "load_job_config" not in src
    assert "\nimport jobmeta" not in src
    assert "yaml" not in src.lower().replace("job-config.yaml", "")


# --------------------------------------------------------------------------- #
# 4. job-config.yaml: validation, derivation, and the ride to the supervisor
# --------------------------------------------------------------------------- #
def _mkjob(tmp_path, **extra):
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "run.sh").write_text("echo hi\n")
    cfg = {"version": 1, "name": "probe-01", "entrypoint": "run.sh",
           "timeout_s": 60, "results": ["out/**"]}
    cfg.update(extra)
    (d / "job-config.json").write_text(json.dumps(cfg))
    return d


def test_config_derives_cheap_for_a_bench_shaped_bundle(tmp_path):
    d = _mkjob(tmp_path)
    cfg, warn = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert cfg["defend"] == "cheap" and warn == []


def test_config_derives_dear_for_a_checkpointing_bundle(tmp_path):
    d = _mkjob(tmp_path, checkpoint_s=300)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert cfg["defend"] == "dear"


@pytest.mark.parametrize("explicit,ckpt", [("dear", None), ("cheap", 300)])
def test_an_explicit_key_overrides_the_derivation(tmp_path, explicit, ckpt):
    extra = {"defend": explicit}
    if ckpt:
        extra["checkpoint_s"] = ckpt
    d = _mkjob(tmp_path, **extra)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert cfg["defend"] == explicit


@pytest.mark.parametrize("bad", ["expensive", "", 1, True, ["dear"]])
def test_an_invalid_defend_is_refused_at_submit(tmp_path, bad):
    d = _mkjob(tmp_path, defend=bad)
    with pytest.raises(jm.JobmetaError) as e:
        jm.validate_job_config(jm.load_job_config(str(d)), str(d))
    assert "defend" in str(e.value)


def test_the_fold_carries_defend_to_the_supervisor():
    """The supervisor folds events; it never reads the bundle. So the hint has
    to ride `submitted`, the same way timeout_s does."""
    ev = {"schema": 1, "job_id": "20260814T000000-x-0001", "event": "submitted",
          "ts": "2026-08-14T00:00:00.000Z", "nonce": "a" * 8, "actor": "cli:t",
          "name": "x", "entrypoint": "run.sh", "timeout_s": 60,
          "defend": "dear"}
    view = jm.fold_events([json.dumps(ev).encode()])
    assert view["defend"] == "dear"


def test_a_pre_2026_08_14_ticket_folds_to_no_hint():
    ev = {"schema": 1, "job_id": "20260801T000000-x-0001", "event": "submitted",
          "ts": "2026-08-01T00:00:00.000Z", "nonce": "b" * 8, "actor": "cli:t",
          "name": "x", "entrypoint": "run.sh", "timeout_s": 60}
    assert jm.fold_events([json.dumps(ev).encode()])["defend"] is None


def test_the_vocabulary_has_exactly_one_owner():
    """jobmeta validates what bidpolicy decides on — one tuple, not two."""
    assert jm.DEFEND_MODES is bp.DEFEND_MODES


# --------------------------------------------------------------------------- #
# 5. the jobs-lane driver
# --------------------------------------------------------------------------- #
def test_dear_wins_a_mixed_queue():
    """The box runs the whole queue, so one expensive ticket makes the box
    worth defending at the dearer price."""
    assert v._jobs_defend_hint([{"defend": "cheap"}, {"defend": "dear"}]) == "dear"


def test_an_all_cheap_queue_is_cheap():
    assert v._jobs_defend_hint([{"defend": "cheap"}, {"defend": "cheap"}]) == "cheap"


def test_a_legacy_queue_yields_no_hint_and_leaves_the_derivation_alone():
    assert v._jobs_defend_hint([{"job_id": "j1"}, None, "junk"]) is None
    assert v._jobs_defend_hint([]) is None


def test_prior_runtime_is_the_longest_running_attempt():
    views = [{"display_status": "running", "started_at": "20260814T000000000Z"},
             {"display_status": "running", "started_at": "20260814T010000000Z"},
             {"display_status": "submitted", "started_at": "20260813T000000000Z"}]
    now = v._ts_to_epoch("20260814T020000000Z")
    assert round(v._jobs_prior_runtime_h(views, now), 3) == 2.0


def test_prior_runtime_counts_from_the_resume_not_the_first_attempt():
    """Work before a resume is already gone; counting it would bill the ceiling
    twice for the same lost hours."""
    views = [{"display_status": "running", "started_at": "20260814T000000000Z",
              "last_resumed_ts": "20260814T013000000Z"}]
    now = v._ts_to_epoch("20260814T020000000Z")
    assert round(v._jobs_prior_runtime_h(views, now), 3) == 0.5


def test_prior_runtime_is_none_when_nothing_is_running():
    assert v._jobs_prior_runtime_h([{"display_status": "submitted"}], NOW) is None
    assert v._jobs_prior_runtime_h([{"display_status": "running"}], NOW) is None


def test_the_driver_hands_both_inputs_to_the_policy(monkeypatch):
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return bp.Rebid("stop", None, "captured", None, 0)

    monkeypatch.setattr(bp, "rebid_ladder", _capture)
    monkeypatch.setattr(v, "_job_handoff_emit", lambda jc, kind, **kw: None)
    monkeypatch.setattr(v, "_job_ladder_journal", lambda jc, kind, **kw: None)
    monkeypatch.setattr(v, "_jobs_work_horizon_h", lambda views, now, **kw: 0.64)
    import argparse
    import datetime
    now = v._ts_to_epoch("20260814T020000000Z")
    started = datetime.datetime.fromtimestamp(now - 1.135 * 3600.0,
                                              datetime.timezone.utc)
    jc = {"iid": "700", "last_bid": 1.00, "p_alt": 0.896, "p_alt_ts": now - 10,
          "a": argparse.Namespace(budget=100.0, dry_run=False),
          "ladder_journal": [],
          "pending_views": [{"job_id": "j1", "defend": "cheap",
                             "display_status": "running",
                             "started_at": started.strftime("%Y%m%dT%H%M%S000Z")}]}
    v._job_rebid_ladder(jc, jc["a"], "700", 0.95, 6.0, bp.EVICTION_OUTBID, now)
    assert seen["defend"] == "cheap"
    assert round(seen["prior_runtime_h"], 3) == 1.135
