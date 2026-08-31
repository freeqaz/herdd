"""`herdd ls`'s s/it figure: consecutive-step delta, not tqdm's aggregate.

The defect these pin, observed on box 47021787 on 2026-08-06 while
v9-gemma4-dec resumed from a checkpoint: the phase column read

    7.50 → 10.08 → 17.07 → 21.21 → 28.24 → … → 98.54 s/it

across one attempt whose actual step time never moved off ~101 s. tqdm's
displayed rate is an aggregate over the attempt, and HF's `ProgressCallback`
advances the bar by `global_step - current_step` on the first real step after a
resume — so the twenty steps the dataloader fast-forwards through land as ONE
enormous iteration and deflate the figure for the rest of the epoch.

Reading that column at minute five would have said the box was 13× faster than
it was. `PERF_LEVERS_INVESTIGATION_2026-08-06.md` §2.2 measured the same effect
and wrote the rule ("never quote tqdm s/it from a resumed run"); this makes the
tool obey it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402


def _hms(sec):
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(step, elapsed_s, rate, total=156):
    pct = int(100 * step / total)
    return (f"\r {pct:>3}%|██▋       | {step}/{total} "
            f"[{_hms(elapsed_s)}<1:00:00, {rate:.2f}s/it]")


def _log(tokens):
    return ("\r                            \r"
            "{'loss': '0.1366', 'grad_norm': '0.2501', "
            f"'num_tokens': '{tokens}', 'epoch': '0.2692'}}\n")


# The real series. Resume lands on step 21 at 2:37 elapsed (a ~56 s dataloader
# fast-forward plus one 101 s step); every step after that takes 101 s, so
# elapsed(n) = 157 + (n - 21) * 101 — and the s/it tqdm PRINTS is the observed
# one, which shares no arithmetic with that.
OBSERVED = [(21, 157, 7.50), (22, 258, 10.08), (23, 359, 17.07),
            (24, 460, 21.21), (25, 561, 28.24),
            (80, 6116, 97.51), (81, 6217, 98.54)]


def _view(tail, **kw):
    v = {"job_id": "j1", "name": "train-a", "display_status": "running",
         "last_tail": tail}
    v.update(kw)
    return v


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_consecutive_step_delta_is_flat_where_tqdms_figure_climbs_13x():
    """One assertion for the whole defect: over the observed series the emitted
    figure must be the flat truth, not the 7.50 → 98.54 ramp."""
    emitted = []
    for (s0, e0, r0), (s1, e1, r1) in zip(OBSERVED, OBSERVED[1:]):
        if s1 != s0 + 1:
            continue                      # not adjacent in the real series
        tail = _bar(s0, e0, r0) + _log("6.0e+06") + _bar(s1, e1, r1)
        pg = herdd._job_progress(_view(tail))
        emitted.append((pg["rate"], pg["rate_kind"]))
    assert emitted, "fixture built no adjacent pairs"
    assert all(x == ("101s/it", "delta") for x in emitted), emitted
    # …and tqdm's own numbers, which we are refusing to print, span 13x.
    assert max(r for _, _, r in OBSERVED) / min(r for _, _, r in OBSERVED) > 13


def test_the_last_pair_of_the_series_is_the_101s_step():
    """Steps 80 → 81, the pair quoted in the task: 98.54 s/it displayed, 101 s
    actually elapsed."""
    tail = _bar(80, 6116, 97.51) + _log("1.1e+07") + _bar(81, 6217, 98.54)
    pg = herdd._job_progress(_view(tail))
    assert pg["rate"] == "101s/it" and pg["rate_kind"] == "delta"
    assert pg["step"] == 81 and pg["total"] == 156 and pg["pct"] == 51


def test_a_single_bar_falls_back_and_says_so():
    """First heartbeat of an attempt: nothing to subtract from. The cumulative
    figure is still shown — it is better than nothing — but it is prefixed `~`
    and suffixed `(avg)` so no reader mistakes it for a step time."""
    pg = herdd._job_progress(_view(_bar(21, 157, 7.50)))
    assert pg["rate"] == "~7.5s/it(avg)" and pg["rate_kind"] == "avg"


def test_the_label_distinguishes_the_two_kinds_in_the_rendered_cell():
    delta = _bar(80, 6116, 97.51) + _bar(81, 6217, 98.54)
    only = _bar(81, 6217, 98.54)
    assert herdd._job_cell(_view(delta, n_checkpoints=3)) == \
        "train-a:running:51%:101s/it:ckpt3"
    assert herdd._job_cell(_view(only, n_checkpoints=3)) == \
        "train-a:running:51%:~98.5s/it(avg):ckpt3"


# ---------------------------------------------------------------------------
# Things that must NOT pair
# ---------------------------------------------------------------------------

def test_a_described_bar_is_not_a_training_step():
    """`Loading weights:` and `Map:` bars ride in the same heartbeat as the
    training bar and match the same regex. Paired into a delta they produce
    nonsense; read alone they turn a booting job into a finished one. (The
    dashboard's parseTail fixed this in TypeScript; the CLI had not.)"""
    noise = ("Loading weights: 100%|██| 339/339 [00:00<00:00, "
             "9415.48it/s]\nMap:  50%|██ | 2/4 [00:01<00:01, 1.20it/s]\n")
    assert herdd._job_progress(_view(noise)) == {}
    # …and it does not poison a real pair sharing the tail
    tail = noise + _bar(80, 6116, 97.51) + _bar(81, 6217, 98.54)
    assert herdd._job_progress(_view(tail))["rate"] == "101s/it"


def test_a_restart_inside_one_tail_does_not_pair_across_the_boundary():
    """Attempt 2's bar restarts elapsed at 0. Pairing step 81 of attempt 1 with
    step 3 of attempt 2 would emit a negative or wildly wrong delta."""
    tail = _bar(80, 6116, 97.51) + _bar(81, 6217, 98.54) + \
        _bar(1, 120, 120.0) + _bar(2, 240, 120.0)
    pg = herdd._job_progress(_view(tail))
    assert pg["step"] == 2
    assert pg["rate"] == "120s/it" and pg["rate_kind"] == "delta"


def test_steps_faster_than_the_elapsed_resolution_fall_back():
    """tqdm's elapsed stamp is whole seconds. An eval bar at 0.08 s/it would
    give a delta of 0 or 1 — worse than the aggregate it replaced."""
    tail = _bar(151, 12, 0.083, total=156) + _bar(152, 12, 0.083, total=156)
    pg = herdd._job_progress(_view(tail))
    assert pg["rate_kind"] == "avg" and pg["rate"] == "~0.1s/it(avg)"


def test_a_repainted_identical_bar_does_not_produce_a_zero_delta():
    """tqdm repaints the same step around each log dict; the duplicate must be
    skipped, not divided by."""
    tail = (_bar(80, 6116, 97.51) + _log("1.0e+07") + _bar(80, 6116, 97.51)
            + _bar(81, 6217, 98.54) + _log("1.1e+07") + _bar(81, 6217, 98.54))
    assert herdd._job_progress(_view(tail))["rate"] == "101s/it"


def test_a_multi_step_gap_is_divided_by_the_gap():
    """A throttled bar can skip a repaint. 3 steps in 303 s is 101 s/step, not
    303."""
    tail = _bar(78, 5914, 90.0) + _bar(81, 6217, 98.54)
    assert herdd._job_progress(_view(tail))["rate"] == "101s/it"


# ---------------------------------------------------------------------------
# Everything else the cell carries must be unchanged
# ---------------------------------------------------------------------------

def test_tokens_per_second_still_comes_off_the_freshest_bars_elapsed():
    tail = _bar(80, 6116, 97.51) + _bar(81, 6217, 98.54) + _log("1.2434e+07")
    pg = herdd._job_progress(_view(tail))
    assert pg["toks"] == 1.2434e7 / 6217


def test_no_bar_no_progress():
    assert herdd._job_progress(_view("")) == {}
    assert herdd._job_progress(_view("Traceback (most recent call last):")) == {}
    assert herdd._job_progress(_view("", n_checkpoints=2)) == {"ckpt": 2}


def test_an_it_per_s_bar_still_yields_a_seconds_per_step_delta():
    """Fast-but-not-too-fast bars print `it/s`. The delta is in seconds either
    way; only the fallback echoes tqdm's unit."""
    one = "\r  7%|▋ | 7/100 [00:21<01:33, 0.33it/s]"
    two = "\r  8%|▋ | 8/100 [00:24<01:30, 0.33it/s]"
    assert herdd._job_progress(_view(one + two))["rate"] == "3.0s/it"
    assert herdd._job_progress(_view(one))["rate"] == "~0.3it/s(avg)"
