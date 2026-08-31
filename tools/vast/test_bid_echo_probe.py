"""Pure-logic tests for the bid-echo probe (FLEET_REVIEW_2026-08-14 item 3).

Nothing here touches the network, vast, or fleetd: the probe funnels every API
call through three seam functions (`read_box`, `read_floor`, `put_bid`), and
every test below monkeypatches all three onto a scripted fake plus a fake clock.
That is the whole reason the seams exist — the flip-detection state machine is
the part that can be wrong, and it must be testable without renting anything.

What is pinned, and why each one is a way the probe could quietly lie:

  * a flip needs TWO consecutive new-bid reads. One read is not a flip: the
    offers query alternates between our rented chunk and the machine's free
    sibling chunk (AUTOBID_DESIGN, probe v2 finding 1), so a single new read can
    be attribution churn rather than the echo catching up.
  * an "old" read RESTARTS the confirming run and the lag clock — the echo is
    demonstrably still stale at that moment.
  * an "other" read (a real competitor, or a sibling chunk's genuine floor) and
    an "unread" (failed read) are evidence about NEITHER price: recorded, and
    neutral to the streak. Treating them as refutations would bias the measured
    lag upward on exactly the busy machines we most want to measure.
  * `censored-at-max-wait` and `no-echo-observed` are OUTCOMES, not errors. The
    whole point of the probe is that passive data is censored at the window; a
    probe that reported its own censoring as a failure would repeat the defect.
  * the original bid is restored on every exit path, including an exception
    mid-poll and an interrupt.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bid_echo_probe as bep  # noqa: E402

BOX = 47316203
MACHINE = 52305
T0 = 1_000_000.0


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeClock:
    """Time only moves when the probe sleeps, so every lag in these tests is an
    exact multiple of the poll interval."""

    def __init__(self, t0=T0):
        self.t = float(t0)

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += float(dt)


class FakeVast:
    """A scripted machine. `floors` is consumed one entry per poll and the last
    entry repeats forever; None means the market read FAILED (ignorance)."""

    def __init__(self, floors, bid, *, is_bid=True, status="running",
                 put_ok=True, raise_on_read=None):
        self.floors = list(floors)
        self.bid = float(bid)
        self.is_bid = is_bid
        self.status = status
        self.put_ok = put_ok
        self.raise_on_read = raise_on_read      # read index (1-based) that dies
        self.reads = 0
        self.puts = []

    # -- seams ---------------------------------------------------------- #
    def read_floor(self, machine_id, num_gpus=None):
        self.reads += 1
        if self.raise_on_read and self.reads == self.raise_on_read:
            raise RuntimeError("boom: offers read exploded")
        f = self.floors[0] if len(self.floors) == 1 else self.floors.pop(0)
        if f is None:
            return {"ok": False, "listed": False, "min_bid": None,
                    "floors": [], "scaled": False}
        return {"ok": True, "listed": True, "min_bid": f, "floors": [f],
                "scaled": False}

    def read_box(self, iid):
        return {"id": iid, "machine_id": MACHINE, "num_gpus": 2,
                "is_bid": self.is_bid, "actual_status": self.status,
                "standing_bid": self.bid, "dph_total": self.bid + 0.01}

    def put_bid(self, iid, price):
        self.puts.append((iid, price))
        if not self.put_ok:
            return False, "429 rate limited"
        self.bid = float(price)
        return True, None


def install(monkeypatch, fake):
    monkeypatch.setattr(bep, "read_floor", fake.read_floor)
    monkeypatch.setattr(bep, "read_box", fake.read_box)
    monkeypatch.setattr(bep, "put_bid", fake.put_bid)
    # a seam that is never reached must still be un-callable in a test
    monkeypatch.setattr(bep, "_herdd",
                        lambda: pytest.fail("the probe touched the vast API"))
    return fake


def make_probe(tmp_path, fake, clock, **kw):
    log = bep.EventLog(str(tmp_path / "echo.ndjson"), box=BOX, machine=MACHINE,
                       now=clock.now)
    kw.setdefault("interval", 60.0)
    kw.setdefault("max_wait", 600.0)
    kw.setdefault("baseline_wait", 300.0)
    kw.setdefault("max_price", 1.0)
    return bep.EchoProbe(box=BOX, machine=MACHINE, num_gpus=2,
                         orig_bid=fake.bid, log=log, now=clock.now,
                         sleep=clock.sleep, quiet=True, **kw), log


def read_ndjson(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_classify_floor_labels_each_read():
    assert bep.classify_floor(0.034, 0.032, 0.034) == "new"
    assert bep.classify_floor(0.032, 0.032, 0.034) == "old"
    assert bep.classify_floor(0.099, 0.032, 0.034) == "other"
    assert bep.classify_floor(None, 0.032, 0.034) == "unread"
    # the echo comes back QUANTIZED TO 3 DECIMALS against a 4-decimal bid:
    # 0.034 IS 0.0336, and the eps exists to say so.
    assert bep.classify_floor(0.034, 0.0336, 0.0353) == "old"


def test_eps_does_not_drift_from_the_guard():
    """The probe's classifier and the shipped self-floor guard must agree on
    what 'this floor is our own bid' means, or the probe measures a lag the
    guard would not have suppressed."""
    import bidpolicy
    assert bep.ECHO_EPS == bidpolicy.BID_SELF_FLOOR_EPS


def test_build_plan_raises_then_returns_to_the_original_bid():
    plan = bep.build_plan(0.032, 5, ["raise", "lower"])
    assert [(p.name, p.from_bid, p.to_bid) for p in plan] == [
        ("raise", 0.032, 0.034), ("lower", 0.034, 0.032)]
    assert plan[-1].to_bid == 0.032, "the box must end at its starting price"


def test_build_plan_lower_alone_moves_down():
    plan = bep.build_plan(0.100, 10, ["lower"])
    assert [(p.name, p.to_bid) for p in plan] == [("lower", 0.09)]


def test_build_plan_rejects_an_unknown_phase():
    with pytest.raises(bep.ProbeError):
        bep.build_plan(0.1, 5, ["sideways"])


# --------------------------------------------------------------------------- #
# preflight refusals
# --------------------------------------------------------------------------- #
def _box(**kw):
    b = {"id": BOX, "machine_id": MACHINE, "num_gpus": 2, "is_bid": True,
         "actual_status": "running", "standing_bid": 0.032, "dph_total": 0.04}
    b.update(kw)
    return b


def test_preflight_refuses_a_phase_above_max_price():
    plan = bep.build_plan(0.032, 5, ["raise", "lower"])
    errs = bep.preflight(_box(), plan, max_price=0.033)
    assert any("max-price" in e and "raise" in e for e in errs)
    assert bep.preflight(_box(), plan, max_price=0.05) == []


def test_preflight_refuses_when_the_current_bid_already_exceeds_the_cap():
    plan = bep.build_plan(0.5, 5, ["raise"])
    errs = bep.preflight(_box(standing_bid=0.5), plan, max_price=0.1)
    assert any("CURRENT bid" in e for e in errs)


def test_preflight_refuses_a_non_bid_box():
    plan = bep.build_plan(0.5, 5, ["raise"])
    errs = bep.preflight(_box(is_bid=False), plan, max_price=10)
    assert any("not a bid (spot) box" in e for e in errs)


def test_preflight_refuses_a_stopped_box():
    plan = bep.build_plan(0.5, 5, ["raise"])
    errs = bep.preflight(_box(actual_status="exited"), plan, max_price=10)
    assert any("actual_status" in e for e in errs)


def test_preflight_refuses_a_move_smaller_than_the_price_grid():
    """0.5% of $0.032 rounds to the same 3-decimal price: a flip that could
    never be observed is a refusal, not a run."""
    plan = bep.build_plan(0.032, 0.5, ["raise"])
    errs = bep.preflight(_box(), plan, max_price=10)
    assert any("price-grid step" in e for e in errs)


def test_preflight_refuses_a_missing_box():
    assert bep.preflight(None, [], max_price=1) == ["box not found (or the "
                                                    "instance read failed)"]


def test_move_bid_refuses_above_max_price_even_if_the_plan_did_not(tmp_path,
                                                                   monkeypatch):
    """Defence in depth: the cap is enforced at the PUT, not only at the plan."""
    clock = FakeClock()
    fake = install(monkeypatch, FakeVast([0.032], 0.032))
    probe, _log = make_probe(tmp_path, fake, clock, max_price=0.033)
    with pytest.raises(bep.ProbeError):
        probe._move_bid(0.05, why="test")
    assert fake.puts == [], "no PUT may leave the process above the cap"


# --------------------------------------------------------------------------- #
# the flip state machine
# --------------------------------------------------------------------------- #
def test_flip_needs_two_consecutive_new_reads(tmp_path, monkeypatch):
    # baseline: old, old -> stable. flip polls: old, new, new.
    fake = install(monkeypatch, FakeVast(
        [0.032, 0.032, 0.032, 0.034, 0.034], 0.032))
    clock = FakeClock()
    probe, log = make_probe(tmp_path, fake, clock)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "flipped"
    assert res.lag_s == 60.0, "lag = first read of the confirming run - the PUT"
    assert res.confirm_lag_s == 120.0
    assert res.counts == {"old": 1, "new": 2}
    assert fake.puts == [(BOX, 0.034)]
    assert res.baseline["outcome"] == "stable"


def test_a_single_new_read_is_not_a_flip(tmp_path, monkeypatch):
    """new, old, new, old ... never confirms: the echo is flapping, and the
    probe must censor rather than report the first new read as the lag."""
    fake = install(monkeypatch, FakeVast(
        [0.032, 0.032] + [0.034, 0.032] * 12, 0.032))
    clock = FakeClock()
    probe, _log = make_probe(tmp_path, fake, clock, max_wait=300.0)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "censored-at-max-wait"
    assert res.lag_s is None


def test_other_and_unread_reads_are_recorded_but_reset_nothing(tmp_path,
                                                               monkeypatch):
    """A real competitor's floor and a failed read are evidence about neither
    of our two prices, so the confirming run survives them."""
    fake = install(monkeypatch, FakeVast(
        [0.032, 0.032,                       # baseline
         0.034, 0.099, None, 0.034], 0.032))  # new, other, unread, new
    clock = FakeClock()
    probe, _log = make_probe(tmp_path, fake, clock)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "flipped"
    assert res.counts == {"new": 2, "other": 1, "unread": 1}
    # the lag still dates from the FIRST new read, three polls earlier
    assert res.lag_s == 0.0 and res.confirm_lag_s == 180.0


def test_an_old_read_restarts_the_confirming_run(tmp_path, monkeypatch):
    fake = install(monkeypatch, FakeVast(
        [0.032, 0.032,                        # baseline
         0.034, 0.032, 0.034, 0.034], 0.032))  # new, OLD (still stale), new, new
    clock = FakeClock()
    probe, _log = make_probe(tmp_path, fake, clock)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "flipped"
    assert res.lag_s == 120.0, "the lag clock restarts at the second new run"
    assert res.counts == {"new": 3, "old": 1}


def test_censored_at_max_wait_is_an_outcome_not_an_error(tmp_path, monkeypatch):
    fake = install(monkeypatch, FakeVast([0.032], 0.032))   # echoes forever
    clock = FakeClock()
    probe, log = make_probe(tmp_path, fake, clock, max_wait=300.0)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "censored-at-max-wait"
    assert (res.lag_s, res.confirm_lag_s) == (None, None)
    assert res.polls == 6                                   # t=0..300 @ 60 s
    summ = [r for r in log.rows if r["event"] == "phase_summary"]
    assert summ and summ[0]["outcome"] == "censored-at-max-wait"
    assert summ[0]["max_wait_s"] == 300.0


def test_no_echo_observed_stops_without_moving_the_bid(tmp_path, monkeypatch):
    """The floor never matches our standing bid at all: this machine may simply
    not echo. That is a finding, and it is measured for free — so the probe must
    NOT move the bid to chase it."""
    fake = install(monkeypatch, FakeVast([0.099], 0.032))
    clock = FakeClock()
    probe, log = make_probe(tmp_path, fake, clock, baseline_wait=180.0)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "no-echo-observed"
    assert fake.puts == [], "no bid move on a machine that never echoed"
    assert res.baseline["matches"] == 0 and res.baseline["polls"] == 4


def test_an_intermittent_baseline_still_runs_and_says_so(tmp_path, monkeypatch):
    """Matched, but never twice in a row: the echo is intermittent (attribution
    churn). The phase runs; the summary records `unstable` so the number can be
    read with that caveat."""
    fake = install(monkeypatch, FakeVast(
        [0.032, 0.099] * 3 + [0.034, 0.034], 0.032))
    clock = FakeClock()
    probe, _log = make_probe(tmp_path, fake, clock, baseline_wait=300.0)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.baseline["outcome"] == "unstable"
    assert res.outcome == "flipped"
    assert fake.puts == [(BOX, 0.034)]


def test_a_refused_bid_move_ends_the_phase(tmp_path, monkeypatch):
    fake = install(monkeypatch, FakeVast([0.032], 0.032, put_ok=False))
    clock = FakeClock()
    probe, log = make_probe(tmp_path, fake, clock)
    res = probe.run_phase(bep.Phase("raise", 0.032, 0.034))
    assert res.outcome == "bid-move-failed"
    moves = [r for r in log.rows if r["event"] == "bid_move"]
    assert moves and moves[0]["ok"] is False and "429" in moves[0]["error"]


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #
def test_the_bid_is_restored_when_a_seam_raises(tmp_path, monkeypatch):
    out = tmp_path / "echo.ndjson"
    # reads: 2 baseline, then the 3rd (first post-move) read explodes
    fake = install(monkeypatch, FakeVast([0.032], 0.032, raise_on_read=3))
    clock = FakeClock()
    rc = bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "1.0",
                   "--phases", "raise", "--quiet"],
                  now=clock.now, sleep=clock.sleep, install_signals=False)
    assert rc == 1
    assert fake.puts == [(BOX, 0.034), (BOX, 0.032)], \
        "the raise, then the restore back to the original bid"
    assert fake.bid == 0.032
    rows = read_ndjson(out)
    restores = [r for r in rows if r["event"] == "bid_restore"]
    assert restores and restores[-1]["ok"] is True
    end = [r for r in rows if r["event"] == "probe_end"][-1]
    assert end["restored"] is True and end["final_bid"] == 0.032
    assert "RuntimeError" in end["failure"]


def test_an_interrupt_restores_the_bid_and_exits_130(tmp_path, monkeypatch):
    out = tmp_path / "echo.ndjson"

    class Interrupting(FakeVast):
        def read_floor(self, machine_id, num_gpus=None):
            if self.reads >= 2:                  # first read after the move
                raise KeyboardInterrupt
            return super().read_floor(machine_id, num_gpus)

    fake = install(monkeypatch, Interrupting([0.032], 0.032))
    clock = FakeClock()
    rc = bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "1.0",
                   "--phases", "raise", "--quiet"],
                  now=clock.now, sleep=clock.sleep, install_signals=False)
    assert rc == 130
    assert fake.puts[-1] == (BOX, 0.032) and fake.bid == 0.032


def test_restore_retries_and_reports_a_failure(tmp_path, monkeypatch):
    fake = install(monkeypatch, FakeVast([0.032], 0.032))
    clock = FakeClock()
    probe, log = make_probe(tmp_path, fake, clock)
    probe.current_bid = 0.034            # pretend a move landed
    fake.put_ok = False
    assert probe.restore() is False
    attempts = [r for r in log.rows if r["event"] == "bid_restore"]
    assert len(attempts) == bep.RESTORE_ATTEMPTS
    assert all(r["ok"] is False for r in attempts)


def test_restore_is_a_no_op_when_the_bid_never_moved(tmp_path, monkeypatch):
    fake = install(monkeypatch, FakeVast([0.032], 0.032))
    clock = FakeClock()
    probe, _log = make_probe(tmp_path, fake, clock)
    assert probe.restore() is True
    assert fake.puts == []


# --------------------------------------------------------------------------- #
# CLI / event log
# --------------------------------------------------------------------------- #
def _both_phase_script():
    return [0.032, 0.032,                 # raise baseline
            0.032, 0.034, 0.034,          # raise flip (lag 60 s)
            0.034, 0.034,                 # lower baseline
            0.034, 0.034, 0.032, 0.032]   # lower flip (lag 120 s — the slow
                                          # direction, as probe v2 measured)


def test_main_runs_both_phases_and_ends_at_the_original_price(tmp_path,
                                                              monkeypatch):
    out = tmp_path / "echo.ndjson"
    fake = install(monkeypatch, FakeVast(_both_phase_script(), 0.032))
    clock = FakeClock()
    rc = bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "0.05",
                   "--quiet"], now=clock.now, sleep=clock.sleep,
                  install_signals=False)
    assert rc == 0
    assert fake.puts == [(BOX, 0.034), (BOX, 0.032)]
    assert fake.bid == 0.032, "the box ends at the price it started at"
    summ = [r for r in read_ndjson(out) if r["event"] == "phase_summary"]
    assert [(s["phase"], s["outcome"], s["lag_s"]) for s in summ] == [
        ("raise", "flipped", 60.0), ("lower", "flipped", 120.0)]


def test_main_refuses_over_max_price_without_touching_the_bid(tmp_path,
                                                              monkeypatch):
    out = tmp_path / "echo.ndjson"
    fake = install(monkeypatch, FakeVast([0.032], 0.032))
    clock = FakeClock()
    rc = bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "0.033",
                   "--quiet"], now=clock.now, sleep=clock.sleep,
                  install_signals=False)
    assert rc == 2
    assert fake.puts == []
    assert not os.path.exists(out), "a refusal opens no event log"


def test_main_refuses_a_non_spot_box(tmp_path, monkeypatch, capsys):
    out = tmp_path / "echo.ndjson"
    fake = install(monkeypatch, FakeVast([0.032], 0.032, is_bid=False))
    clock = FakeClock()
    rc = bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "1.0",
                   "--quiet"], now=clock.now, sleep=clock.sleep,
                  install_signals=False)
    assert rc == 2 and fake.puts == []
    assert "not a bid (spot) box" in capsys.readouterr().err


def test_main_clamps_the_interval_to_the_politeness_floor(tmp_path, monkeypatch):
    out = tmp_path / "echo.ndjson"
    fake = install(monkeypatch, FakeVast(_both_phase_script(), 0.032))
    clock = FakeClock()
    bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "0.05",
              "--interval", "5", "--quiet"], now=clock.now, sleep=clock.sleep,
             install_signals=False)
    start = [r for r in read_ndjson(out) if r["event"] == "probe_start"][0]
    assert start["interval_s"] == bep.INTERVAL_FLOOR_S


def test_every_ndjson_row_is_well_formed(tmp_path, monkeypatch):
    out = tmp_path / "echo.ndjson"
    fake = install(monkeypatch, FakeVast(_both_phase_script(), 0.032))
    clock = FakeClock()
    bep.main(["--box", str(BOX), "--out", str(out), "--max-price", "0.05",
              "--quiet"], now=clock.now, sleep=clock.sleep,
             install_signals=False)
    rows = read_ndjson(out)
    assert [r["event"] for r in rows][0] == "probe_start"
    assert [r["event"] for r in rows][-1] == "probe_end"
    ts = [r["ts"] for r in rows]
    assert ts == sorted(ts), "timestamps must be non-decreasing"
    for r in rows:
        assert set(("ts", "iso", "event", "box", "machine")) <= set(r)
        assert r["box"] == BOX and r["machine"] == MACHINE
        assert r["iso"].endswith("+00:00")
    polls = [r for r in rows if r["event"] == "poll"]
    assert len(polls) == len(_both_phase_script())
    for p in polls:
        assert set(("phase", "stage", "our_bid", "floor_read", "matched",
                    "observed_bid", "floors", "listed")) <= set(p)
        assert p["matched"] in ("old", "new", "other", "unread")
        assert p["phase"] in bep.PHASE_NAMES
    # the poll rows are the raw evidence: the summary's lag must be re-derivable
    flip = [p for p in polls if p["stage"] == "flip" and p["phase"] == "raise"]
    assert [p["matched"] for p in flip] == ["old", "new", "new"]
    assert flip[1]["since_move_s"] == 60.0


def test_the_seams_wrap_symbols_that_still_exist_in_herdd():
    """The three seams are thin wrappers, NOT reimplementations — so the only
    way they can rot is a rename on the herdd side. Attribute existence only:
    nothing here calls the API."""
    import inspect

    import herdd
    for name in ("_get_instance_soft", "_instance_standing_bid", "_num_dph",
                 "_market_min_bid_read", "set_bid"):
        assert callable(getattr(herdd, name, None)), \
            f"bid_echo_probe's seams wrap herdd.{name}, which is gone"
    assert set(herdd.MarketRead._fields) >= {"ok", "listed", "min_bid",
                                               "floors", "scaled"}
    assert list(inspect.signature(herdd.set_bid).parameters) == ["iid", "price"]


def test_the_probe_summary_names_the_censoring(tmp_path, monkeypatch):
    res = [bep.PhaseResult("raise", 0.032, 0.034, "censored-at-max-wait", None,
                           None, 90, {"old": 90}, {"outcome": "stable"})]
    text = bep.format_summary(res, 0.032, "/dev/null")
    assert "censored-at-max-wait" in text and "--max-wait" in text
