"""`herdd search`'s CPU lane: measured ranking, and a floor that can only
refuse silicon it has actually measured.

The lane has two orderings that must not be confused. Without a CPU ask the
order is price-ascending and every other lane depends on it, so a GPU-only
search must come out of here byte-identical to before. With one, offers are
ranked on MEASURED work per dollar — a different unit from the `cpu_score`
prior, which is why measured and unmeasured rows cannot share a sort key.

The floor is the part worth pinning hardest. It is armed by default (owner
decision, 2026-08-27) and it is one-directional: an offer we have measured and
found pathological is refused; an offer we have never measured cannot be shown
to be slow and is kept. Roughly 70% of the cheap market is unmeasured, so a
floor that treated unknown as bad would empty the board and read as
selectivity.
"""
from __future__ import annotations

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vastlib.cli import search as cli_search  # noqa: E402
from vastlib.market import offers as market_offers  # noqa: E402

_TABLE = {
    "units": "pyops", "rate_is": "per_core_s", "generated": "t",
    "n_machines": 2, "n_models": 2, "fleet_median": 4.0e6, "fleet_spread": 3.0,
    "by_machine": {},
    "by_model": {"Fast CPU": {"rate": 8.0e6, "n_machines": 3, "spread": 1.2},
                 "Slow CPU": {"rate": 1.0e6, "n_machines": 1, "spread": None}},
}

def _offer(**kw):
    """A row shaped enough for `fmt.fmt_offer`, which is not the surface under
    test here but does render every line."""
    base = {"id": 0, "num_gpus": 1, "gpu_name": "RTX 4090", "gpu_ram": 24576,
            "min_bid": 0.01, "disk_space": 100, "storage_cost": 0.1,
            "reliability": 0.99, "inet_down": 1000, "host_id": 7,
            "geolocation": "US", "machine_id": 0, "cpu_name": "",
            "cpu_cores_effective": 8, "cpu_ghz": 3.0, "dph_total": 0.10}
    base.update(kw)
    return base


FAST = _offer(id=1, machine_id=11, cpu_name="Fast CPU",
              cpu_cores_effective=8, dph_total=0.10)
SLOW = _offer(id=2, machine_id=22, cpu_name="Slow CPU",
              cpu_cores_effective=256, cpu_ghz=2.2, dph_total=0.01)
UNKNOWN = _offer(id=3, machine_id=33, cpu_name="Never Seen",
                 cpu_cores_effective=64, cpu_ghz=4.0, dph_total=0.02)


@pytest.fixture(autouse=True)
def _table(monkeypatch):
    """Pin the calibration so a test never depends on the tracked table's
    contents — that file changes every time a box is rented."""
    monkeypatch.setattr(market_offers, "cpu_calibration",
                        lambda *a, **k: _TABLE)


def _ns(**kw):
    base = dict(cpu_cores=0, cpu_ghz=0, host_ram=0, json=False, type="bid",
                any_cpu=False, min_cpu_perf=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _run(offers, monkeypatch, capsys, **kw):
    monkeypatch.setattr(market_offers, "search_offers", lambda a: list(offers))
    cli_search.run(_ns(**kw))
    return capsys.readouterr().out


def test_a_search_with_no_cpu_ask_is_left_exactly_alone(monkeypatch, capsys):
    """Price-ascending is the default order and every other lane depends on it.
    No CPU bracket, no floor, no reordering."""
    out = _run([SLOW, FAST], monkeypatch, capsys)
    assert "[cpu" not in out and "floor" not in out
    assert out.splitlines()[0].startswith("2 offers (bid):")


def test_measured_offers_rank_above_unmeasured_ones(monkeypatch, capsys):
    """A measured rate and a GHz*cores prior are different units, so they cannot
    share a sort key. Unmeasured sorts LAST rather than as zero."""
    out = _run([UNKNOWN, FAST], monkeypatch, capsys, cpu_cores=4)
    body = [ln for ln in out.splitlines() if "[cpu" in ln]
    assert "model" in body[0] and "UNMEASURED" in body[1]


def test_the_bracket_always_names_the_tier_it_came_from(monkeypatch, capsys):
    """A measured rate and a prior must never look alike on the page."""
    out = _run([FAST, UNKNOWN], monkeypatch, capsys, cpu_cores=4)
    assert "model:n=3,spread=1.2x" in out
    assert "UNMEASURED score=" in out


def test_the_default_floor_drops_a_measured_slow_offer(monkeypatch, capsys):
    out = _run([FAST, SLOW], monkeypatch, capsys, cpu_cores=4)
    assert "1 offers" in out
    assert "Slow CPU" in out and "dropped 1 MEASURED-slow" in out


def test_the_floor_keeps_an_unmeasured_offer_however_cheap(monkeypatch, capsys):
    """The whole safety property: unknown is not bad. Were this to fail, a
    CPU-shaped search would return a near-empty board."""
    out = _run([UNKNOWN], monkeypatch, capsys, cpu_cores=4)
    assert "1 offers" in out and "dropped 0 MEASURED-slow" in out


def test_any_cpu_disarms_the_floor(monkeypatch, capsys):
    """The escape hatch that matters: SLOW is 256 threads at $0.01 and wins on
    throughput per dollar outright. The floor refuses it on single-compile
    latency, and this is how a caller overrules that."""
    out = _run([FAST, SLOW], monkeypatch, capsys, cpu_cores=4, any_cpu=True)
    assert "2 offers" in out and "DISARMED" in out
    # and it ranks FIRST: 2.56e10 work/$ against the fast box's 6.4e8. This is
    # the tension the floor is deliberately accepting, made visible.
    rows = [ln for ln in out.splitlines() if "[cpu" in ln]
    assert "m=22" in rows[0] and "m=11" in rows[1]


def test_min_cpu_perf_retunes_the_floor(monkeypatch, capsys):
    """A ratio of the fleet median, so a caller raises the bar in the same unit
    the table is written in."""
    kept = _run([FAST, SLOW], monkeypatch, capsys, cpu_cores=4,
                min_cpu_perf=0.1)
    assert "2 offers" in kept
    strict = _run([FAST, SLOW], monkeypatch, capsys, cpu_cores=4,
                  min_cpu_perf=3.0)                # above FAST's 8.0e6/4.0e6=2x
    assert "0 offers" in strict and "dropped 2" in strict


def test_the_note_says_zero_when_nothing_was_dropped(monkeypatch, capsys):
    """Silence would read as "nothing was rejected" whether or not that was
    true, and the count is the only thing that distinguishes them."""
    out = _run([FAST], monkeypatch, capsys, cpu_cores=4)
    assert "dropped 0 MEASURED-slow offer(s)" in out


def test_with_no_calibration_the_lane_says_so_and_drops_nothing(monkeypatch,
                                                                capsys):
    """Degrading to the old GHz*cores prior is fine; degrading SILENTLY is not,
    because the output would otherwise look identical to a measured ranking."""
    monkeypatch.setattr(market_offers, "cpu_calibration", lambda *a, **k: None)
    out = _run([FAST, SLOW], monkeypatch, capsys, cpu_cores=4)
    assert "2 offers" in out
    assert "no calibration table" in out and "BLIND TO IPC" in out


def test_json_output_is_the_offers_not_the_commentary(monkeypatch, capsys):
    """`--json` is a machine surface; the floor still applies, but nothing may
    be printed alongside the array."""
    out = _run([FAST, SLOW], monkeypatch, capsys, cpu_cores=4, json=True)
    assert out.lstrip().startswith("[") and "note:" not in out
