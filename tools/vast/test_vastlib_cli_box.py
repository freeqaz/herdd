"""`herdd box` — the one-call per-box view (cli/box.py).

What these tests pin: the dict is the contract (human lines render over the
same dict `--json` prints), the watch row matches by current iid AND by the
original watch key so an old id still answers after a handoff, degradation is
soft when fleetd is absent, and only gone-AND-unwatched exits 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vastlib.boxes import health, lifecycle, reap  # noqa: E402
from vastlib.cli import box as cli_box  # noqa: E402
from vastlib.fleet import client  # noqa: E402
from vastlib.jobs import view as jobs_view  # noqa: E402

_BOXES = [
    {"id": 48671690, "actual_status": "running", "num_gpus": 1,
     "gpu_name": "H200", "gpu_util": 0.0, "dph_total": 1.1415,
     "min_bid": 1.1415, "is_bid": True, "label": "upstream-monorepo",
     "ssh_host": "h1", "ssh_port": 2222, "disk_space": 50.0,
     "disk_usage": 0.0, "image_uuid": "registry.example.com/train:t215-latest"},
    {"id": 47, "actual_status": "exited", "num_gpus": 1, "gpu_name": "X",
     "dph_total": 0.1, "storage_total_cost": 0.05},
]

_STATUS = {
    "rows": [{"iid": "48671690", "target": "48670932", "profile": "jobs",
              "state": "watched", "spend_usd": 0.067, "budget_usd": 4.0,
              "remaining_usd": 3.933, "last_action": "tick"}],
}


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(lifecycle, "_instances", lambda: _BOXES)
    monkeypatch.setattr(jobs_view, "_fold_fleet_jobs", lambda live: {
        "48671690": [{"job_id": "20260824T074105-screen-v1-mint-0520",
                      "name": "screen-v1-mint", "status": "running",
                      "display_status": "running", "instance_id": 48671690}]})
    monkeypatch.setattr(reap, "_idle_secs_map", lambda ins, live: {})
    # keep the test off-network: health classification is its own suite's job
    monkeypatch.setattr(health, "gather_fleet_health",
                        lambda ins, jobs, **k: {})
    monkeypatch.setattr(client, "fleet_sock_path", lambda: "/nonexistent-sock")
    monkeypatch.setattr(client, "fleet_state_dir", lambda: "/nonexistent-dir")


def _wire_fleetd(monkeypatch, payload=_STATUS):
    monkeypatch.setattr(client, "fleet_sock_path", lambda: "/dev/null")
    monkeypatch.setattr(client, "fleet_request",
                        lambda op, **k: (True, payload, None))


def test_box_json_is_the_whole_answer(wired, monkeypatch, capsys):
    _wire_fleetd(monkeypatch)
    cli_box.run(argparse.Namespace(id="48671690", json=True))
    d = json.loads(capsys.readouterr().out)
    assert d["found"] is True
    assert d["box"]["state"] == "active"           # live + running job
    assert d["box"]["mode"] == "spot"
    assert d["box"]["ssh"] == "h1:2222"
    assert d["jobs"][0]["cell"].startswith("screen-v1-mint:running")
    assert "last_tail" not in json.dumps(d)        # raw tails never leave the fold
    w = d["watch"]
    assert w["watch"] == "48670932" and w["handed_off"] is True
    assert w["budget_usd"] == 4.0 and w["remaining_usd"] == 3.933


def test_box_state_reflects_the_active_job(wired, monkeypatch, capsys):
    _wire_fleetd(monkeypatch)
    cli_box.run(argparse.Namespace(id="48671690", json=True))
    d = json.loads(capsys.readouterr().out)
    assert d["box"]["state"] == "active"


def test_box_human_lines_carry_the_handoff(wired, monkeypatch, capsys):
    _wire_fleetd(monkeypatch)
    cli_box.run(argparse.Namespace(id="48671690", json=False))
    out = capsys.readouterr().out
    assert out.startswith("box 48671690: active (running)")
    assert "watch 48670932: watched" in out
    assert "$4.000" in out and "$3.933" in out
    assert "this box replaced 48670932" in out


def test_box_answers_for_the_replaced_original_id(wired, monkeypatch, capsys):
    """`box <old-id>` after a handoff: no instance, but the watch row still
    matches by key and names the current box — exit 0, it IS the answer."""
    _wire_fleetd(monkeypatch)
    cli_box.run(argparse.Namespace(id="48670932", json=False))
    out = capsys.readouterr().out
    assert "NO instance" in out
    assert "REPLACED — the current box is 48671690" in out


def test_box_unknown_id_exits_2(wired, monkeypatch, capsys):
    _wire_fleetd(monkeypatch)
    with pytest.raises(SystemExit) as ei:
        cli_box.run(argparse.Namespace(id="99", json=True))
    assert ei.value.code == 2
    d = json.loads(capsys.readouterr().out)     # the dict still printed first
    assert d["found"] is False and d["watch"] is None


def test_box_degrades_soft_without_fleetd(wired, capsys):
    cli_box.run(argparse.Namespace(id="48671690", json=True))
    d = json.loads(capsys.readouterr().out)
    assert d["found"] is True
    assert d["watch"] is None
    assert d["watch_note"] == "fleetd not installed"
