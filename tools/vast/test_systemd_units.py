"""Portable checks over the shipped systemd unit templates.

These files were installed and running before anything asserted a property of
them, which is how a `Type=oneshot` unit reached the operator's machine with
`TimeoutStartUSec=infinity`.
"""
import configparser
import glob
import os

import pytest

_UNIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "systemd")


def _parse(path):
    # systemd allows repeated keys; configparser does not, and strict=False is
    # what lets a real unit file through unchanged.
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(path, encoding="utf-8")
    return cp


def _units(suffix):
    return sorted(glob.glob(os.path.join(_UNIT_DIR, f"*.{suffix}.in")))


def test_there_are_units_to_check():
    assert _units("service"), "no service templates found — did the dir move?"


@pytest.mark.parametrize("path", _units("service"), ids=os.path.basename)
def test_a_oneshot_service_bounds_its_own_start(path):
    """`Type=oneshot` defaults TimeoutStartSec to infinity.

    A wedged ExecStart then leaves the unit 'activating' forever, and a timer
    cannot trigger a unit that never went inactive — so a single hang silently
    ends every future run rather than failing one.
    """
    svc = _parse(path)["Service"]
    if svc.get("Type") != "oneshot":
        pytest.skip("only oneshot defaults to an unbounded start")
    t = svc.get("TimeoutStartSec", "")
    assert t and t not in ("infinity", "0"), (
        f"{os.path.basename(path)} is Type=oneshot with TimeoutStartSec={t!r} — "
        "a hung run would block all later triggers")


@pytest.mark.parametrize("path", _units("timer"), ids=os.path.basename)
def test_a_timer_measures_its_interval_from_completion(path):
    """OnUnitActiveSec measures from START, so a job whose duration grows
    toward the interval schedules itself back-to-back. Anything that sweeps a
    store must use OnUnitInactiveSec to keep a real gap."""
    tmr = _parse(path)["Timer"]
    if "OnUnitActiveSec" not in tmr:
        return
    assert "OnCalendar" in tmr or "OnUnitInactiveSec" in tmr, (
        f"{os.path.basename(path)} uses OnUnitActiveSec alone; if the run can "
        "approach the interval, use OnUnitInactiveSec")


@pytest.mark.parametrize("path", _units("service") + _units("timer"),
                         ids=os.path.basename)
def test_a_template_carries_no_absolute_machine_path(path):
    """CLAUDE.md: no `/home/<user>/…` in a committed file. That is the whole
    reason these are @DS_ROOT@ templates and not unit files."""
    body = open(path, encoding="utf-8").read()
    assert "/home/" not in body, f"{os.path.basename(path)} hardcodes a home dir"


@pytest.mark.parametrize("path", _units("service"), ids=os.path.basename)
def test_a_service_template_is_rooted_at_the_substitution_token(path):
    svc = _parse(path)["Service"]
    assert svc.get("ExecStart", "").startswith("@DS_ROOT@"), (
        "ExecStart must resolve through the installer's substitution")
