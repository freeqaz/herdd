"""The co-tenant CPU compile-farm is a DEAD FEATURE: off by default, opt-in only.

Owner ruling 2026-08-21. The sidecar's rb3-objcache grew to 69 GB and took a
live serving box to 110/110 GB — one write from killing a serve mid-eval — on
top of the 2026-07-10 finding that it starves a CPU-sensitive train 16x.

These are text pins on the shipped bash surface rather than behavioural runs:
the farm block sits mid-script behind a B2/rclone probe and an `exec`, so the
cheapest honest gate on "did someone flip the default back" is the default
expansion itself. A revert therefore fails here instead of on a rented box.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
ONSTART = os.path.join(_HERE, "onstart")


@pytest.mark.parametrize("script", ["serve_vllm.sh", "train.sh"])
def test_onstart_defaults_cpu_farm_off(script):
    text = open(os.path.join(ONSTART, script), encoding="utf-8").read()
    defaults = re.findall(r'CPU_FARM="\$\{CPU_FARM:-([^}]*)\}"', text)
    assert defaults == ["0"], f"{script}: CPU_FARM default must be 0, got {defaults}"


@pytest.mark.parametrize("script", ["serve_vllm.sh", "train.sh"])
def test_onstart_shouts_when_the_farm_is_opted_into(script):
    """Opting into a dead feature must be loud in the boot log, not silent."""
    text = open(os.path.join(ONSTART, script), encoding="utf-8").read()
    assert "DEAD co-tenant compile farm" in text


def test_launch_serve_has_an_explicit_opt_in_flag_and_keeps_the_old_one():
    text = open(os.path.join(_HERE, "launch_serve.sh"), encoding="utf-8").read()
    assert re.search(r"^\s*--cpu-farm\)\s*CPU_FARM_ON=1;", text, re.M)
    assert re.search(r"^\s*--no-cpu-farm\)\s*CPU_FARM_ON=0;", text, re.M)
    assert re.search(r'^CPU_FARM_ON=0\b', text, re.M)


def test_launch_serve_default_mint_is_the_scoped_pair():
    """Farm-OFF is now the default, so the tighter serve/-scoped write key is
    what a plain `launch_serve.sh` mints; the bucket-wide single key is only
    reachable by asking for the farm. Behaviour is covered end-to-end in
    test_b2_mint_key.py — this pins the branch polarity at the seam."""
    text = open(os.path.join(_HERE, "launch_serve.sh"), encoding="utf-8").read()
    branch = text[text.index('if [ "$CPU_FARM_ON" = "1" ]; then'):]
    branch = branch[:branch.index("\n    fi\n")]
    on_arm, off_arm = branch.split("    else\n", 1)
    assert "mint-pair" not in on_arm and "--write-prefix" not in on_arm
    assert "mint-pair" in off_arm and "--write-prefix serve/" in off_arm
