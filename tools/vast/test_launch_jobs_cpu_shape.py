"""A bundle's CPU-shape needs must reach the offer search, not stop at the ticket.

`needs.cpu_cores` landed 2026-08-21 as SCHEMA ONLY: jobmeta validated it and
wrote it into the ticket, and nothing ever read it back. The launcher's needs
reader is a fixed-width `read -r` of six positional fields and `cpu_cores` was
not one of them, so a bundle could declare 64 cores and still be rented a
4-core slice — the declaration was inert, and inert in the silent direction.

`needs.host_ram_gb` is the axis that actually decides whether a CPU-shaped job
runs at all (a bf16 CPU merge holds the whole base resident), so it is wired
the same way and tested here beside it.

These are OFFER-SELECTION filters with no ticket contract behind them: jobd has
no host-RAM or core gate to refuse against, so unlike `--gpu-ram` there is
nothing to compare an explicit flag to and it simply wins. That asymmetry is
the thing worth pinning — if a box-side gate is ever added, these tests should
start failing and gain a `--force-` guard like the VRAM one has.

Offline: every assertion lands on the `--dry-run` plan, before B2 or the vast
API is touched.
"""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "launch_jobs_box.sh")

FAKE_ENV = {"VASTAI_API_KEY": "test-not-a-real-key",
            "B2_BUCKET": "test-not-a-real-bucket"}

CPU_SHAPED_CONFIG = """version: 1
name: synthetic-cpu-bundle
entrypoint: run.sh
timeout_s: 600
needs:
  gpu: false
  cpu_cores: 64
  host_ram_gb: 96
"""

NO_SHAPE_CONFIG = """version: 1
name: synthetic-plain-bundle
entrypoint: run.sh
timeout_s: 600
needs:
  gpu: false
"""


def run(args, env=None):
    e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
         "HOME": os.environ.get("HOME", "/tmp")}
    e.update(FAKE_ENV)
    e.update(env or {})
    return subprocess.run(["bash", SCRIPT, *args], capture_output=True,
                          text=True, env=e, timeout=120)


def _bundle(tmp_path, cfg=CPU_SHAPED_CONFIG, name="synthetic-cpu"):
    d = tmp_path / name
    d.mkdir()
    (d / "job-config.yaml").write_text(cfg)
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    return str(d)


@pytest.fixture
def bundle(tmp_path):
    return _bundle(tmp_path)


def test_the_bundles_host_ram_need_becomes_a_search_filter(bundle):
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--host-ram 96" in r.stdout


def test_the_bundles_core_count_becomes_a_search_filter(bundle):
    """The regression this file exists for: declared, ticketed, and never read."""
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--cpu-cores 64" in r.stdout


def test_an_explicit_flag_overrides_the_bundle(bundle):
    r = run([bundle, "--host-ram", "252", "--cpu-cores", "128", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--host-ram 252" in r.stdout
    assert "--cpu-cores 128" in r.stdout
    assert "--host-ram 96" not in r.stdout


def test_a_bundle_declaring_neither_asks_for_neither(tmp_path):
    """Absent must stay ABSENT, never become a floor of 0 — a `gte 0` filter is
    a filter, and one that silently narrows nothing today can narrow something
    the day the field's meaning shifts."""
    b = _bundle(tmp_path, NO_SHAPE_CONFIG, name="plain")
    r = run([b, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--host-ram" not in r.stdout
    assert "--cpu-cores" not in r.stdout


def test_the_widened_needs_read_did_not_shift_the_other_fields(tmp_path):
    """`read -r` is POSITIONAL: appending fields to the python `print()` and
    the `read` list in different orders silently rebinds every variable after
    the first mismatch. This asserts the pre-existing fields still land where
    they belong."""
    cfg = """version: 1
name: synthetic-mixed-bundle
entrypoint: run.sh
timeout_s: 600
needs:
  gpu: true
  gpus: 1
  gpu_ram_gb: 48
  venv: serve
  cpu_cores: 32
  host_ram_gb: 128
"""
    b = _bundle(tmp_path, cfg, name="mixed")
    r = run([b, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--gpu-ram 48" in r.stdout          # field 1 still field 1
    assert "--host-ram 128" in r.stdout        # ...and the new ones are 7 and 8
    assert "--cpu-cores 32" in r.stdout
