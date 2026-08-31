"""--gpu-ram below the bundle's needs.gpu_ram_gb must be refused BEFORE the rent.

The trap, live on 2026-08-14: `launch_jobs_box.sh` had

    [ -z "$GPU_RAM" ] && GPU_RAM="$NEED_RAM"

so an explicit `--gpu-ram` silently WON — no comparison, no warning. Passing
`--gpu-ram 23` against a bundle declaring `needs.gpu_ram_gb: 30` rented a 24 GB
card whose ticket jobd was guaranteed to refuse (jobd fails a ticket when the
largest card is smaller than the declared need, and a 4090's 24082 MiB rounds
to 24). The box booted, jobd started, the job never ran, the meter did.

What made it invisible is that the two numbers look like one knob and are not:

    --gpu-ram          OFFER SELECTION — a vast search filter, spent once at
                       rent time, workstation-side.
    needs.gpu_ram_gb   the TICKET's CONTRACT — re-checked by jobd against the
                       cards actually present, on the box, after the spend.

Lowering the filter cannot lower the contract, so a below-need `--gpu-ram` buys
a box that will not run the job. These tests pin the refusal, its escape hatch,
and the cases that must stay allowed (equal, over-provisioned, absent).

Offline: every assertion lands before the script touches B2 or the vast API.
(Kept in its own file rather than folded into test_launch_jobs_box.py because a
concurrent session owned that file when this landed — merge them when
convenient; `run`/`FAKE_ENV` are deliberately identical.)
"""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "launch_jobs_box.sh")

FAKE_ENV = {"VASTAI_API_KEY": "test-not-a-real-key",
            "B2_BUCKET": "test-not-a-real-bucket"}

MINIMAL_CONFIG = """version: 1
name: synthetic-test-bundle
entrypoint: run.sh
timeout_s: 600
needs:
  gpu: true
  gpus: "all"
  gpu_ram_gb: 30
env:
  GRAD_ACCUM: "32"
  MODE: "autotune"
"""

NO_NEED_CONFIG = MINIMAL_CONFIG.replace("  gpu_ram_gb: 30\n", "")


def run(args, env=None):
    e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
         "HOME": os.environ.get("HOME", "/tmp")}
    e.update(FAKE_ENV)
    e.update(env or {})
    return subprocess.run(["bash", SCRIPT, *args], capture_output=True,
                          text=True, env=e, timeout=120)


def _bundle(tmp_path, cfg=MINIMAL_CONFIG, name="synthetic"):
    d = tmp_path / name
    d.mkdir()
    (d / "job-config.yaml").write_text(cfg)
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    return str(d)


@pytest.fixture
def bundle(tmp_path):
    return _bundle(tmp_path)


def test_gpu_ram_below_the_bundle_need_refuses_before_any_spend(bundle):
    r = run([bundle, "--gpu-ram", "23", "--dry-run"])
    assert r.returncode != 0
    assert "is BELOW" in r.stderr and "needs.gpu_ram_gb: 30" in r.stderr
    # the distinction that made the trap invisible must be IN the message
    assert "OFFER" in r.stderr and "TICKET" in r.stderr
    # and it must land before the B2/asset preflight — i.e. before any spend
    assert "preflight" not in r.stdout
    assert "would launch" not in r.stdout


def test_the_refusal_names_the_escape_hatch(bundle):
    r = run([bundle, "--gpu-ram", "23", "--dry-run"])
    assert "--force-gpu-ram" in r.stderr


def test_force_gpu_ram_proceeds_but_says_what_it_bought(bundle):
    r = run([bundle, "--gpu-ram", "23", "--force-gpu-ram", "--dry-run"])
    assert r.returncode == 0
    assert "--force-gpu-ram" in r.stdout
    assert "buying a box, not a run" in r.stdout
    assert "--gpu-ram 23" in r.stdout          # the override still applies


def test_equal_gpu_ram_is_allowed(bundle):
    r = run([bundle, "--gpu-ram", "30", "--dry-run"])
    assert r.returncode == 0
    assert "is BELOW" not in r.stderr


def test_over_provisioning_is_allowed(bundle):
    """A bigger card always satisfies the ticket — headroom is not a mistake."""
    r = run([bundle, "--gpu-ram", "80", "--dry-run"])
    assert r.returncode == 0
    assert "is BELOW" not in r.stderr
    assert "--gpu-ram 80" in r.stdout


def test_absent_gpu_ram_still_defaults_to_the_bundle(bundle):
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0
    assert "--gpu-ram 30" in r.stdout


def test_a_bundle_that_declares_no_need_cannot_be_compared_against(tmp_path):
    """No declaration is not a declaration of 0: with nothing to compare, the
    explicit flag is the only number there is and must pass through."""
    b = _bundle(tmp_path, NO_NEED_CONFIG, name="no-need")
    r = run([b, "--gpu-ram", "23", "--dry-run"])
    assert r.returncode == 0
    assert "is BELOW" not in r.stderr
    assert "--gpu-ram 23" in r.stdout


def test_the_real_perf_levers_d1_bundle_refuses_the_launch_that_burned_a_box():
    """The live instance, not a synthetic one: perf-levers-d1 declares 30 and a
    `--gpu-ram 23` launch against it is exactly what rented a 24 GB box."""
    repo = os.path.dirname(os.path.dirname(HERE))
    b = os.path.join(repo, "tools", "witness", "jobs", "perf-levers-d1")
    if not os.path.isdir(b):
        pytest.skip("perf-levers-d1 bundle is gone")
    r = run([b, "--gpu-ram", "23", "--dry-run"])
    assert r.returncode != 0
    assert "needs.gpu_ram_gb: 30" in r.stderr
