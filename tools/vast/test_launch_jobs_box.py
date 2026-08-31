"""test_launch_jobs_box.py — portable-lane tests for launch_jobs_box.sh.

Every test here exercises the checks that run BEFORE the script touches B2 or
the vast API, so the whole file is offline and costs nothing. That is not a
testing convenience — it is the property under test. The script's reason to
exist is that the hand-run launch order rents a box and THEN discovers the
bundle is unshippable, so "the refusals happen before any spend" is the
behaviour, and these tests pin it.
"""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "launch_jobs_box.sh")

# Fake, but PRESENT: the script prefers an already-exported environment over the
# repo .env precisely so a caller can pin these. If it ever went back to
# sourcing .env unconditionally, these tests would start reading the real
# bucket — and the `test_env_file_does_not_clobber` case below fails loudly.
FAKE_ENV = {"VASTAI_API_KEY": "test-not-a-real-key",
            "B2_BUCKET": "test-not-a-real-bucket"}

MINIMAL_CONFIG = """version: 1
name: synthetic-test-bundle
entrypoint: run.sh
timeout_s: 600
needs:
  gpu: true
  gpus: "all"
  gpu_ram_gb: 48
env:
  GRAD_ACCUM: "32"
  MODE: "autotune"
"""


def run(args, env=None, cwd=None):
    e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
         "HOME": os.environ.get("HOME", "/tmp")}
    e.update(FAKE_ENV)
    e.update(env or {})
    return subprocess.run(["bash", SCRIPT, *args], capture_output=True,
                          text=True, env=e, cwd=cwd, timeout=120)


@pytest.fixture
def bundle(tmp_path):
    """A whole-box bundle with GRAD_ACCUM 32 and no assets/tracks, so nothing
    in the local-check path wants a network."""
    d = tmp_path / "synthetic"
    d.mkdir()
    (d / "job-config.yaml").write_text(MINIMAL_CONFIG)
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    return str(d)


def test_missing_bundle_dir_refuses():
    r = run(["/nonexistent/bundle", "--dry-run"])
    assert r.returncode != 0
    assert "no such bundle dir" in r.stderr


def test_a_dir_that_is_not_a_bundle_refuses(tmp_path):
    (tmp_path / "empty").mkdir()
    r = run([str(tmp_path / "empty"), "--dry-run"])
    assert r.returncode != 0
    assert "neither matrix.py nor job-config.yaml" in r.stderr


def test_bad_instance_type_refuses(bundle):
    r = run([bundle, "--type", "preemptible", "--dry-run"])
    assert r.returncode != 0
    assert "--type must be ondemand|bid" in r.stderr


def test_unset_b2_bucket_refuses(bundle):
    """B2_BUCKET missing does not FAIL the tracks preflight — it makes it print
    'UNVERIFIED ... proceeding without the check' and pass. So an unset bucket
    silently downgrades the one gate that stops a stale trainer reaching a
    rented box (verified 2026-08-04: a bare `. .env`, which sets shell vars
    without exporting them, produced exactly that note). Refuse instead.

    LAUNCH_JOBS_NO_ENV_FILE pins the no-credentials case: the repo .env would
    otherwise supply the bucket, and its path is script-relative so `cd` cannot
    opt out.
    """
    r = run([bundle, "--dry-run"],
            env={"B2_BUCKET": "", "LAUNCH_JOBS_NO_ENV_FILE": "1"})
    assert r.returncode != 0
    assert "B2_BUCKET unset" in r.stderr
    assert "silently skip" in r.stderr


def test_unset_api_key_refuses(bundle):
    r = run([bundle, "--dry-run"],
            env={"VASTAI_API_KEY": "", "LAUNCH_JOBS_NO_ENV_FILE": "1"})
    assert r.returncode != 0
    assert "VASTAI_API_KEY unset" in r.stderr


def test_indivisible_card_count_refuses_before_touching_the_network(bundle):
    """A whole-box bundle's card count IS its DDP world size, and the on-box
    planner exits 12 rather than round the effective batch. Catching it here
    turns a rented-box failure into a free one.
    """
    r = run([bundle, "--num-gpus", "3", "--dry-run"])
    assert r.returncode != 0
    assert "does not divide the bundle's GRAD_ACCUM 32" in r.stderr
    # the point of the ordering: we never got as far as the B2 preflight
    assert "preflight" not in r.stdout


@pytest.mark.parametrize("gpus,per_rank", [("1", 32), ("2", 16), ("4", 8), ("8", 4)])
def test_divisible_card_counts_report_the_rebalance(bundle, gpus, per_rank):
    r = run([bundle, "--num-gpus", gpus, "--dry-run"])
    assert f"grad-accum 32 -> {per_rank}/rank" in r.stdout
    assert "eff-batch held" in r.stdout


def test_only_is_rejected_on_a_single_arm_bundle(bundle):
    r = run([bundle, "--only", "asm", "--dry-run"])
    assert r.returncode != 0
    assert "--only applies to a matrix bundle" in r.stderr


def test_gpu_ram_floor_is_taken_from_the_bundle(bundle):
    """Sizing the card by hand is how a bundle ends up on a box too small for
    it; the bundle already declares the floor, so read it."""
    r = run([bundle, "--dry-run"])
    assert "--gpu-ram 48" in r.stdout


def test_dry_run_plans_all_three_steps_and_rents_nothing(bundle):
    r = run([bundle, "--dry-run", "--budget", "7"])
    assert r.returncode == 0
    assert "would launch" in r.stdout and "--jobs" in r.stdout
    assert "--fleet-watch" in r.stdout          # no launch->watch gap
    assert "would submit" in r.stdout
    assert "fleet watch <IID> --profile jobs --budget 7" in r.stdout
    assert "--standing" in r.stdout             # FLEET_REVIEW item 2: default on
    assert "launched instance" not in r.stdout  # nothing was rented


def test_standing_watch_is_the_default(bundle):
    """FLEET_REVIEW_2026-08-20 item 2: 119 journaled LAPSED cycles came from a
    watch that just ended on queue drain, leaving the ceiling behind with no
    armed ladder. --standing (FLEETD_DESIGN.md §4a-i) must ship on by default."""
    r = run([bundle, "--dry-run"])
    watch = [ln for ln in r.stdout.splitlines() if "would watch" in ln][0]
    assert "--standing" in watch


def test_standing_watch_opt_out(bundle):
    r = run([bundle, "--dry-run"], env={"STANDING_WATCH": "0"})
    watch = [ln for ln in r.stdout.splitlines() if "would watch" in ln][0]
    assert "--standing" not in watch


def test_env_pins_reach_the_submit_command(bundle):
    """`--env K=V` must land in the SUBMIT argv, and therefore in the step-1
    preflight as well as the real submit — the whole point of the flag is to run
    a bundle in a submit-time variant (the padfree fit probe alone,
    FIT_PROBE_ONLY=1) without minting a new content-addressed bundle id."""
    r = run([bundle, "--dry-run", "--env", "FIT_PROBE_ONLY=1", "--env", "BENCH_GEMM=0"])
    assert r.returncode == 0
    submit = [ln for ln in r.stdout.splitlines() if "would submit" in ln][0]
    assert "--env FIT_PROBE_ONLY=1" in submit
    assert "--env BENCH_GEMM=0" in submit          # repeatable


def test_artifact_pins_reach_the_submit_command(bundle):
    """`--artifact PREFIX=SLUG` is the whole point of the registry: a
    `${PREFIX_B2}` asset prefix resolves from committed JSON instead of a
    hand-typed B2 path. Without this passthrough the one-command wrapper the
    runbook names forced the raw --env escape hatch."""
    r = run([bundle, "--dry-run", "--artifact", "ADAPTER=mergeddemoa",
             "--artifact", "BASE=qwen36-27b"])
    assert r.returncode == 0
    submit = [ln for ln in r.stdout.splitlines() if "would submit" in ln][0]
    assert "--artifact ADAPTER=mergeddemoa" in submit
    assert "--artifact BASE=qwen36-27b" in submit      # repeatable


def test_artifact_folds_before_env_so_a_raw_pin_still_wins(bundle):
    """`job submit` folds --artifact first and --env second, and the raw pin is
    the documented escape hatch. The wrapper must not invert that order."""
    r = run([bundle, "--dry-run", "--env", "ADAPTER_B2=checkpoints/hand-typed",
             "--artifact", "ADAPTER=mergeddemoa"])
    submit = [ln for ln in r.stdout.splitlines() if "would submit" in ln][0]
    assert submit.index("--artifact") < submit.index("--env ADAPTER_B2")


def test_artifact_is_rejected_on_a_matrix_bundle(tmp_path):
    """`jobmatrix.py submit` has no --artifact, so accepting it here would drop
    the registry-composed prefixes and launch every arm against whatever the
    bundle's `env:` already said — the silent-wrong-experiment shape --env is
    refused for."""
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl"}})
    (tmp_path / "synthetic-matrix" / "data").mkdir(exist_ok=True)
    (tmp_path / "synthetic-matrix" / "data" / "a.jsonl").write_text("{}\n")
    r = run([str(d), "--artifact", "ADAPTER=mergeddemoa", "--dry-run"])
    assert r.returncode != 0
    assert "--artifact applies to a single-arm bundle" in r.stderr


def test_env_is_rejected_on_a_matrix_bundle(tmp_path):
    """A matrix arm's env comes from matrix.py and `jobmatrix.py submit` has no
    --env at all, so accepting the flag there would drop it silently and launch
    the wrong experiment on a rented box. Refuse instead."""
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl"}})
    (tmp_path / "synthetic-matrix" / "data").mkdir(exist_ok=True)
    (tmp_path / "synthetic-matrix" / "data" / "a.jsonl").write_text("{}\n")
    r = run([str(d), "--env", "FIT_PROBE_ONLY=1", "--dry-run"])
    assert r.returncode != 0
    assert "--env applies to a single-arm bundle" in r.stderr


def test_the_watch_is_armed_after_the_submit_not_before(bundle):
    """A `jobs` watch parks the box as soon as every ticket it can see is
    terminal, so arming it before the tickets exist parks a box you just
    rented (box 46648873, 2026-08-03). Pin the order in the planned output.
    """
    r = run([bundle, "--dry-run"])
    assert r.stdout.index("would submit") < r.stdout.index("would watch")


def test_the_launch_step_suppresses_the_hand_run_watch_hint():
    """`launch --jobs` nudges the operator to arm the ladder, because on the
    hand-run path nobody else will. Here step 4 IS that command, so the nudge
    would be an alarm firing on the workflow the docs recommend. Read from the
    source: the dry-run path never reaches the rent step."""
    src = open(SCRIPT).read()
    rent = src.index('LAUNCH_OUT="$(')
    assert "HERDD_WATCH_HINT=0" in src[rent:src.index("\n", rent)], (
        "the rent step must suppress the hint it is about to satisfy")


def test_env_file_does_not_clobber_an_exported_environment(bundle):
    """The script reads the repo .env only when the environment is incomplete.
    If that regressed to an unconditional source, this run (executed from the
    repo, with a fake bucket exported) would pick up the REAL bucket."""
    r = run([bundle, "--dry-run"], cwd=HERE)
    assert r.returncode == 0
    assert "example-runs-bucket" not in r.stdout


# ---------------------------------------------------------------------------
# STEP 1a-bis: the corpus-presence/sha guard. The corpora are gitignored by
# design and copied into data/ at submit time, so "present in git" is no
# evidence — a worktree-submitted bundle shipped NO corpora on 2026-08-06 and
# the arm died at rc=13 AFTER the box rented (doc 79 §6). These tests pin that
# the refusal happens here, before any spend, on BOTH submit surfaces.
# ---------------------------------------------------------------------------
import hashlib  # noqa: E402

CORPUS_BYTES = b'{"prompt": "p", "completion": "c"}\n'
CORPUS_SHA = hashlib.sha256(CORPUS_BYTES).hexdigest()


def _matrix_bundle(tmp_path, arm_envs):
    """A matrix bundle whose `arm` axis carries the given {key: env} variants.
    Mirrors the real q6-round1-arms shape: whole-box, GRAD_ACCUM 32."""
    d = tmp_path / "synthetic-matrix"
    d.mkdir()
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    # Real matrix bundles ship a job-config.yaml beside matrix.py (rehearse.sh
    # and STEP 1a's box sizing read it); the SUBMIT surface is still matrix.py.
    (d / "job-config.yaml").write_text(MINIMAL_CONFIG)
    (d / "data").mkdir()
    axes = ",\n            ".join(
        f"{k!r}: Variant(env={env!r})" for k, env in arm_envs.items())
    (d / "matrix.py").write_text(f"""\
from jobmatrix import Experiment, Variant

EXPERIMENT = Experiment(
    name="synthetic-matrix",
    entrypoint="run.sh",
    timeout_s=600,
    needs={{"gpu": True, "gpus": "all", "gpu_ram_gb": 48}},
    env={{"GRAD_ACCUM": "32", "MODE": "autotune"}},
    axes={{
        "arm": {{
            {axes}
        }},
    }},
)
""")
    return d


def test_matrix_bundle_with_absent_corpus_refuses_before_renting(tmp_path):
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl",
                                        "EXPECT_SHA256": CORPUS_SHA}})
    r = run([str(d), "--dry-run"])
    assert r.returncode != 0
    assert "data/a.jsonl is ABSENT" in r.stderr
    assert "nothing rented" in r.stderr
    assert "would launch" not in r.stdout      # never got to the plan


def test_matrix_bundle_with_sha_mismatch_refuses(tmp_path):
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl",
                                        "EXPECT_SHA256": "0" * 64}})
    (d / "data" / "a.jsonl").write_bytes(CORPUS_BYTES)
    r = run([str(d), "--dry-run"])
    assert r.returncode != 0
    assert "sha256" in r.stderr and "pins" in r.stderr
    assert "nothing rented" in r.stderr


def test_matrix_bundle_with_matching_sha_passes_the_guard(tmp_path):
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl",
                                        "EXPECT_SHA256": CORPUS_SHA}})
    (d / "data" / "a.jsonl").write_bytes(CORPUS_BYTES)
    r = run([str(d), "--dry-run"])
    assert "arm corpora: 1 arm(s) for '*': 1 sha-matched" in r.stdout


def test_single_arm_bundle_with_absent_corpus_refuses(bundle):
    """GAP pinned here: the guard used to be wrapped in `if [ -f matrix.py ]`,
    so a single-arm bundle (DATA_FILE in job-config.yaml env, empty data/) was
    entirely unguarded — it rented a box and died at rc=13 on it. The refusal
    must happen at $0, exactly like the matrix path."""
    with open(os.path.join(bundle, "job-config.yaml"), "a") as fh:
        fh.write('  DATA_FILE: "corpus.jsonl"\n'
                 f'  EXPECT_SHA256: "{CORPUS_SHA}"\n')
    r = run([bundle, "--dry-run"])
    assert r.returncode != 0
    assert "data/corpus.jsonl is ABSENT" in r.stderr
    assert "nothing rented" in r.stderr
    assert "would launch" not in r.stdout


def test_single_arm_bundle_with_matching_sha_passes(bundle):
    with open(os.path.join(bundle, "job-config.yaml"), "a") as fh:
        fh.write('  DATA_FILE: "corpus.jsonl"\n'
                 f'  EXPECT_SHA256: "{CORPUS_SHA}"\n')
    os.makedirs(os.path.join(bundle, "data"), exist_ok=True)
    with open(os.path.join(bundle, "data", "corpus.jsonl"), "wb") as fh:
        fh.write(CORPUS_BYTES)
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0
    assert "1 sha-matched" in r.stdout


def test_no_expect_sha_reports_presence_only_not_sha_matched(tmp_path):
    """The old success line said "present and sha-matched" even when no arm
    carried an EXPECT_SHA256 — a reassurance the check never earned."""
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl"}})
    (d / "data" / "a.jsonl").write_bytes(CORPUS_BYTES)
    r = run([str(d), "--dry-run"])
    assert "1 present, no EXPECT_SHA256 (presence-only)" in r.stdout
    assert "sha-matched" not in r.stdout


def test_only_glob_matching_zero_arms_says_nothing_checked(tmp_path):
    """`--only` with a typo'd glob used to print the same reassuring success
    line while checking nothing. The dry-run submit still dies on it later, so
    a note is enough — but it must say NOTHING was checked."""
    d = _matrix_bundle(tmp_path, {"a": {"DATA_FILE": "a.jsonl",
                                        "EXPECT_SHA256": CORPUS_SHA}})
    # data/a.jsonl deliberately ABSENT: an unmatched arm must not be checked,
    # and the note must not read like a pass.
    r = run([str(d), "--only", "zzz-*", "--dry-run"])
    assert "0 arms matched 'zzz-*'" in r.stdout
    assert "nothing checked" in r.stdout
    assert "sha-matched" not in r.stdout


def test_arm_without_data_file_is_skipped_with_a_note(bundle):
    """A bundle whose env has no DATA_FILE (the existing minimal fixture) must
    pass the guard and say the skip out loud rather than claim a check ran."""
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0
    assert "1 with no DATA_FILE (skipped)" in r.stdout


@pytest.mark.parametrize("flag", ["--allow-stale-assets", "--no-asset-check"])
def test_asset_policy_flag_reaches_the_submit_command(bundle, flag):
    r = run([bundle, flag, "--dry-run"])
    assert r.returncode == 0
    submit_line = [l for l in r.stdout.splitlines() if "would submit" in l]
    assert submit_line and flag in submit_line[0]


def test_both_asset_policy_flags_are_an_explicit_error(bundle):
    """The two flags used to last-wins into one scalar; they mean opposite
    things about the provenance check, so the combination must refuse."""
    r = run([bundle, "--allow-stale-assets", "--no-asset-check", "--dry-run"])
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr
    # order must not matter
    r2 = run([bundle, "--no-asset-check", "--allow-stale-assets", "--dry-run"])
    assert r2.returncode != 0
    assert "mutually exclusive" in r2.stderr


def test_repeating_the_same_asset_policy_flag_is_not_an_error(bundle):
    r = run([bundle, "--allow-stale-assets", "--allow-stale-assets", "--dry-run"])
    assert r.returncode == 0


# ---------------------------------------------------------------------------
# THE EVAL-ENV PIN. A `needs.venv: eval` bundle grades against the pre-baked
# eval env, and jobd's `check_venv eval` -> onstart/fetch_eval_env.sh picks the
# tarball from the CONTAINER env at provision time — before the entrypoint
# subshell where the job's own `.job.env` is sourced. So only the BOX launch env
# steers the fetch, and this script had no way to set one: it built
# `herdd launch --jobs ...` with no --env passthrough at all.
#
# Measured 2026-08-09: `eval-env/LATEST` was 20260807-0503-84d35a08 while
# q6-round1-evals pinned 20260806-2152-76cd109a, so a box rented here
# provisioned a different rb3-xenon tree than the fixture's spans were resolved
# against and the job died rc 6 at S0.b2/S0.c — after the boot and the ~15 GB
# base-model pull. These pin the injection, and that it fails CLOSED.
# ---------------------------------------------------------------------------

EVAL_ENV_VER = "20260806-2152-76cd109a"

EVAL_CONFIG = """version: 1
name: synthetic-eval-bundle
entrypoint: run.sh
timeout_s: 600
needs:
  gpu: true
  gpus: 1
  gpu_ram_gb: 24
  venv: eval
env:
  EVAL_ENV_VER: "%s"
""" % EVAL_ENV_VER

EVAL_CONFIG_UNPINNED = "\n".join(
    l for l in EVAL_CONFIG.splitlines() if "EVAL_ENV_VER" not in l) + "\n"


def _eval_bundle(tmp_path, config=EVAL_CONFIG, name="synthetic-eval"):
    d = tmp_path / name
    d.mkdir()
    (d / "job-config.yaml").write_text(config)
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    return str(d)


def _line(out, needle):
    hits = [l for l in out.splitlines() if needle in l]
    assert hits, f"no {needle!r} line in:\n{out}"
    return hits[0]


def test_eval_bundle_pin_is_read_from_the_bundle_and_injected_into_the_box(tmp_path):
    """The pin is never retyped: it comes out of the bundle and goes onto the
    box as EVAL_ENV_VER, which is the only copy fetch_eval_env.sh can see."""
    r = run([_eval_bundle(tmp_path), "--num-gpus", "1", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert f"eval-env pin: {EVAL_ENV_VER} (from bundle job-config env:)" in r.stdout
    assert f"--env EVAL_ENV_VER={EVAL_ENV_VER}" in _line(r.stdout, "would launch")


def test_eval_bundle_submit_asserts_the_pin_actually_landed(tmp_path):
    """Injecting is not the same as landing. The submit re-reads the instance's
    extra_env; --require-box-eval-pin turns the M4 gate's job-pin-only NOTE
    into a refusal, which is correct for a box we rented seconds ago."""
    r = run([_eval_bundle(tmp_path), "--num-gpus", "1", "--dry-run"])
    assert "--require-box-eval-pin" in _line(r.stdout, "would submit")


def test_a_non_eval_bundle_gets_no_pin_and_no_assertion(bundle):
    """`needs.venv` is not `eval` -> there is no baked env to steer. Injecting
    an EVAL_ENV_VER there would be cargo cult."""
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0
    assert "EVAL_ENV_VER" not in _line(r.stdout, "would launch")
    assert "--require-box-eval-pin" not in _line(r.stdout, "would submit")
    assert "eval-env pin:" not in r.stdout


def test_eval_bundle_with_no_pin_anywhere_refuses_before_renting(tmp_path):
    """FAIL CLOSED. An unpinned box resolves eval-env/LATEST at boot, which can
    be older OR newer than what the bundle was preflighted against. `herdd
    job submit` refuses this too — but in STEP 3, one step after the rent."""
    d = _eval_bundle(tmp_path, EVAL_CONFIG_UNPINNED, name="unpinned")
    r = run([d, "--num-gpus", "1", "--dry-run"])
    assert r.returncode != 0
    assert "declares needs.venv: eval but names no EVAL_ENV_VER" in r.stderr
    assert "would launch" not in r.stdout          # nothing was planned, let alone rented


def test_eval_env_ver_flag_supplies_a_pin_the_bundle_lacks(tmp_path):
    """...and it must reach the TICKET as well as the box: STEP 1b preflights
    against placeholder box 0, whose launch env is unreadable by construction,
    so a box-only pin would be invisible to the M4 gate and refused."""
    d = _eval_bundle(tmp_path, EVAL_CONFIG_UNPINNED, name="unpinned2")
    r = run([d, "--num-gpus", "1", "--eval-env-ver", EVAL_ENV_VER, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert f"eval-env pin: {EVAL_ENV_VER} (from --eval-env-ver)" in r.stdout
    assert f"--env EVAL_ENV_VER={EVAL_ENV_VER}" in _line(r.stdout, "would launch")
    assert f"--env EVAL_ENV_VER={EVAL_ENV_VER}" in _line(r.stdout, "would submit")


def test_eval_env_ver_flag_conflicting_with_the_bundle_refuses(tmp_path):
    """The box would fetch one version while the job asserts another. The M4
    gate calls that EVAL_ENV_VER CONFLICT and refuses — after the rent. Here."""
    r = run([_eval_bundle(tmp_path), "--num-gpus", "1",
             "--eval-env-ver", "20260807-0503-84d35a08", "--dry-run"])
    assert r.returncode != 0
    assert "conflicts with the bundle's own pin" in r.stderr
    assert "would launch" not in r.stdout


def test_restating_the_bundle_pin_on_the_flag_is_accepted(tmp_path):
    """Passing the version the bundle already names is legal and is NOT a
    conflict. It re-states the same value on the submit as an override, which
    is a no-op the ticket cannot tell from the bundle's own — deliberately: the
    alternative is a special case whose only job is to save one argv entry."""
    r = run([_eval_bundle(tmp_path), "--num-gpus", "1",
             "--eval-env-ver", EVAL_ENV_VER, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "conflicts with" not in r.stderr
    assert f"eval-env pin: {EVAL_ENV_VER} (from --eval-env-ver)" in r.stdout
    assert f"--env EVAL_ENV_VER={EVAL_ENV_VER}" in _line(r.stdout, "would launch")


def test_machine_pin_does_not_also_pin_the_default_card(bundle):
    """`--machine <ID> --gpu <name>` is an AND of two filters, and a machine
    picked for CPU cores or VRAM usually holds some other card — so appending
    the default GPU to a machine pin intersects to zero offers ("no offers
    match filters") on a host that was right all along."""
    r = run([bundle, "--machine", "34898", "--dry-run"])
    assert "--machine 34898" in r.stdout
    assert "--gpu " not in _line(r.stdout, "would launch")
    assert "--any-gpu" in r.stdout


def test_an_explicit_card_still_narrows_a_machine_pin(bundle):
    """The widening is a DEFAULT, not a policy: a caller naming both means it."""
    r = run([bundle, "--machine", "34898", "--gpu", "h100", "--dry-run"])
    assert "--machine 34898" in r.stdout
    assert "--gpu h100" in r.stdout
    assert "--any-gpu" not in r.stdout


# ---------------------------------------------------------------------------
# THE ARCHITECTURE ALLOWLIST (`needs.cc_allow` -> `--cc-allow`). The bundle
# knows which silicon its kernels exist for; before this, the constraint was
# something a human had to remember to type at launch, about a property of the
# bundle, on every launch and every relaunch. It failed that way three times
# (k4 2026-08-17, pk2's launch and both of its eviction replacements
# 2026-08-18/19) — each misfire an FA2 bundle born on sm_120, where the baked
# flash_attn has no kernel image.
#
# The launch flag is what stamps LAUNCH_CC_ALLOW, which is what fleetd's
# replacement lane inherits, so this is the first of the three links in
#   bundle declares -> launch stamps -> replacement inherits.
# ---------------------------------------------------------------------------

CC_CONFIG = MINIMAL_CONFIG.replace(
    "  gpu_ram_gb: 48\n", "  gpu_ram_gb: 48\n  cc_allow: [80, 86, 89, 90]\n")


@pytest.fixture
def cc_bundle(tmp_path):
    d = tmp_path / "synthetic-cc"
    d.mkdir()
    (d / "job-config.yaml").write_text(CC_CONFIG)
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    return str(d)


def test_a_declared_allowlist_reaches_the_launch(cc_bundle):
    r = run([cc_bundle, "--num-gpus", "1", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "arch allowlist: sm 80,86,89,90 (from bundle needs.cc_allow)" in r.stdout
    assert "--cc-allow 80,86,89,90" in _line(r.stdout, "would launch")


def test_a_declared_allowlist_drops_the_default_card_name(cc_bundle):
    """THE INTERACTION. The default card is h100 (sm_90) and the allowlist is a
    second server-side filter, so a bundle whose list excludes the default would
    search for an impossible offer and get "no offers match filters" — a thin
    market for what is really two filters that cannot both hold. Same shape as
    the --machine widening: the narrowing the caller DID ask for stands, the
    default one does not."""
    r = run([cc_bundle, "--num-gpus", "1", "--dry-run"])
    launch = _line(r.stdout, "would launch")
    assert "--cc-allow" in launch
    assert "--gpu " not in launch          # --gpu-ram survives; --gpu does not
    assert "--any-gpu" not in launch       # the tier policy stays ON
    assert "the sm allowlist (80,86,89,90) picks the architecture" in r.stdout


def test_no_declaration_means_no_filter_at_all(bundle):
    """The regression guard. Absent `needs.cc_allow` must be EXACTLY today's
    behaviour — unconstrained, default card, no --cc-allow anywhere — never an
    allowlist of nothing."""
    r = run([bundle, "--num-gpus", "1", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--cc-allow" not in r.stdout
    assert "arch allowlist" not in r.stdout
    assert "--gpu h100" in _line(r.stdout, "would launch")


def test_an_empty_declaration_is_also_unconstrained(tmp_path):
    d = tmp_path / "synthetic-cc-empty"
    d.mkdir()
    (d / "job-config.yaml").write_text(
        MINIMAL_CONFIG.replace("  gpu_ram_gb: 48\n",
                               "  gpu_ram_gb: 48\n  cc_allow: []\n"))
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    r = run([str(d), "--num-gpus", "1", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--cc-allow" not in r.stdout
    assert "--gpu h100" in _line(r.stdout, "would launch")


def test_an_explicit_flag_overrides_the_bundle_and_says_so(cc_bundle):
    """A flag may override the declaration — but SILENTLY overriding a declared
    device constraint is the failure this whole seam exists to stop, so it is
    announced, with the reason the declaration is the durable form."""
    r = run([cc_bundle, "--num-gpus", "1", "--cc-allow", "90", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "OVERRIDES" in r.stdout and "needs.cc_allow" in r.stdout
    assert "arch allowlist: sm 90 (from --cc-allow)" in r.stdout
    assert "--cc-allow 90" in _line(r.stdout, "would launch")


def test_restating_the_declared_allowlist_is_not_reported_as_an_override(cc_bundle):
    r = run([cc_bundle, "--num-gpus", "1", "--cc-allow", "80,86,89,90", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "OVERRIDES" not in r.stdout


def test_a_flag_allowlist_works_without_any_declaration(bundle):
    r = run([bundle, "--num-gpus", "1", "--cc-allow", "90", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "arch allowlist: sm 90 (from --cc-allow)" in r.stdout
    assert "--cc-allow 90" in _line(r.stdout, "would launch")


def test_a_card_that_contradicts_the_allowlist_refuses_before_renting(cc_bundle):
    """An explicit --gpu still wins over the default — but a named card OUTSIDE
    the allowlist can never match, so refuse where the operator can see which
    two filters disagree, not after a search that reports an empty market."""
    r = run([cc_bundle, "--num-gpus", "1", "--gpu", "rtxpro6000", "--dry-run"])
    assert r.returncode != 0
    assert "is sm_120" in r.stderr and "EXCLUDES" in r.stderr
    assert "would launch" not in r.stdout
    assert "preflight" not in r.stdout      # refused before even the B2 round-trip


def test_a_card_inside_the_allowlist_is_kept(cc_bundle):
    r = run([cc_bundle, "--num-gpus", "1", "--gpu", "a100", "--dry-run"])
    assert r.returncode == 0, r.stderr
    launch = _line(r.stdout, "would launch")
    assert "--gpu a100" in launch and "--cc-allow 80,86,89,90" in launch


def test_a_card_we_cannot_place_does_not_refuse(cc_bundle):
    """The sm table cannot know every SKU, and a gate that fires on ignorance is
    one nobody reads. An unrecognised name is passed through to the search."""
    r = run([cc_bundle, "--num-gpus", "1", "--gpu", "some-2027-card", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--gpu some-2027-card" in _line(r.stdout, "would launch")


def test_a_malformed_declaration_stops_the_launch(tmp_path):
    """A bundle we cannot read is a STOP, not a shrug — the same rule the rest
    of the needs block follows. Dropping an unparseable constraint would rent a
    box the bundle said it could not use."""
    d = tmp_path / "synthetic-cc-bad"
    d.mkdir()
    (d / "job-config.yaml").write_text(
        MINIMAL_CONFIG.replace("  gpu_ram_gb: 48\n",
                               "  gpu_ram_gb: 48\n  cc_allow: [hopper]\n"))
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    r = run([str(d), "--num-gpus", "1", "--dry-run"])
    assert r.returncode != 0
    assert "refusing to size a box by guess" in r.stderr
    assert "would launch" not in r.stdout


# --- --disk vs the bundle's own estimate ------------------------------------ #
# Until 2026-08-25 an explicit --disk skipped the estimator OUTRIGHT — no
# comparison, no note — so `--disk 40` against a bundle whose merge needs 80
# rented the box, pulled the ~19 GB image, pulled the base, booted vLLM, passed a
# 12/12 positive control and THEN died rc 5 on the bundle's own pre-merge disk
# guard. A running instance's disk cannot be resized, so the box was a total
# loss. The comparison is free and belongs here, where nothing has been rented.

@pytest.fixture
def scratch_bundle(tmp_path):
    """A bundle that declares what its entrypoint writes. No assets, so the
    estimate needs no network: scratch 42 + venv 0 + overhead 12 -> 60G."""
    d = tmp_path / "synthetic-scratch"
    d.mkdir()
    (d / "job-config.yaml").write_text(
        MINIMAL_CONFIG.replace("  gpu_ram_gb: 48\n",
                               "  gpu_ram_gb: 48\n  scratch_gb: 42\n"))
    (d / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    return str(d)


def test_a_disk_below_the_bundles_estimate_refuses_before_renting(scratch_bundle):
    r = run([scratch_bundle, "--disk", "10", "--dry-run"])
    assert r.returncode != 0
    assert "--disk 10 is BELOW" in r.stderr and "60GB" in r.stderr
    assert "would launch" not in r.stdout


def test_the_refusal_names_the_durable_fix_not_just_the_flag(scratch_bundle):
    """A refusal that only says "pass a bigger number" gets a bigger number
    typed at it next time too. The bundle is where the size belongs."""
    r = run([scratch_bundle, "--disk", "10", "--dry-run"])
    assert "needs.scratch_gb" in r.stderr and "needs.disk_gb" in r.stderr


def test_force_disk_rents_under_the_estimate_and_says_so(scratch_bundle):
    """Same escape hatch as --force-gpu-ram: the operator may know the
    derivation is wrong, but the launch must not be silent about it."""
    r = run([scratch_bundle, "--disk", "10", "--force-disk", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--force-disk" in r.stdout
    assert "--disk 10" in _line(r.stdout, "would launch")


def test_a_disk_at_or_above_the_estimate_is_kept(scratch_bundle):
    """Over-provisioning stays an operator's call — the gate is one-directional,
    exactly like the --gpu-ram one."""
    r = run([scratch_bundle, "--disk", "200", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "--disk 200" in _line(r.stdout, "would launch")
    assert "estimate 60GB" in _line(r.stdout, ">> disk:")


def test_the_auto_size_says_what_it_could_not_see(bundle):
    """The default path's label is the whole reason the 40G box looked fine:
    "measured-from-assets" was true and read as complete. A bundle that declares
    no scratch gets told so, next to the number."""
    r = run([bundle, "--dry-run"])
    assert r.returncode == 0, r.stderr
    note = _line(r.stdout, ">> disk:")
    assert "needs.scratch_gb" in note and "entrypoint" in note


def test_the_auto_size_stays_quiet_for_a_bundle_that_declared(scratch_bundle):
    r = run([scratch_bundle, "--dry-run"])
    assert r.returncode == 0, r.stderr
    note = _line(r.stdout, ">> disk:")
    assert "60GB" in note and "no needs.scratch_gb" not in note
