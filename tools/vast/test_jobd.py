"""End-to-end test for onstart/jobd.sh under a FAKE rclone shim.

No network, no B2, no real vast box: a tiny bash `rclone` shim maps
`b2:<bucket>/<key>` onto files in a tmpdir standing in for the bucket, and
onstart/jobd.sh runs one poll pass (JOBD_ONCE=1) against a trivial entrypoint.
Mirrors the fake-transport discipline of test_supervise.py / test_farm_ingest.py.

Skipped automatically if `bash`/`timeout` aren't available.
"""
import fcntl
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobmeta as jm  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
JOBD_SH = os.path.join(_HERE, "onstart", "jobd.sh")

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("timeout")),
    reason="needs bash + timeout")


# The bash rclone shim (b2:<bucket>/<key> <-> $FAKE_BUCKET/<key> on disk) is the
# shared local-B2 used by rehearse.sh too — single source of truth in testlib/.
_SHIM_PATH = os.path.join(_HERE, "testlib", "rclone_shim.sh")
with open(_SHIM_PATH) as _f:
    _RCLONE_SHIM = _f.read()


# Subprocess envs are built from an ALLOWLIST, never `dict(os.environ)`.
# jobd.sh / jobd_boot.sh read ~70 environment variables (every `JOBD_*`, the
# `B2_*` set, `CRED_*`, `TS_AUTHKEY`, `VASTAI_API_KEY`/`CONTAINER_API_KEY`), and
# the repo `.env` puts real values for several of them into `os.environ` — so a
# wholesale copy (a) lets a `.env` edit change a test with no code change and
# (b) hands the daemon a real self-control key plus a real `$HOME` holding
# `~/.vast_api_key` (jobd.sh's park-key fallback). Only machine-shaped vars come
# from the environment; everything else is what the caller passes.
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")


def _hermetic_env(tmp_path, **overrides):
    """Allowlisted env for a shelled jobd/bootstrap run: PATH + locale from the
    machine, HOME redirected into tmp_path (never the real one — see above),
    plus exactly the caller's overrides."""
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    env.setdefault("PATH", os.defpath)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env.update(overrides)
    return env


def _make_bucket(tmp_path):
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    shimdir = tmp_path / "bin"
    shimdir.mkdir()
    shim = shimdir / "rclone"
    shim.write_text(_RCLONE_SHIM)
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bucket, shimdir


def _authfail_shimdir(tmp_path, shimdir):
    """A PATH dir whose `rclone` wraps the shared shim but AUTH-FAILS the PERIODIC
    checkpoint-sync copy (the only `copy` that passes --min-age into checkpoints/),
    standing in for a dead/rotated B2 key. Scoped that narrowly so event/log
    transport (rcat) and the final publish (no --min-age, into results/) still land
    — a test can thus observe the loud checkpoint_sync_failed event while the job
    still runs. Keeps the auth-failure simulation inside this suite (the shared shim
    is not forked)."""
    d = tmp_path / "authbin"
    d.mkdir()
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        'if [ "${1:-}" = copy ]; then\n'
        '  _mina=0; _res=0\n'
        '  for a in "$@"; do\n'
        '    case "$a" in\n'
        '      --min-age) _mina=1 ;;\n'
        '      */checkpoints/|*/checkpoints) _res=1 ;;\n'
        '    esac\n'
        '  done\n'
        '  if [ "$_mina" = 1 ] && [ "$_res" = 1 ]; then\n'
        '    echo "ERROR: SerializeHTTPError InvalidAccessKeyId: The key '
        "'004deadKEY0000000000003' is not valid\" >&2\n"
        '    exit 1\n'
        '  fi\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _stalecat_shimdir(tmp_path, shimdir, key_suffix, n_stale):
    """A PATH dir whose `rclone` wraps the shared shim but serves an EMPTY body
    for the first `n_stale` `cat` reads of any key ending in `key_suffix` —
    standing in for B2 overwrite eventual-consistency (a stale pre-publish
    version of a results key served for minutes AFTER the final copy landed;
    the 2026-07-15/16/19 validate_generation_artifact false-fails). Each
    intercepted stale read also records into `<counter>.violation` whether
    results.DONE.json was ALREADY visible in the fake bucket: publish verify's
    ordering guarantee is that a stale read is never observable after DONE
    exists. Returns (pathdir, counter_file)."""
    d = tmp_path / "stalebin"
    d.mkdir()
    cnt = d / "catcount"
    cnt.write_text("0")
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        f'CNT={shlex.quote(str(cnt))}\n'
        'if [ "${1:-}" = cat ]; then\n'
        '  case "${2:-}" in\n'
        f'    *{key_suffix})\n'
        '      n=$(cat "$CNT"); echo $((n+1)) > "$CNT"\n'
        f'      if [ "$n" -lt {int(n_stale)} ]; then\n'
        '        for dm in "$FAKE_BUCKET"/jobs/*/results.DONE.json; do\n'
        '          [ -e "$dm" ] && echo "$dm" >> "$CNT.violation"\n'
        '        done\n'
        '        exit 0\n'
        '      fi ;;\n'
        '  esac\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d, cnt


def _vanish_shimdir(tmp_path, shimdir, *, corrupt=False):
    """A PATH dir whose `rclone` wraps the shared shim and DELETES the queue
    ticket from the fake bucket during the `copyto` that downloads it — the
    `job retarget` / fleetd-replacement delete landing between poll_once's LIST
    and its read (JOB_RETARGET_RACE_2026-08-20.md).

    Two shapes, because they exercise different guards:
      corrupt=False — delete BEFORE the copy, so `copyto` itself fails (the
        already-guarded "ticket download failed — skip this pass" path).
      corrupt=True  — copy, then delete the source AND leave unparseable bytes
        on disk, so `prepare` fails on a ticket that no longer exists anywhere.
        That is the path that used to emit a spurious `failed` and latch
        `.terminal`.
    """
    d = tmp_path / ("vanishbin_c" if corrupt else "vanishbin")
    d.mkdir()
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        'if [ "${1:-}" = copyto ]; then\n'
        '  case "${2:-}" in\n'
        '    */jobs/queue/*/*.json)\n'
        '      src="$FAKE_BUCKET/${2#*:*/}"\n'
        + ('      "$REAL" "$@" || exit $?\n'
           '      rm -f "$src"\n'
           '      printf \'{ not json\' > "$3"\n'
           '      exit 0 ;;\n' if corrupt else
           '      rm -f "$src" ;;\n')
        + '  esac\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _queuecatfail_shimdir(tmp_path, shimdir):
    """A PATH dir whose `rclone` wraps the shared shim and fails every `cat` of a
    queue ticket — a transport blip on the one read that decides whether an
    operator requeued the job."""
    d = tmp_path / "qcatfailbin"
    d.mkdir()
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        'if [ "${1:-}" = cat ]; then\n'
        '  case "${2:-}" in\n'
        '    */jobs/queue/*/*.json) exit 3 ;;\n'
        '  esac\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _copylog_shimdir(tmp_path, shimdir):
    """A PATH dir whose `rclone` wraps the shared shim and APPENDS every `copy`
    op's full argv to a log file, so a test can count how many copy operations
    targeted each B2 prefix (the write-side invariant that jobs/<id>/results/ is
    written exactly once, at finalize). Returns (pathdir, logfile)."""
    d = tmp_path / "copylogbin"
    d.mkdir()
    logf = d / "copies.log"
    logf.write_text("")
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        f'LOG={shlex.quote(str(logf))}\n'
        'if [ "${1:-}" = copy ]; then printf "%s\\n" "$*" >> "$LOG"; fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d, logf


def _stage_job(tmp_path, bucket, iid, entry="mkdir -p out\necho \"hello $FOO job=$JOB_ID\"\n"
               "echo done > out/result.txt\n", rc0=True, config=None):
    src = tmp_path / "jobsrc"
    (src).mkdir()
    (src / "run.sh").write_text(entry if rc0 else entry + "exit 7\n")
    (src / "job-config.yaml").write_text(config or (
        "version: 1\nname: e2e-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "env:\n  FOO: \"world\"\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n"))
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(src)), str(src))
    # bundle -> bucket
    tmp_bundle = tmp_path / "b.tar.zst"
    info = jm.write_bundle(str(src), str(tmp_bundle))
    sha = info["sha256"]
    bdir = bucket / "jobs" / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_bundle), str(bdir / f"{sha}.tar.zst"))
    # ticket -> queue
    job_id = jm.mint_job_id(cfg["name"])
    ticket = jm.make_ticket(job_id, sha, "cli:test", cfg, str(iid))
    qdir = bucket / "jobs" / "queue" / str(iid)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{job_id}.json").write_text(json.dumps(ticket))
    return job_id, sha


def _cred_hermetic(env, tmp_path):
    """Cred-refresh hermeticity for EVERY jobd run: markers land in a tmp dir
    (never the repo's onstart/, jobd.sh's default $JOBD_DIR when run in-tree)
    and no broker identity leaks in from the developer environment — so all
    pre-broker tests exercise the exact no-op path a pre-broker box sees."""
    for k in ("BOX_IDENTITY_NONCE", "B2_KEY_EXPIRES_AT", "CRED_BROKER_URL",
              "CRED_ROLE", "TS_AUTHKEY"):
        env.pop(k, None)
    env["JOBD_CRED_DIR"] = str(tmp_path / "credstate")


def _fake_shm(tmp_path):
    """The container-boot nonce lives on a tmpfs (/dev/shm) on a real box; the
    suite pins it under tmp_path so no test ever touches the machine's /dev/shm
    (hermeticity) and a test can simulate a box stop/start by wiping the dir —
    exactly what vast restarting the container does to the real tmpfs."""
    d = tmp_path / "shm"
    d.mkdir(exist_ok=True)
    return d / "jobd_boot_nonce"


def _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=None):
    env = _hermetic_env(tmp_path)
    env["PATH"] = f"{shimdir}:{env['PATH']}"
    env["FAKE_BUCKET"] = str(bucket)
    env["B2_BUCKET"] = "testbucket"
    env["JOBD_IID"] = str(iid)
    env["JOBD_ROOT"] = str(tmp_path / "workspace")
    env["JOBD_BOOT_NONCE_FILE"] = str(_fake_shm(tmp_path))
    env["JOBD_ONCE"] = "1"
    env["JOBD_SKIP_GPU"] = "1"
    env["JOBD_SKIP_B2CONFIG"] = "1"
    env["JOBD_HEARTBEAT_S"] = "1"
    env["JOBD_PYTHON"] = sys.executable
    # The boot scratch probe's mount attempt (P4e) is OFF for the suite: no test
    # needs the daemon touching this machine's mount table, and a root CI
    # container would really mount a tmpfs inside tmp_path. The two probe tests
    # that care re-enable it explicitly.
    env["JOBD_TMPFS_PROBE"] = "0"
    # The CPU probe deliberately ignores JOBD_SKIP_GPU (a GPU-less box is the
    # one it most wants to measure), so unlike gemm_probe nothing else here
    # suppresses it. Off by name: on an idle machine it would add seconds to
    # every boot in this file, and its own busy-box refusal would make that
    # cost depend on host load. test_jobd_cpu_probe.py owns the stanza.
    env["JOBD_CPU_PROBE"] = "0"
    _cred_hermetic(env, tmp_path)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(["bash", JOBD_SH], env=env, capture_output=True, text=True,
                          timeout=120)


def _events(bucket, job_id):
    d = bucket / "jobs" / job_id / "events"
    bodies = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            # SKIP the shim's in-flight temp files. rclone_shim.sh's `rcat` is
            # atomic by writing `.rclone_shim.XXXXXX` beside the target and
            # renaming — so a listing taken mid-emit contains a half-written
            # file that a REAL B2 listing could never return (an object appears
            # whole or not at all). Reading it raised "JSONDecodeError: line 1
            # column 1" in whichever test happened to be listing at that
            # instant, which is a defect in this reader's model of B2, not in
            # the code under test. Sampled 2026-08-09 on the interrupted-
            # transfer test at checkpoint_s=1 under suite load; passes standing
            # alone, which is exactly what made it look like flakiness.
            if f.name.startswith("."):
                continue
            bodies.append(f.read_text())
    return bodies


def _popen_jobd(tmp_path, bucket, shimdir, iid, extra_env=None):
    """Like _run_jobd but launches the DAEMON loop (no JOBD_ONCE) detached in its
    own session, so a test can deliver an external SIGTERM to the daemon shell mid
    job and inspect the preemption trap. Returns the Popen."""
    env = _hermetic_env(tmp_path)
    env["PATH"] = f"{shimdir}:{env['PATH']}"
    env["FAKE_BUCKET"] = str(bucket)
    env["B2_BUCKET"] = "testbucket"
    env["JOBD_IID"] = str(iid)
    env["JOBD_ROOT"] = str(tmp_path / "workspace")
    env["JOBD_BOOT_NONCE_FILE"] = str(_fake_shm(tmp_path))
    env["JOBD_SKIP_GPU"] = "1"
    env["JOBD_SKIP_B2CONFIG"] = "1"
    env["JOBD_HEARTBEAT_S"] = "1"
    env["JOBD_POLL"] = "1"
    env["JOBD_PYTHON"] = sys.executable
    # The boot scratch probe's mount attempt (P4e) is OFF for the suite: no test
    # needs the daemon touching this machine's mount table, and a root CI
    # container would really mount a tmpfs inside tmp_path. The two probe tests
    # that care re-enable it explicitly.
    env["JOBD_TMPFS_PROBE"] = "0"
    # The CPU probe deliberately ignores JOBD_SKIP_GPU (a GPU-less box is the
    # one it most wants to measure), so unlike gemm_probe nothing else here
    # suppresses it. Off by name: on an idle machine it would add seconds to
    # every boot in this file, and its own busy-box refusal would make that
    # cost depend on host load. test_jobd_cpu_probe.py owns the stanza.
    env["JOBD_CPU_PROBE"] = "0"
    _cred_hermetic(env, tmp_path)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(["bash", JOBD_SH], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)


def _wait_for_event(bucket, job_id, kind, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        for b in _events(bucket, job_id):
            try:
                if json.loads(b).get("event") == kind:
                    return True
            except ValueError:
                pass
        time.sleep(0.1)
    return False


def test_jobd_runs_job_end_to_end(tmp_path):
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90001
    job_id, sha = _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, f"jobd failed: {r.stderr}"

    # events folded -> done
    bodies = _events(bucket, job_id)
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"events={bodies} stderr={r.stderr}"
    kinds = {json.loads(b)["event"] for b in bodies}
    assert {"claimed", "started", "done"} <= kinds

    # results uploaded + marker-last + log
    res = bucket / "jobs" / job_id / "results" / "out" / "result.txt"
    assert res.is_file() and res.read_text().strip() == "done"
    done_marker = bucket / "jobs" / job_id / "results.DONE.json"
    assert done_marker.is_file()
    dm = json.loads(done_marker.read_text())
    assert dm["rc"] == 0 and dm["n_results"] >= 1
    log = bucket / "jobs" / job_id / "log.txt"
    assert log.is_file() and "hello world" in log.read_text()


def test_jobd_idempotent_second_pass_skips(tmp_path):
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90002
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    _run_jobd(tmp_path, bucket, shimdir, iid)
    n1 = len(_events(bucket, job_id))
    # second pass: job already has events -> skipped, no new claimed
    _run_jobd(tmp_path, bucket, shimdir, iid)
    n2 = len(_events(bucket, job_id))
    claims = sum(1 for b in _events(bucket, job_id) if json.loads(b)["event"] == "claimed")
    assert n2 == n1 and claims == 1


def test_jobd_failing_entrypoint_emits_failed(tmp_path):
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90003
    job_id, _ = _stage_job(tmp_path, bucket, iid, rc0=False)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed" and v["rc"] == 7


def test_jobd_publish_verify_waits_out_stale_read_before_done(tmp_path):
    """The e2-paired ordering race (three live controller false-fails,
    2026-07-15/16/19): the final results copy has returned success but B2 keeps
    serving the STALE (empty) version of an overwritten results key. Publish
    verify must re-read until the published bytes are actually served and only
    THEN write results.DONE.json — so the controller (which keys off DONE) can
    never observe a declared result that is not yet readable."""
    bucket, shimdir = _make_bucket(tmp_path)
    staledir, cnt = _stalecat_shimdir(tmp_path, shimdir, "out/result.txt", 2)
    iid = 90031
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, staledir, iid,
                  extra_env={"JOBD_PUBLISH_VERIFY_BACKOFF_S": "1"})
    assert r.returncode == 0, r.stderr
    kinds = {json.loads(b)["event"] for b in _events(bucket, job_id)}
    assert "done" in kinds and "publish_verify_failed" not in kinds
    assert (bucket / "jobs" / job_id / "results.DONE.json").is_file(), r.stderr
    # verify actually waited through the stale window: 2 stale reads + a good one
    assert int(cnt.read_text()) >= 3, r.stderr
    # THE ordering guarantee: no stale read ever happened after DONE was visible
    assert not (staledir / "catcount.violation").exists()


def test_jobd_publish_verify_exhaustion_still_writes_done_and_emits(tmp_path):
    """A results read that NEVER converges must not strand the job: after the
    verify budget the job still reaches its terminal state (DONE written, done
    event) but publish_verify_failed is emitted so the controller's own
    stale-read retry backstop and the operator see the unverified publish."""
    bucket, shimdir = _make_bucket(tmp_path)
    staledir, cnt = _stalecat_shimdir(tmp_path, shimdir, "out/result.txt", 10 ** 6)
    iid = 90032
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, staledir, iid,
                  extra_env={"JOBD_PUBLISH_VERIFY_TIMEOUT_S": "3",
                             "JOBD_PUBLISH_VERIFY_BACKOFF_S": "1"})
    assert r.returncode == 0, r.stderr
    assert (bucket / "jobs" / job_id / "results.DONE.json").is_file(), r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "publish_verify_failed" in kinds, r.stderr
    assert "done" in kinds
    # a FAILED verify is never ALSO a positive signal (mutually exclusive)
    assert "publish_verified" not in kinds, r.stderr


def test_jobd_emits_publish_verified_on_all_match(tmp_path):
    """Part A of the 5x cross-client false-fail fix: when publish-verify reads
    every uploaded result back from B2 at its final local sha, jobd stamps a
    positive `publish_verified` event (listing the results/-relative paths) that
    the controller trusts to SKIP its own redundant, racy per-arm re-read."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90033
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    evs = [json.loads(b) for b in _events(bucket, job_id)]
    pv = [e for e in evs if e.get("event") == "publish_verified"]
    assert len(pv) == 1, f"events={[e.get('event') for e in evs]} stderr={r.stderr}"
    # the verified-files list carries the results/-relative path the manifest arm
    # paths normalize to; the job's single result is out/result.txt.
    files = pv[0].get("files")
    assert "out/result.txt" in (files.split(",") if isinstance(files, str) else files)
    assert pv[0].get("n_files") == 1
    # and it never co-occurs with a failure signal on the happy path
    assert not any(e.get("event") == "publish_verify_failed" for e in evs)


def test_jobd_no_publish_verified_when_uploaded_empty(tmp_path):
    """publish_verified is a signal about VERIFIED results — a job that publishes
    nothing (its results glob matches no file) must NOT emit it (the verify block
    is skipped on an empty .uploaded set), so the controller never spuriously
    trusts a zero-coverage job."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90034
    # results glob matches nothing (job writes no `nope/` tree) -> empty .uploaded
    cfg = ("version: 1\nname: e2e-empty\nentrypoint: run.sh\ntimeout_s: 60\n"
           "env:\n  FOO: \"world\"\n"
           "results:\n  - \"nope/**\"\n"
           "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid,
                           entry="echo \"hello $FOO job=$JOB_ID\"\n", config=cfg)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "done" in kinds, r.stderr
    assert "publish_verified" not in kinds, r.stderr


def test_jobd_checkpoint_sync_survives_timeout_kill(tmp_path):
    """The lost-LoRA regression (2026-07-10): an entrypoint that writes periodic
    state then dies to its own timeout must leave the last synced checkpoint in
    checkpoints/ on B2 — with checkpoint_s set, the mid-run sync ships it BEFORE the
    kill; without it, everything is lost. (The mid-run sync targets checkpoints/, NOT
    results/, so the canonical results/ prefix stays a single new-object write at
    finalize — the B2-overwrite-eventual-consistency fix.)"""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90004
    # writes a "checkpoint" immediately, then sleeps far past timeout_s=4
    entry = "mkdir -p out\necho step-100 > out/ckpt.txt\nsleep 60\n"
    config = (
        "version: 1\nname: ckpt-probe\nentrypoint: run.sh\ntimeout_s: 4\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr

    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed" and v["rc"] == 124        # killed by timeout
    assert v["n_checkpoints"] >= 1 and v["last_checkpoint_ts"], \
        f"no checkpoint events: {v}"
    # the mid-run sync shipped the checkpoint despite the kill — into checkpoints/
    # (the timeout still runs the finalize publish, so results/ is populated too;
    # the invariant that results/ is never OVERWRITTEN mid-run is proven separately
    # by test_jobd_results_prefix_written_once_at_finalize).
    res = bucket / "jobs" / job_id / "checkpoints" / "out" / "ckpt.txt"
    assert res.is_file() and res.read_text().strip() == "step-100"
    # no DONE marker — partial results are distinguishable from a clean finish
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file() or \
        json.loads((bucket / "jobs" / job_id / "results.DONE.json").read_text())["rc"] != 0


# --- box-disk checkpoint lifecycle, END TO END --------------------------------
# The refusal paths (B2 unreadable, torn upload, stale handoff epoch) are unit-
# tested in test_jobd_ckpt_lifecycle.py, which drives the functions directly. What
# these three add is the wiring: that a REAL jobd run prunes mid-run, scrubs at the
# end, leaves B2 untouched, and can fire a sync before its timer.

_LIFECYCLE_ENTRY = (
    "mkdir -p out\n"
    "for s in 10 20 30 40; do\n"
    "  mkdir -p out/checkpoint-$s\n"
    "  printf '%*s' 2048 '' > out/checkpoint-$s/optimizer.pt\n"
    "  printf '{\"global_step\": %s}\\n' $s > out/checkpoint-$s/trainer_state.json\n"
    "done\n"
    "echo adapter > out/adapter_model.safetensors\n"
    "sleep 6\n"
)
_LIFECYCLE_CONFIG = (
    "version: 1\nname: ckpt-lifecycle\nentrypoint: run.sh\ntimeout_s: 60\n"
    "checkpoint_s: 2\n"
    "checkpoints:\n  - \"out/**\"\n"
    "results:\n  - \"out/**\"\n"
    "needs:\n  gpu: false\n  venv: none\n")
_LIFECYCLE_ENV = {
    "JOBD_CKPT_MIN_AGE": "0s",     # the fixture's files are seconds old
    "JOBD_CKPT_SETTLE_S": "1",     # ditto — the real default is 5
    "JOBD_CKPT_WATCH_S": "1",
}


def test_jobd_prunes_and_scrubs_box_disk_but_never_b2(tmp_path):
    """The whole lifecycle in one run. A job that writes four checkpoints must end
    with ZERO checkpoint dirs on the box (mid-run prune of the old ones + the
    end-of-run scrub of the rest) and ALL FOUR still on B2 — B2 holds the dose
    curve, and nothing in this change may delete from it."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90410
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=_LIFECYCLE_ENTRY,
                           config=_LIFECYCLE_CONFIG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=dict(_LIFECYCLE_ENV))
    assert r.returncode == 0, r.stderr
    evs = [json.loads(b) for b in _events(bucket, job_id)]
    kinds = [e["event"] for e in evs]
    assert "done" in kinds, (kinds, r.stderr[-3000:])

    # 1. B2 keeps EVERY checkpoint — the dose curve is untouched.
    ck = bucket / "jobs" / job_id / "checkpoints" / "out"
    for s in (10, 20, 30, 40):
        assert (ck / f"checkpoint-{s}" / "optimizer.pt").is_file(), \
            f"checkpoint-{s} missing from B2: {sorted(p.name for p in ck.iterdir())}"

    # 2. the box disk has none of them left
    work = tmp_path / "workspace" / "jobs" / job_id / "work" / "out"
    left = sorted(p.name for p in work.iterdir() if p.name.startswith("checkpoint-"))
    assert left == [], f"checkpoint dirs left on the box disk: {left}"
    # non-checkpoint output is NOT touched
    assert (work / "adapter_model.safetensors").is_file()

    # 3. both levers actually fired, and said so
    assert any(e["event"] == "checkpoint" and int(e.get("pruned", 0)) > 0
               for e in evs), \
        f"no mid-run prune: {[e for e in evs if e['event'] == 'checkpoint']}"
    scrub = [e for e in evs if e["event"] == "checkpoints_scrubbed"]
    assert scrub and int(scrub[0]["n"]) >= 1, f"no end-of-run scrub: {kinds}"

    # 4. the narrowing of results/ is RECORDED, not silent. A dir pruned mid-run is
    # gone before the finalize publish globs out/**, so it lives on B2 under
    # checkpoints/ ONLY — `herdd job pull` (which reads results/) and any
    # bucket-side retention sweep reasoning about "is results/ a superset of
    # checkpoints/" must be able to find that out. Adversarial-review finding.
    marker = bucket / "jobs" / job_id / "CHECKPOINTS_PRUNED.json"
    assert marker.is_file(), "no CHECKPOINTS_PRUNED.json marker"
    mk = json.loads(marker.read_text())
    assert mk["job_dirs"] and mk["prefix"].endswith("checkpoints/"), mk
    # finalize REPLACES the mid-run partial marker with the complete list
    assert "partial" not in mk, mk
    done = json.loads((bucket / "jobs" / job_id / "results.DONE.json").read_text())
    assert done["checkpoints_pruned"] == mk["job_dirs"], done
    # and those dirs really are absent from results/ while present under checkpoints/
    for rel in mk["job_dirs"]:
        assert not (bucket / "jobs" / job_id / "results" / rel).exists(), rel
        assert (bucket / "jobs" / job_id / "checkpoints" / rel).is_dir(), rel


def test_jobd_raises_the_pruned_marker_mid_run_not_only_at_finalize(tmp_path):
    """`ckpt_retention.py` gates on the PRESENCE of CHECKPOINTS_PRUNED.json, so the
    marker has to exist from the moment the invariant breaks — a box that prunes and
    then dies before finalize would otherwise leave a pruned job with no marker and
    a bucket sweep free to delete the only copy. The job here is killed by its own
    timeout, so the finalize path that writes the complete marker never runs
    normally; what must survive is the mid-run `partial` one."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90413
    config = _LIFECYCLE_CONFIG.replace("timeout_s: 60", "timeout_s: 8")
    job_id, _ = _stage_job(tmp_path, bucket, iid,
                           entry=_LIFECYCLE_ENTRY.replace("sleep 6", "sleep 120"),
                           config=config)
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=dict(_LIFECYCLE_ENV))
    assert r.returncode == 0, r.stderr
    marker = bucket / "jobs" / job_id / "CHECKPOINTS_PRUNED.json"
    assert marker.is_file(), \
        f"no mid-run CHECKPOINTS_PRUNED.json: {r.stderr[-3000:]}"
    mk = json.loads(marker.read_text())
    assert mk["job_dirs"] and mk["prefix"].endswith("checkpoints/"), mk


def test_jobd_checkpoint_scrub_can_be_disabled(tmp_path):
    """JOBD_CKPT_SCRUB=0 leaves the run dir exactly as the entrypoint left it —
    the escape hatch for anyone who needs the box-local copy for triage."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90411
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=_LIFECYCLE_ENTRY,
                           config=_LIFECYCLE_CONFIG)
    env = dict(_LIFECYCLE_ENV)
    env.update({"JOBD_CKPT_SCRUB": "0", "JOBD_CKPT_PRUNE": "0"})
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    work = tmp_path / "workspace" / "jobs" / job_id / "work" / "out"
    left = sorted(p.name for p in work.iterdir() if p.name.startswith("checkpoint-"))
    assert left == ["checkpoint-10", "checkpoint-20",
                    "checkpoint-30", "checkpoint-40"], left
    assert not any(json.loads(b)["event"] == "checkpoints_scrubbed"
                   for b in _events(bucket, job_id))


def test_jobd_checkpoint_sync_fires_on_arrival_not_on_the_timer(tmp_path):
    """Owner 2026-08-05: "begin sync as soon as the checkpoint hits the disk". With
    checkpoint_s=60 and a job that lives ~8s, the PERIODIC pass can never run — so
    a checkpoint on B2 at the end proves the fire-on-arrival watcher shipped it.
    Box 46859541 lost 51 steps to exactly this gap on 2026-08-05 (checkpoint-50
    written 7s after a pass; box died ~3min later with no SIGTERM, so the preempt
    trap's final flush never fired either)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90412
    entry = (
        "mkdir -p out/checkpoint-50\n"
        "printf '%*s' 512 '' > out/checkpoint-50/optimizer.pt\n"
        "printf '{\"global_step\": 50}\\n' > out/checkpoint-50/trainer_state.json\n"
        "sleep 8\n")
    config = (
        "version: 1\nname: ckpt-onarrival\nentrypoint: run.sh\ntimeout_s: 60\n"
        "checkpoint_s: 60\n"                       # the timer can NEVER fire here
        "checkpoints:\n  - \"out/**\"\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    # DEFAULT min-age (45s) deliberately left in place: the fast path must ship a
    # file only seconds old, which the periodic pass's age filter would skip.
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_WATCH_S": "1", "JOBD_CKPT_SETTLE_S": "1"})
    assert r.returncode == 0, r.stderr
    obj = (bucket / "jobs" / job_id / "checkpoints" / "out" / "checkpoint-50"
           / "optimizer.pt")
    assert obj.is_file(), \
        f"checkpoint never reached B2 inside the 60s timer: {r.stderr[-3000:]}"
    trig = [json.loads(b) for b in _events(bucket, job_id)
            if json.loads(b)["event"] == "checkpoint"]
    assert any(e.get("trigger") == "new-checkpoint" for e in trig), trig


def test_jobd_checkpoint_reports_zero_files_when_all_younger_than_min_age(tmp_path):
    """Issue B (v1 canary silent-data-loss): rclone --min-age skips files YOUNGER
    than the window, so a job checkpointing faster than JOBD_CKPT_MIN_AGE ships
    ZERO bytes each pass — yet the event used to report files=<glob count>, a lie
    that made a later retarget "resume" from nothing. When every match is younger
    than the window the checkpoint event must report files=0 (the honest shipped
    count) with matched=<glob count> retained for context."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90200
    # write the checkpoint file once (mtime ~now), then sleep past timeout_s=3 —
    # every sync pass sees a file only seconds old, far younger than the default
    # 45s --min-age window, so rclone would ship nothing.
    entry = "mkdir -p out\necho step-100 > out/ckpt.txt\nsleep 60\n"
    config = (
        "version: 1\nname: young-ckpt-probe\nentrypoint: run.sh\ntimeout_s: 3\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    # DEFAULT min-age (45s) — do NOT override it; that is the whole point.
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    ckpts = [json.loads(b) for b in _events(bucket, job_id)
             if json.loads(b)["event"] == "checkpoint"]
    assert ckpts, f"no checkpoint events: {[json.loads(b)['event'] for b in _events(bucket, job_id)]}"
    for e in ckpts:
        assert e.get("files") == 0, \
            f"checkpoint reported files={e.get('files')} (should be 0 — all younger than min-age): {e}"
        assert e.get("matched", 0) >= 1, f"pre-filter matched context missing/zero: {e}"


def test_jobd_results_prefix_written_once_at_finalize(tmp_path):
    """WRITE-SIDE fix for the four B2 eventual-consistency false-fails
    (2026-07-15/16/19/20). The mid-run checkpoint loop used to OVERWRITE
    jobs/<id>/results/<arm>.jsonl repeatedly (the e2 gen bundles list the arm files
    in BOTH the `checkpoints:` and `results:` globs), then finalize overwrote the
    SAME key — and a B2 OBJECT OVERWRITE is eventually consistent, so a different
    client (validate_generation_artifact) kept reading the stale empty/partial
    version for ~90s. With the split, the recurring mid-run sync targets
    jobs/<id>/checkpoints/, leaving jobs/<id>/results/ written by EXACTLY ONE copy op
    (finalize) as a NEW object — which B2 serves with strong read-after-write, so no
    client can ever read a stale results/ key. This is the multi-client-staleness
    insight encoded structurally: results/ is never overwritten."""
    bucket, shimdir = _make_bucket(tmp_path)
    logdir, logf = _copylog_shimdir(tmp_path, shimdir)
    iid = 90300
    # a gen-style arm file listed in BOTH globs; checkpoint every 1s and stay alive
    # across several ticks so the mid-run sync fires repeatedly, then finish clean.
    entry = ("mkdir -p out\necho '{\"row\":1}' > out/gens.jsonl\nsleep 4\n"
             "echo '{\"row\":2}' >> out/gens.jsonl\nexit 0\n")
    config = (
        "version: 1\nname: once-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "checkpoint_s: 1\n"
        "checkpoints:\n  - \"out/**\"\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    r = _run_jobd(tmp_path, bucket, logdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v}"
    # the mid-run sync ran (>=1 checkpoint event) and landed under checkpoints/
    assert v["n_checkpoints"] >= 1, f"mid-run sync never fired: {v}"
    assert (bucket / "jobs" / job_id / "checkpoints" / "out" / "gens.jsonl").is_file()
    # the final results published to the canonical prefix
    assert (bucket / "jobs" / job_id / "results" / "out" / "gens.jsonl").is_file()
    # THE INVARIANT: the canonical results/ prefix received EXACTLY ONE copy op
    # (finalize) — never a pre-finalize checkpoint overwrite; the recurring mid-run
    # copies all targeted checkpoints/.
    res_copies = [l for l in logf.read_text().splitlines()
                  if f"jobs/{job_id}/results/" in l]
    ckpt_copies = [l for l in logf.read_text().splitlines()
                   if f"jobs/{job_id}/checkpoints/" in l]
    assert len(res_copies) == 1, \
        f"results/ written {len(res_copies)}x (expected exactly 1 at finalize): {res_copies}"
    assert len(ckpt_copies) >= 1, \
        f"mid-run sync did not target checkpoints/: {ckpt_copies}"


def test_jobd_pullback_ignores_legacy_results_prefix(tmp_path):
    """The resume pull-back source moved WITH the write: it restores mid-run state
    from jobs/<id>/checkpoints/, NOT the legacy results/ prefix. State placed ONLY
    under the old results/ location is no longer pulled — the entrypoint restarts
    from scratch and times out. (The positive round-trip — real sync to checkpoints/
    then real pull-back from it — is covered by test_jobd_resumes_interrupted_job and
    test_jobd_retarget_pulls_checkpoint_back.)"""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90310, 90311
    # timeout_s small so a scratch restart (missed pull-back) FAILS with rc 124.
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="legr",
                                      timeout_s=4)
    # relocate the pre-synced checkpoint from checkpoints/ (the current source) to
    # the LEGACY results/ prefix — the pull-back must NOT read it there anymore.
    ck = bucket / "jobs" / job_id / "checkpoints" / "out" / "state.txt"
    dst = bucket / "jobs" / job_id / "results" / "out"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ck), str(dst / "state.txt"))
    r = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(new_iid)})
    # no pull-back from results/ => the entrypoint starts fresh, writes state, sleeps,
    # and times out — proving the old prefix is no longer a pull-back source.
    assert v["status"] == "failed" and v["rc"] == 124, \
        f"pull-back wrongly read the legacy results/ prefix (job resumed): {v}"


def test_validate_job_config_checkpoint_fields(tmp_path):
    src = tmp_path / "cfgsrc"
    src.mkdir()
    (src / "run.sh").write_text("true\n")
    base = {"version": 1, "name": "cfg-probe", "entrypoint": "run.sh",
            "results": ["out/**"], "needs": {"gpu": False, "venv": "none"}}

    # checkpoint_s alone -> checkpoints defaults to results globs
    cfg, _ = jm.validate_job_config({**base, "checkpoint_s": 120}, str(src))
    assert cfg["checkpoint_s"] == 120 and cfg["checkpoints"] == ["out/**"]

    # checkpoints alone -> default interval
    cfg, _ = jm.validate_job_config({**base, "checkpoints": ["out/ckpt-*/**"]}, str(src))
    assert cfg["checkpoint_s"] == 300 and cfg["checkpoints"] == ["out/ckpt-*/**"]

    # neither -> absent from canonical config (old jobd compatible)
    cfg, _ = jm.validate_job_config(dict(base), str(src))
    assert "checkpoint_s" not in cfg and "checkpoints" not in cfg

    # bad values
    with pytest.raises(jm.JobmetaError):
        jm.validate_job_config({**base, "checkpoint_s": -5}, str(src))
    with pytest.raises(jm.JobmetaError):
        jm.validate_job_config({**base, "checkpoint_s": True}, str(src))
    with pytest.raises(jm.JobmetaError):
        jm.validate_job_config({**base, "checkpoints": ["/abs/path/**"]}, str(src))
    with pytest.raises(jm.JobmetaError):
        jm.validate_job_config(
            {**base, "results": [], "checkpoint_s": 60}, str(src))

    # aggressive interval -> warning, not an error
    _, warns = jm.validate_job_config({**base, "checkpoint_s": 5}, str(src))
    assert any("aggressive" in w for w in warns)


def test_jobd_preemption_trap_flushes_and_emits(tmp_path):
    """SPOT_DESIGN §3.3 (jobd trap): an EXTERNAL SIGTERM to the daemon mid-job must
    emit exactly ONE non-terminal 'preempted' event AND kick one bounded final
    checkpoint-glob flush into checkpoints/ — narrowing preemption loss to <1
    interval. (Checkpoint globs flush to checkpoints/, matching the mid-run sync +
    resume pull-back; results/ stays a finalize-only write.)"""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90005
    # write a checkpoint immediately, then sleep far past the SIGTERM. checkpoint_s
    # is large so the periodic mid-run sync loop NEVER fires (it sleeps first) — the
    # only flush that can land the checkpoint is the trap's.
    entry = "mkdir -p out\necho step-200 > out/ckpt.txt\nsleep 60\n"
    config = (
        "version: 1\nname: preempt-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "checkpoint_s: 3600\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        # WAIT for the trap's two real preconditions instead of sleeping a guessed
        # interval. A fixed `time.sleep(1.5)` here made this test flaky under suite
        # load: SIGTERM could land before the entrypoint had written the checkpoint
        # or before the daemon had dropped the job's `.running` breadcrumb, and the
        # trap then correctly flushed nothing — a green-vs-red decided by machine
        # speed, not by jobd's behavior. (`_jobd_preempt` iterates $STATE_DIR/
        # *.running and ships $wdir/work through the checkpoint globs, so these two
        # files ARE the precondition; the trap itself is armed from daemon start.)
        state = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.running"
        ckpt = tmp_path / "workspace" / "jobs" / job_id / "work" / "out" / "ckpt.txt"
        assert _wait_for_file(ckpt), "entrypoint never wrote out/ckpt.txt"
        assert _wait_for_file(state), "jobd never marked the job .running"
        os.kill(p.pid, signal.SIGTERM)        # EXTERNAL signal to the daemon shell
        p.wait(timeout=60)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)   # reap orphaned sleep/heartbeat
        except (ProcessLookupError, PermissionError):
            pass

    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert kinds.count("preempted") == 1, f"kinds={kinds}"
    # the trap's checkpoint-glob flush shipped the checkpoint into checkpoints/
    # despite the abrupt stop (the trap ALSO flushes results globs into results/ as
    # the N1b unpublished-results safety net, so results/out/ckpt.txt exists too —
    # here checkpoints defaults to the results globs).
    res = bucket / "jobs" / job_id / "checkpoints" / "out" / "ckpt.txt"
    assert res.is_file() and res.read_text().strip() == "step-200", f"kinds={kinds}"
    # a preempted run is NOT a clean finish — no DONE marker
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file()
    # 'preempted' is non-terminal: the fold never reads it as done/failed
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] not in jm.TERMINAL


def test_jobd_cancels_running_job(tmp_path):
    """`herdd job cancel` writes jobs/<id>/CANCEL; a box RUNNING the job must see
    the marker, kill the entrypoint tree, and record a TERMINAL, non-resumable
    `cancelled` — not `failed`, not `interrupted`. JOBD_CANCEL_POLL=1 makes the
    watch fire fast for the test."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90040
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry="mkdir -p out\nsleep 120\n")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid,
                    extra_env={"JOBD_CANCEL_POLL": "1"})
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        # operator cancel: drop the CANCEL marker into the bucket (what the CLI's
        # write_cancel_marker does), simulating a queued-ticket delete by leaving
        # the marker as the sole signal.
        cdir = bucket / "jobs" / job_id
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "CANCEL").write_text(json.dumps({"reason": "doomed"}))
        # the `cancelled` event is emitted only AFTER `wait epid` returns, so its
        # arrival proves the entrypoint tree (sleep 120) was actually killed.
        assert _wait_for_event(bucket, job_id, "cancelled", timeout=30), \
            "cancel marker not honored (entrypoint still running)"
        # jobd emits the `cancelled` EVENT and only THEN writes the local
        # .terminal marker, so the event we just awaited does not imply the file
        # exists yet. Wait for it HERE — once the `finally` SIGKILLs the daemon
        # the marker can never appear, and asserting after the kill is a race the
        # test loses under machine load (observed 2026-07-31).
        term = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.terminal"
        end = time.time() + 15
        while time.time() < end and not term.is_file():
            time.sleep(0.1)
    finally:
        _kill(p)
    bodies = _events(bucket, job_id)
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "cancelled", f"events={[json.loads(b)['event'] for b in bodies]}"
    assert v["status"] in jm.TERMINAL
    # local terminal cache marks it cancelled so a resume never reconsiders it
    assert term.is_file() and term.read_text().split()[0] == "cancelled"


def test_jobd_cancel_before_claim_never_runs(tmp_path):
    """A CANCEL marker present when jobd first sees the ticket (delete lagged the
    listing): jobd must NOT run the job — it marks it terminal locally and skips,
    emitting no `claimed`/`started`."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90041
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    cdir = bucket / "jobs" / job_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "CANCEL").write_text(json.dumps({"reason": "pre-claim"}))
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    assert _events(bucket, job_id) == [], "cancelled-before-claim job was run"
    term = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.terminal"
    assert term.is_file() and term.read_text().split()[0] == "cancelled"
    # it never executed: no results marker
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file()


def _terminal_marker(tmp_path, job_id):
    return tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.terminal"


def test_jobd_a_ticket_deleted_before_the_download_is_not_condemned(tmp_path):
    """A queue ticket deleted between the LIST and the `copyto` costs the pass and
    nothing else: no `failed` event, no `.terminal` breadcrumb."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90211
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    vbin = _vanish_shimdir(tmp_path, shimdir)
    r = _run_jobd(tmp_path, bucket, vbin, iid)
    assert r.returncode == 0, r.stderr
    assert _events(bucket, job_id) == [], \
        f"a vanished ticket emitted events: {_events(bucket, job_id)}"
    assert not _terminal_marker(tmp_path, job_id).exists(), \
        "a vanished ticket latched .terminal — this box would skip the JOB_ID forever"


def test_jobd_a_ticket_that_vanishes_under_an_unparseable_read_is_not_condemned(tmp_path):
    """The retarget race proper: `prepare` fails on a ticket the retarget delete
    already removed from B2, and jobd must emit no `failed` and latch no
    `.terminal` — the ticket belongs to whichever box it was moved to."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90212
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    vbin = _vanish_shimdir(tmp_path, shimdir, corrupt=True)
    r = _run_jobd(tmp_path, bucket, vbin, iid)
    assert r.returncode == 0, r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "failed" not in kinds, f"spurious terminal event for a moved ticket: {kinds}"
    assert not _terminal_marker(tmp_path, job_id).exists(), \
        "a moved ticket latched .terminal — this box would skip the JOB_ID forever"
    assert "VANISHED" in r.stdout + r.stderr, "the suppression must say so in the log"


def test_jobd_a_ticket_that_is_present_and_malformed_still_fails_terminal(tmp_path):
    """The positive control for the vanish suppression: a ticket that is STILL IN
    THE QUEUE and genuinely unparseable keeps its `failed` event and its
    `.terminal` latch, so a bad ticket cannot become an infinite re-read loop."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90213
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    (bucket / "jobs" / "queue" / str(iid) / f"{job_id}.json").write_text("{ not json")
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert kinds == ["failed"], f"a real bad ticket must fail terminal, got {kinds}"
    term = _terminal_marker(tmp_path, job_id)
    assert term.is_file() and term.read_text().split()[0] == "failed"
    # the kept stderr is what makes the next rc attributable
    reason = json.loads(_events(bucket, job_id)[0]).get("reason") or ""
    assert "JSONDecodeError" in reason or "json" in reason.lower(), reason


def test_jobd_skips_a_job_it_already_latched_terminal(tmp_path):
    """The latch this whole lane exists to protect: a pre-seeded
    `<JOB_ID>.terminal` makes jobd skip the ticket before any B2 read — which is
    why `job retarget` must refuse a box that carries one."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90214
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    state = tmp_path / "workspace" / "jobs" / ".state"
    state.mkdir(parents=True, exist_ok=True)
    (state / f"{job_id}.terminal").write_text("failed 20260819T000000Z\n")
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    assert _events(bucket, job_id) == [], "a latched job was claimed anyway"
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file()


def test_jobd_no_preempted_on_normal_completion(tmp_path):
    """A job that runs to completion (no external signal) must NEVER emit
    'preempted' — the trap is TERM/INT, not EXIT."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90006
    job_id, _ = _stage_job(tmp_path, bucket, iid)     # fast rc=0 job
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    kinds = {json.loads(b)["event"] for b in _events(bucket, job_id)}
    assert "done" in kinds and "preempted" not in kinds


def test_fold_tolerates_preempted_event():
    """The 'preempted' event is a NON-terminal extension: the fold counts it,
    never flips status to terminal, and never errors (TERMINAL stays done/failed)."""
    iid = "77001"
    jid = "20260710T000000-x-ab"
    evs = [
        jm.make_event(jid, "claimed", f"box:{iid}", instance_id=iid),
        jm.make_event(jid, "started", f"box:{iid}", instance_id=iid),
        jm.make_event(jid, "preempted", f"box:{iid}", instance_id=iid),
    ]
    v = jm.fold_events([json.dumps(e) for e in evs], live_iids=set())
    assert v["parse_errors"] == 0 and v["n_events"] == 3
    assert v["status"] not in jm.TERMINAL
    assert "preempted" not in jm.TERMINAL


def test_jobd_echoes_experiment_association(tmp_path):
    """A matrix-arm ticket (config.experiment block, MATRIX_DESIGN.md audit
    seam): jobd exports it via prepare and echoes exp_id/arm on every
    lifecycle event from `started` on. Entrypoint sees EXP_ID/ARM_ID env."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90007
    exp_id = "20260710T000000-exp3-ab12"
    # stage by hand: build the config dict directly (the block is normally
    # written by jobmatrix.py submit, never authored in YAML)
    src = tmp_path / "jobsrc"
    src.mkdir()
    (src / "run.sh").write_text("mkdir -p out\necho \"arm=$ARM_ID exp=$EXP_ID\" > out/who.txt\n")
    raw = {"version": 1, "name": "mx-arm", "entrypoint": "run.sh", "timeout_s": 60,
           "env": {"EXP_ID": exp_id, "ARM_ID": "qwen-r16"},
           "results": ["out/**"], "needs": {"gpu": False, "venv": "none"},
           "experiment": {"exp_id": exp_id, "arm": "qwen-r16",
                          "axes": {"base": "qwen", "rank": "r16"}}}
    cfg, _ = jm.validate_job_config(raw, str(src))
    tmp_bundle = tmp_path / "b.tar.zst"
    sha = jm.write_bundle(str(src), str(tmp_bundle))["sha256"]
    bdir = bucket / "jobs" / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_bundle), str(bdir / f"{sha}.tar.zst"))
    job_id = jm.mint_job_id(cfg["name"])
    qdir = bucket / "jobs" / "queue" / str(iid)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{job_id}.json").write_text(
        json.dumps(jm.make_ticket(job_id, sha, "cli:test", cfg, str(iid))))

    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr

    by_kind = {}
    for b in _events(bucket, job_id):
        e = json.loads(b)
        by_kind.setdefault(e["event"], []).append(e)
    # v2: the scheduler parses the ticket BEFORE claiming (it needs needs.gpus),
    # so even `claimed` carries the association now (v1 claimed pre-parse).
    for kind in ("claimed", "started", "done"):
        assert by_kind[kind][-1].get("exp_id") == exp_id, by_kind
        assert by_kind[kind][-1].get("arm") == "qwen-r16", by_kind

    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done" and v["exp_id"] == exp_id and v["arm"] == "qwen-r16"
    who = bucket / "jobs" / job_id / "results" / "out" / "who.txt"
    assert who.read_text().strip() == f"arm=qwen-r16 exp={exp_id}"


# ---------------------------------------------------------------------------
# v2: interruption tolerance + GPU scheduling
# ---------------------------------------------------------------------------
def _stage_named(tmp_path, bucket, iid, name, entry, config_yaml, nonce4):
    """Stage with a deterministic JOB_ID suffix so FIFO order is controllable."""
    src = tmp_path / f"jobsrc-{name}"
    src.mkdir()
    (src / "run.sh").write_text(entry)
    (src / "job-config.yaml").write_text(config_yaml)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(src)), str(src))
    tmp_bundle = tmp_path / f"{name}.tar.zst"
    sha = jm.write_bundle(str(src), str(tmp_bundle))["sha256"]
    bdir = bucket / "jobs" / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_bundle), str(bdir / f"{sha}.tar.zst"))
    job_id = jm.mint_job_id(cfg["name"], ts="20260711T000000", nonce4=nonce4)
    qdir = bucket / "jobs" / "queue" / str(iid)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{job_id}.json").write_text(
        json.dumps(jm.make_ticket(job_id, sha, "cli:test", cfg, str(iid))))
    return job_id


def test_jobd_resumes_interrupted_job(tmp_path):
    """THE v2 bug fix (owner 2026-07-11): a job interrupted by preemption/park/
    daemon-death must be picked BACK UP on the next boot — `resumed` event,
    checkpoint pull-back into the fresh workdir, and completion. v1 skipped the
    ticket forever ("this box already claimed it")."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90010
    # first attempt: writes state, syncs it (checkpoint_s=1), then sleeps.
    # resumed attempt: finds the pulled-back state and finishes cleanly.
    entry = (
        "mkdir -p out\n"
        "if [ -f out/state.txt ]; then\n"
        "  echo \"restart=$JOB_RESTART_COUNT\" > out/resumed.txt\n"
        "  echo done > out/result.txt\n  exit 0\nfi\n"
        "echo step-1 > out/state.txt\nsleep 60\n")
    config = (
        "version: 1\nname: resume-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id = _stage_named(tmp_path, bucket, iid, "a", entry, config, "aaaa")

    p = _popen_jobd(tmp_path, bucket, shimdir, iid,
                    extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    try:
        assert _wait_for_event(bucket, job_id, "checkpoint"), "state never synced"
        # hard box death: SIGKILL the whole group (no trap, like a real preemption
        # with no grace) — the .running file + attempts survive on 'disk'
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.wait(timeout=30)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # next boot (same JOBD_ROOT = same box disk): resume + finish
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "resumed" in kinds, f"kinds={kinds} stderr={r.stderr}"
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"kinds={kinds}"
    assert v["attempts"] == 2 and v["last_resumed_ts"]
    resumed = bucket / "jobs" / job_id / "results" / "out" / "resumed.txt"
    assert resumed.is_file() and resumed.read_text().strip() == "restart=1"


def test_jobd_restart_cap_fails_terminally(tmp_path):
    """max_restarts=0 = never resume: after an interruption the next boot must
    emit a terminal `failed` (restart cap), not loop forever."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90011
    config = (
        "version: 1\nname: cap-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "max_restarts: 0\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id = _stage_named(tmp_path, bucket, iid, "b", "mkdir -p out\nsleep 60\n",
                          config, "bbbb")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.wait(timeout=30)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed" and "restart cap" in (v["fail_reason"] or ""), v


def test_jobd_schedules_concurrent_gpu_jobs(tmp_path):
    """v2 scheduler: two 1-GPU jobs on a faked 2-GPU box run CONCURRENTLY on
    DISJOINT cards (each entrypoint blocks until it sees the other's marker —
    both finishing rc=0 proves overlap; CUDA_VISIBLE_DEVICES proves assignment)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90012
    sync = tmp_path / "sync"
    sync.mkdir()
    entry = (
        "mkdir -p out\necho \"$CUDA_VISIBLE_DEVICES\" > out/gpu.txt\n"
        f"touch '{sync}'/$JOB_ID.here\n"
        "for i in $(seq 1 200); do\n"
        f"  n=$(ls '{sync}' | wc -l); [ \"$n\" -ge 2 ] && exit 0; sleep 0.1\n"
        "done\nexit 3\n")
    config = (
        "version: 1\nname: par-NAME\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpus: 1\n  venv: none\n")
    ja = _stage_named(tmp_path, bucket, iid, "c", entry,
                      config.replace("NAME", "a"), "aaaa")
    jb = _stage_named(tmp_path, bucket, iid, "d", entry,
                      config.replace("NAME", "b"), "bbbb")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:32,1:32", "JOBD_SKIP_GPU": "0"})
    assert r.returncode == 0, r.stderr
    seen = set()
    for jid in (ja, jb):
        v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
        assert v["status"] == "done", f"{jid}: {v} stderr={r.stderr}"
        gpu = (bucket / "jobs" / jid / "results" / "out" / "gpu.txt").read_text().strip()
        seen.add(gpu)
    assert seen == {"0", "1"}, f"assignments not disjoint: {seen}"


def test_jobd_fifo_whole_box_job_blocks_younger(tmp_path):
    """Strict FIFO: an older needs.gpus=all ticket takes every card; a younger
    1-GPU ticket must NOT be claimed in the same pass (no backfill starvation
    of whole-box jobs — the younger job waits for the next pass)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90013
    cfg_all = (
        "version: 1\nname: aa-whole-box\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpus: all\n  venv: none\n")
    cfg_one = cfg_all.replace("aa-whole-box", "zz-one-card").replace("gpus: all", "gpus: 1")
    entry = "mkdir -p out\necho \"$CUDA_VISIBLE_DEVICES\" > out/gpu.txt\necho ok > out/r.txt\n"
    j_all = _stage_named(tmp_path, bucket, iid, "e", entry, cfg_all, "aaaa")
    j_one = _stage_named(tmp_path, bucket, iid, "f", entry, cfg_one, "bbbb")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:32,1:32", "JOBD_SKIP_GPU": "0"})
    assert r.returncode == 0, r.stderr
    v_all = jm.fold_events(_events(bucket, j_all), live_iids={str(iid)})
    assert v_all["status"] == "done"
    gpu = (bucket / "jobs" / j_all / "results" / "out" / "gpu.txt").read_text().strip()
    assert gpu == "0,1"
    # the younger ticket was left for the next pass (single ONCE pass ran)
    assert _events(bucket, j_one) == [], "younger job claimed past a FIFO barrier"


def test_jobd_gpu_ram_floor_respected_per_card(tmp_path):
    """needs.gpu_ram_gb is a PER-CARD floor: on a box with one 16GB and one 48GB
    card, a 24GB job must land on the 48GB card only."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90014
    config = (
        "version: 1\nname: vram-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpus: 1\n  gpu_ram_gb: 24\n  venv: none\n")
    entry = "mkdir -p out\necho \"$CUDA_VISIBLE_DEVICES\" > out/gpu.txt\n"
    jid = _stage_named(tmp_path, bucket, iid, "g", entry, config, "aaaa")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:16,1:48", "JOBD_SKIP_GPU": "0"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
    assert v["status"] == "done", v
    gpu = (bucket / "jobs" / jid / "results" / "out" / "gpu.txt").read_text().strip()
    assert gpu == "1"


def test_jobd_exports_gpu_ram_and_cpu_cores_to_entrypoint(tmp_path):
    """The scheduler must hand the launch planner the box FACTS it needs: the
    entrypoint sees JOB_GPU_RAM_GB (min GB across the assigned cards) and
    CPU_CORES, not just JOB_GPU_COUNT. Absent JOB_GPU_RAM_GB, launch_plan.sh's
    VRAM-safety grad-ckpt floor + quant-by-VRAM silently never fire on-box. A
    faked single 24GB card is exactly the case that must trip the floor
    (off does not fit at 24GB -> flip to on) — so 24 must reach the entrypoint."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90015
    config = (
        "version: 1\nname: envprobe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpus: 1\n  venv: none\n")
    entry = ("mkdir -p out\n"
             "echo \"$JOB_GPU_RAM_GB|$JOB_GPU_COUNT|$CPU_CORES\" > out/env.txt\n")
    jid = _stage_named(tmp_path, bucket, iid, "h", entry, config, "aaaa")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:24", "JOBD_SKIP_GPU": "0"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
    assert v["status"] == "done", v
    line = (bucket / "jobs" / jid / "results" / "out" / "env.txt").read_text().strip()
    ram, count, cores = line.split("|")
    assert ram == "24", f"JOB_GPU_RAM_GB not exported (got {ram!r}) — floor cannot fire"
    assert count == "1", f"JOB_GPU_COUNT wrong: {count!r}"
    assert cores.isdigit() and int(cores) >= 1, f"CPU_CORES not a positive int: {cores!r}"


def test_jobd_exports_jobd_root_to_entrypoint(tmp_path):
    """The box root is part of the entrypoint contract, not an accident.

    A dozen bundles resolve the asset cache / the baked train env through
    `${JOBD_ROOT:-/workspace}` (tools/witness/jobs/{perf-levers,perf-levers-
    padfree,frontier-wave}/run.sh). That only ever worked because rehearse.sh and
    joblocal.py happen to pass JOBD_ROOT through jobd's OWN environment, so a
    child inherited it — nothing guaranteed it. Any caller that set the root some
    other way would silently send every such bundle back to a literal /workspace
    that a rehearsal cannot write, and the bundle dies in `mkdir -p /workspace`
    (Permission denied) with nothing downstream of its data gate ever exercised.

    On a rented box $ROOT is `${JOBD_ROOT:-/workspace}` = /workspace, so this
    export changes nothing there — it only makes the guarantee real off-box."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90016
    config = ("version: 1\nname: rootprobe\nentrypoint: run.sh\ntimeout_s: 60\n"
              "results:\n  - \"out/**\"\n"
              "needs:\n  gpu: false\n  venv: none\n")
    entry = ("mkdir -p out\n"
             "echo \"${JOBD_ROOT:-UNSET}\" > out/root.txt\n")
    jid = _stage_named(tmp_path, bucket, iid, "r", entry, config, "bbbb")
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
    assert v["status"] == "done", v
    seen = (bucket / "jobs" / jid / "results" / "out" / "root.txt").read_text().strip()
    assert seen == str(tmp_path / "workspace"), (
        f"JOBD_ROOT reached the entrypoint as {seen!r}, not the daemon's root")


def _cgroup_v2(tmp_path, body):
    """A fake $JOBD_CGROUP_ROOT holding a cgroup-v2 `cpu.max` (\"<quota> <period>\"
    or \"max <period>\")."""
    d = tmp_path / ("cg2-" + str(abs(hash(body)) % 10**6))
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpu.max").write_text(body + "\n")
    return d


def _cpu_cores_seen(tmp_path, iid, name, cgroup_root):
    """Run one job under a faked cgroup root; return the CPU_CORES its entrypoint
    saw (plus the JOBD_STATE_DIR it was handed)."""
    bucket, shimdir = _make_bucket(tmp_path)
    config = ("version: 1\nname: cpuprobe\nentrypoint: run.sh\ntimeout_s: 60\n"
              "results:\n  - \"out/**\"\n"
              "needs:\n  gpu: false\n  venv: none\n")
    entry = ("mkdir -p out\n"
             "echo \"$CPU_CORES|$JOBD_STATE_DIR\" > out/cpu.txt\n")
    jid = _stage_named(tmp_path, bucket, iid, name, entry, config, "cccc")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CGROUP_ROOT": str(cgroup_root)})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
    assert v["status"] == "done", v
    line = (bucket / "jobs" / jid / "results" / "out" / "cpu.txt").read_text().strip()
    cores, _, state_dir = line.partition("|")
    return cores, state_dir


def test_jobd_cpu_cores_is_the_cgroup_quota_not_nproc(tmp_path):
    """MEASURED (BOX_SATURATION_AUDIT 2026-07-30 §5 rec 8 / §8): vast hands the
    container a cpuset far wider than the CFS quota — `nproc` 384 vs `cpu.max
    18432000 100000` = 184.32 (2.08x), and 96 vs 36.86 (2.6x) on the successor
    box. CPU_CORES is what an autotune / a SCORE_WORKERS=$CPU_CORES/N heuristic
    trusts, so it must be min(nproc, floor(quota/period)) — exceeding a CFS quota
    stalls every thread in the cgroup for the rest of each 100ms period."""
    cores, state_dir = _cpu_cores_seen(
        tmp_path, 90017, "q", _cgroup_v2(tmp_path, "200000 100000"))
    assert cores == str(min(os.cpu_count() or 1, 2)), \
        f"CPU_CORES {cores!r} did not honor the 2-core cgroup quota"
    # the live sibling census the on-box CPU-rightsizing reads (cpu_budget.py)
    assert state_dir.endswith("/.state"), f"JOBD_STATE_DIR not exported: {state_dir!r}"


def test_jobd_cpu_cores_unlimited_cgroup_falls_back_to_nproc(tmp_path):
    """`cpu.max` = "max <period>" means NO quota — the cpuset width is then the
    honest answer and CPU_CORES must not shrink."""
    cores, _ = _cpu_cores_seen(tmp_path, 90018, "u",
                               _cgroup_v2(tmp_path, "max 100000"))
    assert cores == str(os.cpu_count() or 1), \
        f"CPU_CORES {cores!r} != nproc under an unlimited cgroup"


def test_jobd_cpu_cores_reads_cgroup_v1_quota(tmp_path):
    """cgroup v1 boxes (no cpu.max) carry the same allowance in the
    cfs_quota_us / cfs_period_us pair; -1 quota = unlimited."""
    d = tmp_path / "cg1"
    (d / "cpu").mkdir(parents=True)
    (d / "cpu" / "cpu.cfs_quota_us").write_text("300000\n")
    (d / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    cores, _ = _cpu_cores_seen(tmp_path, 90019, "v", d)
    assert cores == str(min(os.cpu_count() or 1, 3)), \
        f"CPU_CORES {cores!r} did not honor the cgroup-v1 3-core quota"


def test_jobd_gpu_ram_export_is_min_across_assigned_cards(tmp_path):
    """A whole-box (gpus:all) job on a HETEROGENEOUS 24GB+48GB box must see the
    MIN (24) — the OOM-safe floor, since the smallest card bounds what fits."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90016
    config = (
        "version: 1\nname: minprobe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpus: all\n  venv: none\n")
    entry = "mkdir -p out\necho \"$JOB_GPU_RAM_GB|$JOB_GPU_COUNT\" > out/env.txt\n"
    jid = _stage_named(tmp_path, bucket, iid, "i", entry, config, "bbbb")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:24,1:48", "JOBD_SKIP_GPU": "0"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
    assert v["status"] == "done", v
    line = (bucket / "jobs" / jid / "results" / "out" / "env.txt").read_text().strip()
    ram, count = line.split("|")
    assert ram == "24", f"min-across-cards wrong (got {ram!r}, want 24)"
    assert count == "2", f"JOB_GPU_COUNT wrong: {count!r}"


# ---------------------------------------------------------------------------
# v2.1: default idle self-park on queue drain
# ---------------------------------------------------------------------------
def _box_events(bucket, iid):
    d = bucket / "jobs" / "nodes" / str(iid) / "events"
    out = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            out.append(f.read_text())
    return out


def _park_daemon(tmp_path, bucket, shimdir, iid, marker, extra_env=None):
    """Run the jobd DAEMON loop (not JOBD_ONCE) with the self-park replaced by a
    test seam (JOBD_PARK_CMD touches `marker`, then jobd exits). Fast poll +
    tiny/zero deadlines so a drain parks within a second. Returns the Popen."""
    env = {
        "JOBD_POLL": "1",
        "JOBD_PARK_CMD": f"touch {shlex.quote(str(marker))}",
        "JOBD_VAPI": "http://127.0.0.1:0/never",   # belt+braces: no real API even if seam missed
    }
    if extra_env:
        env.update(extra_env)
    return _popen_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)


def _kill(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def test_jobd_parks_after_drain_grace(tmp_path):
    """A box that ran a job to completion self-parks once the queue drains — and
    every result was already pushed to B2 BEFORE the park event (results.DONE.json
    present alongside parked_self)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90020
    marker = tmp_path / "parked"
    job_id, _ = _stage_job(tmp_path, bucket, iid)     # fast rc=0 job
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK_S": "0", "JOBD_NO_JOB_PARK_S": "600"})
    try:
        p.wait(timeout=60)                            # JOBD_PARK_CMD -> jobd exits
    finally:
        _kill(p)
    assert marker.exists(), "box never self-parked after drain"
    # the job finished (results pushed) BEFORE the park
    assert (bucket / "jobs" / job_id / "results.DONE.json").is_file()
    assert jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})["status"] == "done"
    # a parked_self box event landed, reason=drained, with the done tally
    box = jm.fold_box_events(_box_events(bucket, iid))
    assert box["parked"] and box["park_reason"] == "drained", box
    assert box["n_done"] == 1 and box["n_failed"] == 0, box


def test_jobd_single_arm_uses_short_grace(tmp_path):
    """A single-arm box (exactly one job ran) with NO pinned JOBD_IDLE_PARK_S
    parks on the shorter SINGLE grace, not the 600s multi-job default — a done
    training box should reclaim GPU billing fast (FIX task #60)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90030
    marker = tmp_path / "parked"
    job_id, _ = _stage_job(tmp_path, bucket, iid)     # one fast rc=0 job
    # No JOBD_IDLE_PARK_S (so the auto-lower is eligible); single grace 0 = park
    # immediately once drained; a HIGH multi-job default would (wrongly) hold it.
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK_S_SINGLE": "0",
                                "JOBD_NO_JOB_PARK_S": "600"})
    try:
        p.wait(timeout=60)
    finally:
        _kill(p)
    assert marker.exists(), "single-arm box never parked on the short grace"
    box = jm.fold_box_events(_box_events(bucket, iid))
    assert box["parked"] and box["park_reason"] == "drained", box
    assert box["n_done"] == 1 and box["n_failed"] == 0, box


def test_jobd_pinned_grace_not_auto_lowered(tmp_path):
    """An EXPLICIT JOBD_IDLE_PARK_S always wins — a single-arm box does NOT get
    silently re-graced by the single-arm auto-lower (operator intent is law)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90031
    marker = tmp_path / "parked"
    job_id, _ = _stage_job(tmp_path, bucket, iid)     # one fast rc=0 job
    # Operator PINS a long grace; the single default is 0. Explicit must win ->
    # the box must NOT park within a few poll ticks.
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK_S": "600",
                                "JOBD_IDLE_PARK_S_SINGLE": "0",
                                "JOBD_NO_JOB_PARK_S": "600"})
    try:
        assert _wait_for_event(bucket, job_id, "done"), "job never finished"
        time.sleep(4)                                 # several poll ticks
        assert not marker.exists(), "pinned grace was auto-lowered (should not be)"
    finally:
        _kill(p)


def test_jobd_never_any_job_parks(tmp_path):
    """The owner's stated worst case: a box that NEVER receives a job must still
    park (longer deadline), reason=no_job."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90021
    marker = tmp_path / "parked"
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK_S": "600", "JOBD_NO_JOB_PARK_S": "0"})
    try:
        p.wait(timeout=60)
    finally:
        _kill(p)
    assert marker.exists(), "idle box with no job never parked"
    box = jm.fold_box_events(_box_events(bucket, iid))
    assert box["parked"] and box["park_reason"] == "no_job", box
    assert box["n_done"] == 0 and box["n_failed"] == 0, box


def test_jobd_does_not_park_with_running_job(tmp_path):
    """A box with a job RUNNING must not park, even past the drain grace — a
    running job resets the idle clock every tick."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90022
    marker = tmp_path / "parked"
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry="mkdir -p out\nsleep 30\n")
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK_S": "0", "JOBD_NO_JOB_PARK_S": "0"})
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        time.sleep(4)                                 # several poll ticks
        assert not marker.exists(), "parked while a job was still running"
    finally:
        _kill(p)


def test_jobd_does_not_park_with_queued_job(tmp_path):
    """A box with a non-terminal ticket it has not finished (here: unschedulable
    because CPU slots are 0) must not park — pending work keeps it alive."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90023
    marker = tmp_path / "parked"
    job_id, _ = _stage_job(tmp_path, bucket, iid)     # a plain CPU job
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK_S": "0", "JOBD_NO_JOB_PARK_S": "0",
                                "JOBD_CPU_SLOTS": "0"})   # nothing can be claimed
    try:
        time.sleep(4)
        assert not marker.exists(), "parked with a queued, unfinished ticket"
        # and indeed the ticket was never claimed (stayed queued)
        assert _events(bucket, job_id) == [], "ticket unexpectedly claimed"
    finally:
        _kill(p)


def test_jobd_idle_park_opt_out(tmp_path):
    """JOBD_IDLE_PARK=0 disables self-park entirely (old always-on behavior)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90024
    marker = tmp_path / "parked"
    p = _park_daemon(tmp_path, bucket, shimdir, iid, marker,
                     extra_env={"JOBD_IDLE_PARK": "0",
                                "JOBD_IDLE_PARK_S": "0", "JOBD_NO_JOB_PARK_S": "0"})
    try:
        time.sleep(4)
        assert not marker.exists(), "parked despite JOBD_IDLE_PARK=0"
        # No PARK-lane box event. (The boot scratch probe, P4e, also writes to
        # this stream and is unrelated to parking — filter it out rather than
        # asserting the stream is empty.)
        kinds = [json.loads(b)["event"] for b in _box_events(bucket, iid)]
        assert [k for k in kinds if k != "scratch_probe"] == [], \
            f"emitted a box event with park opted out: {kinds}"
    finally:
        _kill(p)


def test_jobd_drained_event_when_no_self_control_key(tmp_path):
    """If the box has NO self-control key (and no test seam), jobd cannot stop
    itself: it emits ONE `drained` box event so the laptop supervise/CLI parks it,
    instead of silently looping. (Here we drop JOBD_PARK_CMD and null the key.)"""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90025
    p = _popen_jobd(tmp_path, bucket, shimdir, iid,
                    extra_env={"JOBD_POLL": "1", "JOBD_NO_JOB_PARK_S": "0",
                               "JOBD_IDLE_PARK_S": "0",
                               "CONTAINER_API_KEY": "", "VASTAI_API_KEY": "",
                               "HOME": str(tmp_path / "nohome")})
    try:
        end = time.time() + 20
        while time.time() < end:
            box = jm.fold_box_events(_box_events(bucket, iid))
            if box["drained_pending"]:
                break
            time.sleep(0.2)
        box = jm.fold_box_events(_box_events(bucket, iid))
        assert box["drained_pending"] and not box["parked"], box
        assert box["park_reason"] == "no_job", box
        # exactly one drained event (not re-emitted every tick)
        kinds = [json.loads(b)["event"] for b in _box_events(bucket, iid)]
        assert kinds.count("drained") == 1, kinds
    finally:
        _kill(p)


# ---------------------------------------------------------------------------
# v2.1: provision-time jobd bootstrap (launch --jobs -> onstart/jobd_boot.sh)
# ---------------------------------------------------------------------------
def _stage_boot(tmp_path, bucket, iid, shimdir):
    """Stage the flat daemon tar exactly like herdd._stage_jobd_bootstrap and
    return (snippet-with-sha, env) for running the onstart stanza offline."""
    import hashlib
    files = [os.path.join(_HERE, "onstart", "jobd.sh"),
             os.path.join(_HERE, "onstart", "jobd.py"),
             os.path.join(_HERE, "jobmeta.py"),
             os.path.join(_HERE, "runmeta.py"),
             os.path.join(_HERE, "b2_sync.sh")]
    staging = tmp_path / "stage"
    staging.mkdir()
    for f in files:
        shutil.copy(f, staging / os.path.basename(f))
    tar = jm.deterministic_tar_bytes(str(staging))
    sha = hashlib.sha256(tar).hexdigest()
    boot_dir = bucket / "jobs" / "jobd-boot"
    boot_dir.mkdir(parents=True)
    (boot_dir / f"{sha}.tar").write_bytes(tar)
    snippet = open(os.path.join(_HERE, "onstart", "jobd_boot.sh")).read() \
        .replace("@JOBD_BUNDLE_SHA@", sha)
    env = _hermetic_env(tmp_path)                 # HOME=<tmp>/home: never the
    env["PATH"] = f"{shimdir}:{env['PATH']}"      # real rclone.conf, and no
    env["FAKE_BUCKET"] = str(bucket)              # broker identity leaks in
    env["B2_BUCKET"] = "testbucket"
    env["B2_KEY_ID"] = "kid-xyz"
    env["B2_APPLICATION_KEY"] = "akey-xyz"
    env["B2_S3_ENDPOINT"] = "https://s3.example"
    env["B2_REGION"] = "us-west-004"
    env["INSTANCE_ID"] = str(iid)
    env["JOBD_BOOT_WS"] = str(tmp_path / "ws")    # never touch the real /workspace
    env["JOBD_BOOT_DIR"] = str(tmp_path / "install")
    env["JOBD_BOOT_NO_START"] = "1"
    return snippet, env


def test_jobd_boot_pulls_and_extracts(tmp_path):
    """The launch-onstart stanza: given the daemon tar pre-staged on B2 (as
    herdd's _stage_jobd_bootstrap does), jobd_boot.sh pulls it, extracts the
    flat daemon files, and writes jobd.env with the instance id + idle-park
    passthrough — WITHOUT baking any secret literal into the stanza itself."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90030
    snippet, env = _stage_boot(tmp_path, bucket, iid, shimdir)
    env["JOBD_IDLE_PARK"] = "0"                    # opt-out passthrough
    # the stanza has NO secret literal — only $B2_* references
    assert "secret_access_key = ${B2_APPLICATION_KEY}" in snippet

    install = tmp_path / "install"
    # setsid'd from a non-leader background job -> the worker stays a direct
    # child of this bash, so `wait` still sees it finish
    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    bootlog = tmp_path / "ws" / "jobd-boot.log"
    diag = bootlog.read_text() if bootlog.exists() else "(no boot log)"
    assert (install / "jobd.sh").is_file(), f"stderr={r.stderr} log={diag}"
    assert (install / "jobmeta.py").is_file() and (install / "runmeta.py").is_file()
    envtext = (install / "jobd.env").read_text()
    assert f"export INSTANCE_ID={iid}" in envtext
    assert "export JOBD_IDLE_PARK=0" in envtext
    assert "export B2_BUCKET=testbucket" in envtext
    # bootstrap output landed in the persistent log (diagnosable after death)
    assert "extracted, not starting jobd" in diag


def test_jobd_boot_stanza_is_reap_proof_by_construction():
    """The live 44482324 regression, statically: vast's onstart runner reaps its
    process group on exit, so the stanza must daemonize the WHOLE bootstrap —
    setsid (nohup fallback), stdio fully detached, output to a persistent log —
    and must NOT background a bare subshell from the onstart shell."""
    text = open(os.path.join(_HERE, "onstart", "jobd_boot.sh")).read()
    assert "setsid bash" in text and "nohup bash" in text
    # both spawn paths detached: stdin from /dev/null, stdout+stderr to the log
    for launcher in ("setsid bash", "nohup bash"):
        line = next(l for l in text.splitlines() if l.strip().startswith(launcher))
        assert "</dev/null" in line and "jobd-boot.log" in line and "2>&1" in line \
            and line.rstrip().endswith("&"), line
    # the old broken shape is gone: no inline-backgrounded `( set -u ...` subshell
    assert "( set -u" not in text
    # CONTAINER_ID fallback (INSTANCE_ID absent on real boxes — live-verified)
    assert "${INSTANCE_ID:-${CONTAINER_ID:-}}" in text


def test_jobd_boot_survives_parent_reap(tmp_path):
    """Dynamic replay of the incident: the onstart shell exits and its process
    group is SIGKILLed (what vast's runner does) while the bootstrap is still
    mid-pull — the setsid'd worker must survive and finish the install."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90031
    snippet, env = _stage_boot(tmp_path, bucket, iid, shimdir)
    # slow transport: every rclone call sleeps first, so the install is still
    # in flight when the parent group is reaped
    slowdir = tmp_path / "slowbin"
    slowdir.mkdir()
    slow = slowdir / "rclone"
    slow.write_text(f"#!/usr/bin/env bash\nsleep 1\nexec {shimdir}/rclone \"$@\"\n")
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    env["PATH"] = f"{slowdir}:{env['PATH']}"

    p = subprocess.Popen(["bash", "-c", snippet], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    p.wait(timeout=30)                 # stanza returns right after backgrounding
    try:                               # vast reaps the onstart process group
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    install = tmp_path / "install"
    end = time.time() + 30
    while time.time() < end and not (install / "jobd.env").is_file():
        time.sleep(0.1)
    assert (install / "jobd.sh").is_file() and (install / "jobd.env").is_file(), \
        "bootstrap died with the onstart process group (the 44482324 regression)"


def test_jobd_preempt_does_not_mark_running_job_failed(tmp_path):
    """Regression for the 2026-07-12 budget-park bug (box 44566398): when vast stops
    a box (eviction, `supervise` budget-park, idle self-park) while a job is
    RUNNING, the daemon's SIGTERM trap raises .preempting — and the runner, seeing
    its killed entrypoint exit NON-zero, must treat that as an INTERRUPTION, not a
    crash: no `failed` event, no terminal marker, no results.DONE.json. Otherwise
    the job is skipped forever on resume and a run that finished locally is lost."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90050
    # entrypoint: prove it started, sleep past the SIGTERM, then exit NON-zero —
    # stands in for a torchrun rank the box-stop SIGKILLs after training finished.
    entry = "mkdir -p out\necho run > out/x.txt\nsleep 3\nexit 9\n"
    config = (
        "version: 1\nname: preempt-fail-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    pgid = os.getpgid(p.pid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        os.kill(p.pid, signal.SIGTERM)     # park: trap raises .preempting + preempted
        p.wait(timeout=60)                 # daemon exits; the orphaned runner lives on
        time.sleep(7)                      # entrypoint sleeps 3s, then the runner decides
        term = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.terminal"
        assert not term.is_file(), \
            f"preempted job wrongly marked terminal: {term.read_text().strip()!r}"
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "failed" not in kinds, f"kinds={kinds}"
    assert kinds.count("preempted") == 1, f"kinds={kinds}"
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] not in jm.TERMINAL, f"status={v['status']}"
    # the runner returned before publish — no clean-finish claim for an unfinished job
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file()


def test_jobd_boot_surfaces_b2_auth_failure(tmp_path):
    """A dead / revoked box B2 key must produce a LOUD, diagnosable boot log — the
    2026-07-12 incident's log said only 'pull/extract failed — no jobd' with no
    cause, hiding an InvalidAccessKeyId (the box's ephemeral key was revoked
    mid-session by a colliding concurrent `launch --jobs`). jobd_boot.sh must name
    the auth failure, print the rotate command, and NOT waste all its backoff
    rounds on a non-transient error."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90032
    snippet, env = _stage_boot(tmp_path, bucket, iid, shimdir)
    # rclone that AUTH-FAILS the bundle pull (simulates the dead key)
    authdir = tmp_path / "authbin"
    authdir.mkdir()
    rc = authdir / "rclone"
    rc.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  listremotes) echo 'b2:'; exit 0 ;;\n"
        "  cat) echo \"ERROR: SerializeHTTPError: InvalidAccessKeyId: The key "
        "'004e658565e1dd30000000033' is not valid\" >&2; exit 1 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n")
    rc.chmod(rc.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    env["PATH"] = f"{authdir}:{env['PATH']}"

    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stderr
    bootlog = (tmp_path / "ws" / "jobd-boot.log").read_text()
    assert "B2 AUTH FAILURE" in bootlog, bootlog
    assert "InvalidAccessKeyId" in bootlog, bootlog
    assert "herdd job attach" in bootlog, bootlog
    # non-transient -> broke out early, did NOT grind through all 5 backoff rounds
    assert "attempt 5/5" not in bootlog, bootlog
    # no daemon was installed from a failed pull
    assert not (tmp_path / "install" / "jobd.sh").is_file()


# ---------------------------------------------------------------------------
# N1(a): loud checkpoint-sync failure (never a silent freeze)
# ---------------------------------------------------------------------------
def test_jobd_checkpoint_sync_auth_failure_is_loud_and_nonfatal(tmp_path):
    """box 44566398: a dead/rotated B2 key silently FROZE checkpoint sync while
    compute ran on, stranding a finished adapter. The mid-run sync loop must no
    longer swallow rclone errors: on an auth-class failure it emits a distinct
    `checkpoint_sync_failed` event (auth cause named) + drops a named breadcrumb
    file, KEEPS the job running, and keeps retrying — never another silent freeze.
    The failing sync never kills the job."""
    bucket, shimdir = _make_bucket(tmp_path)
    authdir = _authfail_shimdir(tmp_path, shimdir)
    iid = 90060
    # write a checkpoint immediately, keep alive across several sync ticks, finish
    # clean. checkpoint_s small so the (scoped) auth failure bites repeatedly.
    entry = ("mkdir -p out\necho step-1 > out/ckpt.txt\nsleep 3\n"
             "echo done > out/result.txt\nexit 0\n")
    config = (
        "version: 1\nname: authfail-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    r = _run_jobd(tmp_path, bucket, authdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s", "RCLONE_FAIL_COPY_AUTH": "1"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    # LOUD: at least one distinct sync-failure event, naming the auth cause
    fails = [json.loads(b) for b in bodies
             if json.loads(b)["event"] == "checkpoint_sync_failed"]
    assert fails, f"no checkpoint_sync_failed event: kinds={kinds}"
    assert any("InvalidAccessKeyId" in (f.get("reason") or "")
               or "AUTH" in (f.get("reason") or "").upper() for f in fails), fails
    # named diagnostic breadcrumb persisted on the box disk (survives a dead B2)
    crumb = tmp_path / "workspace" / "jobs" / job_id / ".checkpoint_sync_failed"
    assert crumb.is_file() and crumb.read_text().strip(), "no breadcrumb"
    # NON-FATAL: the sync failure never killed the job — it ran on and finished
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"sync failure wrongly killed the job: {v}"


def test_jobd_checkpoint_sync_failure_event_is_rate_limited(tmp_path):
    """A long outage must not spam the event log: the checkpoint_sync_failed event
    is rate-limited (1st failure, then every JOBD_SYNC_FAIL_EVERY-th). With a tiny
    interval and a long-lived job the raw failure count far exceeds the events."""
    bucket, shimdir = _make_bucket(tmp_path)
    authdir = _authfail_shimdir(tmp_path, shimdir)
    iid = 90063
    entry = ("mkdir -p out\necho step-1 > out/ckpt.txt\nsleep 6\n"
             "echo done > out/result.txt\nexit 0\n")
    config = (
        "version: 1\nname: ratelimit-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    r = _run_jobd(tmp_path, bucket, authdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s", "RCLONE_FAIL_COPY_AUTH": "1",
                             "JOBD_SYNC_FAIL_EVERY": "1000"})
    assert r.returncode == 0, r.stderr
    n_fail_ev = sum(1 for b in _events(bucket, job_id)
                    if json.loads(b)["event"] == "checkpoint_sync_failed")
    # with a rate of 1000, only the FIRST failure emits over a ~6s outage
    assert n_fail_ev == 1, f"rate-limit broken: {n_fail_ev} events"
    crumb = tmp_path / "workspace" / "jobs" / job_id / ".checkpoint_sync_failed"
    assert crumb.is_file(), "breadcrumb missing despite rate-limited events"


def test_train_sh_sync_loop_is_loud_on_auth_failure():
    """N1(a) run-path (onstart/train.sh): the periodic checkpoint-stream loop must
    also stop swallowing rclone errors — capture stderr, and on an auth-class
    failure emit a distinct checkpoint_sync_failed event + a persistent breadcrumb,
    while never killing the run. Verified structurally (train.sh has no on-box test
    harness; same discipline as the jobd_boot reap-proof structural test)."""
    text = open(os.path.join(_HERE, "onstart", "train.sh")).read()
    lo = text.index("stream checkpoints out")
    hi = text.index("--- 2b.", lo) if "--- 2b." in text[lo:] else text.index("--- 3.", lo)
    loop = text[lo:hi]
    # the checkpoint-sync copy captures stderr to a file — it must NOT swallow it
    sync_line = next(l for l in loop.splitlines()
                     if "rclone copy" in l and "--min-age 45s" in l)
    assert "2>/dev/null" not in sync_line, \
        f"train.sh sync copy still swallows rclone stderr: {sync_line.strip()}"
    assert "2>" in sync_line, f"sync copy does not capture stderr: {sync_line.strip()}"
    assert "checkpoint_sync_failed" in loop, "no distinct sync-failure event"
    assert ".checkpoint_sync_failed" in loop, "no breadcrumb file"
    # auth-class detection present
    assert "InvalidAccessKeyId" in loop and "SignatureDoesNotMatch" in loop, \
        "no auth-class classification in the sync loop"


# ---------------------------------------------------------------------------
# N1(b): the final preempt flush must cover results globs too
# ---------------------------------------------------------------------------
def test_jobd_preempt_flush_covers_results_globs(tmp_path):
    """N1(b): _jobd_preempt flushed only .checkpoint.globs. A job with NO
    checkpoint_s that had already WRITTEN its result files, interrupted mid-run by
    a box stop, would strand them (the runner's publish runs post-wait, which the
    box stop pre-empts). The trap now also flushes .results.globs (bounded timeout,
    no --min-age) so the freshest results land before the box dies."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90061
    entry = "mkdir -p out\necho final-result > out/final.txt\nsleep 60\n"
    config = (
        "version: 1\nname: preempt-results-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")   # NOTE: no checkpoint_s
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        time.sleep(1.5)                    # entrypoint writes out/final.txt
        os.kill(p.pid, signal.SIGTERM)     # external stop -> trap
        p.wait(timeout=60)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # the trap's RESULTS flush shipped the result despite no checkpoint_s
    res = bucket / "jobs" / job_id / "results" / "out" / "final.txt"
    assert res.is_file() and res.read_text().strip() == "final-result", \
        "results glob not flushed on preempt"
    # preempted (non-terminal) emitted; no clean-finish DONE marker
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert kinds.count("preempted") == 1, kinds
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file()


# ---------------------------------------------------------------------------
# N1(c): preempt-resumes must NOT burn max_restarts
# ---------------------------------------------------------------------------
def test_jobd_preempt_resume_does_not_burn_restart_cap(tmp_path):
    """N1(c): a preempt-mediated resume must NOT count against max_restarts. With
    max_restarts=0 (crash budget = 1 run) a single graceful box-stop preempt used
    to fail the job terminally on the next boot (box 44566398 lineage: three
    outbids -> spurious terminal `failed`). The trap now drops a per-job .preempted
    breadcrumb; the next claim counts it against a separate, generous preempt cap,
    leaving the crash budget intact, so the job resumes and finishes."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90062
    entry = (
        "mkdir -p out\n"
        "if [ -f out/state.txt ]; then\n"
        "  echo done > out/result.txt\n  exit 0\nfi\n"
        "echo step-1 > out/state.txt\nsleep 60\n")
    config = (
        "version: 1\nname: preempt-cap-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "max_restarts: 0\ncheckpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id = _stage_named(tmp_path, bucket, iid, "pc", entry, config, "pcpc")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid,
                    extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    pgid = os.getpgid(p.pid)               # capture BEFORE wait (reaped pid -> getpgid raises)
    try:
        assert _wait_for_event(bucket, job_id, "checkpoint"), "state never synced"
        os.kill(p.pid, signal.SIGTERM)     # graceful preempt: trap raises .preempting + breadcrumb
        p.wait(timeout=60)                  # daemon runs the trap, then exits 143
        # promptly reap the ORPHANED runner so it can't self-complete its sleep and
        # publish `done` on its own — a real box stop SIGKILLs the container.
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(1)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # the per-job breadcrumb survived the box stop (it, unlike .preempting, is not
    # cleared at the next boot — it is what tells the claim this is a preempt-resume)
    crumb = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.preempted"
    assert crumb.is_file(), "trap did not drop the per-job .preempted breadcrumb"

    # next boot: the preempt-resume finishes cleanly, NOT a restart-cap `failed`
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "resumed" in kinds, f"kinds={kinds} stderr={r.stderr}"
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", \
        f"preempt-resume wrongly hit restart cap: kinds={kinds} status={v['status']}"
    assert "restart cap" not in (v["fail_reason"] or "")
    # the breadcrumb is consumed once (not left to double-count a later crash)
    assert not crumb.is_file(), "preempt breadcrumb not consumed at claim"
    # provenance: the resume records WHICH evidence classified it (trap beats
    # the boot-nonce inference when a signal actually arrived)
    resumed = [json.loads(b) for b in bodies if json.loads(b)["event"] == "resumed"]
    assert resumed and resumed[-1].get("kind") == "preempt", resumed
    assert resumed[-1].get("detect") == "trap", resumed
    # the trap-evidenced `preempted` and the `resumed{kind:preempt}` it precedes
    # are the SAME interruption — the view must not double-count them
    assert v["attempts"] == 1 and v["n_preempted"] == 1


# ---------------------------------------------------------------------------
# Signal-less preempt inference (2026-08-06): vast delivers NO stop signal —
# measured three independent ways (SPOT_DESIGN §1: 0 `preempted`/0 `final_flush`
# across 61 runs; HANDOFF_DESIGN O1 negative x3; every post-eviction resume of
# the v11 training job read "kind":"crash"). So the trap breadcrumb above almost
# never exists on a REAL eviction, and the class must be inferred at resume time
# from the container-boot nonce (tmpfs — dies with the box).
# ---------------------------------------------------------------------------
def test_jobd_signalless_box_restart_classifies_preempt(tmp_path):
    """A hard box death with NO signal (the measured vast reality), followed by a
    box stop/start (tmpfs wiped), must classify the resume as kind=preempt and
    drain JOBD_PREEMPT_CAP — not max_restarts. Staged with max_restarts=0 so the
    misclassification is TERMINAL: under the old kind=crash reading this exact
    sequence fails the job at the restart cap on the resume boot (the v11
    livelock contributor: a healthy job on a contested spot market burning a
    crash budget of 2 instead of a preempt budget of 20)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90071
    entry = (
        "mkdir -p out\n"
        "if [ -f out/state.txt ]; then\n"
        "  echo done > out/result.txt\n  exit 0\nfi\n"
        "echo step-1 > out/state.txt\nsleep 60\n")
    config = (
        "version: 1\nname: evict-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "max_restarts: 0\ncheckpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id = _stage_named(tmp_path, bucket, iid, "ev", entry, config, "evev")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid,
                    extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    pgid = os.getpgid(p.pid)
    try:
        assert _wait_for_event(bucket, job_id, "checkpoint"), "state never synced"
        # a REAL eviction: the box dies with NO signal to the daemon
        os.killpg(pgid, signal.SIGKILL)
        p.wait(timeout=30)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    state = tmp_path / "workspace" / "jobs" / ".state"
    # the whole point: no trap ran, so no breadcrumb exists
    assert not (state / f"{job_id}.preempted").is_file(), \
        "test premise broken: a trap breadcrumb appeared without a signal"

    # the box is stopped and restarted: the container tmpfs dies with it.
    # (missing_ok so that against a jobd WITHOUT the nonce mechanism this test
    # reaches — and fails on — the behavioral assertion below, not the wipe)
    _fake_shm(tmp_path).unlink(missing_ok=True)

    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", \
        f"signal-less eviction resume hit the crash cap: kinds={kinds} " \
        f"reason={v['fail_reason']!r}"
    resumed = [json.loads(b) for b in bodies if json.loads(b)["event"] == "resumed"]
    assert resumed and resumed[-1].get("kind") == "preempt", \
        f"eviction resume misclassified: {resumed}"
    assert resumed[-1].get("detect") == "boot_change", resumed
    # the budgets drained the right way round: one preempt, one (fresh) attempt
    assert (state / f"{job_id}.preempts").read_text().strip() == "1"
    assert (state / f"{job_id}.attempts").read_text().strip() == "1"
    # `job status`'s view must mirror those durable on-box counters (2026-08-09
    # drill: the pre-fix fold read attempts=2/n_preempted=0 from this exact
    # signal-less shape — a clean eviction reported as a crash that burned the
    # restart budget)
    assert v["attempts"] == 1 and v["n_preempted"] == 1


def test_jobd_same_boot_resume_stays_crash(tmp_path):
    """The inference must not over-correct: a runner that died while the
    CONTAINER stayed up (nonce unchanged — e.g. the OOM killer took the process
    tree) is a genuine crash and still drains max_restarts. With max_restarts=0
    the resume boot refuses at the restart cap, exactly the pre-inference
    behavior — a crash-loop must never ride the generous preempt cap."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90072
    config = (
        "version: 1\nname: livecrash-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "max_restarts: 0\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id = _stage_named(tmp_path, bucket, iid, "lc", "mkdir -p out\nsleep 60\n",
                          config, "lclc")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    pgid = os.getpgid(p.pid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        os.killpg(pgid, signal.SIGKILL)
        p.wait(timeout=30)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # the nonce file SURVIVES (same container boot) -> the next daemon reads the
    # same nonce and must classify the death as a crash
    assert _fake_shm(tmp_path).is_file(), "test premise broken: nonce file gone"
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed" and "restart cap" in (v["fail_reason"] or ""), v


# ---------------------------------------------------------------------------
# N4: declarative `assets:` staging (shared pull primitive + integrity)
# ---------------------------------------------------------------------------
def _put_asset_files(bucket, b2prefix, files):
    """Write {relpath: str|bytes} under $bucket/<b2prefix>/ (the fake B2 tree)."""
    base = bucket
    for part in b2prefix.split("/"):
        base = base / part
    for rel, content in files.items():
        p = base
        for part in rel.split("/"):
            p = p / part
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
    return base


def _stage_asset_job(tmp_path, bucket, iid, name, entry, assets, nonce4="a1a1",
                     results=("out/**",)):
    """Stage a job whose ticket carries an `assets:` list (built as a dict — the
    YAML fallback can't express a list-of-maps, same as the experiment block)."""
    src = tmp_path / f"asrc-{name}-{nonce4}"
    src.mkdir()
    (src / "run.sh").write_text(entry)
    raw = {"version": 1, "name": name, "entrypoint": "run.sh", "timeout_s": 60,
           "results": list(results), "needs": {"gpu": False, "venv": "none"},
           "assets": assets}
    cfg, _ = jm.validate_job_config(raw, str(src))
    tmp_bundle = tmp_path / f"{name}-{nonce4}.tar.zst"
    sha = jm.write_bundle(str(src), str(tmp_bundle))["sha256"]
    bdir = bucket / "jobs" / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_bundle), str(bdir / f"{sha}.tar.zst"))
    job_id = jm.mint_job_id(cfg["name"], ts="20260712T000000", nonce4=nonce4)
    qdir = bucket / "jobs" / "queue" / str(iid)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{job_id}.json").write_text(
        json.dumps(jm.make_ticket(job_id, sha, "cli:test", cfg, str(iid))))
    return job_id


def _prepare_assets_tsv(tmp_path, cfg, out):
    """Run the REAL `jobd.py prepare` for one config and return its asset TSV
    path — the wire format jobd.sh parses, not a reimplementation of it."""
    ticket = tmp_path / f"ticket-{out.name}.json"
    ticket.write_text(json.dumps(
        jm.make_ticket("j-1", "0" * 64, "cli:test", cfg, "1")))
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "onstart", "jobd.py"), "prepare",
         str(ticket), "--env-out", str(tmp_path / f"env-{out.name}"),
         "--results-out", str(tmp_path / f"res-{out.name}"),
         "--assets-out", str(out)],
        check=True, capture_output=True)
    return out


def _asset_env(pulllog=None):
    env = {"JOBD_ASSET_BACKOFF": "0"}    # no backoff sleeps in tests
    if pulllog is not None:
        env["JOBD_ASSET_PULLLOG"] = str(pulllog)
    return env


def test_jobd_asset_fresh_pull(tmp_path):
    """A fresh box pulls the asset from B2 into the stable cache
    /workspace/assets/<name> before the entrypoint, writes a .complete byte-total
    marker, and the job runs to done."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90070
    _put_asset_files(bucket, "assets-src/base",
                     {"config.json": '{"ok":1}', "model.safetensors": "WEIGHTS"})
    pulllog = tmp_path / "pulllog"
    job_id = _stage_asset_job(tmp_path, bucket, iid, "afresh",
                              "mkdir -p out\necho ok > out/r.txt\n",
                              [{"name": "base", "b2": "assets-src/base"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env(pulllog))
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"
    cache = tmp_path / "workspace" / "assets" / "base"
    assert (cache / "config.json").read_text() == '{"ok":1}'
    assert (cache / "model.safetensors").read_text() == "WEIGHTS"
    marker = tmp_path / "workspace" / "assets" / ".base.complete"
    assert marker.is_file() and int(marker.read_text().strip()) > 0
    assert pulllog.read_text().split().count("base") == 1


def test_jobd_asset_skip_on_complete_marker(tmp_path):
    """Second job (same box cache) does NOT re-pull: with the .complete marker the
    asset is reused even after its B2 source is deleted (proves skip-if-present)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90071
    srcdir = _put_asset_files(bucket, "assets-src/base",
                              {"config.json": "{}", "w.safetensors": "W"})
    pulllog = tmp_path / "pulllog"
    ja = _stage_asset_job(tmp_path, bucket, iid, "askip", "mkdir -p out\necho a>out/r.txt\n",
                          [{"name": "base", "b2": "assets-src/base"}], nonce4="aaaa")
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env(pulllog))
    assert r.returncode == 0, r.stderr
    assert jm.fold_events(_events(bucket, ja), live_iids={str(iid)})["status"] == "done"
    # obliterate the B2 source — a re-pull would now fail
    shutil.rmtree(str(srcdir))
    jb = _stage_asset_job(tmp_path, bucket, iid, "askip", "mkdir -p out\necho b>out/r.txt\n",
                          [{"name": "base", "b2": "assets-src/base"}], nonce4="bbbb")
    r2 = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env(pulllog))
    assert r2.returncode == 0, r2.stderr
    assert jm.fold_events(_events(bucket, jb), live_iids={str(iid)})["status"] == "done", \
        "second job failed — cache skip did not kick in"
    # pulled exactly once, ever (the second boot reused the cache)
    assert pulllog.read_text().split().count("base") == 1


def test_jobd_asset_cache_survives_park_resume(tmp_path):
    """The asset cache lives on box disk (survives park/resume) and is reused
    across daemon boots: a sentinel dropped into the cache after boot 1 survives
    boot 2 (no re-pull would touch it)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90072
    _put_asset_files(bucket, "assets-src/m", {"config.json": "{}", "w.bin": "W"})
    ja = _stage_asset_job(tmp_path, bucket, iid, "apark", "mkdir -p out\necho a>out/r.txt\n",
                          [{"name": "m", "b2": "assets-src/m"}], nonce4="aaaa")
    assert _run_jobd(tmp_path, bucket, shimdir, iid,
                     extra_env=_asset_env()).returncode == 0
    assert jm.fold_events(_events(bucket, ja), live_iids={str(iid)})["status"] == "done"
    sentinel = tmp_path / "workspace" / "assets" / "m" / ".sentinel"
    sentinel.write_text("keep")
    # second boot (== a resume: same box disk, same $ROOT), new job, same asset
    jb = _stage_asset_job(tmp_path, bucket, iid, "apark", "mkdir -p out\necho b>out/r.txt\n",
                          [{"name": "m", "b2": "assets-src/m"}], nonce4="bbbb")
    assert _run_jobd(tmp_path, bucket, shimdir, iid,
                     extra_env=_asset_env()).returncode == 0
    assert jm.fold_events(_events(bucket, jb), live_iids={str(iid)})["status"] == "done"
    assert sentinel.is_file(), "cache was re-pulled instead of reused across boots"


def test_jobd_asset_require_glob_miss_fails_pre_entrypoint(tmp_path):
    """A `require:` glob that matches nothing post-pull fails the (non-optional)
    job terminal BEFORE the entrypoint runs, with the asset_stage_failed reason."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90073
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})   # no *.safetensors
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "areq",
        "mkdir -p out\necho RAN > out/ran.txt\n",
        [{"name": "base", "b2": "assets-src/base",
          "require": ["*.safetensors"]}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", f"kinds={kinds}"
    assert "asset_stage_failed:base" in (v["fail_reason"] or ""), v
    # the entrypoint NEVER started — no `started` event, no result marker
    assert "started" not in kinds, f"entrypoint ran despite a missing asset: {kinds}"
    assert not (bucket / "jobs" / job_id / "results.DONE.json").is_file()


def test_jobd_asset_truncated_shard_caught_by_index(tmp_path):
    """The automatic index-aware completeness check (ported from
    ensure_base_model.sh): a *.index.json naming two shards but only one present
    (a truncated download) fails the job pre-entrypoint — no require: needed."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90074
    index = json.dumps({"weight_map": {
        "a.weight": "model-00001-of-00002.safetensors",
        "b.weight": "model-00002-of-00002.safetensors"}})
    _put_asset_files(bucket, "assets-src/sharded", {
        "config.json": "{}",
        "model.safetensors.index.json": index,
        "model-00001-of-00002.safetensors": "SHARD1"})   # shard 2 MISSING (truncated)
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "ashard", "mkdir -p out\necho RAN>out/ran.txt\n",
        [{"name": "sharded", "b2": "assets-src/sharded"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", f"kinds={kinds}"
    assert "asset_stage_failed:sharded" in (v["fail_reason"] or ""), v
    assert "started" not in kinds, kinds


def test_jobd_asset_optional_tolerates_absence(tmp_path):
    """An `optional: true` asset whose postcondition can't be met is tolerated —
    the job runs to done anyway (contrast the non-optional require-miss test)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90075
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})   # no *.bin
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "aopt", "mkdir -p out\necho ok>out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base",
          "require": ["*.bin"], "optional": True}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"kinds={kinds}"
    assert "started" in kinds and "failed" not in kinds, kinds


def test_jobd_asset_dest_symlink(tmp_path):
    """A `dest:` different from the cache path is symlinked, so the entrypoint sees
    the asset at its expected relative location inside the workdir."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90076
    _put_asset_files(bucket, "assets-src/payload", {"hello.txt": "HELLO"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "adest",
        "mkdir -p out\necho \"asset=$(cat base/hello.txt)\" > out/x.txt\n",
        [{"name": "payload", "b2": "assets-src/payload", "dest": "base"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"
    x = bucket / "jobs" / job_id / "results" / "out" / "x.txt"
    assert x.read_text().strip() == "asset=HELLO"
    # the workdir dest is a symlink into the stable cache
    link = tmp_path / "workspace" / "jobs" / job_id / "work" / "base"
    assert link.is_symlink()
    assert (tmp_path / "workspace" / "assets" / "payload") == link.resolve()


def test_jobd_asset_dest_is_a_link_not_a_second_copy(tmp_path):
    """`dest:` COSTS NO EXTRA BYTES. Pins the fact a 51.77 GiB base model was
    kept OUT of `assets:` for fear of — a declared asset stages once, into the
    cache, and `dest` is a symlink to it. Measured, not assumed: the bytes exist
    at exactly one inode, and writing through the link writes the cache."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90085
    _put_asset_files(bucket, "assets-src/big", {"w.bin": "W" * 4096})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "a1copy", "mkdir -p out\necho ok>out/r.txt\n",
        [{"name": "big", "b2": "assets-src/big", "dest": "models/big"}])
    assert _run_jobd(tmp_path, bucket, shimdir, iid,
                     extra_env=_asset_env()).returncode == 0
    assert jm.fold_events(_events(bucket, job_id),
                          live_iids={str(iid)})["status"] == "done"
    ws = tmp_path / "workspace"
    cache = ws / "assets" / "big"
    dest = ws / "jobs" / job_id / "work" / "models" / "big"
    assert dest.is_symlink() and dest.resolve() == cache
    # ONE inode for the payload, reachable both ways
    assert (cache / "w.bin").stat().st_ino == (dest / "w.bin").stat().st_ino
    # …and exactly one materialized copy anywhere under the workspace root
    # (os.walk does not descend symlinks, which is exactly the question)
    payloads = [os.path.join(r, "w.bin")
                for r, _d, fs in os.walk(ws) if "w.bin" in fs]
    assert payloads == [str(cache / "w.bin")], payloads


# --- completeness receipts (`receipt:`) ------------------------------------

def _pushed(files, complete=True):
    return json.dumps({"complete": complete, "files": files,
                       "ts_utc": "2026-08-23T00:00:00Z"})


def test_jobd_asset_receipt_present_stages_and_excludes_the_marker(tmp_path):
    """Happy path: the marker gates the pull, corroborates the file count, and
    does NOT land in the staged dir (it is transport metadata — landing it adds
    a file the consumer's fingerprint has never seen)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90080
    _put_asset_files(bucket, "assets-src/merged", {
        "config.json": "{}", "model.safetensors": "W", "PUSHED.json": _pushed(2)})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcok",
        "mkdir -p out\nls merged > out/listing.txt\n",
        [{"name": "merged", "b2": "assets-src/merged", "dest": "merged",
          "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"
    cache = tmp_path / "workspace" / "assets" / "merged"
    assert (cache / "model.safetensors").read_text() == "W"
    assert not (cache / "PUSHED.json").exists(), "the receipt landed in the staged dir"
    # and the ENTRYPOINT saw the same clean tree through its dest symlink
    listing = (bucket / "jobs" / job_id / "results" / "out" / "listing.txt").read_text()
    assert "PUSHED.json" not in listing, listing
    assert "model.safetensors" in listing


def test_jobd_asset_receipt_missing_refuses_before_pulling(tmp_path):
    """No marker on B2 => the prefix is truncated or still uploading. Fail fast:
    its own terminal reason, no pull attempted, no entrypoint."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90081
    _put_asset_files(bucket, "assets-src/half", {"config.json": "{}", "w.bin": "W"})
    pulllog = tmp_path / "pulllog"
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcmiss", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "half", "b2": "assets-src/half", "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env(pulllog))
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", f"kinds={kinds}"
    assert "asset_receipt_missing:half" in (v["fail_reason"] or ""), v
    assert "started" not in kinds, kinds
    # BEFORE the pull, not after it: no bytes were spent learning this
    assert not pulllog.exists() or "half" not in pulllog.read_text().split()
    assert not (tmp_path / "workspace" / "assets" / "half" / "w.bin").exists()


def test_jobd_asset_receipt_complete_false_refuses(tmp_path):
    """A marker that is PRESENT and says complete:false is an explicit denial,
    not a missing measurement — it refuses like an absent one."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90082
    _put_asset_files(bucket, "assets-src/aborted", {
        "w.bin": "W", "PUSHED.json": _pushed(1, complete=False)})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcfalse", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "aborted", "b2": "assets-src/aborted", "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", v
    assert "asset_receipt_missing:aborted" in (v["fail_reason"] or ""), v


def test_jobd_asset_receipt_count_shortfall_refuses(tmp_path):
    """The receipt claims more files than landed => the pull dropped something
    the require: globs and the shard index are not watching."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90083
    _put_asset_files(bucket, "assets-src/short", {
        "config.json": "{}", "w.bin": "W", "PUSHED.json": _pushed(9)})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcshort", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "short", "b2": "assets-src/short", "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", f"kinds={kinds}"
    assert "asset_receipt_mismatch:short" in (v["fail_reason"] or ""), v
    assert "started" not in kinds, kinds


def test_jobd_asset_receipt_unparseable_body_never_refuses(tmp_path):
    """A marker we cannot read is not evidence of an incomplete publish: presence
    alone stands, no count is checked, and the job runs."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90084
    _put_asset_files(bucket, "assets-src/opaque", {
        "w.bin": "W", "DONE": "not json, just a touch marker\n"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcopaque", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "opaque", "b2": "assets-src/opaque", "receipt": "DONE"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"
    cache = tmp_path / "workspace" / "assets" / "opaque"
    assert (cache / "w.bin").read_text() == "W"
    assert not (cache / "DONE").exists()        # still excluded


def test_jobd_asset_receipt_optional_asset_is_tolerated(tmp_path):
    """`optional: true` keeps its meaning under the receipt gate: an unpublished
    optional asset is logged and skipped, and the job still runs."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90086
    _put_asset_files(bucket, "assets-src/extra", {"w.bin": "W"})   # no marker
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcopt", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "extra", "b2": "assets-src/extra", "optional": True,
          "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"


def test_jobd_asset_receipt_sweeps_a_marker_already_in_the_cache(tmp_path):
    """A cache staged BEFORE the declaration existed still holds the marker, and
    `sync` will not delete a file its own --exclude hides from it. The explicit
    rm is what makes the exclusion retroactive."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90087
    _put_asset_files(bucket, "assets-src/legacy", {
        "w.bin": "W", "PUSHED.json": _pushed(1)})
    cache = tmp_path / "workspace" / "assets" / "legacy"
    cache.mkdir(parents=True)
    (cache / "PUSHED.json").write_text("stale marker from a pre-receipt pull\n")
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arcsweep", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "legacy", "b2": "assets-src/legacy", "mode": "sync",
          "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"
    assert not (cache / "PUSHED.json").exists()


def test_jobd_asset_receipt_not_required_for_a_cache_already_complete(tmp_path):
    """The gate sits BELOW the .complete skip: an asset this box already holds
    must not be made to depend on B2 being reachable. Boot 2 runs with the whole
    remote prefix — marker included — deleted."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90088
    src = _put_asset_files(bucket, "assets-src/warm", {
        "w.bin": "W", "PUSHED.json": _pushed(1)})
    ja = _stage_asset_job(
        tmp_path, bucket, iid, "arcwarm", "mkdir -p out\necho a>out/r.txt\n",
        [{"name": "warm", "b2": "assets-src/warm", "receipt": "PUSHED.json"}],
        nonce4="aaaa")
    assert _run_jobd(tmp_path, bucket, shimdir, iid,
                     extra_env=_asset_env()).returncode == 0
    assert jm.fold_events(_events(bucket, ja), live_iids={str(iid)})["status"] == "done"
    shutil.rmtree(str(src))                      # B2 prefix gone entirely
    jb = _stage_asset_job(
        tmp_path, bucket, iid, "arcwarm", "mkdir -p out\necho b>out/r.txt\n",
        [{"name": "warm", "b2": "assets-src/warm", "receipt": "PUSHED.json"}],
        nonce4="bbbb")
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jb), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"


def test_jobd_asset_spec_tsv_survives_an_empty_dest_before_a_receipt(tmp_path):
    """A tab is IFS WHITESPACE to bash, so `…\\t\\treceipt` COLLAPSES and would
    land the receipt in `dest`. Pins the `-` encoding that prevents it: an asset
    with NO dest and a receipt must still stage into the cache and refuse on an
    absent marker (not silently symlink a workdir path named PUSHED.json)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90089
    _put_asset_files(bucket, "assets-src/nodest", {"w.bin": "W"})     # no marker
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "arctsv", "mkdir -p out\necho RAN>out/r.txt\n",
        [{"name": "nodest", "b2": "assets-src/nodest", "receipt": "PUSHED.json"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert "asset_receipt_missing:nodest" in (v["fail_reason"] or ""), v
    assert not (tmp_path / "workspace" / "jobs" / job_id / "work" / "PUSHED.json").exists()


def test_jobd_prepare_asset_tsv_shape(tmp_path):
    """The wire format itself: six tab-separated columns, `-` for every absent
    optional field, and a bash split that recovers each field intact."""
    cfg = {"version": 1, "name": "tsv", "entrypoint": "run.sh", "timeout_s": 60,
           "results": ["out/**"], "needs": {"gpu": False, "venv": "none"},
           "assets": [{"name": "a", "b2": "x/y", "receipt": "PUSHED.json"},
                      {"name": "b", "b2": "p/q", "dest": "here"},
                      {"name": "c", "b2": "m/n"}]}
    src = tmp_path / "src"
    src.mkdir()
    (src / "run.sh").write_text("true\n")
    cfg, _ = jm.validate_job_config(cfg, str(src))
    spec = tmp_path / "assets.tsv"
    _prepare_assets_tsv(tmp_path, cfg, spec)
    rows = [ln.split("\t") for ln in spec.read_text().splitlines()]
    assert rows == [["a", "x/y", "copy", "0", "-", "PUSHED.json"],
                    ["b", "p/q", "copy", "0", "here", "-"],
                    ["c", "m/n", "copy", "0", "-", "-"]]
    # …and bash recovers them (the collapse bug lives here, not in the writer)
    out = subprocess.run(
        ["bash", "-c",
         'while IFS=$\'\\t\' read -r n b m o d r; do echo "$n|$b|$m|$o|$d|$r"; '
         'done < "$1"', "_", str(spec)],
        capture_output=True, text=True, check=True).stdout.split()
    assert out == ["a|x/y|copy|0|-|PUSHED.json", "b|p/q|copy|0|here|-",
                   "c|m/n|copy|0|-|-"]


# ---------------------------------------------------------------------------
# P4e: free-space precheck BEFORE the staging loop pulls anything
#
# Before this, staging had no `df` gate at all: an undersized box killed rclone
# partway through and reported `asset_stage_failed:<name>` — a transport-shaped
# reason that never mentions disk — so the operator retried onto the same box.
# The refusal must be distinct (`insufficient_disk`) and carry the numbers.
# ---------------------------------------------------------------------------
def _sizeshim_dir(tmp_path, shimdir, match, nbytes, name="sizebin"):
    """A PATH dir whose `rclone` wraps the shared shim and answers `size --json`
    (the shared shim has no `size` op — it exits 2, which is deliberately the
    UNKNOWN path) with `nbytes` for any target containing `match`. This is how a
    test states "the B2 source is N bytes" without inventing a second shim."""
    d = tmp_path / name
    d.mkdir()
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        'if [ "${1:-}" = size ]; then\n'
        '  for a in "$@"; do\n'
        f'    case "$a" in *{match}*) echo \'{{"count":1,"bytes":{int(nbytes)}}}\';'
        ' exit 0 ;; esac\n'
        '  done\n'
        '  echo \'{"count":0,"bytes":0}\'; exit 0\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _broken_df_dir(tmp_path, body="echo 'df: unavailable'\nexit 1\n"):
    """A PATH dir holding a `df` that cannot answer — the measurement-unreadable
    case. Default: noise on stdout + non-zero exit."""
    d = tmp_path / "dfbin"
    d.mkdir()
    w = d / "df"
    w.write_text("#!/usr/bin/env bash\n" + body)
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _path_env(*dirs):
    return {"PATH": ":".join([str(d) for d in dirs]
                             + [os.environ.get("PATH", os.defpath)])}


def _failed_event(bucket, job_id):
    for b in _events(bucket, job_id):
        e = json.loads(b)
        if e.get("event") == "failed":
            return e
    return None


def test_jobd_asset_precheck_refuses_when_short_of_space(tmp_path):
    """The asset does not fit: the job fails terminal with the DISTINCT
    `insufficient_disk` reason carrying free/required GB and the asset names —
    BEFORE a single byte is pulled (no cache dir, no `started`)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90077
    _put_asset_files(bucket, "assets-src/huge", {"config.json": "{}", "w.bin": "W"})
    # 900 TiB "on B2": larger than any dev box or CI runner, so the refusal is
    # deterministic without faking df.
    sizedir = _sizeshim_dir(tmp_path, shimdir, "assets-src/huge", 900 * 1024 ** 4)
    pulllog = tmp_path / "pulllog"
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "adisk", "mkdir -p out\necho RAN > out/ran.txt\n",
        [{"name": "huge", "b2": "assets-src/huge"}])
    env = _asset_env(pulllog)
    env.update(_path_env(sizedir, shimdir))
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", f"kinds={kinds} stderr={r.stderr}"
    reason = v["fail_reason"] or ""
    assert reason.startswith("insufficient_disk"), reason
    assert "asset_stage_failed" not in reason, \
        "refused with the OLD transport-shaped reason — the whole point is a named disk failure"
    ev = _failed_event(bucket, job_id)
    assert ev["required_gb"] > ev["free_gb"] >= 0, ev
    assert ev["assets"] == "huge" and ev["largest"] == "huge", ev
    assert ev["path"].endswith("/assets"), ev
    # nothing was pulled: no `started`, no cache dir, no pull attempt logged
    assert "started" not in kinds, kinds
    assert not (tmp_path / "workspace" / "assets" / "huge").exists()
    assert not pulllog.exists() or pulllog.read_text().strip() == ""


def test_jobd_asset_precheck_proceeds_when_df_unreadable(tmp_path):
    """THE safety property: a precheck that cannot MEASURE free space must never
    block the job. With `df` broken AND a requirement (999999 GB) that would
    otherwise refuse every box on earth, the job still runs to done — an
    unreadable measurement is not evidence of a full disk."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90078
    dfdir = _broken_df_dir(tmp_path)
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "adfblind", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _asset_env()
    env["JOBD_MIN_FREE_GB"] = "999999"
    env.update(_path_env(dfdir, shimdir))
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"kinds={kinds} stderr={r.stderr}"
    assert "insufficient_disk" not in (v["fail_reason"] or ""), v
    assert "disk precheck SKIPPED" in r.stderr, r.stderr
    # and it really staged: the asset landed despite the blind precheck
    assert (tmp_path / "workspace" / "assets" / "base" / "config.json").is_file()


def test_jobd_asset_precheck_unparseable_df_also_proceeds(tmp_path):
    """Same invariant, second shape: `df` exits 0 but prints something we cannot
    parse (busybox variants, a wrapper). Unparseable == unknown == proceed."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90079
    dfdir = _broken_df_dir(tmp_path, body="echo 'totally not df output'\nexit 0\n")
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "adfjunk", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _asset_env()
    env["JOBD_MIN_FREE_GB"] = "999999"
    env.update(_path_env(dfdir, shimdir))
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"


def test_jobd_asset_precheck_ignores_already_cached_asset(tmp_path):
    """A resumed box must not be asked to find room for bytes it already holds.
    Boot 1 stages the asset (writing the `.complete` marker); boot 2 declares the
    SAME asset as 900 TiB on B2 — the precheck must count it as 0 and let the job
    run, because staging would skip the pull anyway."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90080
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}", "w.bin": "W"})
    pulllog = tmp_path / "pulllog"
    ja = _stage_asset_job(tmp_path, bucket, iid, "acached",
                          "mkdir -p out\necho a > out/r.txt\n",
                          [{"name": "base", "b2": "assets-src/base"}], nonce4="aaaa")
    assert _run_jobd(tmp_path, bucket, shimdir, iid,
                     extra_env=_asset_env(pulllog)).returncode == 0
    assert jm.fold_events(_events(bucket, ja), live_iids={str(iid)})["status"] == "done"
    assert (tmp_path / "workspace" / "assets" / ".base.complete").is_file()

    sizedir = _sizeshim_dir(tmp_path, shimdir, "assets-src/base", 900 * 1024 ** 4)
    jb = _stage_asset_job(tmp_path, bucket, iid, "acached",
                          "mkdir -p out\necho b > out/r.txt\n",
                          [{"name": "base", "b2": "assets-src/base"}], nonce4="bbbb")
    env = _asset_env(pulllog)
    env.update(_path_env(sizedir, shimdir))
    r2 = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r2.returncode == 0, r2.stderr
    v = jm.fold_events(_events(bucket, jb), live_iids={str(iid)})
    assert v["status"] == "done", \
        f"cached asset was double-counted against free space: {v} stderr={r2.stderr}"
    assert pulllog.read_text().split().count("base") == 1   # still never re-pulled


def test_jobd_asset_precheck_excludes_optional_assets(tmp_path):
    """An `optional: true` asset that cannot fit must NOT fail the job: staging
    already tolerates its absence, so refusing the whole job for want of room for
    it would be strictly worse than the behavior this precheck replaces."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90090
    _put_asset_files(bucket, "assets-src/nice2have", {"config.json": "{}"})
    sizedir = _sizeshim_dir(tmp_path, shimdir, "assets-src/nice2have", 900 * 1024 ** 4)
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "aoptdisk", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "nice2have", "b2": "assets-src/nice2have", "optional": True}])
    env = _asset_env()
    env.update(_path_env(sizedir, shimdir))
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"optional asset drove a refusal: {v} stderr={r.stderr}"


def test_jobd_asset_precheck_reads_ticket_declared_requirement(tmp_path):
    """The `needs.disk_gb` WIRING HOOK: jobd reads a ticket-declared requirement
    from JOB_NEEDS_DISK_GB (which onstart/jobd.py `prepare` will echo once
    jobmeta.py carries the key). Declared 999999 GB with a trivially small asset
    => refused, so the hook is proven live end to end today."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90081
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "adecl", "mkdir -p out\necho RAN > out/ran.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _asset_env()
    env["JOB_NEEDS_DISK_GB"] = "999999"
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", f"{v} stderr={r.stderr}"
    assert (v["fail_reason"] or "").startswith("insufficient_disk"), v
    assert _failed_event(bucket, job_id)["required_gb"] == 999999
    assert "started" not in [json.loads(b)["event"] for b in bodies]


def test_jobd_asset_precheck_opt_out(tmp_path):
    """JOBD_DISK_PRECHECK=0 restores the old always-stage behavior (escape hatch
    for a box where the measurement is wrong rather than the disk)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90082
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "adiskoff", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _asset_env()
    env["JOB_NEEDS_DISK_GB"] = "999999"
    env["JOBD_DISK_PRECHECK"] = "0"
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"


# ---------------------------------------------------------------------------
# P4e: boot scratch/filesystem probe (observability only — no policy)
# ---------------------------------------------------------------------------
def _probe_event(bucket, iid):
    for b in _box_events(bucket, iid):
        e = json.loads(b)
        if e.get("event") == "scratch_probe":
            return e
    return None


def _disk_usage_event(bucket, iid):
    for b in _box_events(bucket, iid):
        e = json.loads(b)
        if e.get("event") == "disk_usage":
            return e
    return None


def test_jobd_emits_disk_usage_at_a_job_terminal(tmp_path):
    """The half `scratch_probe` cannot see: what the job actually used against
    what the launch bought. Emitted before the checkpoint scrub, so the figure
    is the job's peak footprint and not the post-cleanup one."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90201
    _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    ev = _disk_usage_event(bucket, iid)
    assert ev is not None, f"no disk_usage box event: {_box_events(bucket, iid)}"
    assert ev["phase"] == "terminal"
    for k in ("workspace_size_mb", "workspace_free_mb", "workspace_used_mb",
              "box_high_water_mb", "job_dir_mb", "job"):
        assert k in ev, f"{k} missing from {ev}"
    for k in ("workspace_size_mb", "workspace_free_mb", "workspace_used_mb"):
        assert isinstance(ev[k], int), f"{k}={ev[k]!r}"
    assert ev["workspace_used_mb"] == ev["workspace_size_mb"] - ev["workspace_free_mb"]
    rec = __import__("disksize").disk_usage_from_event(ev)
    assert rec["allocated_gb"] is not None and rec["peak_gb"] is not None


def test_jobd_disk_usage_survives_a_broken_df(tmp_path):
    """A df that cannot answer must not cost the job its terminal event — every
    unreadable field degrades to the literal "unknown", never a guessed 0."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90202
    _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env=_path_env(_broken_df_dir(tmp_path), shimdir))
    assert r.returncode == 0, r.stderr
    ev = _disk_usage_event(bucket, iid)
    assert ev is not None, f"no disk_usage box event: {_box_events(bucket, iid)}"
    assert ev["workspace_size_mb"] == "unknown", ev
    assert ev["workspace_used_mb"] == "unknown", ev
    rec = __import__("disksize").disk_usage_from_event(ev)
    assert rec["allocated_gb"] is None and rec["slack_gb"] is None


def test_jobd_scratch_probe_emits_box_event(tmp_path):
    """At boot jobd records what backs /tmp, $ROOT and /dev/shm plus system RAM,
    on the per-box stream. Every field must be present, and every value is either
    a number or the literal "unknown" — never a guessed 0 (this evidence is what
    decides whether RAM-backed scratch is viable at all)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90083
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    ev = _probe_event(bucket, iid)
    assert ev is not None, f"no scratch_probe box event: {_box_events(bucket, iid)}"
    for k in ("tmp_fs", "tmp_size_mb", "tmp_free_mb",
              "workspace_fs", "workspace_size_mb", "workspace_free_mb",
              "shm_fs", "shm_size_mb", "shm_free_mb",
              "mem_total_mb", "mem_avail_mb", "cgroup_mem_limit_mb",
              "cgroup_mem_current_mb", "tmpfs_mount", "workspace_path"):
        assert k in ev, f"{k} missing from {ev}"
    for k in ("tmp_size_mb", "tmp_free_mb", "workspace_size_mb", "workspace_free_mb",
              "shm_size_mb", "shm_free_mb", "mem_total_mb", "mem_avail_mb"):
        assert isinstance(ev[k], int) or ev[k] == "unknown", f"{k}={ev[k]!r}"
    assert ev["workspace_path"] == str(tmp_path / "workspace")
    # the suite never touches the mount table (see _run_jobd)
    assert ev["tmpfs_mount"] == "disabled", ev
    assert "scratch probe:" in r.stderr, r.stderr


def test_jobd_scratch_probe_reads_cgroup_memory_limit(tmp_path):
    """/proc/meminfo reports the HOST's memory, not the container's allowance —
    the same overstatement `effective_cpu_cores` corrects for CPU. Since tmpfs
    pages are charged to the cgroup's memory limit, THAT is the number any future
    RAM-scratch budget is bounded by, so the probe reads it (v2 first, v1
    fallback) and reports a no-limit cgroup as "max", never as a number."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90086
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.max").write_text(str(64 * 1024 ** 3) + "\n")     # 64 GiB
    (cg / "memory.current").write_text(str(4 * 1024 ** 3) + "\n")  # 4 GiB
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CGROUP_ROOT": str(cg)})
    assert r.returncode == 0, r.stderr
    ev = _probe_event(bucket, iid)
    assert ev["cgroup_mem_limit_mb"] == 65536, ev
    assert ev["cgroup_mem_current_mb"] == 4096, ev

    # cgroup v1 spells "no limit" as a near-2^63 sentinel; v2 spells it "max".
    # Both must read as "max", never as ~8.8 million MB of headroom.
    for sentinel in ("max", "9223372036854771712"):
        (cg / "memory.max").write_text(sentinel + "\n")
        iid += 1
        assert _run_jobd(tmp_path, bucket, shimdir, iid,
                         extra_env={"JOBD_CGROUP_ROOT": str(cg)}).returncode == 0
        assert _probe_event(bucket, iid)["cgroup_mem_limit_mb"] == "max", sentinel


def test_jobd_scratch_probe_tmpfs_attempt_is_honest_and_nonfatal(tmp_path):
    """With the mount attempt enabled the probe reports what actually happened —
    `ok` where the container may mount a tmpfs, `denied` where it may not (the
    common unprivileged case) — and the daemon boots either way."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90084
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_TMPFS_PROBE": "1"})
    assert r.returncode == 0, r.stderr
    ev = _probe_event(bucket, iid)
    assert ev is not None
    assert ev["tmpfs_mount"] in ("ok", "denied", "no_mount_cmd", "no_tmpdir",
                                "mounted_no_umount"), ev
    # observability only: the job is unaffected
    assert jm.fold_events(_events(bucket, job_id),
                          live_iids={str(iid)})["status"] == "done", r.stderr


def test_jobd_scratch_probe_survives_a_broken_df(tmp_path):
    """Every read degrades to "unknown" rather than killing daemon startup: with
    `df` broken the probe still emits, with unknown filesystem facts, and the job
    still runs."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90085
    dfdir = _broken_df_dir(tmp_path)
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env=_path_env(dfdir, shimdir))
    assert r.returncode == 0, r.stderr
    ev = _probe_event(bucket, iid)
    assert ev is not None, r.stderr
    assert ev["tmp_size_mb"] == "unknown" and ev["workspace_free_mb"] == "unknown", ev
    assert jm.fold_events(_events(bucket, job_id),
                          live_iids={str(iid)})["status"] == "done", r.stderr


def test_jobd_hard_crash_still_counts_against_restart_cap(tmp_path):
    """N1(c) guard: a HARD kill with NO trap (no .preempted breadcrumb) is a
    crash-restart — it still burns max_restarts and still fails terminally at the
    cap. max_restarts=0 => one hard interruption is terminal `failed`."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90064
    config = (
        "version: 1\nname: crash-cap-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "max_restarts: 0\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id = _stage_named(tmp_path, bucket, iid, "cc", "mkdir -p out\nsleep 60\n",
                          config, "cccc")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)   # hard death, no trap
        p.wait(timeout=30)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # no breadcrumb was dropped (the trap never ran)
    crumb = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.preempted"
    assert not crumb.is_file(), "a hard kill must not leave a preempt breadcrumb"
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed" and "restart cap" in (v["fail_reason"] or ""), v


# --- N5(a): REAL venv provisioning (check_venv self-heal) ---------------------
# check_venv used to only SOURCE a pre-existing env (`serve`/`eval` were no-ops
# when absent). N5 makes an ABSENT env self-provision by invoking the matching
# provisioner script. Tests stub the provisioner via PATH injection (a stub
# `job_serve.sh` / `fetch_eval_env.sh` dropped into shimdir, which _run_jobd
# prepends to PATH) — NO real pip. The stub records that it was invoked and
# materializes the env dir so the post-provision source succeeds.

def _write_stub(shimdir, name, body):
    p = shimdir / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


_VENV_SERVE_CFG = (
    "version: 1\nname: venv-serve-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
    "env:\n  FOO: \"world\"\n"
    "results:\n  - \"out/**\"\n"
    "needs:\n  gpu: false\n  venv: serve\n")
_VENV_EVAL_CFG = (
    "version: 1\nname: venv-eval-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
    "env:\n  FOO: \"world\"\n"
    "results:\n  - \"out/**\"\n"
    "needs:\n  gpu: false\n  venv: eval\n")
_VENV_EVAL_PINNED_CFG = (
    "version: 1\nname: venv-eval-pinned-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
    "env:\n  FOO: \"world\"\n  EVAL_ENV_VER: \"20260816-1813-3c0a5f5b\"\n"
    "results:\n  - \"out/**\"\n"
    "needs:\n  gpu: false\n  venv: eval\n")


def test_check_venv_serve_provisions_when_absent(tmp_path):
    """needs.venv: serve + no /workspace/serve => jobd invokes job_serve.sh
    --build-venv (PATH-injected stub), which materializes the venv; job runs."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90070
    marker = tmp_path / "serve_build.marker"
    _write_stub(shimdir, "job_serve.sh",
                'echo "STUB job_serve $*" >&2\n'
                'if [ "${1:-}" = "--build-venv" ]; then\n'
                f'  touch {shlex.quote(str(marker))}\n'
                '  mkdir -p "${JOBD_ROOT:?}/serve/bin"\n'
                '  echo ": provisioned" > "${JOBD_ROOT}/serve/bin/activate"\n'
                '  exit 0\n'
                'fi\n'
                'echo "STUB job_serve: unexpected non-build invocation" >&2; exit 9\n')
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_SERVE_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    assert marker.is_file(), f"provisioner never invoked; stderr={r.stderr}"
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"events={_events(bucket, job_id)} stderr={r.stderr}"


def test_check_venv_eval_provisions_when_absent(tmp_path):
    """needs.venv: eval + no /workspace/eval/env.sh => jobd invokes
    fetch_eval_env.sh (PATH-injected stub), which materializes env.sh; job runs."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90071
    marker = tmp_path / "eval_fetch.marker"
    _write_stub(shimdir, "fetch_eval_env.sh",
                'echo "STUB fetch_eval_env $*" >&2\n'
                f'touch {shlex.quote(str(marker))}\n'
                'mkdir -p "${JOBD_ROOT:?}/eval"\n'
                'echo "export EVAL_ENV=stub" > "${JOBD_ROOT}/eval/env.sh"\n'
                'exit 0\n')
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_EVAL_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    assert marker.is_file(), f"provisioner never invoked; stderr={r.stderr}"
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"events={_events(bucket, job_id)} stderr={r.stderr}"


def _eval_pin_stub(shimdir, seen):
    """A fetch_eval_env.sh stub that records the EVAL_ENV_VER it was RUN with."""
    return _write_stub(
        shimdir, "fetch_eval_env.sh",
        f'printf "%s" "${{EVAL_ENV_VER-<unset>}}" > {shlex.quote(str(seen))}\n'
        'mkdir -p "${JOBD_ROOT:?}/eval"\n'
        'echo "export EVAL_ENV=stub" > "${JOBD_ROOT}/eval/env.sh"\n'
        'exit 0\n')


def test_check_venv_eval_provisions_at_the_ticket_pin(tmp_path):
    """The ticket's EVAL_ENV_VER must reach fetch_eval_env.sh.

    It resolves eval-env/LATEST when the var is unset, and a pinned bake does
    not advance LATEST — so without this the box provisions a DIFFERENT bake
    than the one the job grades against and every readout names the wrong
    instrument."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90081
    seen = tmp_path / "fetch_pin.txt"
    _eval_pin_stub(shimdir, seen)
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_EVAL_PINNED_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    assert seen.is_file(), f"provisioner never invoked; stderr={r.stderr}"
    assert seen.read_text() == "20260816-1813-3c0a5f5b", r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"events={_events(bucket, job_id)}"


def test_check_venv_eval_unpinned_ticket_leaves_the_fetch_env_alone(tmp_path):
    """No ticket pin => jobd must not inject an EMPTY EVAL_ENV_VER, which would
    shadow a pin the BOX was launched with (`herdd launch --eval-env-ver`)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90082
    seen = tmp_path / "fetch_pin_unset.txt"
    _eval_pin_stub(shimdir, seen)
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_EVAL_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"EVAL_ENV_VER": "20260806-2152-76cd109a"})
    assert r.returncode == 0, r.stderr
    assert seen.is_file(), f"provisioner never invoked; stderr={r.stderr}"
    assert seen.read_text() == "20260806-2152-76cd109a", r.stderr


def test_check_venv_serve_provision_failure_fails_job(tmp_path):
    """A provisioner that FAILS takes the job terminal `failed` BEFORE the
    entrypoint runs, with a distinct venv-provisioning reason (loud)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90072
    ran = tmp_path / "entrypoint_ran.marker"
    _write_stub(shimdir, "job_serve.sh",
                'echo "STUB job_serve FAIL $*" >&2\n'
                'exit 3\n')
    # entrypoint touches a marker so we can prove it NEVER ran (failed pre-entry)
    entry = (f"touch {shlex.quote(str(ran))}\n"
             "mkdir -p out\necho hi > out/result.txt\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=_VENV_SERVE_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"events={_events(bucket, job_id)}"
    blob = "".join(_events(bucket, job_id))
    assert "provisioning failed" in blob, blob
    assert not ran.is_file(), "entrypoint must NOT run when venv provisioning fails"


def test_check_venv_serve_sources_existing_no_provision(tmp_path):
    """When /workspace/serve/bin/activate ALREADY exists, check_venv sources it
    and does NOT invoke the provisioner (idempotent warm-box fast path)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90073
    marker = tmp_path / "should_not_run.marker"
    _write_stub(shimdir, "job_serve.sh",
                f'touch {shlex.quote(str(marker))}\nexit 0\n')
    # pre-create the serve venv activate on the box disk (JOBD_ROOT/serve)
    serve_bin = tmp_path / "workspace" / "serve" / "bin"
    serve_bin.mkdir(parents=True)
    (serve_bin / "activate").write_text(": warm\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_SERVE_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", _events(bucket, job_id)
    assert not marker.is_file(), "provisioner ran despite a present serve venv"


def test_check_venv_provision_timeout_fails_the_ticket(tmp_path):
    """A provisioner that never returns fails ITS ticket on the install bound
    instead of holding the box-global venv lock forever; the reason names which
    bound tripped."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90074
    _write_stub(shimdir, "fetch_eval_env.sh", 'sleep 300\n')
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_EVAL_CFG)
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_VENV_PROVISION_TIMEOUT_S": "1"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"events={_events(bucket, job_id)}"
    assert "install exceeded 1s" in (v["fail_reason"] or ""), v


def test_check_venv_lock_wait_bound_fails_the_ticket(tmp_path):
    """A peer wedged inside the per-kind venv lock costs the waiting ticket the
    lock-wait bound and no more; the failure names the lock, not the install."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90075
    ran = tmp_path / "provisioner_ran.marker"
    _write_stub(shimdir, "fetch_eval_env.sh", f'touch {shlex.quote(str(ran))}\n')
    lockf = tmp_path / "workspace" / ".venv-eval.lock"
    lockf.parent.mkdir(parents=True, exist_ok=True)
    lockf.touch()
    job_id, _ = _stage_job(tmp_path, bucket, iid, config=_VENV_EVAL_CFG)
    holder = os.open(str(lockf), os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        r = _run_jobd(tmp_path, bucket, shimdir, iid,
                      extra_env={"JOBD_VENV_LOCK_WAIT_S": "1"})
    finally:
        os.close(holder)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"events={_events(bucket, job_id)}"
    assert "lock wait exceeded 1s" in (v["fail_reason"] or ""), v
    assert not ran.is_file(), "the provisioner ran without holding the lock"


# ---------------------------------------------------------------------------
# T1 (HANDOFF_DESIGN §4/§9): retarget-resume pull-back gap + final_flush fence
# ---------------------------------------------------------------------------
def _stage_retargeted_job(tmp_path, bucket, new_iid, *, retargeted_from,
                          checkpoint_seed="step-1", nonce4="rtrt", timeout_s=8,
                          requeued_ts=None, done_marker=False):
    """Stage a checkpointing job whose ticket is ALREADY sitting in new_iid's queue
    (as `job retarget` leaves it), with a prior checkpoint pre-synced into
    jobs/<id>/checkpoints/. The entrypoint resumes-and-exits when it finds the pulled
    state, else writes state and sleeps past timeout_s (so a MISSING pull-back
    regresses to a timeout `failed`). `retargeted_from` (str|None) sets the ticket
    marker; None models the no-marker fallback path (prior event drives pull-back).

    `requeued_ts` stamps the ticket the way `herdd job requeue` does, and
    `done_marker` plants the results.DONE.json that a rc!=0 attempt publishes
    BEFORE it emits `failed` — together they model a requeued terminal-failed job.
    Returns (job_id, sha)."""
    entry = (
        "mkdir -p out\n"
        "if [ -f out/state.txt ]; then\n"
        "  echo \"restart=$JOB_RESTART_COUNT\" > out/resumed.txt\n"
        "  echo done > out/result.txt\n  exit 0\nfi\n"
        "echo step-1 > out/state.txt\nsleep 60\n")
    config = (
        f"version: 1\nname: retarget-probe\nentrypoint: run.sh\ntimeout_s: {timeout_s}\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    src = tmp_path / f"rtsrc-{nonce4}"
    src.mkdir()
    (src / "run.sh").write_text(entry)
    (src / "job-config.yaml").write_text(config)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(src)), str(src))
    sha = jm.write_bundle(str(src), str(tmp_path / f"rt-{nonce4}.tar.zst"))["sha256"]
    bdir = bucket / "jobs" / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_path / f"rt-{nonce4}.tar.zst"), str(bdir / f"{sha}.tar.zst"))
    job_id = jm.mint_job_id(cfg["name"], ts="20260713T000000", nonce4=nonce4)
    # the OLD box already synced a checkpoint into jobs/<id>/checkpoints/
    res = bucket / "jobs" / job_id / "checkpoints" / "out"
    res.mkdir(parents=True, exist_ok=True)
    (res / "state.txt").write_text(checkpoint_seed + "\n")
    # the ticket now lives under the NEW box's queue (what `job retarget` writes)
    ticket = jm.make_ticket(job_id, sha, "cli:test", cfg, str(new_iid))
    if retargeted_from is not None:
        ticket["retargeted_from"] = str(retargeted_from)
    if requeued_ts is not None:
        ticket[jm.REQUEUE_TICKET_MARK] = requeued_ts
    if done_marker:
        (bucket / "jobs" / job_id / "results.DONE.json").write_text(
            json.dumps({"rc": 16, "n_results": 0}))
    qdir = bucket / "jobs" / "queue" / str(new_iid)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{job_id}.json").write_text(json.dumps(ticket))
    return job_id, sha


def test_jobd_retarget_pulls_checkpoint_back(tmp_path):
    """THE T1 gap (HANDOFF_DESIGN §4): a checkpointing job moved by `job retarget`
    onto a FRESH box must pull its prior checkpoints back before the entrypoint —
    even though the box-local restart_count is 0 (no .attempts/.preempts breadcrumb
    exists here). The old predicate gated pull-back on restart_count>0 ALONE, so the
    job silently restarted from scratch. The ticket's `retargeted_from` marker now
    drives the pull-back."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90080, 90081
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid)
    # fresh box: NO local breadcrumbs -> restart_count=0; the marker must still pull.
    r = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(new_iid)})
    assert v["status"] == "done", f"kinds={kinds} stderr={r.stderr}"
    # the resume branch ran (state.txt was pulled back) at restart_count=0 —
    # a scratch restart would instead sleep past timeout_s and fail (rc 124).
    resumed = bucket / "jobs" / job_id / "results" / "out" / "resumed.txt"
    assert resumed.is_file() and resumed.read_text().strip() == "restart=0", \
        f"retargeted job did not pull the checkpoint back (scratch restart): kinds={kinds}"


def test_jobd_prior_checkpoint_event_triggers_pullback(tmp_path):
    """The second continuation signal (HANDOFF_DESIGN §4): a fresh box whose ticket
    carries NO `retargeted_from` marker still pulls the checkpoint back when the
    job's B2 event history already holds a `checkpoint` event (best-effort probe) —
    i.e. some prior box demonstrably synced state for this job."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90082, 90083
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=None, nonce4="ckev")
    # no ticket marker — instead a prior `checkpoint` event from the OLD box exists
    # (actor box:<old> => its key never matches this box's -box_<new>- resume probe,
    # so this is still a FRESH claim with restart_count=0).
    ev = jm.make_event(job_id, "checkpoint", f"box:{old_iid}",
                       instance_id=str(old_iid), n=1, files=1)
    evdir = bucket / "jobs" / job_id / "events"
    evdir.mkdir(parents=True, exist_ok=True)
    (evdir / jm.event_key(ev)).write_text(json.dumps(ev, separators=(",", ":")))

    r = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(new_iid)})
    assert v["status"] == "done", f"kinds={kinds} stderr={r.stderr}"
    resumed = bucket / "jobs" / job_id / "results" / "out" / "resumed.txt"
    assert resumed.is_file() and resumed.read_text().strip() == "restart=0", \
        f"prior-checkpoint-event continuation did not pull back: kinds={kinds}"


def test_jobd_retarget_pullback_emits_resumed_continuity(tmp_path):
    """Issue C (canary-job): a `job retarget` CONTINUATION lands on a FRESH box
    (restart_count=0), so the box-local resume path never fires and the understudy
    emitted only claimed+started — a resume off another box was invisible in the
    event log (continuity only readable from heartbeat tails). The fresh-box
    pull-back path must emit a `resumed` continuity event (kind=retarget, from_box
    carrying the source) so fold_events counts it (last_resumed_ts) and `job status`
    shows the pull-back."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90090, 90091
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="cont")
    r = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    evs = [json.loads(b) for b in bodies]
    kinds = [e["event"] for e in evs]
    # a fresh-box retarget claim now carries a `resumed` continuity event...
    resumed = [e for e in evs if e["event"] == "resumed"]
    assert resumed, f"no resumed continuity event on retarget pull-back: kinds={kinds}"
    assert resumed[-1].get("kind") == "retarget", f"resumed not tagged retarget: {resumed[-1]}"
    assert str(resumed[-1].get("from_box")) == str(old_iid), \
        f"resumed missing/wrong from_box: {resumed[-1]}"
    # ...and the fold makes the continuation visible (last_resumed_ts set).
    v = jm.fold_events(bodies, live_iids={str(new_iid)})
    assert v["last_resumed_ts"], f"fold did not register the continuity resume: {v}"


def test_jobd_skips_a_ticket_whose_done_marker_exists(tmp_path):
    """The behavior a requeue has to get past, pinned as the NEGATIVE control: a
    plain (non-requeued) ticket whose job already has results.DONE.json is skipped
    as remote-done and never claimed. This is correct for a retargeted twin — and
    it is also why a same-JOB_ID re-open was impossible before `job requeue`: a
    run that exits rc!=0 publishes results.DONE.json BEFORE it emits `failed`, so
    the marker is present for EVERY infra-killed job."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90100, 90101
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="dnmk",
                                      done_marker=True)
    r = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "claimed" not in kinds and "started" not in kinds, \
        f"jobd ran a job whose DONE marker exists: kinds={kinds}"


def test_jobd_unreadable_ticket_does_not_latch_remote_done(tmp_path):
    """A DONE marker plus an unreadable ticket is UNKNOWN, not "no requeue": the
    `.terminal` latch is permanent, so jobd must leave no breadcrumb and retry
    next poll rather than swallow an operator `job requeue` on a transport blip."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90120, 90121
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="qcfl",
                                      done_marker=True)
    r = _run_jobd(tmp_path, bucket, _queuecatfail_shimdir(tmp_path, shimdir), new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    assert not _terminal_marker(tmp_path, job_id).exists(), \
        "an unread ticket latched remote-done — a later requeue is now unreachable"
    assert _events(bucket, job_id) == [], _events(bucket, job_id)


def test_jobd_persistently_unreadable_ticket_latches_remote_done(tmp_path):
    """The bound on the retry above: after JOBD_DONE_UNKNOWN_MAX consecutive
    unreadable polls jobd stops re-reading B2 and trusts the DONE marker, and the
    consecutive count survives between polls in $STATE_DIR."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90122, 90123
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="qcpl",
                                      done_marker=True)
    failbin = _queuecatfail_shimdir(tmp_path, shimdir)
    env = {"JOBD_CKPT_MIN_AGE": "0s", "JOBD_DONE_UNKNOWN_MAX": "2"}
    counter = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.done_unknown"

    r = _run_jobd(tmp_path, bucket, failbin, new_iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    assert not _terminal_marker(tmp_path, job_id).exists(), "latched on poll 1 of 2"
    assert counter.read_text().strip() == "1", "the unknown count did not persist"

    r = _run_jobd(tmp_path, bucket, failbin, new_iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    assert _terminal_marker(tmp_path, job_id).read_text().startswith("remote-done"), \
        "a persistently unreadable ticket still re-reads B2 every poll"
    assert not counter.exists(), "the counter outlived the latch"
    assert _events(bucket, job_id) == [], _events(bucket, job_id)


def test_jobd_a_readable_ticket_resets_the_unknown_count(tmp_path):
    """A blip does not accumulate toward the bound: one unreadable poll then a
    readable one clears the $STATE_DIR counter, so the job runs normally."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90124, 90125
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="qcrs",
                                      done_marker=True,
                                      requeued_ts="20260731T010203Z")
    env = {"JOBD_CKPT_MIN_AGE": "0s", "JOBD_DONE_UNKNOWN_MAX": "2"}
    counter = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.done_unknown"

    r = _run_jobd(tmp_path, bucket, _queuecatfail_shimdir(tmp_path, shimdir),
                  new_iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    assert counter.read_text().strip() == "1", "the unknown count did not persist"

    r = _run_jobd(tmp_path, bucket, shimdir, new_iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    assert not counter.exists(), "a readable ticket did not reset the unknown count"
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(new_iid)})
    assert v["status"] == "done", f"events={_events(bucket, job_id)} stderr={r.stderr}"


def test_jobd_requeue_ticket_overrides_the_done_marker(tmp_path):
    """`herdd job requeue` re-opens a TERMINAL-FAILED job under the SAME JOB_ID
    by re-minting the ticket with `requeued_ts` (jobmeta.REQUEUE_TICKET_MARK). jobd
    must honour that explicit operator re-open over the prior attempt's
    results.DONE.json — otherwise the requeue is swallowed in silence — and still
    take the `retargeted_from` pull-back path, so the re-run CONTINUES from
    jobs/<JOB_ID>/checkpoints/ instead of restarting from scratch."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90102, 90103
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="rqmk",
                                      done_marker=True,
                                      requeued_ts="20260731T010203Z")
    r = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(new_iid)})
    assert v["status"] == "done", f"requeued job did not run: kinds={kinds} {r.stderr}"
    # the checkpoint pull-back still fired (same JOB_ID prefix, no copy needed):
    # a scratch restart would sleep past timeout_s and fail rc=124.
    resumed = bucket / "jobs" / job_id / "results" / "out" / "resumed.txt"
    assert resumed.is_file(), f"requeued job restarted from scratch: kinds={kinds}"
    assert "REQUEUE" in r.stdout + r.stderr, \
        "the DONE-marker override must say so in the log"


def test_jobd_honours_a_requeue_only_until_it_goes_terminal(tmp_path):
    """The loop guard. The queue ticket is never deleted, so a `requeued_ts` that
    kept overriding would re-run a re-failing job forever. No extra state carries
    that bound: the moment this attempt goes terminal, `mark_terminal` writes the
    box-local breadcrumb that poll_once checks BEFORE any B2 read — so a second
    pass on the same box is a clean no-op."""
    bucket, shimdir = _make_bucket(tmp_path)
    old_iid, new_iid = 90104, 90105
    job_id, _ = _stage_retargeted_job(tmp_path, bucket, new_iid,
                                      retargeted_from=old_iid, nonce4="rqlp",
                                      done_marker=True,
                                      requeued_ts="20260731T010203Z")
    _run_jobd(tmp_path, bucket, shimdir, new_iid,
              extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    n1 = len(_events(bucket, job_id))
    r2 = _run_jobd(tmp_path, bucket, shimdir, new_iid,
                   extra_env={"JOBD_CKPT_MIN_AGE": "0s"})
    assert r2.returncode == 0, r2.stderr
    assert len(_events(bucket, job_id)) == n1, \
        "the requeue mark re-ran an already-terminal job (loop!)"
    claims = [json.loads(b)["event"] for b in _events(bucket, job_id)].count("claimed")
    assert claims == 1


def test_jobd_preempt_emits_final_flush(tmp_path):
    """T1 fence signal (HANDOFF_DESIGN §4): after the preempt trap flushes the last
    checkpoint/results to B2, jobd emits a `final_flush` event — the cutover fence
    the handoff understudy waits on before write-enabling. Exactly one per running
    job, AFTER `preempted`, and AFTER the flush landed."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90084
    # write a checkpoint immediately then sleep past the SIGTERM; checkpoint_s is
    # large so the periodic sync NEVER fires — only the trap's final flush lands it.
    entry = "mkdir -p out\necho step-9 > out/ckpt.txt\nsleep 60\n"
    config = (
        "version: 1\nname: flush-probe\nentrypoint: run.sh\ntimeout_s: 120\n"
        "checkpoint_s: 3600\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid)
    try:
        assert _wait_for_event(bucket, job_id, "started"), "job never started"
        time.sleep(1.5)                       # entrypoint writes ckpt; jobd reaches wait()
        os.kill(p.pid, signal.SIGTERM)        # external stop -> preempt trap
        p.wait(timeout=60)
    finally:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert kinds.count("final_flush") == 1, f"kinds={kinds}"
    assert "preempted" in kinds, f"kinds={kinds}"
    # the flush landed (the checkpoint is on B2, under checkpoints/) — final_flush
    # marks its completion
    res = bucket / "jobs" / job_id / "checkpoints" / "out" / "ckpt.txt"
    assert res.is_file() and res.read_text().strip() == "step-9", f"kinds={kinds}"
    # final_flush is a NON-terminal marker: the fold never reads it as done/failed
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] not in jm.TERMINAL, f"status={v['status']}"


# ---------------------------------------------------------------------------
# T6 (HANDOFF_DESIGN §4/§9): box-side epoch guard — the mid-run checkpoint sync
# must REFUSE to push once a strictly-newer handoff epoch was PROMOTED over the
# job's B2 state (two-writer corruption fence), and push normally when the epoch
# is current/absent — including when only ARM-time <epoch>.json markers exist
# (a still-canonical box mid-window / after an aborted attempt is never refused).
# ---------------------------------------------------------------------------
def _run_ckpt_epoch_job(tmp_path, bucket, shimdir, iid, *, epoch_env, marker_epoch,
                        nonce4, promoted_epoch=None):
    """Stage a checkpointing job (writes out/ckpt.txt at once, then sleeps past its
    4s timeout so only the mid-run sync can ship it), optionally seed a handoff ARM
    marker jobs/<id>/handoff/<marker_epoch>.json and/or a promoted marker naming
    promoted_epoch, run jobd with HANDOFF_EPOCH=epoch_env. Returns (job_id, fold-view)."""
    entry = "mkdir -p out\necho step-100 > out/ckpt.txt\nsleep 60\n"
    config = (
        "version: 1\nname: epoch-probe\nentrypoint: run.sh\ntimeout_s: 4\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    # distinct src per call so bundles/job-ids don't collide across sub-runs
    src = tmp_path / f"epsrc-{nonce4}"
    src.mkdir()
    (src / "run.sh").write_text(entry)
    (src / "job-config.yaml").write_text(config)
    cfg, _ = jm.validate_job_config(jm.load_job_config(str(src)), str(src))
    sha = jm.write_bundle(str(src), str(tmp_path / f"ep-{nonce4}.tar.zst"))["sha256"]
    bdir = bucket / "jobs" / "bundles"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_path / f"ep-{nonce4}.tar.zst"), str(bdir / f"{sha}.tar.zst"))
    job_id = jm.mint_job_id(cfg["name"], ts="20260713T000000", nonce4=nonce4)
    if marker_epoch is not None or promoted_epoch is not None:
        hdir = bucket / "jobs" / job_id / "handoff"
        hdir.mkdir(parents=True, exist_ok=True)
        if marker_epoch is not None:
            (hdir / f"{marker_epoch}.json").write_text('{"epoch":%d}' % marker_epoch)
        if promoted_epoch is not None:
            (hdir / "promoted").write_text(
                '{"job_id":"%s","understudy":"999","epoch":%d,'
                '"promoted_at":"t","reason":"post_flush"}' % (job_id, promoted_epoch))
    ticket = jm.make_ticket(job_id, sha, "cli:test", cfg, str(iid))
    qdir = bucket / "jobs" / "queue" / str(iid)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{job_id}.json").write_text(json.dumps(ticket))
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_MIN_AGE": "0s", "HANDOFF_EPOCH": str(epoch_env)})
    assert r.returncode == 0, r.stderr
    return job_id, jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})


def test_jobd_checkpoint_sync_refuses_stale_epoch(tmp_path):
    """A box carrying a STALE handoff epoch (a newer epoch marker exists on B2) must
    NOT ship MID-RUN checkpoints — the understudy owns the job's B2 state after
    cutover. Zero `checkpoint` events fire (the discriminator: the terminal publish
    path is intentionally ungated, so the file itself may still land at job end; what
    the epoch fence controls is the recurring mid-run sync, counted here as events)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90090
    job_id, v = _run_ckpt_epoch_job(tmp_path, bucket, shimdir, iid,
                                    epoch_env=3, marker_epoch=5, nonce4="stal",
                                    promoted_epoch=5)
    assert v["status"] == "failed" and v["rc"] == 124            # timed out as usual
    assert v["n_checkpoints"] == 0, f"stale-epoch box still ran the mid-run sync: {v}"


def test_jobd_checkpoint_sync_allows_current_epoch(tmp_path):
    """The control: an epoch EQUAL to the promoted epoch is NOT stale (only a
    strictly-greater promotion refuses — the promoted understudy itself keeps
    pushing), so the mid-run sync ships normally."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90091
    job_id, v = _run_ckpt_epoch_job(tmp_path, bucket, shimdir, iid,
                                    epoch_env=5, marker_epoch=5, nonce4="curr",
                                    promoted_epoch=5)
    assert v["status"] == "failed" and v["rc"] == 124
    assert v["n_checkpoints"] >= 1, f"current-epoch box was wrongly refused: {v}"
    res = bucket / "jobs" / job_id / "checkpoints" / "out" / "ckpt.txt"
    assert res.is_file() and res.read_text().strip() == "step-100"


def test_jobd_checkpoint_sync_arm_marker_alone_not_stale(tmp_path):
    """A newer ARM-time <epoch>.json WITHOUT a promotion must NOT refuse the box:
    the still-canonical primary keeps syncing through a second handoff's
    ARM->cutover window, and an ABORTED attempt (whose ARM marker is never cleaned
    up) never silences the survivor. This is the regression the promoted-keyed
    predicate exists for — the old max-<epoch>.json keying failed it."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90093
    job_id, v = _run_ckpt_epoch_job(tmp_path, bucket, shimdir, iid,
                                    epoch_env=1, marker_epoch=2, nonce4="armo",
                                    promoted_epoch=None)
    assert v["status"] == "failed" and v["rc"] == 124
    assert v["n_checkpoints"] >= 1, f"ARM marker alone wrongly refused the box: {v}"
    assert (bucket / "jobs" / job_id / "checkpoints" / "out" / "ckpt.txt").is_file()


def test_jobd_checkpoint_sync_no_handoff_env_is_noop(tmp_path):
    """FAIL-SAFE: with HANDOFF_EPOCH unset (every normal, non-handoff job) the guard
    is a pure no-op even if stray handoff markers exist — the sync ships as before."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90092
    # markers present but no HANDOFF_EPOCH env => not a handoff box => never stale
    job_id, v = _run_ckpt_epoch_job(tmp_path, bucket, shimdir, iid,
                                    epoch_env="", marker_epoch=9, nonce4="noop",
                                    promoted_epoch=9)
    assert v["status"] == "failed" and v["rc"] == 124
    assert v["n_checkpoints"] >= 1, f"unset-epoch box was wrongly refused: {v}"
    assert (bucket / "jobs" / job_id / "checkpoints" / "out" / "ckpt.txt").is_file()


# ---------------------------------------------------------------------------
# cred-broker refresh hook (C6 — docs/plans/cred-broker-buildout.md §2.6)
# ---------------------------------------------------------------------------
def _cred_stub(tmp_path, rc=0, new_expiry=None):
    """Recorder stand-in for cred_client.py (jobd runs it via $JOBD_CRED_CLIENT):
    appends one line per invocation, optionally rewrites $JOBD_ENV_FILE with a
    fresh expiry (what the real client does on success), exits rc."""
    rec = tmp_path / "cred_recorder.txt"
    stub = tmp_path / "cred_stub.py"
    body = ["import os, sys",
            f"open({str(rec)!r}, 'a').write('call\\n')"]
    if new_expiry is not None:
        body.append("open(os.environ['JOBD_ENV_FILE'], 'w').write("
                    f"'export B2_KEY_EXPIRES_AT={int(new_expiry)}\\n')")
    body.append(f"sys.exit({rc})")
    stub.write_text("\n".join(body) + "\n")
    return stub, rec


def _wait_for_file(path, timeout=25):
    end = time.time() + timeout
    while time.time() < end and not path.exists():
        time.sleep(0.2)
    return path.exists()


def _wait_for_text(path, needle, timeout=60):
    """Bounded wait for a marker to appear IN a growing log file.

    The condition, not a stand-in for it. A fixed `sleep(N)` that means "by now
    the daemon will have ticked" is correct only on an idle box: under load the
    tick lands late, the marker is absent, and the assertion fails on timing
    rather than on behaviour.
    """
    end = time.time() + timeout
    while time.time() < end:
        try:
            if needle in path.read_text():
                return True
        except (FileNotFoundError, UnicodeDecodeError):
            pass
        time.sleep(0.2)
    return False


def test_jobd_cred_refresh_on_imminent_expiry_is_throttled(tmp_path):
    """B2_KEY_EXPIRES_AT inside the 24 h pre-expiry window triggers EXACTLY ONE
    cred_client invocation across many poll loops — the >=1/h throttle holds even
    though the (stubbed) refresh left the expiry imminent."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90100
    stub, rec = _cred_stub(tmp_path)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid, extra_env={
        "BOX_IDENTITY_NONCE": "ab" * 16,
        "B2_KEY_EXPIRES_AT": str(int(time.time()) + 600),   # << now+86400
        "JOBD_CRED_CLIENT": str(stub),
    })
    try:
        assert _wait_for_file(rec), "cred_client never invoked on imminent expiry"
        time.sleep(4)                                       # ~4 more poll loops
        assert rec.read_text().count("call") == 1, "throttle broken"
        assert (tmp_path / "credstate" / ".cred_refresh_last").is_file()
        assert p.poll() is None, "refresh path killed the daemon"
    finally:
        _kill(p)


def test_jobd_cred_refresh_success_resources_jobd_env(tmp_path):
    """On success jobd re-sources jobd.env: the stub installs a far-future
    B2_KEY_EXPIRES_AT, and even with the throttle DISABLED no second attempt
    fires — proof the re-sourced expiry cleared the trigger."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90101
    far = int(time.time()) + 30 * 86400
    stub, rec = _cred_stub(tmp_path, new_expiry=far)
    envfile = tmp_path / "jobd.env"
    p = _popen_jobd(tmp_path, bucket, shimdir, iid, extra_env={
        "BOX_IDENTITY_NONCE": "ab" * 16,
        "B2_KEY_EXPIRES_AT": str(int(time.time()) + 600),
        "JOBD_CRED_CLIENT": str(stub),
        "JOBD_ENV_FILE": str(envfile),
        "JOBD_CRED_THROTTLE_S": "0",                        # throttle out of the way
    })
    try:
        assert _wait_for_file(rec), "cred_client never invoked"
        time.sleep(4)
        assert rec.read_text().count("call") == 1, \
            "expiry trigger survived a successful refresh (jobd.env not re-sourced)"
        assert f"B2_KEY_EXPIRES_AT={far}" in envfile.read_text()
    finally:
        _kill(p)


def test_jobd_ckpt_authfail_flags_and_triggers_refresh(tmp_path):
    """The checkpoint-sync auth grep touches .cred_refresh_now; the main loop sees
    it and runs cred_client. The stub FAILS (rc=1): the flag persists for a later
    retry, the daemon survives, and the job still runs to done."""
    bucket, shimdir = _make_bucket(tmp_path)
    authdir = _authfail_shimdir(tmp_path, shimdir)
    iid = 90102
    entry = ("mkdir -p out\necho step-1 > out/ckpt.txt\nsleep 5\n"
             "echo done > out/result.txt\nexit 0\n")
    config = (
        "version: 1\nname: credflag-probe\nentrypoint: run.sh\ntimeout_s: 60\n"
        "checkpoint_s: 1\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    stub, rec = _cred_stub(tmp_path, rc=1)
    flag = tmp_path / "credstate" / ".cred_refresh_now"
    p = _popen_jobd(tmp_path, bucket, authdir, iid, extra_env={
        "JOBD_CKPT_MIN_AGE": "0s",
        "BOX_IDENTITY_NONCE": "cd" * 16,          # no B2_KEY_EXPIRES_AT: flag-only
        "JOBD_CRED_CLIENT": str(stub),
    })
    try:
        assert _wait_for_file(flag), "auth failure never touched .cred_refresh_now"
        assert _wait_for_file(rec), "flag never triggered a cred_client run"
        assert _wait_for_event(bucket, job_id, "done"), "refresh path broke the job"
        assert flag.is_file(), "failed refresh must LEAVE the flag for a retry"
        assert p.poll() is None, "failing cred_client crashed the daemon"
    finally:
        _kill(p)


def _asset_authfail_shimdir(tmp_path, shimdir, marker="assets-src"):
    """rclone wrapper that AUTH-FAILS any `copy` touching <marker> (the asset
    source prefix) — a dead key seen by asset_pull, everything else real."""
    d = tmp_path / "assetauthbin"
    d.mkdir()
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        'if [ "${1:-}" = copy ]; then\n'
        '  for a in "$@"; do case "$a" in *' + marker + '*)\n'
        '    echo "ERROR: InvalidAccessKeyId: the key is not valid" >&2\n'
        '    exit 1 ;; esac; done\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def test_jobd_asset_authfail_touches_refresh_flag(tmp_path):
    """The asset_pull auth grep is the second refresh hook: a dead-key asset pull
    keeps its existing behavior (early break-out, asset_stage_failed) AND touches
    .cred_refresh_now."""
    bucket, shimdir = _make_bucket(tmp_path)
    authdir = _asset_authfail_shimdir(tmp_path, shimdir)
    iid = 90103
    _put_asset_files(bucket, "assets-src/base", {"w.bin": "W"})
    job_id = _stage_asset_job(tmp_path, bucket, iid, "acredfail",
                              "mkdir -p out\necho ok > out/r.txt\n",
                              [{"name": "base", "b2": "assets-src/base"}])
    r = _run_jobd(tmp_path, bucket, authdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"existing auth-fail behavior changed: {v}"
    assert (tmp_path / "credstate" / ".cred_refresh_now").is_file(), \
        "asset auth failure did not touch .cred_refresh_now"


def test_jobd_no_broker_box_makes_zero_refresh_attempts(tmp_path):
    """A pre-broker box (no BOX_IDENTITY_NONCE) must behave byte-identically:
    zero cred_client invocations and zero marker files even with an imminent
    B2_KEY_EXPIRES_AT, while a staged job runs to done as always."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90104
    stub, rec = _cred_stub(tmp_path)
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid, extra_env={
        "B2_KEY_EXPIRES_AT": str(int(time.time()) + 600),   # imminent, but no nonce
        "JOBD_CRED_CLIENT": str(stub),
    })
    try:
        assert _wait_for_event(bucket, job_id, "done"), "job did not finish"
        time.sleep(2)
        assert not rec.exists(), "no-broker box attempted a cred refresh"
        cs = tmp_path / "credstate"
        assert not (cs / ".cred_refresh_now").exists()
        assert not (cs / ".cred_refresh_last").exists()
    finally:
        _kill(p)


def test_jobd_boot_persists_broker_identity(tmp_path):
    """jobd_boot.sh persists the §2.1 broker identity vars into jobd.env (0600)
    when present, so a RESUMED boot re-sources them for the refresh hook."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90105
    snippet, env = _stage_boot(tmp_path, bucket, iid, shimdir)
    env["B2_KEY_EXPIRES_AT"] = "1783000000"
    env["CRED_BROKER_URL"] = "http://broker.example.ts.net:8651"
    env["BOX_IDENTITY_NONCE"] = "ef" * 16
    env["CRED_ROLE"] = "jobs"
    env["TS_AUTHKEY"] = "tskey-test-not-a-real-key"
    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    envfile = tmp_path / "install" / "jobd.env"
    text = envfile.read_text()
    assert "export B2_KEY_EXPIRES_AT=1783000000" in text
    assert "export CRED_BROKER_URL=http://broker.example.ts.net:8651" in text
    assert f"export BOX_IDENTITY_NONCE={'ef' * 16}" in text
    assert "export CRED_ROLE=jobs" in text
    assert "export TS_AUTHKEY=tskey-test-not-a-real-key" in text
    assert stat.S_IMODE(envfile.stat().st_mode) == 0o600


def test_jobd_boot_omits_broker_identity_when_absent(tmp_path):
    """Pre-broker launch (none of the §2.1 vars set): jobd.env carries NO broker
    lines — the refresh hook stays a no-op on such boxes."""
    bucket, shimdir = _make_bucket(tmp_path)
    snippet, env = _stage_boot(tmp_path, bucket, 90106, shimdir)
    for k in ("B2_KEY_EXPIRES_AT", "CRED_BROKER_URL", "BOX_IDENTITY_NONCE",
              "CRED_ROLE", "TS_AUTHKEY"):
        env.pop(k, None)
    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    text = (tmp_path / "install" / "jobd.env").read_text()
    for k in ("B2_KEY_EXPIRES_AT", "CRED_BROKER_URL", "BOX_IDENTITY_NONCE",
              "CRED_ROLE", "TS_AUTHKEY"):
        assert k not in text, f"pre-broker jobd.env leaked {k}"


def _slow_cred_stub(tmp_path, sleep_s=25):
    """cred_client stand-in that records, then HANGS (a wedged transport) —
    exercises the background-refresh path: the poll loop must keep working."""
    rec = tmp_path / "cred_recorder.txt"
    stub = tmp_path / "cred_stub.py"
    stub.write_text("import sys, time\n"
                    f"open({str(rec)!r}, 'a').write('call\\n')\n"
                    f"time.sleep({sleep_s})\n"
                    "sys.exit(0)\n")
    return stub, rec


def test_jobd_cred_refresh_does_not_block_poll_loop(tmp_path):
    """A wedged cred_client (hung transport) must NOT stall the daemon: the
    refresh runs in the background, and a job staged AFTER the refresh started
    still gets claimed and run to done while the client is still hanging."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90107
    stub, rec = _slow_cred_stub(tmp_path, sleep_s=25)
    p = _popen_jobd(tmp_path, bucket, shimdir, iid, extra_env={
        "BOX_IDENTITY_NONCE": "ab" * 16,
        "B2_KEY_EXPIRES_AT": str(int(time.time()) + 600),   # imminent -> refresh
        "JOBD_CRED_CLIENT": str(stub),
    })
    try:
        assert _wait_for_file(rec), "cred_client never invoked"
        job_id, _ = _stage_job(tmp_path, bucket, iid)       # staged mid-refresh
        assert _wait_for_event(bucket, job_id, "done", timeout=20), \
            "poll loop stalled behind an in-flight cred refresh"
        assert rec.read_text().count("call") == 1, \
            "in-flight guard broken (second cred_client spawned)"
        assert p.poll() is None, "refresh path killed the daemon"
    finally:
        _kill(p)


def _pair_cred_stub(tmp_path, write_pair):
    """cred_client stand-in for a key-SHAPE change: rewrites $JOBD_ENV_FILE with
    a far expiry and (optionally) a scoped write pair, like the real client's
    build_jobd_env after a single<->pair rotation."""
    rec = tmp_path / "cred_recorder.txt"
    stub = tmp_path / "cred_stub.py"
    far = int(time.time()) + 30 * 86400
    lines = [f"export B2_KEY_EXPIRES_AT={far}"]
    if write_pair:
        lines += ["export B2_WRITE_KEY_ID=wkid-rotated",
                  "export B2_WRITE_APPLICATION_KEY=wkey-rotated"]
    envtext = "\\n".join(lines) + "\\n"
    stub.write_text("import os, sys\n"
                    f"open({str(rec)!r}, 'a').write('call\\n')\n"
                    f"open(os.environ['JOBD_ENV_FILE'], 'w').write('{envtext}')\n"
                    "sys.exit(0)\n")
    return stub, rec


def _run_cred_shape_flip(tmp_path, launch_write_key, stub_write_pair):
    """Boot jobd (poll loop), let one successful (stubbed) refresh land, kill,
    and return the daemon stderr — the 'cred refresh OK' line carries the
    recomputed write_remote. cwd=tmp_path: post-flip status writes hit the
    b2w: remote, which the shim treats as a local relative path (contained)."""
    bucket, shimdir = _make_bucket(tmp_path)
    stub, rec = _pair_cred_stub(tmp_path, write_pair=stub_write_pair)
    env = _hermetic_env(tmp_path)
    env["PATH"] = f"{shimdir}:{env['PATH']}"
    env["FAKE_BUCKET"] = str(bucket)
    env["B2_BUCKET"] = "testbucket"
    env["JOBD_IID"] = "90108"
    env["JOBD_ROOT"] = str(tmp_path / "workspace")
    env["JOBD_SKIP_GPU"] = "1"
    env["JOBD_SKIP_B2CONFIG"] = "1"
    env["JOBD_POLL"] = "1"
    env["JOBD_PYTHON"] = sys.executable
    _cred_hermetic(env, tmp_path)
    env.update({
        "BOX_IDENTITY_NONCE": "ab" * 16,
        "B2_KEY_EXPIRES_AT": str(int(time.time()) + 600),
        "JOBD_CRED_CLIENT": str(stub),
        "JOBD_ENV_FILE": str(tmp_path / "jobd.env"),
    })
    if launch_write_key:
        env["B2_WRITE_KEY_ID"] = "wkid-launch"
        env["B2_WRITE_APPLICATION_KEY"] = "wkey-launch"
    # stderr to a FILE, not a pipe, so the wait below can read the daemon's log
    # while it is still running. A pipe is only readable at communicate() time,
    # which is after the kill — too late to wait on anything it says.
    errf = tmp_path / "jobd.stderr"
    with open(errf, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(["bash", JOBD_SH], env=env, text=True,
                             cwd=str(tmp_path), stdout=subprocess.DEVNULL,
                             stderr=fh, start_new_session=True)
        try:
            assert _wait_for_file(rec), "cred_client never invoked"
            # Wait for the LINE, not for a guess at how long it takes. This was
            # `sleep(3)` for ">=1 tick + the re-source" with JOBD_POLL=1, which
            # is a wall-clock stand-in for a condition: green when the root runs
            # alone, red under `scripts/test_tools.py`, whose parallel roots put
            # three heavy suites on the box at once. Caught by a land gate.
            assert _wait_for_text(errf, "cred refresh OK"), \
                f"no cred refresh logged within 60s: {errf.read_text()}"
        finally:
            _kill(p)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    return errf.read_text()


def test_jobd_cred_refresh_single_to_pair_recomputes_b2w(tmp_path):
    """Scenario A of the frozen-B2W finding: a box launched on a single
    bucket-wide key gets a scoped PAIR from the broker (read half READ-ONLY).
    After the refresh jobd must move its writes to [b2w] — a launch-time-frozen
    B2W would 403 every heartbeat/result/status write until restart."""
    err = _run_cred_shape_flip(tmp_path, launch_write_key=False,
                               stub_write_pair=True)
    assert "cred refresh OK" in err, err
    assert "write_remote=b2w:testbucket" in err, \
        f"B2W not recomputed after single->pair rotation: {err}"


def test_jobd_cred_refresh_pair_to_single_recomputes_b2w(tmp_path):
    """Downgrade shape: the rotated jobd.env DROPS the write pair. Sourcing
    cannot unset vars, so jobd must clear B2_WRITE_* first — B2W falls back to
    [b2] instead of pointing at a remote cred_client stripped from rclone.conf."""
    err = _run_cred_shape_flip(tmp_path, launch_write_key=True,
                               stub_write_pair=False)
    assert "cred refresh OK" in err, err
    assert "write_remote=b2:testbucket" in err, \
        f"stale B2_WRITE_KEY_ID kept B2W on the dropped [b2w] remote: {err}"


def test_jobd_boot_resume_preserves_rotated_jobd_env(tmp_path):
    """Park/resume re-runs the onstart stanza with the ORIGINAL launch-time env.
    If cred_client (or `job attach`) rotated the key mid-session by rewriting
    jobd.env, the resumed boot must NOT clobber it back to launch creds."""
    bucket, shimdir = _make_bucket(tmp_path)
    snippet, env = _stage_boot(tmp_path, bucket, 90109, shimdir)
    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    envfile = tmp_path / "install" / "jobd.env"
    assert "export B2_KEY_ID=kid-xyz" in envfile.read_text()
    # cred_client rotates the key in place (fresh key id + far expiry)
    rotated = ("export B2_BUCKET=testbucket\n"
               "export B2_KEY_ID=kid-rotated\n"
               "export B2_APPLICATION_KEY=akey-rotated\n"
               "export B2_S3_ENDPOINT=https://s3.example\n"
               "export B2_KEY_EXPIRES_AT=9999999999\n")
    envfile.write_text(rotated)
    # ... the box parks; resume re-runs the SAME stanza with launch-time env
    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    text = envfile.read_text()
    assert "kid-rotated" in text, \
        "resume clobbered the cred_client-rotated jobd.env with launch creds"
    assert "kid-xyz" not in text, "launch-time key resurrected on resume"


def test_jobd_boot_regenerates_torn_jobd_env(tmp_path):
    """A jobd.env torn by a boot killed mid-write (no B2_KEY_ID line) is NOT
    treated as a rotation — the resumed boot regenerates it from launch env."""
    bucket, shimdir = _make_bucket(tmp_path)
    snippet, env = _stage_boot(tmp_path, bucket, 90110, shimdir)
    install = tmp_path / "install"
    install.mkdir()
    (install / "jobd.env").write_text("export B2_BUCKET=testbucket\n")  # torn
    r = subprocess.run(["bash", "-c", snippet + "\nwait\n"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    text = (install / "jobd.env").read_text()
    assert "export B2_KEY_ID=kid-xyz" in text, \
        "torn jobd.env (no B2_KEY_ID) was preserved instead of regenerated"


def test_jobd_scratch_shortfall_warns_but_never_refuses(tmp_path):
    """`needs.scratch_gb` is what the ENTRYPOINT writes (a ninja build tree, N
    compile worktrees), not bytes staging must land — so it must NOT refuse: a
    box that cannot fit an author's scratch estimate may still run the job fine,
    and refusing on an estimate is worse than the silence it replaces.

    But silence is how a job stages cleanly and then dies with a compiler ENOSPC
    nobody connects to disk. So: the job RUNS, and the log names the term."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90090
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "ascratch", "mkdir -p out\necho RAN > out/ran.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _asset_env()
    env["JOB_NEEDS_SCRATCH_GB"] = "999999"        # cannot possibly fit
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"scratch must not refuse: {v} stderr={r.stderr}"
    assert "disk WARNING" in r.stderr and "needs.scratch_gb" in r.stderr, r.stderr


def test_jobd_no_scratch_declaration_is_silent(tmp_path):
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90091
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "anoscr", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", v
    assert "disk WARNING" not in r.stderr


# ---------------------------------------------------------------------------
# B2 transfer guards on the box (owner scope-out 2026-08-02)
# ---------------------------------------------------------------------------
# Stall DETECTION for a box is a control-plane function — a wedged box is exactly
# the one that will not answer a probe, so fleetd owns that and these tests do
# NOT go near it. What is tested here is the carved-out corollary: a TRANSFER
# bounding its own runtime, so a slow or flaky host surfaces as a NAMED, distinct
# failure instead of a silent hang.
#
# The property that matters is DISTINGUISHABILITY. Before this, an asset pull
# that hung produced nothing at all, and one that failed produced
# `asset_stage_failed:<name>` — a transport-shaped reason that sends the operator
# to retry the same job on the same shape of box. A timeout and a slow host are
# different findings with a different action (re-rent elsewhere), so they get
# their own reasons.

def _hang_shimdir(tmp_path, shimdir, match, *, trickle=None):
    """A PATH `rclone` wrapping the shared shim that HANGS on a `copy` whose
    source matches `match` — the silent-peer shape a timeout must catch.

    `exec sleep` on purpose: the guard kills the PID it launched, and if the
    wrapper stayed alive as a parent the real sleeper would survive the kill and
    the test would hang instead of failing. With exec, the launched PID IS the
    sleeper.

    trickle=<dict>: files written into the destination BEFORE hanging, so the
    transfer has visibly started and then crawled — that is the throughput-floor
    case, which must be told apart from the never-started one."""
    d = tmp_path / f"hangbin{abs(hash(match)) % 10000}"
    d.mkdir()
    w = d / "rclone"
    pre = ""
    if trickle:
        for rel, body in trickle.items():
            pre += (f'  mkdir -p "$(dirname "$_dst/{rel}")" 2>/dev/null\n'
                    f'  printf %s {shlex.quote(body)} > "$_dst/{rel}"\n')
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        f'MATCH={shlex.quote(match)}\n'
        'if [ "${1:-}" = copy ] || [ "${1:-}" = sync ]; then\n'
        '  _hit=0; _dst=""\n'
        '  for a in "$@"; do\n'
        '    case "$a" in --*) continue ;; esac\n'
        '    case "$a" in *"$MATCH"*) _hit=1 ;; esac\n'
        '    case "$a" in b2:*) : ;; *) _dst="$a" ;; esac\n'
        '  done\n'
        '  if [ "$_hit" = 1 ]; then\n'
        f'{pre}'
        '    exec sleep 900\n'
        '  fi\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def _guard_env(**kw):
    env = _asset_env()
    # Ceiling arithmetic is bytes/floor + slack, floored at MIN_TIMEOUT_S. Zero
    # the slack so a test asset's ceiling collapses to the minimum, which is the
    # only number a test then has to set.
    env["JOBD_ASSET_SLACK_S"] = "0"
    env.update({k: str(v) for k, v in kw.items()})
    return env


def test_jobd_asset_pull_timeout_is_named_not_hung(tmp_path):
    """A host that accepts the connection and then goes silent must NOT park the
    daemon. The pull is killed at its ceiling and the job goes terminal with
    `asset_stage_timeout:` — a reason distinct from `asset_stage_failed:`."""
    bucket, shimdir = _make_bucket(tmp_path)
    hangdir = _hang_shimdir(tmp_path, shimdir, "assets-src/base")
    iid = 90120
    _put_asset_files(bucket, "assets-src/base", {"model.safetensors": "WEIGHTS"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "atimeout", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _guard_env(JOBD_ASSET_MIN_TIMEOUT_S=3, JOBD_ASSET_RETRIES=1,
                     JOBD_ASSET_MIN_MBPS=0)          # floor off: isolate the ceiling
    t0 = time.time()
    r = _run_jobd(tmp_path, bucket, hangdir, iid, extra_env=env)
    elapsed = time.time() - t0

    assert r.returncode == 0, r.stderr
    # It must return in something like its ceiling, not the 120s subprocess cap.
    assert elapsed < 60, f"guard did not bound the pull: {elapsed:.0f}s"
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"{v} stderr={r.stderr}"
    reason = v["fail_reason"] or ""
    assert "asset_stage_timeout:base" in reason, reason
    assert "asset_stage_failed" not in reason, \
        f"a timeout must not be reported as a generic failure: {reason}"
    # and the log must name the cause AND the knob that governs it
    assert "TIMEOUT" in r.stderr and "JOBD_ASSET_TIMEOUT_S" in r.stderr, r.stderr


def test_jobd_asset_pull_slow_host_is_named_and_distinct(tmp_path):
    """A transfer that STARTED and then crawled is a HOST verdict, not a broken
    pull: it is condemned by the throughput floor and reported as
    `asset_stage_slow:`, distinct from both timeout and failure."""
    bucket, shimdir = _make_bucket(tmp_path)
    hangdir = _hang_shimdir(tmp_path, shimdir, "assets-src/base",
                            trickle={"config.json": "{}"})
    iid = 90121
    _put_asset_files(bucket, "assets-src/base", {"model.safetensors": "WEIGHTS"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "aslow", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    # Window well inside the ceiling, so the FLOOR is what fires. A real box uses
    # 300s/3 MB/s; the ratio, not the magnitude, is what is under test.
    env = _guard_env(JOBD_ASSET_MIN_TIMEOUT_S=120, JOBD_ASSET_RETRIES=1,
                     JOBD_ASSET_MBPS_WINDOW_S=3, JOBD_ASSET_MIN_MBPS=3)
    t0 = time.time()
    r = _run_jobd(tmp_path, bucket, hangdir, iid, extra_env=env)
    elapsed = time.time() - t0

    assert r.returncode == 0, r.stderr
    assert elapsed < 60, f"floor did not fire inside the ceiling: {elapsed:.0f}s"
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"{v} stderr={r.stderr}"
    reason = v["fail_reason"] or ""
    assert "asset_stage_slow:base" in reason, reason
    assert "SLOW HOST" in r.stderr and "JOBD_ASSET_MIN_MBPS" in r.stderr, r.stderr


def test_jobd_asset_guard_never_touches_a_healthy_pull(tmp_path):
    """The false-positive rail. A guard that fires on a normal pull would be
    switched off within a week, which is worse than having none — so a healthy
    pull under the SHIPPED defaults must be completely unaffected."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90122
    _put_asset_files(bucket, "assets-src/base",
                     {"config.json": '{"ok":1}', "model.safetensors": "WEIGHTS"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "afast", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_asset_env())
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", f"{v} stderr={r.stderr}"
    assert (tmp_path / "workspace" / "assets" / "base" / "config.json").read_text() \
        == '{"ok":1}'
    for word in ("TIMEOUT", "SLOW HOST"):
        assert word not in r.stderr, f"guard fired on a healthy pull: {r.stderr}"
    # and it announced its budget, so an operator can see what it was judged by
    assert "pull guard" in r.stderr, r.stderr


def test_jobd_asset_guard_can_be_disarmed(tmp_path):
    """JOBD_ASSET_GUARD=0 restores the pre-guard path exactly. A safety net with
    no field kill-switch is one that gets deleted under pressure instead."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90123
    _put_asset_files(bucket, "assets-src/base", {"config.json": "{}"})
    job_id = _stage_asset_job(
        tmp_path, bucket, iid, "aoff", "mkdir -p out\necho ok > out/r.txt\n",
        [{"name": "base", "b2": "assets-src/base"}])
    env = _asset_env()
    env["JOBD_ASSET_GUARD"] = "0"
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=env)
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "done", v
    assert "pull guard" not in r.stderr, r.stderr


def _bundle_shimdir(tmp_path, shimdir, mode):
    """A PATH `rclone` that breaks ONLY the bundle `copyto` — either hanging
    (mode='hang') or returning a revoked-key error (mode='auth'). Everything else
    (events, tickets, results) goes to the real shim, so the test observes the
    bundle path in isolation."""
    d = tmp_path / f"bundlebin-{mode}"
    d.mkdir()
    w = d / "rclone"
    broken = ('    exec sleep 900\n' if mode == "hang" else
              '    echo "ERROR: InvalidAccessKeyId: The key '
              "'004deadKEY0000000000003' is not valid\" >&2\n"
              '    exit 1\n')
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        'if [ "${1:-}" = copyto ]; then\n'
        '  case "${2:-}" in\n'
        '    */jobs/bundles/*)\n'
        f'{broken}'
        '      ;;\n'
        '  esac\n'
        'fi\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


def test_jobd_bundle_download_timeout_is_named(tmp_path):
    """The bundle pull was the LEAST guarded B2 read on the box despite being the
    one every job passes through: a bare copyto, no timeout, no retry. A silent
    peer must now bound it and say so."""
    bucket, shimdir = _make_bucket(tmp_path)
    bdir = _bundle_shimdir(tmp_path, shimdir, "hang")
    iid = 90124
    job_id = _stage_asset_job(tmp_path, bucket, iid, "btimeout",
                              "mkdir -p out\necho ok > out/r.txt\n", [])
    t0 = time.time()
    r = _run_jobd(tmp_path, bucket, bdir, iid,
                  extra_env={"JOBD_BUNDLE_TIMEOUT_S": "3", "JOBD_BUNDLE_RETRIES": "1"})
    elapsed = time.time() - t0

    assert r.returncode == 0, r.stderr
    assert elapsed < 60, f"bundle pull was not bounded: {elapsed:.0f}s"
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"{v} stderr={r.stderr}"
    reason = v["fail_reason"] or ""
    assert "TIMEOUT" in reason and "JOBD_BUNDLE_TIMEOUT_S" in reason, reason


def test_jobd_bundle_download_auth_failure_names_the_cause(tmp_path):
    """THE governing precedent. The old code sent bundle stderr to /dev/null and
    emitted 'bundle download failed', so a revoked key's InvalidAccessKeyId was
    invisible and the operator had to guess. The cause must reach the event."""
    bucket, shimdir = _make_bucket(tmp_path)
    bdir = _bundle_shimdir(tmp_path, shimdir, "auth")
    iid = 90125
    job_id = _stage_asset_job(tmp_path, bucket, iid, "bauth",
                              "mkdir -p out\necho ok > out/r.txt\n", [])
    r = _run_jobd(tmp_path, bucket, bdir, iid,
                  extra_env={"JOBD_BUNDLE_RETRIES": "2", "JOBD_BUNDLE_BACKOFF": "0"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v["status"] == "failed", f"{v} stderr={r.stderr}"
    reason = v["fail_reason"] or ""
    assert "AUTH FAILURE" in reason, reason
    assert "InvalidAccessKeyId" in reason, \
        f"the actual B2 error must survive into the reason: {reason}"


def test_jobd_ships_a_live_append_style_checkpoint(tmp_path):
    """A CONTINUOUSLY-APPENDED checkpoint file must reach B2 while it is still
    being written.

    LANDED FIX (task #110). This carried `xfail(strict=True)` from `c663e717`
    until the tail-snapshot path landed; the marker came off the moment the
    behaviour changed, which is exactly what strict=True was for. The `before`
    reading is preserved: on the unpatched daemon this test FAILS with
    `checkpoints/out/gens.jsonl` absent.

    jobd's periodic pass ships with `rclone --min-age ${JOBD_CKPT_MIN_AGE:-45s}`
    so it never uploads a file the entrypoint is mid-write on. But an
    append-style producer (gen_probe_resumable writes + flushes + fsyncs a row
    after EVERY function) refreshes the file's mtime every few seconds, so it is
    NEVER old enough to be eligible and was NEVER shipped. The fire-on-arrival
    fast path, which does bypass the age window, arms only on `checkpoint-<N>/`
    DIRECTORIES — a trainer shape an append-style .jsonl can never match.

    Measured cost, drift-roster-r3 2026-08-06: twelve checkpoint passes uploaded
    the FINISHED arm exactly once (at the second it went quiet) and the IN-FLIGHT
    arm not once in ten minutes of generating; ~200 rows were lost. It is worse
    than a box-death exposure, because jobd's job start does `rm -rf "$wdir/work"`
    and re-pulls the checkpoints/ prefix — so ANY restart destroys whatever never
    shipped, even when the box's disk survived intact.

    The job below appends to one file for ~12 s with checkpoint_s=3, so several
    periodic passes run while the file is live. At exit the file must be on B2 —
    and every line of it must be a COMPLETE record, because durability that
    ships a torn tail is a worse defect than the one it fixes.
    """
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90455
    # append a row every ~0.5s for ~12s: mtime is never more than ~0.5s old, so
    # the 45s age window can never be cleared while the job is running.
    entry = (
        "mkdir -p out\n"
        "i=0\n"
        "while [ $i -lt 24 ]; do\n"
        "  printf '{\"id\": \"fn%d\", \"generations\": [\"x\"]}\\n' \"$i\" >> out/gens.jsonl\n"
        "  i=$((i+1))\n"
        "  sleep 0.5\n"
        "done\n")
    config = (
        "version: 1\nname: ckpt-append\nentrypoint: run.sh\ntimeout_s: 60\n"
        "checkpoint_s: 3\n"
        "checkpoints:\n  - \"out/gens.jsonl\"\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpu: false\n  venv: none\n")
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry, config=config)
    # DEFAULT min-age deliberately left in place — the fix must NOT work by
    # widening the age window; the window is what keeps a mid-write copy out.
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_CKPT_WATCH_S": "1"})
    assert r.returncode == 0, r.stderr
    obj = bucket / "jobs" / job_id / "checkpoints" / "out" / "gens.jsonl"
    assert obj.is_file(), (
        "a live append-style checkpoint never reached B2 — every periodic pass "
        "skipped it because --min-age 45s can never be cleared by a file being "
        "appended to every 0.5s. A reclaim or ANY restart here loses the whole "
        f"in-flight arm.\n{r.stderr[-3000:]}")
    body = obj.read_bytes()
    assert body.endswith(b"\n"), (
        "the shipped snapshot must end on a record boundary — a trailing "
        f"fragment is a torn tail: {body[-120:]!r}")
    rows = [json.loads(l) for l in body.decode().splitlines() if l.strip()]
    assert rows, "shipped an empty snapshot"
    # ids are fn0..fn23 in order; whatever prefix landed must be exactly that
    # prefix, with no gap and no half-record.
    assert [r_["id"] for r_ in rows] == [f"fn{i}" for i in range(len(rows))], rows
    # It has to be MID-flight to be worth anything: shipping only the finished
    # file at the last quiet pass is the defect wearing a hat. At least one
    # checkpoint event must report tail>=1 while the entrypoint was still running.
    ck = [e for e in (json.loads(b) for b in _events(bucket, job_id))
          if e.get("event") == "checkpoint"]
    assert any(int(e.get("tail") or 0) > 0 for e in ck), \
        f"no checkpoint pass shipped a live tail: {[e.get('tail') for e in ck]}"


# --------------------------------------------------------------------------- #
# stall detection (JOBD_STALL_S) — a wedged entrypoint writes NOTHING and burns
# the rental. Measured 2026-08-06 on job 20260806T082213-v11-...: a NCCL
# collective deadlocked on its FIRST all-reduce and sat silent until torch's own
# 1800s watchdog fired — twice, on two boxes, ~30 min of paid silence each, and
# nothing in jobd was looking at the log having stopped growing.
# --------------------------------------------------------------------------- #
def test_jobd_emits_stall_suspected_when_entrypoint_log_goes_quiet(tmp_path):
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90101
    # writes once, then goes silent well past the threshold, then finishes
    entry = 'mkdir -p out\necho "starting"\nsleep 6\necho done > out/result.txt\n'
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry)
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env={"JOBD_STALL_S": "2"})
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "stall_suspected" in kinds, f"kinds={kinds} stderr={r.stderr}"

    ev = [json.loads(b) for b in bodies if json.loads(b)["event"] == "stall_suspected"][0]
    assert int(ev["threshold_s"]) == 2
    assert int(ev["quiet_s"]) >= 2
    assert int(ev["log_bytes"]) > 0        # it HAD written; it then stopped

    # it is an ALARM, never a kill: the job still runs to completion
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"kinds={kinds}"

    # one event per quiet window, not one per heartbeat (HB=1 over a ~6s quiet
    # stretch would otherwise emit ~4 and spam the log)
    assert kinds.count("stall_suspected") == 1, kinds


def test_jobd_stall_detector_stays_silent_on_a_chatty_entrypoint(tmp_path):
    """A job that keeps writing must never be flagged — the detector watches
    log GROWTH, not elapsed time."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90102
    entry = ('mkdir -p out\nfor i in 1 2 3 4 5 6 7 8 9 10; do echo "tick $i"; '
             'sleep 0.5; done\necho done > out/result.txt\n')
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry)
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env={"JOBD_STALL_S": "2"})
    assert r.returncode == 0, r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "stall_suspected" not in kinds, kinds


def test_jobd_stall_detector_disabled_by_zero(tmp_path):
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90103
    entry = 'mkdir -p out\necho "starting"\nsleep 6\necho done > out/result.txt\n'
    job_id, _ = _stage_job(tmp_path, bucket, iid, entry=entry)
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env={"JOBD_STALL_S": "0"})
    assert r.returncode == 0, r.stderr
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "stall_suspected" not in kinds, kinds


# ------------------------------------------------------------------------------
# shared Triton JIT cache (triton_cache_boot_pull / TRITON_CACHE_DIR export /
# triton_cache_push_bg) — the $0 end-to-end proof of the whole loop
# ------------------------------------------------------------------------------
def test_triton_cache_gpu_export_push_and_second_boot_pull(tmp_path):
    """One jobd pass on a (fake) GPU box: the GPU job's entrypoint sees
    TRITON_CACHE_DIR pointing at the box-level dir while the CPU job sees
    nothing; the kernel the GPU job 'compiles' is pushed to the remote after
    the job; and a SECOND boot on a fresh workspace pulls it back (hit=True)
    — the cross-box reuse this whole lane exists for."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90071
    cfg_gpu = (
        "version: 1\nname: aa-tc-gpu\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  gpus: 1\n  venv: none\n")
    cfg_cpu = (
        "version: 1\nname: bb-tc-cpu\nentrypoint: run.sh\ntimeout_s: 60\n"
        "results:\n  - \"out/**\"\n"
        "needs:\n  venv: none\n")
    entry_gpu = (
        "mkdir -p out\n"
        "echo \"TCD=${TRITON_CACHE_DIR:-unset}\" > out/tc.txt\n"
        "mkdir -p \"$TRITON_CACHE_DIR/deadbeefkernel\"\n"
        "echo fakecubin > \"$TRITON_CACHE_DIR/deadbeefkernel/kernel.cubin\"\n")
    entry_cpu = "mkdir -p out\necho \"TCD=${TRITON_CACHE_DIR:-unset}\" > out/tc.txt\n"
    j_gpu = _stage_named(tmp_path, bucket, iid, "tcg", entry_gpu, cfg_gpu, "aaaa")
    j_cpu = _stage_named(tmp_path, bucket, iid, "tcc", entry_cpu, cfg_cpu, "bbbb")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:32", "JOBD_SKIP_GPU": "0"})
    assert r.returncode == 0, r.stderr
    for jid in (j_gpu, j_cpu):
        v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
        assert v["status"] == "done", f"{jid}: {v} stderr={r.stderr}"
    ws = tmp_path / "workspace"
    # (a) the GPU entrypoint saw the box-level dir; the CPU one saw nothing
    tcd = (bucket / "jobs" / j_gpu / "results" / "out" / "tc.txt").read_text().strip()
    assert tcd == f"TCD={ws}/triton-cache", tcd
    assert (bucket / "jobs" / j_cpu / "results" / "out" / "tc.txt"
            ).read_text().strip() == "TCD=unset"
    # (b) the post-job push published tarball + digest sidecar to the remote
    remote = bucket / "triton-cache"
    tars = list(remote.glob("*.tar.gz")) if remote.is_dir() else []
    assert len(tars) == 1, f"push did not land: {list(remote.iterdir()) if remote.is_dir() else 'no dir'}\nstderr={r.stderr[-2000:]}"
    assert tars[0].with_suffix("").with_suffix(".sha256").exists() or \
        (remote / (tars[0].name[:-len(".tar.gz")] + ".sha256")).exists()
    # (c) a SECOND boot on a FRESH workspace (same fake remote, empty queue)
    # pulls the kernel back — the cross-box hit
    ws2 = tmp_path / "workspace2"
    r2 = _run_jobd(tmp_path, bucket, shimdir, iid,
                   extra_env={"JOBD_FAKE_GPUS": "0:32", "JOBD_SKIP_GPU": "0",
                              "JOBD_ROOT": str(ws2)})
    assert r2.returncode == 0, r2.stderr
    assert (ws2 / "triton-cache" / "deadbeefkernel" / "kernel.cubin").read_text() \
        == "fakecubin\n"
    pull = json.loads((ws2 / ".triton_cache.pull.json").read_text())
    assert pull["hit"] is True and pull["entries_installed"] == 1, pull


def test_triton_cache_kill_switch_disables_everything(tmp_path):
    """JOBD_TRITON_CACHE=0: no pull, no export, no push — byte-for-byte the
    pre-lane behavior."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90072
    cfg = ("version: 1\nname: tc-off\nentrypoint: run.sh\ntimeout_s: 60\n"
           "results:\n  - \"out/**\"\n"
           "needs:\n  gpus: 1\n  venv: none\n")
    entry = "mkdir -p out\necho \"TCD=${TRITON_CACHE_DIR:-unset}\" > out/tc.txt\n"
    jid = _stage_named(tmp_path, bucket, iid, "tco", entry, cfg, "cccc")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env={"JOBD_FAKE_GPUS": "0:32", "JOBD_SKIP_GPU": "0",
                             "JOBD_TRITON_CACHE": "0"})
    assert r.returncode == 0, r.stderr
    v = jm.fold_events(_events(bucket, jid), live_iids={str(iid)})
    assert v["status"] == "done", v
    assert (bucket / "jobs" / jid / "results" / "out" / "tc.txt"
            ).read_text().strip() == "TCD=unset"
    assert not (tmp_path / "workspace" / ".triton_cache.pull.json").exists()
    assert not (bucket / "triton-cache").exists()


# ---------------------------------------------------------------------------
# Interrupted transfer: self-heal + ONE bounded retry (defect #77, 2026-08-09)
#
# The shape these tests reproduce is the real one, not an abstraction of it. A
# bundle stages weights OUTSIDE the job workdir (so a resume does not re-pull
# 34 GB) behind a presence guard:
#     if [ ! -f "$dir/model.safetensors.index.json" ]; then ... rclone copy ...
# (driftr3-h200-gen/run.sh:383, :449). The index lands EARLY, so an interrupted
# pull leaves the guard ARMED plus a `<name>.<rand>.partial` corpse — and every
# later attempt skips the pull and fails the bundle's own shard gate identically.
# The entrypoint below is exactly that: guard, early index, one shard, a partial
# for the shard that never landed, then the gate that `die`s with rc=4.
# ---------------------------------------------------------------------------
_SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
_INDEX_JSON = json.dumps({"weight_map": {"a.weight": _SHARDS[0], "b.weight": _SHARDS[1]}})


def _weights_src(tmp_path):
    """A stand-in for the B2 side of a weights pull (plain files on disk — the
    entrypoint `cp`s from it, so no shim op is involved and the test is only
    about what jobd does with what the pull LEFT BEHIND)."""
    d = tmp_path / "weights-src"
    d.mkdir(exist_ok=True)
    (d / "model.safetensors.index.json").write_text(_INDEX_JSON)
    (d / _SHARDS[0]).write_text("SHARD-ONE")
    (d / _SHARDS[1]).write_text("SHARD-TWO")
    return d


def _transfer_entry(tmp_path, succeed_from=99):
    """The driftr3-shaped pull block + completeness gate. `succeed_from` is the
    attempt number from which the (re-armed) pull manages to land both shards;
    99 = never. Writes an attempt counter and a pull log OUTSIDE the workdir,
    because poll_once wipes the workdir on every claim."""
    src = _weights_src(tmp_path)
    models = tmp_path / "workspace" / "models"       # under $ROOT, outside $JOBS_DIR
    state = tmp_path / "entry-state"
    state.mkdir(exist_ok=True)
    return (
        f'MD="{models}"\nST="{state}"\nSRC="{src}"\n'
        'mkdir -p "$MD" "$ST" out\n'
        'N=$(cat "$ST/n" 2>/dev/null || echo 0); N=$((N+1)); echo "$N" > "$ST/n"\n'
        '# --- the bundle\'s own pull block, guarded the way real bundles guard it\n'
        'if [ ! -f "$MD/model.safetensors.index.json" ]; then\n'
        '  echo "pull-$N" >> "$ST/pulls"\n'
        '  cp "$SRC/model.safetensors.index.json" "$MD/"\n'
        f'  cp "$SRC/{_SHARDS[0]}" "$MD/"\n'
        f'  if [ "$N" -ge {succeed_from} ]; then\n'
        f'    cp "$SRC/{_SHARDS[1]}" "$MD/"\n'
        '  else\n'
        f'    : > "$MD/{_SHARDS[1]}.4f2a1b.partial"\n'
        '  fi\n'
        'fi\n'
        '# --- the bundle\'s own completeness gate (`die ... 4`)\n'
        f'if [ ! -f "$MD/{_SHARDS[1]}" ]; then\n'
        '  echo "9B base pull failed: shards missing under $MD" >&2\n'
        '  exit 4\n'
        'fi\n'
        'echo ok > out/r.txt\n')


def _transfer_config(name, max_restarts=None):
    return (f"version: 1\nname: {name}\nentrypoint: run.sh\ntimeout_s: 60\n"
            + (f"max_restarts: {max_restarts}\n" if max_restarts is not None else "")
            + "results:\n  - \"out/**\"\n"
              "needs:\n  gpu: false\n  venv: none\n")


def _transfer_env(**over):
    env = {"JOBD_TRANSFER_BACKOFF_S": "0"}   # no backoff sleeps in tests
    env.update(over)
    return env


def _state_int(tmp_path, job_id, suffix):
    p = tmp_path / "workspace" / "jobs" / ".state" / f"{job_id}.{suffix}"
    return int(p.read_text().strip()) if p.is_file() else 0


def _models_dir(tmp_path):
    return tmp_path / "workspace" / "models"


def test_jobd_sweep_removes_index_when_shards_are_missing(tmp_path):
    """THE SKIP-GUARD DEFEAT — the piece without which the whole retry is a no-op.

    After an interrupted pull the model dir holds an index.json naming two shards,
    one shard, and a `.partial` for the other. A bare retry would hit the bundle's
    `if [ ! -f .../model.safetensors.index.json ]` guard, SKIP the pull, and fail
    the identical gate forever. jobd must therefore delete BOTH the orphan partial
    and the index whose shards never landed, so the bundle's own guard re-arms."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90080
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-heal",
                          _transfer_entry(tmp_path), _transfer_config("xfer-heal"),
                          "hea1")
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r.returncode == 0, r.stderr

    md = _models_dir(tmp_path)
    # the interrupted pull really did happen (guard armed, shard 2 absent)
    assert (md / _SHARDS[0]).is_file(), "the test's own pull block never ran"
    # ...and jobd healed it: index GONE (guard re-armed), partial GONE
    assert not (md / "model.safetensors.index.json").exists(), \
        "index survived — the pull guard is still armed and every retry will skip the pull"
    assert not list(md.glob("*.partial")), \
        f"orphan partials survived: {[p.name for p in md.glob('*.partial')]}"
    # the shard that DID land is untouched: the re-pull must stay cheap
    assert (md / _SHARDS[0]).read_text() == "SHARD-ONE"
    # non-terminal + breadcrumb: this is a retry, not a failure
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "transfer_retry" in kinds, f"kinds={kinds} stderr={r.stderr}"
    assert "failed" not in kinds, kinds
    state = tmp_path / "workspace" / "jobs" / ".state"
    assert (state / f"{job_id}.transfer_retry").is_file(), "no retry breadcrumb"
    assert not (state / f"{job_id}.terminal").exists(), \
        "job was marked terminal — the ticket would be skipped forever"
    assert not (bucket / "jobs" / job_id / "results.DONE.json").exists(), \
        "a DONE marker would make poll_once skip the job on every later pass"


def test_jobd_no_partials_means_terminal_first_try(tmp_path):
    """OVER-FIRING GUARD. The classifier fires on FILESYSTEM EVIDENCE, never on
    the rc integer (rc=4 is b2x's own '404' — b2x/main.go:46-54 — and entrypoints
    use small rcs for everything). An ordinary rc!=0 with no `.partial` anywhere
    must stay terminal on the FIRST attempt, exactly as before this lane existed."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90081
    # rc=4 deliberately: the SAME rc the interrupted-transfer case exits with.
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-nofire",
                          "mkdir -p out\necho boom >&2\nexit 4\n",
                          _transfer_config("xfer-nofire"), "nof1")
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "transfer_retry" not in kinds, \
        f"retried on the rc alone, with no evidence on disk: {kinds}"
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", v
    assert "interrupted_transfer" not in (v["fail_reason"] or ""), v
    state = tmp_path / "workspace" / "jobs" / ".state"
    assert (state / f"{job_id}.terminal").is_file()
    assert not (state / f"{job_id}.transfer_retries").exists()


def test_jobd_transfer_retry_does_not_burn_max_restarts(tmp_path):
    """The retry rides its OWN counter. Staged with max_restarts=0 (crash budget
    = 1 run), so a retry counted as a crash-restart would fail the second claim at
    the restart cap — the same misclassification that made three outbids fail a
    healthy job (box 44566398 lineage). Two boots, both interrupted:
      boot 1  claim   -> .attempts=1, heal, transfer_retry
      boot 2  resume  -> .transfer_retries=1, .attempts UNCHANGED, budget spent
                         -> terminal with the NAMED interrupted_transfer reason
    """
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90082
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-cap",
                          _transfer_entry(tmp_path),
                          _transfer_config("xfer-cap", max_restarts=0), "cap1")
    r1 = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r1.returncode == 0, r1.stderr
    assert _state_int(tmp_path, job_id, "attempts") == 1
    assert _state_int(tmp_path, job_id, "transfer_retries") == 0

    r2 = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r2.returncode == 0, r2.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "resumed" in kinds, f"kinds={kinds} stderr={r2.stderr}"
    resumed = [json.loads(b) for b in bodies if json.loads(b)["event"] == "resumed"]
    assert resumed[-1].get("kind") == "transfer", resumed
    assert resumed[-1].get("detect") == "transfer", resumed
    # the crash budget was NEVER touched
    assert _state_int(tmp_path, job_id, "attempts") == 1, \
        "the transfer retry spent max_restarts"
    assert _state_int(tmp_path, job_id, "transfer_retries") == 1
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", v
    assert "interrupted_transfer" in (v["fail_reason"] or ""), v
    assert "restart cap" not in (v["fail_reason"] or ""), \
        f"the retry was counted as a crash: {v['fail_reason']}"
    # bounded: exactly one retry, then terminal
    assert kinds.count("transfer_retry") == 1, kinds


def test_jobd_interrupted_transfer_heals_then_second_attempt_succeeds(tmp_path):
    """HAPPY PATH end to end: partials + missing shards -> self-heal -> retry ->
    the re-armed guard lets the pull run AGAIN -> both shards land -> done.

    The pull log is the load-bearing assertion. Two entries means the bundle's
    pull block really re-executed; one would mean the index survived and the
    retry skipped the pull (which is exactly how a naive retry fails)."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90083
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-ok",
                          _transfer_entry(tmp_path, succeed_from=2),
                          _transfer_config("xfer-ok", max_restarts=0), "ok11")
    r1 = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r1.returncode == 0, r1.stderr
    v1 = jm.fold_events(_events(bucket, job_id), live_iids={str(iid)})
    assert v1["status"] != "failed", v1

    r2 = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r2.returncode == 0, r2.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "done", f"kinds={kinds} v={v} stderr={r2.stderr}"
    pulls = (tmp_path / "entry-state" / "pulls").read_text().split()
    assert pulls == ["pull-1", "pull-2"], \
        f"the pull did not re-run — the skip-guard was not defeated: {pulls}"
    md = _models_dir(tmp_path)
    assert (md / _SHARDS[1]).read_text() == "SHARD-TWO"
    assert not list(md.glob("*.partial"))
    assert (bucket / "jobs" / job_id / "results" / "out" / "r.txt").is_file()


def test_jobd_disk_full_is_not_classified_as_interrupted(tmp_path):
    """ENOSPC leaves partials INDISTINGUISHABLE from an interrupted pull, and a
    retry into a full disk just buys the same failure twice. Under the free-space
    floor the verdict is its own terminal reason carrying the number — never a
    retry. (Floor pinned absurdly high so the check fires on any real machine.)"""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90084
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-disk",
                          _transfer_entry(tmp_path), _transfer_config("xfer-disk"),
                          "dsk1")
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env=_transfer_env(JOBD_TRANSFER_MIN_FREE_GB="100000000"))
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "transfer_retry" not in kinds, f"retried into a full disk: {kinds}"
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", v
    assert "insufficient_disk" in (v["fail_reason"] or ""), v
    assert _state_int(tmp_path, job_id, "transfer_retries") == 0
    # and nothing was deleted: the disk gate runs BEFORE the heal
    assert (_models_dir(tmp_path) / "model.safetensors.index.json").is_file()


def test_jobd_transfer_corruption_is_terminal_not_retried(tmp_path):
    """MIXED EVIDENCE. A file at its FINAL path whose recorded sha256 disagrees is
    not an interrupt — something renamed bad bytes into place, and a re-pull may
    skip it entirely (rclone compares size/mtime, not content). Terminal."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90085
    md = _models_dir(tmp_path)
    # the manifest goes down FIRST — the gate below `exit`s, so anything appended
    # after _transfer_entry would never run.
    entry = (f'mkdir -p "{md}"\n'
             f'printf "%s  %s\\n" "{"0" * 64}" "{_SHARDS[0]}" > "{md}/SHA256SUMS"\n'
             + _transfer_entry(tmp_path))
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-corrupt", entry,
                          _transfer_config("xfer-corrupt"), "cor1")
    r = _run_jobd(tmp_path, bucket, shimdir, iid, extra_env=_transfer_env())
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    kinds = [json.loads(b)["event"] for b in bodies]
    assert "transfer_retry" not in kinds, f"retried over corruption: {kinds}"
    v = jm.fold_events(bodies, live_iids={str(iid)})
    assert v["status"] == "failed", v
    assert "transfer_corruption" in (v["fail_reason"] or ""), v


def test_jobd_preempt_during_transfer_bumps_preempts_not_transfer_retries(tmp_path):
    """ORDERING IS LOAD-BEARING. A box stop kills the in-flight pull too and leaves
    the SAME `.partial` corpses behind — so the interrupted-transfer classifier
    sits strictly AFTER run_job_body's preempt block, which returns first. An
    eviction must drain JOBD_PREEMPT_CAP, never the transfer budget."""
    bucket, shimdir = _make_bucket(tmp_path)
    iid = 90086
    md = _models_dir(tmp_path)
    # leave the interrupted-pull debris, then block: the box stop lands mid-"pull"
    entry = (f'MD="{md}"\nmkdir -p "$MD" out\n'
             f'cp /dev/null "$MD/{_SHARDS[1]}.4f2a1b.partial"\n'
             f'printf "%s" \'{_INDEX_JSON}\' > "$MD/model.safetensors.index.json"\n'
             'echo step-1 > out/state.txt\nsleep 60\n')
    config = _transfer_config("xfer-preempt", max_restarts=0).replace(
        "timeout_s: 60\n", "timeout_s: 120\ncheckpoint_s: 1\n")
    job_id = _stage_named(tmp_path, bucket, iid, "xfer-preempt", entry, config, "pre1")
    p = _popen_jobd(tmp_path, bucket, shimdir, iid,
                    extra_env=_transfer_env(JOBD_CKPT_MIN_AGE="0s"))
    pgid = os.getpgid(p.pid)
    try:
        assert _wait_for_event(bucket, job_id, "checkpoint"), "state never synced"
        os.kill(p.pid, signal.SIGTERM)     # graceful preempt: trap raises the marker
        p.wait(timeout=60)
        os.killpg(pgid, signal.SIGKILL)    # a real box stop SIGKILLs the container
        time.sleep(1)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    state = tmp_path / "workspace" / "jobs" / ".state"
    assert (state / f"{job_id}.preempted").is_file(), "no preempt breadcrumb"
    assert not (state / f"{job_id}.transfer_retry").exists(), \
        "the transfer classifier ran on a PREEMPT — it must be ordered after the preempt block"
    kinds = [json.loads(b)["event"] for b in _events(bucket, job_id)]
    assert "transfer_retry" not in kinds, kinds
    # the debris is still there, untouched: nothing healed a preempted job
    assert (md / "model.safetensors.index.json").is_file()

    # resume boot: it drains the PREEMPT budget, and the transfer budget stays 0
    r = _run_jobd(tmp_path, bucket, shimdir, iid,
                  extra_env=_transfer_env(JOBD_CKPT_MIN_AGE="0s"))
    assert r.returncode == 0, r.stderr
    bodies = _events(bucket, job_id)
    resumed = [json.loads(b) for b in bodies if json.loads(b)["event"] == "resumed"]
    assert resumed and resumed[-1].get("kind") == "preempt", resumed
    assert _state_int(tmp_path, job_id, "preempts") == 1
    assert _state_int(tmp_path, job_id, "transfer_retries") == 0, \
        "an eviction spent the interrupted-transfer budget"


# ---------------------------------------------------------------------------
# Queue-delete invariant (audit finding out of defect #75's investigation)
# ---------------------------------------------------------------------------
def _oplog_shimdir(tmp_path, shimdir):
    """A PATH dir whose `rclone` records EVERY invocation (one argv per line)
    before delegating to the shared shim. Returns (pathdir, logfile)."""
    d = tmp_path / "opbin"
    d.mkdir()
    logf = d / "ops.log"
    w = d / "rclone"
    w.write_text(
        "#!/usr/bin/env bash\n"
        f'REAL={shlex.quote(str(shimdir / "rclone"))}\n'
        f'printf "%s\\n" "$*" >> {shlex.quote(str(logf))}\n'
        'exec "$REAL" "$@"\n')
    w.chmod(w.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d, logf


_DESTRUCTIVE_OPS = ("delete", "deletefile", "purge", "rmdir", "rmdirs", "move", "moveto")


def _queue_deletes(logf):
    """Destructive rclone invocations that name a jobs/queue/ path."""
    hits = []
    if not logf.is_file():
        return hits
    for line in logf.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in _DESTRUCTIVE_OPS and any(
                "jobs/queue/" in a for a in parts[1:]):
            hits.append(line)
    return hits


def test_poll_once_never_deletes_a_queue_ticket(tmp_path):
    """THE INVARIANT jobd.sh STATES BUT NOTHING ENFORCED. The header ("NOTHING
    here ever deletes anything on B2") and poll_once's own comments assume the
    daemon only ever READS jobs/queue/<IID>/ — the ticket is the operator's, and
    `herdd job cancel`/`retarget` are the only things that may remove one. A box
    that deleted its own ticket would erase the record that the job was ever
    scheduled here, and `job requeue` (which re-mints the SAME key) would have
    nothing to re-open. Nothing checked it; this does.

    Covers a full lifecycle — claim, run, done, plus a second pass over the same
    ticket (the terminal-skip path) — and carries a POSITIVE CONTROL, because a
    detector that cannot see a queue delete would pass this test vacuously."""
    bucket, shimdir = _make_bucket(tmp_path)
    opdir, logf = _oplog_shimdir(tmp_path, shimdir)
    iid = 90087
    job_id, _ = _stage_job(tmp_path, bucket, iid)
    ticket = bucket / "jobs" / "queue" / str(iid) / f"{job_id}.json"

    for _ in range(2):     # pass 1 runs it; pass 2 takes the terminal-skip path
        r = _run_jobd(tmp_path, bucket, opdir, iid)
        assert r.returncode == 0, r.stderr
    assert jm.fold_events(_events(bucket, job_id),
                          live_iids={str(iid)})["status"] == "done"

    assert logf.is_file() and logf.read_text().strip(), "the op log recorded nothing"
    assert _queue_deletes(logf) == [], \
        f"poll_once issued a destructive op against jobs/queue/: {_queue_deletes(logf)}"
    assert ticket.is_file(), "the queue ticket was removed by the daemon"

    # POSITIVE CONTROL: the detector must actually be able to see one. (Without
    # this, a typo in _DESTRUCTIVE_OPS or a shim that logged nothing would make
    # the assertion above pass on a daemon that deleted every ticket it saw.)
    env = _hermetic_env(tmp_path, FAKE_BUCKET=str(bucket))
    subprocess.run([str(opdir / "rclone"), "deletefile",
                    f"b2:testbucket/jobs/queue/{iid}/{job_id}.json"],
                   env=env, capture_output=True, timeout=30)
    assert _queue_deletes(logf), \
        "the queue-delete detector is blind — the assertion above proved nothing"
    assert not ticket.is_file()
