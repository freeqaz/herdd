"""joblocal.py — the LOCAL execution lane for jobs-v2 bundles (LOCAL_GPU_LANE.md).

Owner directive (2026-07-30): *"we should be standardizing local training to
_also_ leverage our training run config setup so that it is agnostic to local vs
remote GPU infrastructure."*

THE IDEA IN ONE PARAGRAPH. A jobs-v2 bundle used to be executable only on a
rented box, so local training meant hand-rolled `torchrun` scripts in gitignored
`out/` dirs — which is how the repair-lifter adapters sat at `--max-seq 4096` for
months where no config governed them. Both executors of the job system reach B2
through exactly one seam each (`jobmeta._default_runner` shells out to `rclone`;
`onstart/jobd.sh` likewise), and `testlib/rclone_shim.sh` already implements a
local-dir bucket for both. So the local lane is NOT a second code path: it is the
SAME laptop CLI and the SAME daemon pointed at a local-dir bucket by three env
vars. Identity of behavior is by construction, not by parallel maintenance.

This module owns only the local lane's own facts — where the bucket lives, what
the box is called, which env switches an executor onto it, how a local asset
override is seeded, and how local box liveness is probed. Everything else is the
existing job machinery, unmodified.

PURITY: every function here is either pure or touches ONLY the local root (never
the network, never the vast API, never a credential). That is the lane's safety
property: no flag, no local bucket; `--local`, no B2.
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM_SRC = os.path.join(HERE, "testlib", "rclone_shim.sh")
JOBD_SH = os.path.join(HERE, "onstart", "jobd.sh")

#: `B2_BUCKET` value for the local lane. The shim maps `b2:<bucket>/<key>` ->
#: `$FAKE_BUCKET/<key>`, so the name is cosmetic — but it must be SOMETHING, and
#: a distinctive one makes a stray local path obvious in a log line.
LOCAL_BUCKET_NAME = "local"

#: Default local root. Deliberately OUTSIDE the repo: nothing to .gitignore, and
#: no absolute machine path can ever be committed.
DEFAULT_HOME = "~/.cache/upstream-monorepo/joblocal"

_SANITIZE_RE = re.compile(r"[^a-z0-9-]+")


class JoblocalError(Exception):
    pass


# --------------------------------------------------------------------------- #
# identity + paths (pure)
# --------------------------------------------------------------------------- #
def local_home() -> str:
    """Root of the local lane ($JOBLOCAL_HOME, else ~/.cache/upstream-monorepo/joblocal)."""
    return os.path.abspath(os.path.expanduser(
        os.environ.get("JOBLOCAL_HOME") or DEFAULT_HOME))


def local_box_id(host: str | None = None) -> str:
    """`local-<hostname>` — the queue prefix / `box=` field of a local job.

    The `local-` prefix is belt-and-braces: the REAL separation is that the local
    bucket is a different filesystem root, so a local job physically cannot show
    up in a remote `job ls` (and vice versa). The prefix just makes it obvious in
    output. Sanitized to [a-z0-9-] because it is used raw as a path segment."""
    h = host if host is not None else _hostname()
    h = _SANITIZE_RE.sub("-", h.strip().lower()).strip("-")
    return f"local-{h or 'host'}"


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:                                    # pragma: no cover
        return "host"


def bucket_dir(home: str | None = None) -> str:
    return os.path.join(home or local_home(), "bucket")


def workspace_dir(home: str | None = None) -> str:
    """Local stand-in for `/workspace` — jobd's `$JOBD_ROOT`."""
    return os.path.join(home or local_home(), "workspace")


def bin_dir(home: str | None = None) -> str:
    return os.path.join(home or local_home(), "bin")


def asset_map_path(home: str | None = None) -> str:
    return os.path.join(home or local_home(), "assets.map")


def lock_path(root: str) -> str:
    """jobd's single-instance flock, which doubles as the liveness probe."""
    return os.path.join(root, ".jobd.lock")


# --------------------------------------------------------------------------- #
# transport activation
# --------------------------------------------------------------------------- #
def install_shim(home: str | None = None) -> str:
    """Install `testlib/rclone_shim.sh` as `<home>/bin/rclone`; return the path.

    Copied (not symlinked) so the installed name is a plain executable, and
    refreshed on every activation so an edit to the shared shim can never leave a
    stale local copy behind — the lane's correctness argument is that laptop side
    and box side run the SAME transport."""
    d = bin_dir(home)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "rclone")
    shutil.copyfile(SHIM_SRC, dst)
    os.chmod(dst, 0o755)
    return dst


def transport_env(home: str | None = None, base_env: dict | None = None) -> dict:
    """The three variables that point an executor at the local bucket.

    Returns a NEW dict (base_env or os.environ, copied). `PATH` is PREPENDED so
    the shim wins over any real rclone."""
    h = home or local_home()
    env = dict(os.environ if base_env is None else base_env)
    env["PATH"] = bin_dir(h) + os.pathsep + env.get("PATH", "")
    env["FAKE_BUCKET"] = bucket_dir(h)
    env["B2_BUCKET"] = LOCAL_BUCKET_NAME
    # A stray real-B2 credential in the environment must not be able to make the
    # shim (or anything downstream) talk to B2. It cannot — the shim has no
    # network code — but dropping them makes the "never touches a credential"
    # property visible rather than merely true.
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_WRITE_KEY_ID",
              "B2_WRITE_APPLICATION_KEY", "B2_S3_ENDPOINT", "B2_REGION"):
        env.pop(k, None)
    return env


def ensure_layout(home: str | None = None) -> str:
    """Create the local root (bucket/, workspace/, bin/rclone) and return it."""
    h = home or local_home()
    for d in (bucket_dir(h), workspace_dir(h), bin_dir(h)):
        os.makedirs(d, exist_ok=True)
    install_shim(h)
    return h


def activate(home: str | None = None) -> str:
    """Switch THIS PROCESS onto the local bucket, and return the local root.

    Mutates os.environ, so every later `jobmeta`/`runmeta` call in the process
    (whose `_default_runner` shells out to `rclone` on PATH) transparently reads
    and writes the local bucket. This is the whole `--local` implementation."""
    h = ensure_layout(home)
    os.environ.update({k: v for k, v in transport_env(h).items()
                       if k in ("PATH", "FAKE_BUCKET", "B2_BUCKET")})
    for k in ("B2_KEY_ID", "B2_APPLICATION_KEY", "B2_WRITE_KEY_ID",
              "B2_WRITE_APPLICATION_KEY", "B2_S3_ENDPOINT", "B2_REGION"):
        os.environ.pop(k, None)
    return h


# --------------------------------------------------------------------------- #
# liveness — the local answer to `_live_iids_set()`
# --------------------------------------------------------------------------- #
def daemon_running(root: str) -> bool:
    """Is a local jobd holding $JOBD_ROOT/.jobd.lock right now?

    jobd takes an exclusive flock on that file for its whole lifetime (its
    one-daemon-per-box guard). Trying to take it non-blocking is therefore a REAL
    liveness probe, not a stub: ^C the runner mid-job and `job status --local`
    correctly folds the job to `interrupted`, exactly as a dead box does."""
    p = lock_path(root)
    if not os.path.exists(p):
        return False
    import fcntl
    try:
        fd = os.open(p, os.O_RDWR)
    except OSError:                                      # pragma: no cover
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True                                  # someone holds it
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def live_boxes(home: str | None = None) -> set:
    """The live-box set to inject into a local job fold (`{local-<host>}` or empty)."""
    h = home or local_home()
    return {local_box_id()} if daemon_running(workspace_dir(h)) else set()


# --------------------------------------------------------------------------- #
# local asset overrides
# --------------------------------------------------------------------------- #
def parse_asset_arg(spec: str) -> tuple[str, str]:
    """`NAME=DIR` -> (name, abspath). Raises JoblocalError on a bad spec."""
    if "=" not in spec:
        raise JoblocalError(f"--asset needs NAME=DIR (got {spec!r})")
    name, _, path = spec.partition("=")
    name = name.strip()
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not name:
        raise JoblocalError(f"--asset needs a NAME (got {spec!r})")
    if not os.path.isdir(path):
        raise JoblocalError(f"--asset {name}: not a directory: {path}")
    return name, path


def load_asset_map(home: str | None = None) -> dict:
    """Persisted `name<TAB>dir` overrides. Unreadable/garbage lines are skipped —
    this is a convenience cache, never a source of truth."""
    out = {}
    try:
        with open(asset_map_path(home), encoding="utf-8") as fh:
            for line in fh:
                name, tab, path = line.rstrip("\n").partition("\t")
                if tab and name.strip() and path.strip():
                    out[name.strip()] = path.strip()
    except OSError:
        pass
    return out


def save_asset_map(mapping: dict, home: str | None = None) -> str:
    """Write the override map atomically (the local root is shared with a running
    daemon; a half-written map read by the next invocation is a silent wrong path)."""
    p = asset_map_path(home)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for name in sorted(mapping):
            fh.write(f"{name}\t{mapping[name]}\n")
    os.replace(tmp, p)
    return p


def seed_asset(root: str, name: str, src_dir: str) -> str:
    """Point jobd's asset cache at an existing local directory, with NO copy.

    Writes `$root/assets/<name>` as a symlink to `src_dir` plus the marker
    `$root/assets/.<name>.local`, which jobd reads as "this cache is
    operator-provided; do not pull over it" (see jobd.sh `_stage_one_asset_body`).
    `require:` postconditions and the `dest` symlink still run normally, so a
    wrong local path is still caught exactly where a truncated B2 pull would be.

    Returns the cache path."""
    src = os.path.abspath(os.path.expanduser(src_dir))
    if not os.path.isdir(src):
        raise JoblocalError(f"asset {name!r}: not a directory: {src}")
    assets = os.path.join(root, "assets")
    os.makedirs(assets, exist_ok=True)
    cache = os.path.join(assets, name)
    if os.path.islink(cache):
        os.unlink(cache)
    elif os.path.isdir(cache):
        raise JoblocalError(
            f"asset {name!r}: {cache} is a real directory (a previous PULL landed "
            f"there). Remove it before overriding with a local path.")
    os.symlink(src, cache)
    # The marker is what jobd keys on; a bare symlink is not enough, because a
    # `.complete` byte-total check against a symlinked cache reads 0 bytes and
    # would trigger a re-pull straight THROUGH the link into the source tree.
    with open(os.path.join(assets, f".{name}.local"), "w", encoding="utf-8") as fh:
        fh.write(src + "\n")
    return cache


# --------------------------------------------------------------------------- #
# GPUs
# --------------------------------------------------------------------------- #
def probe_gpus() -> list:
    """[(index, total_MiB, name), …] from nvidia-smi; [] when there is no GPU.

    Honors jobd's own `JOBD_FAKE_GPUS="0:24,1:24"` test hook so the CLI preflight
    and the daemon agree about the inventory — otherwise a fake-GPU test would
    preflight against the real machine (and be unrunnable on a GPU-less host)."""
    fake = os.environ.get("JOBD_FAKE_GPUS")
    if fake:
        out = []
        for tok in fake.split(","):
            idx, _, gb = tok.strip().partition(":")
            if idx.isdigit():
                out.append((int(idx), int(gb or 0) * 1024, "fake-gpu"))
        return out
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    out = []
    for line in (p.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out.append((int(parts[0]), int(parts[1]),
                        parts[2] if len(parts) > 2 else "?"))
    return out


def foreign_gpu_procs(allow: list | None = None) -> list:
    """[(pid, gpu_index, name), …] — compute processes NOT started by us.

    Why this exists: jobd's `reap_orphan_gpu_procs` KILLS every compute process
    whose ppid is 1 when the daemon boots with nothing adopted. On a rented box
    that is right (the only such procs are wedged contexts from the previous
    boot). On a workstation ppid==1 is the NORMAL state of any nohup'd / setsid'd
    / systemd-started training run, so the same heuristic would kill the
    operator's own work. The local lane therefore forces `JOBD_GPU_REAP=0` AND
    refuses to start when a card it was asked to use is already busy.

    Returns [] under `JOBD_FAKE_GPUS`, matching jobd's own rule that the reaper
    never touches GPUs when the inventory is faked."""
    if os.environ.get("JOBD_FAKE_GPUS"):
        return []
    try:
        p = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,gpu_uuid,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if p.returncode != 0:
        return []
    uuid_to_idx = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        for line in (q.stdout or "").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                uuid_to_idx[parts[1]] = int(parts[0])
    except (OSError, subprocess.SubprocessError):        # pragma: no cover
        pass
    out = []
    for line in (p.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        idx = uuid_to_idx.get(parts[1])
        if allow is not None and idx is not None and idx not in allow:
            continue
        out.append((int(parts[0]), idx, parts[2]))
    return out


def parse_gpu_allow(spec: str | None) -> list:
    """`"0,1"` -> [0, 1]; None/empty -> [] (meaning "every card jobd probes")."""
    if not spec:
        return []
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise JoblocalError(f"--gpus takes device indices, got {tok!r}")
        out.append(int(tok))
    return out


# --------------------------------------------------------------------------- #
# the daemon environment
# --------------------------------------------------------------------------- #
def jobd_env(home: str | None = None, *, root: str | None = None,
             gpu_allow: list | None = None, once: bool = True,
             cpu_slots: int = 2, python: str | None = None,
             base_env: dict | None = None) -> dict:
    """Full environment for a LOCAL `onstart/jobd.sh` run.

    Everything here is either the local transport, a box-concept that has no
    local meaning, or a real safety guard. Each is spelled out in
    LOCAL_GPU_LANE.md's differences table; keep them in step."""
    h = home or local_home()
    env = transport_env(h, base_env=base_env)
    env["JOBD_IID"] = local_box_id()
    env["JOBD_ROOT"] = root or workspace_dir(h)
    # the rclone remote is the shim: there is no `b2_sync.sh config` to run.
    env["JOBD_SKIP_B2CONFIG"] = "1"
    # NO BOX TO PARK. Self-park PUTs vast's instance API with a scoped key; there
    # is neither a box nor a key, and "park the operator's workstation" is not a
    # thing we ever want to almost-do.
    env["JOBD_IDLE_PARK"] = "0"
    # NO ZOMBIE REAP. See foreign_gpu_procs() — the ppid==1 heuristic is correct
    # on a fresh box and catastrophic on a workstation.
    env["JOBD_GPU_REAP"] = "0"
    # No cred broker locally (also implied by an absent BOX_IDENTITY_NONCE, but
    # an inherited one from a shell would arm a pointless refresh loop).
    env.pop("BOX_IDENTITY_NONCE", None)
    env.pop("B2_KEY_EXPIRES_AT", None)
    env["JOBD_CPU_SLOTS"] = str(int(cpu_slots))
    if once:
        # one claim pass, then DRAIN the running jobs and exit — the right shape
        # for "run this bundle", and re-running is exactly jobd's resume path.
        env["JOBD_ONCE"] = "1"
    else:
        env.pop("JOBD_ONCE", None)
    if gpu_allow:
        env["JOBD_GPU_ALLOW"] = ",".join(str(i) for i in gpu_allow)
    if python:
        env["JOBD_PYTHON"] = python
    return env


def differences_banner() -> str:
    """The honest local-vs-remote delta, printed on every run. A local PASS must
    never be mistaken for a box-certified one."""
    return (
        ">> LOCAL LANE — same bundle, same jobd, same config; NOT the same as a box:\n"
        ">>   no spot preemption · no self-park · no cred rotation · no GPU-zombie reap\n"
        ">>   results stay on LOCAL disk (never pushed to B2)\n"
        ">>   entrypoint runs in YOUR host env, not the baked image — "
        "`rehearse.sh --image` is still the env gate before you rent\n"
        ">>   (details: tools/vast/LOCAL_GPU_LANE.md)")
