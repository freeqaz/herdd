#!/usr/bin/env python3
"""local_smoke.py — run a jobs-v2 bundle's REAL entrypoint on the LOCAL GPUs, $0.

WHAT THIS IS FOR. Between `rehearse.sh` (CPU-only, fake bucket, proves jobd
plumbing) and renting a box (real money) there was nothing: no way to answer
"does this training config come up AT ALL — does the trainer start, does loss
move, do the post-train gates pass?" without spending. This is that step. It
runs the bundle's own `run.sh` inside the pinned image on the local cards, with
a SMOKE CONFIG that overrides the of-record env (8-bit instead of bf16, a short
window instead of 32k, ten steps instead of an epoch) while leaving everything
else — the data-identity gate, the launch planner, the trainer, the gates —
exactly as the box would run it.

WHAT IT CERTIFIES: the entrypoint AND the live train env, on the local GPU
architecture. WHAT IT DOES NOT: the of-record shape (VRAM tail, s/it at the
real sequence length), the of-record numerics (bf16 — a smoke runs quantized to
fit), the B2 branches (assets are local, publish is forced off), jobd's own
plumbing (bypassed entirely — that is rehearse.sh's and run-local's job), any
other GPU architecture, and anything at all about training QUALITY. Ten steps of
loss is machinery evidence and nothing else. Full runbook, including what to do
when a stage fails: tools/vast/LOCAL_SMOKE_RUNBOOK.md.

WHY A THIRD $0 LANE. rehearse.sh runs the image but is CPU-only by
construction; `herdd job run-local` (LOCAL_GPU_LANE.md) runs real GPUs but in
YOUR HOST env, and its own differences table says so — "a passing local run does
not certify the remote run's environment… rehearse.sh --image remains the env
gate", which is CPU-only. So image-and-GPU-together — where liger/triton,
flash-attn, bitsandbytes and sm-arch failures actually live — was covered by
neither. That hole is the whole justification. The tidier end state is
`run-local --image` with this file's smoke-config merge folded in; that touches
a tested of-record executor, so it is written down in the runbook rather than
done in passing.

    tools/vast/local_smoke.py tools/witness/jobs/v7-longctx-train
    tools/vast/local_smoke.py <job> --dry-run          # print the podman argv
    tools/vast/local_smoke.py <job> --env MAX_SEQ=8192 --gpus 1

THE SMOKE CONFIG IS A SEPARATE FILE, deliberately. `<job>/smoke.yaml` holds the
overrides; `job-config.yaml` stays the box's config and is never edited for a
local run. Two reasons it is not a `smoke:` block inside job-config.yaml:
(1) that file is the SUBMIT surface — jobmeta's no-PyYAML fallback parser
handles exactly one level of nesting, so a `smoke.env.*` two-level map would
mis-parse on a box that lacks PyYAML, and a mis-parse there is a spend-time
failure; (2) a separate file can carry things the submit schema has no business
knowing (which local dir stands in for which B2 asset, how many cards to use).

Precedence, lowest to highest:
    job-config.yaml `env:`  ->  smoke.yaml `env:`  ->  --env K=V
So a smoke inherits the REAL config — including EXPECT_SHA256 and the rest of
the fail-closed identity gate — and overrides only what it names. That is the
point: a smoke that skipped the gates would prove less than nothing.

GPU PASSTHROUGH — two paths, auto-selected.

  cdi (PREFERRED, available since nvidia-container-toolkit was installed here
  2026-08-05): `--device nvidia.com/gpu=<i>`. The toolkit's generated spec
  injects the driver libs, the device nodes AND the userspace binaries, and its
  createContainer hook updates the ld cache for us. Verified in-container:
  `ctypes.CDLL("libcuda.so")` resolves — the bare SONAME-less lookup that
  triton does — and `nvidia-smi` works, which the manual path never gave us
  (run.sh calls it with `|| true`, so it was silently printing nothing).

  manual (FALLBACK, and what the first run of this lane used): mount the host's
  libcuda/libnvidia-ml in, symlink the SONAMEs beside them, add the dir to
  ld.so.conf, and run `ldconfig` IN-CONTAINER. Skip that last step and torch
  imports and even reports cuda_available=True, but every triton JIT (liger!)
  dies "libcuda.so cannot be found" — the footgun recorded in
  n5prime_sft_run/training-infra/119_SPEEDUP_DEFAULTS_LANDED. Kept because
  hosts without the toolkit still exist and this is the only thing that works
  there; `--gpu-mode manual` forces it.

`--shm-size` matters on both paths: podman defaults to 64 MB, which DDP +
dataloader workers exhaust immediately.

GUARDS, because the local cards are small and shared with whatever else is
running: the tool REFUSES to start if the GPUs already hold memory (another
agent's training run — `--force` overrides), forces `PUBLISH=0` so a smoke can
never write to B2, never exports B2_BUCKET (so a missing asset fails loudly
instead of silently pulling 14 GB), and wraps the container in a wall-clock
timeout. Output lands in a real run directory (never /tmp); by default only the
small evidence is kept and the weights are pruned — see --keep-weights.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import jobmeta as jm  # noqa: E402  (same directory; one job-config parser, not two)
import vastconf  # noqa: E402  (the local-GPU switch lives there, not restated here)

# Keep in lockstep with rehearse.sh's DEFAULT_IMAGE and herdd.yaml
# default_image. A stale tag here does not error — it runs the WRONG env and
# reports PASS, which is worse than failing.
# Must equal herdd.yaml default_image — test_rehearse.py pins every copy.
DEFAULT_IMAGE = "registry.example.com/train:latest"
DEFAULT_TIMEOUT_S = 1800
# Where the host driver libs get mounted inside the container. Deliberately NOT
# /usr/lib: overmounting the image's own lib dir would shadow the CUDA runtime
# the image ships.
CTR_LIB_DIR = "/usr/lib/nvhost"
# Small files worth keeping after a smoke; everything else under out/ is
# throwaway weights (a 10-step adapter is ~160 MB and means nothing).
EVIDENCE = ("train.log", "train_summary.json", "artifact-manifest.json",
            "corpus-identity.json", "adapter_config.json", "dataset_manifest.json",
            "PUBLISH_DRYRUN.json",
            # A bench bundle's whole output is its report; pruning it left only
            # the per-cell summaries and made the run's own conclusion
            # re-derivable but not readable.
            "bench_results.json", "bench_report.txt", "gemm_ceiling.json",
            "gemm_ceiling.log", "cell_meta.json")


class SmokeError(Exception):
    """Anything the operator has to fix before a run can be attempted."""


# ---------------------------------------------------------------- host probes

def probe_gpus(smi_csv: "str | None" = None) -> list:
    """[{index, name, ram_gb, used_mb}, ...] from nvidia-smi.

    ram_gb floors MiB->GiB so a 24576 MiB card reads 24, matching how
    `needs.gpu_ram_gb` and autotune's fit rule are written."""
    if smi_csv is None:
        try:
            smi_csv = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError) as e:
            raise SmokeError(f"nvidia-smi unavailable ({e}) — no local GPUs to smoke on")
    out = []
    for line in smi_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            out.append({"index": int(parts[0]), "name": parts[1],
                        "ram_gb": int(int(parts[2]) // 1024), "used_mb": int(parts[3])})
        except ValueError:
            continue
    if not out:
        raise SmokeError("nvidia-smi listed no GPUs")
    return out


def find_driver_libs(ldconfig_out: "str | None" = None) -> dict:
    """{soname_stem: absolute host path} for libcuda + libnvidia-ml.

    Resolved from `ldconfig -p` rather than a hardcoded /usr/lib glob, because
    the driver lands in different places per distro (/usr/lib on Arch,
    /usr/lib/x86_64-linux-gnu on Debian). Falls back to globbing when ldconfig
    is unreadable. Returns the VERSIONED file (libcuda.so.610.43.03), not the
    .so.1 symlink: podman would mount the symlink as a dangling link."""
    want = ("libcuda.so.1", "libnvidia-ml.so.1")
    found: dict = {}
    if ldconfig_out is None:
        try:
            ldconfig_out = subprocess.run(["ldconfig", "-p"], capture_output=True,
                                          text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            ldconfig_out = ""
    for line in ldconfig_out.splitlines():
        # "	libcuda.so.1 (libc6,x86-64) => /usr/lib/libcuda.so.1"
        m = re.match(r"\s*(\S+)\s+\([^)]*\)\s+=>\s+(\S+)\s*$", line)
        if not m or m.group(1) not in want or m.group(1) in found:
            continue
        found[m.group(1)] = m.group(2)
    for soname in want:
        if soname not in found:
            for d in ("/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"):
                hits = sorted(glob.glob(os.path.join(d, soname.rsplit(".1", 1)[0] + ".*")))
                real = [h for h in hits if re.search(r"\.so\.\d+[\d.]*$", h)]
                if real:
                    found[soname] = real[-1]
                    break
    missing = [s for s in want if s not in found]
    if missing:
        raise SmokeError(
            f"host driver libs not found: {', '.join(missing)}. The container needs "
            "the HOST driver's libcuda/libnvidia-ml mounted in (no nvidia CDI here); "
            "install/repair the driver or pass --no-gpu for an argv-shape check.")
    return {k: os.path.realpath(v) for k, v in found.items()}


CDI_SPEC_DIRS = ("/etc/cdi", "/var/run/cdi")


def detect_cdi(spec_dirs: "tuple | None" = None) -> bool:
    """True when a CDI spec declaring nvidia.com/gpu devices is on this host.

    Read from the spec FILES rather than by shelling out to `nvidia-ctk`: podman
    resolves the device name against these same files, so a host where the
    binary exists but the spec was never generated must take the manual path."""
    for d in (spec_dirs or CDI_SPEC_DIRS):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for fn in names:
            if not fn.endswith((".yaml", ".yml", ".json")):
                continue
            try:
                with open(os.path.join(d, fn), errors="replace") as fh:
                    if "nvidia.com/gpu" in fh.read(65536):
                        return True
            except OSError:
                continue
    return False


def cdi_device_args(indices: list) -> list:
    """--device nvidia.com/gpu=<i> per selected card.

    Named per index rather than `=all` so `--gpus 1` genuinely denies the other
    card (verified: the container then sees exactly one device)."""
    args = []
    for i in indices:
        args += ["--device", f"nvidia.com/gpu={i}"]
    return args


def gpu_device_args(indices: list) -> list:
    """MANUAL path --device flags: selected render nodes plus the control nodes.

    Only the SELECTED /dev/nvidia<i> are passed, so `--gpus 1` genuinely denies
    the second card rather than relying on CUDA_VISIBLE_DEVICES (which a
    subprocess can simply reset)."""
    devs = [f"/dev/nvidia{i}" for i in indices]
    for ctl in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
                "/dev/nvidia-modeset"):
        if os.path.exists(ctl):
            devs.append(ctl)
    args = []
    for d in devs:
        if not os.path.exists(d):
            raise SmokeError(f"device node missing: {d}")
        args += ["--device", d]
    return args


def check_width_pin(env: dict) -> None:
    """Mirror a bundle's EXPECT_GPU_COUNT gate BEFORE paying for a container start.

    A bundle may pin its DDP width fail-closed (v7 does: `EXPECT_GPU_COUNT: "2"`,
    run.sh exit 13) so a box's card count cannot become a silent variable in a
    paired A/B. That pin is correct and this tool does not override it — but
    run.sh only carves out `JOB_GPU_COUNT=0` (the CPU rehearsal), so a smoke on
    ONE card (the obvious move when a peer holds the other 3090) would otherwise
    die minutes in, advising the reader to rent more cards. Fail here instead,
    naming the two legitimate resolutions."""
    expect = str(env.get("EXPECT_GPU_COUNT", "")).strip()
    have = str(env.get("JOB_GPU_COUNT", "")).strip()
    if not expect or not have or have == "0" or have == expect:
        return
    raise SmokeError(
        f"this bundle pins EXPECT_GPU_COUNT={expect} but the smoke selected "
        f"{have} card(s) — run.sh would exit 13. Either run on {expect} card(s) "
        f"(--gpus {expect}, and check no peer is holding one), or override "
        f"EXPECT_GPU_COUNT in the bundle's smoke.yaml `env:` if you deliberately "
        f"want a narrower machinery check. Do NOT edit job-config.yaml — that is "
        f"the submit surface, and the pin there is what the A/B depends on.")


# --------------------------------------------------------------- config merge

def load_smoke_config(path: "str | None") -> dict:
    """Parse <job>/smoke.yaml (or an explicit --smoke-config). Absent is OK.

    Uses jobmeta's parser so there is exactly ONE yaml dialect in this tree:
    top-level scalars, one-level nested maps (`env:`, `assets:`), lists."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as fh:
        data = jm._parse_job_yaml(fh.read())
    if not isinstance(data, dict):
        raise SmokeError(f"{path}: smoke config must be a mapping at top level")
    for k in ("env", "assets"):
        if k in data and data[k] is not None and not isinstance(data[k], dict):
            raise SmokeError(f"{path}: `{k}:` must be a map of key: value")
    return data


def merge_env(job_cfg: dict, smoke_cfg: dict, cli_env: list) -> "tuple[dict, dict]":
    """Resolve the container env. Returns (env, provenance{key: source}).

    job-config `env:` (the box's config) < smoke.yaml `env:` < --env K=V. The
    provenance map exists so the run banner can show WHICH knobs a smoke moved —
    an override you did not intend is the likeliest way to get a green smoke
    that means nothing."""
    env: dict = {}
    prov: dict = {}
    for src, block in (("job-config", job_cfg.get("env") or {}),
                       ("smoke.yaml", smoke_cfg.get("env") or {})):
        for k, v in block.items():
            env[str(k)] = "" if v is None else str(v)
            prov[str(k)] = src
    for item in cli_env or []:
        if "=" not in item:
            raise SmokeError(f"--env needs KEY=VALUE (got {item!r})")
        k, v = item.split("=", 1)
        env[k.strip()] = v
        prov[k.strip()] = "--env"
    return env, prov


def check_asset_provenance(job_cfg: dict, assets: dict,
                           repo_root: str = REPO_ROOT) -> list:
    """Compare each staged asset file against the repo file `tracks:` says it mirrors.

    A bundle's `tracks:` map is the provenance declaration `herdd job submit`
    checks its B2 objects against. The LOCAL lane never went through that check:
    it mounts `tools/vast/runsets/<name>/_build`, a staged copy nothing re-stages
    automatically, so the trainer a smoke exercises can drift from the one a box
    pulls — in a direction no gate anywhere was looking. Measured 2026-08-06: the
    witness-lifter `_build` trainer sat 109 lines behind
    tools/pipeline/ml_infra/ while B2 matched it exactly, which is a smoke
    certifying code no box would run.

    Returns [(asset, filename, staged_path, truth_path, status)] where status is
    match | drift | missing-staged | missing-truth. Reporting rather than raising
    keeps the policy decision (refuse vs warn) with the caller."""
    out: list = []
    for asset in (job_cfg.get("assets") or []):
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        tracks = asset.get("tracks")
        if not name or not isinstance(tracks, dict) or name not in assets:
            continue
        staged_dir = assets[name][0]
        for fname, repo_rel in sorted(tracks.items()):
            staged = os.path.join(staged_dir, str(fname))
            truth = os.path.join(repo_root, str(repo_rel))
            if not os.path.isfile(staged):
                status = "missing-staged"
            elif not os.path.isfile(truth):
                status = "missing-truth"
            else:
                status = ("match" if _sha256(staged) == _sha256(truth) else "drift")
            out.append((name, str(fname), staged, truth, status))
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_assets(job_cfg: dict, smoke_cfg: dict, cli_assets: dict,
                   repo_root: str = REPO_ROOT) -> dict:
    """{asset name: (host dir, container dest)} for every asset the job declares.

    Resolution order per asset: --asset NAME=DIR, then smoke.yaml `assets:`,
    then a convention for the two shapes that actually occur:
      base-models/<slug>  ->  ~/base-models/<slug>   (the pinned B2 base, pulled once)
      runsets/<name>      ->  tools/vast/runsets/<name>/_build
    Anything unresolved is an error naming what was tried — silently skipping an
    asset would send run.sh down its B2-pull branch, which is a 14 GB surprise."""
    out: dict = {}
    for asset in (job_cfg.get("assets") or []):
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        dest = str(asset.get("dest") or "")
        b2 = str(asset.get("b2") or "")
        if not name or not dest:
            raise SmokeError(f"asset with no name/dest in job-config: {asset!r}")
        if os.path.isabs(dest):
            raise SmokeError(
                f"asset {name!r} has an absolute dest ({dest}). jobd refuses those too; "
                "the bundle should use a relative dest (see run.sh's asset notes).")
        tried = []
        cand = cli_assets.get(name)
        if not cand:
            cand = (smoke_cfg.get("assets") or {}).get(name)
        if cand:
            cand = os.path.abspath(os.path.expanduser(str(cand)))
            tried.append(cand)
        else:
            for guess in _convention_dirs(b2, repo_root):
                tried.append(guess)
                if os.path.isdir(guess):
                    cand = guess
                    break
        if not cand or not os.path.isdir(cand):
            raise SmokeError(
                f"asset {name!r} (b2:{b2}) has no local stand-in. Tried: "
                + (", ".join(tried) or "<nothing>")
                + f". Pass --asset {name}=<dir>, or add it under `assets:` in smoke.yaml.")
        out[name] = (cand, dest)
    return out


def bootstrap_script(libs: dict, entrypoint: str) -> str:
    """The in-container preamble: SONAME symlinks, ld.so.conf, ldconfig, exec.

    `ldconfig` is the load-bearing line. Skip it and torch still imports and may
    even report cuda_available — but every triton JIT dies looking for
    libcuda.so (doc 119). Written as a here-doc-free single string so it can be
    passed to `bash -lc` verbatim and asserted in tests."""
    if not libs:
        # CDI path: the toolkit's createContainer hook already updated the ld
        # cache, so there is nothing to bootstrap.
        return f"exec bash {entrypoint}"
    lines = ["set -euo pipefail"]
    for soname, host_path in sorted(libs.items()):
        base = os.path.basename(host_path)
        lines.append(f"ln -sf {base} {CTR_LIB_DIR}/{soname}")
        stem = soname.rsplit(".1", 1)[0]          # libcuda.so.1 -> libcuda.so
        lines.append(f"ln -sf {soname} {CTR_LIB_DIR}/{stem}")
    lines += [
        f"echo {CTR_LIB_DIR} > /etc/ld.so.conf.d/nvhost.conf",
        "ldconfig",
        f"exec bash {entrypoint}",
    ]
    return "\n".join(lines)


def podman_argv(*, image: str, workdir: str, job_mounts: list, env: dict,
                devices: list, libs: dict, entrypoint: str, name: str,
                shm: str = "8g") -> list:
    """The full `podman run` argv. Pure, so --dry-run and the tests see it."""
    argv = ["podman", "run", "--rm", "--name", name, "--shm-size", shm]
    argv += devices
    for soname, host_path in sorted(libs.items()):
        argv += ["-v", f"{host_path}:{CTR_LIB_DIR}/{os.path.basename(host_path)}:ro"]
    argv += ["-v", f"{workdir}:/job"]
    for host_dir, ctr_dest in job_mounts:
        argv += ["-v", f"{host_dir}:/job/{ctr_dest.strip('/')}:ro"]
    argv += ["-w", "/job"]
    for k in sorted(env):
        argv += ["-e", f"{k}={env[k]}"]
    argv += [image, "bash", "-lc", bootstrap_script(libs, entrypoint)]
    return argv


def _convention_dirs(b2: str, repo_root: str) -> list:
    b2 = b2.strip("/")
    if b2.startswith("base-models/"):
        return [os.path.expanduser(os.path.join("~/base-models", b2.split("/", 1)[1]))]
    if b2.startswith("runsets/"):
        name = b2.split("/", 1)[1]
        return [os.path.join(repo_root, "tools", "vast", "runsets", name, "_build"),
                os.path.join(repo_root, "tools", "vast", "runsets", name)]
    return []


# ------------------------------------------------------------------- workdir

SKIP_DIRS = {"out", "assets", "results", "__pycache__", ".git", "assets-fixture"}


def stage_workdir(job_dir: str, workdir: str) -> list:
    """Copy the bundle into a writable workdir; return the ro bind-mounts.

    The entrypoint writes out/ into its own cwd, so the bundle folder itself
    must not be the cwd — a smoke would litter the repo with adapters and, worse,
    could leave a DRY_RUN-stamped artifact manifest where a later `submit` would
    tar it up (run.sh purges exactly that case for a reason). data/ is
    bind-mounted read-only instead of copied: the corpora are tens of MB and the
    identity gate only reads them.

    Declared `includes:` are overlaid after the copy, the same way
    `jobmeta.materialize_bundle` does it for the tar. This lane COPIES rather
    than tars, so it does not inherit that path — and a smoke that ran without
    the shared files would be testing a bundle no box will ever receive."""
    os.makedirs(workdir, exist_ok=True)
    mounts = []
    for entry in sorted(os.listdir(job_dir)):
        src = os.path.join(job_dir, entry)
        if entry in SKIP_DIRS or entry.endswith(".pyc"):
            continue
        if entry == "data" and os.path.isdir(src):
            mounts.append((os.path.abspath(src), "data"))
            continue
        dst = os.path.join(workdir, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    for name, inc_src in sorted(jm.resolve_includes(job_dir).items()):
        dst = os.path.join(workdir, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(inc_src, dst)
    os.makedirs(os.path.join(workdir, "out"), exist_ok=True)
    return mounts


def default_runs_root() -> str:
    """Run output goes to a REAL directory, never /tmp (owner ruling 2026-08-05).

    Prefers the canonical resolver so a smoke's evidence lands beside every other
    run's; falls back to ~/upstream-smoke-runs when upstream-bench is not present."""
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "grind"))
        import bench_paths  # type: ignore
        return os.path.join(bench_paths.runs_dir(create=True), "local-smoke")
    except Exception:
        return os.path.expanduser("~/upstream-smoke-runs")


def prune_weights(out_dir: str) -> int:
    """Delete the throwaway weights, keep the evidence. Returns bytes freed.

    A ten-step adapter is ~160 MB and says nothing; its train.log and
    train_summary.json are the whole point of having run it."""
    freed = 0
    if not os.path.isdir(out_dir):
        return 0
    for root, dirs, files in os.walk(out_dir, topdown=False):
        for f in files:
            if f in EVIDENCE:
                continue
            p = os.path.join(root, f)
            try:
                freed += os.path.getsize(p)
                os.remove(p)
            except OSError:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass
    return freed


# ---------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local_smoke.py",
        description="Run a jobs-v2 bundle's real entrypoint on the local GPUs ($0).")
    p.add_argument("job_dir", help="bundle folder (the one holding job-config.yaml)")
    p.add_argument("--smoke-config", default=None,
                   help="override the default <job>/smoke.yaml")
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="highest-precedence env override; repeatable")
    p.add_argument("--asset", action="append", default=[], metavar="NAME=DIR",
                   help="local dir standing in for a B2 asset; repeatable")
    p.add_argument("--gpus", type=int, default=None,
                   help="how many local cards to use (default: all)")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--workdir", default=None,
                   help="where the run writes (default: a stamped dir under the runs root)")
    p.add_argument("--timeout-s", type=int, default=None,
                   help=f"wall-clock ceiling (default: smoke.yaml timeout_s, else {DEFAULT_TIMEOUT_S})")
    p.add_argument("--shm-size", default="8g")
    p.add_argument("--keep-weights", action="store_true",
                   help="keep out/ weights + checkpoints (default: prune, keep evidence)")
    p.add_argument("--force", action="store_true",
                   help="run even if the GPUs are already busy")
    p.add_argument("--allow-stale-runset", action="store_true",
                   help="run even though a staged asset differs from the repo file "
                        "its `tracks:` names (mirrors `job submit --allow-stale-assets`)")
    p.add_argument("--no-gpu", action="store_true",
                   help="skip GPU passthrough (argv/plumbing check only; training will fail)")
    p.add_argument("--gpu-mode", choices=("auto", "cdi", "manual"), default="auto",
                   help="auto (default) = CDI when a spec is present, else the manual "
                        "lib-mount + in-container ldconfig recipe")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved env + podman argv, run nothing. NOTE: this is "
                        "the TOOL's dry run; the container always gets DRY_RUN=0 so run.sh "
                        "takes its real lane.")
    return p


def _kv_list(items: list, flag: str) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SmokeError(f"{flag} needs NAME=VALUE (got {it!r})")
        k, v = it.split("=", 1)
        out[k.strip()] = v
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Owner ruling 2026-08-06: the local GPU is off limits unless authorized.
    # ONE switch, in vastconf — do not restate the policy here. --dry-run only
    # prints the podman argv and touches no card, so it stays open: refusing to
    # show someone what WOULD run helps nobody.
    if not getattr(args, "dry_run", False):
        vastconf.require_local_gpu("local_smoke.py")
    job_dir = os.path.abspath(args.job_dir)
    if not os.path.isdir(job_dir):
        raise SmokeError(f"not a directory: {job_dir}")

    # `--env` folds PRE-validation, the same slot `job submit` uses: a `${VAR}`
    # asset prefix is unresolvable without it. `merge_env` re-applies the CLI
    # layer last, so the banner's provenance still reads `--env` for these keys.
    _raw = jm.load_job_config(job_dir)
    if args.env:
        _e = dict(_raw.get("env") or {})
        for _kv in args.env:
            if "=" not in _kv:
                raise SmokeError(f"--env needs KEY=VALUE (got {_kv!r})")
            _k, _v = _kv.split("=", 1)
            _e[_k.strip()] = _v
        _raw["env"] = _e
    job_cfg, warns = jm.validate_job_config(_raw, job_dir)
    for w in warns:
        print(f"warn: {w}", file=sys.stderr)
    entrypoint = str(job_cfg.get("entrypoint") or "run.sh")
    smoke_path = args.smoke_config or os.path.join(job_dir, "smoke.yaml")
    smoke_cfg = load_smoke_config(smoke_path)
    if not smoke_cfg:
        print(f">> no smoke config at {smoke_path} — running the OF-RECORD env "
              "unchanged. That is usually the box's shape (bf16, long context) and "
              "will not fit a small card; add a smoke.yaml or pass --env.",
              file=sys.stderr)

    env, prov = merge_env(job_cfg, smoke_cfg, args.env)
    assets = resolve_assets(job_cfg, smoke_cfg, _kv_list(args.asset, "--asset"))

    # Provenance: is the staged copy we are about to mount the same code a box
    # would pull? Runs in --dry-run too, deliberately — it needs no GPU, and
    # while the local lane is gated that is the only mode anyone can reach.
    prov_rows = check_asset_provenance(job_cfg, assets)
    drift = [r for r in prov_rows if r[4] == "drift"]
    for name, fname, staged, truth, status in prov_rows:
        if status == "missing-truth":
            print(f">> provenance SKIPPED: {name}/{fname} — tracks: names "
                  f"{truth}, which does not exist here", file=sys.stderr)
        elif status == "missing-staged":
            print(f">> provenance SKIPPED: {name}/{fname} not in the staged dir",
                  file=sys.stderr)
    if drift and not args.allow_stale_runset:
        lines = "\n".join(
            f"    {n}/{f}\n      staged: {s}\n      truth : {t}"
            for n, f, s, t, _ in drift)
        raise SmokeError(
            "staged asset(s) differ from the repo file `tracks:` says they mirror, "
            "so this smoke would exercise code a box will not run:\n" + lines +
            "\n  Re-stage (tools/vast/runsets/<name>/build.sh), or pass "
            "--allow-stale-runset to test the staged bytes on purpose.")
    if drift:
        print(f">> provenance OVERRIDDEN: {len(drift)} staged file(s) differ from "
              "the repo source of truth; testing the staged bytes on purpose",
              file=sys.stderr)
    elif prov_rows:
        ok = sum(1 for r in prov_rows if r[4] == "match")
        print(f">> provenance OK: {ok}/{len(prov_rows)} staged file(s) match the "
              "repo source of truth", file=sys.stderr)

    # Guards. PUBLISH is forced off unless the operator says otherwise IN THE
    # SAME BREATH; B2_BUCKET is never exported, so an unresolved asset dies on
    # run.sh's own error instead of quietly pulling gigabytes.
    if env.get("PUBLISH", "0") != "0":
        print("!! PUBLISH is not 0 — a smoke must never write to b2:checkpoints/. "
              "Forcing PUBLISH=0.", file=sys.stderr)
    env["PUBLISH"] = "0" if "PUBLISH" not in _kv_list(args.env, "--env") else env["PUBLISH"]
    env.pop("B2_BUCKET", None)
    env["DRY_RUN"] = env.get("DRY_RUN", "0")

    gpus = []
    libs: dict = {}
    gpu_mode = "none"
    if not args.no_gpu:
        gpu_mode = args.gpu_mode
        if gpu_mode == "auto":
            gpu_mode = "cdi" if detect_cdi() else "manual"
        if gpu_mode == "cdi" and not detect_cdi():
            raise SmokeError(
                "--gpu-mode cdi but no CDI spec declaring nvidia.com/gpu was found in "
                + " or ".join(CDI_SPEC_DIRS)
                + ". Generate one (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) "
                  "or use --gpu-mode manual.")
        gpus = probe_gpus()
        if args.gpus is not None:
            if args.gpus < 1 or args.gpus > len(gpus):
                raise SmokeError(f"--gpus {args.gpus} but {len(gpus)} card(s) present")
            gpus = gpus[:args.gpus]
        busy = [g for g in gpus if g["used_mb"] > 1024]
        if busy and not args.force:
            raise SmokeError(
                "GPU(s) already in use: "
                + ", ".join(f"#{g['index']} {g['name']} {g['used_mb']} MiB" for g in busy)
                + ". Another agent may be training — these cards are shared. "
                  "Wait, use --gpus to take only the idle ones, or --force.")
        if gpu_mode == "manual":
            libs = find_driver_libs()
        env.setdefault("JOB_GPU_COUNT", str(len(gpus)))
        env.setdefault("JOB_GPU_RAM_GB", str(min(g["ram_gb"] for g in gpus)))
        check_width_pin(env)
    env.setdefault("CPU_CORES", str(os.cpu_count() or 1))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = os.path.basename(job_dir.rstrip("/")) or "job"
    workdir = os.path.abspath(args.workdir or
                              os.path.join(default_runs_root(), f"{slug}-{stamp}"))
    timeout_s = int(args.timeout_s or smoke_cfg.get("timeout_s") or DEFAULT_TIMEOUT_S)
    name = f"smoke-{slug}-{stamp}"[:60]

    idx = [g["index"] for g in gpus]
    devices = (cdi_device_args(idx) if gpu_mode == "cdi"
               else gpu_device_args(idx) if gpus else [])
    mounts = [(host, dest) for host, dest in assets.values()]
    data_dir = os.path.join(job_dir, "data")
    preview_data = [(data_dir, "data")] if os.path.isdir(data_dir) else []
    argv_preview = podman_argv(image=args.image, workdir=workdir,
                               job_mounts=mounts + preview_data,
                               env=env, devices=devices, libs=libs,
                               entrypoint=entrypoint, name=name, shm=args.shm_size)

    overridden = sorted(k for k, v in prov.items() if v != "job-config")
    print(f"== LOCAL SMOKE: {slug} ==")
    print(f"   image     {args.image}")
    print(f"   gpus      " + (", ".join(f"#{g['index']} {g['name']} {g['ram_gb']}GB"
                                        for g in gpus) if gpus else "NONE (--no-gpu)")
          + f"   [passthrough: {gpu_mode}]")
    print(f"   workdir   {workdir}")
    print(f"   timeout   {timeout_s}s")
    for n, (host, dest) in sorted(assets.items()):
        print(f"   asset     {n} -> {host} (at /job/{dest})")
    print(f"   overrides {', '.join(f'{k}={env[k]}' for k in overridden) or '<none>'}")

    if args.keep_weights and args.workdir is None:
        print("!! --keep-weights with the default workdir writes ~0.5 GB into the "
              "upstream-bench archive, which tracks its bytes in git. Pass --workdir "
              "somewhere outside it if you mean to keep the weights.", file=sys.stderr)

    if args.dry_run:
        print("\n-- podman argv --")
        print(" ".join(_shquote(a) for a in argv_preview))
        return 0

    data_mounts = stage_workdir(job_dir, workdir)
    run_argv = podman_argv(image=args.image, workdir=workdir,
                           job_mounts=mounts + data_mounts, env=env, devices=devices,
                           libs=libs, entrypoint=entrypoint, name=name,
                           shm=args.shm_size)
    log_path = os.path.join(workdir, "smoke.log")
    print(f"\n>> running (log: {log_path})\n")
    t0 = time.time()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(run_argv, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        try:
            for chunk in iter(lambda: proc.stdout.read(4096), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
            rc = proc.wait(timeout=max(1, timeout_s - int(time.time() - t0)))
        except subprocess.TimeoutExpired:
            subprocess.run(["podman", "kill", name], capture_output=True)
            proc.wait()
            print(f"\n!! TIMEOUT after {timeout_s}s — container killed", file=sys.stderr)
            rc = 124
    return report(workdir, rc, time.time() - t0, keep_weights=args.keep_weights)


def _shquote(s: str) -> str:
    return s if re.fullmatch(r"[\w@%+=:,./-]+", s or "") else "'" + s.replace("'", "'\\''") + "'"


def report(workdir: str, rc: int, secs: float, *, keep_weights: bool) -> int:
    """Print the verdict and the numbers worth reading, then prune."""
    out = os.path.join(workdir, "out")
    summary_path = os.path.join(out, "train_summary.json")
    print("\n" + "=" * 60)
    print(f"{'SMOKE PASSED' if rc == 0 else f'SMOKE FAILED (rc={rc})'}  in {secs:.0f}s")
    s = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            pass
    if s:
        vram = s.get("peak_vram_reserved_gb_per_gpu") or s.get("peak_vram_reserved_gb")
        print(f"  steps       {s.get('global_steps')} @ {s.get('step_time_seconds')}s/step"
              f"  (world_size {s.get('world_size')}, eff_batch {s.get('eff_batch')})")
        print(f"  loss        first {s.get('loss_first')} -> last {s.get('loss_last')}"
              f"  min {s.get('loss_min')}  learned={s.get('loss_learned')}")
        print(f"  shape       quant={s.get('quant_mode')} max_seq={s.get('max_seq')}"
              f" r={s.get('lora_r')} alpha={s.get('lora_alpha')} gc={s.get('grad_checkpointing')}")
        print(f"  peak VRAM   {vram} GB reserved")
    log = os.path.join(out, "train.log")
    if os.path.isfile(log):
        try:
            with open(log, errors="replace") as fh:
                n = sum(1 for ln in fh if "Mismatch between tokenized prompt" in ln)
            # Corpus health, not a smoke property: TRL emits one line per row
            # whose prompt/completion boundary is not token-aligned, and those
            # rows silently lose their first completion characters from the loss
            # (D_TRAIN_FULL_RESULT.md SS1b uses this count as the instrument).
            print(f"  boundary    {n} TRL prompt/completion tokenizer-mismatch row(s)"
                  + ("" if n == 0 else "  <- corpus defect, see the runbook"))
        except OSError:
            pass
    if not keep_weights:
        freed = prune_weights(out)
        if freed:
            print(f"  pruned      {freed / 1e6:.0f} MB of throwaway weights "
                  "(--keep-weights to retain)")
    print(f"  evidence    {out}")
    print("=" * 60)
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeError as e:
        print(f"!! {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
