"""vastlib.jobs.runlocal — run a jobs-v2 bundle on THIS machine's GPUs.

Why this exists
---------------
`job run-local` is a TRANSPORT switch, not a second code path: it points
`jobmeta`'s injectable rclone runner at `testlib/rclone_shim.sh` over a local
bucket directory, and then runs the shipped `onstart/jobd.sh` against it. The
bundle, the config validation, the asset staging, the results globs and the
checkpoint-resume semantics are the ones a rented box gets — which is the whole
value: a plumbing bug found here is a box never rented.

`_JOB_LOCAL` — the mutable global, and why it lives HERE
--------------------------------------------------------
Exactly two places in the jobs lane reach for the vast API rather than for B2:
the liveness injection (`view._live_iids_set`) and the presence read
(`view._present_iids_set`). `_JOB_LOCAL` is the flag that gives both of them
their local answer, and it is flipped at RUNTIME by `_job_local_activate()`.

It lives in this module because this module is its sole writer, and `view`
reads it as **`runlocal._JOB_LOCAL`, at call time** (plan §8b). A
`from .runlocal import _JOB_LOCAL` over in `view` would bind `False` once at
import and the local lane would silently start hitting the real vast API —
precisely the credential touch `LOCAL_GPU_LANE.md` promises never to make. The
same indirection is what makes `monkeypatch.setattr(runlocal, "_JOB_LOCAL",
True)` steer the readers, which is how three existing tests drive the lane.

`view` and `runlocal` import each other (this module needs `_print_job_view`
and `_live_iids_set`; `view` needs the flag). The cycle is real, and it holds
because neither module touches the other's attributes at IMPORT time — only
inside function bodies. `test_vastlib_jobs_runlocal.py` imports the pair in
both orders to prove it.

What is deliberately NOT here
-----------------------------
* **`require_local_gpu`.** The local-GPU lane is authorized by CONFIG, not by
  this call site: one switch (`allow_local_gpu` in `herdd.yaml`), one home
  (`core.config`), owner ruling 2026-08-11. `_run_local_preflight` calls DOWN
  into it and does not re-implement the policy.
* **`joblocal.py` itself.** It is an absorbed sibling (plan §3) still living as
  a flat file and is imported bare-name until step 7. Its `live_boxes()` is
  documented as "the local answer to `_live_iids_set()`".
* **The argparse registration.** `pjrl.set_defaults(jobfunc=cmd_job_run_local)`
  and the `_JOB_LOCAL_SUBCOMMANDS` gate on which subparsers grow `--local` are
  `cli/` territory at step 6, and part of the §4 full-CLI-surface diff.

Provenance: behavior-preserving move of 7 symbols from `tools/vast/herdd.py`
(plan §8 step 5, 2026-08-16), each carrying its `# moved-from:` marker.
ADD-ONLY: `herdd.py` keeps its live copies until step 6.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

import joblocal

from vastlib.core import config
from vastlib.jobs import submit, view

import jobmeta

# --------------------------------------------------------------------------- #
# THE LOCAL LANE FLAG. False at import, True after `_job_local_activate()`.
# Read by `view._live_iids_set` and `view._present_iids_set` — the only two
# places that reach for the VAST API rather than for B2 — as
# `runlocal._JOB_LOCAL`, at call time. See the module docstring.
# --------------------------------------------------------------------------- #
# moved-from: herdd._JOB_LOCAL
_JOB_LOCAL = False

#: subcommands that make sense against a local-dir bucket. `attach`/`retarget`/
#: `requeue`/`supervise` are box concepts (install a daemon on / move work between
#: / babysit the bid of a rented machine) and deliberately have no --local.
# moved-from: herdd._JOB_LOCAL_SUBCOMMANDS
_JOB_LOCAL_SUBCOMMANDS = ("submit", "status", "wait", "logs", "pull", "ls", "cancel")


# moved-from: herdd._job_local_activate
def _job_local_activate() -> str:
    """Switch this process onto the local bucket. Idempotent."""
    global _JOB_LOCAL
    home = joblocal.activate()
    _JOB_LOCAL = True
    return home


# moved-from: herdd.cmd_job
def cmd_job(a: Any) -> Any:  # noqa: ANN401 — argparse.Namespace in, nested dispatch out
    """Dispatch `herdd job <action>`."""
    if getattr(a, "local", False):
        _job_local_activate()
    a.jobfunc(a)


# moved-from: herdd._run_local_preflight
def _run_local_preflight(a: Any,  # noqa: ANN401 — argparse.Namespace
                         home: str | None,
                         gpu_allow: Any,  # noqa: ANN401 — joblocal's list[int]|None
                         ) -> tuple[str, list[str]]:
    """Everything that must be true BEFORE a local jobd touches a card.

    Returns (root, warnings). Exits on a hard refusal."""
    # The local-GPU lane is authorized by config, not by this call site. One
    # switch, in vastconf — see local_gpu_allowed(). Checked FIRST, before any
    # directory is created or card probed, so a refusal leaves no debris.
    config.require_local_gpu("`job run-local`")

    root = os.path.abspath(os.path.expanduser(a.root)) if a.root \
        else joblocal.workspace_dir(home)
    os.makedirs(root, exist_ok=True)
    warn = []

    cards = joblocal.probe_gpus()
    if not cards:
        sys.exit("error: no GPU visible to nvidia-smi — this is the LOCAL GPU lane. "
                 "For a CPU-only plumbing check use `tools/vast/rehearse.sh <folder>`.")
    usable = [c for c in cards if not gpu_allow or c[0] in gpu_allow]
    if not usable:
        sys.exit(f"error: --gpus {a.gpus} selects no probed card "
                 f"(available: {', '.join(str(c[0]) for c in cards)})")
    print(">> local GPUs: " + ", ".join(
        f"[{i}] {name} {mib // 1024} GiB" for i, mib, name in usable))

    # THE GUARD THAT MATTERS. jobd's reap_orphan_gpu_procs SIGKILLs every compute
    # process whose ppid is 1 when it boots with nothing adopted — correct on a
    # fresh rented box, catastrophic here, where ppid==1 is the normal state of
    # any detached training run. jobd_env() forces JOBD_GPU_REAP=0 so we can
    # never do that; this refusal is the second half: do not silently contend for
    # a card someone (probably you, in another terminal) is already training on.
    busy = joblocal.foreign_gpu_procs([c[0] for c in usable])
    if busy:
        for pid, idx, name in busy:
            print(f"!! GPU {idx if idx is not None else '?'} is BUSY: pid={pid} {name}",
                  file=sys.stderr)
        if not a.force:
            sys.exit("error: refusing to start — the cards this job would use are "
                     "already running compute. Wait, pick free cards with --gpus, "
                     "or override with --force (we will NOT kill anything either way).")
        warn.append("started with --force over busy GPUs — expect OOM/contention")
    return root, warn


# moved-from: herdd._run_local_asset_warnings
def _run_local_asset_warnings(cfg: Any,  # noqa: ANN401 — validated job config
                              root: str) -> list[str]:
    """`_link_asset_dest` refuses any dest outside $JOBD_ROOT, so an absolute
    `dest: /workspace/...` silently does not link when the local root is
    elsewhere. Name it rather than let the entrypoint fail confusingly."""
    out = []
    for asset in (cfg.get("assets") or []):
        dest = asset.get("dest")
        if not dest or not str(dest).startswith("/"):
            continue
        if not (str(dest) + "/").startswith(root.rstrip("/") + "/"):
            out.append(
                f"asset {asset['name']!r} has an absolute dest {dest!r} outside the "
                f"local root {root} — jobd will REFUSE to link it. Use a relative "
                f"dest (works on a box too), or run with --root /workspace")
    return out


# moved-from: herdd.cmd_job_run_local
def cmd_job_run_local(a: Any) -> None:  # noqa: ANN401 — argparse.Namespace
    """Execute a jobs-v2 bundle on this machine's GPUs (LOCAL_GPU_LANE.md).

    With a folder: submit it to the local queue, then drain. Without: just drain
    whatever is already queued locally (which is how a resume works — re-running
    is jobd's own resume path, same JOB_ID, checkpoint pulled back)."""
    # jobd logs to stderr (unbuffered) while our own prints are block-buffered
    # when piped, which interleaves the two into nonsense. Line-buffer ours.
    try:
        sys.stdout.reconfigure(line_buffering=True)   # type: ignore[union-attr]
    except (AttributeError, OSError):                     # pragma: no cover
        pass
    home = _job_local_activate()
    print(joblocal.differences_banner())
    try:
        gpu_allow = joblocal.parse_gpu_allow(a.gpus)
    except joblocal.JoblocalError as e:
        sys.exit(f"error: {e}")

    root, warns = _run_local_preflight(a, home, gpu_allow)

    # --asset NAME=DIR: seed the cache as a symlink (NO copy — the base model is
    # already on this disk) and remember it, so later runs need no flags.
    amap = joblocal.load_asset_map(home)
    try:
        for spec in (a.asset or []):
            name, path = joblocal.parse_asset_arg(spec)
            amap[name] = path
        for name, path in list(amap.items()):
            if not os.path.isdir(path):
                warns.append(f"asset override {name!r} -> {path} no longer exists — ignoring")
                amap.pop(name, None)
                continue
            joblocal.seed_asset(root, name, path)
    except joblocal.JoblocalError as e:
        sys.exit(f"error: {e}")
    joblocal.save_asset_map(amap, home)
    if amap:
        print(">> local assets (no copy, symlinked into the cache): "
              + ", ".join(f"{k} -> {v}" for k, v in sorted(amap.items())))

    submitted = None
    if a.dir:
        try:
            # Fold the same overrides the real submit below will: a `${VAR}`
            # asset prefix is unresolvable without them, so a preflight that
            # skipped them would refuse a job that submits fine.
            _raw = jobmeta.load_job_config(a.dir)
            submit._apply_artifact_env(_raw, getattr(a, "artifact", None))
            submit._apply_env_overrides(_raw, getattr(a, "env", None))
            cfg, _ = jobmeta.validate_job_config(_raw, a.dir)
        except jobmeta.JobmetaError as e:
            sys.exit(f"error: {e}")
        warns.extend(_run_local_asset_warnings(cfg, root))
        for name in (x["name"] for x in (cfg.get("assets") or [])):
            if name not in amap:
                warns.append(
                    f"asset {name!r} has no local override — jobd will try to pull it "
                    f"from the (empty) local bucket and any `require:` glob will fail "
                    f"with asset_stage_failed:{name}. Add --asset {name}=<dir>")
    for w in warns:
        print(f"warn: {w}", file=sys.stderr)
    if a.dry_run:
        print(">> [dry-run] preflight only — nothing submitted, no daemon started.")
        return
    if a.dir:
        sub = argparse.Namespace(
            dir=a.dir, box=joblocal.local_box_id(), name=a.name, timeout=a.timeout,
            env=a.env, artifact=getattr(a, "artifact", None),
            dry_run=False, strict_assets=False,
            # the staleness preflight compares a LOCAL source against a B2 prefix;
            # there is no B2 in this lane and the source IS local.
            no_asset_check=True, local=True)
        submitted = submit.cmd_job_submit(sub)

    env = joblocal.jobd_env(home, root=root, gpu_allow=gpu_allow,
                            once=not a.watch, cpu_slots=a.cpu_slots,
                            python=sys.executable)
    print(f">> running jobd: root={root} box={joblocal.local_box_id()} "
          f"bucket={joblocal.bucket_dir(home)} mode={'watch' if a.watch else 'drain'}")
    rc = subprocess.call(["bash", joblocal.JOBD_SH], env=env)
    prog = os.path.basename(sys.argv[0])
    if submitted:
        try:
            v = jobmeta.read_job(submitted, live_iids=view._live_iids_set())
            print()
            view._print_job_view(v)
        except Exception as e:                            # never mask jobd's rc
            print(f"(could not fold {submitted}: {e})", file=sys.stderr)
        print(f">>   logs  : {prog} job logs {submitted} --local")
        print(f">>   pull  : {prog} job pull {submitted} --local")
        print(f">>   resume: {prog} job run-local        # re-runs the resume path")
    if rc != 0:
        sys.exit(f"!! jobd exited {rc}")
