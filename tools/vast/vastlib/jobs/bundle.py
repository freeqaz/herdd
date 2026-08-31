"""vastlib.jobs.bundle — what ships to a box, and the two gates in front of it.

Why this module exists
----------------------
`_job_attach_files()` is the repo's SECOND "what ships" manifest (the first is
`ship_manifest.txt`). It decides which files land flat in `/workspace/jobd/`
on a rented machine, and **three independent readers** consume that decision:
`test_jobd_bundle_imports_flat.py` (dynamic — every shipped leaf must import
bare-name under `python3 -P` in a flat, repo-invisible directory),
`shipcheck.py`'s `JOBD_FLAT_NAMESPACES` (its static twin), and
`test_broker_env.py`'s `_PINNED_JOBD_BUNDLE` name pin. A list with three
readers must have exactly one home; this is it.

The failure mode the whole module is shaped around is **silence**. `jobd.sh`
runs its python calls with `|| true`, so a bundle that cannot import does not
crash a box — it produces a machine that writes `JOBD_STATUS`, emits no events,
parses no tickets, looks idle, and bills. That is what happened when
`jobmeta.py`'s module-scope `from bidpolicy import DEFEND_*` shipped without
`bidpolicy.py` (84d09ab1), and it is why `_jobd_import_gate` is fail-closed.

The three things a port of this module can silently break
--------------------------------------------------------
* **`__file__` depth.** All three path-deriving functions here read
  `os.path.dirname(os.path.abspath(__file__))` in the flat module, where that
  is `tools/vast/`. This file sits **two directories deeper**, so the walk-back
  is hoisted into the two module constants below (`TOOLS_VAST_DIR`,
  `_REPO_ROOT`) rather than re-counted at each site. Get the count wrong and
  you get an empty bundle (nothing `os.path.isfile`s), or a `repo_root` that
  makes `shipcheck`'s `relpath` produce `../..` keys — and `_jobd_import_gate`
  swallows the resulting exception as a one-line NOTE, so the gate goes quiet
  and every bundle ships unchecked. Pinned by
  `test_vastlib_jobs_bundle.py::test_tools_vast_dir_*` /
  `::test_repo_root_matches_the_herdd_computation`, which compare against the
  expression applied to `herdd.py`'s own path. Nothing else fails when these
  drift.
* **The raw `rclone rcat`.** `_stage_jobd_bootstrap` uploads with a bare
  `subprocess.run([...], input=tar)` where `tar` is **bytes**. It does NOT go
  through `storage.b2._b2_rcat`, which passes `text=True` and would decode the
  tar into mojibake; the box would then pull a corrupt bundle and `jobd.sh`
  would swallow it with `|| true`. The raw call is kept verbatim, and
  `test_vastlib_jobs_bundle.py` stubs **this module's** `subprocess` attribute
  (not the storage seam) precisely because patching the seam leaves this write
  live.
* **The `sys.path.insert` in front of `import shipcheck`.** It is the ONE
  `sys.path` mutation in Zone P, and it is kept deliberately against the
  package's own no-`sys.path` rule (README §1) because it is what makes the
  bare-name import resolve for a caller that did not bootstrap `tools/vast`.
  It sits INSIDE the gate's blanket `except Exception`, so dropping it does not
  raise: it turns the fail-closed gate into a permanent one-line NOTE, which is
  the exact failure this gate exists to prevent. Removing it is a step-6 change
  (once every entry point is a `vastlib` console script), not a port-time
  cleanup, and it needs the positive control below to prove the gate still
  fires.
* **The `files` argument to `shipcheck`.** It is passed explicitly "so
  shipcheck does not re-exec this module". Call
  `shipcheck.jobd_import_closure_gaps()` with `shipped=None` and shipcheck
  `exec_module()`s the source file again under a synthetic module name,
  producing a second live copy with its own state.

What is deliberately NOT here
-----------------------------
* **`cmd_job_attach`.** It is the third consumer of these four functions
  (`_job_attach_files`, `_jobd_import_gate`, `_stage_jobd_bootstrap` and, via
  the launch lane, `_jobd_boot_snippet`) but it belongs with retarget/requeue/
  cancel/orphans in `jobs.control` (plan §5). Its `_stage_jobd_bootstrap` call
  sits inside `except (Exception, SystemExit)` — a staging refusal degrades to
  a loud warning rather than failing the attach — and that breadth is
  load-bearing; whoever ports it must keep both the exception set and the
  warning text.
* **`jobmatrix.submit_experiment`'s parallel pipeline.** It reimplements the
  same vram gate + `write_bundle` + sha + `os.replace` sequence that
  `jobs.submit` runs, with its own cwd-relative staging dir and its own
  `MatrixError` instead of `sys.exit`. It is absorbed into `vastlib.workflows`
  (plan §3); the divergence is recorded, not unified, exactly as the supervise
  lane-mirroring rule does.
* **No B2 key policy.** `jobs/jobd-boot/<sha>.tar` is a content-addressed,
  immutable object; the bucket comes from `B2_BUCKET` and the transport from
  `storage.b2`. Credential *minting* is `launch.spec`.

The layering note (plan §8 step 5, ruled 2026-08-16)
----------------------------------------------------
`compose_jobs_launch_env` lives HERE, in `jobs/`, and `vastlib.launch.launch`
keeps a **raising** seam for it until the `cli/` composition root binds the two
together at step 6. The import-linter contract puts `jobs` in the ring ABOVE
`launch`, so `launch` may never import this module — at module scope or inside
a function (import-linter reads the AST; a deferred import does not dodge it).
The direction is the problem, not the timing. `workflowctl.build_box_resolver`
already injects this function as `jobs_composer`, which is the same idiom the
composition root will use, and preserving that injectability (plus
`bootstrap_stager`) is a hard requirement of the port: it is the defect-#6
parity seam that makes a workflow-launched box compose IDENTICALLY to a manual
`launch --jobs` one.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 5,
2026-08-16), rev ea8360dc. Behavior-preserving: bodies and comments copied,
annotations added, plus the one documented mechanical change — the two
`__file__` walk-back constants in place of the inline `dirname` chains, which
moved two directories deeper with the file.
"""

from __future__ import annotations

import hashlib
import os
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Sequence

from vastlib.launch import spec
from vastlib.storage import b2

import jobmeta

__all__ = [
    "TOOLS_VAST_DIR",
    "_job_attach_files",
    "_jobd_boot_snippet",
    "_jobd_import_gate",
    "_sync_file_list",
    "_sync_import_gate",
    "_stage_jobd_bootstrap",
    "compose_jobs_launch_env",
]

# `tools/vast/` — the directory the flat `herdd.py` lived in, and the one
# `onstart/`, `jobmeta.py`, `bidpolicy.py` and the rest of the bundle still live
# in. In the flat module every one of the three path-deriving functions below
# spelled this as `os.path.dirname(os.path.abspath(__file__))`; this file sits
# two directories deeper, so the same path has to be walked back up. Hoisted to
# a module constant for the same reason `core.config._HERE` and
# `boxes.ssh._REPO_ROOT` are: the depth is a property of the module's location,
# not of the function, and a package that moves again should have to fix exactly
# one line. Pinned by `test_vastlib_jobs_bundle.py` against the expression
# applied to `herdd.py`'s own path — the drift is SILENT otherwise (a bundle
# of files that do not exist, refused one by one by `_stage_jobd_bootstrap`).
TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The repo root, as `_jobd_import_gate` computes it: three `dirname`s above
# `tools/vast/herdd.py`, five above this file. It is the base `shipcheck`
# relpaths the bundle against, so an off-by-one here does not raise — it
# produces `../..`-prefixed keys that match nothing, `shipcheck` reports gaps or
# throws, and the gate degrades to a NOTE on a green run.
_REPO_ROOT = os.path.dirname(os.path.dirname(TOOLS_VAST_DIR))


# moved-from: herdd._job_attach_files
def _job_attach_files() -> list[str]:
    here = TOOLS_VAST_DIR
    return [
        os.path.join(here, "onstart", "jobd.sh"),
        os.path.join(here, "onstart", "jobd.py"),
        os.path.join(here, "jobmeta.py"),
        os.path.join(here, "runmeta.py"),
        # jobmeta imports bidpolicy AT MODULE SCOPE (the `defend:` vocabulary is
        # owned by the bid ladder, 84d09ab1) — unguarded, so it is not a
        # degrade-quietly optional. In the repo that import resolves because
        # jobd.py puts BOTH the flat dir and its PARENT on sys.path and the
        # parent is tools/vast/ where bidpolicy.py lives; the shipped bundle is
        # FLAT under /workspace/jobd/ and has no such parent to fall back on, so
        # without this line every `python3 jobd.py ...` on a box dies at
        # "ModuleNotFoundError: No module named 'bidpolicy'". jobd.sh calls it
        # `>/dev/null 2>&1 || true`, so the box keeps writing JOBD_STATUS while
        # emitting no events and parsing no tickets: alive-looking, idle
        # forever, billing. bidpolicy's own closure is runmeta, already above.
        os.path.join(here, "bidpolicy.py"),
        os.path.join(here, "b2_sync.sh"),
        os.path.join(here, "metrics_probe.py"),  # host-metrics on job heartbeats
        # Boot GEMM ceiling (host-acceptance telemetry, 2026-08-07). jobd
        # resolves it at $JOBD_DIR/gemm_probe.py and it imports metrics_probe as
        # a FLAT sibling for its busy-GPU guard — ship the pair or the probe
        # refuses on every box ("cannot prove the GPU is idle"). Same delivery
        # defect as preempt_save.py below, caught before it shipped.
        os.path.join(here, "gemm_probe.py"),
        # Boot CPU probe (2026-08-25), and its closure. jobd resolves it at
        # $JOBD_DIR/cpu_probe.py; `cpu_probe drop` imports hostfacts as a FLAT
        # sibling to build the record, and imports gemm_probe (above) to reuse
        # the running-job census. hostfacts.py has never ridden THIS bundle —
        # the harvested producers get it from their own job bundle's
        # jobcommon/ symlink — so shipping cpu_probe.py alone would put a probe
        # on every box that dies at `import hostfacts` and drops nothing. Same
        # delivery defect as gemm_probe/metrics_probe above, caught the same way.
        os.path.join(here, "cpu_probe.py"),
        os.path.join(here, "hostfacts.py"),
        # cred-broker refresh lane (cred-broker-buildout.md §2.5/§2.6): jobd.sh
        # resolves cred_client at $JOBD_DIR/cred_client.py, and cred_client runs
        # tailnet_join.sh from ITS OWN dir — both must ride the same flat bundle
        # or maybe_refresh_creds hits its "no client — skipping" branch forever.
        os.path.join(here, "onstart", "cred_client.py"),
        os.path.join(here, "onstart", "tailnet_join.sh"),
        # needs.venv:eval provisioner (jobd.sh check_venv -> _venv_provisioner):
        # on any image without a baked eval tree /workspace/eval/env.sh is absent, so a
        # score job dies at "needs.venv=eval provisioning failed" unless this
        # self-contained fetcher rides the bundle (defect #6 score-stage sibling).
        os.path.join(here, "onstart", "fetch_eval_env.sh"),
        # needs.venv:serve provisioner (the eval sibling above): jobd.sh resolves
        # `job_serve.sh` at $JOBD_DIR and runs it `--build-venv`. Without it a
        # `launch --jobs` box has NO copy, so `needs.venv: serve` — and any job
        # calling the build-venv seam directly (frontier-wave S0.b2's gen
        # interpreter) — dies on a fresh box. Found by the 2026-07-29 spot smoke
        # (REMOTE_WAVE_PLAN §7 defect 2), which had to work around it by
        # shipping a bundle-local copy.
        os.path.join(here, "onstart", "job_serve.sh"),
        # …and the helper job_serve.sh resolves as its OWN sibling
        # ($HERE/serve_vllm.sh): without it the build-venv path still works but
        # a real in-job serve (eval_job_lib's ejl_serve_up) exits 2
        # "serve_vllm.sh not found". Ship the pair or the serve lane is half a
        # bundle. serve_vllm.sh's own sidecars are pulled from B2 at runtime and
        # it finds metrics_probe.py as a flat sibling, which this bundle carries.
        os.path.join(here, "onstart", "serve_vllm.sh"),
        # preempt-forced checkpoint, BOTH halves (2026-08-06). The jobs lane
        # carried NEITHER, which is why `preempt_save.py` had never once run on a
        # box: the trainer resolves it by probing $JOBD_DIR//workspace and found
        # nothing, logged "ModuleNotFoundError: No module named 'preempt_save'",
        # and never advertised a pid — after which jobd's preempt path had
        # nothing to signal even once it learned how to ask.
        #   preempt_save.py  — imported by tools/pipeline/ml_infra/train_proposer_lora.py
        #                      to arm the SIGUSR1 handler + advertise per-rank pids.
        #   preempt_trap.sh  — sourced by jobd.sh with PREEMPT_TRAP_NO_INSTALL=1
        #                      for `_preempt_local_save` ONLY (jobd keeps its own
        #                      TERM/INT trap; see the guard at the file's end).
        # Ship the pair or the jobs lane is back to flushing stale bytes.
        os.path.join(here, "onstart", "preempt_save.py"),
        os.path.join(here, "onstart", "preempt_trap.sh"),
        # shared cross-box Triton JIT cache: jobd's boot pull / post-job push
        # (triton_cache_boot_pull / triton_cache_push_bg) resolve this as
        # $JOBD_DIR/triton_cache.py and silently skip when absent — shipping it
        # is what turns the hook on.
        os.path.join(here, "triton_cache.py"),
        # b2x transport shim. jobd.sh sources it from $JOBD_DIR (or
        # $JOBD_DIR/onstart/) and, finding NEITHER, installs
        # `b2x_pull() { return 1; }` / `b2x_push() { return 1; }` — so every
        # b2x call site in the jobs lane has always fallen straight through to
        # the rclone line beside it. That is not a degraded transport, it is the
        # transport never running: vast shapes per TCP flow, and the measured
        # peak-flow spread is rclone-stock 4 / rclone-tuned 9 / b2x 68 on the
        # same object. The tell was invisible because the shim logs only
        # failures and this path never failed — it was simply never defined.
        # Same delivery defect as gemm_probe/preempt_save above, and it also
        # revives fetch_eval_env.sh, which sources b2x_boot.sh as its own
        # sibling (fetch_eval_env.sh:99) and has been falling back since it
        # shipped.
        os.path.join(here, "onstart", "b2x_boot.sh"),
        # b2x_boot.sh's rung-0 CDN tier looks for this beside itself and does
        # nothing without it, so shipping the shim alone ships half the ladder.
        os.path.join(here, "cdn_pull.py"),
    ]


# moved-from: herdd._jobd_import_gate
def _jobd_import_gate(files: list[str], warn_only: bool = False) -> None:
    """Import-closure gate on the JOBD BUNDLE, before it is staged or pushed.

    The bundle is the repo's SECOND "what ships" manifest (the first is
    ship_manifest.txt, gated by _sync_import_gate above) and until 2026-08-14
    it was guarded only by a name-pinning test — which is how jobmeta.py's
    module-scope `from bidpolicy import DEFEND_*` shipped without bidpolicy.py
    and killed jobd on every box launched after 84d09ab1, silently, for hours.
    Same detector as the sync gate, aimed at the other list.

    Fail-closed: a bundle that cannot import is a box that bills for nothing.
    The file set is passed in so shipcheck does not re-exec this module.
    Any shipcheck-internal problem degrades to a NOTE — a guard must never be
    the reason a box cannot be attached. Escape hatch: JOBD_NO_IMPORT_CHECK=1.
    """
    root = _REPO_ROOT
    try:
        sys.path.insert(0, TOOLS_VAST_DIR)
        import shipcheck
        shipped = {os.path.relpath(os.path.abspath(f), root).replace(os.sep, "/")
                   for f in files}
        gaps = shipcheck.jobd_import_closure_gaps(root, shipped)
        lines = shipcheck.format_import_gaps(gaps, shipcheck.JOBD_MANIFEST_HINT,
                                             "jobd bundle")
    except Exception as e:                       # defense in depth
        print(f"note: jobd import-closure check skipped ({e})", file=sys.stderr)
        return
    # Box-floor syntax gate, SAME fail-closed contract as the closure check.
    # The workstation's newer python imports PEP 701 code fine, so the closure
    # check alone passes a bundle the box's 3.11 cannot parse — and the shipper
    # is not always a tested checkout (fleetd re-stages from its OWN tree on
    # every replacement launch and re-attach; boxes 48094838/48132001,
    # 2026-08-19). See pyfloor.py / test_box_python_floor.py.
    try:
        import pyfloor
        floor_bad = pyfloor.floor_gaps(files)
        floor_ver = ".".join(map(str, pyfloor.BOX_PYTHON_FLOOR))
    except Exception as e:                       # defense in depth
        print(f"note: jobd box-floor syntax check skipped ({e})", file=sys.stderr)
        floor_bad = []
    if floor_bad:
        lines = list(lines) + [
            f"jobd bundle uses syntax newer than python {floor_ver} "
            f"(the box floor):",
            *(f"  {ln}" for ln in floor_bad)]
    if not gaps and not floor_bad:
        return
    for ln in lines:
        print(ln, file=sys.stderr)
    if warn_only or os.environ.get("JOBD_NO_IMPORT_CHECK") == "1":
        print("note: JOBD_NO_IMPORT_CHECK=1 — shipping the broken bundle anyway",
              file=sys.stderr)
        return
    sys.exit("error: refusing to ship the jobd bundle — it is not import-closed. "
             "jobd.sh runs its python calls with `|| true`, so on a box this is "
             "SILENT: no events, no ticket parsing, a box that looks idle and "
             "bills. Add the module(s) above to _job_attach_files() (and to "
             "_PINNED_JOBD_BUNDLE in test_broker_env.py), or set "
             "JOBD_NO_IMPORT_CHECK=1 if you truly mean it.")


# moved-from: herdd._stage_jobd_bootstrap
def _stage_jobd_bootstrap(dry_run: bool = False) -> str:
    """Stage the jobd daemon files as a content-addressed PLAIN tar to
    b2:$B2_BUCKET/jobs/jobd-boot/<sha>.tar so a `launch --jobs` onstart can
    pull + exec them (they exceed Vast's 16 KiB onstart cap). FLAT layout ==
    exactly what `job attach` scp's into /workspace/jobd/. Reuses jobmeta's
    deterministic tar so unchanged files dedupe across launches. Returns the
    sha256 (dry-run computes+returns it without any B2 write)."""
    files = _job_attach_files()
    for f in files:
        if not os.path.isfile(f):
            sys.exit(f"error: missing jobd file to stage: {f}")
    _jobd_import_gate(files)                   # never stage an unimportable bundle
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            shutil.copy(f, os.path.join(td, os.path.basename(f)))
        tar = jobmeta.deterministic_tar_bytes(td)
    sha = hashlib.sha256(tar).hexdigest()
    key = f"jobs/jobd-boot/{sha}.tar"
    if dry_run:
        print(f">> [dry-run/--jobs] would stage jobd bundle -> {key} ({len(tar)} B)")
        return sha
    b2._ensure_b2_remote()
    bucket = os.environ.get("B2_BUCKET")
    rc, out = b2._rclone(["lsf", f"b2:{bucket}/{key}"])   # dedupe: immutable object
    if rc == 0 and (out or "").strip():
        print(f">> jobd bundle present ({sha[:12]}…) — reusing")
        return sha
    # rcat streams a PUT (no HeadObject 403 flake); binary body -> no text mode
    r = subprocess.run(["rclone", "rcat", f"b2:{bucket}/{key}"], input=tar)
    if r.returncode != 0:
        sys.exit("error: failed to stage jobd bootstrap bundle to B2")
    print(f">> staged jobd bundle -> {key} ({len(tar)} B)")
    return sha


# moved-from: herdd._jobd_boot_snippet
def _jobd_boot_snippet(sha: str) -> str:
    """The onstart prelude that pulls + starts jobd at boot (onstart/jobd_boot.sh
    with the staged bundle sha baked in). Mirrors hf_login_snippet's shape."""
    path = os.path.join(TOOLS_VAST_DIR, "onstart", "jobd_boot.sh")
    with open(path) as fh:
        return fh.read().replace("@JOBD_BUNDLE_SHA@", sha) + "\n"


# moved-from: herdd.compose_jobs_launch_env
def compose_jobs_launch_env(env: dict[str, str], onstart: str | None, *,
                            dry_run: bool = False,
                            key_base: str | None = None,
                            no_idle_park: bool = False,
                            idle_park_grace: object = None,
                            no_job_deadline: object = None,
                            timeout_s: float | None = None,
                            bootstrap_stager: Callable[..., str] | None = None,
                            ) -> tuple[str, str]:
    """Compose the `--jobs` provision-time jobd boot onto (env, onstart) — the
    ONE reusable jobs-launch-body builder shared by BOTH `_do_launch`'s --jobs
    block and `workflowctl.build_box_resolver`, so a workflow-launched box
    boots jobd EXACTLY like the manual `herdd launch --jobs` path (fixing the
    2026-07-15 defect #6: a workflow box launched BARE — no jobd onstart, no
    scoped-B2 env — never ran jobd and no job was ever claimed).

    It mutates `env` in place (setdefault throughout: an explicit value already
    present always wins) with the scoped B2 cred pair (bucket-wide read +
    jobs/-restricted write), the queue transport vars
    (B2_BUCKET/B2_S3_ENDPOINT/B2_REGION), CRED_ROLE=jobs, the minted key's
    expiry, the EU read-replica pairs, and the idle-park knobs; stages the
    content-addressed jobd bundle (via `bootstrap_stager`, default
    `_stage_jobd_bootstrap`); and prepends `_jobd_boot_snippet(sha)` to
    `onstart`. Returns `(onstart_with_prelude, bootstrap_sha)`.

    A UNIQUE per-launch mint-key base (time+random nonce) is used unless
    `key_base` is given, so a second `--jobs`/workflow launch never revokes a
    still-running box's key (the 2026-07-12 box-44566398 incident).

    Without B2_BUCKET/B2_S3_ENDPOINT in env/.env there is no queue transport,
    so this is a hard error — a jobd box with no queue can never claim a job."""
    bucket = os.environ.get("B2_BUCKET")
    endpoint = os.environ.get("B2_S3_ENDPOINT")
    region = os.environ.get("B2_REGION", "us-west-004")
    if not (bucket and endpoint):
        sys.exit("error: --jobs needs B2_BUCKET/B2_S3_ENDPOINT in env/.env "
                 "(the box's queue transport)")
    # cred-broker identity (§2.1): a fresh per-box nonce gates the B2-mediated
    # cred-refresh lane (broker-URL-less by design). setdefault so a manual
    # `_do_launch` that already stamped the nonce pre-jobs is unchanged, while a
    # workflow-launched box (which had NONE before defect #6) now gets one too.
    env.setdefault("BOX_IDENTITY_NONCE", secrets.token_hex(16))
    if os.environ.get("CRED_BROKER_URL"):
        env.setdefault("CRED_BROKER_URL", os.environ["CRED_BROKER_URL"])
        if os.environ.get("TS_AUTHKEY"):
            env.setdefault("TS_AUTHKEY", os.environ["TS_AUTHKEY"])
    key_nonce = f"{int(time.time())}-{random.randint(0, 0xffff):04x}"
    base = key_base or f"job-launch-{key_nonce}"
    hours = spec._ephemeral_hours(timeout_s)
    _b2env = spec._ship_b2_env(base, hours=hours, write_prefix="jobs/", dry_run=dry_run)
    stager = bootstrap_stager if bootstrap_stager is not None else _stage_jobd_bootstrap
    sha = stager(dry_run=dry_run)
    env.setdefault("CRED_ROLE", "jobs")     # broker policy role (§2.1); ships
    # even with no CRED_BROKER_URL — the B2-mediated lane (§2.4) is nonce-gated
    _exp = spec._minted_expiry(base, hours)
    if _exp:
        env.setdefault("B2_KEY_EXPIRES_AT", str(_exp))
    env.setdefault("B2_BUCKET", bucket)
    for _k, _v in _b2env:
        env.setdefault(_k, _v)
    env.setdefault("B2_S3_ENDPOINT", endpoint)
    env.setdefault("B2_REGION", region)
    for _k, _v in spec._b2_eu_pairs():       # region-aware read replica (opt-in via .env)
        env.setdefault(_k, _v)
    for _k, _v in spec._r2_tc_pairs():       # shared Triton JIT cache (opt-in via .env)
        env.setdefault(_k, _v)
    for _k, _v in spec._cdn_pairs():         # CDN weights mirror (opt-in via .env)
        env.setdefault(_k, _v)
    if no_idle_park:
        env["JOBD_IDLE_PARK"] = "0"
    if idle_park_grace is not None:
        env["JOBD_IDLE_PARK_S"] = str(idle_park_grace)
    if no_job_deadline is not None:
        env["JOBD_NO_JOB_PARK_S"] = str(no_job_deadline)
    onstart = _jobd_boot_snippet(sha) + (onstart or "")
    return onstart, sha


# --------------------------------------------------------------------------- #
# The SYNC side of the same question: which repo files reach a box, and does
# that set import-close?
# --------------------------------------------------------------------------- #
# `_jobd_import_gate` above gates the jobd BUNDLE (`_job_attach_files`);
# these two gate the rsync SHIP SET (`ship_manifest.txt`). Same detector
# (`shipcheck`), two lists, one failure mode — a box that rents, pulls, and dies
# on an ImportError after the meter started (the 2026-07-30 frontier wave
# shipped `witness_frontier.py` without `inplace_build.py` and burned a box on
# an S0 import gate; jobmeta.py shipped without bidpolicy.py and killed jobd
# silently on every box launched after 84d09ab1). Keeping the pair in one module
# is what makes "the two gates use the same shipcheck entry points" checkable by
# reading one file.
#
# `cli-surface.json` files both under `cli/sync.py` because `sync` is their only
# caller; they land one ring DOWN instead, next to their twin, and `cli/sync.py`
# imports them. `_load_ship_manifest` — the pathspec PARSER — deliberately stays
# with `cli/sync.py`: it reads a CLI-facing allowlist file and has no bundle
# twin here.
#
# THE `sys.path.insert` IS PORTED, and `TOOLS_VAST_DIR` replaces the flat
# `dirname(abspath(__file__))` — see this module's docstring for why it is kept
# against the package's own no-`sys.path` rule, and why dropping it would turn a
# fail-closed gate into a permanent one-line NOTE rather than raising.


# moved-from: herdd._sync_file_list
def _sync_file_list(repo_root: str, paths: Sequence[str]) -> list[str]:
    """Tracked files under `paths` (git ls-files), relative to repo_root.
    Tracked-only is the point: caches, .env/secrets, *.db, build junk and
    worktree scratch never ship. Exits on a path that matches nothing."""
    r = subprocess.run(["git", "-C", repo_root, "ls-files", "-z", "--", *paths],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"error: git ls-files failed: {r.stderr.strip()}")
    files = [f for f in r.stdout.split("\0") if f]
    if not files:
        sys.exit(f"error: no tracked files match {paths!r}")
    return files


# moved-from: herdd._sync_import_gate
def _sync_import_gate(repo_root: str, warn_only: bool = False) -> None:
    """Import-closure gate on the DEFAULT manifest path set, before any rsync.

    The bake catches a broken closure late (eval-env/smoke_check.py check (f)
    imports it against the pruned tree); `sync` had no equivalent, which is how
    the 2026-07-30 frontier wave shipped witness_frontier.py without
    inplace_build.py and burned a box on an S0 import gate. Fail-closed here:
    the cost of the check is milliseconds and the cost of missing it is a rented
    GPU. Any shipcheck-internal problem degrades to a NOTE — a guard must never
    be the reason a sync cannot run."""
    try:
        sys.path.insert(0, TOOLS_VAST_DIR)
        import shipcheck
        gaps = shipcheck.import_closure_gaps(repo_root)
        lines = shipcheck.format_import_gaps(gaps)
    except Exception as e:                       # defense in depth
        print(f"note: import-closure check skipped ({e})", file=sys.stderr)
        return
    if not gaps:
        print(lines[0])
        return
    for ln in lines:
        print(ln, file=sys.stderr)
    if warn_only:
        print("note: --no-import-check — syncing anyway", file=sys.stderr)
        return
    sys.exit("error: refusing to sync — ship_manifest.txt is not import-closed "
             "(add the module(s) above, or pass --no-import-check)")
