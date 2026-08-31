#!/usr/bin/env bash
# onstart/job_serve.sh — reusable, idempotent, IN-JOB vLLM serve bring-up.
#
# A jobs-v2 entrypoint (onstart/jobd.sh) that needs a base+LoRA endpoint on the
# SAME box it runs on calls this ONCE per attempt. It reuses onstart/serve_vllm.sh
# verbatim (the single source of truth for the `vllm serve` argv, B2 base/adapter
# pull, SERVE_STATUS markers) but backgrounds it and gates on a box-local
# /v1/models poll — no tunnel, no laptop round-trip. It is safe to re-run after a
# spot preemption / park+resume: it kills any stale server first, so the READY
# state is reconstructed idempotently on every job restart.
#
# Why this exists (vs launch_serve.sh): launch_serve is a laptop wrapper that
# rents/attaches a box and drives serve from OUTSIDE. In the box-side spot-tolerant
# job model, the serve is one phase of the entrypoint that already owns the box —
# so serve bring-up must be a plain function the entrypoint can call and re-call.
#
# Required env (forwarded to serve_vllm.sh — see its header for semantics):
#   MODEL_B2        b2 subpath of the base-model dir (HF-Xet-deadlock-safe pull)
#   SERVED_NAME     --served-model-name for the base (e.g. proposer-base-7b)
#   VLLM_API_KEY    bearer (localhost-only here, but serve_vllm still wants it)
#   B2_KEY_ID B2_APPLICATION_KEY B2_BUCKET B2_S3_ENDPOINT [B2_REGION]
# Optional env:
#   LORA_SPECS      name=b2subpath[,name2=...]  (e.g. proposer-v4-gen=artifacts/corpus-v4-01/serve/gen)
#   MAX_LORA_RANK   default 32                  MAX_LEN         default 16384
#   GPU_UTIL        default 0.90 (0.95 + a 2nd job OOMs)
#   KV_CACHE_DTYPE  e.g. fp8 (fine on sm>=8.9 / 5090; NOT on A6000/SM8.6 — bf16 there)
#   VLLM_SPEC       pip spec for the serve venv — FALLBACK ONLY. Its default
#                   (vllm==0.24.0) is the RETIRED stock lane, NOT the pin of
#                   record (tools/vast/train-env/VLLM_PIN). On the shipped image
#                   it is never used: vLLM is baked into system dist-packages, so
#                   build_serve_venv's global `import vllm` probe wins and nothing
#                   is installed. It fires only on an image carrying no vLLM.
#                   DATED PROVENANCE (2026-07-30, both frontier waves): a stock
#                   0.24.0 install pulled torch 2.11+cu130 and died "The NVIDIA
#                   driver on your system is too old" on a cuda_max_good 12.9 box.
#                   That is WHY ensure_cuda_init exists — it is history, not the
#                   current stack, which is cu129 end to end and rented at the
#                   image's CUDA-12 family floor (vastlib LAUNCH_CUDA_MAX_GOOD).
#                   BLACKWELL NOTE: if the pinned vLLM/torch can't init on
#                   sm_120, the readiness gate below TIMES OUT and this returns
#                   non-zero — the caller's serve gate then fails CHEAPLY (no arm
#                   budget spent). Bump VLLM_SPEC to a Blackwell-capable pin if
#                   that happens.
#   JOB_SERVE_SKIP_CUDA_PROBE=1  skip the CUDA-init probe (already skipped when
#                   no nvidia-smi is present, i.e. CPU boxes / rehearsal lanes)
#   SERVE_VENV      default /workspace/serve (persists across job restarts)
#   READY_TIMEOUT   seconds to wait for /v1/models (default 1200)
#   SERVE_LOG       default /workspace/serve.log
#   SERVE_VLLM_SH   path to serve_vllm.sh (default: sibling of this script)
#
# Exit: 0 = READY (all served ids listed), non-zero = not ready in time / setup fail.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SERVE_VLLM_SH="${SERVE_VLLM_SH:-$HERE/serve_vllm.sh}"
SERVE_VENV="${SERVE_VENV:-/workspace/serve}"
SERVE_LOG="${SERVE_LOG:-/workspace/serve.log}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
# Retired stock spec, kept as the no-vLLM-image fallback only — see the header.
# Unused on the shipped image, whose baked vLLM satisfies the probe below.
VLLM_SPEC="${VLLM_SPEC:-vllm==0.24.0}"

# --- serve venv build (FACTORED so jobd's `needs.venv: serve` provisioning can
# --- build the venv WITHOUT starting a serve — see `--build-venv` below) -------
# A fresh venv (NOT --system-site-packages) so pip pulls vLLM's own torch/CUDA
# wheels rather than binding the image's torch. Idempotent: the `import vllm`
# probe skips the (multi-minute, multi-GB) install on a job restart / warm box.
# Returns 0 on a usable venv, 3 on any build/import failure. Emits progress to
# STDOUT so the caller (jobd routes ours to its log) reads it as breadcrumbs.
# Resolves a python interpreter with vLLM importable into SERVE_PY (a global,
# read by the caller to put the right `vllm` console script on PATH). Prefers an
# ISOLATED venv; falls back to a global `--break-system-packages` install when no
# image interpreter can build one. RATIONALE: the axolotl eval image's
# /usr/bin/python3.* ship WITHOUT ensurepip, so `python3 -m venv` on them dies
# "ensurepip is not available" (observed 2026-07-12, box 44612403). The validated
# training scripts hit the identical wall and fall back to a global install
# (corpus-v4/train.sh:130, n5prime-datagrowth/train.sh:97) — mirror that here
# rather than hard-failing. The venv attempt now tests CREATION SUCCESS (not mere
# interpreter existence) and tries the PATH `python3` (axolotl's, ensurepip-capable
# when conda-based) FIRST, so a clean isolated venv is still the common outcome.
SERVE_PY=""
build_serve_venv() {
  # warm-box fast paths (job restart / park+resume): skip the multi-GB install.
  if [ -x "$SERVE_VENV/bin/python" ] && "$SERVE_VENV/bin/python" -c 'import vllm' >/dev/null 2>&1; then
    SERVE_PY="$SERVE_VENV/bin/python"
    echo ">> job_serve: serve venv already usable at $SERVE_VENV (vllm importable)"; return 0
  fi
  # THE SHIPPED PATH, not an edge case: the image bakes vLLM into system
  # dist-packages (and the train venv inherits it via --system-site-packages), so
  # this returns before $VLLM_SPEC is ever read and no pip runs on a cold box.
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import vllm' >/dev/null 2>&1; then
    SERVE_PY="$(command -v python3)"
    echo ">> job_serve: vllm already importable in global python3 ($SERVE_PY)"; return 0
  fi
  echo ">> job_serve: building serve env ($VLLM_SPEC)"
  # 1) prefer an isolated venv — first interpreter whose ensurepip actually works.
  local c bs=""
  SERVE_PY=""
  rm -rf "$SERVE_VENV"   # clear any half-built venv from a prior failed attempt
  for c in python3 /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -m venv "$SERVE_VENV" >/dev/null 2>&1; then
      SERVE_PY="$SERVE_VENV/bin/python"; echo ">> job_serve: venv built with $c"; break
    fi
    rm -rf "$SERVE_VENV"
  done
  # 2) fallback: no interpreter could build a venv (ensurepip missing) — install
  #    into the global python with --break-system-packages. vLLM's own torch cu128
  #    wheels replace the image torch in place; fine for a serve-only box.
  if [ -z "$SERVE_PY" ]; then
    SERVE_PY="$(command -v python3 || true)"
    [ -n "$SERVE_PY" ] || { echo "!! job_serve: no python3 to build serve env" >&2; return 3; }
    bs="--break-system-packages"
    echo "!! job_serve: venv unavailable (ensurepip missing) — global install via $SERVE_PY $bs"
  fi
  "$SERVE_PY" -m pip install $bs -q --upgrade pip wheel >/dev/null 2>&1 || true
  # Minimal fp serve deps: vLLM + peft (LoRA). We deliberately do NOT install
  # arctic-inference / bitsandbytes from requirements-serve.txt — the paired-eval
  # instrument serves fp/fp8-KV only (no --quantization, no spec-decode).
  # NOT -q: pip's real output (the "The conflict is caused by:" block) goes to a
  # log file, so the success path stays quiet in the job log but a FAILURE is
  # self-diagnosing. Re-running pip just to capture the message would be wrong —
  # a failure that happened at DOWNLOAD time would re-pull multiple GB.
  # PIP_CONSTRAINT is dumped because it is the single most likely cause: the
  # image bakes one pinning torch to a +cu129 LOCAL version that exists only on
  # download.pytorch.org, which no EMPTY venv can resolve (measured 2026-08-05 —
  # 8 s to ResolutionImpossible, and the -q above meant the log named a symptom
  # and no cause).
  local _piplog="${SERVE_VENV%/}.pip.log"
  if ! "$SERVE_PY" -m pip install $bs --progress-bar off "$VLLM_SPEC" peft >"$_piplog" 2>&1; then
    echo "!! job_serve: pip install '$VLLM_SPEC' peft failed" >&2
    echo "!! job_serve:   SERVE_PY=$SERVE_PY [$("$SERVE_PY" -V 2>&1)] real=$("$SERVE_PY" -c 'import sys;print(sys.executable)' 2>&1)" >&2
    echo "!! job_serve:   PIP_CONSTRAINT=${PIP_CONSTRAINT:-<unset>} PIP_INDEX_URL=${PIP_INDEX_URL:-<default>} PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-<unset>}" >&2
    if [ -n "${PIP_CONSTRAINT:-}" ] && [ -r "${PIP_CONSTRAINT}" ]; then
      echo "!! job_serve:   --- ${PIP_CONSTRAINT} ---" >&2
      sed 's/^/!! job_serve:   /' "${PIP_CONSTRAINT}" >&2
    fi
    echo "!! job_serve:   --- last 60 lines of ${_piplog} ---" >&2
    tail -n 60 "$_piplog" >&2
    return 3
  fi
  "$SERVE_PY" -c 'import vllm' >/dev/null 2>&1 \
    || { echo "!! job_serve: vllm still not importable via $SERVE_PY" >&2; return 3; }
  echo ">> job_serve: serve env ready ($SERVE_PY)"
  return 0
}

# Restore an importable distutils for $SERVE_PY. vllm serve's engine import chain
# pulls torch.utils.cpp_extension -> setuptools -> distutils.core at RUNTIME (NOT
# exercised by a plain `import vllm`, so build_serve_venv's probe can't catch it).
# ROOT CAUSE (box 44612403, 2026-07-12): the baked eval-env venv ships setuptools
# 59.6.0, which defaults to SETUPTOOLS_USE_DISTUTILS=stdlib — but this Debian
# python3.10-minimal has only a PARTIAL stdlib distutils (distutils/__init__.py,
# no core.py), so `import distutils.core` dies. setuptools>=60 defaults to the
# VENDORED distutils ("local") and setuptools<74 still SHIPS it — so the fix is a
# version in [60,74). A plain `setuptools<74` install is a NO-OP here (59.6.0
# already satisfies it), and `-U setuptools` would jump to >=74 (vendored distutils
# REMOVED) and reintroduce the break — so pin the range >=69,<74 to force the
# upgrade off 59.6.0 while staying below the removal. apt distutils is the fallback.
# Runs on BOTH cold and warm serve paths; no-op once distutils imports.
# --break-system-packages is harmless in a real venv.
ensure_distutils() {
  "$SERVE_PY" -c 'import distutils.core' >/dev/null 2>&1 && return 0
  echo ">> job_serve: distutils.core missing for $SERVE_PY — upgrading setuptools into [69,74) for its vendored distutils"
  "$SERVE_PY" -m pip install --break-system-packages -q 'setuptools>=69,<74' >/dev/null 2>&1 || true
  "$SERVE_PY" -c 'import distutils.core' >/dev/null 2>&1 && return 0
  local pv; pv="$("$SERVE_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  echo ">> job_serve: distutils still missing — apt install python${pv}-distutils"
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq "python${pv}-distutils" >/dev/null 2>&1 || true
  "$SERVE_PY" -c 'import distutils.core' >/dev/null 2>&1 && return 0
  echo "!! job_serve: distutils.core still unavailable — vllm serve will fail" >&2
  return 3
}

# Install the Python dev headers (Python.h) for $SERVE_PY. At vLLM engine-core
# init, Triton/Inductor JIT-COMPILES a `cuda_utils.cpython-<ver>.so` extension
# with gcc, which #includes <Python.h>. The slim python3.10-minimal eval-env venv
# has no dev headers (its include dir /usr/include/python3.10 is empty), so the
# compile dies "Python.h: No such file or directory" and engine init fails (box
# 44612403, 2026-07-12; verified apt python3.10-dev makes the exact compile pass).
# gcc + libcuda.so.1 + triton's bundled cuda headers are already present — only
# Python.h is missing. Runs on cold+warm serve paths; no-op once Python.h exists.
ensure_py_headers() {
  local inc; inc="$("$SERVE_PY" -c 'import sysconfig;print(sysconfig.get_path("include"))' 2>/dev/null)"
  [ -n "$inc" ] && [ -f "$inc/Python.h" ] && return 0
  local pv; pv="$("$SERVE_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  echo ">> job_serve: Python.h missing (${inc:-?}) — apt install python${pv}-dev (triton runtime compile)"
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq "python${pv}-dev" >/dev/null 2>&1 || apt-get install -y -qq python3-dev >/dev/null 2>&1 || true
  [ -n "$inc" ] && [ -f "$inc/Python.h" ] && return 0
  echo "!! job_serve: Python.h still missing — triton cuda_utils compile will fail" >&2
  return 3
}

# Prove the installed torch can actually TALK TO THIS BOX'S DRIVER before anyone
# spends an arm on it. ORIGIN (2026-07-30, both live frontier waves — dated, and
# NOT the current stack): the then-default stock vllm==0.24.0 pulled torch
# 2.11+cu130, the rented box had cuda_max_good 12.9, and engine init died "The
# NVIDIA driver on your system is too old (found version 12090)". `import vllm`
# DOES NOT initialize CUDA, so build_serve_venv's import probe passed and the
# failure only surfaced at gen start — after a full S0 stage and a multi-GB pip
# install. A CUDA context costs ~2 s and moves that discovery to provisioning.
# STILL EARNS ITS LINE on the current stack: it probes whichever torch $SERVE_PY
# resolved to — normally the image's baked cu129 build — against THIS box's
# driver. Same check, different pair, and the pair is what it reports.
# Runs on BOTH cold and warm paths: a WARM venv on a freshly-rented box is exactly
# the shape that breaks (same wheels, different driver). Skipped with an explicit
# note when no nvidia-smi is present (CPU boxes, rehearsal lanes, --image
# rehearsals) — never a failure there. JOB_SERVE_SKIP_CUDA_PROBE=1 forces the skip.
ensure_cuda_init() {
  if [ "${JOB_SERVE_SKIP_CUDA_PROBE:-0}" = "1" ]; then
    echo ">> job_serve: CUDA-init probe skipped (JOB_SERVE_SKIP_CUDA_PROBE=1)"; return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ">> job_serve: no nvidia-smi — skipping CUDA-init probe (CPU/rehearsal lane)"
    return 0
  fi
  local out
  if out="$("$SERVE_PY" -c 'import torch; torch.zeros(1).cuda(); print("cuda_ok", torch.version.cuda)' 2>&1)"; then
    echo ">> job_serve: CUDA init OK ($(printf '%s' "$out" | tail -1))"
    return 0
  fi
  # Report the torch that actually failed, not a hardcoded spec: $SERVE_PY is
  # usually the IMAGE's interpreter, so naming $VLLM_SPEC here would misdirect.
  # `torch.version.cuda` is a plain attribute — no CUDA context, no second probe.
  local tcu; tcu="$("$SERVE_PY" -c 'import torch;print(torch.version.cuda)' 2>/dev/null)"
  echo "!! job_serve: CUDA INIT FAILED for $SERVE_PY (torch CUDA build ${tcu:-unknown})" >&2
  printf '%s\n' "$out" | tail -5 >&2
  case "$out" in
    *"driver on your system is too old"*|*"CUDA driver version is insufficient"*|\
    *"Found no NVIDIA driver"*)
      echo "!! job_serve: the box DRIVER is too old for this torch build. Rent a host whose cuda_max_good covers CUDA ${tcu:-<the torch build below>} (herdd search --cuda ${tcu:-<that version>}). NOTE the CLI's default floor is vastlib LAUNCH_CUDA_MAX_GOOD, which tracks the IMAGE's CUDA build (train-env/VLLM_PIN) — it does not cover a venv built later on the box, so pin VLLM_SPEC to an older-CUDA torch if that is what failed." >&2 ;;
  esac
  return 3
}

# --- build-only entry: `job_serve.sh --build-venv` (or JOB_SERVE_ACTION=build-venv)
# Provisions the serve venv and EXITS — no MODEL_B2/SERVED_NAME, no serve_vllm.sh,
# no server. This is the callable seam jobd's check_venv invokes for `venv: serve`
# so the pip lines live in ONE place. Idempotent + safe on a warm box.
if [ "${1:-}" = "--build-venv" ] || [ "${JOB_SERVE_ACTION:-}" = "build-venv" ]; then
  build_serve_venv || exit $?
  ensure_distutils || exit $?
  ensure_py_headers || exit $?
  ensure_cuda_init || exit $?
  echo ">> job_serve: --build-venv complete"
  exit 0
fi

[ -f "$SERVE_VLLM_SH" ] || { echo "!! job_serve: serve_vllm.sh not found at $SERVE_VLLM_SH" >&2; exit 2; }
# MODEL_B2 is required ONLY when there is no local MODEL_ID — this mirrors
# serve_vllm.sh's own contract ("MODEL_ID … required unless MODEL_B2", :5-7),
# which this wrapper was stricter than. A job that has already STAGED the base
# as an asset must be able to serve it from local disk: re-pulling 15 GB from
# B2 on every spot resume is exactly the waste `assets:` exists to prevent.
# Found live 2026-08-05: the q6 round-1 of-record eval set MODEL_ID to its
# staged asset and unset MODEL_B2, and died here (rc 3) AFTER passing every
# identity gate. Rehearsal cannot catch it — `--stub-vllm` replaces this file.
if [ -z "${MODEL_B2:-}" ] && [ -z "${MODEL_ID:-}" ]; then
  echo "!! job_serve: MODEL_B2 or MODEL_ID required (neither set)" >&2; exit 2
fi
: "${SERVED_NAME:?job_serve: SERVED_NAME required}"

# --- 1. clean any stale server (serve_vllm.sh does NOT self-clean :8000) -------
# This is the launch_serve.sh --restart step; makes a resume-time re-run safe.
echo ">> job_serve: clearing any stale vllm/haproxy on :8000"
pkill -f 'vllm serve' 2>/dev/null || true
pkill -x haproxy      2>/dev/null || true
sleep 2

# --- 2. serve venv: vLLM pip-in-venv on a non-vllm image (train.sh:477 pattern) -
build_serve_venv || exit $?
ensure_distutils || exit $?
ensure_py_headers || exit $?
ensure_cuda_init || exit $?
# Put the resolved interpreter's console-scripts dir (venv bin OR the global
# scripts dir when we fell back to --break-system-packages) on PATH, so the
# peer-owned serve_vllm.sh finds `vllm` regardless of which path build took.
SERVE_BIN_DIR="$("$SERVE_PY" -c 'import sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null || dirname "$SERVE_PY")"
export PATH="$SERVE_BIN_DIR:$(dirname "$SERVE_PY"):$PATH"

# --- 3. launch serve_vllm.sh detached ----------------------------------------
# CPU_FARM=0: belt-and-braces. It is serve_vllm.sh's default since 2026-08-21,
#   but a job box must never inherit a stray CPU_FARM=1 from its env — the
#   sidecar would soak the CPUs this job needs for its OWN compile/score arms.
# MAX_HOURS=0: the JOB (jobd timeout_s + supervise budget) owns the box lifetime;
#   the serve must never self-park/destruct out from under a running eval.
# VLLM_USE_DEEP_GEMM=0: sm_120 fp8 DeepGEMM warmup crash ("Unknown recipe").
#   SCOPE: this serve path with an fp8 KV cache — NOT sm_120 in general. Measured
#   2026-07-29 (box 46193810, RTX 5090, vLLM 0.24.0 + torch 2.11.0+cu130): the
#   in-process bf16 gen lane runs a full wave with DeepGEMM ENABLED
#   ("DeepGEMM PDL enabled on vllm.third_party.deep_gemm", FLASH_ATTN backend).
#   Do not propagate this flag to bf16 gen jobs on Blackwell.
# Sampler: NOT pinned here — serve_vllm.sh defaults it to 0 itself (flashinfer
#   measured no win at <=8-way, readout 119) and honors an explicit
#   caller-provided VLLM_USE_FLASHINFER_SAMPLER, inherited by the child below.
# SERVE_MTP=0 unless the BUNDLE says otherwise: serve_vllm.sh flipped MTP on by
#   default on 2026-08-27 (owner directive, big measured win — see its header),
#   but that directive is about NEW serve runs launched through launch_serve.sh.
#   This path is how a jobs-v2 bundle stands up its endpoint, and several of
#   those bundles are terms in FROZEN comparisons whose numbers were banked on
#   the OFF cohort. MTP is not output-identical, so inheriting the flip here
#   would move a comparand nobody chose to move. `:-0` not `=0`: a bundle opts
#   in with one line of `env:` (`SERVE_MTP: auto`), which is where the choice
#   belongs.
echo ">> job_serve: starting serve_vllm.sh (log: $SERVE_LOG)"
CPU_FARM=0 MAX_HOURS=0 \
SERVE_MTP="${SERVE_MTP:-0}" \
VLLM_USE_DEEP_GEMM=0 \
  setsid bash "$SERVE_VLLM_SH" >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!
disown || true

# --- 4. box-local readiness gate (no tunnel) ---------------------------------
# Liveness is gated on $SERVE_PID, NOT `pgrep 'vllm serve'`. serve_vllm.sh pulls
# the base model + adapter (multi-GB, MINUTES on a cold box) BEFORE it `exec`s
# `vllm serve` in place — so during the pull there is no 'vllm serve' process yet,
# and a pgrep-based death check false-positives ("serve died" mid-pull; observed
# box 44612403, 2026-07-12). Because serve_vllm.sh execs the engine, $SERVE_PID
# tracks the launcher THROUGH the pull and INTO the running engine; only a dead
# $SERVE_PID means the stack actually exited.
auth=(); [ -n "${VLLM_API_KEY:-}" ] && auth=(-H "Authorization: Bearer ${VLLM_API_KEY}")
echo ">> job_serve: waiting up to ${READY_TIMEOUT}s for http://127.0.0.1:8000/v1/models (serve pid $SERVE_PID)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if curl -fsS "${auth[@]}" http://127.0.0.1:8000/v1/models -o /tmp/job_serve_models.json 2>/dev/null; then
    ids="$(python3 -c 'import json;print(",".join(m["id"] for m in json.load(open("/tmp/job_serve_models.json")).get("data",[])))' 2>/dev/null || true)"
    echo ">> job_serve: READY — served: ${ids:-<parse-failed>}"
    exit 0
  fi
  # surface a hard serve death early instead of waiting out the full timeout
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    sleep 3
    # one last READY probe in case the engine bound the port as it exited-then-reforked
    if curl -fsS "${auth[@]}" http://127.0.0.1:8000/v1/models -o /tmp/job_serve_models.json 2>/dev/null; then
      echo ">> job_serve: READY (late)"; exit 0
    fi
    echo "!! job_serve: serve process $SERVE_PID exited before ready — serve died. Tail of $SERVE_LOG:" >&2
    tail -n 60 "$SERVE_LOG" >&2 2>/dev/null || true
    exit 4
  fi
  sleep 5
done
echo "!! job_serve: /v1/models not ready after ${READY_TIMEOUT}s. Tail of $SERVE_LOG:" >&2
tail -n 40 "$SERVE_LOG" >&2 2>/dev/null || true
exit 4
