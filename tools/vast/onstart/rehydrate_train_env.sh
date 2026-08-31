#!/usr/bin/env bash
# rehydrate_train_env.sh — box-side fast-boot rehydrate for the TRAINING env.
#
# Pulled from b2:.../train-env/rehydrate.sh by onstart/train.sh (kept out of the
# inline onstart, which is near Vast's 16 KB arg cap — same pattern as the eval
# sidecar). On the SLIM base image (tools/vast/train-env/IMAGE) the incremental
# training stack is NOT baked in; this pulls the pre-baked tarball from FAST B2
# (built by tools/vast/train-env/bake.sh; the base already ships python3.11 +
# torch cu128 + cudnn), sha256-verifies it against the manifest, unpacks it at
# the baked prefix /workspace/train-env, and — on success — writes the venv
# activate path to /workspace/.train_env_activate for train.sh to source.
#
# Best-effort: any failure logs + exits non-zero and train.sh proceeds on the
# runset preflight's pip (SAFE-FALLBACK). Requires: rclone already configured (a
# [b2] remote), B2_BUCKET, and optional TRAIN_ENV_VER (default: train-env/LATEST).
set -uo pipefail

# Baked-image fast path: if the env is already present (prebaked into the launch
# image by train-env/Dockerfile), there is nothing to rehydrate — write the
# activate marker and exit 0 before touching B2 (a baked-image box needs no B2
# creds). See tools/vast/train-env/bake.sh `image`.
if [ -f /workspace/train-env/env.sh ]; then
  echo /workspace/train-env/env.sh > /workspace/.train_env_activate
  echo ">> FAST_BOOT: train env already baked into image — skipping B2 rehydrate"
  exit 0
fi

B2="b2:${B2_BUCKET:?B2_BUCKET required}"

ver="${TRAIN_ENV_VER:-}"
[ -n "$ver" ] || ver="$(rclone cat "$B2/train-env/LATEST" 2>/dev/null | tr -d '[:space:]')"
[ -n "$ver" ] || { echo "!! FAST_BOOT: no train-env/LATEST — falling back to preflight deps"; exit 1; }
echo ">> FAST_BOOT: rehydrating train env $ver"

tb="/workspace/te-${ver}.tar.zst"; mf="/workspace/te-${ver}.json"
# b2x transport (sibling; no-op stub when absent, so the rclone line below is
# unchanged). This was the worst-tuned bulk site in the repo: 8 streams and no
# --transfers at all, on a multi-GB tarball. One object means rclone's
# --transfers is a no-op and the 64M chunk default caps it at 8 flows, while
# b2x plans parts from the object size.
_B2XD="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
[ -f "$_B2XD/b2x_boot.sh" ] && . "$_B2XD/b2x_boot.sh"
command -v b2x_pull >/dev/null 2>&1 || b2x_pull() { return 1; }

b2x_pull "$B2/train-env/env-${ver}.tar.zst" "$tb" 2>/dev/null \
  || rclone copyto --multi-thread-streams "${RCLONE_STREAMS:-8}" --multi-thread-cutoff 64M \
  "$B2/train-env/env-${ver}.tar.zst"       "$tb" 2>/dev/null || { echo "!! FAST_BOOT: pull tarball failed"; exit 1; }
rclone copyto "$B2/train-env/env-${ver}.MANIFEST.json" "$mf" 2>/dev/null || { echo "!! FAST_BOOT: pull manifest failed"; exit 1; }

# sha256 vs manifest (partial/torn-upload guard) before unpacking
want="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["tar_sha256"])' "$mf" 2>/dev/null || true)"
got="$(sha256sum "$tb" | awk '{print $1}')"
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "!! FAST_BOOT: sha256 mismatch (want ${want:-?} got $got)"; exit 1; }
echo ">> FAST_BOOT: sha256 OK"

command -v zstd >/dev/null || { timeout 120 apt-get update -qq >/dev/null 2>&1 || true; \
  timeout 180 apt-get install -y -qq zstd >/dev/null 2>&1 || true; }
command -v zstd >/dev/null || { echo "!! FAST_BOOT: zstd unavailable"; exit 1; }
# the venv bakes absolute /workspace/train-env paths — unpack there and nowhere else
zstd -dc "$tb" | tar -C /workspace -xf - || { echo "!! FAST_BOOT: unpack failed"; exit 1; }
[ -f /workspace/train-env/env.sh ] || { echo "!! FAST_BOOT: env.sh missing after unpack"; exit 1; }
rm -f "$tb"

# C compiler for triton's runtime JIT: bitsandbytes>=0.49 imports triton kernels
# that compile a cuda_utils helper on first GPU use, needing cc/gcc. The pytorch
# *runtime* base ships no compiler (only the *devel* image does), so `import
# bitsandbytes` hard-fails on a GPU box with "Failed to find C compiler" — the
# axolotl image had gcc, so this is fast-boot-specific. ~30-60s apt, one-time,
# still far cheaper than the image-pull tax we removed. Best-effort: a miss only
# resurfaces the original error, it does not corrupt anything.
if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
  echo ">> FAST_BOOT: installing build-essential (triton needs a C compiler at runtime)"
  { DEBIAN_FRONTEND=noninteractive timeout 120 apt-get update -qq >/dev/null 2>&1 || true; \
    DEBIAN_FRONTEND=noninteractive timeout 300 apt-get install -y -qq build-essential >/dev/null 2>&1; } \
    && echo ">> FAST_BOOT: gcc $(gcc -dumpversion 2>/dev/null)" \
    || echo "!! FAST_BOOT: build-essential install failed — bitsandbytes/triton may error"
fi

echo /workspace/train-env/env.sh > /workspace/.train_env_activate
echo ">> FAST_BOOT: train env $ver ready"
