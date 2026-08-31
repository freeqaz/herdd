#!/usr/bin/env bash
# onstart/fetch_eval_env.sh — reusable box-side ACQUISITION of the baked eval env.
#
# Pulls the pre-baked compile+score environment (env-<ver>.tar.zst) from B2 and
# unpacks it at the ONE baked-in prefix /workspace/eval — the toolchains
# (wibo/objdiff-cli/dtk binaries), the upstream-monorepo venv, and all three target
# repos ninja-built. This is the env-acquisition half of onstart/eval_sidecar.sh
# (~L181-293) factored out so a jobs-v2 entrypoint that owns its own run loop
# (it does NOT run batch_validate, does NOT self-teardown) can obtain the env and
# then drive its own eval (e.g. run_paired_eval.sh). eval_sidecar.sh is left
# untouched; this is purely additive.
#
# Idempotent: a no-op fast path returns immediately if /workspace/eval/env.sh is
# already present (warm box / job restart), so it is safe to call every attempt.
#
# Required env:
#   B2_KEY_ID B2_APPLICATION_KEY B2_BUCKET B2_S3_ENDPOINT [B2_REGION]
# Optional env:
#   EVAL_ENV_VER    env tarball version (default: rclone cat eval-env/LATEST)
#   EVAL_PREFIX     unpack prefix (MUST be /workspace/eval — refuses otherwise)
#
# Exit: 0 = /workspace/eval/env.sh present & venv importable; non-zero on failure.
# On success the CALLER should `source /workspace/eval/env.sh` in its own shell.
set -uo pipefail

EVAL_PREFIX="${EVAL_PREFIX:-/workspace/eval}"
SYS_PY=/usr/bin/python3.10; command -v "$SYS_PY" >/dev/null || SYS_PY=python3

fail() { echo "!! fetch_eval_env: $1" >&2; exit 1; }

# --- fast path: already unpacked ---------------------------------------------
if [ -f "$EVAL_PREFIX/env.sh" ]; then
  echo ">> fetch_eval_env: $EVAL_PREFIX/env.sh already present — skipping pull"
  exit 0
fi

: "${B2_BUCKET:?fetch_eval_env: B2_BUCKET required}"
: "${B2_KEY_ID:?fetch_eval_env: B2_KEY_ID required}"
: "${B2_APPLICATION_KEY:?fetch_eval_env: B2_APPLICATION_KEY required}"
: "${B2_S3_ENDPOINT:?fetch_eval_env: B2_S3_ENDPOINT required}"
B2="b2:${B2_BUCKET}"

# --- rclone remote (idempotent; copied from eval_sidecar.sh) ------------------
if ! command -v rclone >/dev/null; then
  # every link BOUNDED and re-verified: a blackholed mirror hung an unbounded
  # curl 11min at boot, and `curl|bash` returns bash's 0 so links 2-3 never ran.
  { curl -fsSL --connect-timeout 10 --max-time 90 -o /tmp/rclone.deb \
      https://downloads.rclone.org/rclone-current-linux-amd64.deb \
      && dpkg -i /tmp/rclone.deb >/dev/null 2>&1 && rm -f /tmp/rclone.deb \
      && command -v rclone >/dev/null 2>&1; } \
    || { curl -fsSL --connect-timeout 10 --max-time 90 -o /tmp/rclone-inst.sh \
        https://rclone.org/install.sh \
        && timeout 120 bash /tmp/rclone-inst.sh >/dev/null 2>&1 \
        && command -v rclone >/dev/null 2>&1; } \
    || { timeout 120 apt-get update -qq >/dev/null 2>&1 || true; \
        timeout 180 apt-get install -y -qq rclone >/dev/null 2>&1; } || true
fi
command -v rclone >/dev/null || fail "rclone install failed — cannot reach B2"
# Honour RCLONE_CONFIG: rclone itself reads it, so a writer that hardcodes $HOME
# writes one file while every later `rclone` call reads another.
RCONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
mkdir -p "$(dirname "$RCONF")"
if ! grep -qs '^\[b2\]' "$RCONF" 2>/dev/null; then
  cat > "$RCONF" <<EOF
[b2]
type = s3
provider = Other
access_key_id = ${B2_KEY_ID}
secret_access_key = ${B2_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
no_check_bucket = true
EOF
  chmod 600 "$RCONF"
fi

# --- resolve env version ------------------------------------------------------
VER="${EVAL_ENV_VER:-}"
if [ -z "$VER" ]; then
  VER="$(rclone cat "$B2/eval-env/LATEST" 2>/dev/null | tr -d '[:space:]')"
fi
[ -n "$VER" ] || fail "could not resolve EVAL_ENV_VER (eval-env/LATEST empty?)"
echo ">> fetch_eval_env: env version $VER"

TARBALL="/workspace/env-${VER}.tar.zst"
MANIFEST="/workspace/env-${VER}.MANIFEST.json"
# TUNED TRANSPORT for the ONE multi-GB tarball: a single-flow copyto crawls on a
# per-flow-shaped host (1-16 MB/s/flow, vast-per-flow-image-layering). For ONE
# big file --multi-thread-streams is the lever — it splits the object into N
# ranged GETs — so the NIC saturates instead of a lone shaped flow (mirrors
# onstart/train.sh's RC_FAST 16/16/64M). --stats writes a live one-line MB/s to
# the per-tarball stats file for inspectability. Flags trail the src/dst so the
# local-bucket test shim (positional copyto) is undisturbed. All env-overridable.
# b2x first: the tuned rclone spelling below asks for 16 streams but rclone caps
# them at ceil(size/64MiB) = 9 for this ~555 MB tarball (measured, rclone
# 1.74.4). b2x computes 66 parts from the object size instead. It writes the same
# live-MB/s stats file the callers inspect. Falls back to the original line.
_EE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
[ -f "$_EE_DIR/b2x_boot.sh" ] && . "$_EE_DIR/b2x_boot.sh"
command -v b2x_pull >/dev/null 2>&1 || b2x_pull() { return 1; }

b2x_pull "$B2/eval-env/env-${VER}.tar.zst" "$TARBALL" 2>"/workspace/env-${VER}.pull.stats" \
  || rclone copyto "$B2/eval-env/env-${VER}.tar.zst" "$TARBALL" \
  --transfers "${EVAL_ENV_TRANSFERS:-16}" \
  --multi-thread-streams "${EVAL_ENV_STREAMS:-16}" \
  --multi-thread-cutoff 64M \
  --stats 30s --stats-one-line --stats-log-level NOTICE \
  2>"/workspace/env-${VER}.pull.stats" || fail "pull tarball failed"
rclone copyto "$B2/eval-env/env-${VER}.MANIFEST.json" "$MANIFEST" 2>/dev/null || fail "pull manifest failed"

# --- verify sha256 against the manifest --------------------------------------
WANT_SHA="$("$SYS_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tar_sha256"])' "$MANIFEST" 2>/dev/null || true)"
[ -n "$WANT_SHA" ] || fail "manifest missing tar_sha256"
GOT_SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
[ "$WANT_SHA" = "$GOT_SHA" ] || fail "tarball sha256 mismatch (want $WANT_SHA got $GOT_SHA)"
echo ">> fetch_eval_env: sha256 OK ($GOT_SHA)"

# --- unpack at the ONE baked-in prefix (SPEC MUST 7) --------------------------
# build.ninja + dc3's PCH embed absolute paths; a moved tree silently scores 0.0.
[ "$EVAL_PREFIX" = "/workspace/eval" ] || fail "EVAL_PREFIX=$EVAL_PREFIX — tarball paths bake to /workspace/eval; refusing"
command -v zstd >/dev/null || { echo ">> installing zstd"; timeout 120 apt-get update -qq || true; \
  timeout 180 apt-get install -y -qq zstd; }
command -v zstd >/dev/null || fail "zstd unavailable — cannot unpack"
echo ">> fetch_eval_env: unpacking $TARBALL -> /workspace (top-level eval/)"
zstd -dc "$TARBALL" | tar -C /workspace -xf - || fail "unpack failed"
[ -f "$EVAL_PREFIX/env.sh" ] || fail "$EVAL_PREFIX/env.sh missing after unpack"
rm -f "$TARBALL"   # multi-GB; reclaim disk

# --- venv self-heal: cross-image python mismatch (eval_sidecar.sh L272-293) ----
# The baked venv's cp310 C-extensions may not import on the box image's python;
# heal by installing the same deps into the box python + PYTHONPATH the tree.
# shellcheck disable=SC1091
source "$EVAL_PREFIX/env.sh"
if ! python3 -c 'import tree_sitter' 2>/dev/null; then
  echo ">> fetch_eval_env: baked venv unusable on this image ($(python3 -V 2>&1)) — self-healing"
  python3 -m pip install -q tree-sitter tree-sitter-cpp tree-sitter-c graphviz numpy 2>/dev/null \
    || python3 -m pip install -q --break-system-packages tree-sitter tree-sitter-cpp tree-sitter-c graphviz numpy \
    || echo "!! self-heal pip install failed"
  export PYTHONPATH="$EVAL_PREFIX/upstream-monorepo${PYTHONPATH:+:$PYTHONPATH}"
  python3 -c 'import tree_sitter, upstream_monorepo' 2>/dev/null \
    && echo ">> fetch_eval_env: self-heal OK" \
    || echo "!! fetch_eval_env: self-heal FAILED — tree_sitter/upstream_monorepo still not importable"
fi
echo ">> fetch_eval_env: env ready at $EVAL_PREFIX"
