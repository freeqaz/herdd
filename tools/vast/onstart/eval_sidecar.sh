#!/usr/bin/env bash
# onstart/eval_sidecar.sh — thin box-side runtime for upstream-monorepo remote evals.
#
# Runs compile+score evals on a rented box's idle CPU while the GPU trains. It
# pulls a pre-baked env tarball from B2 (built locally, zero builds on rented
# compute — see tools/vast/eval-env/SPEC.md), unpacks it at the ONE baked-in
# prefix /workspace/eval, runs the eval per target, and streams results back to
# b2:.../evals/<RUN_ID>/. It is deliberately best-effort: a target failure never
# aborts the others, and nothing here touches the training checkpoints/STATUS.
#
# Two run modes:
#   * co-tenant  — launched in the background by onstart/train.sh (EVAL_TARGETS
#     set). train.sh owns the box lifecycle; this script just runs+streams.
#   * standalone — EVAL_STANDALONE=1 (e.g. `herdd launch --onstart
#     eval_sidecar.sh`) for a smoke test with no training. It then self-destructs
#     the instance on completion (same CONTAINER_API_KEY mechanism as train.sh)
#     and arms its own MAX_HOURS watchdog since no train.sh watchdog exists.
#
# Resource fencing (SPEC MUST 16): the GPU trainer is the paying workload, so the
# sidecar re-execs under its own process group (killable as a tree), runs
# renice 19 + ionice -c3, sets oom_score_adj=800 on itself (children inherit, so
# the OOM killer prefers the sidecar over the trainer), and caps parallelism at
# EVAL_JOBS (default 4) — never derived from nproc.
#
# Required env:
#   RUN_ID                 results land under evals/<RUN_ID>/
#   EVAL_TARGETS           comma list of dc3|rb3|rb3-xenon
#   B2_KEY_ID B2_APPLICATION_KEY B2_BUCKET B2_S3_ENDPOINT [B2_REGION]
# Optional env:
#   EVAL_ENV_VER           env tarball version (default: rclone cat eval-env/LATEST)
#   EVAL_CMD               per-target command, CWD=target dir
#                          (default: python -m upstream_monorepo.batch_validate
#                                     --limit 25 --no-apply --json)
#   EVAL_JOBS              parallelism cap for the harness (default 4)
#   EVAL_STANDALONE=1      self-destruct + own watchdog (no train.sh present)
#   KEEP_ON_FAIL=1         standalone: stay up on failure for debugging
#   MAX_HOURS=N            standalone watchdog (default 6; 0 disables)
#   LLM_BASE_URL LLM_API_KEY OPENROUTER_API_KEY   passed through if present
set -uo pipefail

# --- self-fence: own process group + low priority (SPEC MUST 16) -------------
# Re-exec under setsid unless we are already a process-group leader (train.sh
# may already have launched us with setsid). Guard var stops an infinite loop
# on hosts where the pgid probe misbehaves.
if [ -z "${EVAL_SIDECAR_FENCED:-}" ]; then
  export EVAL_SIDECAR_FENCED=1
  _pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
  if [ -z "$_pgid" ] || [ "$_pgid" != "$$" ]; then
    exec setsid bash "$0" "$@"
  fi
fi
# yield stack: renice 19 + ionice -c3 + oom_score_adj 800, plus (NEW, best-effort)
# a low cgroup-v2 cpu.weight leaf so the trainer's dataloader/checkpoint threads
# always preempt the farm. Sourced from yield_fence.sh (the canonical stack, also
# used by farm_worker.sh + the dry-run) when present — train.sh pulls it next to
# this script. When absent (e.g. a bare standalone smoke) the inline fallback is
# byte-identical to the original three lines, so default behavior is unchanged.
_YF="$(dirname "$0")/yield_fence.sh"; [ -f "$_YF" ] || _YF=/workspace/yield_fence.sh
if [ -f "$_YF" ]; then
  # shellcheck source=/dev/null
  { . "$_YF" && yield_fence_self "eval-sidecar"; } || true
else
  renice 19 -p $$ >/dev/null 2>&1 || true
  ionice -c3 -p $$ >/dev/null 2>&1 || true
  echo 800 > "/proc/$$/oom_score_adj" 2>/dev/null || true
fi
# fixed (default, existing per-target eval) | farm (B2 work-unit compile loop)
EVAL_MODE="${EVAL_MODE:-fixed}"
export EVAL_JOBS="${EVAL_JOBS:-4}"
# total permuter parallelism cap consumed by upstream_monorepo.batch_validate —
# EVAL_JOBS alone only sets job-level fan-out; without this the harness sizes
# workers off nproc, which belongs to the trainer (SPEC MUST 16)
export UPSTREAM_MONOREPO_MAX_TOTAL_WORKERS="$EVAL_JOBS"
# Pin a SYSTEM interpreter, resolved the same way provision.sh does. PATH
# python3 is NOT it on either image: the axolotl lane's was a venv under
# /workspace, and t211's is /opt/train-bin/python3 (the trainer's). The old
# hardcoded /usr/bin/python3.10 fell back to bare `python3` on t211 — i.e.
# straight into the train venv — which is exactly what this is meant to avoid.
SYS_PY=""
for _c in /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3; do
  [ -x "$_c" ] || continue
  SYS_PY="$_c"; break
done
[ -n "$SYS_PY" ] || SYS_PY=python3   # last resort; better than an empty command
unset _c

mkdir -p /workspace/eval-out
exec > >(tee -a /workspace/eval_sidecar.log) 2>&1
echo "=== eval_sidecar $(date -u) RUN_ID=${RUN_ID:-?} TARGETS=${EVAL_TARGETS:-?} PID=$$ PGID=$(ps -o pgid= -p $$ | tr -d ' ') ==="

EVAL_CMD="${EVAL_CMD:-python -m upstream_monorepo.batch_validate --limit 25 --no-apply --json}"
EVAL_PREFIX="${EVAL_PREFIX:-/workspace/eval}"
[ -n "${B2_BUCKET:-}" ] && B2="b2:${B2_BUCKET}" || B2=""
B2W="$B2"               # write-side remote; may be re-pointed by the write gate
STATUS_LINE=""          # final EVAL_STATUS payload, set before exit
PUSH_PID=""

# self-teardown via the per-instance key Vast injects into every container
# (CONTAINER_API_KEY may only manage this one instance). Mirrors onstart/train.sh
# — standalone mode has no train.sh to tear the box down. Default action is PARK
# (2026-07-10 suspend-by-default): GPU billing ends, disk kept for `herdd
# start`; TEARDOWN=destroy restores the old self-destruct; a park that doesn't
# take within 180s falls back to destroy. NOTE: resuming a parked standalone
# sidecar box re-runs onstart and thus re-runs the eval (results are already on
# B2; the re-run is wasteful, not corrupting).
_iid_key() {
  IID="${INSTANCE_ID:-${CONTAINER_ID:-}}"; KEY="${VASTAI_API_KEY:-${CONTAINER_API_KEY:-}}"
  [ -z "$KEY" ] && [ -f ~/.vast_api_key ] && KEY="$(cat ~/.vast_api_key)"
  [ -n "$IID" ] && [ -n "$KEY" ]
}
self_destruct() {
  _iid_key || { echo "!! no iid/key — destroy manually: herdd destroy <id> -y"; return 1; }
  echo ">> self-destruct instance ${IID}"
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X DELETE -H "Authorization: Bearer ${KEY}" \
    "https://console.vast.ai/api/v0/instances/${IID}/" || true
}
self_park() {
  _iid_key || { echo "!! no iid/key — park manually: herdd stop <id>"; return 1; }
  echo ">> self-park instance ${IID} (resume: herdd start ${IID}; disk bills until destroyed)"
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X PUT -H "Authorization: Bearer ${KEY}" -H 'Content-Type: application/json' \
    -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/${IID}/" || true
  sleep 180
  echo "!! park did not take within 180s — self-destructing"
  self_destruct
}
self_teardown() {
  case "${TEARDOWN:-park}" in destroy) self_destruct ;; keep) : ;; *) self_park ;; esac
}

eval_status() { echo "$1" | rclone rcat "$B2W/evals/${RUN_ID}/EVAL_STATUS" 2>/dev/null || true; }

# final flush + status write on ANY exit path (torn-write mitigation lives in
# the periodic pusher; here we do a plain full copy so nothing is lost).
finish() {
  local ec=$?
  [ -n "$PUSH_PID" ] && kill "$PUSH_PID" 2>/dev/null || true
  if [ -n "${B2:-}" ] && [ -n "${RUN_ID:-}" ] && command -v rclone >/dev/null 2>&1; then
    rclone copy --fast-list /workspace/eval-out "$B2W/evals/${RUN_ID}" 2>/dev/null || true
    rclone rcat "$B2W/evals/${RUN_ID}/eval_sidecar.log" < /workspace/eval_sidecar.log 2>/dev/null || true
    eval_status "${STATUS_LINE:-FAILED (aborted rc=$ec) $(date -u +%FT%TZ)}"
  fi
  if [ "${EVAL_STANDALONE:-0}" = "1" ]; then
    if [ "${EVAL_RC:-1}" -eq 0 ] || [ "${KEEP_ON_FAIL:-0}" != "1" ]; then
      [ -n "${WATCHDOG:-}" ] && kill "$WATCHDOG" 2>/dev/null || true
      self_teardown
    else
      # watchdog stays ARMED: even a forgotten KEEP_ON_FAIL debug box dies at
      # MAX_HOURS (same policy as train.sh)
      echo ">> KEEP_ON_FAIL=1 — standalone box left RUNNING (billing!) for debugging; park/destroy yourself."
      [ -n "${WATCHDOG:-}" ] && echo ">> (watchdog still armed: parks at MAX_HOURS=${MAX_HOURS:-6}h)"
    fi
  fi
  echo "=== eval_sidecar done $(date -u) ==="
}
trap finish EXIT
# route the group-kill from train.sh's teardown through the EXIT trap so the
# final flush + EVAL_STATUS write still happen (bash does not run EXIT on an
# untrapped signal).
trap 'exit 143' TERM INT

fail() { STATUS_LINE="FAILED ($1) $(date -u +%FT%TZ)"; echo "!! $1"; exit 1; }

# --- standalone watchdog (SPEC MUST 17: no train.sh watchdog in this mode) ----
if [ "${EVAL_STANDALONE:-0}" = "1" ]; then
  MAX_HOURS="${MAX_HOURS:-6}"
  if [ "$MAX_HOURS" != "0" ]; then
    ( sleep "$(( MAX_HOURS * 3600 ))"
      echo "!! MAX_HOURS=${MAX_HOURS} exceeded — teardown (${TEARDOWN:-park})"
      if [ "${TEARDOWN:-park}" = destroy ]; then self_destruct; else self_park; fi ) & WATCHDOG=$!
    echo ">> standalone watchdog pid $WATCHDOG (${TEARDOWN:-park} after ${MAX_HOURS}h)"
  fi
fi

# --- required env ------------------------------------------------------------
# Checked only now, AFTER the EXIT trap + standalone watchdog are armed: a
# typo'd --env on a standalone launch must still self-destruct the box, not
# exit silently and bill forever.
: "${RUN_ID:?eval_sidecar: RUN_ID required}"
# farm mode takes its targets from the B2 manifest (per unit) and saturate mode
# self-selects from the baked artifacts.db, so neither needs EVAL_TARGETS; fixed mode
# still requires it.
[ "$EVAL_MODE" = farm ] || [ "$EVAL_MODE" = saturate ] \
  || : "${EVAL_TARGETS:?eval_sidecar: EVAL_TARGETS required}"
: "${B2_BUCKET:?eval_sidecar: B2_BUCKET required}"
: "${B2_KEY_ID:?eval_sidecar: B2_KEY_ID required}"
: "${B2_APPLICATION_KEY:?eval_sidecar: B2_APPLICATION_KEY required}"
: "${B2_S3_ENDPOINT:?eval_sidecar: B2_S3_ENDPOINT required}"
B2="b2:${B2_BUCKET}"
B2W="$B2"

# --- rclone remote (idempotent; sidecar may run before/without train.sh) ------
rclone_bootstrap() {
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
}
command -v rclone >/dev/null || rclone_bootstrap
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

# --- b2x transport (optional; every use falls back to its rclone line) --------
_ES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
[ -f "$_ES_DIR/b2x_boot.sh" ] && . "$_ES_DIR/b2x_boot.sh"
command -v b2x_pull >/dev/null 2>&1 || { b2x_pull() { return 1; }; b2x_push() { return 1; }; }

# --- write gate: pick a remote that can actually land evals/<RUN_ID>/ ---------
# launch_serve.sh ships a mint-pair by default: [b2] is built from a bucket-wide
# READ-ONLY key and the write key (B2_WRITE_*) is scoped to serve/. Every write
# this sidecar makes targets evals/<RUN_ID>/ — with the RO key each one 403s,
# and they are all `|| true`-swallowed, so a whole session's farm output used
# to vanish with no error anywhere. Probe the real first write (EVAL_STATUS)
# through [b2], then [b2w] when shipped, and refuse LOUDLY when neither can
# write: burning the CPU farm all session with unshippable output is strictly
# worse than exiting (co-tenant: serve/train unaffected; standalone: fail()'s
# exit still runs the finish() teardown so the box never bills forever).
# BEGIN b2-write-gate (sourced by test_eval_sidecar_write_gate.py — keep markers)
b2w_remote_config() {  # idempotent [b2w] from B2_WRITE_* (same shape as serve_vllm.sh)
  [ -n "${B2_WRITE_KEY_ID:-}" ] && [ -n "${B2_WRITE_APPLICATION_KEY:-}" ] || return 1
  # Resolved here rather than from the preamble's $RCONF: this block is extracted
  # between the markers and sourced standalone under `set -u` by its test.
  local rc="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
  grep -qs '^\[b2w\]' "$rc" 2>/dev/null && return 0
  cat >> "$rc" <<EOF

[b2w]
type = s3
provider = Other
access_key_id = ${B2_WRITE_KEY_ID}
secret_access_key = ${B2_WRITE_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
no_check_bucket = true
EOF
  chmod 600 "$rc"
}
# the probe IS the initial "RUNNING" status write (same payload as before) —
# no litter object, and rclone's stderr stays visible in eval_sidecar.log.
probe_status_write() {
  echo "RUNNING $(date -u +%FT%TZ)" | rclone rcat "$1:${B2_BUCKET}/evals/${RUN_ID}/EVAL_STATUS"
}
pick_write_remote() {  # sets B2W; retries so a transient flake is not a 403
  local i
  for i in 1 2 3; do
    if probe_status_write b2; then B2W="b2:${B2_BUCKET}"; return 0; fi
    if b2w_remote_config && probe_status_write b2w; then
      B2W="b2w:${B2_BUCKET}"
      echo ">> [b2] cannot write evals/ — writes routed via [b2w]"
      return 0
    fi
    echo "!! evals/ write probe failed (attempt $i/3)"
    [ "$i" -lt 3 ] && sleep "$((i * ${B2_PROBE_BACKOFF:-15}))"
  done
  return 1
}
# END b2-write-gate
pick_write_remote \
  || fail "cannot write $B2/evals/${RUN_ID}/ with the shipped B2 key(s) — read-only pair key? (serve write keys are scoped to serve/, not evals/) — refusing to run: every result would be lost silently"

# --- resolve env version -----------------------------------------------------
VER="${EVAL_ENV_VER:-}"
if [ -z "$VER" ]; then
  VER="$(rclone cat "$B2/eval-env/LATEST" 2>/dev/null | tr -d '[:space:]')"
fi
[ -n "$VER" ] || fail "could not resolve EVAL_ENV_VER (eval-env/LATEST empty?)"
echo ">> env version: $VER"

TARBALL="/workspace/env-${VER}.tar.zst"
MANIFEST="/workspace/env-${VER}.MANIFEST.json"
echo ">> pulling env tarball + manifest from $B2/eval-env/"
# The multi-GB env tarball. This was on STOCK rclone (4 flows) AND silenced with
# 2>/dev/null, so a slow pull was invisible; fetch_eval_env.sh pulls the SAME
# artifact with tuned flags, which is exactly the per-site drift b2x removes.
# (Even the tuned spelling only reached 9 flows — see tools/vast/B2X_DESIGN.md.)
b2x_pull "$B2/eval-env/env-${VER}.tar.zst" "$TARBALL" \
  || rclone copyto "$B2/eval-env/env-${VER}.tar.zst" "$TARBALL" 2>/dev/null \
  || fail "pull tarball failed"
rclone copyto "$B2/eval-env/env-${VER}.MANIFEST.json" "$MANIFEST" 2>/dev/null || fail "pull manifest failed"

# --- verify sha256 against the manifest (jq may be absent — use python3) ------
WANT_SHA="$("$SYS_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tar_sha256"])' "$MANIFEST" 2>/dev/null || true)"
[ -n "$WANT_SHA" ] || fail "manifest missing tar_sha256"
GOT_SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
if [ "$WANT_SHA" != "$GOT_SHA" ]; then
  fail "tarball sha256 mismatch (want $WANT_SHA got $GOT_SHA)"
fi
echo ">> sha256 OK ($GOT_SHA)"

# best-effort image_digest cross-check (podman writes /run/.containerenv). We
# only WARN — the box image and bake image are pinned by IMAGE, but a live
# digest is not reliably readable inside a rootless container.
MANIFEST_IMG="$("$SYS_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("image_digest",""))' "$MANIFEST" 2>/dev/null || true)"
if [ -n "$MANIFEST_IMG" ] && [ -r /run/.containerenv ]; then
  RUN_IMG="$(sed -n 's/^image="\(.*\)"$/\1/p' /run/.containerenv 2>/dev/null || true)"
  if [ -n "$RUN_IMG" ] && [ "$RUN_IMG" != "$MANIFEST_IMG" ]; then
    echo "!! WARN: running image ($RUN_IMG) != manifest image_digest ($MANIFEST_IMG) — proceeding best-effort"
  fi
fi

# --- unpack at the ONE baked-in prefix (SPEC MUST 7) --------------------------
# build.ninja + dc3's PCH embed absolute paths; a moved tree silently scores 0.0
# on ~42% of dc3 TUs. Refuse anything but /workspace/eval.
[ "$EVAL_PREFIX" = "/workspace/eval" ] || fail "EVAL_PREFIX=$EVAL_PREFIX — tarball paths bake to /workspace/eval; refusing"
command -v zstd >/dev/null || { echo ">> installing zstd"; timeout 120 apt-get update -qq || true; \
  timeout 180 apt-get install -y -qq zstd; }
command -v zstd >/dev/null || fail "zstd unavailable — cannot unpack"
echo ">> unpacking $TARBALL -> /workspace (top-level eval/)"
zstd -dc "$TARBALL" | tar -C /workspace -xf - || fail "unpack failed"
[ -f "$EVAL_PREFIX/env.sh" ] || fail "$EVAL_PREFIX/env.sh missing after unpack"
rm -f "$TARBALL"   # multi-GB; the unpacked env is what we need — reclaim disk

# copy the manifest into the results dir for attribution (SPEC MUST 18)
rclone rcat "$B2W/evals/${RUN_ID}/env-${VER}.MANIFEST.json" < "$MANIFEST" 2>/dev/null || true

# --- background streaming loop (SPEC MUST 18) --------------------------------
# --min-age 45s skips files the harness is mid-writing (sqlite/json torn-write
# window), same mitigation as train.sh's checkpoint watcher.
( while true; do
    rclone copy --fast-list --min-age 45s /workspace/eval-out "$B2W/evals/${RUN_ID}" 2>/dev/null \
      && echo "[eval-sync] pushed $(date -u +%T)" || echo "[eval-sync] push FAILED $(date -u +%T) (retry)"
    sleep 120
  done ) & PUSH_PID=$!
echo ">> results watcher pid $PUSH_PID -> $B2W/evals/${RUN_ID}/"

# --- run the eval per target (a failure never aborts the rest) ----------------
# shellcheck disable=SC1091
source "$EVAL_PREFIX/env.sh"
echo ">> sourced env.sh; EVAL_JOBS=$EVAL_JOBS; EVAL_CMD=$EVAL_CMD"

# --- venv self-heal: cross-image python mismatch ------------------------------
# The baked venv's bin/python3 symlinks the BAKE image's interpreter (e.g.
# /usr/bin/python3.10). On a box whose image ships a different python (observed
# live 2026-07-10: the torch-2.10 train image is py3.12-only — the symlink
# dangled, PATH lookup fell through to system python, and saturate died on
# `import tree_sitter` within a minute of launch), the whole baked venv is
# unusable: even re-pointing the symlink can't load its cp310 C extensions.
# Self-heal instead of dying: install the wheels into the box python and put
# the upstream-monorepo tree on PYTHONPATH — same dep set provision.sh installs.
# One pip fetch (~30s) on mismatched images; a no-op wherever the venv works.
if ! python3 -c 'import tree_sitter' 2>/dev/null; then
  echo ">> baked venv unusable on this image (python3=$(command -v python3), $(python3 -V 2>&1)) — self-healing"
  python3 -m pip install -q tree-sitter tree-sitter-cpp tree-sitter-c graphviz numpy 2>/dev/null \
    || python3 -m pip install -q --break-system-packages tree-sitter tree-sitter-cpp tree-sitter-c graphviz numpy \
    || echo "!! self-heal pip install failed"
  export PYTHONPATH="$EVAL_PREFIX/upstream-monorepo${PYTHONPATH:+:$PYTHONPATH}"
  if python3 -c 'import tree_sitter, upstream_monorepo' 2>/dev/null; then
    echo ">> self-heal OK ($(python3 -V 2>&1))"
  else
    echo "!! self-heal FAILED — tree_sitter/upstream_monorepo still not importable (harness will fail)"
  fi
fi

# --- farm mode: run the B2 work-unit compile loop instead of the fixed targets -
# Reuses everything above (fence, verified/unpacked env, background streamer) and
# delegates to farm_worker.sh (pulls farm/<RUN_ID>/inbox manifest, runs the same
# score_real_anchor/crack_live oracle, pushes farm/<RUN_ID>/results incrementally,
# resumable+idempotent). The EXIT trap finish() handles the final flush +
# EVAL_STATUS + (standalone) self-destruct, same as the fixed path.
if [ "$EVAL_MODE" = farm ]; then
  FARM="$(dirname "$0")/farm_worker.sh"; [ -f "$FARM" ] || FARM=/workspace/farm_worker.sh
  [ -f "$FARM" ] || FARM="$EVAL_PREFIX/upstream-monorepo/tools/vast/onstart/farm_worker.sh"
  if [ ! -f "$FARM" ]; then
    STATUS_LINE="FAILED (farm_worker.sh not found) $(date -u +%FT%TZ)"; echo "!! $STATUS_LINE"
    EVAL_RC=1; exit "$EVAL_RC"
  fi
  echo ">> === farm mode: $FARM (FARM_RUN_ID=${FARM_RUN_ID:-$RUN_ID}) ==="
  bash "$FARM" > /workspace/eval-out/farm_summary.json 2> >(tee -a /workspace/eval-out/farm.log >&2)
  EVAL_RC=${PIPESTATUS[0]}
  if [ "$EVAL_RC" -eq 0 ]; then
    STATUS_LINE="DONE (farm $(tr -d '\n' < /workspace/eval-out/farm_summary.json)) $(date -u +%FT%TZ)"
  else
    STATUS_LINE="FAILED (farm rc=$EVAL_RC) $(date -u +%FT%TZ)"
  fi
  echo ">> $STATUS_LINE"
  exit "$EVAL_RC"
fi

# --- saturate mode: run the manifest-FREE continuous crack-farm saturator -------
# Like farm mode, reuses the whole prelude (fence, verified/unpacked env, the
# background streamer that ships /workspace/eval-out -> evals/<RUN_ID>/). Unlike
# farm mode, needs NO B2 manifest: saturator.py self-selects the near-miss frontier
# from the baked artifacts.db and runs an unbounded (function x pattern-config) search.
# It writes NDJSON shards + SAT_STATUS.json into /workspace/eval-out/corpus so the
# streamer delivers them. Runs in the FOREGROUND so this sidecar stays alive (and
# the streamer keeps pushing) for the life of the box; the box's lifetime is the
# training job's — no self-destruct (co-tenant; EVAL_STANDALONE unset).
if [ "$EVAL_MODE" = saturate ]; then
  SAT="$(dirname "$0")/saturator.py"; [ -f "$SAT" ] || SAT=/workspace/saturator.py
  [ -f "$SAT" ] || SAT="$EVAL_PREFIX/upstream-monorepo/tools/vast/onstart/saturator.py"
  if [ ! -f "$SAT" ]; then
    STATUS_LINE="FAILED (saturator.py not found) $(date -u +%FT%TZ)"; echo "!! $STATUS_LINE"
    EVAL_RC=1; exit "$EVAL_RC"
  fi
  SAT_WORKERS="${SAT_WORKERS:-$(python3 -c 'import os;print(max(1,round(0.7*(os.cpu_count() or 8))))')}"
  mkdir -p /workspace/eval-out/corpus
  echo ">> === saturate mode: $SAT workers=$SAT_WORKERS (FARM_RUN_ID=${FARM_RUN_ID:-$RUN_ID}) ==="
  SAT_OUTBOX="${SAT_OUTBOX:-/workspace/eval-out/corpus}" \
  SAT_HOST="${SAT_HOST:-${FARM_RUN_ID:-$RUN_ID}}" SAT_WORKERS="$SAT_WORKERS" \
    python3 "$SAT" > /workspace/eval-out/saturate.log 2> >(tee -a /workspace/eval-out/saturate.err >&2)
  EVAL_RC=${PIPESTATUS[0]}
  STATUS_LINE="DONE (saturate rc=$EVAL_RC) $(date -u +%FT%TZ)"
  echo ">> $STATUS_LINE"
  exit "$EVAL_RC"
fi

declare -A RC_MAP
IFS=',' read -ra _TARGETS <<< "$EVAL_TARGETS"
for t in "${_TARGETS[@]}"; do
  t="$(echo "$t" | tr -d '[:space:]')"
  [ -z "$t" ] && continue
  tdir="$EVAL_PREFIX/$t"
  outdir="/workspace/eval-out/$t"
  mkdir -p "$outdir"
  if [ ! -d "$tdir" ]; then
    echo "!! target dir missing: $tdir (not in this env build?)" | tee -a "$outdir/eval.log"
    RC_MAP["$t"]=127
    continue
  fi
  echo ">> === eval target: $t ($(date -u +%FT%TZ)) ===" | tee -a "$outdir/eval.log"
  # stdout (batch_validate --json) -> eval.log AND result.json; stderr -> eval.log.
  # PIPESTATUS[0] is the harness rc, unaffected by the tee tail.
  ( cd "$tdir" && eval "$EVAL_CMD" ) \
    2> >(tee -a "$outdir/eval.log" >&2) \
    | tee -a "$outdir/eval.log" > "$outdir/result.json"
  rc=${PIPESTATUS[0]}
  RC_MAP["$t"]=$rc
  echo ">> target $t exited rc=$rc" | tee -a "$outdir/eval.log"
done

# --- summarize + final status ------------------------------------------------
EVAL_RC=0
summary=""
for t in "${!RC_MAP[@]}"; do
  summary+="$t=${RC_MAP[$t]} "
  [ "${RC_MAP[$t]}" -eq 0 ] || EVAL_RC=1
done
summary="$(echo "$summary" | sed 's/[[:space:]]*$//')"
if [ "$EVAL_RC" -eq 0 ]; then
  STATUS_LINE="DONE ($summary) $(date -u +%FT%TZ)"
else
  STATUS_LINE="FAILED ($summary) $(date -u +%FT%TZ)"
fi
echo ">> $STATUS_LINE"

# trap finish() handles the final flush, EVAL_STATUS write, and (standalone)
# self-destruct. Exit non-zero on any target failure for onstart visibility.
exit "$EVAL_RC"
