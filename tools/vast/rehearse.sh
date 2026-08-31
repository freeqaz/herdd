#!/usr/bin/env bash
# rehearse.sh — a one-shot LOCAL rehearsal of a job folder. No B2, no GPU, no
# money. It runs the REAL onstart/jobd.sh against a local-dir "bucket" (the
# rclone shim in testlib/rclone_shim.sh maps b2:<bucket>/<key> -> a tmpdir) so
# you can prove a job's config + entrypoint + results-glob wiring BEFORE spending
# a cent on a bid box. Same validation path as `herdd job submit`; same
# transport the pytest harness (test_jobd.py) uses — just outside pytest and
# reporting a human PASS/FAIL.
#
# WHAT THIS CERTIFIES (and what it does NOT): the rehearsal proves PLUMBING —
# job-folder validity, jobd claim/extract/needs/results wiring, asset staging,
# and entrypoint control-flow — NOT the live serve env. --stub-vllm starts
# eval_stub_server.py, never real vLLM, so a broken vllm/CUDA install or a stale
# LIVE-B2 asset (rehearsal reads LOCAL fixtures) is INVISIBLE here by design. The
# fix is PIN, DON'T SIMULATE — pin the box's runtime inputs (vllm version +
# PIP_CONSTRAINT, named image, submit-time B2-staleness check), don't make the
# stub more realistic. For the live vLLM env, run the on-box fail-fast serve
# smoke (EVAL_SERVE_SMOKE in runsets/base-reader-train/train.sh; see JOBS_DESIGN.md
# "What the local rehearsal certifies (and does NOT)").
#
# Usage:
#   tools/vast/rehearse.sh <job-folder> [--stub-vllm] [--stub-eval-env] \
#                          [--asset N=DIR ...] [--assets-fixture DIR] \
#                          [--image IMG] [--allow-image-fallback] [--keep]
#
# `--image` that cannot run (no podman, image absent — this script NEVER pulls)
# exits 4 RESULT: INCONCLUSIVE, because a local-lane PASS cannot speak for the
# box image. `--allow-image-fallback` accepts the local-lane-only result.
#
# Flags:
#   --stub-vllm         start eval_stub_server.py on 127.0.0.1:<stub-port> before
#                       jobd and kill it after, so eval-shaped entrypoints find an
#                       OpenAI endpoint (/v1/models, /v1/completions,
#                       /v1/chat/completions, /health).
#   --stub-eval-env     satisfy `needs.venv: eval` with testlib/fetch_eval_env_stub.sh
#                       (via jobd's JOBD_FETCH_EVAL_SH seam) instead of the real
#                       multi-GB credentialed B2 pull, which cannot succeed in a
#                       rehearsal. Writes only $JOBD_ROOT/eval/env.sh — NO game
#                       tree, NO compiler, NO objdiff-cli: an entrypoint that
#                       would touch the tree must take its own DRY_RUN path.
#   --stub-port N       stub vLLM port                             (default 8000)
#   --stub-models CSV   models advertised on /v1/models
#                       (default: job env STUB_MODELS, else stub-base,reader)
#   --asset N=DIR       seed the fake bucket prefix of asset `N` from local DIR
#                       BEFORE jobd runs, so the REAL asset-staging path (pull ->
#                       .complete marker -> require: postconditions) executes
#                       against the local bucket. Repeatable. Overrides the
#                       fixture convention below for that asset.
#   --assets-fixture DIR  base dir for the asset-fixture CONVENTION: an asset `N`
#                       with no --asset override is seeded from DIR/<N>/ when that
#                       dir exists (default: <job-folder>/assets-fixture).
#   --env K=V           override/add ONE job-config `env:` entry in the seeded
#                       TICKET, exactly like `herdd job submit --env` — for
#                       rehearsing a submit-time override (ARMS=w5b,
#                       CONC_SWEEP=8,16) without editing the bundle. Repeatable.
#                       Folded into the ticket rather than exported: a host
#                       export cannot win against `.job.env` for keys the
#                       job-config defines, so an export-based passthrough
#                       silently rehearses the WRONG env.
#   --image IMG         run jobd INSIDE podman with IMG, CPU-only, mounting the
#                       repo's tools/vast + the fake bucket. Skips gracefully when
#                       podman or the image is unavailable — NEVER pulls (a
#                       multi-GB pull in a rehearsal is never what you want).
#                       (default when bare --image given: DEFAULT_IMAGE below,
#                        which MUST equal herdd.yaml's default_image)
#   --concurrent N      seed N tickets for the SAME job folder instead of one, and
#                       let jobd claim them in a single poll pass (JOBD_CPU_SLOTS=N)
#                       so they run CONCURRENTLY against ONE workspace. This is the
#                       only lane that exercises BOX-GLOBAL MUTABLE STATE: the venv
#                       provisioners ($JOBD_ROOT/{serve,eval}), the content-addressed
#                       bundle cache, and the asset cache + its `.complete` markers
#                       (all N jobs share the same asset names). Every job must PASS.
#                       Motivating incident: 2026-07-30 frontier wave round 3 — three
#                       concurrent jobs raced job_serve.sh --build-venv into the shared
#                       /workspace/serve and the parallel pip self-upgrade corrupted
#                       pip (`ModuleNotFoundError: pip._internal` in 2 of 3 jobs). The
#                       single-job rehearsal could not see it: shared state needs
#                       CONTENTION, not realism, to break.
#   --keep              keep the tmp bucket/workspace for inspection.
#
# Exit: 0 PASS · 1 FAIL (bad config / entrypoint rc / missing results /
#       asset_stage_failed) · 2 usage.
#
# Rehearsal — assets end-to-end (the N6 workflow):
#   A job's `assets:` list names B2 prefixes that don't exist in a fresh fake
#   bucket, so rehearse SEEDS them first from local dirs, then lets the REAL
#   onstart/jobd.sh stage them exactly as a live box would:
#     1. author job-config with `assets: [{name, b2, require: [...], ...}]`;
#     2. put the bytes each asset should contain in a local dir;
#     3. rehearse:  tools/vast/rehearse.sh <job> --asset <name>=<local-dir>
#                   (or drop them under <job>/assets-fixture/<name>/);
#     4. jobd pulls b2:<bucket>/<prefix>/ (the shim maps it to the seeded dir),
#        writes the .complete byte-total marker, and enforces every `require:`
#        glob + index-aware shard completeness — the SAME code path as a bid box.
#   An asset MISSING from the fixture (or seeded TRUNCATED so a `require:` glob
#   or a *.index.json shard is absent) fails the rehearsal PRE-entrypoint with
#   the box-identical `asset_stage_failed:<name>` reason (surfaced from the job's
#   event log at [5/5]); no `started` event, no results. Pair with --stub-vllm
#   for a full eval-shaped run: assets staged -> entrypoint curls the stub ->
#   NDJSON + summary land under the results globs -> PASS.
#
# GPU-MODE FOOTGUN (for a future --image --gpu lane, NOT used here): podman has
# no nvidia CDI on our hosts — you must mount /dev/nvidia* + libcuda/libnvidia-ml
# AND run `ldconfig` IN-container, else torch imports but every triton JIT (liger)
# dies "libcuda.so cannot be found". CPU-only rehearsal (this script) sidesteps it
# entirely. Ref: docs/.../n5prime_sft_run/training-infra/119_SPEEDUP_DEFAULTS_LANDED_2026-07-12.md.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PY="${REHEARSE_PYTHON:-python3}"
JOBD_SH="$HERE/onstart/jobd.sh"
STUB_SRC="$HERE/runsets/modelzoo-reader/eval_stub_server.py"
SHIM_SRC="$HERE/testlib/rclone_shim.sh"
EVALSTUB_SRC="$HERE/testlib/fetch_eval_env_stub.sh"
IID="rehearse"

# ---- args -------------------------------------------------------------------
JOB_DIR=""; STUB=0; EVALSTUB=0; STUB_PORT=8000; STUB_MODELS=""; IMAGE=""; KEEP=0
IMAGE_FALLBACK_OK=0; IMAGE_SKIPPED=0
ASSETS_FIXTURE=""; CONCURRENT=1
declare -A ASSET_OVERRIDE=()    # asset name -> local seed dir (--asset N=DIR)
EXTRA_ENV=()                    # K=V ticket-env overrides (--env), submit-style
# The image the box actually runs (herdd.yaml default_image). Keep these in
# sync: rehearsal NEVER pulls, so a stale tag here does not error — it silently
# skips, turning the pre-spend preflight into a no-op. Flipped to the unified
# t211 lane 2026-08-01 with herdd.yaml + launch_serve.sh.
# MUST equal herdd.yaml's default_image — test_rehearse.py pins the pair.
# This drifted at the 2026-08-07 R2 cutover, which moved herdd.yaml to
# registry.example.com and left this on registry.gitlab.com, and the drift is
# SILENT in the direction that matters: if the named image is not present
# locally, rehearse.sh falls back to LOCAL jobd with a `>> SKIP --image` note
# rather than failing, so a rehearsal reports OK having never exercised the
# image the box will actually run. (Observed 2026-08-08 — a bare `--image`
# rehearsal ran the GitLab t211 image while the fleet default was the R2 one.)
DEFAULT_IMAGE="registry.example.com/train:latest"
# Print the header from line 2 through the "Exit:" block, addressed by CONTENT
# rather than a line number — a hardcoded range silently drops the newest flag
# from --help every time this header grows (fixed twice: 4541d3c3, and again
# when --concurrent landed).
usage() { sed -n '2,/^#       asset_stage_failed/p' "$0" >&2; exit "${1:-2}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --stub-vllm)   STUB=1; shift ;;
    --stub-eval-env) EVALSTUB=1; shift ;;
    --stub-port)   STUB_PORT="$2"; shift 2 ;;
    --stub-models) STUB_MODELS="$2"; shift 2 ;;
    --asset)       # --asset NAME=DIR : seed asset NAME's fake-bucket prefix from DIR
      case "${2:-}" in
        *=*) ASSET_OVERRIDE["${2%%=*}"]="${2#*=}"; shift 2 ;;
        *)   echo "!! --asset needs NAME=DIR (got ${2:-<empty>})" >&2; usage 2 ;;
      esac ;;
    --assets-fixture) ASSETS_FIXTURE="$2"; shift 2 ;;
    --env)
      case "${2:-}" in
        *=*) EXTRA_ENV+=("$2"); shift 2 ;;
        *)   echo "!! --env needs K=V (got ${2:-<empty>})" >&2; usage 2 ;;
      esac ;;
    --concurrent)
      case "${2:-}" in
        ''|*[!0-9]*) echo "!! --concurrent needs a positive integer (got ${2:-<empty>})" >&2; usage 2 ;;
      esac
      [ "$2" -ge 1 ] || { echo "!! --concurrent must be >= 1" >&2; usage 2; }
      CONCURRENT="$2"; shift 2 ;;
    --image)       # optional value: bare flag -> default image
      if [ $# -ge 2 ] && [ "${2#-}" = "$2" ]; then IMAGE="$2"; shift 2
      else IMAGE="$DEFAULT_IMAGE"; shift; fi ;;
    --allow-image-fallback) IMAGE_FALLBACK_OK=1; shift ;;
    --keep)        KEEP=1; shift ;;
    -h|--help)     usage 0 ;;
    -*)            echo "!! unknown flag: $1" >&2; usage 2 ;;
    *)             if [ -z "$JOB_DIR" ]; then JOB_DIR="$1"; else echo "!! unexpected arg: $1" >&2; usage 2; fi; shift ;;
  esac
done
[ -n "$JOB_DIR" ] || { echo "!! job-folder required" >&2; usage 2; }
[ -d "$JOB_DIR" ] || { echo "!! not a directory: $JOB_DIR" >&2; exit 2; }
JOB_DIR="$(cd "$JOB_DIR" && pwd)"
[ -n "$ASSETS_FIXTURE" ] || ASSETS_FIXTURE="$JOB_DIR/assets-fixture"
[ -f "$JOBD_SH" ] || { echo "!! jobd.sh not found: $JOBD_SH" >&2; exit 1; }
[ -f "$SHIM_SRC" ] || { echo "!! rclone shim not found: $SHIM_SRC" >&2; exit 1; }

echo "== REHEARSAL: $JOB_DIR =="

# ---- [1/5] validate config (same path as `herdd job submit`) --------------
# Also emits, on the last line, the entrypoint filename and the STUB_MODELS from
# the job's env block (so --stub-vllm can default its model list from the job).
META="$("$PY" - "$HERE" "$JOB_DIR" ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import jobmeta as jm
src = sys.argv[2]
raw = jm.load_job_config(src)
# --env folds HERE too, not just at ticket time ([3/5]): an `assets[].b2`
# carrying `${VAR}` is unresolvable without it, so a validation that skipped the
# overrides would fail a bundle the rehearsal is about to run correctly.
if sys.argv[3:]:
    env = dict(raw.get("env") or {})
    for kv in sys.argv[3:]:
        k, sep, v = kv.partition("=")
        if sep:
            env[k] = v
    raw["env"] = env
try:
    cfg, warns = jm.validate_job_config(raw, src)
except jm.JobmetaError as e:
    print("ERR " + str(e), file=sys.stderr)
    sys.exit(1)
for w in warns:
    print("warn: " + w, file=sys.stderr)
env = cfg.get("env") or {}
print("name=%s" % cfg["name"])
print("entrypoint=%s" % cfg["entrypoint"])
print("results=%s" % ",".join(cfg.get("results") or []))
print("stub_models=%s" % (env.get("STUB_MODELS") or ""))
# One `include=<name>` per shared file. The entrypoint check below needs it:
# jobmeta lets a bundle name an INCLUDED file as its entrypoint, so testing
# only the folder would fail a bundle that submits fine.
for n in (cfg.get("includes") or []):
    print("include=%s" % n)
# One `asset=<name>\t<b2prefix>\t<receipt-or-dash>` line per declared asset (all
# validated slugs/paths, no tab/newline — the TSV split below is unambiguous), so
# rehearse can seed each asset's fake-bucket prefix before jobd stages it.
# Optionality is NOT echoed: jobd enforces `optional`/`require` itself; rehearse
# only seeds. `receipt` IS, because seeding a prefix without its completeness
# marker seeds an UNPUBLISHED prefix — see the mint below.
for a in (cfg.get("assets") or []):
    print("asset=%s\t%s\t%s" % (a["name"], a["b2"], a.get("receipt") or "-"))
# B2 WRITE-SCOPE gate — the same seam `herdd job submit` / `jobmatrix submit`
# call (jm.b2_write_preflight). The rehearsal is DRY_RUN and never touches B2 by
# construction, so a publish stage's entitlement can ONLY be checked statically;
# without this, "RESULT: PASS" said nothing about the write that ended the v7
# run (403 on checkpoints/ with the read remote, after 45 min of training).
lines, refuse = jm.b2_write_scope_report(jm.b2_write_preflight(cfg, src))
for ln in lines:
    print(ln, file=sys.stderr)
if refuse:
    print("ERR B2 write-scope: this bundle writes a prefix no box key grants",
          file=sys.stderr)
    sys.exit(1)
PY
)" || { echo "[1/5] validate config ... FAIL"; echo "RESULT: FAIL (config or B2 write scope invalid — fix before spending)"; exit 1; }
JOB_NAME="$(printf '%s\n' "$META" | sed -n 's/^name=//p')"
ENTRY="$(printf '%s\n' "$META" | sed -n 's/^entrypoint=//p')"
RESULTS_CSV="$(printf '%s\n' "$META" | sed -n 's/^results=//p')"
JOB_STUB_MODELS="$(printf '%s\n' "$META" | sed -n 's/^stub_models=//p')"
ASSET_LINES="$(printf '%s\n' "$META" | sed -n 's/^asset=//p')"
echo "[1/5] validate config ... OK (name=$JOB_NAME entrypoint=$ENTRY results=[$RESULTS_CSV])"
# entrypoint file must exist (validate_job_config checks this, but be explicit).
# It may come from `includes:` rather than the folder — jobmeta permits that
# deliberately, so a bare -f test here would FAIL a bundle that submits fine.
if [ ! -f "$JOB_DIR/$ENTRY" ] \
   && ! printf '%s\n' "$META" | sed -n 's/^include=//p' | grep -qxF "$ENTRY"; then
  echo "[1/5] entrypoint file missing: $ENTRY (not in the bundle, not in includes:)" >&2
  echo "RESULT: FAIL"; exit 1
fi

# ---- [2/5] fake bucket + rclone shim ----------------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rehearse.XXXXXX")"
BUCKET="$WORK/bucket"; BINDIR="$WORK/bin"; WS="$WORK/workspace"
mkdir -p "$BUCKET" "$BINDIR" "$WS"
cp "$SHIM_SRC" "$BINDIR/rclone"; chmod +x "$BINDIR/rclone"
echo "[2/5] fake bucket + rclone shim ... OK ($WORK)"

STUB_PID=""
cleanup() {
  [ -n "$STUB_PID" ] && kill "$STUB_PID" 2>/dev/null
  if [ "$KEEP" = 1 ]; then echo ">> kept: $WORK" >&2
  else rm -rf "$WORK"; fi
}
trap cleanup EXIT

# ---- [3/5] seed bundle + ticket(s) into the fake bucket ---------------------
# --concurrent N mints N tickets for the SAME bundle sha (content-addressed, so
# one upload) — jobd's poll_once claims every runnable ticket in ONE pass and
# spawns each run_job_body in the background, so N tickets + JOBD_CPU_SLOTS=N is
# genuine concurrency against ONE $JOBD_ROOT. One job id per line.
JOB_IDS="$("$PY" - "$HERE" "$JOB_DIR" "$BUCKET" "$IID" "$CONCURRENT" \
    ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} <<'PY'
import json, os, shutil, sys, tempfile
sys.path.insert(0, sys.argv[1])
import jobmeta as jm
src, bucket, iid, n = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
raw = jm.load_job_config(src)
# --env K=V overrides, folded onto env: PRE-validation — the same slot
# `herdd job submit --env` uses (_apply_env_overrides), so the rehearsed
# ticket carries exactly what a real submit with the same flags would.
overrides = sys.argv[6:]
if overrides:
    env = dict(raw.get("env") or {})
    for kv in overrides:
        k, sep, v = kv.partition("=")
        assert sep, f"--env needs K=V, got {kv!r}"
        env[k] = v
    raw["env"] = env
    print("env override (rehearse-time): %s"
          % ", ".join(kv.partition("=")[0] for kv in overrides),
          file=sys.stderr)
cfg, _ = jm.validate_job_config(raw, src)
tmp = tempfile.mkdtemp()
try:
    bpath = os.path.join(tmp, "b.tar.zst")
    sha = jm.write_bundle(src, bpath)["sha256"]
    bdir = os.path.join(bucket, "jobs", "bundles"); os.makedirs(bdir, exist_ok=True)
    shutil.copy(bpath, os.path.join(bdir, sha + ".tar.zst"))
    qdir = os.path.join(bucket, "jobs", "queue", str(iid)); os.makedirs(qdir, exist_ok=True)
    seen = set()
    for _ in range(n):
        job_id = jm.mint_job_id(cfg["name"])
        while job_id in seen:           # mint_job_id carries a ts: force distinct
            job_id = jm.mint_job_id(cfg["name"])
        seen.add(job_id)
        ticket = jm.make_ticket(job_id, sha, "cli:rehearse", cfg, str(iid))
        with open(os.path.join(qdir, job_id + ".json"), "w") as f:
            json.dump(ticket, f)
        print(job_id)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PY
)" || { echo "[3/5] seed bundle+ticket ... FAIL" >&2; echo "RESULT: FAIL"; exit 1; }
JOB_ID="$(printf '%s\n' "$JOB_IDS" | head -n1)"
N_SEEDED="$(printf '%s\n' "$JOB_IDS" | grep -c .)"
[ "$N_SEEDED" = "$CONCURRENT" ] || {
  echo "[3/5] seed bundle+ticket ... FAIL (minted $N_SEEDED of $CONCURRENT)" >&2
  echo "RESULT: FAIL"; exit 1; }
if [ "$CONCURRENT" = 1 ]; then
  echo "[3/5] seed bundle + ticket ... OK (job_id=$JOB_ID)"
else
  echo "[3/5] seed bundle + $CONCURRENT tickets ... OK (job_id=$JOB_ID +$((CONCURRENT-1)) more)"
  printf '   %s\n' $JOB_IDS
fi

# ---- seed assets into the fake bucket (so the REAL jobd staging path runs) ---
# Each declared asset's B2 prefix is materialized on the fake bucket from a local
# dir: an explicit --asset NAME=DIR override, else the fixture convention
# <assets-fixture>/<name>/. An UNSEEDED asset is left absent on purpose — jobd's
# pull "succeeds" copying zero files (exactly as a real empty B2 prefix would),
# and any `require:` glob / index shard check then fails the job the same way a
# live box does. So a missing/truncated fixture reproduces `asset_stage_failed`.
if [ -n "$ASSET_LINES" ]; then
  n_seed=0; n_skip=0
  while IFS=$'\t' read -r a_name a_b2 a_recp; do
    [ -n "$a_name" ] || continue
    [ "${a_recp:-}" = "-" ] && a_recp=""
    fixdir=""
    if [ -n "${ASSET_OVERRIDE[$a_name]:-}" ]; then
      fixdir="${ASSET_OVERRIDE[$a_name]}"
      [ -d "$fixdir" ] || { echo "!! --asset $a_name=$fixdir is not a directory" >&2; echo "RESULT: FAIL"; exit 1; }
    elif [ -d "$ASSETS_FIXTURE/$a_name" ]; then
      fixdir="$ASSETS_FIXTURE/$a_name"
    fi
    if [ -n "$fixdir" ]; then
      dst="$BUCKET/$a_b2"; mkdir -p "$dst"
      # DEREFERENCE (-L). A HuggingFace cache snapshot — the fixture source this
      # tool's own docs recommend (`--asset base=~/…/snapshots/<sha>`,
      # JOBS_PREFLIGHT.md) — is entirely symlinks into ../../blobs/. A plain
      # `cp -a` preserves them, so the fake bucket gets dangling links, the pull
      # copies 0 B, and the failure surfaces as an unrelated-looking
      # "asset postcondition FAILED (require globs unmatched)" — the documented
      # invocation could not pass. Real B2 stores file bytes, so -L is also the
      # faithful reproduction.
      cp -aL "$fixdir"/. "$dst"/ 2>/dev/null || cp -rL "$fixdir"/. "$dst"/
      seeded_bytes=$(du -sLb "$fixdir" 2>/dev/null | cut -f1)
      # A fixture that seeds ~nothing is a fixture problem, not a bundle
      # problem: say so HERE rather than letting jobd report it as a staging
      # failure 30 s later.
      if [ -n "$seeded_bytes" ] && [ "$seeded_bytes" -lt 1024 ]; then
        echo "!! asset '$a_name' fixture $fixdir holds only ${seeded_bytes}B — seeding an empty asset; expect asset_stage_failed:$a_name" >&2
      fi
      # MINT THE COMPLETENESS RECEIPT the fixture almost certainly lacks. On B2
      # the marker is written by the publisher; here the "publish" is a local
      # directory the operator handed us whole, so a seeded fixture IS complete
      # by construction and refusing it would test the harness, not the bundle —
      # a $0 gate no receipt-declaring job could ever pass. A fixture that
      # carries its own marker keeps it (that is a real captured publish, and
      # rehearsing a DELIBERATELY unpublished prefix stays possible by seeding
      # one whose marker says complete:false).
      if [ -n "$a_recp" ] && [ ! -f "$dst/$a_recp" ]; then
        mkdir -p "$(dirname "$dst/$a_recp")"
        printf '{"complete": true, "files": %s, "ts_utc": "%s"}\n' \
          "$(find "$dst" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')" \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$dst/$a_recp"
        echo ">> asset '$a_name' receipt '$a_recp' MINTED for the rehearsal (the fixture carried none)"
      fi
      echo ">> asset '$a_name' seeded from $fixdir -> b2:$a_b2/"
      n_seed=$((n_seed+1))
    else
      [ -n "$a_recp" ] && echo "!! asset '$a_name' declares receipt: $a_recp and is UNSEEDED — expect asset_receipt_missing:$a_name" >&2
      echo ">> asset '$a_name' NOT seeded (no --asset override, no $ASSETS_FIXTURE/$a_name) — jobd require:/optional decides"
      n_skip=$((n_skip+1))
    fi
  done <<< "$ASSET_LINES"
  echo "[3/5] seed assets ... OK ($n_seed seeded, $n_skip unseeded)"
fi

# ---- optional: start the stub vLLM ------------------------------------------
if [ "$STUB" = 1 ]; then
  [ -f "$STUB_SRC" ] || { echo "!! stub server not found: $STUB_SRC" >&2; echo "RESULT: FAIL"; exit 1; }
  MODELS="${STUB_MODELS:-${JOB_STUB_MODELS:-stub-base,reader}}"
  "$PY" "$STUB_SRC" --host 127.0.0.1 --port "$STUB_PORT" --models "$MODELS" &
  STUB_PID=$!
  # wait for /health
  up=0
  for _ in $(seq 1 50); do
    if curl -fsS -o /dev/null "http://127.0.0.1:${STUB_PORT}/health" 2>/dev/null; then up=1; break; fi
    sleep 0.1
  done
  [ "$up" = 1 ] || { echo "!! stub vLLM never became healthy on :$STUB_PORT" >&2; echo "RESULT: FAIL"; exit 1; }
  echo ">> stub vLLM up on 127.0.0.1:$STUB_PORT (models=$MODELS, pid=$STUB_PID)"
  # eval-shaped entrypoints read these to find the endpoint (inherited by jobd).
  export STUB_ENDPOINT="http://127.0.0.1:${STUB_PORT}"
  export STUB_PORT STUB_MODELS="$MODELS"
fi

# ---- [4/5] run the REAL jobd ------------------------------------------------
run_jobd_local() {
  # exported (not an assignment prefix): a prefix produced by expansion is not
  # parsed as one and bash would try to EXECUTE it.
  [ -n "$EVALSTUB_ENV" ] && export JOBD_FETCH_EVAL_SH="$EVALSTUB_SRC"
  PATH="$BINDIR:$PATH" \
  FAKE_BUCKET="$BUCKET" B2_BUCKET="rehearse-bucket" \
  JOBD_IID="$IID" JOBD_ROOT="$WS" \
  JOBD_ONCE=1 JOBD_SKIP_GPU=1 JOBD_SKIP_B2CONFIG=1 \
  JOBD_CPU_SLOTS="$CONCURRENT" \
  JOBD_HEARTBEAT_S=1 JOBD_PYTHON="$PY" \
    bash "$JOBD_SH"
}

run_jobd_podman() {
  # CPU-only: mount tools/vast (ro) + the fake bucket (rw). The shim is on PATH
  # via /work/tools/vast/testlib installed to /usr/local/bin. Never --pull.
  if ! command -v podman >/dev/null 2>&1; then
    echo ">> SKIP --image: podman not installed — falling back to LOCAL jobd" >&2
    return 3
  fi
  if ! podman image exists "$IMAGE" 2>/dev/null; then
    echo ">> SKIP --image: image not present locally ($IMAGE) — NOT pulling; LOCAL jobd" >&2
    return 3
  fi
  echo ">> running jobd inside podman image $IMAGE (CPU-only)"
  podman run --rm --network host \
    -v "$HERE:/work/tools/vast:ro" \
    -v "$BUCKET:/bucket:rw" -v "$WS:/ws:rw" \
    -e FAKE_BUCKET=/bucket -e B2_BUCKET=rehearse-bucket \
    -e JOBD_IID="$IID" -e JOBD_ROOT=/ws \
    -e JOBD_ONCE=1 -e JOBD_SKIP_GPU=1 -e JOBD_SKIP_B2CONFIG=1 \
    -e JOBD_CPU_SLOTS="$CONCURRENT" \
    -e JOBD_HEARTBEAT_S=1 \
    ${DRY_RUN:+-e} ${DRY_RUN:+DRY_RUN=$DRY_RUN} \
    ${STUB:+-e} ${STUB:+STUB_PORT=$STUB_PORT} \
    ${EVALSTUB_ENV:+-e} ${EVALSTUB_ENV:+JOBD_FETCH_EVAL_SH=/work/tools/vast/testlib/fetch_eval_env_stub.sh} \
    "$IMAGE" bash -c '
      set -e
      install -m 0755 /work/tools/vast/testlib/rclone_shim.sh /usr/local/bin/rclone
      exec bash /work/tools/vast/onstart/jobd.sh'
}

# `needs.venv: eval` cannot self-provision in a rehearsal (fetch_eval_env.sh
# pulls the multi-GB baked env from B2 with real creds), so --stub-eval-env
# points jobd's documented JOBD_FETCH_EVAL_SH seam at the testlib stub. The
# container sees it at its MOUNTED path (/work/tools/vast/...), not the host one.
EVALSTUB_ENV=""
if [ "$EVALSTUB" = 1 ]; then
  [ -f "$EVALSTUB_SRC" ] || { echo "!! eval-env stub not found: $EVALSTUB_SRC" >&2; echo "RESULT: FAIL"; exit 1; }
  EVALSTUB_ENV=1
  echo ">> needs.venv=eval will be satisfied by the REHEARSAL STUB (no toolchain, no game trees)"
fi

JOBD_LOG="$WORK/jobd.out"
JOBD_LANE="local"
if [ -n "$IMAGE" ]; then
  run_jobd_podman >"$JOBD_LOG" 2>&1; jrc=$?
  JOBD_LANE="image ($IMAGE)"
  if [ "$jrc" = 3 ]; then
    # SURFACE THE FALLBACK. `>> SKIP --image: …` is written into $JOBD_LOG, which
    # is only tailed when jobd FAILS — so a rehearsal that silently dropped to
    # the local lane and then PASSed told the operator nothing, and they believed
    # they had exercised the box image. That is not hypothetical: the 2026-08-09
    # "stage 5 dies in /workspace" reading of perf-levers-padfree was the LOCAL
    # lane all along (DEFAULT_IMAGE absent on that host), while the same bundle
    # PASSed in the image. Print it on the console, every time.
    grep -h '^>> SKIP --image' "$JOBD_LOG" >&2 \
      || echo ">> SKIP --image: podman/image unavailable — falling back to LOCAL jobd" >&2
    echo ">> NOTE: the image lane did NOT run; what follows is the LOCAL lane," \
         "which inherits this host's nvidia-smi, /workspace and toolchain" >&2
    JOBD_LANE="local (--image SKIPPED)"
    IMAGE_SKIPPED=1
    run_jobd_local >"$JOBD_LOG" 2>&1; jrc=$?
  fi
else
  run_jobd_local >"$JOBD_LOG" 2>&1; jrc=$?
fi
if [ "$jrc" != 0 ]; then
  echo "[4/5] run jobd ... FAIL (jobd exited $jrc, lane=$JOBD_LANE)"
  echo "----- jobd log tail -----"; tail -n 40 "$JOBD_LOG"
  echo "RESULT: FAIL"; exit 1
fi
echo "[4/5] run jobd (JOBD_ONCE=1 JOBD_SKIP_GPU=1 lane=$JOBD_LANE) ... OK"

# ---- [5/5] assert results.DONE.json + every results glob landed -------------
# When there is NO DONE marker, fold the job's event log for a terminal `failed`
# reason and surface it verbatim — so an asset that failed staging shows the
# box-identical `asset_stage_failed:<name>` shape (not a generic "didn't finish").
assert_job() {   # assert_job <job-id> -> echoes an OK.../ERR... line, rc 0/1
  "$PY" - "$BUCKET" "$1" "$RESULTS_CSV" <<'PY'
import glob, json, os, sys
bucket, job_id, results_csv = sys.argv[1], sys.argv[2], sys.argv[3]
jdir = os.path.join(bucket, "jobs", job_id)
donef = os.path.join(jdir, "results.DONE.json")
if not os.path.isfile(donef):
    reason = None
    started = False
    evd = os.path.join(jdir, "events")
    if os.path.isdir(evd):
        for nm in sorted(os.listdir(evd)):
            try:
                e = json.load(open(os.path.join(evd, nm)))
            except (ValueError, OSError):
                continue
            if e.get("event") == "started":
                started = True
            if e.get("event") == "failed" and e.get("reason"):
                reason = e.get("reason")
    if reason is not None:
        where = "post-entrypoint" if started else "pre-entrypoint"
        print("ERR job failed %s: %s" % (where, reason)); sys.exit(1)
    print("ERR no results.DONE.json (job did not finish cleanly)"); sys.exit(1)
dm = json.load(open(donef))
if int(dm.get("rc", 1)) != 0:
    print("ERR entrypoint rc=%s (nonzero) — job failed" % dm.get("rc")); sys.exit(1)
rdir = os.path.join(jdir, "results")
globs = [g for g in results_csv.split(",") if g]
missing = []
for g in globs:
    if not glob.glob(os.path.join(rdir, g), recursive=True):
        missing.append(g)
if missing:
    print("ERR results glob(s) matched nothing on the bucket: %s" % ", ".join(missing)); sys.exit(1)
print("OK rc=0 n_results=%s globs=%d" % (dm.get("n_results"), len(globs)))
PY
}

# EVERY seeded job must land: under --concurrent N a single survivor is exactly
# the shape the round-3 pip corruption took (1 of 3 jobs "passed").
FAILED=0
for jid in $JOB_IDS; do
  if ASSERT="$(assert_job "$jid")"; then
    [ "$CONCURRENT" = 1 ] || echo "   $jid ... $ASSERT"
  else
    FAILED=$((FAILED+1))
    echo "[5/5] assert results ... FAIL ($jid)"; echo "   ${ASSERT#ERR }"
    # entrypoint log FIRST (the actual failure, e.g. a missing endpoint / bad cmd),
    # jobd's operational log second. jobd rcat's the entrypoint's log.txt to the
    # bucket even on failure (crash-safe publish), so surface it when present.
    ELOG="$BUCKET/jobs/$jid/log.txt"
    if [ -s "$ELOG" ]; then echo "----- entrypoint log tail -----"; tail -n 40 "$ELOG"; fi
  fi
done
if [ "$FAILED" != 0 ]; then
  echo "----- jobd log tail -----"; tail -n 40 "$JOBD_LOG"
  echo "RESULT: FAIL ($FAILED of $CONCURRENT job(s) did not land)"; exit 1
fi
if [ "$CONCURRENT" = 1 ]; then
  echo "[5/5] assert results ... ${ASSERT}"
else
  echo "[5/5] assert results ... OK ($CONCURRENT/$CONCURRENT concurrent job(s) landed)"
fi

# ---- optional results-capture seam: let a driver (e.g. `workflow rehearse`) --
# pull this stage's produced results out of the fake bucket before $WORK is
# removed by the EXIT trap. Only fires on the PASS path; no-op when unset.
if [ -n "${REHEARSE_RESULTS_OUT:-}" ]; then
  mkdir -p "$REHEARSE_RESULTS_OUT"
  src="$BUCKET/jobs/$JOB_ID/results"
  cp -a "$src"/. "$REHEARSE_RESULTS_OUT"/ 2>/dev/null || cp -r "$src"/. "$REHEARSE_RESULTS_OUT"/
  echo ">> results copied to: $REHEARSE_RESULTS_OUT" >&2
fi

# A PASS may only speak for the lane that RAN. `--image` was requested here and
# the image lane did not run, so what passed was this host's python/toolchain —
# structurally unable to see an image-specific fault. Printing a bare PASS was
# tried and failed twice: 2026-08-09 (perf-levers-padfree read as "stage 5 dies
# in /workspace" off the local lane) and 2026-08-19 (box 48089639 parked, whole
# python half dead, because the image ships python 3.11 and this host runs 3.13
# — the rehearsal PASSed minutes earlier). The console warning existed both
# times; the VERDICT is what gets read, so the verdict carries it now.
if [ "$IMAGE_SKIPPED" = 1 ] && [ "$IMAGE_FALLBACK_OK" != 1 ]; then
  echo "RESULT: INCONCLUSIVE — the local lane passed, but --image ($IMAGE) never"
  echo "  ran, so the box image is UNTESTED. Make the image present"
  echo "  (podman pull $IMAGE) and re-run, or pass --allow-image-fallback to"
  echo "  accept a local-lane-only result."
  exit 4
fi
if [ "$IMAGE_SKIPPED" = 1 ]; then
  echo "RESULT: PASS (LOCAL lane only — --image was SKIPPED, box image UNTESTED)"
  exit 0
fi
echo "RESULT: PASS${IMAGE:+ (image lane: $IMAGE)}"
exit 0
