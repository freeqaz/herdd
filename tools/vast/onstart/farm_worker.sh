#!/usr/bin/env bash
# onstart/farm_worker.sh — CPU compile-farm work-unit runner for the eval sidecar.
#
# WHAT: generalizes the fixed-target eval into a resumable, idempotent work-unit
# loop. It pulls a manifest of compile jobs from B2 (farm/<RUN_ID>/inbox/), runs
# each niced through the SAME oracle the local box uses (chain_to_escore +
# score_real_anchor for `escore` units; crack_live for `crack` units), and pushes
# each unit's result to farm/<RUN_ID>/results/ the instant it finishes — so a box
# that dies mid-run resumes exactly where it left off (already-done units are
# skipped by their result marker on B2).
#
# HOW IT RUNS: as the sidecar's EVAL_CMD, or auto-launched by onstart/train.sh in
# farm mode (EVAL_MODE=farm). Either way it inherits the sidecar's yield fence
# (renice/ionice/oom/cgroup — children inherit) and its unpacked env at
# /workspace/eval. It re-applies the fence defensively so a standalone invocation
# still yields. It never touches checkpoints/<RUN_ID>/STATUS.
#
# B2 WORK-UNIT PROTOCOL
#   farm/<RUN_ID>/inbox/manifest.json     list of units (schema below)
#   farm/<RUN_ID>/inbox/<path>            per-unit input DB(s), referenced by manifest
#   farm/<RUN_ID>/results/<id>/           per-unit output DBs (escore.db, ...)
#   farm/<RUN_ID>/results/<id>.json       per-unit summary — the DONE marker (idempotency)
#   farm/<RUN_ID>/FARM_STATUS             coarse heartbeat (RUNNING/DONE/FAILED)
#
# manifest.json:
#   { "version": 1, "units": [
#       { "id": "u0001", "kind": "escore", "target": "dc3",
#         "repo_tag": "dc3", "input_db": "u0001/input.db",
#         "chain_args": "", "score_args": "--limit 200" },
#       { "id": "u0002", "kind": "crack", "target": "rb3-xenon",
#         "crack_args": "--from-frontier --min-pct 90 --limit 5 --budget 24" }
#   ] }
#   kind=escore : input_db (under inbox/) required. kind=crack : crack_args required.
#   Result marker written LAST, after the output DBs land, so an interrupted push
#   never leaves a unit falsely "done".
#
# Env contract (RUN_ID + B2_BUCKET come from the box; the rest are optional):
#   FARM_RUN_ID           overrides RUN_ID for the farm inbox/results namespace
#   EVAL_JOBS             per-unit oracle parallelism (default 4; sidecar sets it)
#   FARM_MAX_UNITS        stop after N units this boot (default 0 = all)
#   FARM_REFRESH=1        re-run a unit even if its result marker exists on B2
set -uo pipefail

# --- defensive re-fence (a no-op when already fenced by the sidecar) ----------
_YF="$(dirname "$0")/yield_fence.sh"; [ -f "$_YF" ] || _YF=/workspace/yield_fence.sh
if [ -f "$_YF" ]; then
  # shellcheck source=/dev/null
  { . "$_YF" && yield_fence_self "farm-worker"; } || true
else
  renice 19 -p "$$" >/dev/null 2>&1 || true
  ionice -c3 -p "$$" >/dev/null 2>&1 || true
  echo 800 > "/proc/$$/oom_score_adj" 2>/dev/null || true
fi

# --- env.sh (venv + tool PATH); marker-guarded, same as escore_job.sh ---------
if [ -z "${ESCORE_ENV_SOURCED:-}" ] && [ -f /workspace/eval/env.sh ]; then
  # shellcheck source=/dev/null
  source /workspace/eval/env.sh
  export ESCORE_ENV_SOURCED=1
fi

RUN_ID="${FARM_RUN_ID:-${RUN_ID:?farm_worker: RUN_ID (or FARM_RUN_ID) required}}"
B2="${B2:-b2:${B2_BUCKET:?farm_worker: B2_BUCKET required}}"
JOBS="${EVAL_JOBS:-4}"
DS="/workspace/eval/upstream-monorepo"
INBOX="$B2/farm/$RUN_ID/inbox"
RESULTS="$B2/farm/$RUN_ID/results"
SCRATCH="/workspace/farm-scratch/$RUN_ID"
OUTMIRROR="/workspace/eval-out/farm"      # streamed to evals/<RUN_ID>/farm/ by the sidecar
mkdir -p "$SCRATCH" "$OUTMIRROR"

log() { echo ">> [farm $RUN_ID] $*" >&2; }
farm_status() { echo "$1 $(date -u +%FT%TZ)" | rclone rcat "$B2/farm/$RUN_ID/FARM_STATUS" 2>/dev/null || true; }

# --- fail-open: no manifest => loud no-op, training unaffected ----------------
MANIFEST="$SCRATCH/manifest.json"
if ! rclone copyto "$INBOX/manifest.json" "$MANIFEST" 2>/dev/null; then
  log "no manifest at $INBOX/manifest.json — CPU farm has no work (no-op; training unaffected)"
  echo '{"farm_run": "'"$RUN_ID"'", "status": "no-manifest", "units_done": 0}'
  exit 0
fi
farm_status RUNNING

# --- expand manifest -> one shell-sourceable file per unit (python; jq absent)-
# Fields are shlex.quoted so args with spaces survive the bash `source`.
UNITDIR="$SCRATCH/units"; rm -rf "$UNITDIR"; mkdir -p "$UNITDIR"
python - "$MANIFEST" "$UNITDIR" <<'PY' >&2
import json, sys, shlex, os
manifest, outdir = sys.argv[1], sys.argv[2]
data = json.load(open(manifest))
units = data.get("units", [])
for i, u in enumerate(units):
    uid = str(u.get("id") or f"u{i:04d}")
    def q(k, default=""):
        return shlex.quote(str(u.get(k, default) or default))
    with open(os.path.join(outdir, f"{i:05d}.unit"), "w") as f:
        f.write(f"U_ID={shlex.quote(uid)}\n")
        f.write(f"U_KIND={q('kind','escore')}\n")
        f.write(f"U_TARGET={q('target')}\n")
        f.write(f"U_REPO_TAG={q('repo_tag')}\n")
        f.write(f"U_INPUT_DB={q('input_db')}\n")
        f.write(f"U_CHAIN_ARGS={q('chain_args')}\n")
        f.write(f"U_SCORE_ARGS={q('score_args')}\n")
        f.write(f"U_CRACK_ARGS={q('crack_args')}\n")
print(f"expanded {len(units)} units", file=sys.stderr)
PY

DONE=0; SKIPPED=0; FAILED=0; MAXU="${FARM_MAX_UNITS:-0}"
shopt -s nullglob
for uf in "$UNITDIR"/*.unit; do
  # reset per-unit vars then source (source=/dev/null: dynamic file)
  U_ID=""; U_KIND=""; U_TARGET=""; U_REPO_TAG=""; U_INPUT_DB=""
  U_CHAIN_ARGS=""; U_SCORE_ARGS=""; U_CRACK_ARGS=""
  # shellcheck source=/dev/null
  . "$uf"
  [ -n "$U_ID" ] || continue

  # idempotent skip: the result marker on B2 is the source of truth (survives box death)
  if [ "${FARM_REFRESH:-0}" != "1" ] && rclone lsf "$RESULTS/$U_ID.json" 2>/dev/null | grep -q .; then
    log "unit $U_ID already done (result marker present) — skip"
    SKIPPED=$((SKIPPED+1)); continue
  fi

  tdir="/workspace/eval/$U_TARGET"
  tag="${U_REPO_TAG:-$U_TARGET}"
  usc="$SCRATCH/$U_ID"; rm -rf "$usc"; mkdir -p "$usc"
  ok=1
  if [ ! -d "$tdir" ]; then
    log "unit $U_ID: target dir missing ($tdir — not in this env build) — marking failed"
    ok=0
  else
    case "$U_KIND" in
      escore)
        [ -n "$U_INPUT_DB" ] || { log "unit $U_ID: escore needs input_db"; ok=0; }
        if [ "$ok" = 1 ]; then
          log "unit $U_ID escore target=$U_TARGET tag=$tag"
          if rclone copyto "$INBOX/$U_INPUT_DB" "$usc/input.db" 2>/dev/null; then
            # SAME oracle path as escore_job.sh (chain_to_escore --per-state -> score_real_anchor).
            # U_*_ARGS are intentionally word-split into argv (e.g. "--limit 200").
            # shellcheck disable=SC2086
            ( cd "$tdir" \
              && python "$DS/tools/pipeline/chain/chain_to_escore.py" --per-state \
                   --repo-tag "$tag" --repo-root "$tdir" --src "$usc/input.db" \
                   --out "$usc/escore_input.db" --mapping-out "$usc/escore_map.json" \
                   ${U_CHAIN_ARGS:-} \
              && python "$DS/tools/pipeline/score_real_anchor.py" \
                   --db "$usc/escore_input.db" --out "$usc/escore.db" \
                   --repo-tag "$tag" --repo-root "$tdir" --workers "$JOBS" \
                   ${U_SCORE_ARGS:-} ) >&2 || ok=0
          else
            log "unit $U_ID: input pull failed ($INBOX/$U_INPUT_DB)"; ok=0
          fi
        fi ;;
      crack)
        [ -n "$U_CRACK_ARGS" ] || { log "unit $U_ID: crack needs crack_args"; ok=0; }
        if [ "$ok" = 1 ]; then
          log "unit $U_ID crack target=$U_TARGET"
          # SAME oracle: crack_live.py drives the compile+score search. Args are
          # opaque pass-through (symbol/frontier selection lives in the manifest).
          # --out is a JSON report (crack_live writes json.dumps to it), so name
          # it .json — the *.json publish glob ships it; the old crack.db name
          # made any ingester that sqlite3-opens it fail ('file is not a
          # database'). The (prompt, proposals, verdicts) transcript stream —
          # the v2 ranker's training data (crack_live.py docstring) — defaults
          # to a LOCAL cache that dies with the box: route it into the unit
          # scratch so the *.db publish glob ships it to B2. Guarded to
          # llm-generator units (crack_live ap.error()s on --transcript-db
          # without --generator llm).
          xtra=""
          case "$U_CRACK_ARGS" in
            *--generator?llm*)
              case "$U_CRACK_ARGS" in
                *--transcript-db*) ;;
                *) xtra="--transcript-db $usc/crack_transcripts.db" ;;
              esac ;;
          esac
          # shellcheck disable=SC2086
          ( cd "$tdir" \
            && python "$DS/tools/pipeline/crack_live.py" \
                 --repo-root "$tdir" --out "$usc/crack.json" ${U_CRACK_ARGS:-} $xtra ) >&2 || ok=0
        fi ;;
      *)
        log "unit $U_ID: unknown kind '$U_KIND'"; ok=0 ;;
    esac
  fi

  # publish per unit: output DBs FIRST, result marker LAST (crash-safe idempotency)
  rm -f "$usc/input.db"    # the pulled input is not a result — don't echo it back
  status_word="ok"; [ "$ok" = 1 ] || status_word="failed"
  nfiles=0
  for f in "$usc"/*.db "$usc"/*.json; do
    [ -f "$f" ] || continue
    if rclone rcat "$RESULTS/$U_ID/$(basename "$f")" < "$f" 2>/dev/null; then
      nfiles=$((nfiles+1))
    fi
  done
  # one-line JSON marker (mirrored locally so the sidecar streams it to evals/ too).
  # Provenance rides IN the marker (not only in DB meta tables) because crack
  # output has no meta table — without this its git sha is unrecoverable — and
  # because the ingester (tools/vast/farm_ingest.py) reads the marker first and
  # must survive units whose DBs failed to upload. args_b64 = base64 of the
  # unit's arg strings: base64 avoids JSON-escaping hazards in a printf-built
  # one-liner. Readers must tolerate the added fields (farm_ingest does).
  rgit=$(git -C "$tdir" rev-parse HEAD 2>/dev/null || echo unknown)
  dsgit=$(git -C "$DS" rev-parse HEAD 2>/dev/null || echo unknown)
  args_b64=$(printf '%s\037%s\037%s' "${U_CHAIN_ARGS:-}" "${U_SCORE_ARGS:-}" "${U_CRACK_ARGS:-}" | base64 -w0 2>/dev/null || true)
  marker="$usc/$U_ID.result.json"
  printf '{"unit":"%s","kind":"%s","target":"%s","status":"%s","files":%d,"ts":"%s","repo_tag":"%s","repo_git_sha":"%s","ds_git_sha":"%s","input_db":"%s","args_b64":"%s"}\n' \
    "$U_ID" "$U_KIND" "$U_TARGET" "$status_word" "$nfiles" "$(date -u +%FT%TZ)" \
    "$tag" "$rgit" "$dsgit" "${U_INPUT_DB:-}" "$args_b64" > "$marker"
  cp -f "$marker" "$OUTMIRROR/$U_ID.result.json" 2>/dev/null || true
  rclone rcat "$RESULTS/$U_ID.json" < "$marker" 2>/dev/null || true
  rm -rf "$usc"

  if [ "$ok" = 1 ]; then DONE=$((DONE+1)); else FAILED=$((FAILED+1)); fi
  if [ "$MAXU" != "0" ] && [ "$((DONE+FAILED))" -ge "$MAXU" ]; then
    log "FARM_MAX_UNITS=$MAXU reached this boot — stopping (rest resume next boot)"; break
  fi
done

TOTAL=$((DONE+SKIPPED+FAILED))
if [ "$FAILED" -eq 0 ]; then farm_status "DONE done=$DONE skip=$SKIPPED"; else farm_status "PARTIAL done=$DONE skip=$SKIPPED failed=$FAILED"; fi
log "farm complete: done=$DONE skipped=$SKIPPED failed=$FAILED total=$TOTAL"
# stdout summary -> result.json when run as the sidecar's EVAL_CMD
echo '{"farm_run":"'"$RUN_ID"'","done":'"$DONE"',"skipped":'"$SKIPPED"',"failed":'"$FAILED"',"total":'"$TOTAL"'}'
[ "$FAILED" -eq 0 ]
