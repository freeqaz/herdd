#!/usr/bin/env bash
# serve_flip.sh — install a DURABLE model flip on a serve box.
#
# WHY THIS EXISTS. The model a serve box serves arrives as vast `extra_env`
# (MODEL_B2 / MODEL_ID / SERVED_NAME / MAX_LEN), and that env is FIXED AT CREATE
# TIME for the life of the instance. A flip done at runtime — kill the base
# vLLM, hand-start one on a merged dir — is process state and nothing else, so
# when a spot eviction's rescue re-creates the container, PID 1 re-runs the
# stored onstart against that same immutable env and the box comes back serving
# the LAUNCH model under the flip's endpoint label. Renaming the base dir does
# not help: the boot re-pulls it from B2 before serving it.
#
# TWO DURABLE SEAMS, and a real flip writes both:
#   /workspace/serve_model_override.json  — on the box; onstart/serve_vllm.sh
#       resolves it before any pull, on every start (launch onstart, onstart
#       re-run after a resume, jobs-lane attempt re-run, --on-box attach).
#   b2:$B2_BUCKET/serve/<SERVE_ID>/serve_model_override.json — beside the
#       serve_main.sh that serve_boot.sh already re-pulls every boot. Writable
#       from the workstation with no ssh, and it survives the box entirely.
# `stage` also re-stages serve_main.sh from THIS checkout: a box launched before
# the override existed runs a serve_main.sh that will never look for it.
#
# The box-side half is fail-closed by construction — a present-but-unusable
# override is a refusal, never a fallback to the launch model — and `write`
# drops /workspace/.serve_flipped so that even a lost override file makes the
# next boot refuse rather than quietly serve base.
#
# Box-side:
#   serve_flip.sh write --model-path /workspace/merged/adapter-merged \
#       --marker .v4_relayout_ok.json --reason "cell B stage 1" [--restart]
#   serve_flip.sh check              # what the next start would resolve
#   serve_flip.sh clear              # back to the launch model, deliberately
# Workstation-side (B2 creds from tools/vast/.env):
#   serve_flip.sh stage --serve-id serve-260826-0616-2d6f \
#       --model-path /workspace/merged/adapter-merged --marker .v4_relayout_ok.json
#
# Flags:
#   --model-path DIR     what to serve instead (an ON-BOX path)
#   --marker NAME|PATH   completion marker inside DIR (or absolute). Its absence
#                        at serve time is a REFUSAL — this is what keeps a
#                        half-written merge from being served.
#   --served-name N      --served-model-name (default: keep the launch label,
#                        which is usually the point of a flip)
#   --max-len N          --max-model-len for the override target
#   --identity-expect P  on-box identity_expect.json for the TARGET. Required
#                        when the launch armed SERVE_IDENT_REQUIRED=1.
#   --allow-lora         keep the launch's LORA_SPECS. Default refuses them: if
#                        the override IS the merge, the adapter applies twice.
#   --reason TEXT        recorded in the file and printed on every start
#   --file PATH          override path (default /workspace/serve_model_override.json)
#   --sentinel PATH      flip sentinel (default /workspace/.serve_flipped)
#   --restart            write only: kill the running vLLM and re-run the serve
#   --serve-id ID        stage only: the SERVE_ID whose B2 prefix to write
#   --src PATH           stage only: serve payload (default onstart/serve_vllm.sh)
#   --no-restage         stage only: write the override object, leave serve_main.sh
#
# Exit: 0 ok · 1 refusal / failure · 2 usage
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
usage() { grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'; exit "${1:-2}"; }

CMD="${1:-}"; [ $# -gt 0 ] && shift || true
case "$CMD" in write|clear|check|stage) ;; -h|--help) usage 0;; *) echo "!! first arg must be write|clear|check|stage" >&2; usage 2;; esac

MODEL_PATH=""; MARKER=""; SERVED_NAME=""; MAX_LEN=""; IDENTITY_EXPECT=""
ALLOW_LORA=0; REASON=""; RESTART=0; SERVE_ID=""; SRC="$HERE/onstart/serve_vllm.sh"
RESTAGE=1
OVFILE="${SERVE_MODEL_OVERRIDE:-/workspace/serve_model_override.json}"
SENTINEL="${SERVE_FLIP_SENTINEL:-/workspace/.serve_flipped}"
while [ $# -gt 0 ]; do case "$1" in
  --model-path) MODEL_PATH="$2"; shift 2;;
  --marker) MARKER="$2"; shift 2;;
  --served-name) SERVED_NAME="$2"; shift 2;;
  --max-len) MAX_LEN="$2"; shift 2;;
  --identity-expect) IDENTITY_EXPECT="$2"; shift 2;;
  --allow-lora) ALLOW_LORA=1; shift;;
  --reason) REASON="$2"; shift 2;;
  --file) OVFILE="$2"; shift 2;;
  --sentinel) SENTINEL="$2"; shift 2;;
  --restart) RESTART=1; shift;;
  --serve-id) SERVE_ID="$2"; shift 2;;
  --src) SRC="$2"; shift 2;;
  --no-restage) RESTAGE=0; shift;;
  -h|--help) usage 0;;
  *) echo "!! unknown arg $1" >&2; usage 2;;
esac; done

# Compose the document in ONE place: `write` and `stage` must never be able to
# install differently-shaped overrides for the same flip.
compose() {
  FL_MODEL_PATH="$MODEL_PATH" FL_MARKER="$MARKER" FL_SERVED_NAME="$SERVED_NAME" \
  FL_MAX_LEN="$MAX_LEN" FL_IDENTITY_EXPECT="$IDENTITY_EXPECT" \
  FL_ALLOW_LORA="$ALLOW_LORA" FL_REASON="$REASON" python3 - <<'PY'
import json, os, time
def s(k):
    v = os.environ.get("FL_" + k, "").strip()
    return v or None
doc = {"schema_version": 1,
       "model_path": s("MODEL_PATH"),
       "marker": s("MARKER"),
       "served_name": s("SERVED_NAME"),
       "max_len": s("MAX_LEN"),
       "identity_expect": s("IDENTITY_EXPECT"),
       "allow_lora": os.environ.get("FL_ALLOW_LORA") == "1",
       "reason": s("REASON"),
       "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
print(json.dumps(doc, indent=1))
PY
}

# temp-in-the-same-dir + rename: a reader at boot sees the old file or the new
# one, never a half-written one.
atomic_write() {   # $1=dest, stdin=content
  local dest="$1" dir tmp
  dir="$(dirname "$dest")"; mkdir -p "$dir"
  tmp="$(mktemp "$dir/.serve_flip.XXXXXX")"
  cat > "$tmp"
  chmod 644 "$tmp"
  mv "$tmp" "$dest"
}

validate_local() {   # only where the target is actually reachable
  local m
  [ -d "$MODEL_PATH" ] || { echo "!! --model-path '$MODEL_PATH' is not a directory on this host" >&2; exit 1; }
  [ -f "$MODEL_PATH/config.json" ] || { echo "!! '$MODEL_PATH' holds no config.json — not a servable model dir" >&2; exit 1; }
  if [ -n "$MARKER" ]; then
    case "$MARKER" in /*) m="$MARKER";; *) m="$MODEL_PATH/$MARKER";; esac
    [ -e "$m" ] || { echo "!! completion marker '$m' is absent — the merge has not finished; refusing to install a flip that would refuse at boot" >&2; exit 1; }
  else
    echo ">> note: no --marker declared. A flip with no completion marker cannot tell a finished merge from a half-written one." >&2
  fi
}

case "$CMD" in

check)
  if [ ! -e "$OVFILE" ]; then
    echo ">> no override at $OVFILE — the next start serves the LAUNCH model"
    [ -e "$SENTINEL" ] && { echo "!! but the flip sentinel $SENTINEL exists, so the next start will REFUSE (fail-closed)."; exit 1; }
    exit 0
  fi
  echo ">> override at $OVFILE:"; cat "$OVFILE"
  MODEL_PATH="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("model_path") or "")' "$OVFILE")"
  MARKER="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("marker") or "")' "$OVFILE")"
  validate_local
  echo ">> OK — the next start would serve $MODEL_PATH"
  ;;

write)
  [ -n "$MODEL_PATH" ] || { echo "!! write needs --model-path" >&2; usage 2; }
  validate_local
  compose | atomic_write "$OVFILE"
  printf 'flipped %s -> %s\n' "$(date -u +%FT%TZ)" "$MODEL_PATH" | atomic_write "$SENTINEL"
  echo ">> wrote $OVFILE (and the fail-closed sentinel $SENTINEL)"
  echo ">> every start from now on serves $MODEL_PATH — including an onstart re-run after an eviction/resume."
  if [ -n "$SERVE_ID" ]; then
    echo ">> NOTE: --serve-id is not used by write; the B2 half is a separate step:" >&2
    echo ">>   $0 stage --serve-id $SERVE_ID --model-path $MODEL_PATH ${MARKER:+--marker $MARKER}" >&2
  fi
  if [ "$RESTART" -eq 1 ]; then
    # `[v]llm serve` compiles to the same regex but is NOT the literal in this
    # shell's own argv — a bare pattern kills the shell running this script.
    echo ">> --restart: stopping the running engine"
    pkill -f '[v]llm serve' >/dev/null 2>&1 || true
    sleep 3
    SERVE_SH=""
    for p in "${SERVE_VLLM_SH:-}" /workspace/serve_main.sh /workspace/serve_vllm.sh \
             /workspace/jobd/serve_vllm.sh "$HERE/onstart/serve_vllm.sh"; do
      [ -n "$p" ] && [ -f "$p" ] && { SERVE_SH="$p"; break; }
    done
    if [ -z "$SERVE_SH" ]; then
      echo "!! --restart: no serve payload found (looked at /workspace/serve_main.sh and friends)." >&2
      echo "!!   The override IS installed — restart the serve however this box normally does." >&2
      exit 1
    fi
    # The launch env persists to /etc/environment (serve_vllm.sh does this on
    # every run), so a manual restart gets the same B2 creds and SERVE_ID.
    [ -f /etc/environment ] && { set -a; . /etc/environment; set +a; }
    LOG="${SERVE_LOG:-/workspace/serve.log}"
    setsid bash "$SERVE_SH" >>"$LOG" 2>&1 &
    echo ">> restarted $SERVE_SH as pid $! (log $LOG)"
  fi
  ;;

clear)
  rm -f "$OVFILE" "$SENTINEL"
  echo ">> removed $OVFILE and $SENTINEL — the next start serves the LAUNCH model again."
  ;;

stage)
  [ -n "$SERVE_ID" ] || { echo "!! stage needs --serve-id" >&2; usage 2; }
  [ -n "$MODEL_PATH" ] || { echo "!! stage needs --model-path (the path ON THE BOX)" >&2; usage 2; }
  ENVF="${_SERVE_FLIP_ENV:-$HERE/../../.env}"; [ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
  : "${B2_BUCKET:?stage needs B2_BUCKET (source tools/vast/.env)}"
  : "${B2_KEY_ID:?stage needs B2_KEY_ID}"
  bash "$HERE/b2_sync.sh" config >/dev/null
  # The target lives on the box, so nothing here can check it. Say so once,
  # loudly: the box-side gate is what actually refuses a bad target.
  echo ">> stage: '$MODEL_PATH' is an ON-BOX path — unverifiable from here. The box refuses at boot if it is absent or unfinished."
  compose | rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/serve_model_override.json"
  echo ">> staged b2:$B2_BUCKET/serve/$SERVE_ID/serve_model_override.json"
  if [ "$RESTAGE" -eq 1 ]; then
    [ -f "$SRC" ] || { echo "!! stage: serve payload not found: $SRC" >&2; exit 1; }
    rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/serve_main.sh" < "$SRC"
    echo ">> re-staged serve_main.sh ($(wc -c < "$SRC")B) — a box whose serve_main.sh predates the override now reads it on its next boot"
  fi
  ;;
esac
