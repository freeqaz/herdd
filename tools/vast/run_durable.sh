#!/usr/bin/env bash
# run_durable.sh — run a training command on an ALREADY-RENTED box with B2
# checkpoint streaming + resume, so the run survives box death.
#
# Complements `herdd train`: that provisions a FRESH box and runs onstart/train.sh.
# Use THIS when you already have a box up (with B2 creds + a working `b2:` rclone
# remote in its env, as the vast boxes have) and want to train on it durably.
#
#   RUN_ID=gemma4-prop-01 tools/vast/run_durable.sh /workspace/out \
#     python3 train_proposer_lora.py --save-steps 50 --resume auto --out {OUT} <args...>
#
# {OUT} in the command is replaced by OUT_DIR (the local checkpoint dir). While the
# command runs, OUT_DIR streams to b2:$B2_BUCKET/checkpoints/$RUN_ID every
# CKPT_INTERVAL sec (default 180); on start any prior checkpoints for $RUN_ID are
# pulled back first, so the trainer's --resume auto continues after a death.
# On success the final OUT_DIR is copied to b2:.../artifacts/$RUN_ID.
#
# Resume after a box died: on the new (or same) box, run the SAME command with the
# SAME RUN_ID — the pull-back + --resume auto pick up from the last streamed step.
set -uo pipefail
: "${RUN_ID:?set RUN_ID}"; : "${B2_BUCKET:?B2 env (B2_BUCKET) not set — is this a vast box?}"
OUT="${1:?usage: RUN_ID=<id> run_durable.sh <OUT_DIR> <CMD...>}"; shift
[ "$#" -gt 0 ] || { echo "!! no command given" >&2; exit 2; }
INTERVAL="${CKPT_INTERVAL:-180}"
B2="b2:${B2_BUCKET}"
mkdir -p "$OUT"

# STATUS marker (same convention as onstart/train.sh) so `herdd runs` can show
# this run's live state. Written directly to B2, never into OUT.
status() { echo "$1 $(date -u +%FT%TZ)" | rclone rcat "$B2/checkpoints/$RUN_ID/STATUS" 2>/dev/null || true; }

# append-only run-metadata events (runs/<RUN_ID>/events/) — same inline emitter as
# onstart/train.sh (box is a no-repo actor). One immutable object per event.
_json_str() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys;sys.stdout.write(json.dumps(sys.stdin.read().rstrip("\n")))'
  else
    local s; s=$(cat); s=${s//\\/\\\\}; s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}; s=${s//$'\r'/\\r}; s=${s//$'\t'/\\t}; printf '"%s"' "$s"
  fi
}
emit_event() {                      # emit_event <event> [reason]; env: STEP,DPH
  local ev=$1 reason=${2:-} ts nonce iid actor key
  ts=$(date -u +%Y%m%dT%H%M%S%3NZ); nonce=$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')
  iid="${INSTANCE_ID:-${CONTAINER_ID:-unknown}}"; actor="box:${iid}"
  key="runs/${RUN_ID}/events/${ts}-box_${iid}-${nonce}.json"
  { printf '{"v":1,"ts":"%s","actor":"%s","event":"%s","run_id":"%s","nonce":"%s"' \
      "$ts" "$actor" "$ev" "$RUN_ID" "$nonce"
    [ -n "$reason"   ] && printf ',"reason":%s' "$(printf '%s' "$reason" | _json_str)"
    [ -n "${STEP:-}" ] && printf ',"step":%s'   "$STEP"
    printf '}\n'
  } | rclone rcat "$B2/${key}" 2>/dev/null
}
_emit_terminal() { local ev=$1 reason=${2:-} _i; for _i in 1 2 3; do emit_event "$ev" "$reason" && return 0; sleep 5; done; return 0; }

echo ">> [durable] RUN_ID=$RUN_ID OUT=$OUT stream -> $B2/checkpoints/$RUN_ID every ${INTERVAL}s"
status RUNNING
# resume: pull any prior checkpoints for this RUN_ID back to OUT before training
rclone copy --fast-list --transfers 16 --exclude STATUS "$B2/checkpoints/$RUN_ID" "$OUT" 2>/dev/null || true
[ -n "$(ls -A "$OUT" 2>/dev/null)" ] && echo ">> [durable] pulled prior state for $RUN_ID (will --resume)"

# background streamer: --min-age 45s skips files the trainer is mid-write on
( LAST_EMIT_STEP=0
  while true; do
    rclone copy --fast-list --min-age 45s --exclude STATUS "$OUT" "$B2/checkpoints/$RUN_ID" 2>/dev/null \
      && echo "[durable] ckpt pushed $(date -u +%T)"
    status RUNNING
    # glob both flat ($OUT/checkpoint-*) and arm-nested ($OUT/arms/<name>/
    # checkpoint-*) layouts so latest_step is emitted either way — see the
    # matching note in onstart/train.sh (2026-07-09 fix).
    newstep=$(ls -1d "$OUT"/checkpoint-* "$OUT"/arms/*/checkpoint-* 2>/dev/null \
               | sed 's#.*/checkpoint-##' \
               | grep -E '^[0-9]+$' | sort -n | tail -1)
    if [ -n "$newstep" ] && [ "$newstep" -gt "$LAST_EMIT_STEP" ] 2>/dev/null; then
      STEP="$newstep" emit_event checkpoint || true; LAST_EMIT_STEP="$newstep"
    fi
    sleep "$INTERVAL"
  done ) & WATCH=$!
trap 'kill $WATCH 2>/dev/null || true' EXIT

# preemption trap (SPOT_DESIGN §3.3; best-effort BONUS over the periodic push above,
# which stays the PRIMARY defense). On an EXTERNAL SIGTERM/SIGINT (box preemption /
# docker stop) BEFORE the command returns: emit ONE non-terminal 'preempted' event +
# ONE bounded final flush (no --min-age; trailing-slash dest; --exclude STATUS;
# `timeout 45` so the death path can't hang), then exit. Once the command has
# returned (RC set) the normal terminal path below owns teardown, so the trap
# no-ops — a normal completion never emits 'preempted' (this is TERM/INT, not EXIT).
# `trap -` disarms first so a second signal can't re-enter the handler.
_preempt() {
  trap - TERM INT
  [ -n "${RC:-}" ] && exit "${RC:-0}"
  kill $WATCH 2>/dev/null || true
  emit_event preempted || true
  timeout 45 rclone copy --fast-list --exclude STATUS "$OUT" "$B2/checkpoints/$RUN_ID/" 2>/dev/null || true
  exit 143
}
trap _preempt TERM INT

# substitute {OUT} in the command, then exec it
CMD=(); for a in "$@"; do CMD+=("${a//\{OUT\}/$OUT}"); done
echo ">> [durable] exec: ${CMD[*]}"
emit_event running || true          # once, at train start
"${CMD[@]}"; RC=$?

kill $WATCH 2>/dev/null || true
# final flush (retry a few times — teardown must persist the last checkpoint)
for i in 1 2 3; do
  rclone copy --fast-list --exclude STATUS "$OUT" "$B2/checkpoints/$RUN_ID" && break
  echo "!! [durable] final flush failed ($i/3)"; sleep 10
done
if [ "$RC" -eq 0 ]; then
  rclone copy --fast-list "$OUT" "$B2/artifacts/$RUN_ID" && echo ">> [durable] artifacts -> $B2/artifacts/$RUN_ID"
  _emit_terminal done; status DONE                 # terminal EVENT before STATUS (I4)
else
  _emit_terminal failed "rc=$RC"; status "FAILED rc=$RC"
fi
echo ">> [durable] done rc=$RC"
exit $RC
