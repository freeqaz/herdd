#!/usr/bin/env bash
# serve_ready.sh — the GATE before a paid paired eval.
#
# Waits for a vLLM serve box (launched with a SERVE_ID marker, see
# onstart/serve_vllm.sh) to reach READY, opens the SSH tunnel the driver reads,
# then VERIFIES that every model id you expect is actually being served — the
# [[qwen36-27b-sft-plan]] silent-no-op guard: a mis-named adapter serves base
# weights or all-zeros and the eval scores garbage without ever erroring. On
# success it prints the exact LLM_BASE_URL / LLM_API_KEY that run_paired_eval.sh
# consumes, plus a still-running cost nag. Exit 0 == safe to start the eval.
#
# Usage:
#   tools/vast/serve_ready.sh <SERVE_ID> [flags]
#   tools/vast/serve_ready.sh --base-url URL --expect-models CSV [flags]   # verify-only
#
# Flags:
#   --expect-models CSV   names that must ALL appear in /v1/models
#                         (default: the model list parsed from the READY marker)
#   --expect-ident SHA12  the grade-A identity the box must have VERIFIED on
#                         itself, from the READY marker's `ident=` field. Fails
#                         on absent as well as on mismatched: absent means the
#                         box never gated its own weights, and "no claim" is not
#                         a passing claim. Names prove the LABEL; only this
#                         proves the WEIGHTS. Marker mode only — --base-url has
#                         no marker to read it from.
#   --local-port N        tunnel local port                 (default 28087 = driver TUNNEL_PORT)
#   --timeout SECS        max wait for the READY marker      (default 2700)
#   --poll SECS           marker poll interval               (default 20)
#   --api-key-file PATH   bearer for /v1/models + probe      (default out/serve_api_key.txt)
#   --base-url URL        VERIFY-ONLY: skip B2 poll + instance resolve + tunnel;
#                         verify (+probe) directly against URL (local / pre-tunneled)
#   --no-probe            skip the 1-token completion probe  (default: probe ON)
#   --status-only         poll+print the marker state and exit (round-trip test hook)
#
# Exit codes: 0 ready+verified · 2 usage/env · 3 marker=FAILED · 4 timeout waiting
#             READY · 5 no live instance for label · 6 tunnel failed · 7 expected
#             model missing · 8 probe failed · 9 --expect-ident absent/mismatched.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
ENV="$REPO_ROOT/.env"; [ -f "$ENV" ] && set -a && . "$ENV" && set +a
VCTL="python3 $HERE/herdd.py"

usage() { sed -n '2,40p' "$0" >&2; exit "${1:-2}"; }

# --- args --------------------------------------------------------------------
SERVE_ID=""
EXPECT=""
EXPECT_IDENT=""
LOCAL_PORT=28087
TIMEOUT=2700
POLL=20
API_KEY_FILE="$REPO_ROOT/out/serve_api_key.txt"  # CWD-independent; same file launch_serve writes + the driver reads
API_KEY_FILE_EXPLICIT=0
BASE_URL=""
PROBE=1
STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --expect-models) EXPECT="$2"; shift 2 ;;
    --expect-ident)  EXPECT_IDENT="$2"; shift 2 ;;
    --local-port)    LOCAL_PORT="$2"; shift 2 ;;
    --timeout)       TIMEOUT="$2"; shift 2 ;;
    --poll)          POLL="$2"; shift 2 ;;
    --api-key-file)  API_KEY_FILE="$2"; API_KEY_FILE_EXPLICIT=1; shift 2 ;;
    --base-url)      BASE_URL="${2%/}"; shift 2 ;;
    --no-probe)      PROBE=0; shift ;;
    --status-only)   STATUS_ONLY=1; shift ;;
    -h|--help)       usage 0 ;;
    -*)              echo "!! unknown flag: $1" >&2; usage 2 ;;
    *)               if [ -z "$SERVE_ID" ]; then SERVE_ID="$1"; else echo "!! unexpected arg: $1" >&2; usage 2; fi; shift ;;
  esac
done

# --- bearer token (same file the driver reads, run_paired_eval.sh:124-125) ----
# `launch_serve.sh --serve-id X` writes out/serve_api_key_X.txt, not the bare
# default, so a caller who names a serve-id and omits --api-key-file would
# otherwise poll unauthenticated. Prefer the serve-id file when it exists.
if [ "$API_KEY_FILE_EXPLICIT" -eq 0 ] && [ -n "$SERVE_ID" ] \
   && [ -f "$REPO_ROOT/out/serve_api_key_$SERVE_ID.txt" ]; then
  API_KEY_FILE="$REPO_ROOT/out/serve_api_key_$SERVE_ID.txt"
fi
API_KEY=""
# tr, not $(cat): a trailing newline inside the Authorization header is itself
# a 401, and it makes the key's sha compare unequal to the server's.
[ -f "$API_KEY_FILE" ] && API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"
auth_args() { [ -n "$API_KEY" ] && printf '%s\0%s\0' "-H" "Authorization: Bearer ${API_KEY}"; }

if [ -z "$API_KEY" ]; then
  echo "!! no API key at $API_KEY_FILE — polling UNAUTHENTICATED." >&2
  echo "!! If the server sets VLLM_API_KEY, every poll returns 401 and this" >&2
  echo "!! script reports a timeout that is indistinguishable from a box that" >&2
  echo "!! never booted. Pass --api-key-file if the serve used --serve-id." >&2
fi

# --- verify every expected id is served (curl GET /v1/models + set-compare) ---
# Args: <base_url_with_/v1>. Uses EXPECT (CSV). Returns 0 all present, 7 missing.
verify_models() {
  local base="$1" body raw code
  mapfile -d '' -t _A < <(auth_args)
  # -w the status so an auth refusal can be named. Without this a 401 poll
  # loop and a box that never booted produce the same timeout, and the
  # operator debugs the box instead of the client.
  raw="$(curl -sS -w $'\n%{http_code}' "${_A[@]}" "${base}/models" 2>/dev/null)" || raw=$'\n000'
  code="${raw##*$'\n'}"; body="${raw%$'\n'*}"
  if [ "$code" = "401" ] || [ "$code" = "403" ]; then
    echo "!! GET ${base}/models -> HTTP $code: the SERVER IS ALIVE and rejecting" >&2
    echo "!! this client's credentials. This is NOT a boot failure." >&2
    if [ -n "$API_KEY" ]; then
      echo "!! Sent a bearer from: $API_KEY_FILE" >&2
      echo "!! Compare fingerprints (strip whitespace on BOTH sides, or an" >&2
      echo "!! identical key reads as a mismatch):" >&2
      echo "!!   tr -d '[:space:]' < $API_KEY_FILE | sha256sum" >&2
      echo "!!   herdd ssh <IID> --exec 'printf %s \"\$VLLM_API_KEY\" | sha256sum'" >&2
    else
      echo "!! Sent NO Authorization header (no key file at $API_KEY_FILE)." >&2
    fi
    return 7
  fi
  if [ "$code" != "200" ]; then
    echo "!! GET ${base}/models -> HTTP ${code} (server up? token right?)" >&2; return 7
  fi
  EXPECT="$EXPECT" python3 - "$body" <<'PY' || return 7
import json, os, sys
want = [x for x in os.environ.get("EXPECT", "").split(",") if x]
served = [m.get("id") for m in json.loads(sys.argv[1]).get("data", [])]
missing = [w for w in want if w not in served]
print(">> served models: " + (", ".join(served) or "<none>"), file=sys.stderr)
if missing:
    print("!! MISSING expected model(s): " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print(">> all %d expected model(s) present" % len(want), file=sys.stderr)
PY
  return 0
}

# --- ONE 1-token completion probe against the first expected model -----------
probe() {
  local base="$1" first body
  first="${EXPECT%%,*}"
  mapfile -d '' -t _A < <(auth_args)
  body="$(curl -fsS "${_A[@]}" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${first}\",\"prompt\":\"ping\",\"max_tokens\":1,\"temperature\":0}" \
    "${base}/completions" 2>/dev/null)" || { echo "!! probe POST ${base}/completions failed" >&2; return 8; }
  printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("choices") else 1)' \
    || { echo "!! probe returned no choices: $body" >&2; return 8; }
  echo ">> probe OK: 1-token completion from '${first}'" >&2
}

# ============================ VERIFY-ONLY MODE ===============================
# --base-url: no B2, no herdd, no tunnel — for a local or already-tunneled serve.
if [ -n "$BASE_URL" ]; then
  [ -n "$EXPECT" ] || { echo "!! --base-url requires --expect-models (no marker to default from)" >&2; exit 2; }
  # Refuse rather than silently ignore: an --expect-ident that quietly does
  # nothing is worse than no gate at all, because the caller believes it ran.
  [ -z "$EXPECT_IDENT" ] || { echo "!! --expect-ident needs the SERVE_STATUS marker (the box writes its verified identity there); it cannot be checked in --base-url verify-only mode" >&2; exit 2; }
  verify_models "$BASE_URL" || exit 7
  [ "$PROBE" = "1" ] && { probe "$BASE_URL" || exit 8; }
  echo "======================================================================"
  echo "export LLM_BASE_URL=${BASE_URL}"
  [ -n "$API_KEY" ] && echo "export LLM_API_KEY=\$(cat ${API_KEY_FILE})"
  echo "# verify-only (--base-url) — no tunnel opened, no cost tracking"
  echo "======================================================================"
  exit 0
fi

# ============================ FULL FLOW ======================================
[ -n "$SERVE_ID" ] || { echo "!! SERVE_ID required (or use --base-url)" >&2; usage 2; }
: "${B2_BUCKET:?serve marker needs B2_BUCKET (source tools/vast/.env)}"
: "${B2_KEY_ID:?serve marker needs B2_KEY_ID}"
MARKER="b2:${B2_BUCKET}/serve/${SERVE_ID}/SERVE_STATUS"

# --- the READY line's grammar, in ONE place ----------------------------------
#   READY <ts> <model1,model2,...|-> [ident=<sha12>]
# POSITIONAL and APPEND-ONLY: field 3 has been the model CSV since this marker
# existed, so `ident=` could only ever go after it. `-` is the placeholder the
# box writes when it carried an identity but parsed no model ids, so `ident=`
# is always field 4 when present — a reader never has to guess the column.
# A marker with no 4th field is a box that never gated its own weights.
marker_models() {   # $1 = the READY line
  local m; m="$(printf '%s\n' "$1" | awk '{print $3}')"
  [ "$m" = "-" ] && m=""       # placeholder, not a model named "-"
  printf '%s' "$m"
}
marker_ident() {    # $1 = the READY line -> the sha12, or empty
  printf '%s\n' "$1" | awk '{for (i=4; i<=NF; i++) if ($i ~ /^ident=/) { sub(/^ident=/, "", $i); print $i; exit } }'
}

# --- poll SERVE_STATUS until READY/FAILED/timeout ----------------------------
# Sets MARKER_STATE, MARKER_MODELS and MARKER_IDENT. Exit 3 FAILED · 4 timeout.
MARKER_STATE=""; MARKER_MODELS=""; MARKER_IDENT=""
poll_marker() {
  local deadline line
  deadline=$(( $(date +%s) + TIMEOUT ))
  while :; do
    line="$(rclone cat "$MARKER" 2>/dev/null || true)"
    MARKER_STATE="${line%% *}"
    case "$MARKER_STATE" in
      READY)
        MARKER_MODELS="$(marker_models "$line")"
        MARKER_IDENT="$(marker_ident "$line")"
        echo ">> marker READY: served=${MARKER_MODELS:-<none>} ident=${MARKER_IDENT:-<none>}" >&2
        return 0 ;;
      FAILED)
        echo "!! marker FAILED: ${line#FAILED }" >&2
        exit 3 ;;
      *)
        if [ "$STATUS_ONLY" = "1" ]; then
          echo "state=${MARKER_STATE:-<none>} line='${line}'"; exit 0
        fi
        [ "$(date +%s)" -ge "$deadline" ] && { echo "!! timeout ${TIMEOUT}s waiting for READY (last: '${line:-<no marker>}')" >&2; exit 4; }
        echo "   [$(date -u +%H:%M:%S)] marker=${MARKER_STATE:-<none>} — waiting ${POLL}s" >&2
        sleep "$POLL" ;;
    esac
  done
}

# --status-only prints the current state once and exits (round-trip test hook).
if [ "$STATUS_ONLY" = "1" ]; then
  line="$(rclone cat "$MARKER" 2>/dev/null || true)"
  st="${line%% *}"
  case "$st" in FAILED) echo "state=FAILED line='${line}'"; exit 3 ;; esac
  echo "state=${st:-<none>} models=$(marker_models "$line") ident=$(marker_ident "$line") line='${line}'"
  exit 0
fi

poll_marker

# --- identity gate: what the BOX verified about its own weights --------------
# Runs before the tunnel and before /v1/models, because it is the only check
# here that is not a name comparison. A serve whose weights are wrong passes
# --expect-models, passes the probe, and scores like the baseline.
if [ -n "$EXPECT_IDENT" ]; then
  if [ -z "$MARKER_IDENT" ]; then
    echo "!! --expect-ident $EXPECT_IDENT but the READY marker carries NO ident= field." >&2
    echo "!!   This box never verified its own weights — either it was launched without" >&2
    echo "!!   --model-artifact, or it is running a serve payload from before the gate." >&2
    echo "!!   Absent is a FAILURE here on purpose: 'no claim' is not a passing claim." >&2
    exit 9
  fi
  if [ "$MARKER_IDENT" != "$EXPECT_IDENT" ]; then
    echo "!! IDENTITY MISMATCH: box verified '$MARKER_IDENT', expected '$EXPECT_IDENT'." >&2
    echo "!!   The box proved it is serving SOMETHING coherently — just not the artifact" >&2
    echo "!!   you are about to score against. Do not start the eval." >&2
    exit 9
  fi
  echo ">> identity OK: box-verified grade-A fingerprint $MARKER_IDENT" >&2
fi

[ -n "$EXPECT" ] || EXPECT="$MARKER_MODELS"
[ -n "$EXPECT" ] || { echo "!! no --expect-models and READY marker carried no model list" >&2; exit 2; }

# --- resolve the live instance by its serve:<id> label -----------------------
resolve_instance() {
  local iid
  iid="$($VCTL ls --json 2>/dev/null | SERVE_ID="$SERVE_ID" python3 -c '
import json, os, sys
want = "serve:" + os.environ["SERVE_ID"]
live = {"running", "loading", "created"}
for i in json.load(sys.stdin):
    if (i.get("label") or "") == want and (i.get("actual_status") or "").lower() in live:
        print(i.get("id")); break
' 2>/dev/null)"
  [ -n "$iid" ] || { echo "!! no LIVE instance labelled serve:${SERVE_ID}" >&2; exit 5; }
  echo "$iid"
}
IID="$(resolve_instance)"
echo ">> serve instance: ${IID} (label serve:${SERVE_ID})" >&2

# --- cost nag (orphan serve boxes are the runbook's #1 loss) -----------------
cost_nag() {
  local dph
  dph="$($VCTL show "$IID" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("dph_total") or "?")' 2>/dev/null || echo "?")"
  echo ">> serve box ${IID} still running at \$${dph}/hr — after the eval: python3 tools/vast/herdd.py destroy ${IID} -y" >&2
}

# --- open the tunnel (background; the driver needs it to stay up) -------------
PIDFILE="$REPO_ROOT/out/serve_tunnel_${SERVE_ID}.pid"
open_tunnel() {
  mkdir -p "$REPO_ROOT/out"
  # herdd tunnel execs `ssh -N -L <local>:localhost:<remote>` — nohup+& detaches it.
  # shellcheck disable=SC2086  # $VCTL is "python3 <path>" and MUST word-split into argv
  nohup $VCTL tunnel "$IID" --local "$LOCAL_PORT" --remote 8000 >"$REPO_ROOT/out/serve_tunnel_${SERVE_ID}.log" 2>&1 &
  echo "$!" > "$PIDFILE"
  for _ in $(seq 1 30); do
    if curl -fsS -o /dev/null "http://127.0.0.1:${LOCAL_PORT}/v1/models" 2>/dev/null \
       || bash -c ": >/dev/tcp/127.0.0.1/${LOCAL_PORT}" 2>/dev/null; then
      echo ">> tunnel up: 127.0.0.1:${LOCAL_PORT} -> instance ${IID} :8000 (pid $(cat "$PIDFILE"))" >&2
      return 0
    fi
    sleep 1
  done
  echo "!! tunnel did not come up on 127.0.0.1:${LOCAL_PORT} within 30s (see $REPO_ROOT/out/serve_tunnel_${SERVE_ID}.log)" >&2
  return 6
}
open_tunnel || { cost_nag; exit 6; }

# on a failed gate the box is still billing and the tunnel is still up — nag before exiting
gate_fail() {
  echo "!! gate failed — tunnel pid $(cat "$PIDFILE" 2>/dev/null || echo '?') still up (kill: kill \$(cat ${PIDFILE}))" >&2
  cost_nag
  exit "$1"
}

BASE="http://127.0.0.1:${LOCAL_PORT}/v1"
verify_models "$BASE" || gate_fail 7
[ "$PROBE" = "1" ] && { probe "$BASE" || gate_fail 8; }

echo "======================================================================"
echo "export LLM_BASE_URL=http://127.0.0.1:${LOCAL_PORT}/v1"
echo "export LLM_API_KEY=\$(cat ${API_KEY_FILE})"
echo "# tunnel pid $(cat "$PIDFILE") (kill: kill \$(cat ${PIDFILE}))  — leave it up for the driver"
echo "======================================================================"
cost_nag
exit 0
