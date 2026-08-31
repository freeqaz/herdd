#!/usr/bin/env bash
# onstart/eval_job_lib.sh — sourceable phase library for box-side eval jobs.
#
# Every eval job today hand-rolls the same serve -> gate -> (probe) -> teardown
# orchestration (h1_paired_eval/run.sh ~259 lines, base-reader-train/train.sh
# run_eval, sampler-concurrency-ab). This library factors those phases into
# functions so an eval entrypoint just SOURCEs it and calls the phases in order:
#
#     source "$(dirname "$0")/eval_job_lib.sh"   # or the onstart/ copy
#     ejl_serve_up        || exit 4              # base+LoRA up (delegates job_serve.sh)
#     ejl_gate            || exit 5              # all models present + probe (+divergence)
#     ...run the probe/grade the job actually owns...
#     ejl_teardown                               # idempotent serve kill (also trap it)
#
# Design constraints:
#   * DEPENDENCY-LIGHT bash — only curl + python3 (stdlib json), same as the rest
#     of onstart/. No jq, no extra pips.
#   * SOURCEABLE — defines functions only; sets NO shell options (`set -e`/`-u`)
#     that would leak into the caller. Every function returns a status; callers
#     decide fatality.
#   * Breadcrumb lines carry a STABLE prefix ("[ejl]") so `herdd job logs`
#     reads like phases (grep '\[ejl\]' for the eval timeline).
#   * IDEMPOTENT / resume-safe — ejl_serve_up re-brings-up after a preemption
#     (job_serve.sh kills any stale server first); ejl_teardown is a no-op when
#     nothing is running.
#
# Serve config is passed the SAME way job_serve.sh reads it — via exported env
# (MODEL_B2, SERVED_NAME, LORA_SPECS, MAX_LORA_RANK, KV_CACHE_DTYPE, MAX_LEN,
# GPU_UTIL, VLLM_API_KEY, ...). See job_serve.sh's header for the contract.
#
# Gate/divergence env (all optional except EJL_EXPECT_MODELS for ejl_gate):
#   EJL_BASE_URL            default http://127.0.0.1:8000/v1
#   EJL_EXPECT_MODELS       CSV of ids that must ALL appear on /v1/models
#                           (or pass as ejl_gate's first arg)
#   EJL_API_KEY             bearer (falls back to VLLM_API_KEY)
#   EJL_GATE_TIMEOUT        seconds to wait for all models (default 900)
#   EJL_PROBE               1 (default) = 1-token /v1/completions probe on id #1
#   EJL_DIVERGENCE          "base_id=lora_id" => assert greedy chat output DIFFERS
#                           (the silent-no-op adapter guard; mirrors divergence_smoke.sh)
#   EJL_DIVERGENCE_PROMPT   override the divergence prompt
#   EJL_DIVERGENCE_MAX_TOKENS default 64
#   EJL_JOB_SERVE           path to job_serve.sh (default: sibling of this lib,
#                           then PATH, then the onstart/ copy)

EJL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EJL_PREFIX="[ejl]"

ejl_log()  { echo ">> $EJL_PREFIX $(date -u +%T) $*"; }
ejl_warn() { echo "!! $EJL_PREFIX $(date -u +%T) $*" >&2; }

# _ejl_resolve <script-name> <env-override> — env override, then PATH, then sibling.
_ejl_resolve() {
  local name="$1" override="$2" p
  if [ -n "$override" ]; then echo "$override"; return; fi
  p="$(command -v "$name" 2>/dev/null || true)"
  [ -n "$p" ] || p="$EJL_HERE/$name"
  echo "$p"
}

# --- phase: serve-up (delegates to job_serve.sh) ------------------------------
# job_serve.sh builds/reuses the serve venv, kills any stale server, launches
# serve_vllm.sh detached, and blocks on its OWN /v1/models readiness gate. So on
# return the base (+ any LORA_SPECS) is already listed; ejl_gate then applies the
# finer all-expected + probe + divergence checks. Safe to call on every attempt.
ejl_serve_up() {
  local js; js="$(_ejl_resolve job_serve.sh "${EJL_JOB_SERVE:-}")"
  [ -f "$js" ] || { ejl_warn "serve_up: job_serve.sh not found ($js)"; return 2; }
  ejl_log "serve_up: base=${SERVED_NAME:-?} lora=${LORA_SPECS:-<none>} via $js"
  if bash "$js"; then
    ejl_log "serve_up: serve READY"
    return 0
  fi
  ejl_warn "serve_up: job_serve.sh failed (see ${SERVE_LOG:-/workspace/serve.log})"
  return 4
}

# --- helpers for the gate -----------------------------------------------------
# _ejl_all_present <want-csv> <have-csv> — every want id appears in have.
_ejl_all_present() {
  local want="$1" have="$2" m
  [ -n "$have" ] || return 1
  local IFS=,
  for m in $want; do
    [ -n "$m" ] || continue
    case ",$have," in *",$m,"*) : ;; *) return 1 ;; esac
  done
  return 0
}

# _ejl_models_csv <base_url> [auth...] — echo the served model-id CSV, or nothing.
_ejl_models_csv() {
  local base_url="$1"; shift
  curl -fsS "$@" "$base_url/models" -o /tmp/ejl_models.json 2>/dev/null || return 1
  python3 -c 'import json,sys
try:
    d=json.load(open("/tmp/ejl_models.json"))
    print(",".join(m.get("id","") for m in d.get("data",[])))
except Exception:
    sys.exit(1)' 2>/dev/null
}

# _ejl_probe <base_url> <model> [auth...] — a 1-token completion must succeed.
_ejl_probe() {
  local base_url="$1" model="$2"; shift 2
  local body
  body="$(python3 -c 'import json,sys;print(json.dumps({"model":sys.argv[1],"prompt":"ping","max_tokens":1,"temperature":0}))' "$model")"
  curl -fsS "$@" -H "Content-Type: application/json" -d "$body" \
    "$base_url/completions" -o /dev/null 2>/dev/null
}

# _ejl_chat <base_url> <model> <prompt> <max_tokens> [auth...] — greedy chat text.
_ejl_chat() {
  local base_url="$1" model="$2" prompt="$3" mx="$4"; shift 4
  local body
  body="$(python3 -c 'import json,sys;print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":int(sys.argv[3]),"temperature":0,"seed":0}))' "$model" "$prompt" "$mx")"
  curl -fsS "$@" -H "Content-Type: application/json" -d "$body" \
    "$base_url/chat/completions" 2>/dev/null \
    | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["choices"][0]["message"]["content"])
except Exception:
    sys.exit(1)' 2>/dev/null
}

# --- phase: readiness gate ----------------------------------------------------
# ejl_gate [expect-csv] — poll /v1/models until ALL expected ids present (bounded),
# then a 1-token /v1/completions probe (EJL_PROBE=1), then an OPTIONAL divergence
# check (EJL_DIVERGENCE="base=lora"). Return: 0 pass · 2 no expect list · 6
# divergence fail · 7 models never present · 8 probe fail.
ejl_gate() {
  local expect="${1:-${EJL_EXPECT_MODELS:-}}"
  local base_url="${EJL_BASE_URL:-http://127.0.0.1:8000/v1}"; base_url="${base_url%/}"
  local timeout="${EJL_GATE_TIMEOUT:-900}"
  local api_key="${EJL_API_KEY:-${VLLM_API_KEY:-}}"
  [ -n "$expect" ] || { ejl_warn "gate: no expected model ids (EJL_EXPECT_MODELS)"; return 2; }
  local -a auth=(); [ -n "$api_key" ] && auth=(-H "Authorization: Bearer $api_key")

  ejl_log "gate: waiting up to ${timeout}s for models [$expect] at $base_url/models"
  local deadline got=""
  deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    got="$(_ejl_models_csv "$base_url" "${auth[@]}" || true)"
    if _ejl_all_present "$expect" "$got"; then
      ejl_log "gate: all expected models present ($got)"
      break
    fi
    sleep 3
  done
  _ejl_all_present "$expect" "$got" \
    || { ejl_warn "gate: models never all present (want [$expect] got [${got:-<none>}])"; return 7; }

  if [ "${EJL_PROBE:-1}" = "1" ]; then
    local first="${expect%%,*}"
    ejl_log "gate: 1-token /v1/completions probe on '$first'"
    _ejl_probe "$base_url" "$first" "${auth[@]}" \
      || { ejl_warn "gate: 1-token probe failed on '$first'"; return 8; }
  fi

  if [ -n "${EJL_DIVERGENCE:-}" ]; then
    local a="${EJL_DIVERGENCE%%=*}" b="${EJL_DIVERGENCE#*=}"
    local prompt="${EJL_DIVERGENCE_PROMPT:-Rewrite this C++ function body to match the target object code exactly. Output only the function.}"
    local mx="${EJL_DIVERGENCE_MAX_TOKENS:-64}"
    ejl_log "gate: divergence check '$a' vs '$b' (silent-no-op adapter guard)"
    local oa ob
    oa="$(_ejl_chat "$base_url" "$a" "$prompt" "$mx" "${auth[@]}")" \
      || { ejl_warn "gate: divergence request to '$a' failed"; return 6; }
    ob="$(_ejl_chat "$base_url" "$b" "$prompt" "$mx" "${auth[@]}")" \
      || { ejl_warn "gate: divergence request to '$b' failed"; return 6; }
    if [ -z "$oa" ] || [ "$oa" = "$ob" ]; then
      ejl_warn "gate: DIVERGENCE FAIL — '$a' == '$b' greedy output (adapter is a silent no-op)"
      return 6
    fi
    ejl_log "gate: divergence OK ($a != $b)"
  fi

  ejl_log "gate: PASSED"
  return 0
}

# --- phase: teardown ----------------------------------------------------------
# Idempotent serve kill. Call at the end AND trap it (trap 'ejl_teardown' EXIT)
# so a mid-eval death still frees the GPU for the next attempt.
ejl_teardown() {
  ejl_log "teardown: stopping vllm/haproxy (idempotent)"
  pkill -f 'vllm serve' 2>/dev/null || true
  pkill -x haproxy      2>/dev/null || true
  return 0
}
