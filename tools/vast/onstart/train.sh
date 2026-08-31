#!/usr/bin/env bash
# onstart/train.sh — compute-efficient training entrypoint for a Vast instance.
#
# Pre-staged run in B2 (runsets/<RUNSET>/, stage_run.sh): pulls once, resumes
# from latest checkpoint if outbid, streams checkpoints out, pushes artifact,
# self-parks.
#
# Teardown (2026-07-10 default change: SUSPEND, don't destroy): DONE -> self-PARK
# (stop via Vast CONTAINER_API_KEY: GPU billing ends, disk/HF-cache/env stay warm
# for `herdd start`; storage still bills until destroyed — `herdd ls` flags it).
# FAILED/STAGED -> holds FAIL_HOLD_MINUTES (default 15; KEEP_ON_FAIL=1 = indefinite)
# then self-parks; MAX_HOURS (default 24) hard watchdog also parks. TEARDOWN=destroy
# restores the old self-destruct; TEARDOWN=keep (or KEEP_ON_DONE=1) leaves the box
# RUNNING. A park that doesn't take within 180s falls back to self-destruct so a
# wedged box can't bill GPU forever. A parked box re-runs onstart on resume: the
# resume guard (/workspace/.run_terminal) idles instead of re-running TRAIN_CMD.
# Checkpoints stream every 3 min (resume by RUN_ID). `herdd train --babysit`
# mirrors STATUS. Debug: tools/vast/debug_box.sh {stop,extend} <RUN_ID>.
#
# Required env (`herdd launch --env` or launch_train.sh):
#   RUN_ID              unique id for this run (checkpoints/<RUN_ID>)
#   RUNSET              name of the pre-staged bundle in B2 (runsets/<RUNSET>)
#   B2_KEY_ID B2_APPLICATION_KEY B2_BUCKET B2_S3_ENDPOINT B2_REGION
# Optional:
#   BASE_MODEL_B2       B2 subpath of pre-staged base model (skips HF download)
#   HF_TOKEN            for gated/base-model HF pulls
#   TRAIN_CMD           override training command (default: runset/train.sh)
#   FAST_BOOT=1         rehydrate train stack from b2:.../train-env (see 0b +
#                       onstart/rehydrate_train_env.sh) instead of fat image
#   TEARDOWN=park|destroy|keep
#                       terminal action on DONE/FAILED/MAX_HOURS. Default park
#                       (self-stop; resume with `herdd start`). destroy = old
#                       self-destruct. keep = leave RUNNING (GPU billing!).
#   FAIL_HOLD_MINUTES=N debug window on FAILED before teardown (default
#                       15; 0=immediate). KEEP_ON_FAIL=1 = indefinite.
#                       stop/extend: tools/vast/debug_box.sh.
#   KEEP_ON_DONE=1      legacy alias for TEARDOWN=keep on the DONE path only.
#   MAX_HOURS=N         hard teardown cap (default 24; 0 disables)
#   EVAL_TARGETS        comma list (dc3|rb3|rb3-xenon): compile+score evals on
#                       idle CPU via eval sidecar. Also EVAL_GRACE_MINUTES/
#                       EVAL_CMD/EVAL_JOBS/EVAL_ENV_VER (onstart/eval_sidecar.sh).
set -uo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1
echo "=== onstart $(date -u) RUN_ID=${RUN_ID:-?} RUNSET=${RUNSET:-?} ==="
env | grep -E '^(B2_|RUN_ID|RUNSET|HF_|WANDB_|BASE_MODEL_B2|TRAIN_CMD|TEARDOWN|KEEP_ON_|FAIL_HOLD_MINUTES|MAX_HOURS|CONTAINER_|EVAL_|LLM_|OPENROUTER_|CPU_FARM|FARM_)' >> /etc/environment || true

WS=/workspace; RUNSET_DIR=$WS/runset; CKPT_DIR=$WS/out/${RUN_ID}
mkdir -p "$RUNSET_DIR" "$CKPT_DIR"

# --- boot-phase timing (PURE-LOCAL; made visible off-box by the boot pusher below) ---
# The 15-18min boot used to be one opaque gap: STATUS said only 'RUNNING', runmeta
# had zero events between 'launched' and 'running' (train start), and onstart.log
# reached B2 only at teardown. boot_mark appends '<epoch>\t<elapsed_s>\t<phase>\t
# <detail>' to a local TSV and echoes '>> [boot +<Ns>] <phase> <detail>' to the log.
# It NEVER calls B2 (rclone may not exist yet) and must NEVER fail the boot (|| true).
BOOT_T0=$(date +%s)
BOOT_PHASES=/workspace/boot_phases.tsv
boot_mark() {                       # boot_mark <phase> [detail]
  local _now _el _phase=$1 _detail=${2:-}
  _now=$(date +%s); _el=$(( _now - BOOT_T0 ))
  printf '%s\t%s\t%s\t%s\n' "$_now" "$_el" "$_phase" "$_detail" >> "$BOOT_PHASES" 2>/dev/null || true
  echo ">> [boot +${_el}s] ${_phase} ${_detail}"
}
boot_mark onstart_start "RUN_ID=${RUN_ID:-?} RUNSET=${RUNSET:-?}"

# self-teardown via per-instance CONTAINER_API_KEY; curl-only.
# Default action is PARK (stop): GPU billing ends, disk kept for `herdd start`.
# TEARDOWN=destroy restores the old self-destruct; a park that doesn't take
# within 180s falls back to destroy so a wedged box can't bill GPU forever
# (a SUCCESSFUL stop kills this process mid-sleep, so the fallback never runs).
VAPI=https://console.vast.ai/api/v0/instances
_iid_key() {
  IID="${INSTANCE_ID:-${CONTAINER_ID:-}}"; KEY="${VASTAI_API_KEY:-${CONTAINER_API_KEY:-}}"
  [ -z "$KEY" ] && [ -f ~/.vast_api_key ] && KEY="$(cat ~/.vast_api_key)"
  [ -n "$IID" ] && [ -n "$KEY" ]
}
self_destruct() {
  _iid_key || { echo "!! no iid/key — run: herdd destroy <id> -y"; return 1; }
  echo ">> self-destruct ${IID}"
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X DELETE -H "Authorization: Bearer ${KEY}" "$VAPI/${IID}/" || true
}
self_park() {
  _iid_key || { echo "!! no iid/key — run: herdd stop <id>"; return 1; }
  echo ">> self-park ${IID} (resume: herdd start ${IID})"
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X PUT -H "Authorization: Bearer ${KEY}" -H 'Content-Type: application/json' \
    -d '{"state":"stopped"}' "$VAPI/${IID}/" || true
  sleep 180
  echo "!! park failed — self-destructing"
  self_destruct
}
self_teardown() {
  case "${TEARDOWN:-park}" in destroy) self_destruct ;; keep) : ;; *) self_park ;; esac
}

# resume guard: a parked box re-runs onstart on `herdd start`. If this run
# already went terminal, idle for interactive/eval reuse instead of re-running
# TRAIN_CMD (which would re-pull, re-train and re-emit terminal events). The
# idle box re-parks itself at MAX_HOURS (no STATUS/event writes — those are
# already terminal on B2 and must not be clobbered).
if [ -f /workspace/.run_terminal ]; then
  echo ">> resume guard: terminal ($(cat /workspace/.run_terminal)) — idle, no re-run"
  [ "${MAX_HOURS:-24}" != "0" ] && { ( sleep "$(( ${MAX_HOURS:-24} * 3600 ))"; self_park ) & }
  sleep infinity
  exit 0
fi

# hard runtime cap
MAX_HOURS="${MAX_HOURS:-24}"
if [ "$MAX_HOURS" != "0" ]; then
  ( sleep "$(( MAX_HOURS * 3600 ))"
    echo "!! MAX_HOURS=${MAX_HOURS} — teardown (${TEARDOWN:-park})"
    boot_mark max_hours_kill "MAX_HOURS=${MAX_HOURS}"
    # UNFORGEABLE TERMINAL (AUTOMATION_PLAN Phase 2): the watchdog forks before
    # status()/emit_event() are defined, so write the terminal STATUS + event
    # INLINE here. Without this the box just 404s with STATUS stuck at RUNNING —
    # indistinguishable from an eviction, which would trigger an endless relaunch
    # loop burning a full MAX_HOURS window each time. reason=max_hours (no free text).
    if [ -n "${B2_KEY_ID:-}" ]; then
      echo "FAILED max-hours $(date -u +%FT%TZ)" \
        | rclone rcat "b2:${B2_BUCKET}/checkpoints/${RUN_ID}/STATUS" 2>/dev/null || true
      _wts=$(date -u +%Y%m%dT%H%M%S%3NZ); _wn=$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')
      _wi="${INSTANCE_ID:-${CONTAINER_ID:-unknown}}"
      printf '{"v":1,"ts":"%s","actor":"box:%s","event":"failed","run_id":"%s","nonce":"%s","reason":"max_hours"}\n' \
        "$_wts" "$_wi" "$RUN_ID" "$_wn" \
        | rclone rcat "b2:${B2_BUCKET}/runs/${RUN_ID}/events/${_wts}-box_${_wi}-${_wn}.json" 2>/dev/null || true
      # the watchdog never pushed onstart.log before — a MAX_HOURS kill during a
      # wedged boot left ZERO off-box log. Best-effort push both (inline, same style).
      rclone rcat "b2:${B2_BUCKET}/checkpoints/${RUN_ID}/onstart.log" < /workspace/onstart.log 2>/dev/null || true
      rclone rcat "b2:${B2_BUCKET}/checkpoints/${RUN_ID}/boot_phases.tsv" < /workspace/boot_phases.tsv 2>/dev/null || true
    fi
    echo "FAILED max_hours $(date -u +%FT%TZ)" > /workspace/.run_terminal
    # even TEARDOWN=keep parks here — MAX_HOURS is the forgotten-box net
    if [ "${TEARDOWN:-park}" = destroy ]; then self_destruct; else self_park; fi ) & WATCHDOG=$!
  echo ">> watchdog pid $WATCHDOG (${TEARDOWN:-park} @ ${MAX_HOURS}h)"
fi

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


status() { [ -n "${B2_KEY_ID:-}" ] && echo "$1 $(date -u +%FT%TZ)" | \
  rclone rcat "b2:${B2_BUCKET}/checkpoints/${RUN_ID}/STATUS" 2>/dev/null || true; }

# --- append-only run-metadata events (runs/<RUN_ID>/events/, see runmeta.py) ---
# The box is a no-repo actor, so it emits via this inline bash helper (NOT a
# pulled runmeta.py — a B2 pull would fail exactly when a box is dying). One
# immutable object per event; unique urandom nonce => concurrent writers never
# collide. Returns rclone's rc so callers can retry a terminal emit.
_json_str() {                       # stdin -> JSON string literal (escape reason)
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys;sys.stdout.write(json.dumps(sys.stdin.read().rstrip("\n")))'
  else
    local s; s=$(cat); s=${s//\\/\\\\}; s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}; s=${s//$'\r'/\\r}; s=${s//$'\t'/\\t}; printf '"%s"' "$s"
  fi
}
emit_event() {                      # emit_event <event> [free-text reason]; env: STEP,DPH
  [ -n "${B2_KEY_ID:-}" ] || return 0
  local ev=$1 reason=${2:-} ts nonce iid actor key
  ts=$(date -u +%Y%m%dT%H%M%S%3NZ)                          # colon-free, ms (== runmeta now_ts)
  nonce=$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')        # NOT $RANDOM (15-bit, collides)
  iid="${INSTANCE_ID:-${CONTAINER_ID:-unknown}}"; actor="box:${iid}"
  key="runs/${RUN_ID}/events/${ts}-box_${iid}-${nonce}.json"
  { printf '{"v":1,"ts":"%s","actor":"%s","event":"%s","run_id":"%s","nonce":"%s"' \
      "$ts" "$actor" "$ev" "$RUN_ID" "$nonce"
    [ -n "$reason"   ] && printf ',"reason":%s' "$(printf '%s' "$reason" | _json_str)"
    [ -n "${STEP:-}" ] && printf ',"step":%s'   "$STEP"
    [ -n "${DPH:-}"  ] && printf ',"dph":%s'    "$DPH"
    # boot heartbeat fields (runmeta fold tolerates unknown fields; PHASE is a
    # string, the rest are integers from shell arithmetic == valid JSON numbers).
    [ -n "${PHASE:-}"    ] && printf ',"phase":%s'   "$(printf '%s' "$PHASE" | _json_str)"
    [ -n "${T_BOOT_S:-}" ] && printf ',"t_boot_s":%s' "$T_BOOT_S"
    [ -n "${BYTES:-}"    ] && printf ',"bytes":%s'   "$BYTES"
    [ -n "${SECS:-}"     ] && printf ',"secs":%s'    "$SECS"
    [ -n "${MBPS:-}"     ] && printf ',"mbps":%s'    "$MBPS"
    # host_metrics: compact "gpu_util:..,cpu:..,net_rx:.." string (a JSON string,
    # not a number). Folded as last_metrics; the runs view tolerates it.
    [ -n "${HOST_METRICS:-}" ] && printf ',"host_metrics":%s' "$(printf '%s' "$HOST_METRICS" | _json_str)"
    printf '}\n'
  } | rclone rcat "b2:${B2_BUCKET}/${key}" 2>/dev/null
}
host_metrics() {   # compact k:v host-metrics line for a heartbeat's host_metrics
  # Prefer the full probe (GPU + cpu/net/disk) if it's on the box; else a GPU-only
  # inline nvidia-smi fallback so at least utilization is captured. Never fails —
  # a metrics read must not wedge the checkpoint watcher.
  local probe
  for probe in "${METRICS_PROBE:-}" \
               "$(dirname "$0")/../metrics_probe.py" \
               "$(dirname "$0")/metrics_probe.py" \
               /workspace/eval/upstream-monorepo/tools/vast/metrics_probe.py; do
    [ -n "$probe" ] && [ -f "$probe" ] || continue
    python3 "$probe" fields 2>/dev/null && return 0
  done
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  # power.limit and name ride along even in the fallback: without the absolute
  # cap, a host that lowered it is indistinguishable from a healthy box, and
  # without the name a throughput number cannot be attributed to a SKU once the
  # box is destroyed (PERF_LEVERS_INVESTIGATION_2026-08-06.md §2.4). The name is
  # squeezed to field-safe characters — no '=', space or ','.
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.limit,name \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' 'NR==1{
        for(i=1;i<=NF;i++){gsub(/^ +| +$/,"",$i)}
        n=$5; sub(/^NVIDIA +/,"",n); gsub(/[^A-Za-z0-9_.+-]+/,"_",n); gsub(/^_+|_+$/,"",n)
        printf "gpu_util:%s,gpu_mem:%s,gpu_temp:%s",$1,$2,$3
        if($4!="" && $4!="[N/A]") printf ",gpu_plim:%d",$4
        if(n!="") printf ",gpu:%s",n
      }'
}
_emit_terminal() {                  # terminal event with 3x retry (I4); best-effort
  local ev=$1 reason=${2:-} _i
  for _i in 1 2 3; do emit_event "$ev" "$reason" && return 0; sleep 5; done; return 0
}

# --- 0. rclone remote (idempotent) ---
command -v rclone >/dev/null || rclone_bootstrap
if ! command -v rclone >/dev/null; then
  echo "!! rclone install failed — no B2; teardown (${TEARDOWN:-park})"
  echo "FAILED no_rclone $(date -u +%FT%TZ)" > /workspace/.run_terminal
  [ "${KEEP_ON_FAIL:-0}" = "1" ] || self_teardown
  exit 1
fi
# Honour RCLONE_CONFIG: rclone itself reads it, so a writer that hardcodes $HOME
# writes one file while every later `rclone` call reads another.
RCONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
mkdir -p "$(dirname "$RCONF")"
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
B2="b2:${B2_BUCKET}"
status RUNNING
boot_mark rclone_ready "$(rclone version 2>/dev/null | head -1)"

# --- b2x transport (optional; every use falls back to its rclone line) --------
# Pulled from B2 like the other companions (preempt_trap.sh, resume_pull.sh)
# so a box on an OLD image still gets it. b2x_ensure then finds the baked
# /usr/local/bin/b2x on new images and skips the fetch entirely.
# eval-env/ is where the eval-env bake ships companions; tools/b2x/ is where
# publish.sh puts it, so the transport is usable BEFORE the next env rebake.
# cdn_pull.py FIRST: b2x_boot.sh resolves it beside itself at source time, and
# /workspace is where this stages both. Without it rung 0 logs a miss.
rclone copyto "$B2/eval-env/cdn_pull.py" /workspace/cdn_pull.py 2>/dev/null || true
{ { rclone copyto "$B2/eval-env/b2x_boot.sh" /workspace/b2x_boot.sh 2>/dev/null \
    || rclone copyto "$B2/tools/b2x/b2x_boot.sh" /workspace/b2x_boot.sh 2>/dev/null; } \
  && . /workspace/b2x_boot.sh; } || true
command -v b2x_pull >/dev/null 2>&1 || { b2x_pull() { return 1; }; b2x_push() { return 1; }; }
b2x_ensure >/dev/null 2>&1 && boot_mark b2x_ready "$("$B2X" version 2>/dev/null)" || true

# --- handoff guards (HANDOFF_DESIGN §4/§6, task T6; NO-OP off the handoff path) ---
# A run under `--handoff` migrates from an expensive PRIMARY box to a cheaper
# UNDERSTUDY (herdd cmd_supervise is the driver). It sets two box-side contracts
# at understudy-launch / handoff time; BOTH are unset on every ordinary run, so the
# whole block below is inert unless a handoff is actually in flight:
#   HANDOFF_EPOCH  monotonic write-generation counter for THIS box. The driver
#                  stamps runs/<RUN_ID>/handoff/<epoch>.json at each ARM (§4); a box
#                  may push to the run's B2 state only while NO strictly-greater
#                  epoch marker exists — else the understudy that superseded it owns
#                  the state. This is the two-writer guard that neuters a parked-husk
#                  auto-resume writing over the understudy after cutover (§4 interleave 3).
#   HANDOFF_TTL_S  dead-man deadline (= HANDOFF_DEADLINE_S + margin) for an UNDERSTUDY:
#                  if no cutover/promotion marker appears by TTL the supervisor
#                  probably died mid-handoff, so self-park rather than run forever
#                  double-billing next to the still-live primary (§6 box-side dead-man).
HANDOFF_EPOCH="${HANDOFF_EPOCH:-}"
# _handoff_epoch_stale: rc 0 == STALE (a newer epoch was PROMOTED over this box —
# the sync site must REFUSE its push); rc 1 == ok to push. Keyed on the `promoted`
# marker's "epoch" field, NOT the max ARM-time <epoch>.json: write ownership
# transfers at PROMOTION (driver writes runs/<ID>/handoff/promoted at cutover),
# and keying on the ARM marker broke a SECOND handoff (HANDOFF_MAX=2) two ways —
# the still-canonical epoch-N primary went silent the moment N+1.json was stamped
# (losing every periodic push for the whole 2x window), and an ABORTED attempt
# left N+1.json behind with no cleanup, silencing the surviving primary for the
# rest of the run. <epoch>.json markers remain as arm-time telemetry only.
# >>> handoff-epoch-stale
# FAIL-SAFE: unset HANDOFF_EPOCH (every normal run), a missing/unreadable promoted
# marker, or an unparsable epoch field => "not stale" => the push proceeds exactly
# as before. A rerun of the same RUN_ID launched WITHOUT HANDOFF_EPOCH is likewise
# never refused by a leftover promoted marker from an earlier campaign.
_handoff_epoch_stale() {
  [ -n "$HANDOFF_EPOCH" ] || return 1
  local pj pe
  pj=$(rclone cat "$B2/runs/${RUN_ID}/handoff/promoted" 2>/dev/null) || return 1
  pe=$(printf '%s' "$pj" | sed -n 's/.*"epoch":[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)
  [ -n "$pe" ] || return 1
  [ "$pe" -gt "$HANDOFF_EPOCH" ] 2>/dev/null
}
# <<< handoff-epoch-stale
# _handoff_stamp_epoch: record which epoch owns the state we just pushed, alongside
# the synced checkpoints (best-effort). No-op off the handoff path.
_handoff_stamp_epoch() {
  [ -n "$HANDOFF_EPOCH" ] || return 0
  printf '%s\n' "$HANDOFF_EPOCH" \
    | rclone rcat "$B2/checkpoints/${RUN_ID}/HANDOFF_EPOCH" 2>/dev/null || true
}
# >>> handoff-deadman-watchdog (understudy-only dead-man; see block comment)
# Park (resumable), NOT destroy: a false-positive park is cheap to undo with
# `herdd start`; a destroy would burn the warm understudy. The promotion signal
# is a B2 marker the driver writes at CUTOVER completion (runs/<RUN_ID>/handoff/
# promoted) — present => this box became canonical and must keep running.
if [ -n "${HANDOFF_TTL_S:-}" ] && [ "${HANDOFF_TTL_S}" -gt 0 ] 2>/dev/null; then
  ( sleep "${HANDOFF_TTL_S}"
    if rclone lsf "$B2/runs/${RUN_ID}/handoff/promoted" 2>/dev/null | grep -q .; then
      echo ">> handoff dead-man: promotion marker present after ${HANDOFF_TTL_S}s — this box is canonical, staying up"
      exit 0
    fi
    echo "!! handoff dead-man: no cutover/promotion within HANDOFF_TTL_S=${HANDOFF_TTL_S}s — supervisor likely died; self-parking (stops double-bill)"
    emit_event handoff_deadman "no cutover/promotion within HANDOFF_TTL_S=${HANDOFF_TTL_S}s" || true
    self_park ) & HANDOFF_DEADMAN=$!
  echo ">> handoff dead-man watchdog pid $HANDOFF_DEADMAN (park @ ${HANDOFF_TTL_S}s unless promoted)"
fi
# <<< handoff-deadman-watchdog

# --- preemption trap (SPOT_DESIGN §3.3; best-effort BONUS over the 180s loop) ---
# On an EXTERNAL SIGTERM/SIGINT (vast preemption / docker stop) the trap emits ONE
# non-terminal 'preempted' event and kicks ONE bounded final checkpoint flush. The
# body lives in the B2-staged companion preempt_trap.sh (externalized ONLY for the
# 16 KiB inline-onstart wire cap, same as resume_pull.sh): it self-installs the trap
# on source, closing over emit_event/$CKPT_DIR/$B2/$RUN_ID here. Missing/unreadable
# companion => no trap; the 180s periodic push (§2) is the PRIMARY defense regardless.
# preempt_save.py is the TRAINER-side half and must land BEFORE the trainer
# starts: train_proposer_lora.py probes /workspace for it and arms a SIGUSR1
# handler, which is the only thing the trap's `_preempt_local_save` can signal.
# Pulling the trap without the module (the state of the world until 2026-08-06)
# gets you a trap that always reports `no_piddir`.
rclone copyto "$B2/eval-env/preempt_save.py" /workspace/preempt_save.py 2>/dev/null || true
{ rclone copyto "$B2/eval-env/preempt_trap.sh" /workspace/preempt_trap.sh 2>/dev/null && . /workspace/preempt_trap.sh; } || true
# Lane-specific emitter for _preempt_save_report (preempt_trap.sh): turn every
# preempt-save outcome into a runmeta event so a skip is visible OFF-BOX, not
# just in an onstart log. Non-terminal + best-effort, like every other emit here.
_preempt_save_emit() { emit_event preempt_save "result=$1 ${2:-}" || true; }

# --- 0a. boot pusher: stream onstart.log + boot_phases.tsv to B2 DURING boot ---
# Until now the whole boot was invisible off-box (onstart.log only reached B2 at
# teardown; runmeta had zero events between 'launched' and 'running'). This
# subshell, every BOOT_PUSH_INTERVAL (default 45s): (a) pushes onstart.log,
# (b) pushes boot_phases.tsv, (c) rewrites STATUS with the latest phase appended
# as EXTRA tokens AFTER the timestamp — safe because babysit (launch_train.sh)
# and herdd only glob-match DONE*/FAILED*/STAGED* and otherwise display STATUS
# raw, and heartbeat is outside runmeta's status lattice — and (d) emits ONE
# runmeta 'heartbeat' event per NEW phase row (cursor-tracked). All best-effort;
# it exits when /workspace/.boot_done appears, then does one final push.
BOOT_PUSH_INTERVAL="${BOOT_PUSH_INTERVAL:-45}"
BOOT_PUSH=""
if [ -n "${B2_KEY_ID:-}" ]; then
  ( c=0
    # pb: push log + phases, rewrite STATUS with the latest phase appended
    pb() {
      rclone rcat "$B2/checkpoints/${RUN_ID}/onstart.log" < /workspace/onstart.log 2>/dev/null || true
      rclone rcat "$B2/checkpoints/${RUN_ID}/boot_phases.tsv" < "$BOOT_PHASES" 2>/dev/null || true
      echo "RUNNING $(date -u +%FT%TZ) phase=$(tail -1 "$BOOT_PHASES" 2>/dev/null | cut -f3)" \
        | rclone rcat "$B2/checkpoints/${RUN_ID}/STATUS" 2>/dev/null || true
    }
    while [ ! -f /workspace/.boot_done ]; do
      pb
      # one heartbeat per NEW phase row (cursor c); row = epoch\telapsed\tphase\tdetail
      while r=$(sed -n "$((c+1))p" "$BOOT_PHASES" 2>/dev/null); [ -n "$r" ]; do
        c=$((c+1))
        PHASE=$(echo "$r" | cut -f3) T_BOOT_S=$(echo "$r" | cut -f2) emit_event heartbeat || true
      done
      sleep "$BOOT_PUSH_INTERVAL"
    done
    pb ) & BOOT_PUSH=$!
  echo ">> boot pusher pid $BOOT_PUSH (${BOOT_PUSH_INTERVAL}s cadence)"
fi

# --- 0b. fast-boot: rehydrate the training env from B2 (opt-in) ---
REHYDRATE_PID=""
if [ "${FAST_BOOT:-0}" = "1" ] && rclone copyto "$B2/train-env/rehydrate.sh" /workspace/rehydrate.sh 2>/dev/null; then
  ( bash /workspace/rehydrate.sh ) & REHYDRATE_PID=$!
  echo ">> FAST_BOOT=1 — train env rehydrate started (pid $REHYDRATE_PID)"
  boot_mark rehydrate_start "pid=$REHYDRATE_PID"
elif [ "${FAST_BOOT:-0}" = "1" ]; then
  echo "!! FAST_BOOT: could not pull rehydrate.sh from B2 — proceeding on preflight deps"
fi

# --- 1. pull run bundle + resume checkpoints (parallel) ---
# rclone tuning: --transfers across files, --multi-thread-streams splits one big
# file into ranged GETs; all env-overridable.
RC_STREAMS="${RCLONE_STREAMS:-16}"; RC_TRANSFERS="${RCLONE_TRANSFERS:-16}"
RC_CUTOFF="${RCLONE_MT_CUTOFF:-64M}"
# --stats/-one-line => per-30s MB/s line in the (now-pushed) onstart.log so the
# pull phase is observable off-box live. RC_FAST is used ONLY by the three boot
# pulls below; env overrides for streams/transfers/cutoff are unchanged.
RC_FAST=(--fast-list --transfers "$RC_TRANSFERS" \
         --multi-thread-streams "$RC_STREAMS" --multi-thread-cutoff "$RC_CUTOFF" \
         --stats 30s --stats-one-line)
_pull_t0=$(date +%s)
boot_mark pull_start "streams=$RC_STREAMS transfers=$RC_TRANSFERS"
echo ">> pulling runset ${RUNSET}"
{ b2x_pull "$B2/runsets/${RUNSET}" "$RUNSET_DIR" \
  || rclone copy "${RC_FAST[@]}" "$B2/runsets/${RUNSET}" "$RUNSET_DIR"; } &
PULL_RUNSET=$!
if [ -n "${BASE_MODEL_B2:-}" ]; then
  echo ">> pulling base model ${BASE_MODEL_B2} (streams=$RC_STREAMS transfers=$RC_TRANSFERS)"
  mkdir -p "$WS/base"
  # THE big one (5-25 GB) and the one that most wants idempotence: a resumed or
  # preempted box often already holds most of these bytes on its kept disk.
  { b2x_pull "$B2/${BASE_MODEL_B2}" "$WS/base" \
    || rclone copy "${RC_FAST[@]}" "$B2/${BASE_MODEL_B2}" "$WS/base"; } &
  PULL_MODEL=$!
fi
# resume prior checkpoints; --exclude STATUS keeps the live marker out of the tree.
# The narrowed pull (newest-2 checkpoint dirs per layout root instead of the
# whole accumulated prefix — a 20-save run pulled ~20x the bytes HF resume needs
# on a billed idle box) lives in the B2-staged companion resume_pull.sh: it is
# too big for this file's 16 KiB wire budget. Companion missing/unreadable =>
# verbatim legacy whole-prefix pull. The companion never exits and does its own
# whole-prefix fallback on lsf failure — resume correctness beats efficiency.
{ rclone copyto "$B2/eval-env/resume_pull.sh" /workspace/resume_pull.sh 2>/dev/null && . /workspace/resume_pull.sh; } || b2x_pull "$B2/checkpoints/${RUN_ID}" "$CKPT_DIR" --exclude STATUS 2>/dev/null || rclone copy "${RC_FAST[@]}" --exclude STATUS "$B2/checkpoints/${RUN_ID}" "$CKPT_DIR" 2>/dev/null || true
# >>> handoff-synced-marker
# Understudy boot proof for the driver's SYNCED gate: the checkpoint resume pull
# above has completed ON THIS BOX. Only handoff understudies carry HANDOFF_EPOCH.
# Without this marker the driver stamped SYNCED off mere API liveness ('loading'
# counts) and fenced the primary against a box that had not even booted (live
# canary handoff-canary-2, 2026-07-15). A fresh run with zero prior checkpoints
# still writes it — nothing to sync IS synced; the fence loses no bytes.
if [ -n "$HANDOFF_EPOCH" ]; then
  printf '{"run_id":"%s","epoch":%s,"synced_at":"%s"}\n' \
    "$RUN_ID" "$HANDOFF_EPOCH" "$(date -u +%FT%TZ)" \
    | rclone rcat "$B2/runs/${RUN_ID}/handoff/${HANDOFF_EPOCH}.synced" 2>/dev/null || true
fi
# <<< handoff-synced-marker
wait $PULL_RUNSET; RC_RUNSET=$?
if [ -n "${PULL_MODEL:-}" ]; then
  wait $PULL_MODEL || { echo "!! base model pull failed"; RC_RUNSET=1; }
fi
if [ "$RC_RUNSET" -ne 0 ] || [ -z "$(ls -A "$RUNSET_DIR" 2>/dev/null)" ]; then
  echo "!! runset/model pull failed or runset empty — aborting (nothing to train)"
  RC=3
fi
# measured pull throughput: bytes = du of runset + base (base absent => 0).
# Integer MB/s is fine; this is the per-host record we later use to score hosts
# (geolocation/inet_down are launcher-side; the box reports MEASURED throughput).
_pull_secs=$(( $(date +%s) - _pull_t0 ))
_pull_bytes=$(du -sbc "$RUNSET_DIR" "$WS/base" 2>/dev/null | tail -1 | cut -f1); _pull_bytes=${_pull_bytes:-0}
_pull_mbps=$(( _pull_bytes / 1048576 / (_pull_secs > 0 ? _pull_secs : 1) ))
# The tally names WHICH transport moved the bytes (cdn / b2x ok / rclone
# fallback). MB/s alone cannot say, and "the CDN tier never engaged" is
# otherwise invisible off-box — the pull still succeeds, just slower.
boot_mark pull_done "bytes=${_pull_bytes} secs=${_pull_secs} MBps=${_pull_mbps} $(command -v b2x_tally_summary >/dev/null 2>&1 && b2x_tally_summary || echo transport=unknown)"
# distinct phase name so throughput consumers don't collide with the pusher's
# generic pull_done heartbeat (which carries only phase + t_boot_s)
PHASE=pull_throughput T_BOOT_S=$(( $(date +%s) - BOOT_T0 )) \
  BYTES="$_pull_bytes" SECS="$_pull_secs" MBPS="$_pull_mbps" emit_event heartbeat || true
echo ">> runset contents:"; ls -la "$RUNSET_DIR"

# --- 2. background: stream checkpoints out every CKPT_INTERVAL sec (default 3 min) ---
# --min-age 45s skips files still mid-write. CKPT_INTERVAL is the run's loss-window
# knob (SPOT_DESIGN §3.3): shorter = less lost on preemption, more B2 churn.
#
# ACCEPTED RISK (documented, not fixed): a HARD SIGKILL (no docker-stop SIGTERM, so
# neither this loop nor the preempt trap runs) loses up to one CKPT_INTERVAL of
# progress; the --min-age 45s window is a second, narrower blind spot closed only by
# the trap's no-min-age final flush.
#
# LOUD on failure (box 44566398): stderr is CAPTURED, not swallowed. A dead/rotated
# B2 key silently FROZE this sync while training ran on and the finished adapter
# stranded. On an auth-class failure we drop a persistent /workspace/.checkpoint_sync_failed
# breadcrumb + emit a DISTINCT, rate-limited `checkpoint_sync_failed` event — never
# silent, never fatal (compute keeps advancing locally; the loop keeps retrying).
( LAST_EMIT_STEP=0; LAST_HB_AT=""; HB_MIN="${METRICS_HEARTBEAT_S:-120}"
  SYNC_FAILS=0; SYNC_ERR=/workspace/.ckpt_sync.err
  while true; do
    # host-metrics heartbeat, throttled to >= HB_MIN sec apart so a short
    # CKPT_INTERVAL doesn't spam the event log (default 120s ~= 720 objs/24h).
    if [ -z "$LAST_HB_AT" ] || [ $(( SECONDS - LAST_HB_AT )) -ge "$HB_MIN" ]; then
      HOST_METRICS="$(host_metrics)" emit_event heartbeat || true
      LAST_HB_AT=$SECONDS
    fi
    # two-writer fence (HANDOFF_DESIGN §4): once a newer handoff epoch owns the
    # run's B2 state, this box must NOT overwrite the understudy's checkpoints.
    # No-op / fail-safe off the handoff path (unset epoch => never stale).
    if _handoff_epoch_stale; then
      echo "[ckpt-sync] handoff-epoch REFUSE $(date -u +%T): epoch ${HANDOFF_EPOCH} is stale — a newer epoch owns runs/${RUN_ID}; not overwriting the understudy"
      sleep "${CKPT_INTERVAL:-180}"; continue
    fi
    : > "$SYNC_ERR"
    if b2x_push "$CKPT_DIR" "$B2/checkpoints/${RUN_ID}/" --min-age 45s --exclude STATUS 2>"$SYNC_ERR" \
       || rclone copy --fast-list --min-age 45s --exclude STATUS "$CKPT_DIR" "$B2/checkpoints/${RUN_ID}/" 2>"$SYNC_ERR"; then
      echo "[ckpt-sync] pushed $(date -u +%T)"; status RUNNING
      SYNC_FAILS=0; rm -f /workspace/.checkpoint_sync_failed; _handoff_stamp_epoch
      # emit a checkpoint EVENT only when max step ADVANCES (per-save, not per
      # 180s loop — a 1:1 map would be ~480 useless objects/24h). Step parsed
      # exactly as cmd_runs does: int after 'checkpoint-'.
      # NOTE: runsets write checkpoints one level deep under arms/<name>/ (e.g.
      # modelzoo-reader: $CKPT_DIR/arms/reader/checkpoint-*), NOT at $CKPT_DIR
      # top level. Glob BOTH so latest_step is emitted for flat- and arm-layout
      # runsets alike — a top-level-only glob left latest_step=None for every
      # arm-layout run even while training advanced (checkpoints synced fine).
      newstep=$(ls -1d "$CKPT_DIR"/checkpoint-* "$CKPT_DIR"/arms/*/checkpoint-* 2>/dev/null \
                 | sed 's#.*/checkpoint-##' \
                 | grep -E '^[0-9]+$' | sort -n | tail -1)
      if [ -n "$newstep" ] && [ "$newstep" -gt "$LAST_EMIT_STEP" ] 2>/dev/null; then
        STEP="$newstep" emit_event checkpoint || true; LAST_EMIT_STEP="$newstep"
      fi
    else
      # NEVER silent, NEVER fatal: keep retrying. Classify the cause; an auth-class
      # failure (InvalidAccessKeyId / SignatureDoesNotMatch / 403) is non-transient
      # and is the box-44566398 freeze mechanism.
      SYNC_FAILS=$(( SYNC_FAILS + 1 ))
      SYNC_TAIL="$(tr '\n' ' ' < "$SYNC_ERR" 2>/dev/null | tail -c 200)"
      if grep -qiE 'InvalidAccessKeyId|SignatureDoesNotMatch|AccessDenied|Unauthorized|not valid| 403 ' "$SYNC_ERR" 2>/dev/null; then
        SYNC_REASON="B2 AUTH FAILURE (dead/rotated key): ${SYNC_TAIL}"
      else
        SYNC_REASON="rclone sync error: ${SYNC_TAIL}"
      fi
      printf '%s consecutive=%s\n%s\n' "$(date -u +%FT%TZ)" "$SYNC_FAILS" "$SYNC_REASON" \
        > /workspace/.checkpoint_sync_failed 2>/dev/null || true
      echo "[ckpt-sync] push FAILED $(date -u +%T) (#$SYNC_FAILS; will retry) — $SYNC_REASON"
      # rate-limit: 1st failure, then every SYNC_FAIL_EVERY-th consecutive one.
      if [ $(( (SYNC_FAILS - 1) % ${SYNC_FAIL_EVERY:-5} )) -eq 0 ]; then
        emit_event checkpoint_sync_failed "consecutive=${SYNC_FAILS}; ${SYNC_REASON}" || true
      fi
    fi
    # keep onstart.log fresh on B2 during training too (it grows slowly); runs
    # each cycle regardless of the checkpoint push result.
    rclone rcat "$B2/checkpoints/${RUN_ID}/onstart.log" < /workspace/onstart.log 2>/dev/null || true
    sleep "${CKPT_INTERVAL:-180}"
  done ) & WATCH=$!
echo ">> checkpoint watcher pid $WATCH -> $B2/checkpoints/${RUN_ID}"

# --- 2b. co-tenant eval sidecar (optional) ---
# EVAL_TARGETS => compile+score evals on idle CPU (setsid group, evals/<RUN_ID>/,
# never touches STATUS). See onstart/eval_sidecar.sh.
SIDE_PID=""; SIDE_PGID=""
# companions (best-effort): yield_fence.sh + farm_worker.sh (CPU_FARM/farm mode)
# + saturator.py (CPU_FARM/saturate mode).
side_companions() {
  rclone copyto "$B2/eval-env/yield_fence.sh"  /workspace/yield_fence.sh  2>/dev/null || true
  rclone copyto "$B2/eval-env/farm_worker.sh"  /workspace/farm_worker.sh  2>/dev/null || true
  rclone copyto "$B2/eval-env/saturator.py"    /workspace/saturator.py    2>/dev/null || true
}
if [ -n "${EVAL_TARGETS:-}" ]; then
  echo ">> eval sidecar requested (EVAL_TARGETS=${EVAL_TARGETS})"
  if rclone copyto "$B2/eval-env/eval_sidecar.sh" /workspace/eval_sidecar.sh 2>/dev/null; then
    side_companions
    # boot log only catches output before the sidecar's own tee takes over
    setsid bash /workspace/eval_sidecar.sh >>/workspace/eval_sidecar.boot.log 2>&1 &
    SIDE_PID=$!
    SIDE_PGID=$(ps -o pgid= -p "$SIDE_PID" 2>/dev/null | tr -d ' '); SIDE_PGID="${SIDE_PGID:-$SIDE_PID}"
    echo ">> eval sidecar pid $SIDE_PID pgid $SIDE_PGID (log: eval_sidecar.log)"
  else
    echo "!! could not pull eval_sidecar.sh from B2 — skipping evals (training unaffected)"
  fi
fi

# --- 2c. CPU compile-farm co-tenant (DEAD FEATURE, opt-IN) ---
# Owner ruling 2026-08-21: dead, disabled by default everywhere, opt-in only —
# the sidecar's rb3-objcache grew to 69 GB and took a live serving box to
# 110/110 GB, one write from killing a serve mid-eval. The starvation record
# below is why it was already default-OFF on this lane.
# Default-OFF since 2026-07-10: a 134-worker saturator starved a CPU-sensitive
# LoRA train 16x (106 s/it @ 4% GPU util -> 6.66 s/it @ 62% after kill;
# base-reader-nanbeige-01). The yield fence's cgroup cpu.weight layer never
# engages inside vast containers (no cgroup delegation — subtree_control is
# empty), leaving nice 19 as the only CPU layer; ~0.7*nproc compile procs at
# nice 19 still rival the trainer's few host threads in aggregate CFS weight,
# and nothing schedules memory bandwidth, which wibo/cl.exe saturates. The
# earlier "no degradation" verdict (corpus-v4-01, 90 workers) held only because
# that train was GPU-bound. Opt in with --cpu-farm (CPU_FARM=1) ONLY for trains
# verified insensitive to host CPU/membw load.
# Two sub-modes, chosen by whether farm work is queued:
#   * a B2 manifest at farm/<RUN_ID>/inbox => EVAL_MODE=farm (finite manifest units)
#   * NO manifest                          => EVAL_MODE=saturate (manifest-free
#     continuous crack-farm; self-selects the near-miss frontier from artifacts.db).
CPU_FARM="${CPU_FARM:-0}"
FARM_RUN_ID="${FARM_RUN_ID:-$RUN_ID}"
if [ "$CPU_FARM" != "0" ] && [ -z "${EVAL_TARGETS:-}" ] && [ -z "$SIDE_PID" ]; then
  echo "!! CPU_FARM=$CPU_FARM — opting IN to the DEAD co-tenant compile farm."
  echo "!! It starves CPU-sensitive trains 16x and filled a serve box's disk"
  echo "!! (69 GB objcache -> 110/110 GB, 2026-08-20). Unset CPU_FARM for the default (off)."
  env_staged=0; farm_queued=0
  rclone lsf "$B2/eval-env/LATEST" 2>/dev/null | grep -q . && env_staged=1
  rclone lsf "$B2/farm/${FARM_RUN_ID}/inbox/manifest.json" 2>/dev/null | grep -q . && farm_queued=1
  if [ "$env_staged" -eq 1 ]; then
    if [ "$farm_queued" -eq 1 ]; then
      SIDE_MODE=farm
      echo ">> CPU_FARM: staged+queued (farm/${FARM_RUN_ID}) — launching farm sidecar"
    else
      SIDE_MODE=saturate
      echo ">> CPU_FARM: no manifest — saturate sidecar"
    fi
    if rclone copyto "$B2/eval-env/eval_sidecar.sh" /workspace/eval_sidecar.sh 2>/dev/null; then
      side_companions
      EVAL_MODE="$SIDE_MODE" FARM_RUN_ID="$FARM_RUN_ID" \
        setsid bash /workspace/eval_sidecar.sh >>/workspace/eval_sidecar.boot.log 2>&1 &
      SIDE_PID=$!
      SIDE_PGID=$(ps -o pgid= -p "$SIDE_PID" 2>/dev/null | tr -d ' '); SIDE_PGID="${SIDE_PGID:-$SIDE_PID}"
      echo ">> $SIDE_MODE sidecar pid $SIDE_PID pgid $SIDE_PGID (log: eval_sidecar.log)"
    else
      echo "!! could not pull eval_sidecar.sh — skipping CPU farm (training unaffected)"
    fi
  else
    echo ">> CPU_FARM no-op: env not staged (env_staged=0)"
  fi
fi   # CPU_FARM=0 is the default and says nothing

# --- 3. train ---
export OUTPUT_DIR="$CKPT_DIR" BASE_DIR="$WS/base" RUNSET_DIR="$RUNSET_DIR"
# fast-boot / baked-image: source the venv before the trainer (best-effort).
# The activate marker is written either by the B2 rehydrate (FAST_BOOT) OR baked
# into the image by train-env/Dockerfile — so source it whenever present, and
# only warn about a MISS when FAST_BOOT actually tried to produce it.
if [ -n "$REHYDRATE_PID" ]; then
  wait "$REHYDRATE_PID" 2>/dev/null || true
fi
boot_mark rehydrate_done   # tier-agnostic: env is ready (rehydrated or baked-image)
_act="$(cat /workspace/.train_env_activate 2>/dev/null || true)"
if [ -n "$_act" ] && [ -f "$_act" ]; then echo ">> sourcing train env: $_act"; . "$_act"
elif [ -n "$REHYDRATE_PID" ]; then echo "!! FAST_BOOT: no train env — using preflight deps"; fi
if [ -z "${RC:-}" ]; then
  CMD="${TRAIN_CMD:-}"
  [ -z "$CMD" ] && [ -f "$RUNSET_DIR/train.sh" ] && CMD="bash $RUNSET_DIR/train.sh"
  if [ -z "$CMD" ]; then
    echo "!! no TRAIN_CMD and no runset/train.sh — nothing to run"; RC=2
  else
    echo ">> TRAIN: $CMD"; emit_event running || true    # once, at train start (not per sync loop)
    # boot is over: signal the boot pusher to stop (final push happens inside it).
    # The runset's own train.sh echoes its env-tier lines from here on, now visible
    # via the checkpoint watcher's per-cycle onstart.log push.
    touch /workspace/.boot_done 2>/dev/null || true
    boot_mark runset_cmd_start
    eval "$CMD"; RC=$?
    echo ">> training exited rc=$RC"
  fi
fi

# --- 4. final flush + artifact + self-destruct ---
# errexit-OFF here: teardown must run even if a flush attempt fails
# defensive: if the train never started (RC=2/3 aborts before .boot_done), the
# boot pusher would still be looping — stop it here too.
touch /workspace/.boot_done 2>/dev/null || true
[ -n "${BOOT_PUSH:-}" ] && kill "$BOOT_PUSH" 2>/dev/null || true
boot_mark teardown "rc=${RC:-na}"
kill $WATCH 2>/dev/null || true
# trailing slash on destinations: B2's S3 HEAD can transiently report a
# nonexistent key as an object, making rclone's dest file-check die CRITICAL
# "is a file not a directory" (ate base-reader-nanbeige-01's artifacts push,
# 2026-07-10); a dir-slash dest skips the check, retries ride out the flake.
# two-writer fence (HANDOFF_DESIGN §4): a stale husk (a superseded primary that
# resumed) must not overwrite the understudy's B2 state on its way down. Fail-safe
# off the handoff path (unset epoch => not stale => flush exactly as before).
if _handoff_epoch_stale; then
  echo "!! final flush SKIPPED — handoff epoch ${HANDOFF_EPOCH} stale (a newer epoch owns runs/${RUN_ID})"
else
  for i in 1 2 3; do
    { b2x_push "$CKPT_DIR" "$B2/checkpoints/${RUN_ID}/" --exclude STATUS \
      || rclone copy --fast-list --exclude STATUS "$CKPT_DIR" "$B2/checkpoints/${RUN_ID}/"; } && { _handoff_stamp_epoch; break; }
    echo "!! final checkpoint flush failed (attempt $i/3)"; sleep 20
  done
fi
if [ "${RC:-1}" -eq 0 ]; then
  if _handoff_epoch_stale; then
    echo "!! artifacts push SKIPPED — handoff epoch ${HANDOFF_EPOCH} stale (a newer epoch owns ${RUN_ID})"
  else
    for i in 1 2 3; do
      { b2x_push "$CKPT_DIR" "$B2/artifacts/${RUN_ID}/" \
        || rclone copy --fast-list "$CKPT_DIR" "$B2/artifacts/${RUN_ID}/"; } && break
      echo "!! artifacts push failed (attempt $i/3)"; sleep 20
    done
    echo ">> artifacts -> $B2/artifacts/${RUN_ID}"
  fi
fi
# --- 4b. eval sidecar teardown (SPEC MUST 17) ---
# Before STATUS->DONE. Grace-wait only if RC==0; else kill now. Group-kill so
# ninja/cargo children die too.
if [ -n "${SIDE_PID:-}" ]; then
  if kill -0 "$SIDE_PID" 2>/dev/null; then
    if [ "${RC:-1}" -eq 0 ]; then
      GRACE_MIN="${EVAL_GRACE_MINUTES:-30}"
      echo ">> training OK — grace-wait ${GRACE_MIN}m for eval sidecar (pgid ${SIDE_PGID})"
      for _ in $(seq 1 $(( GRACE_MIN * 2 )) ); do
        kill -0 "$SIDE_PID" 2>/dev/null || { echo ">> eval sidecar finished on its own"; break; }
        sleep 30
      done
    else
      echo ">> training rc=${RC:-na} — not waiting for eval sidecar"
    fi
    if kill -0 "$SIDE_PID" 2>/dev/null; then
      echo ">> stopping eval sidecar group (kill -- -${SIDE_PGID})"
      kill -- "-${SIDE_PGID}" 2>/dev/null || kill "$SIDE_PID" 2>/dev/null || true
      # let the sidecar's TERM trap flush results + EVAL_STATUS
      for _ in $(seq 1 12); do
        kill -0 "$SIDE_PID" 2>/dev/null || break
        sleep 5
      done
    fi
  fi
  rclone rcat "$B2/evals/${RUN_ID}/eval_sidecar.log" < /workspace/eval_sidecar.log 2>/dev/null || true
fi

# push onstart.log + boot_phases.tsv + write terminal EVENT (before STATUS, before self-destruct, I4)
rclone rcat "$B2/checkpoints/${RUN_ID}/onstart.log" < /workspace/onstart.log 2>/dev/null || true
rclone rcat "$B2/checkpoints/${RUN_ID}/boot_phases.tsv" < /workspace/boot_phases.tsv 2>/dev/null || true
case "${RC:-1}" in
  0) _emit_terminal done ;;
  2) _emit_terminal failed "STAGED (no train.sh/TRAIN_CMD)" ;;
  3) _emit_terminal failed "FAILED pull" ;;
  *) _emit_terminal failed "rc=${RC:-na}" ;;
esac
case "${RC:-1}" in
  0) status DONE ;;
  2) status "STAGED (no train.sh/TRAIN_CMD)" ;;
  3) status "FAILED pull" ;;
  *) status "FAILED rc=${RC:-na}" ;;
esac

# terminal sentinel FIRST (before any teardown action): a parked box re-runs
# onstart on resume — this is what the resume guard keys on.
echo "rc=${RC:-na} $(date -u +%FT%TZ)" > /workspace/.run_terminal
teardown() { [ -n "${WATCHDOG:-}" ] && kill "$WATCHDOG" 2>/dev/null || true; self_teardown; }

if [ "${RC:-1}" -eq 0 ]; then
  if [ "${KEEP_ON_DONE:-0}" = "1" ] || [ "${TEARDOWN:-park}" = "keep" ]; then
    # explicit keep: box stays RUNNING (GPU billing!) for immediate reuse.
    # Watchdog stays armed so a forgotten box still parks at MAX_HOURS.
    echo ">> TEARDOWN=keep — RUNNING (billing!); parks @ ${MAX_HOURS}h"
  else
    # default: self-park — resume with `herdd start <id>` (warm disk/caches)
    teardown
  fi
elif [ "${KEEP_ON_FAIL:-0}" = "1" ]; then
  # watchdog stays armed: forgotten debug box still parks at MAX_HOURS
  echo ">> KEEP_ON_FAIL=1 — RUNNING (billing!); parks @ ${MAX_HOURS}h"
else
  # Bounded debug-hold: SSH-able for FAIL_HOLD_MINUTES; poll B2 stop/extend
  # markers (tools/vast/debug_box.sh, keyed on RUN_ID). MAX_HOURS backstop stays armed.
  HOLD="${FAIL_HOLD_MINUTES:-15}"
  if [ "$HOLD" -gt 0 ] 2>/dev/null; then
    status "FAILED rc=${RC:-na} — debug-hold ${HOLD}m"
    echo ">> FAIL debug-hold ${HOLD}m rc=${RC:-na}; debug_box.sh stop|extend ${RUN_ID}"
    end=$(( $(date +%s) + HOLD * 60 ))
    while [ "$(date +%s)" -lt "$end" ]; do
      if rclone lsf "$B2/checkpoints/${RUN_ID}/STOP" >/dev/null 2>&1; then
        echo ">> STOP marker seen — tearing down early"
        rclone deletefile "$B2/checkpoints/${RUN_ID}/STOP" 2>/dev/null || true
        break
      fi
      if rclone lsf "$B2/checkpoints/${RUN_ID}/EXTEND" >/dev/null 2>&1; then
        echo ">> EXTEND marker seen — +${HOLD}m"
        rclone deletefile "$B2/checkpoints/${RUN_ID}/EXTEND" 2>/dev/null || true
        end=$(( $(date +%s) + HOLD * 60 ))
      fi
      sleep 20
    done
    echo ">> debug-hold ended — tearing down (${TEARDOWN:-park})"
  fi
  teardown
fi
echo "=== onstart done $(date -u) ==="
