#!/usr/bin/env bash
# onstart/serve_vllm.sh — vLLM OpenAI-compatible server for evals; mirrors local qwen path (LLM_BASE_URL=host:port/v1).
#
# Env knobs:
#   MODEL_ID           HF id or local path of the base model (required unless MODEL_B2)
#   MODEL_B2           b2 subpath of a base-model dir; the box pulls the base from B2
#                      instead of HF (HF-Xet-deadlock-safe path) -> MODEL_ID=/workspace/base-model.
#                      OVERRIDES MODEL_ID when both are set (loudly, since 2026-08-21);
#                      a caller that means MODEL_ID must ship MODEL_B2='' alongside it.
#   SERVE_MODEL_OVERRIDE  path to the DURABLE flip file (default
#                      /workspace/serve_model_override.json). When it exists it
#                      REPLACES MODEL_ID/MODEL_B2 (and optionally SERVED_NAME /
#                      MAX_LEN / the identity expectation) on EVERY start — the
#                      launch onstart, an onstart re-run after a vast resume, a
#                      jobs-lane attempt re-run, an --on-box attach. Write it
#                      with tools/vast/serve_flip.sh. A present-but-unusable
#                      override is a REFUSAL, never a fallback to the launch
#                      model: the endpoint label belongs to the override.
#   SERVED_NAME        --served-model-name (default: basename of MODEL_ID)
#   MAX_LEN            --max-model-len (default 16384)
#   GPU_UTIL           vLLM VRAM fraction (default 0.90; 0.95 + a 2nd job OOMs)
#   VLLM_API_KEY       bearer token; the mapped port is PUBLIC — always set it
#   QUANTIZATION       --quantization (e.g. bitsandbytes) — match a local serve
#                      shape when an eval arm must stay quantization-consistent
#   LORA_SPECS         comma list  name=b2subpath[,name2=b2subpath2,...]
#   MAX_LORA_RANK      --max-lora-rank when LORA_SPECS set (default 32)
#   CHAT_TEMPLATE_B2   b2 subpath of a .jinja chat template to attach
#   MAX_HOURS          watchdog teardown after N hours (default 12; 0 off)
#   TEARDOWN           park (default; resume with herdd start) | destroy
#   CPU_FARM           =1 opts IN to the co-tenant CPU saturate-farm — DEAD
#                      feature, default OFF (owner ruling 2026-08-21); its
#                      objcache once filled a serve box's disk to the brim
#   FARM_RUN_ID        farm shards land at evals/<FARM_RUN_ID>/corpus/ (default SERVE_ID)
#   SERVE_DP           vLLM-native data parallelism: N engine replicas inside ONE
#                      `vllm serve` on :8000 ('auto' = GPU count / SERVE_TP). The
#                      PREFERRED multi-GPU saturation path: single endpoint,
#                      queue-aware internal LB (beats HAProxy leastconn, which only
#                      sees connection counts), no haproxy install. Mutually
#                      exclusive with SERVE_REPLICAS>1.
#   SERVE_TP           --tensor-parallel-size per engine (default 1). Composes with
#                      SERVE_DP: DP=2 x TP=2 on 4 cards is the shape for a model
#                      whose bf16 weights don't fit one card's VRAM with KV.
#   SERVE_REPLICAS     LEGACY fallback: N separate vLLM processes behind HAProxy
#                      (default 1; 'auto'=GPU count). Keeps per-replica logs +
#                      partial-degradation (HAProxy ejects a dead replica); use only
#                      when those properties are needed.
#   REPLICA_BASE_PORT  replica i binds 127.0.0.1:(BASE+i) loopback-only (default 8001)
#   HAPROXY_BALANCE    backend balance algo (default leastconn)
#   HAPROXY_TIMEOUT_S  HAProxy client/server/tunnel timeout secs (default 1800)
#   KV_CACHE_DTYPE     --kv-cache-dtype (e.g. fp8) — the sanctioned VRAM fallback
#   MAX_NUM_SEQS       --max-num-seqs (decode batch width). UNSET = flag NOT emitted
#                      and vLLM's own card-dependent default stands (256 <70 GiB /
#                      1024 >=70 GiB at 0.26.0) — this script emitted nothing at all
#                      before 2026-08-09, so leaving it unset is byte-identical to
#                      every serve banked to date. It is NOT a free performance
#                      knob: the r3 drift-roster arms pin 16 and the pin is a
#                      MEASURED term worth -3 solves (V8_DD_EVAL_RESULT_2026-08-05
#                      :135), so changing it on a comparand needs a bridge cell.
#                      Some hybrid archs also FAIL engine init at the 1024 default
#                      ("max_num_seqs (1024) exceeds available Mamba cache blocks"),
#                      which is why the bakeoff roster pins 64.
#   MAX_NUM_BATCHED_TOKENS  --max-num-batched-tokens (prefill token budget per
#                      step). UNSET = vLLM's OWN vllm-serve resolution is computed
#                      here and emitted EXPLICITLY -- 2048 (<70 GiB card) / 8192
#                      (>=70 GiB) -- so the value lands in argv and serve_summary
#                      instead of being inferable-only. Behaviour is unchanged by
#                      construction; `none` suppresses the flag and restores the
#                      pre-2026-08-24 argv exactly. An integer pins it (a comparand
#                      move: the term is ~1%, so pinning is for comparability, not
#                      speed). Resolution is a 2x2 on (>=70 GiB, UsageContext) and
#                      the in-process row is 8192/16384 -- do not splice the rows.
#   SERVE_DTYPE        --dtype (e.g. bfloat16). UNSET = flag NOT emitted and
#                      vLLM's own auto default stands, which is what every serve
#                      banked before this knob existed ran at.
#   SERVE_IDENT_REQUIRED  =1 ARMS the on-box identity gate: after the pull and
#                      before any vllm argv exists, the model dir is fingerprinted
#                      and compared to the expectation the launcher froze from the
#                      COMMITTED registry. Set by launch_serve.sh --model-artifact.
#                      UNSET = skip with one loud line (every pre-artifact caller
#                      is byte-identical). SET but unresolvable = FAILED, never a
#                      skip: a gate that disappears on a transient read is not one.
#   SERVE_IDENT_ARTIFACT  the registry slug, for the log and serve_summary.json
#   SERVE_IDENT_EXPECT / SERVE_IDENT_GATE / SERVE_IDENT_FINGERPRINT / SERVE_IDENT_DIRHASH
#                      explicit paths to the four staged identity assets; each
#                      falls back to /workspace/<name> and then to a pull from
#                      b2:$B2_BUCKET/serve/$SERVE_ID/<name> (same per-serve prefix
#                      as parse_vllm_mem.py — none of them fits the onstart wire)
#   TRUST_REMOTE_CODE  =1 appends --trust-remote-code
#   SERVE_PREFIX_CACHING  --enable-prefix-caching. DEFAULTS ON (=1) -- opt-out,
#                      not opt-in, for THROUGHPUT (2.02x). It is a COMPARAND
#                      TERM: the "output-identical" claim this default was
#                      argued from was MEASURED AND REFUTED 2026-08-24 (cache
#                      OFF 6/6 reproducible, ON 2/6; setting the flag moves
#                      mamba_cache_mode none->align and changes even a COLD
#                      prefill). Record the resolved value per arm and never
#                      flip it inside a frozen comparison.
#                      NOT redundant with vLLM's own default:
#                      0.27 computes `is_prefix_caching_supported and not
#                      is_hybrid`, so every hybrid (Qwen3.5/3.6/3.8) served with
#                      it OFF. =0 emits nothing and restores the per-model default.
#   SERVE_MTP          multi-token-prediction spec decoding. DEFAULT `auto`
#                      since 2026-08-27 (owner directive): ON iff the checkpoint
#                      ships an MTP head, INCLUDING when LoRA is attached. 1 =
#                      force, 0 = the opt-out. Measured +205% output tok/s at
#                      k=1/9/20 on eval-format prompts with the v14 LoRA
#                      attached unmerged (run of record
#                      <upstream-bench>/archive/runs/2026-08-27-v14-lora-mtp/,
#                      RTX PRO 6000 / vLLM 0.27.1.post1+fork.gfb8e9ed57). The
#                      2026-08-22 anchor's -2.3% at k=20 still reproduces on ITS
#                      workload; the discriminator is ACCEPTANCE (0.34-0.75 vs
#                      0.93+), not concurrency. Full argument at the resolution
#                      block below. Two costs: min_p/logit_bias requests are
#                      REFUSED under spec decode (pass 0 for a sampling lane),
#                      and output is not bitwise stable, so this is a COHORT
#                      term -- do not let a frozen comparison gain it
#                      (EVAL_THROUGHPUT_AUDIT_2026-08-09 §447).
#   SERVE_MTP_NUM_SPEC / SERVE_MTP_METHOD  draft depth (DEFAULT 5 since
#                      2026-08-27; n=1 is only ~+45%) and method (default mtp).
#                      The 1-layer head is reused autoregressively, not clamped.
#                      Depth is workload-shaped: sweep it at the REAL
#                      concurrency before pinning something else.
#   VLLM_EXTRA_ARGS    extra `vllm serve` args, whitespace-split into argv (appended last)
#   SERVE_ID           when set WITH B2_*: writes a SERVE_STATUS marker to
#                      b2:$B2_BUCKET/serve/$SERVE_ID/SERVE_STATUS (log-only serve if unset)
#   B2_KEY_ID B2_APPLICATION_KEY B2_BUCKET B2_S3_ENDPOINT [B2_REGION]  required IFF LORA_SPECS/CHAT_TEMPLATE_B2/MODEL_B2 set (or SERVE_ID marker)
#   B2_WRITE_KEY_ID B2_WRITE_APPLICATION_KEY  optional Option-1b serve/-scoped
#                      write pair: serve/ writes (SERVE_STATUS/METRICS) route via
#                      [b2w]; unset => single-key box, writes stay on [b2].
#                      [b2]/[b2w] are REWRITTEN from env on every run (rotation-
#                      safe: a --on-box re-run revokes+re-mints the serve keys)
#   DRY_RUN=1          print resolved vllm serve argv (+ haproxy.cfg if replicas>1) and exit; no pull/serve
#
# Launch (base): herdd.py launch --gpu a100 --gpu-ram 40 --disk 60 --image vllm/vllm-openai:latest --port 8000 --ssh --wait 900 --onstart tools/vast/onstart/serve_vllm.sh --env MODEL_ID=Qwen/Qwen2.5-Coder-32B-Instruct --env VLLM_API_KEY=<secret>
# Launch (4-GPU saturation, native DP): herdd.py launch --gpu 5090 --num-gpus 4 --disk 100 --port 8000 --ssh --wait 900 --image vllm/vllm-openai:latest --onstart tools/vast/onstart/serve_vllm.sh --env MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct --env VLLM_API_KEY=<secret> --env SERVE_DP=auto
# Launch (big model, 4 GPUs): same but --env SERVE_DP=2 --env SERVE_TP=2 (2 engines x 2-way sharding)
set -euo pipefail
# This file OUTLIVES the run that wrote it (an ssh attach inherits it via pam_env),
# so REPLACE each key instead of appending: a stale MODEL_B2 stacked here by an
# earlier launch silently overrode a later attach's MODEL_ID and served BASE
# weights under the new name (2026-08-21). Snapshot first, then rewrite.
_SERVE_ENV_FILE="${SERVE_ENV_FILE:-/etc/environment}"   # override is test-only
_SERVE_ENV_SNAP="$(mktemp)"
# SERVE_* already covers SERVE_DTYPE and every SERVE_IDENT_* below — which is
# load-bearing for the SAME reason MODEL_B2 was: an identity expectation that
# outlives its run and is inherited by a later attach would gate the NEXT model
# against the PREVIOUS one's fingerprint. Replace-not-append is what stops that.
env | grep -E '^(MODEL_ID|MODEL_B2|SERVED_NAME|MAX_LEN|MAX_NUM_|HF_|GPU_UTIL|VLLM_|QUANTIZATION|KV_CACHE_DTYPE|TRUST_REMOTE_CODE|LORA_SPECS|MAX_LORA_RANK|CHAT_TEMPLATE_B2|MAX_HOURS|TEARDOWN|SERVE_|REPLICA_BASE_PORT|HAPROXY_|B2_|CPU_FARM|FARM_)' > "$_SERVE_ENV_SNAP" || true
_persist_serve_env() {
  local keys tmp
  [ -s "$_SERVE_ENV_SNAP" ] || return 0
  # identifier-shaped keys only: a junk token would make the grep below invalid
  # and (behind `|| true`) truncate everything else out of the file.
  keys="$(cut -d= -f1 "$_SERVE_ENV_SNAP" | grep -E '^[A-Za-z_][A-Za-z0-9_]*$' \
          | sort -u | paste -sd'|' -)"
  tmp="$(mktemp)" || return 0
  if [ -f "$_SERVE_ENV_FILE" ]; then
    if [ -n "$keys" ]; then
      grep -vE "^(${keys})=" "$_SERVE_ENV_FILE" > "$tmp" 2>/dev/null || true
    else
      cat "$_SERVE_ENV_FILE" > "$tmp" 2>/dev/null || true
    fi
  fi
  cat "$_SERVE_ENV_SNAP" >> "$tmp"
  cat "$tmp" > "$_SERVE_ENV_FILE" 2>/dev/null || true
  rm -f "$tmp"
}
_persist_serve_env || true
rm -f "$_SERVE_ENV_SNAP"

# MODEL_ID or MODEL_B2 — the B2 path resolves MODEL_ID=/workspace/base-model later.
MODEL_B2="${MODEL_B2:-}"
MODEL_ID="${MODEL_ID:-}"
if [ -z "$MODEL_ID" ] && [ -z "$MODEL_B2" ]; then
  echo "!! set MODEL_ID (HF id / local path) or MODEL_B2 (b2 subpath to a base-model dir)" >&2; exit 1
fi
# Both set is legitimate (a job that stages the base as an asset AND names its B2
# subpath) but it is also the shape a stale inherited MODEL_B2 takes — say which
# one wins, so the argv never disagrees with the log again.
if [ -n "$MODEL_ID" ] && [ -n "$MODEL_B2" ]; then
  echo "!! MODEL_B2='$MODEL_B2' AND MODEL_ID='$MODEL_ID' both set — MODEL_B2 WINS (serving /workspace/base-model)." >&2
  echo "!!   If MODEL_ID is the model you meant, ship MODEL_B2='' with it (launch_serve.sh --on-box does)." >&2
fi
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"   # keep headroom; 0.95 + a second job OOMs (see qwen notes)
DRY_RUN="${DRY_RUN:-0}"
# SERVE_DP defaults to 'auto' (saturate every card): an UNSET SERVE_DP on a
# multi-GPU box was the #1 silent footgun — one card serving, N-1 idle, eval
# walltime Nx (owner 2026-07-11). Explicit SERVE_DP=1 opts back into one engine.
if [ -n "${SERVE_DP:-}" ]; then _DP_DEFAULTED=0; else SERVE_DP=auto; _DP_DEFAULTED=1; fi
SERVE_TP="${SERVE_TP:-1}"                       # tensor-parallel size per engine (composes with SERVE_DP)
SERVE_REPLICAS="${SERVE_REPLICAS:-1}"           # LEGACY: N vLLM replicas behind HAProxy; 'auto' = GPU count; <=1 = single instance
REPLICA_BASE_PORT="${REPLICA_BASE_PORT:-8001}"  # replica i binds 127.0.0.1:(BASE+i), loopback-only (HAProxy fronts :8000)
HAPROXY_BALANCE="${HAPROXY_BALANCE:-leastconn}" # leastconn: route to least-busy replica (variable-length completions)
HAPROXY_TIMEOUT_S="${HAPROXY_TIMEOUT_S:-1800}"  # client/server/tunnel timeout for long/streaming generations

# --- rclone download tuning (shared convention w/ onstart/train.sh) -----------
# --transfers across files, --multi-thread-streams splits one big file into
# ranged GETs; all env-overridable. A single-file adapter pull leans entirely on
# --multi-thread-streams (--transfers is a no-op for one object); the 128M cutoff
# still splits everything relevant.
RC_STREAMS="${RCLONE_STREAMS:-16}"; RC_TRANSFERS="${RCLONE_TRANSFERS:-16}"
RC_CUTOFF="${RCLONE_MT_CUTOFF:-64M}"
RC_FAST=(--fast-list --transfers "$RC_TRANSFERS" \
         --multi-thread-streams "$RC_STREAMS" --multi-thread-cutoff "$RC_CUTOFF")

# b2x transport. Sourced as a sibling; a no-op stub when absent, so every `||`
# fallback below is the pre-existing rclone line unchanged. These are the serve
# lane's BULK pulls (a base model is 5-25 GB) and they were rclone-only, where
# effective concurrency is min(streams, ceil(size/64Mi)) -- the 64M cutoff alone
# clamps a 150 MB adapter to 3 flows.
_B2XD="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
for _c in "$_B2XD/b2x_boot.sh" "$_B2XD/onstart/b2x_boot.sh" /workspace/jobd/b2x_boot.sh; do
  [ -f "$_c" ] && { . "$_c"; break; }
done
command -v b2x_pull >/dev/null 2>&1 || { b2x_pull() { return 1; }; b2x_push() { return 1; }; }


# --- sampler backend: PyTorch-native default, flashinfer opt-in (now viable) ----
# The flashinfer sampler (VLLM_USE_FLASHINFER_SAMPLER=1) needs nvcc: in vLLM
# 0.19.1 it JIT-COMPILES its sampling ops at ENGINE STARTUP and the engine DIES
# if that fails (on the old torch-runtime images: "Could not find nvcc" — which
# killed modelzoo-reader-06's eval; attention/torch.compile/warmup all passed,
# ONLY the sampler needs a compiler). UNSET resolves to forward_native, =0 forces
# native. --enforce-eager does NOT help (sampler, not model graph).
#
# Default stays 0 ON PURPOSE: a paired A/B on serve210v (sm_120, gemma4-12b-text
# 262k vocab, readout 119) measured NO throughput difference — seq 32.8 vs 32.5,
# 8-way 205.4 vs 204.9 tok/s (an initial 2.16x "win" was a confounded
# first-bench-after-resume artifact; the fresh-restart re-run refuted it). So =1
# buys nothing measured at <=8-way while adding a startup-JIT failure surface.
# Opt in explicitly (caller env is always honored) for high-batch workloads worth
# re-measuring; the CPATH shim below makes =1 actually WORK on the baked lanes.
#
# CPATH (opt-in support): the flashinfer JIT #includes curand.h, which the baked
# apt toolchain does NOT ship (nvcc/cudart-dev/cccl only) — measured on
# serve210v/sm_120 the =1 startup died "curand.h: No such file or directory".
# The pip nvidia-*-cu12 packages torch depends on DO ship the headers under
# site-packages/nvidia/*/include, and nvcc reads CPATH — extend it with every
# nvidia include dir the serving python can see. The compiled cache persists per
# box (~/.cache/flashinfer), so the ~30-60s JIT is once per box.
_nvidia_cpath() {
  python3 - <<'PY' 2>/dev/null
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(":".join(sorted(d for d in glob.glob(os.path.join(sp, "nvidia", "*", "include"))
                      if os.path.isdir(d))))
PY
}
if [ -n "${VLLM_USE_FLASHINFER_SAMPLER:-}" ]; then
  echo ">> sampler: VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} (caller-pinned)"
else
  export VLLM_USE_FLASHINFER_SAMPLER=0
  echo ">> sampler: PyTorch-native (default — flashinfer measured no win at <=8-way, readout 119; opt in via VLLM_USE_FLASHINFER_SAMPLER=1)"
fi
if [ "${VLLM_USE_FLASHINFER_SAMPLER}" = "1" ]; then
  _ncp="$(_nvidia_cpath)"
  if [ -n "$_ncp" ]; then
    export CPATH="${_ncp}${CPATH:+:$CPATH}"
    echo ">> sampler: CPATH += pip nvidia include dirs (flashinfer startup JIT needs curand.h)"
  fi
fi

# --- teardown watchdog (mirrors onstart/train.sh) -----------------------------
# Vast injects a per-instance CONTAINER_API_KEY that can manage only this box.
# Default watchdog action is PARK (2026-07-10 suspend-by-default): GPU billing
# ends, warm disk/weights kept for `herdd start` (a resumed serve box comes
# back serving by itself — onstart re-runs). TEARDOWN=destroy restores the old
# self-destruct; a park that doesn't take within 180s falls back to destroy.
_iid_key() {
  IID="${INSTANCE_ID:-${CONTAINER_ID:-}}"; KEY="${VASTAI_API_KEY:-${CONTAINER_API_KEY:-}}"
  [ -z "$KEY" ] && [ -f ~/.vast_api_key ] && KEY="$(cat ~/.vast_api_key)"
  [ -n "$IID" ] && [ -n "$KEY" ]
}
self_destruct() {
  _iid_key || { echo "!! no iid/key — destroy manually: herdd destroy <id> -y" >&2; return 1; }
  echo ">> self-destruct instance ${IID}" >&2
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X DELETE -H "Authorization: Bearer ${KEY}" \
    "https://console.vast.ai/api/v0/instances/${IID}/" || true
}
self_park() {
  _iid_key || { echo "!! no iid/key — park manually: herdd stop <id>" >&2; return 1; }
  echo ">> self-park instance ${IID} (resume: herdd start ${IID}; disk bills until destroyed)" >&2
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X PUT -H "Authorization: Bearer ${KEY}" -H 'Content-Type: application/json' \
    -d '{"state":"stopped"}' "https://console.vast.ai/api/v0/instances/${IID}/" || true
  sleep 180
  echo "!! park did not take within 180s — self-destructing" >&2
  self_destruct
}

# --- Option-1b scoped WRITE remote (same pattern as onstart/jobd.sh) -----------
# When the launcher shipped a prefix-restricted write key (B2_WRITE_KEY_ID), the
# bucket-wide [b2] remote is read-only — every serve/ write (SERVE_STATUS +
# METRICS) goes through [b2w] (written by ensure_b2 below). Single-key box (no
# B2_WRITE_*) => B2W == b2, byte-identical behavior. Keep every $B2W target
# under serve/ or the scoped key 403s. See tools/vast/CREDENTIAL_LIFECYCLE.md.
if [ -n "${B2_WRITE_KEY_ID:-}" ]; then B2W="b2w"; else B2W="b2"; fi

# --- SERVE_STATUS marker (shape copied from onstart/train.sh:106 status()) -----
# Single overwritten object at b2:$B2_BUCKET/serve/$SERVE_ID/SERVE_STATUS; the
# launch_serve wrapper + serve_ready.sh poll it (PULLING/READY/FAILED) so a
# wedged pull or dead replica is distinguishable from a slow model load without a
# hand-ssh curl. DEGRADES to a log-only no-op when SERVE_ID or B2_KEY_ID is unset
# (a raw `herdd launch --onstart serve_vllm.sh` keeps working unchanged).
_FAILED_MARKED=0
status() {
  [ "$1" = "FAILED" ] && _FAILED_MARKED=1
  if [ -n "${SERVE_ID:-}" ] && [ -n "${B2_KEY_ID:-}" ]; then
    echo "$1 $(date -u +%FT%TZ)${2:+ $2}" \
      | rclone rcat "${B2W}:${B2_BUCKET}/serve/${SERVE_ID}/SERVE_STATUS" 2>/dev/null || true
  fi
  return 0
}
# Any un-enumerated `set -e` death still writes FAILED (unless a specific FAILED
# reason was already marked). Does NOT fire on `exec` (process is replaced) nor on
# DRY_RUN's rc=0 exits. rc is assigned inside the trap body (SC2154 false positive).
# shellcheck disable=SC2154
trap 'rc=$?; [ "$rc" -ne 0 ] && [ "$_FAILED_MARKED" = "0" ] && status FAILED "onstart_abort rc=$rc"' EXIT

# Arm BEFORE any pull/serve work so a wedged adapter pull can't bill forever.
# `exec` below replaces this shell, but this already-forked background process
# survives as a separate PID (an orphan reparented to init) and keeps ticking.
MAX_HOURS="${MAX_HOURS:-12}"
if [ "$MAX_HOURS" != "0" ]; then
  if [ "$DRY_RUN" != "1" ]; then
    ( sleep "$(( MAX_HOURS * 3600 ))"
      echo "!! MAX_HOURS=${MAX_HOURS} exceeded — teardown (${TEARDOWN:-park})" >&2
      # SELF_PARKED (not FAILED) on the park path: fleetd's serve watch reads
      # this marker to tell a deliberate watchdog park from an OUTBID eviction
      # (herdd._serve_self_park_soft) — without it a spot serve box would be
      # rescue-resumed forever against its own watchdog.
      if [ "${TEARDOWN:-park}" = destroy ]; then
        status FAILED max_hours
        self_destruct
      else
        status SELF_PARKED max_hours
        self_park
      fi ) & disown
  fi
  echo ">> watchdog: ${TEARDOWN:-park} after ${MAX_HOURS}h (DRY_RUN skips arming)"
fi

# the mapped port is on a PUBLIC IP — pass --env VLLM_API_KEY=... so random
# scanners can't run inference on your billed GPU (harness: OPENAI_API_KEY=same)
# NEVER put the key on vLLM's argv (--api-key): the API server echoes its
# non-default args into the serve log, and serve logs get banked into run
# archives. vLLM reads VLLM_API_KEY from the environment; readiness_poll
# proves enforcement with a no-auth probe and FAILS the serve if the port
# answers open.
EXTRA=()
[ -n "${VLLM_API_KEY:-}" ] && export VLLM_API_KEY \
  || echo "WARNING: no VLLM_API_KEY set — server is open to the internet" >&2
[ -n "${QUANTIZATION:-}" ] && EXTRA+=(--quantization "$QUANTIZATION")
[ -n "${KV_CACHE_DTYPE:-}" ] && EXTRA+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
# Unset => flag not emitted => vLLM's `auto`. Same rule, same reason, as the two
# width knobs below: this path serves of-record comparands.
[ -n "${SERVE_DTYPE:-}" ] && EXTRA+=(--dtype "$SERVE_DTYPE")
# The two width knobs (2026-08-09, doc-120 §5 knob gap / EVAL_THROUGHPUT_AUDIT
# E0). UNSET MUST MEAN "DO NOT PASS THE FLAG": this is an of-record eval serving
# path, so a default here would silently move every banked comparand. There is
# deliberately no `:-<value>` on either.
[ -n "${MAX_NUM_SEQS:-}" ] && EXTRA+=(--max-num-seqs "$MAX_NUM_SEQS")

# `--max-num-batched-tokens` is MADE EXPLICIT here (2026-08-24) at vLLM's OWN
# value, so the argv changes and the behaviour does not. Why it had to: `vllm
# serve` never prints the integer it resolved -- the `Chunked prefill is enabled
# with max_num_batched_tokens=N` line is in-process only -- so on this path the
# prefill budget was inferable and never evidenced, and a $0 census of ~570
# banked run dirs could assert it for NONE of the 12 serve-path runs
# (<upstream-bench> .../2026-08-23-mnbt-2x2/MNBT_CENSUS.md §1). Emitting it turns
# the next census into a grep. The term itself is worth ~1%, monotone, bigger
# slightly better (20 arms, f8d3efcff) -- which is exactly why the value is
# REPRODUCED rather than optimised: moving it is a comparand move and needs a
# bridge cell, and this path serves of-record comparands.
_mnbt_serve_default() {
  # vLLM EngineArgs._set_default_args_v1, API-SERVER row: 2048 under a 70 GiB
  # device, 8192 at or above. The in-process row is 8192/16384 and splicing the
  # two is how the campaign's own doc got this wrong. No card in this estate
  # sits near the boundary (48 GB -> 46.6 GiB, H100 80 GB -> 79.7 GiB), so the
  # nvidia-smi-vs-torch reporting gap cannot flip the answer.
  local mib="${MNBT_DEVICE_TOTAL_MIB:-}"          # override is test-only
  [ -z "$mib" ] && mib="$(nvidia-smi --query-gpu=memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  case "$mib" in ''|*[!0-9]*) return 1 ;; esac
  [ "$mib" -ge 71680 ] && echo 8192 || echo 2048
}
MNBT_PROVENANCE=suppressed
if [ "${MAX_NUM_BATCHED_TOKENS:-}" = "none" ]; then
  MAX_NUM_BATCHED_TOKENS=""
  echo ">> max_num_batched_tokens: flag SUPPRESSED — vLLM resolves it silently, as before 2026-08-24"
elif [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
  MNBT_PROVENANCE=explicit
  EXTRA+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
elif MAX_NUM_BATCHED_TOKENS="$(_mnbt_serve_default)"; then
  MNBT_PROVENANCE=derived
  EXTRA+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  echo ">> max_num_batched_tokens: $MAX_NUM_BATCHED_TOKENS (vLLM's own vllm-serve default for this card, made explicit — behaviour unchanged, now recorded)"
else
  # Fail OPEN to the old behaviour rather than guessing a budget: an unreadable
  # card is exactly the case where a wrong derived value would be invisible.
  MNBT_PROVENANCE=unreadable_device
  MAX_NUM_BATCHED_TOKENS=""
  echo "WARNING: device memory unreadable — --max-num-batched-tokens left off, so the resolved prefill budget goes unrecorded again" >&2
fi

[ "${TRUST_REMOTE_CODE:-0}" = "1" ] && EXTRA+=(--trust-remote-code)

# Prefix caching: OPT-OUT (owner 2026-08-22), for throughput (2.02x) -- but it
# IS a comparand term, exactly like max_num_seqs. The "output-identical" premise
# this default was argued from was tested on 2026-08-24 and REFUTED: a hit does
# not reproduce the prefill it replaces, and setting the flag moves
# mamba_cache_mode none->align, which re-pages state on EVERY request, hit or
# miss. Pin it per comparison and record what the engine resolved.
# It is not redundant with vLLM's own default: 0.27 reads
# `default_prefix_caching = is_prefix_caching_supported and not is_hybrid`, so
# every hybrid (all Qwen3.5/3.6/3.8) silently served with it OFF. SERVE_PREFIX_CACHING=0
# emits nothing and restores vLLM's per-model default (the escape hatch for a
# model that cannot support it); =1/unset forces it on.
if [ "${SERVE_PREFIX_CACHING:-1}" = "1" ]; then
  EXTRA+=(--enable-prefix-caching)
fi

# --- B2 remote (only when we must pull adapters or a chat template) -----------
# Idempotent [b2] remote, copied from onstart/eval_sidecar.sh (~L147-167).
ensure_b2() {
  : "${B2_BUCKET:?LORA_SPECS/CHAT_TEMPLATE_B2 set but B2_BUCKET missing}"
  : "${B2_KEY_ID:?LORA_SPECS/CHAT_TEMPLATE_B2 set but B2_KEY_ID missing}"
  : "${B2_APPLICATION_KEY:?LORA_SPECS/CHAT_TEMPLATE_B2 set but B2_APPLICATION_KEY missing}"
  : "${B2_S3_ENDPOINT:?LORA_SPECS/CHAT_TEMPLATE_B2 set but B2_S3_ENDPOINT missing}"
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
  command -v rclone >/dev/null || { echo "!! rclone install failed — cannot reach B2" >&2; exit 1; }
  # Honour RCLONE_CONFIG: rclone itself reads it, so a writer that hardcodes
  # $HOME writes one file while every later `rclone` call reads another.
  _RCONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
  mkdir -p "$(dirname "$_RCONF")"
  # ALWAYS rewrite [b2]/[b2w] from the shipped env (other remotes preserved) —
  # never keep-if-present. The launch-side mint is revoke-then-mint on the
  # serve-<id> key names, so a --on-box re-run under the SAME --serve-id has
  # just REVOKED whatever keys an existing rclone.conf still holds; a
  # keep-if-present guard would leave the box on dead creds (pulls fail
  # loudly, SERVE_STATUS/METRICS rcats die silently behind '|| true'). A stale
  # [b2w] from a prior scoped-pair run is dropped too when this run is
  # single-key. Write-to-temp + mv keeps the swap atomic for live readers.
  _RTMP="$(mktemp "$(dirname "$_RCONF")/.rclone.conf.XXXXXX")"
  [ -f "$_RCONF" ] && awk '/^\[/{drop=($0=="[b2]"||$0=="[b2w]")} !drop' "$_RCONF" >> "$_RTMP"
  cat >> "$_RTMP" <<EOF
[b2]
type = s3
provider = Other
access_key_id = ${B2_KEY_ID}
secret_access_key = ${B2_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
no_check_bucket = true
EOF
  # Option-1b scoped write remote (same shape as onstart/jobd_boot.sh): when the
  # launcher shipped a prefix-restricted write key, serve/ writes go through
  # [b2w]; reads stay on [b2]. Absent B2_WRITE_* nothing is added (single-key box).
  if [ -n "${B2_WRITE_KEY_ID:-}" ] && [ -n "${B2_WRITE_APPLICATION_KEY:-}" ]; then
    cat >> "$_RTMP" <<EOF
[b2w]
type = s3
provider = Other
access_key_id = ${B2_WRITE_KEY_ID}
secret_access_key = ${B2_WRITE_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
no_check_bucket = true
EOF
  fi
  chmod 600 "$_RTMP"
  mv "$_RTMP" "$_RCONF"
}

if [ -n "${LORA_SPECS:-}" ] || [ -n "${CHAT_TEMPLATE_B2:-}" ] || [ -n "${MODEL_B2:-}" ]; then
  # these MUST reach B2 (adapters / template / base weights) — ensure_b2's :? is hard.
  [ "$DRY_RUN" = "1" ] || ensure_b2
elif [ -n "${SERVE_ID:-}" ] && [ -n "${B2_KEY_ID:-}" ]; then
  # marker-only serve: needs the B2 remote for SERVE_STATUS but must DEGRADE
  # (never abort) if the B2 config is partial — precheck instead of ensure_b2's :?.
  if [ -n "${B2_BUCKET:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ] && [ -n "${B2_S3_ENDPOINT:-}" ]; then
    [ "$DRY_RUN" = "1" ] || ensure_b2
  else
    echo ">> SERVE_STATUS markers disabled (B2_* incomplete for a marker-only serve)" >&2
    SERVE_ID=""   # disable status() writes cleanly
  fi
fi
[ "$DRY_RUN" = "1" ] || status PULLING boot   # first marker: creds proven, pulls next

# --- b2x/CDN shim, second attempt (now that [b2] exists) ----------------------
# The search at the top of this file looks at SIBLINGS ONLY, and a pure serve box
# has neither a jobd bundle nor an eval-env unpack — so it has always found
# nothing, installed the `return 1` stubs, and run this lane's 5-25 GB base pull
# on plain rclone. Same fetch onstart/train.sh does, moved after ensure_b2
# because there is no [b2] remote to read at the top of the file.
# `b2x_ensure` (not `b2x_pull`) is the probe: the stubs above already satisfy
# `command -v b2x_pull`. Best-effort — absent leaves the stubs exactly as they are.
if [ "$DRY_RUN" != "1" ] && ! command -v b2x_ensure >/dev/null 2>&1 \
   && [ -n "${B2_BUCKET:-}" ] && command -v rclone >/dev/null 2>&1; then
  rclone copyto "b2:${B2_BUCKET}/eval-env/cdn_pull.py" /workspace/cdn_pull.py 2>/dev/null || true
  if { rclone copyto "b2:${B2_BUCKET}/eval-env/b2x_boot.sh" /workspace/b2x_boot.sh 2>/dev/null \
       || rclone copyto "b2:${B2_BUCKET}/tools/b2x/b2x_boot.sh" /workspace/b2x_boot.sh 2>/dev/null; }; then
    unset -f b2x_pull b2x_push 2>/dev/null || true
    . /workspace/b2x_boot.sh 2>/dev/null || true
    command -v b2x_pull >/dev/null 2>&1 \
      || { b2x_pull() { return 1; }; b2x_push() { return 1; }; }
    command -v b2x_ensure >/dev/null 2>&1 && echo ">> b2x transport: fetched shim from B2"
  fi
fi

# --- DURABLE serve-model override (the flip that survives a respawn) ----------
# A flip done at runtime — kill the base vLLM, hand-start one on a merged dir —
# lives only in a process. Every restart path re-enters THIS script from the
# ORIGINAL launch env, so an eviction rescue, a park/resume or a jobs attempt
# re-run brings the LAUNCH model back up under the ratified endpoint label.
# The file below is the on-disk half, read on every start whoever starts it.
#
# It is resolved BEFORE the adapter and base pulls: a refusal must cost nothing,
# and an override target needs neither.
SERVE_MODEL_OVERRIDE="${SERVE_MODEL_OVERRIDE:-/workspace/serve_model_override.json}"
# Colon-separated paths whose existence means "somebody flipped this box".
# `.serve_flipped` is serve_flip.sh's sentinel; `base-model.parked` is the
# hand-rolled park defence, which this guard is what finally makes effective.
# Point it at a merged dir's completion marker to cover a flip done by tooling
# that predates the override file.
SERVE_FLIP_EVIDENCE="${SERVE_FLIP_EVIDENCE:-/workspace/.serve_flipped:/workspace/base-model.parked}"
OVERRIDE_ACTIVE=0
SERVE_MODEL_SOURCE=launch-env
_override_refuse() {
  local tok="$1"; shift
  status FAILED "$tok"
  echo "!! serve model override ($SERVE_MODEL_OVERRIDE): $*" >&2
  echo "!!   REFUSING to serve, and NOT falling back to the launch model. This endpoint is" >&2
  echo "!!   labelled '${SERVED_NAME:-?}' for the override target; serving the launch weights" >&2
  echo "!!   under that label is the exact failure this file exists to prevent. A down" >&2
  echo "!!   endpoint is loud, wrong weights are not." >&2
  echo "!!   Fix the target, or drop the override (serve_flip.sh --clear) to go back to the" >&2
  echo "!!   launch model deliberately." >&2
  exit 1
}
resolve_serve_model_override() {
  local ov m was ev
  OVERRIDE_ACTIVE=0
  # The one seam a WORKSTATION can reach on a box it cannot ssh into: the
  # per-serve B2 prefix serve_boot.sh already re-pulls serve_main.sh from on
  # every boot. Landing it on disk here also makes the next boot local-only.
  if [ ! -e "$SERVE_MODEL_OVERRIDE" ] && [ -n "${SERVE_ID:-}" ] \
     && [ -n "${B2_KEY_ID:-}" ] && command -v rclone >/dev/null 2>&1; then
    mkdir -p "$(dirname "$SERVE_MODEL_OVERRIDE")" 2>/dev/null || true
    if rclone copyto "b2:${B2_BUCKET}/serve/${SERVE_ID}/serve_model_override.json" \
         "$SERVE_MODEL_OVERRIDE" 2>/dev/null && [ -s "$SERVE_MODEL_OVERRIDE" ]; then
      echo ">> serve model: flip override pulled from b2:${B2_BUCKET}/serve/${SERVE_ID}/serve_model_override.json"
    else
      rm -f "$SERVE_MODEL_OVERRIDE" 2>/dev/null || true
    fi
  fi
  if [ ! -e "$SERVE_MODEL_OVERRIDE" ]; then
    # No override — but a box that carries flip evidence must not quietly
    # re-materialise and serve the LAUNCH model under the flip's endpoint
    # label. Fail closed: `Connection refused` is a state the eval gates
    # already handle, wrong weights under a ratified label is not.
    IFS=':' read -ra _EV <<< "$SERVE_FLIP_EVIDENCE"
    for ev in "${_EV[@]}"; do
      [ -n "$ev" ] && [ -e "$ev" ] || continue
      status FAILED base_over_flip
      echo "!! serve model: '$ev' says this box was FLIPPED away from the launch model," >&2
      echo "!!   and this start resolved back to it (${MODEL_B2:+b2:$MODEL_B2 }${MODEL_ID:-?})." >&2
      echo "!!   The launch env is immutable for the instance's life, so every boot after an" >&2
      echo "!!   eviction/resume lands here — serving the launch weights under the flip's" >&2
      echo "!!   endpoint label. REFUSING." >&2
      echo "!!   Install the flip durably (tools/vast/serve_flip.sh write / stage --serve-id)," >&2
      echo "!!   or clear the evidence path if the flip is genuinely over." >&2
      exit 1
    done
    SERVE_MODEL_SOURCE="launch-env"
    echo ">> serve model: LAUNCH DEFAULT ${MODEL_B2:+b2:$MODEL_B2 }${MODEL_ID:-} (no override at $SERVE_MODEL_OVERRIDE)"
    return 0
  fi
  ov="$(python3 - "$SERVE_MODEL_OVERRIDE" <<'PY'
import json, shlex, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    sys.exit("not readable as JSON: %s" % e)
if not isinstance(d, dict) or not str(d.get("model_path") or "").strip():
    sys.exit("no 'model_path' in the document")
for k in ("model_path", "marker", "served_name", "max_len", "identity_expect",
          "allow_lora", "reason", "written_at"):
    print("OV_%s=%s" % (k.upper(), shlex.quote(str(d.get(k) or ""))))
PY
)" || _override_refuse serve_override_unreadable "present but not a usable override document"
  eval "$ov"
  [ -d "$OV_MODEL_PATH" ] \
    || _override_refuse serve_override_missing_dir "target '$OV_MODEL_PATH' is not a directory on this box"
  [ -f "$OV_MODEL_PATH/config.json" ] \
    || _override_refuse serve_override_no_config "target '$OV_MODEL_PATH' holds no config.json — not a servable model dir"
  if [ -n "$OV_MARKER" ]; then
    case "$OV_MARKER" in /*) m="$OV_MARKER";; *) m="$OV_MODEL_PATH/$OV_MARKER";; esac
    [ -e "$m" ] \
      || _override_refuse serve_override_no_marker "completion marker '$m' is absent — whatever produces '$OV_MODEL_PATH' has not finished"
  fi
  # The launcher's lora_forbidden refusal, one lane over: when the override IS
  # the merge, the launch's adapter would be applied a second time and every
  # readiness gate would still be green.
  case "$OV_ALLOW_LORA" in
    true|True|1) ;;
    *) [ -z "${LORA_SPECS:-}" ] \
         || _override_refuse serve_override_lora "LORA_SPECS='${LORA_SPECS}' is still set from the launch, and mounting it over the override target applies it TWICE if that target is the merge. Set allow_lora on the override if the adapter really is not in these weights." ;;
  esac
  was="${MODEL_B2:+b2:$MODEL_B2 }${MODEL_ID:-}"
  MODEL_ID="$OV_MODEL_PATH"
  MODEL_B2=""                       # nothing to pull, and nothing may re-resolve MODEL_ID
  [ -n "$OV_SERVED_NAME" ] && SERVED_NAME="$OV_SERVED_NAME"
  [ -n "$OV_MAX_LEN" ] && MAX_LEN="$OV_MAX_LEN"
  [ -n "$OV_IDENTITY_EXPECT" ] && SERVE_IDENT_EXPECT="$OV_IDENTITY_EXPECT"
  OVERRIDE_ACTIVE=1
  SERVE_MODEL_SOURCE="flip-override:$SERVE_MODEL_OVERRIDE"
  echo "======================================================================"
  echo ">> serve model: OVERRIDE ACTIVE -> $MODEL_ID"
  echo ">>   file   : $SERVE_MODEL_OVERRIDE (written ${OV_WRITTEN_AT:-<unknown>})"
  echo ">>   reason : ${OV_REASON:-<none given>}"
  echo ">>   marker : ${OV_MARKER:-<none declared>}"
  echo ">>   served : ${SERVED_NAME:-<basename>}   max-model-len $MAX_LEN"
  echo ">>   launch default was '${was:-<none>}' — NOT served this start."
  echo "======================================================================"
  return 0
}
resolve_serve_model_override

# --- LoRA adapters -------------------------------------------------------------
# NOTE: vLLM's --lora-modules uses a custom parser action that REPLACES its value
# on every occurrence of the flag — repeating `--lora-modules a=... --lora-modules
# b=...` silently keeps only the LAST adapter. Emit the flag ONCE with all
# name=path pairs after it (nargs='+').
LORA_ARGS=()
LORA_PAIRS=()
if [ -n "${LORA_SPECS:-}" ]; then
  [ "$DRY_RUN" = "1" ] || mkdir -p /workspace/adapters
  IFS=',' read -ra _SPECS <<< "$LORA_SPECS"
  for spec in "${_SPECS[@]}"; do
    spec="$(echo "$spec" | tr -d '[:space:]')"
    [ -z "$spec" ] && continue
    name="${spec%%=*}"
    subpath="${spec#*=}"
    if [ "$name" = "$spec" ] || [ -z "$name" ] || [ -z "$subpath" ]; then
      echo "!! LORA_SPECS entry '$spec' is not name=b2subpath" >&2; exit 1
    fi
    dest="/workspace/adapters/$name"
    if [ "$DRY_RUN" != "1" ]; then
      status PULLING "adapter:$name"
      echo ">> pulling adapter '$name' <- b2:${B2_BUCKET}/${subpath}"
      b2x_pull "b2:${B2_BUCKET}/${subpath}" "$dest" --exclude 'checkpoint-*/**' \
      || rclone copy "${RC_FAST[@]}" --exclude 'checkpoint-*/**' "b2:${B2_BUCKET}/${subpath}" "$dest" \
        || { status FAILED "adapter_pull $name"; echo "!! adapter pull failed: $name" >&2; exit 1; }
      # a missing adapter_config.json is a SILENT no-op LoRA at serve time —
      # vLLM would happily start and route requests to the base weights.
      [ -f "$dest/adapter_config.json" ] \
        || { status FAILED "adapter_config_missing $name"; echo "!! $dest/adapter_config.json missing — LoRA '$name' would be a silent no-op; aborting" >&2; exit 1; }
    fi
    LORA_PAIRS+=("${name}=${dest}")
  done
  if [ "${#LORA_PAIRS[@]}" -gt 0 ]; then
    LORA_ARGS+=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-32}"
                --lora-modules "${LORA_PAIRS[@]}")
  fi
fi

# --- chat template (template-less bases like gemma-4) -------------------------
CHAT_ARGS=()
if [ -n "${CHAT_TEMPLATE_B2:-}" ]; then
  if [ "$DRY_RUN" != "1" ]; then
    status PULLING chat_template
    echo ">> pulling chat template <- b2:${B2_BUCKET}/${CHAT_TEMPLATE_B2}"
    rclone copyto "b2:${B2_BUCKET}/${CHAT_TEMPLATE_B2}" /workspace/chat_template.jinja \
      || { status FAILED template_pull; echo "!! chat template pull failed" >&2; exit 1; }
  fi
  CHAT_ARGS+=(--chat-template /workspace/chat_template.jinja)
fi

# --- B2-staged base model (bakeoff roster; HF-Xet-deadlock-safe path) ----------
# When MODEL_B2 is set the box pulls the base weights from B2 instead of HF. This
# is what makes the B2-staged bakeoff roster servable — EXP2's claim that
# serve_vllm "already serves from a B2 base dir" was FALSE (only adapters and the
# chat template were pulled). Resolves MODEL_ID to the local dir afterwards.
if [ -n "${MODEL_B2:-}" ]; then
  if [ "$DRY_RUN" != "1" ]; then
    status PULLING base
    echo ">> pulling base model <- b2:${B2_BUCKET}/${MODEL_B2}"
    b2x_pull "b2:${B2_BUCKET}/${MODEL_B2}" /workspace/base-model \
    || rclone copy "${RC_FAST[@]}" "b2:${B2_BUCKET}/${MODEL_B2}" /workspace/base-model \
      || { status FAILED base_pull; echo "!! base model pull failed" >&2; exit 1; }
    [ -f /workspace/base-model/config.json ] \
      || { status FAILED base_pull; echo "!! /workspace/base-model/config.json missing after pull" >&2; exit 1; }
  fi
  MODEL_ID=/workspace/base-model
  SERVE_MODEL_SOURCE="launch-env:MODEL_B2=$MODEL_B2"
fi

# One line on EVERY start naming what is about to be served and where that
# decision came from. The 2026-08-26 revert had to be reconstructed from the
# instance record because no log line said which of the two it had resolved.
echo ">> serve model RESOLVED: $MODEL_ID  (source: $SERVE_MODEL_SOURCE, served-model-name: ${SERVED_NAME:-$(basename "$MODEL_ID")}, max-model-len: $MAX_LEN)"

# --- ON-BOX IDENTITY GATE (after the pull, before ANY vllm argv) --------------
# The one check that can see the failure every other one is blind to. On
# 2026-08-21 a stale MODEL_B2 inherited through /etc/environment made this
# script serve the wrong weights while /v1/models named the right model and
# serve_ready.sh passed: every name-level gate agreed with every other, because
# they were all reading the same label. Only a claim about BYTES, made here, on
# the dir that was actually downloaded, separates those.
#
# It runs BEFORE build_vllm_argv on purpose — a refusal must cost nothing but
# the pull, and a partially-started engine holding VRAM is not a refusal.
SERVE_IDENT_SHA12=""
SERVE_IDENT_GRADE=""
SERVE_IDENT_SHA256=""

# One staged asset. Local candidates first, then the per-SERVE B2 prefix the
# launcher rcat'd them to (the B2-boot lane ships no repo checkout, and none of
# these fits the 16 KiB onstart wire). Prints the resolved path; rc 1 = nowhere.
_stage_pull() {
  local name="$1" explicit="${2:-}" p
  for p in "$explicit" "/workspace/$name" "$(dirname "$0")/$name"; do
    [ -n "$p" ] && [ -s "$p" ] && { printf '%s' "$p"; return 0; }
  done
  if [ -n "${SERVE_ID:-}" ] && [ -n "${B2_KEY_ID:-}" ] && command -v rclone >/dev/null 2>&1; then
    mkdir -p /workspace
    if rclone copyto "b2:${B2_BUCKET}/serve/${SERVE_ID}/${name}" "/workspace/$name" 2>/dev/null \
       && [ -s "/workspace/$name" ]; then
      printf '%s' "/workspace/$name"; return 0
    fi
  fi
  return 1
}

verify_model_identity() {
  local expect gate fpt dht out verdict
  expect="$(_stage_pull identity_expect.json "${SERVE_IDENT_EXPECT:-}")" || {
    status FAILED identity_expect_missing
    echo "!! identity gate: SERVE_IDENT_REQUIRED=1 but identity_expect.json is not on this box" >&2
    echo "!!   and could not be pulled from b2:${B2_BUCKET:-?}/serve/${SERVE_ID:-?}/. This is a" >&2
    echo "!!   REFUSAL, not a skip: the launcher shipped an expectation, so serving without" >&2
    echo "!!   checking it would be exactly the unverified serve the flag was passed to prevent." >&2
    exit 1; }
  gate="$(_stage_pull serve_identity_gate.py "${SERVE_IDENT_GATE:-}")" || {
    status FAILED identity_gate_missing
    echo "!! identity gate: serve_identity_gate.py could not be staged — refusing (see above)" >&2
    exit 1; }
  fpt="$(_stage_pull merged_fingerprint.py "${SERVE_IDENT_FINGERPRINT:-}")" || {
    status FAILED identity_gate_missing
    echo "!! identity gate: merged_fingerprint.py could not be staged — refusing (see above)" >&2
    exit 1; }
  # grade B only; the launcher stages it iff a content pin exists, and the gate
  # itself refuses when the pin is there and the tool is not.
  dht="$(_stage_pull dirhash.py "${SERVE_IDENT_DIRHASH:-}")" || dht=""

  echo ">> identity gate: verifying $MODEL_ID against artifact '${SERVE_IDENT_ARTIFACT:-?}'"
  out="$(python3 "$gate" --dir "$MODEL_ID" --expect "$expect" \
          --fingerprint-tool "$fpt" ${dht:+--dirhash-tool "$dht"} \
          --out /workspace/identity_report.json)" || {
    verdict="$(printf '%s\n' "$out" | tail -n1)"
    case "$verdict" in
      IDENTITY_CANNOT_CHECK) status FAILED identity_cannot_check ;;
      *)                     status FAILED identity_mismatch ;;
    esac
    echo "!! identity gate REFUSED — not serving. The dir pulled to $MODEL_ID is not the" >&2
    echo "!!   artifact this serve was launched for. Report: /workspace/identity_report.json" >&2
    exit 1; }
  printf '%s\n' "$out"
  # IDENTITY_VERIFIED <grade> <sha12> <sha256>
  verdict="$(printf '%s\n' "$out" | tail -n1)"
  SERVE_IDENT_GRADE="$(printf '%s' "$verdict" | awk '{print $2}')"
  SERVE_IDENT_SHA12="$(printf '%s' "$verdict" | awk '{print $3}')"
  SERVE_IDENT_SHA256="$(printf '%s' "$verdict" | awk '{print $4}')"
}

# An armed expectation describes the LAUNCH artifact. Checking a flipped box
# against it can only fail, and skipping it would disarm the gate — so the
# override has to carry its own, or this is a refusal.
if [ "${SERVE_IDENT_REQUIRED:-}" = "1" ] && [ "${OVERRIDE_ACTIVE:-0}" = "1" ] \
   && [ -z "${OV_IDENTITY_EXPECT:-}" ]; then
  status FAILED serve_override_ungated
  echo "!! identity gate: this serve was launched with an expectation for artifact" >&2
  echo "!!   '${SERVE_IDENT_ARTIFACT:-?}' and ${SERVE_MODEL_OVERRIDE:-the flip override} points serving elsewhere." >&2
  echo "!!   Give the override an 'identity_expect' naming its OWN target, or flip a box" >&2
  echo "!!   that was launched ungated. REFUSING." >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  # Nothing was pulled, so there is nothing to fingerprint — but say which way
  # the gate WOULD have gone, or --dry-run cannot show whether a serve is
  # armed, which is the one thing a caller wants to check before spending.
  if [ "${SERVE_IDENT_REQUIRED:-}" = "1" ]; then
    echo ">> identity gate: DRY_RUN — ARMED, would verify $MODEL_ID against artifact '${SERVE_IDENT_ARTIFACT:-?}'"
  else
    echo ">> identity gate: DRY_RUN — UNARMED, would skip (no --model-artifact)"
  fi
elif [ "${SERVE_IDENT_REQUIRED:-}" = "1" ]; then
  status PULLING identity_gate
  verify_model_identity
elif [ -n "${SERVE_IDENT_EXPECT:-}" ] || [ -s /workspace/identity_expect.json ]; then
  # An expectation is lying around from an EARLIER serve on this box but this
  # run was not armed. Gating against it would compare this model to the last
  # one's fingerprint — the same stale-inheritance shape as the MODEL_B2 bug.
  echo ">> identity gate: SKIPPED — this serve shipped no SERVE_IDENT_REQUIRED, and a" >&2
  echo "   leftover identity_expect.json from an earlier serve on this box is NOT this" >&2
  echo "   run's expectation. Relaunch with --model-artifact <slug> to gate the bytes." >&2
else
  echo ">> identity gate: SKIPPED — no identity expectation was shipped with this serve." >&2
  echo "   /v1/models proves the LABEL, never the WEIGHTS: a wrong-model serve boots," >&2
  echo "   answers, and scores like the baseline. Launch with --model-artifact <slug>" >&2
  echo "   (tools/vast/MERGED_MODEL_ARTIFACTS.md) to gate this." >&2
fi

# --- MTP speculative decoding (ON by default where a head exists -- measured) -
# Must run AFTER the B2 pull above: `auto` reads the checkpoint off disk.
#   SERVE_MTP=auto (default) on IFF the checkpoint ships an MTP head. Fails
#                            CLOSED: no head, nothing on disk, or a bare HF id
#                            -> off.
#   SERVE_MTP=1              force on even with no detectable head
#   SERVE_MTP=0              off. THE OPT-OUT, and it is not decoration -- use
#                            it for a min_p / logit_bias sampling lane and to
#                            hold a frozen comparand on the OFF cohort.
#
# DEFAULT FLIPPED 2026-08-27 (owner directive). The OFF default below, and the
# LoRA stand-down that went with it, rested on ONE measurement. That
# measurement is not wrong -- it is a different WORKLOAD, and both runs of
# record are named here because neither alone predicts the other.
#
#   ANCHOR, 2026-08-22 -- qwen35-9b, one 5090, bench_serve_defaults.py: a
#   SHARED ~2k prefix, 192-token generations, T=0.6.
#     k=20  904 tok/s off vs 883 on  -> -2.3%
#     k=1    72 tok/s off vs  82 on  -> +13.1%
#   It still reproduces: re-run on the 2026-08-27 box with its own harness it
#   gave the same SIGN (+16.2% k=1, -26.1% k=20) at acceptance 0.34-0.75.
#
#   RUN OF RECORD, 2026-08-27 -- <upstream-bench>/archive/runs/2026-08-27-v14-lora-mtp/.
#   1x RTX PRO 6000 Blackwell, vLLM 0.27.1.post1+fork.gfb8e9ed57, v14 LoRA r64
#   attached UNMERGED, eval-format prompts (p50 6,259 prompt tokens), greedy.
#   Output tok/s, off -> n=5:
#     k=1   71.8 -> 218.6  (+205%)
#     k=9   92.9 -> 291.0  (+213%)
#     k=20 190.3 -> 580.1  (+205%)
#
# THE DISCRIMINATOR IS ACCEPTANCE, not concurrency, not the card: 0.34-0.75 on
# the anchor's generic prompts at T=0.6 versus 0.932-0.944 here. Depth only pays
# when the text is predictable, and format-aligned eval prompts are.
#
# n=5 and not 1: n=1 is only ~+45%, so a depth-1 default leaves most of the win
# on the table. The 1-layer head is NOT clamped -- vLLM reuses it
# autoregressively 5x and warns (speculative.py:930) that acceptance MAY fall.
# Measured here it does not.
#
# The LoRA stand-down is REMOVED, not relaxed, because it was measured WRONG:
# attaching the adapter RAISES n=5 acceptance by 21-35 points over base
# (0.9102/0.9169 vs 0.6995/0.5675 at k=1/k=9). The adapter's output on its own
# format is exactly what the shared MTP head predicts most easily.
#
# TWO COSTS, and they are the whole reason `0` exists:
#   * min_p and logit_bias are REFUSED under spec decode -- the engine warns
#     once at startup, then REJECTS each request carrying either (per-request
#     VLLMValidationError, an HTTP 400 -- verified on the fork at the pin,
#     sampling_params.py:887; the first landing of this note said "silently
#     ignored", which was wrong). Loud, but it lands MID-RUN, after the spend
#     -- a sampling lane that uses either must pass --mtp 0 up front. Preflight
#     guard + the runtime discriminator (`vllm:spec_decode_*` on /metrics):
#     tools/vast/serve_sampling_guard.py. A fork fix exists, unlanded:
#     docs/plans/witness/MINP_UNDER_SPEC_DECODE_2026-08-27.md.
#   * Greedy output is NOT bitwise stable across the arms (20/24 identical at
#     k=20 n=5). Rejection sampling preserves the distribution in EXPECTATION,
#     not in realization, and vLLM guarantees nothing here
#     (EVAL_THROUGHPUT_AUDIT_2026-08-09 §447). MTP IS A COHORT TERM: never let a
#     frozen comparison gain it. Cohort note:
#     docs/plans/witness/MTP_SERVE_DEFAULT_COHORT_2026-08-27.md
#
# Detection is local-path only -- a bare HF id is not on disk yet, and stalling
# boot to probe the Hub would trade a throughput knob for a boot-failure mode.
# ASSERT THE ENGINE, NEVER THE FLAG: what actually resolved is
# `speculative_config=` in the engine banner, harvested into the serve_memory/v1
# artifact by parse_vllm_mem.py. The line printed below records INTENT.
SPEC_ARGS=()
SERVE_MTP_RESOLVED=off
_mtp_head_present() {
  local d="$1"
  [ -d "$d" ] || return 1
  # The head is a `*mtp*` tensor. Prefer the index (cheap); fall back to the
  # shard names for single-file or non-indexed checkpoints.
  if ls "$d"/*.safetensors.index.json >/dev/null 2>&1; then
    grep -qil 'mtp' "$d"/*.safetensors.index.json && return 0
  fi
  ls "$d" 2>/dev/null | grep -qi 'mtp' && return 0
  grep -qi 'num_nextn_predict_layers\|mtp' "$d/config.json" 2>/dev/null && return 0
  return 1
}
_mtp_spec_n="${SERVE_MTP_NUM_SPEC:-5}"
case "${SERVE_MTP:-auto}" in
  0|off)
    echo ">> MTP: OFF — explicit --mtp 0. min_p/logit_bias ARE honoured; this serve is on the OFF cohort." ;;
  1|on)     SERVE_MTP_RESOLVED=forced ;;
  auto|"")
    if _mtp_head_present "$MODEL_ID"; then
      SERVE_MTP_RESOLVED=auto
    else
      echo ">> MTP auto: OFF (no MTP head found in $MODEL_ID)"
    fi ;;
  *) echo "!! SERVE_MTP='$SERVE_MTP' not in {auto,1,0} — treating as auto" >&2
     if _mtp_head_present "$MODEL_ID"; then SERVE_MTP_RESOLVED=auto; fi ;;
esac
if [ "$SERVE_MTP_RESOLVED" != "off" ]; then
  # Record the resolved depth so the serve summary reports what ran, not "".
  SERVE_MTP_NUM_SPEC="$_mtp_spec_n"
  SPEC_ARGS=(--speculative-config \
    "{\"method\":\"${SERVE_MTP_METHOD:-mtp}\",\"num_speculative_tokens\":${SERVE_MTP_NUM_SPEC}}")
  echo ">> MTP $SERVE_MTP_RESOLVED: ON  n=${SERVE_MTP_NUM_SPEC} method=${SERVE_MTP_METHOD:-mtp}${LORA_SPECS:+  (+LoRA — measured 2026-08-27 to RAISE acceptance, not lower it)}"
  echo ">>   min_p and logit_bias are IGNORED by the engine under spec decode — pass --mtp 0 for a sampling lane."
  echo ">>   MTP is a COHORT term (greedy output is not bitwise stable) — do not compare a frozen number across it."
  echo ">>   Assert it from the engine: 'speculative_config=' in the banner, and vllm:spec_decode_* on /metrics."
fi

# --- assemble the `vllm serve` argv (one source of truth) ---------------------
# Populate the global VLLM_ARGV array for a given <port> <host>. Both the single
# instance and every HAProxy replica call this, so flags stay identical — only
# host/port differ. The adapter pull + EXTRA/LORA_ARGS/CHAT_ARGS arrays above are
# built ONCE and shared: every replica reads the same /workspace/adapters/*.
build_vllm_argv() {
  local port="$1" host="$2"
  # shellcheck disable=SC2206  # VLLM_EXTRA_ARGS is INTENTIONALLY word-split into argv
  VLLM_ARGV=(vllm serve "$MODEL_ID"
    --host "$host" --port "$port"
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
    --served-model-name "${SERVED_NAME:-$(basename "$MODEL_ID")}"
    ${PAR_ARGS[@]+"${PAR_ARGS[@]}"}
    ${EXTRA[@]+"${EXTRA[@]}"}
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
    ${LORA_ARGS[@]+"${LORA_ARGS[@]}"}
    ${CHAT_ARGS[@]+"${CHAT_ARGS[@]}"}
    ${VLLM_EXTRA_ARGS:-})
}

# Resolve replica count: 'auto' -> GPU count; non-numeric/<1 -> 1.
if [ "$SERVE_REPLICAS" = "auto" ]; then
  SERVE_REPLICAS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
case "$SERVE_REPLICAS" in ''|*[!0-9]*) SERVE_REPLICAS=1 ;; esac
if [ "$SERVE_REPLICAS" -lt 1 ]; then SERVE_REPLICAS=1; fi

# --- native DP/TP resolution (the preferred multi-GPU path) --------------------
# SERVE_DP -> --data-parallel-size, SERVE_TP -> --tensor-parallel-size on the
# SINGLE-instance path: one `vllm serve` on :8000, N engine-core processes with
# queue-aware internal LB (docs/serving/data_parallel_deployment.md in the vLLM
# tree). DP composes with TP (DP*TP GPUs); 'auto' = GPU count / SERVE_TP.
# a DEFAULTED dp-auto yields to the legacy replicas path (an EXPLICIT dp>1
# still trips the mutual-exclusion error below, on purpose)
if [ "${_DP_DEFAULTED:-0}" = "1" ] && [ "$SERVE_REPLICAS" -gt 1 ]; then SERVE_DP=1; fi
case "$SERVE_TP" in ''|*[!0-9]*) SERVE_TP=1 ;; esac
if [ "$SERVE_TP" -lt 1 ]; then SERVE_TP=1; fi
if [ "$SERVE_DP" = "auto" ]; then
  _ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  case "$_ngpu" in ''|*[!0-9]*) _ngpu=0 ;; esac
  SERVE_DP=$(( _ngpu / SERVE_TP ))
fi
case "$SERVE_DP" in ''|*[!0-9]*) SERVE_DP=1 ;; esac
if [ "$SERVE_DP" -lt 1 ]; then SERVE_DP=1; fi
# One multi-GPU mode at a time: native DP/TP owns the GPUs inside one server,
# the HAProxy replica path pins CUDA_VISIBLE_DEVICES per process — mixing them
# double-books cards.
if [ "$SERVE_REPLICAS" -gt 1 ] && { [ "$SERVE_DP" -gt 1 ] || [ "$SERVE_TP" -gt 1 ]; }; then
  status FAILED "config_conflict dp_tp_vs_replicas"
  echo "!! SERVE_REPLICAS>1 is mutually exclusive with SERVE_DP/SERVE_TP>1 — pick native DP (preferred) or HAProxy replicas" >&2
  exit 1
fi
PAR_ARGS=()
[ "$SERVE_TP" -gt 1 ] && PAR_ARGS+=(--tensor-parallel-size "$SERVE_TP")
[ "$SERVE_DP" -gt 1 ] && PAR_ARGS+=(--data-parallel-size "$SERVE_DP")

# --- sampler warmup (best-effort; opt-in =1 lane only) ---------------------------
# With =1, vLLM 0.19.1 pays the flashinfer sampling JIT at engine startup, but
# lazy per-shape paths can still compile on the first real generation — one
# throwaway completion flushes them (and the disk cache is per-arch, so all
# HAProxy replicas benefit) before we mark READY, and proves generation actually
# works. temperature>0 + top_p<1 is required: greedy (temperature=0) is argmax
# and never touches the top-k/top-p sampler. Gated on the flashinfer sampler
# being active — the native (default) lane's readiness path stays byte-identical.
# NON-fatal on any failure. $1 = first served id.
warmup_sampler() {
  [ "${VLLM_USE_FLASHINFER_SAMPLER:-0}" = "1" ] || return 0
  local _id="$1" _auth=() _code
  [ -z "$_id" ] && return 0
  [ -n "${VLLM_API_KEY:-}" ] && _auth=(-H "Authorization: Bearer ${VLLM_API_KEY}")
  _code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 180 \
    ${_auth[@]+"${_auth[@]}"} -H 'Content-Type: application/json' \
    http://127.0.0.1:8000/v1/completions \
    -d "{\"model\":\"${_id}\",\"prompt\":\"warmup\",\"max_tokens\":4,\"temperature\":1.0,\"top_p\":0.9}" 2>/dev/null || true)"
  if [ "$_code" = "200" ]; then
    echo ">> sampler warmup ok — lazy sampler paths flushed (model '${_id}')"
  else
    echo ">> sampler warmup non-200 (http=${_code:-none}) — non-fatal; first real request may JIT" >&2
  fi
}

# --- serve memory profile capture (best-effort; the V10 sec-7 gap) -------------
# Every serve launch should leave behind what vLLM's own memory profiler said,
# because nothing ever recorded it: V10_SPOT_PROVISIONING_2026-08-08.md sec 7
# sized a 96 GB pin with "non-torch + activation peak + CUDA graphs" quoted as a
# 3-5 GiB ESTIMATED band, naming the cause — "the bundle never captures vLLM's
# own memory profiler output". The numbers ARE printed at engine init; they just
# died with the box. tools/vast/parse_vllm_mem.py turns them into
# serve_summary.json; this is the two-call hook that runs it.
#
# NO NEW SYNC CHANNEL. Two destinations, both already carried:
#   * beside the serve log — the jobs lane points SERVE_LOG inside the job's
#     synced output dir (jobs/eval-template/run.sh: SERVE_LOG="$OUT/eval/serve.log"),
#     so the summary rides back with serve.log and eval_summary.json;
#   * serve/<SERVE_ID>/serve_summary.json on B2 — the same prefix (and the same
#     [b2w] write remote) as SERVE_STATUS and METRICS.
#
# STRICTLY ADDITIVE. It runs AFTER the READY marker is already written, is
# wrapped in `|| true` at the call site, and every failure is a note. A box with
# no parser, no python3, no log and no B2 degrades to one stderr line.
_resolve_mem_parser() {
  local p
  for p in "${VLLM_MEM_PARSER:-}" \
           "$(dirname "$0")/../parse_vllm_mem.py" \
           "$(dirname "$0")/parse_vllm_mem.py" \
           /workspace/eval/upstream-monorepo/tools/vast/parse_vllm_mem.py \
           /workspace/parse_vllm_mem.py; do
    [ -n "$p" ] && [ -f "$p" ] && { printf '%s' "$p"; return 0; }
  done
  # The launch_serve.sh B2-boot lane ships no repo checkout, so the launcher
  # stages the parser next to serve_main.sh under this SERVE_ID; pull it.
  if [ -n "${SERVE_ID:-}" ] && [ -n "${B2_KEY_ID:-}" ] && command -v rclone >/dev/null 2>&1; then
    if rclone copyto "b2:${B2_BUCKET}/serve/${SERVE_ID}/parse_vllm_mem.py" \
         /workspace/parse_vllm_mem.py 2>/dev/null && [ -s /workspace/parse_vllm_mem.py ]; then
      printf '%s' /workspace/parse_vllm_mem.py; return 0
    fi
  fi
  return 1
}

capture_serve_summary() {
  local parser out fd1 mode
  command -v python3 >/dev/null 2>&1 || { echo ">> serve summary: no python3 — skipped" >&2; return 0; }
  parser="$(_resolve_mem_parser)" || {
    echo ">> serve summary: parse_vllm_mem.py not on this box — skipped (point VLLM_MEM_PARSER at it)" >&2
    return 0; }
  # Where the engine's stdout actually landed differs per lane, so hand the
  # parser EVERY candidate and let it pick by content (see choose_log): the job
  # lane's $SERVE_LOG, our own fd 1 (job lane redirect / serve_boot's tee), the
  # boot-pull lane's onstart.log, and the HAProxy lane's per-replica logs.
  # $$ is THIS shell's pid on purpose: /proc/self/fd/1 inside `readlink` would be
  # readlink's own stdout, i.e. the command substitution's pipe, not the log.
  # A pipe (serve_boot's tee lane) makes readlink fail -> empty -> skipped.
  fd1="$(readlink -f "/proc/$$/fd/1" 2>/dev/null || true)"
  mode=single
  if [ "${SERVE_REPLICAS:-1}" != "1" ]; then mode=haproxy; fi
  # stdout = the artifact path (captured); stderr = the parser's own breadcrumb,
  # inherited so it lands in this serve log next to the READY banner.
  out="$(python3 "$parser" \
      --log "${VLLM_MEM_LOG:-}" --log "${SERVE_LOG:-}" --log "$fd1" \
      --log /workspace/serve.log --log /workspace/onstart.log \
      --log /workspace/vllm-0.log \
      --nvidia-smi --print \
      --field "serve_id=${SERVE_ID:-}" \
      --field "served_name=${SERVED_NAME:-}" \
      --field "model=${MODEL_B2:-$MODEL_ID}" \
      --field "max_len=${MAX_LEN:-}" \
      --field "gpu_util=${GPU_UTIL:-}" \
      --field "serve_dp=${SERVE_DP:-1}" \
      --field "serve_tp=${SERVE_TP:-1}" \
      --field "serve_replicas=${SERVE_REPLICAS:-1}" \
      --field "serve_mode=$mode" \
      --field "quantization=${QUANTIZATION:-}" \
      --field "kv_cache_dtype=${KV_CACHE_DTYPE:-}" \
      --field "max_num_seqs=${MAX_NUM_SEQS:-}" \
      --field "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-}" \
      --field "max_num_batched_tokens_provenance=${MNBT_PROVENANCE:-unknown}" \
      --field "prefix_caching=${SERVE_PREFIX_CACHING:-1}" \
      --field "mtp=${SERVE_MTP_RESOLVED:-off}" \
      --field "mtp_num_spec=${SERVE_MTP_NUM_SPEC:-}" \
      --field "lora_specs=${LORA_SPECS:-}" \
      --field "max_lora_rank=${MAX_LORA_RANK:-}" \
      --field "served_ids=${1:-}" \
      --field "artifact=${SERVE_IDENT_ARTIFACT:-}" \
      --field "ident_sha256=${SERVE_IDENT_SHA256:-}" \
      --field "ident_grade=${SERVE_IDENT_GRADE:-}")" || true
  if [ -z "$out" ] || [ ! -f "$out" ]; then
    echo ">> serve summary: parser wrote nothing (no readable serve log?) — skipped" >&2
    return 0
  fi
  echo ">> serve summary: $out"
  if [ -n "${SERVE_ID:-}" ] && [ -n "${B2_KEY_ID:-}" ]; then
    if rclone rcat "${B2W}:${B2_BUCKET}/serve/${SERVE_ID}/serve_summary.json" \
         < "$out" 2>/dev/null; then
      echo ">> serve summary -> b2:${B2_BUCKET}/serve/${SERVE_ID}/serve_summary.json"
    else
      echo ">> note: serve_summary.json B2 write failed (local copy kept at $out)" >&2
    fi
  fi
  return 0
}

# --- readiness poll (ONE source of truth; single-instance + HAProxy both use it)
# Polls /v1/models on the public port (:8000) WITH the bearer until 200, then
# writes the READY marker with the comma-joined served ids; FAILED readiness_timeout
# on give-up. Backgrounded by BOTH serve paths so it observes the exec'd foreground
# process (vLLM or HAProxy) and lets that process stay PID-supervised (same orphan-
# survives-exec mechanism as the watchdog above). $1 = banner text for the log.
readiness_poll() {
  local _banner="${1:-vLLM :8000}" _auth=() _ids _mk _noauth
  [ -n "${VLLM_API_KEY:-}" ] && _auth=(-H "Authorization: Bearer ${VLLM_API_KEY}")
  for _ in $(seq 1 900); do
    if curl -fsS ${_auth[@]+"${_auth[@]}"} http://127.0.0.1:8000/v1/models -o /tmp/models.json 2>/dev/null; then
      # The key travels by env (VLLM_API_KEY), never argv — vLLM echoes argv
      # into logs that get banked. Env-based auth is only safe if it is
      # actually ENFORCED, so prove it: a no-auth request must be rejected.
      if [ -n "${VLLM_API_KEY:-}" ]; then
        _noauth="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
          http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
        if [ "$_noauth" = "200" ]; then
          status FAILED auth_unenforced
          echo "!! readiness gate: VLLM_API_KEY is set but a NO-AUTH request got 200 —" >&2
          echo "!!   the engine ignored the env var and the public port is open. Refusing READY." >&2
          return 1
        fi
      fi
      _ids="$(python3 -c 'import json;print(",".join(m["id"] for m in json.load(open("/tmp/models.json")).get("data",[])))' 2>/dev/null || true)"
      # warm the flashinfer sampler BEFORE READY so an eval's first request never JITs
      warmup_sampler "${_ids%%,*}"
      # READY <ts> <ids-csv|-> [ident=<sha12>]
      # APPEND-ONLY: serve_ready.sh's poll_marker reads the id list POSITIONALLY
      # ($3), so a new field may only go after it. When the gate ran, the id
      # list is placeholdered with `-` if empty so `ident=` is always field 4 and
      # a reader never has to guess which column it landed in; when the gate did
      # NOT run the line is byte-identical to every marker written before this.
      _mk="$_ids"
      [ -n "$SERVE_IDENT_SHA12" ] && _mk="${_ids:--} ident=$SERVE_IDENT_SHA12"
      status READY "$_mk"
      echo "======================================================================"
      echo ">> READY: ${_banner} — served: ${_ids:-<parse-failed; see /tmp/models.json>}"
      [ -n "$SERVE_IDENT_SHA12" ] \
        && echo ">> READY: grade-${SERVE_IDENT_GRADE} identity ${SERVE_IDENT_SHA12} verified on box (artifact '${SERVE_IDENT_ARTIFACT:-?}')"
      echo "======================================================================"
      # AFTER the READY marker, never before: the memory-profile capture is
      # observational and must not sit between the engine coming up and the
      # signal an eval driver is blocking on. `|| true` so it cannot fail READY.
      capture_serve_summary "$_ids" || true
      return 0
    fi
    sleep 1
  done
  status FAILED readiness_timeout
  echo "!! readiness gate: /v1/models not 200 after 900s (replicas may still be loading)" >&2
  return 1
}

# --- host-metrics sampler (opt-out: METRICS_SAMPLE=0) --------------------------
# A serve box has no periodic heartbeat loop, so this writes a rolling compact
# host-metrics line (GPU util/mem/pwr/temp + cpu/net/disk via metrics_probe.py,
# GPU-only nvidia-smi fallback) to $B2W:$B2_BUCKET/serve/$SERVE_ID/METRICS every
# METRICS_SAMPLE_S sec — so you can see whether the served model saturates the
# card or idles. Overwrite-in-place like SERVE_STATUS; best-effort; backgrounded
# before the exec (orphan-survives-exec, same as readiness_poll).
metrics_sampler() {
  local probe="" p line
  for p in "${METRICS_PROBE:-}" "$(dirname "$0")/../metrics_probe.py" \
           "$(dirname "$0")/metrics_probe.py" \
           /workspace/eval/upstream-monorepo/tools/vast/metrics_probe.py; do
    [ -n "$p" ] && [ -f "$p" ] && { probe="$p"; break; }
  done
  while true; do
    line=""
    if [ -n "$probe" ]; then
      line="$(python3 "$probe" fields 2>/dev/null || true)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      line="$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' 'NR==1{gsub(/ /,"");printf "gpu_util:%s,gpu_mem:%s,gpu_temp:%s",$1,$2,$3}')"
    fi
    [ -n "$line" ] && printf '%s %s\n' "$(date -u +%FT%TZ)" "$line" \
      | rclone rcat "${B2W}:${B2_BUCKET}/serve/${SERVE_ID}/METRICS" 2>/dev/null || true
    sleep "${METRICS_SAMPLE_S:-60}"
  done
}

# --- co-tenant CPU compile-farm (DEAD FEATURE, opt-IN; mirrors train.sh 2c) ---
# Owner ruling 2026-08-21: the farm is dead and never runs unless explicitly
# requested. Why: the sidecar's rb3-objcache grew to 69 GB and took a live
# serving box to 110/110 GB — one write from killing a serve mid-eval. It also
# starves CPU-sensitive work (see train.sh block-2c). Opt in with
# CPU_FARM=1 (launch_serve.sh --cpu-farm) and expect to babysit the disk.
CPU_FARM="${CPU_FARM:-0}"
FARM_RUN_ID="${FARM_RUN_ID:-${SERVE_ID:-satfarm-$(hostname)}}"
if [ "$DRY_RUN" != "1" ] && [ "$CPU_FARM" != "0" ]; then
  echo "!! CPU_FARM=$CPU_FARM — opting IN to the DEAD co-tenant compile farm."
  echo "!! It filled a serve box's disk (69 GB objcache -> 110/110 GB, 2026-08-20)."
  echo "!! Watch free space, or unset CPU_FARM to get the default (off)."
  if pgrep -f '[e]val_sidecar\.sh' >/dev/null 2>&1; then
    # --on-box --restart re-runs this whole script; don't stack a second farm
    echo ">> CPU_FARM: sidecar already running — leaving it alone"
  elif [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_BUCKET:-}" ] && command -v rclone >/dev/null 2>&1; then
    if rclone lsf "b2:${B2_BUCKET}/eval-env/LATEST" 2>/dev/null | grep -q . \
       && rclone copyto "b2:${B2_BUCKET}/eval-env/eval_sidecar.sh" /workspace/eval_sidecar.sh 2>/dev/null; then
      for _f in yield_fence.sh farm_worker.sh saturator.py; do
        rclone copyto "b2:${B2_BUCKET}/eval-env/${_f}" "/workspace/${_f}" 2>/dev/null || true
      done
      RUN_ID="$FARM_RUN_ID" FARM_RUN_ID="$FARM_RUN_ID" EVAL_MODE=saturate \
        setsid bash /workspace/eval_sidecar.sh >>/workspace/eval_sidecar.boot.log 2>&1 &
      echo ">> CPU_FARM: saturate sidecar pid $! (log: /workspace/eval_sidecar.log; shards: evals/${FARM_RUN_ID}/corpus/ on B2) — unset CPU_FARM to disable"
    else
      echo ">> CPU_FARM on but no-op: eval-env not staged / sidecar pull failed (serve unaffected)"
    fi
  else
    echo ">> CPU_FARM on but no-op: B2 creds or rclone unavailable (serve unaffected)"
  fi
fi   # CPU_FARM=0 is the default and says nothing

# host-metrics sampler — start ONCE before either exec path (covers single +
# HAProxy layouts). No-op without SERVE_ID/B2 creds (raw launches unchanged).
if [ "$DRY_RUN" != "1" ] && [ "${METRICS_SAMPLE:-1}" != "0" ] \
   && [ -n "${SERVE_ID:-}" ] && [ -n "${B2_KEY_ID:-}" ]; then
  metrics_sampler & disown
  echo ">> host-metrics sampler pid $! -> serve/${SERVE_ID}/METRICS (opt out: METRICS_SAMPLE=0)"
fi

# --- single instance (SERVE_REPLICAS<=1) — TODAY'S EXACT BEHAVIOR -------------
if [ "$SERVE_REPLICAS" -le 1 ]; then
  build_vllm_argv 8000 0.0.0.0
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${VLLM_ARGV[@]}"; echo
    exit 0
  fi
  echo "starting vLLM: $MODEL_ID (max-model-len=$MAX_LEN, util=$GPU_UTIL, dp=$SERVE_DP, tp=$SERVE_TP, adapters=${LORA_SPECS:-none})"
  # background the READY marker poll before exec — it survives the exec as an orphan.
  readiness_poll "vLLM :8000 (dp=$SERVE_DP tp=$SERVE_TP) -> ${SERVED_NAME:-$(basename "$MODEL_ID")}" & disown
  exec "${VLLM_ARGV[@]}"
fi

# --- N replicas behind HAProxy (SERVE_REPLICAS>1) -----------------------------
# One vLLM per GPU on a private loopback port; HAProxy L7-balances :8000 across
# them. MUST be `mode http` (L7): a `mode tcp` frontend would pin a persistent
# client connection to one backend and defeat the balancing. The /health check
# is tokenless (vLLM's OpenAI server exposes /health unauthenticated, 200,
# independent of --api-key).
render_haproxy_cfg() {
  local out="$1" i
  {
    cat <<EOF
global
    maxconn 2048
    log stdout format raw local0

defaults
    mode http
    option httplog
    option dontlognull
    timeout connect 10s
    timeout client ${HAPROXY_TIMEOUT_S}s
    timeout server ${HAPROXY_TIMEOUT_S}s
    timeout tunnel ${HAPROXY_TIMEOUT_S}s
    retries 2

frontend vllm_in
    bind *:8000
    default_backend vllm_pool

backend vllm_pool
    balance ${HAPROXY_BALANCE}
    option httpchk GET /health
    http-check expect status 200
    default-server inter 2s fall 3 rise 2
EOF
    for ((i = 0; i < SERVE_REPLICAS; i++)); do
      echo "    server r${i} 127.0.0.1:$((REPLICA_BASE_PORT + i)) check"
    done
    cat <<EOF

listen stats
    bind 127.0.0.1:8404
    stats enable
    stats uri /haproxy?stats
EOF
  } > "$out"
}

if [ "$DRY_RUN" = "1" ]; then
  # Preview every replica's argv (with CUDA_VISIBLE_DEVICES + --port) and the
  # fully-rendered haproxy.cfg. No apt/rclone/serve/network.
  for ((i = 0; i < SERVE_REPLICAS; i++)); do
    build_vllm_argv "$((REPLICA_BASE_PORT + i))" 127.0.0.1
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$i"
    printf '%q ' "${VLLM_ARGV[@]}"; echo
  done
  echo "# --- /workspace/haproxy.cfg ---"
  render_haproxy_cfg /dev/stdout
  exit 0
fi

# haproxy: guard-first install (Debian/Ubuntu image), same shape as ensure_b2's
# rclone guard above.
command -v haproxy >/dev/null || { timeout 120 apt-get update -qq || true; \
  timeout 180 apt-get install -y -qq haproxy; }

# Launch one vLLM per GPU as a BACKGROUND process, each to its own log (the log
# path is what a co-tenant spread test reads to confirm load is distributed).
REPLICA_PIDS=()
for ((i = 0; i < SERVE_REPLICAS; i++)); do
  _port="$((REPLICA_BASE_PORT + i))"
  build_vllm_argv "$_port" 127.0.0.1
  echo ">> replica $i: CUDA_VISIBLE_DEVICES=$i on 127.0.0.1:$_port -> /workspace/vllm-$i.log"
  CUDA_VISIBLE_DEVICES="$i" "${VLLM_ARGV[@]}" > "/workspace/vllm-$i.log" 2>&1 &
  REPLICA_PIDS+=("$!")
done
echo ">> launched ${SERVE_REPLICAS} replicas (pids: ${REPLICA_PIDS[*]})"

render_haproxy_cfg /workspace/haproxy.cfg
echo ">> rendered /workspace/haproxy.cfg (balance=${HAPROXY_BALANCE}, timeout=${HAPROXY_TIMEOUT_S}s)"

# Readiness gate: the SAME readiness_poll THROUGH HAProxy on the public port
# (:8000). Backgrounded so it can observe the exec'd HAProxy (below) and still let
# HAProxy be the foreground supervised process. Per-backend health is HAProxy's own
# httpchk/fall/rise — a replica that crashes+recovers is auto-readmitted, so there
# is no per-replica poll. Writes the READY / FAILED readiness_timeout marker.
readiness_poll "HAProxy :8000 -> ${SERVE_REPLICAS} vLLM replicas (balance=${HAPROXY_BALANCE})" & disown

# HAProxy is the foreground supervised main process (mirrors the single-path exec).
echo "starting HAProxy on :8000 across ${SERVE_REPLICAS} vLLM replicas (adapters=${LORA_SPECS:-none})"
exec haproxy -f /workspace/haproxy.cfg -db
