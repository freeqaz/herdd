#!/usr/bin/env bash
# M4-T2 real-image + version-probe canary. Runs on a GPU box under jobd; certifies
# the of-record image actually runs (GPU model + CUDA/torch/vLLM/bnb versions),
# does one real torch GPU step, a checkpoint round-trip, and emits a receipt that
# the control plane stores (keyed by CANARY_KEY) with a TTL. See README.md.
set -uo pipefail

OUT=out; mkdir -p "$OUT/ckpt"
LOG="$OUT/probe.log"
: > "$LOG"
say() { echo "[canary] $*" | tee -a "$LOG"; }

TS_START=$(date -u +%s)
IMAGE_DIGEST="${CANARY_IMAGE_DIGEST:-${HERDD_IMAGE_DIGEST:-}}"
ALLOW_NO_GPU="${CANARY_ALLOW_NO_GPU:-0}"
# rehearse.sh runs jobd CPU-only with JOBD_SKIP_GPU=1 — treat that as rehearsal
# so the bundle rehearses green; a REAL box (no such marker) stays fail-closed.
[ "${JOBD_SKIP_GPU:-0}" = "1" ] && ALLOW_NO_GPU=1
say "start job=${JOB_ID:-?} restart=${JOB_RESTART_COUNT:-0} image_digest=${IMAGE_DIGEST:-<none>}"

# --- activate the baked of-record train env (best-effort) --------------------
# /workspace/.train_env_activate is a POINTER FILE — it CONTAINS the path of the
# real activate script (/workspace/train-env/env.sh), it is not one. `source`-ing
# it directly runs that path as a COMMAND, so the venv's exports land in a child
# process and vanish: the canary then probes the SYSTEM interpreter while
# reporting success, which on t211 still yields plausible torch/vllm versions and
# hides the miss. cat the pointer, source the target (same shape as
# onstart/train.sh:586 and every jobs run.sh).
if [ -f /workspace/.train_env_activate ]; then
  _act="$(cat /workspace/.train_env_activate 2>/dev/null || true)"
  if [ -n "$_act" ] && [ -f "$_act" ]; then
    # shellcheck disable=SC1090
    . "$_act" 2>>"$LOG" || say "train_env_activate: sourcing $_act failed (non-fatal)"
  else
    say "train_env_activate: pointer present but target '${_act:-<empty>}' missing (non-fatal)"
  fi
fi

# --- checkpoint round-trip (proves the checkpoint transport) -----------------
CKPT_TOKEN="canary-${JOB_ID:-local}-${JOB_RESTART_COUNT:-0}-$TS_START"
CKPT_FILE="$OUT/ckpt/roundtrip.txt"
echo "$CKPT_TOKEN" > "$CKPT_FILE"
if [ "$(cat "$CKPT_FILE" 2>/dev/null)" = "$CKPT_TOKEN" ]; then
  CKPT_OK=1;  say "checkpoint round-trip OK ($CKPT_FILE)"
else
  CKPT_OK=0;  say "checkpoint round-trip FAILED"
fi

# --- GPU model + version + one real torch step -------------------------------
GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
if [ -z "$GPU_MODEL" ] || printf '%s' "$GPU_MODEL" | grep -qiE 'fail|error|mismatch|not found'; then
  GPU_MODEL="unavailable"
fi
say "gpu=$GPU_MODEL"

# python probe: versions + a real cuda matmul; emits key=val lines to $LOG.probe
python3 - "$OUT/versions.json" <<'PY' >>"$LOG" 2>&1
import json, sys
def ver(mod):
    try:
        m = __import__(mod); return getattr(m, "__version__", "present")
    except Exception as e:
        return f"absent:{type(e).__name__}"
info = {"torch": ver("torch"), "transformers": ver("transformers"),
        "bitsandbytes": ver("bitsandbytes"), "vllm": ver("vllm")}
step_ok = False; cuda = "n/a"
try:
    import torch
    cuda = torch.version.cuda or "n/a"
    if torch.cuda.is_available():
        a = torch.randn(256, 256, device="cuda"); b = torch.randn(256, 256, device="cuda")
        c = (a @ b).sum().item()
        step_ok = bool(c == c)   # NaN-free finite result => real GPU compute ran
        info["gpu_step_value"] = c
except Exception as e:
    info["torch_error"] = f"{type(e).__name__}: {e}"
info["cuda"] = cuda; info["gpu_step_ok"] = step_ok
json.dump(info, open(sys.argv[1], "w"), sort_keys=True)
print("versions:", json.dumps(info, sort_keys=True))
PY
VERSIONS_JSON="$OUT/versions.json"
[ -f "$VERSIONS_JSON" ] || echo '{"gpu_step_ok": false, "error": "probe-did-not-write"}' > "$VERSIONS_JSON"
STEP_OK=$(python3 -c "import json;print(str(json.load(open('$VERSIONS_JSON')).get('gpu_step_ok',False)).lower())" 2>/dev/null || echo false)

# --- fail-closed on a broken GPU box (unless rehearsal opted out) -------------
RC=0
if [ "$GPU_MODEL" = "unavailable" ] || [ "$STEP_OK" != "true" ]; then
  if [ "$ALLOW_NO_GPU" = "1" ]; then
    say "no working GPU but CANARY_ALLOW_NO_GPU=1 (rehearsal) -> pass with degraded step"
  else
    say "FAIL: no working GPU step on a real canary box"; RC=21
  fi
fi

# --- emit the receipt (terminal, written last; NOT a checkpoints: glob) -------
# All dynamic values reach python via the environment (exported below) — never
# interpolated into the heredoc, so a bash `true`/`false`/empty never becomes a
# python NameError and no value needs shell-quoting.
TS_END=$(date -u +%s)
export CANARY_RECEIPT_IMAGE_DIGEST="$IMAGE_DIGEST" CANARY_RECEIPT_GPU="$GPU_MODEL"
export CANARY_RECEIPT_CKPT_OK="$CKPT_OK" CANARY_RECEIPT_RC="$RC"
export CANARY_RECEIPT_TS_START="$TS_START" CANARY_RECEIPT_TS_END="$TS_END"
python3 - "$OUT/canary-receipt.json" "$VERSIONS_JSON" <<'PY'
import json, os, sys
e = os.environ.get
versions = json.load(open(sys.argv[2]))
ttl = int(e("CANARY_TTL_S", "86400") or "86400")
ts_end = int(e("CANARY_RECEIPT_TS_END", "0"))
receipt = {
  "v": 1, "kind": "workflow-canary-receipt",
  "key": e("CANARY_KEY", ""),
  "components": {
    "image_digest": e("CANARY_RECEIPT_IMAGE_DIGEST", ""),
    "jobd_sha": e("CANARY_JOBD_SHA", ""),
    "model_manifest_sha": e("CANARY_MODEL_SHA", ""),
    "adapter_manifest_sha": e("CANARY_ADAPTER_SHA", ""),
    "recipe_sha": e("CANARY_RECIPE_SHA", "")},
  "fidelity": "image+version-probe+gpu-step+checkpoint-roundtrip",
  "gpu_model": e("CANARY_RECEIPT_GPU", ""), "versions": versions,
  "gpu_step_ok": bool(versions.get("gpu_step_ok", False)),
  "checkpoint_roundtrip_ok": e("CANARY_RECEIPT_CKPT_OK") == "1",
  "job_id": e("JOB_ID", ""), "restart_count": int(e("JOB_RESTART_COUNT", "0") or "0"),
  "ts_start": int(e("CANARY_RECEIPT_TS_START", "0")), "ts_end": ts_end,
  "ttl_s": ttl, "expires_ts": ts_end + ttl,
  "rc": int(e("CANARY_RECEIPT_RC", "0") or "0"),
}
tmp = sys.argv[1] + ".tmp"
json.dump(receipt, open(tmp, "w"), sort_keys=True, separators=(",", ":"))
os.replace(tmp, sys.argv[1])
print("receipt written:", sys.argv[1])
PY
say "done rc=$RC receipt=$OUT/canary-receipt.json"
exit "$RC"
