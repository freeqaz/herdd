#!/usr/bin/env bash
# hf_login.sh — land $HF_TOKEN on a rented box for auth'd, full-bandwidth HF pulls.
# herdd.py `launch` auto-prepends this to the onstart + passes the token in as
# env HF_TOKEN. Reads $HF_TOKEN from env (NO secret baked in). Idempotent/guarded.
# Writes $HF_HOME/token (hub/hf/vLLM) + /etc/environment (later SSH). Kept terse:
# the whole file rides a near-16KB-capped onstart. Does NOT set
# HF_HUB_ENABLE_HF_TRANSFER (hard-errors without hf_transfer installed).
if [ -n "${HF_TOKEN:-}" ]; then
  _hf_home="${HF_HOME:-$HOME/.cache/huggingface}"; mkdir -p "$_hf_home" 2>/dev/null || true
  printf '%s' "$HF_TOKEN" > "$_hf_home/token" 2>/dev/null && chmod 600 "$_hf_home/token" 2>/dev/null || true
  grep -q '^HF_TOKEN=' /etc/environment 2>/dev/null || echo "HF_TOKEN=${HF_TOKEN}" >> /etc/environment 2>/dev/null || true
  grep -q '^HUGGING_FACE_HUB_TOKEN=' /etc/environment 2>/dev/null || echo "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" >> /etc/environment 2>/dev/null || true
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo ">> hf_login: token installed ($_hf_home/token)"
else
  echo ">> hf_login: no HF_TOKEN in env — anonymous (rate-limited) HF pulls"
fi
