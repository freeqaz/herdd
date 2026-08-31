#!/usr/bin/env bash
# onstart/serve_boot.sh — the tiny boot-pull wire for launch_serve.sh.
#
# WHY: the full onstart/serve_vllm.sh grew past Vast's 16 KiB inline-onstart cap
# (the heavily-documented multi-GPU/HAProxy/farm serve path — the STRIPPED wire is
# ~16.5 KiB), so a launch 400s or silently truncates. Mirrors onstart/train_boot.sh
# (and jobd_boot.sh): the LAPTOP stages the real server to
# b2:<bucket>/serve/<SERVE_ID>/serve_main.sh (per-SERVE, so concurrent launches of
# different serve_vllm.sh versions never race), and only THIS ~2 KiB bootstrap
# ships on the wire. It configures B2, pulls serve_main.sh, and execs it (replacing
# this process, so serve_vllm.sh runs EXACTLY as it did when it was the onstart
# directly — serve_vllm.sh's tail is `exec vllm`/`exec haproxy`, a foreground
# supervised main). A park/resume re-runs THIS wire and its SERVE_ID -> pulls the
# SAME per-SERVE serve_main.sh.
#
# Required env (shipped by launch_serve's --env list when the B2 marker is on):
# SERVE_ID, B2_BUCKET, B2_KEY_ID, B2_APPLICATION_KEY, B2_S3_ENDPOINT, B2_REGION.
# Optional Option-1b scoped write pair (B2_WRITE_KEY_ID/B2_WRITE_APPLICATION_KEY):
# the pull READS serve_main.sh via [b2]; a FAILED marker WRITE routes via [b2w]
# when present (the RO [b2] key can't write serve/). Test seams (no-ops in prod):
#   SERVE_BOOT_WS          retarget /workspace (serve_main.sh dir + log)
#   SERVE_BOOT_RETRY_SLEEP retry backoff seconds (default 10)
#   SERVE_BOOT_NO_EXEC=1   skip the final exec (write a marker, exit 0)
set -uo pipefail
WS="${SERVE_BOOT_WS:-/workspace}"
mkdir -p "$WS"
exec > >(tee -a "$WS/onstart.log") 2>&1
echo "=== serve_boot $(date -u +%FT%TZ) SERVE_ID=${SERVE_ID:-?} ==="

# creds must be present (container env from --env) — without them there is no
# transport and no way to even report failure.
if [ -z "${SERVE_ID:-}" ] || [ -z "${B2_BUCKET:-}" ] || [ -z "${B2_KEY_ID:-}" ] || \
   [ -z "${B2_APPLICATION_KEY:-}" ] || [ -z "${B2_S3_ENDPOINT:-}" ]; then
  echo "!! serve_boot: SERVE_ID/B2_* env not set — cannot pull the server; exiting"
  exit 1
fi

# rclone (idempotent install; same fallback chain as onstart/serve_vllm.sh — the
# pre-t214 images and upstream bases ship no rclone). Without it there is no
# B2, so we cannot
# even write the FAILED marker; echo loud and exit.
if ! command -v rclone >/dev/null 2>&1; then
  echo ">> serve_boot: installing rclone"
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
if ! command -v rclone >/dev/null 2>&1; then
  echo "!! serve_boot: rclone install failed — no B2 transport, cannot pull the server"
  exit 1
fi

# b2: read remote — secret lands in the on-box conf, never in this wire. Also a
# [b2w] write remote when a scoped pair was shipped (so the FAILED marker below
# can write serve/ even though [b2] is read-only). serve_vllm.sh REWRITES both
# from env on every run, so a stale conf here is harmless.
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
if [ -n "${B2_WRITE_KEY_ID:-}" ] && [ -n "${B2_WRITE_APPLICATION_KEY:-}" ]; then
  cat >> "$RCONF" <<EOF
[b2w]
type = s3
provider = Other
access_key_id = ${B2_WRITE_KEY_ID}
secret_access_key = ${B2_WRITE_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
no_check_bucket = true
EOF
  B2W="b2w"
else
  B2W="b2"
fi
chmod 600 "$RCONF"
B2="b2:${B2_BUCKET}"

# pull the full server (per-SERVE object). Bounded retry — box networking can lag
# at boot; a transient blip must not leave the box silent.
SERVE_MAIN="$WS/serve_main.sh"
_sleep="${SERVE_BOOT_RETRY_SLEEP:-10}"
_ok=0
for _try in 1 2 3 4 5; do
  if rclone copyto "$B2/serve/${SERVE_ID}/serve_main.sh" "$SERVE_MAIN" 2>>"$WS/onstart.log" \
       && [ -s "$SERVE_MAIN" ]; then
    _ok=1; break
  fi
  echo ">> serve_boot: server pull attempt ${_try}/5 failed — retrying in ${_sleep}s"
  sleep "$_sleep"
done

if [ "$_ok" != 1 ]; then
  # a silent dead box is exactly what this replaces: write a terminal FAILED
  # SERVE_STATUS (same `FAILED <token>` shape serve_vllm.sh's status() writes;
  # serve_ready.sh exits 3 on FAILED) so a watcher stops waiting instead of hanging.
  echo "!! serve_boot: could not pull serve/${SERVE_ID}/serve_main.sh after 5 tries — FAILED"
  echo "FAILED serve_boot_pull $(date -u +%FT%TZ)" \
    | rclone rcat "${B2W}:${B2_BUCKET}/serve/${SERVE_ID}/SERVE_STATUS" 2>/dev/null || true
  exit 1
fi
chmod +x "$SERVE_MAIN" 2>/dev/null || true
echo ">> serve_boot: pulled server ($(wc -c <"$SERVE_MAIN" 2>/dev/null) B) — handing off"

if [ "${SERVE_BOOT_NO_EXEC:-0}" = "1" ]; then
  echo ">> serve_boot: SERVE_BOOT_NO_EXEC=1 — not exec'ing (test seam)"
  : > "$WS/.serve_boot_would_exec"
  exit 0
fi
exec bash "$SERVE_MAIN"
