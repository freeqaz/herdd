#!/usr/bin/env bash
# onstart/train_boot.sh — the tiny boot-pull wire for `herdd train`.
#
# WHY: the full onstart/train.sh grew past Vast's 16 KiB inline-onstart cap
# (box-side handoff/spot guards pushed the STRIPPED wire to ~19.6 KiB), so a
# launch 400s or silently truncates. Mirrors onstart/jobd_boot.sh: the LAPTOP
# stages the real trainer to b2:<bucket>/runs/<RUN_ID>/train_main.sh (per-RUN,
# so concurrent launches of different train.sh versions never race), and only
# THIS ~2 KiB bootstrap ships on the wire. It configures B2, pulls the trainer,
# and execs it (replacing this process, so train.sh runs exactly as it did when
# it was the onstart directly). A relaunch/understudy reuses the captured wire
# (this file) and its RUN_ID -> pulls the SAME per-RUN train_main.sh.
#
# Required env (shipped by cmd_train's env_list): RUN_ID, B2_BUCKET, B2_KEY_ID,
# B2_APPLICATION_KEY, B2_S3_ENDPOINT, B2_REGION. Test seams (no-ops in prod):
#   TRAIN_BOOT_WS         retarget /workspace (train_main.sh dir + log)
#   TRAIN_BOOT_RETRY_SLEEP retry backoff seconds (default 10)
#   TRAIN_BOOT_NO_EXEC=1  skip the final exec (write a marker, exit 0)
set -uo pipefail
WS="${TRAIN_BOOT_WS:-/workspace}"
mkdir -p "$WS"
exec > >(tee -a "$WS/onstart.log") 2>&1
echo "=== train_boot $(date -u +%FT%TZ) RUN_ID=${RUN_ID:-?} ==="

# creds must be present (container env from --env) — without them there is no
# transport and no way to even report failure.
if [ -z "${RUN_ID:-}" ] || [ -z "${B2_BUCKET:-}" ] || [ -z "${B2_KEY_ID:-}" ] || \
   [ -z "${B2_APPLICATION_KEY:-}" ] || [ -z "${B2_S3_ENDPOINT:-}" ]; then
  echo "!! train_boot: RUN_ID/B2_* env not set — cannot pull the trainer; exiting"
  exit 1
fi

# rclone (idempotent install; same fallback chain as onstart/train.sh — the
# pre-t214 images and upstream bases ship no rclone). Without it there is no
# B2, so we cannot
# even write the FAILED marker; echo loud and exit.
if ! command -v rclone >/dev/null 2>&1; then
  echo ">> train_boot: installing rclone"
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
  echo "!! train_boot: rclone install failed — no B2 transport, cannot pull the trainer"
  exit 1
fi

# b2: remote — EXACT same shape as onstart/train.sh (secret lands in the on-box
# conf, never in this wire). train.sh's preamble configures only [b2]; do the same.
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

# pull the full trainer (per-RUN object). Bounded retry — box networking can lag
# at boot; a transient blip must not leave the box silent.
TRAIN_MAIN="$WS/train_main.sh"
_sleep="${TRAIN_BOOT_RETRY_SLEEP:-10}"
_ok=0
for _try in 1 2 3 4 5; do
  if rclone copyto "$B2/runs/${RUN_ID}/train_main.sh" "$TRAIN_MAIN" 2>>"$WS/onstart.log" \
       && [ -s "$TRAIN_MAIN" ]; then
    _ok=1; break
  fi
  echo ">> train_boot: trainer pull attempt ${_try}/5 failed — retrying in ${_sleep}s"
  sleep "$_sleep"
done

if [ "$_ok" != 1 ]; then
  # a silent dead box is exactly what this replaces: write a terminal FAILED
  # STATUS (same `<TOKEN> <utc-ts>` shape train.sh's status() writes; babysit/
  # fold glob-match FAILED*) so a watcher tears the box down instead of hanging.
  echo "!! train_boot: could not pull runs/${RUN_ID}/train_main.sh after 5 tries — FAILED"
  echo "FAILED train_boot_pull $(date -u +%FT%TZ)" \
    | rclone rcat "$B2/checkpoints/${RUN_ID}/STATUS" 2>/dev/null || true
  exit 1
fi
chmod +x "$TRAIN_MAIN" 2>/dev/null || true
echo ">> train_boot: pulled trainer ($(wc -c <"$TRAIN_MAIN" 2>/dev/null) B) — handing off"

if [ "${TRAIN_BOOT_NO_EXEC:-0}" = "1" ]; then
  echo ">> train_boot: TRAIN_BOOT_NO_EXEC=1 — not exec'ing (test seam)"
  : > "$WS/.train_boot_would_exec"
  exit 0
fi
exec bash "$TRAIN_MAIN"
