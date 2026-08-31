#!/usr/bin/env bash
# stage_run.sh — pre-stage everything a training run needs into B2, so the
# instance pulls one bundle and starts immediately (no slow HF downloads, no
# git clone on GPU-billed time).
#
# A "runset" is just a directory that gets copied to b2:<bucket>/runsets/<name>/.
# Put your data + config + a train.sh entrypoint in it. onstart/train.sh runs
# runset/train.sh with OUTPUT_DIR / BASE_DIR / RUNSET_DIR exported.
#
# Usage:
#   stage_run.sh runset  <name> <local_dir>          # data + config + train.sh
#   stage_run.sh model   <b2_subpath> <local_model_dir>   # one-time base-model cache
#   stage_run.sh helper                              # upload b2_sync.sh for instances
#   stage_run.sh ls [name]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# load B2_* from repo .env
ENV="$(cd "$HERE/../.." && pwd)/.env"; [ -f "$ENV" ] && set -a && . "$ENV" && set +a
: "${B2_BUCKET:?set B2_BUCKET}"
bash "$HERE/b2_sync.sh" config >/dev/null
B2="b2:${B2_BUCKET}"

case "${1:-}" in
  runset) name="${2:?name}"; dir="${3:?local dir}"
    [ -f "$dir/train.sh" ] || echo "note: $dir has no train.sh — set TRAIN_CMD at launch instead" >&2
    # sync (not copy): restaging must also DELETE remote files removed locally,
    # or instances keep pulling stale data/config from an earlier stage
    rclone sync --fast-list -P "$dir" "$B2/runsets/${name}"
    echo "staged runset -> $B2/runsets/${name}" ;;
  model)  sub="${2:?b2 subpath}"; dir="${3:?local model dir}"
    rclone copy --fast-list -P "$dir" "$B2/${sub}"
    echo "staged model -> $B2/${sub}  (launch with --env BASE_MODEL_B2=${sub})" ;;
  helper) rclone copyto "$HERE/b2_sync.sh" "$B2/runsets/_bin/b2_sync.sh"; echo "staged b2_sync.sh" ;;
  ls)     rclone tree "$B2/runsets/${2:-}" 2>/dev/null || rclone lsf "$B2/runsets/${2:-}" ;;
  *) grep -E '^#( |$)' "$0" | sed 's/^# \?//'; exit 1 ;;
esac
