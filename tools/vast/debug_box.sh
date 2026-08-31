#!/usr/bin/env bash
# debug_box.sh — control a training box's post-failure debug-hold by RUN_ID.
#
# When a run FAILS, onstart/train.sh keeps the box SSH-able for
# FAIL_HOLD_MINUTES (default 15) instead of tearing down immediately, so a
# crash can be diagnosed and a fix tried in place (respinning is slow + re-pulls
# the image/base). This helper writes the B2 markers that box polls for — keyed
# on RUN_ID, so you never need to look up the ephemeral instance id.
#
#   tools/vast/debug_box.sh status <RUN_ID>   # show the STATUS marker
#   tools/vast/debug_box.sh stop   <RUN_ID>   # tear the box down NOW (~20s)
#   tools/vast/debug_box.sh extend <RUN_ID>   # add another FAIL_HOLD_MINUTES
#   tools/vast/debug_box.sh ssh    <RUN_ID>   # print the ssh command (best-effort)
#
# stop/extend only act while the box is in the post-failure debug-hold (that's
# when it polls). To kill a box that is still RUNNING, use
# `herdd.py destroy <id>` (the id is printed at launch and by `herdd.py ls`).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV="$(cd "$HERE/../.." && pwd)/.env"; [ -f "$ENV" ] && set -a && . "$ENV" && set +a
: "${B2_BUCKET:?B2_BUCKET not set (source .env)}"
bash "$HERE/b2_sync.sh" config >/dev/null 2>&1 || true

# NB: don't use ${1:?msg} here — the '}' inside "{status|stop|...}" would close
# the parameter expansion early. Explicit checks are unambiguous.
cmd="${1:-}"; RUN="${2:-}"
[ -n "$cmd" ] && [ -n "$RUN" ] || { echo "usage: debug_box.sh {status|stop|extend|ssh} <RUN_ID>" >&2; exit 1; }
CK="b2:$B2_BUCKET/checkpoints/$RUN"

case "$cmd" in
  status)
    echo -n "STATUS: "; rclone cat "$CK/STATUS" 2>/dev/null || echo "(none)"
    ;;
  stop)
    echo "stop $(date -u +%FT%TZ)" | rclone rcat "$CK/STOP"
    echo ">> STOP written for $RUN — box tears down (parks by default) within ~20s IF it"
    echo "   is in the post-failure debug-hold. Still RUNNING? herdd.py stop|destroy <id>"
    ;;
  extend)
    echo "extend $(date -u +%FT%TZ)" | rclone rcat "$CK/EXTEND"
    echo ">> EXTEND written for $RUN — the debug-hold adds another FAIL_HOLD_MINUTES"
    echo "   the next time the box polls (~20s)."
    ;;
  ssh)
    echo ">> find the instance id with: python3 $HERE/herdd.py ls"
    echo ">> then: python3 $HERE/herdd.py ssh <id>"
    ;;
  *)
    echo "usage: debug_box.sh {status|stop|extend|ssh} <RUN_ID>" >&2; exit 1
    ;;
esac
