#!/usr/bin/env bash
# Install (or remove) the systemd USER timer that auto-destroys idle vast.ai
# boxes: runs `herdd reap -y` every 15 minutes, destroying any STOPPED box
# idle past 2h (owner policy 2026-07-21; HERDD_REAP_IDLE_H overrides).
#
# Opt-outs:
#   per-box   : herdd label <ID> keep:<why>   (a `keep` label token)
#   globally  : HERDD_REAP=0 in the repo .env, or `reaper_install.sh --remove`
#
# The generated units live in ~/.config/systemd/user (machine-local, never
# committed); paths are derived from this script's checkout at install time.
# Logs: journalctl --user -u herdd-reaper.service
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
PY=$(command -v python3)

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now herdd-reaper.timer 2>/dev/null || true
  rm -f "$UNIT_DIR"/herdd-reaper.{timer,service}
  systemctl --user daemon-reload
  echo "herdd-reaper timer removed."
  exit 0
fi

mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/herdd-reaper.service" <<EOF
[Unit]
Description=herdd reap — destroy vast.ai boxes idle >2h (label 'keep' opts out)

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=$PY $REPO_ROOT/tools/vast/herdd.py reap -y
EOF

cat > "$UNIT_DIR/herdd-reaper.timer" <<EOF
[Unit]
Description=run herdd reap every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
RandomizedDelaySec=2min

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now herdd-reaper.timer

# Keep the timer firing when no session is open (best-effort; may need sudo).
loginctl enable-linger "$USER" 2>/dev/null \
  || echo "note: 'loginctl enable-linger $USER' failed — timer only runs while logged in."

echo "herdd-reaper installed: reap -y every 15 min from $REPO_ROOT"
echo "  status : systemctl --user list-timers herdd-reaper.timer"
echo "  logs   : journalctl --user -u herdd-reaper.service -n 50"
echo "  remove : $0 --remove"
