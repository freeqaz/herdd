#!/usr/bin/env bash
# Install (or remove) the systemd USER timer that writes a weekly B2 growth
# report: re-indexes the bucket and runs `ckpt_retention.py plan` as a DRY RUN.
#
# **It deletes nothing and cannot be made to from here.** `retention_report.sh`
# has no `--apply` path. Arming a real sweep is an owner ruling against the
# standing "no B2 deletes" invariant — see RETENTION_SWEEP.md.
#
# Modelled on reaper_install.sh: generated units live in ~/.config/systemd/user
# (machine-local, never committed) and paths derive from this checkout.
# Logs: journalctl --user -u b2-growth-report.service
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ "${1:-}" == "--remove" ]]; then
  systemctl --user disable --now b2-growth-report.timer 2>/dev/null || true
  rm -f "$UNIT_DIR"/b2-growth-report.{timer,service}
  systemctl --user daemon-reload
  echo "b2-growth-report timer removed."
  exit 0
fi

mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/b2-growth-report.service" <<EOF
[Unit]
Description=B2 growth report — re-index the bucket, dry-run the retention plan (DELETES NOTHING)

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
# Generous: the inventory pass alone is minutes, and the tool's own
# jobs/*/checkpoints/ listing ran out a 30-min timeout with no output. Both steps
# carry their own timeout and report it, so this is a backstop, not the gate.
TimeoutStartSec=9000
ExecStart=$REPO_ROOT/tools/vast/retention_report.sh --quiet
EOF

cat > "$UNIT_DIR/b2-growth-report.timer" <<EOF
[Unit]
Description=weekly B2 growth report

[Timer]
OnCalendar=Mon 09:00
Persistent=true
RandomizedDelaySec=30min

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now b2-growth-report.timer

loginctl enable-linger "$USER" 2>/dev/null \
  || echo "note: 'loginctl enable-linger $USER' failed — timer only runs while logged in."

echo "b2-growth-report installed (REPORT ONLY): weekly, from $REPO_ROOT"
echo "  status : systemctl --user list-timers b2-growth-report.timer"
echo "  run now: tools/vast/retention_report.sh"
