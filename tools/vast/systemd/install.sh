#!/usr/bin/env bash
# Install this directory's user timers, with the repo root filled in.
#
# The units are templates because a committed unit file cannot carry
# `/home/<user>/...` (CLAUDE.md), and because the two units that already run on
# this box were hand-written into ~/.config and therefore exist nowhere in git —
# which is how `hostfacts ingest` went 17 days without running and nobody could
# see that it was supposed to.
#
#   tools/vast/systemd/install.sh            # install + start
#   tools/vast/systemd/install.sh --status   # what is installed and when it ran
#   tools/vast/systemd/install.sh --uninstall
#
# No sudo: these are `systemctl --user` units.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_ROOT="$(cd "$HERE/../../.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

units() { find "$HERE" -maxdepth 1 -name '*.in' -printf '%f\n' | sed 's/\.in$//'; }

case "${1:-install}" in
--status)
  for u in $(units); do
    printf '%s: ' "$u"
    systemctl --user is-enabled "$u" 2>/dev/null || true
  done
  systemctl --user list-timers --all 2>/dev/null \
    | grep -E "UNIT|$(units | sed 's/\.[^.]*$//' | sort -u | paste -sd'|')" \
    || echo "(no matching timer)"
  exit 0
  ;;
--uninstall)
  for u in $(units); do
    case "$u" in *.timer) systemctl --user disable --now "$u" 2>/dev/null || true ;; esac
    rm -f "$UNIT_DIR/$u"
  done
  systemctl --user daemon-reload
  echo ">> uninstalled from $UNIT_DIR"
  exit 0
  ;;
install) ;;
*) echo "usage: $0 [install|--status|--uninstall]" >&2; exit 2 ;;
esac

[ -x "$DS_ROOT/.venv/bin/python" ] \
  || { echo "!! no venv at $DS_ROOT/.venv — install it first" >&2; exit 1; }

# A unit pinned to a WORKTREE is a unit that stops existing. Worktrees are
# ephemeral by design and `wt.py gc` reclaims them, so an ExecStart into one
# fails silently from then on — the same shape as the fleetd unit that turned
# out to be running a checkout. Install from the primary.
if git -C "$DS_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  common="$(git -C "$DS_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  gitdir="$(git -C "$DS_ROOT" rev-parse --path-format=absolute --git-dir 2>/dev/null || true)"
  if [ -n "$common" ] && [ "$common" != "$gitdir" ]; then
    echo "!! $DS_ROOT is a git WORKTREE, not the primary checkout." >&2
    echo "!! Its files are reclaimable; a unit pointing here breaks on gc." >&2
    echo "!! Run this from ${common%/.git}" >&2
    exit 1
  fi
fi

mkdir -p "$UNIT_DIR"
for u in $(units); do
  sed "s|@DS_ROOT@|$DS_ROOT|g" "$HERE/$u.in" > "$UNIT_DIR/$u"
  echo ">> wrote $UNIT_DIR/$u"
done

systemctl --user daemon-reload
for u in $(units); do
  case "$u" in
  *.timer)
    # enable, THEN restart. `enable --now` is a no-op on a unit that is already
    # active, so on a re-install it would leave the OLD unit file running and
    # report success — the deploy-that-does-nothing shape.
    systemctl --user enable "$u" >/dev/null
    systemctl --user restart "$u"
    echo ">> started $u"
    ;;
  esac
done

echo
echo ">> verify (the timer must appear with a NEXT time):"
systemctl --user list-timers --all \
  | grep -E "NEXT|$(units | sed 's/\.[^.]*$//' | sort -u | paste -sd'|')" || true
