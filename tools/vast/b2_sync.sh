#!/usr/bin/env bash
# b2_sync.sh — durable checkpoint/artifact sync to Backblaze B2 via rclone (S3 API).
#
# Works both ON a Vast instance (push checkpoints out as they're written) and on
# THIS box (pull results back). Interruptible instances can die at any moment, so
# training loops MUST push to B2 (or another durable store) — local disk is lost
# when a bid instance is outbid.
#
# Required env (from .env or the instance environment):
#   B2_KEY_ID           Backblaze keyID   (the 004... string — NOT the K004 secret)
#   B2_APPLICATION_KEY  Backblaze appKey  (the K004... secret)
#   B2_BUCKET           bucket name       (example-runs-bucket)
#   B2_S3_ENDPOINT      e.g. https://s3.us-west-004.backblazeb2.com
#   B2_REGION           e.g. us-west-004
#
# Usage:
#   b2_sync.sh config                         # write ~/.config/rclone/rclone.conf
#   b2_sync.sh doctor                         # structural health of that file
#                                             # (0 ok / 2 absent / 4 poisoned;
#                                             #  prints no key material)
#   b2_sync.sh push LOCAL_DIR  REMOTE_SUBPATH # one-shot copy local -> b2
#   b2_sync.sh pull REMOTE_SUBPATH LOCAL_DIR  # one-shot copy b2 -> local
#   b2_sync.sh watch LOCAL_DIR REMOTE_SUBPATH [INTERVAL_SEC]  # loop push (default 300s)
#   b2_sync.sh ls [REMOTE_SUBPATH]            # list objects
set -euo pipefail

REMOTE=b2                      # rclone remote name (US source; write + default read)
RC="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"

# Region-aware read selection (US source vs EU replica). Optional: if the helper
# or EU creds are absent, every path below stays exactly on the US remote.
_HERE_BS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
[ -f "$_HERE_BS/b2_region.sh" ] && . "$_HERE_BS/b2_region.sh" || true

# --- the live-config guard ---------------------------------------------------
# INLINE, not a sourced sibling: the jobd bundle ships this file FLAT
# (`vastlib/jobs/bundle.py`) and a hard `source` of a file that does not ride
# along would refuse on every box.
#
# WHY IT EXISTS: ~/.config/rclone/rclone.conf's [b2] remote is shared by every
# session on the workstation AND by fleetd, which is a systemd user unit with
# its own environment — so no per-remote RCLONE_CONFIG_* override in one shell
# can rescue it. A test that supplies a placeholder endpoint and inherits the
# real $HOME rewrote [b2] with it four times in 47 minutes on 2026-08-22; every
# B2 call then failed with a DNS error that points a reader at the network, and
# fleetd read its own blindness as an empty queue and stopped defending a box
# that held a live ticket. Long form: <upstream-bench>
# archive/runs/2026-08-22-wave3-p0c-9b-thinking-pilot/analysis/FLEETD_BLIND_QUEUE.md
#
# Fails CLOSED: the refusal costs a fixture a temp path, the alternative costs
# every session its B2 access. Never fires on a box (real endpoint) or behind
# an RCLONE_CONFIG override.
# ONE predicate, shared by the write guard and the on-disk detector below: a
# host refused on the way in must also be recognised once it is already on disk.
# RFC 2606 / RFC 6761 names are reserved for documentation and testing and can
# never resolve to a B2 endpoint, so a caller offering one is a fixture.
# Echoes the reason and returns 0 when HOST is reserved; returns 1 otherwise.
b2_reserved_host_reason() {
  local host="$1"
  host="${host#*://}"; host="${host%%/*}"; host="${host%%:*}"
  case "$host" in
    *.invalid|*.test|*.example|*.localhost|invalid|test|example|localhost)
      echo "host '$host' is an RFC 6761 reserved test name"; return 0 ;;
    example.com|example.net|example.org|*.example.com|*.example.net|*.example.org)
      echo "host '$host' is an RFC 2606 reserved documentation name"; return 0 ;;
  esac
  return 1
}

b2_guard_live_rclone_config() {
  local rc="$1" why="" reason
  [ "$rc" = "${HOME:-/nonexistent-home}/.config/rclone/rclone.conf" ] || return 0
  if reason="$(b2_reserved_host_reason "${B2_S3_ENDPOINT:-}")"; then
    why="B2_S3_ENDPOINT $reason"
  fi
  # Second net, for a fixture whose placeholder endpoint looks plausible. Only
  # fires when pytest's env reached the child; the name check is load-bearing.
  if [ -z "$why" ] && [ -n "${PYTEST_CURRENT_TEST:-}" ]; then
    why="running under pytest (PYTEST_CURRENT_TEST=${PYTEST_CURRENT_TEST%% *})"
  fi
  [ -n "$why" ] || return 0
  echo "b2_sync: REFUSING to write the live rclone config $rc" >&2
  echo "b2_sync:   reason: $why" >&2
  echo "b2_sync:   That file's [b2] remote is shared by every session on this box" >&2
  echo "b2_sync:   and by fleetd; overwriting it breaks ALL B2 i/o until it is" >&2
  echo "b2_sync:   repaired by hand, and fleetd then reads the failure as an empty" >&2
  echo "b2_sync:   job queue and stops defending the box." >&2
  echo "b2_sync:   Point RCLONE_CONFIG at a temp path (or set HOME) and re-run." >&2
  exit 3
}

# --- the on-disk detector ----------------------------------------------------
# The write guard refuses NEW damage; it is blind to damage already written.
# That gap is why the 2026-08-22 clobber outlived the guard by two days: a
# poisoned stanza still matches `grep '^\[b2\]'`, so every presence probe read
# it as "already configured", nothing re-wrote it, and every call failed DNS
# far from the cause. Detection is pure text — rclone need not be installed and
# the endpoint need not resolve.

# Echo the `endpoint` value of one stanza. Returns 1 when absent.
b2_conf_endpoint() {
  local rc="$1" want="[$2]"
  [ -f "$rc" ] || return 1
  awk -v s="$want" '
    /^\[/     { inside = ($0 == s); next }
    inside && /^[[:space:]]*endpoint[[:space:]]*=/ {
      sub(/^[^=]*=[[:space:]]*/, ""); print; found = 1; exit
    }
    END { exit(found ? 0 : 1) }' "$rc"
}

# Echo the reason and return 0 when the stanza on disk names a reserved host.
b2_rclone_conf_poison_reason() {
  local rc="$1" remote="${2:-$REMOTE}" ep reason
  ep="$(b2_conf_endpoint "$rc" "$remote")" || return 1
  [ -n "$ep" ] || return 1
  reason="$(b2_reserved_host_reason "$ep")" || return 1
  echo "[$remote] endpoint $reason"
}

# Refuse LOUDLY, naming the repair. Distinct exit 4: "the file on disk is
# broken" is a different operator action from the guard's exit 3 ("your
# argument was refused").
b2_require_healthy_rclone_config() {
  local rc="$1" why
  why="$(b2_rclone_conf_poison_reason "$rc")" || return 0
  echo "b2_sync: the rclone config ON DISK is POISONED — refusing to run" >&2
  echo "b2_sync:   file:   $rc" >&2
  echo "b2_sync:   damage: $why" >&2
  echo "b2_sync:   A reserved name can never resolve, so every B2 call from this" >&2
  echo "b2_sync:   box fails DNS and fleetd reads its own blindness as an empty" >&2
  echo "b2_sync:   job queue. This is damage already on disk, not a bad argument." >&2
  echo "b2_sync: REPAIR — from the repo root, rewrites [b2] from .env:" >&2
  echo "b2_sync:   set -a; . .env; set +a; bash tools/vast/b2_sync.sh config" >&2
  echo "b2_sync: VERIFY:  bash tools/vast/b2_sync.sh doctor" >&2
  exit 4
}

# Structural health report. Prints NO key material — a name, a host and
# present/absent only, so it is safe to paste into a report or a ticket.
# Exit 0 healthy · 2 absent/unconfigured · 4 poisoned.
doctor() {
  local rc="$1" why ep rem rc_status=0
  if [ ! -f "$rc" ]; then
    echo "b2_sync: doctor: NO CONFIG at $rc"
    echo "b2_sync:   repair: set -a; . .env; set +a; bash tools/vast/b2_sync.sh config"
    return 2
  fi
  echo "b2_sync: doctor: $rc"
  for rem in "$REMOTE" b2w b2p b2eu; do
    ep="$(b2_conf_endpoint "$rc" "$rem")" || continue
    if why="$(b2_rclone_conf_poison_reason "$rc" "$rem")"; then
      echo "  [$rem] POISONED — $why"
      [ "$rem" = "$REMOTE" ] && rc_status=4
    else
      echo "  [$rem] endpoint ${ep#*://} — ok"
    fi
  done
  if ! grep -qs "^\[$REMOTE\]" "$rc"; then
    echo "  [$REMOTE] ABSENT"
    echo "b2_sync:   repair: set -a; . .env; set +a; bash tools/vast/b2_sync.sh config"
    return 2
  fi
  if [ "$rc_status" -ne 0 ]; then
    echo "b2_sync:   repair: set -a; . .env; set +a; bash tools/vast/b2_sync.sh config"
    return "$rc_status"
  fi
  echo "b2_sync: doctor: [$REMOTE] looks healthy (structure only — run 'ls' to prove it reads)"
  return 0
}

# --- throughput tuning ------------------------------------------------------
# A bare `rclone copy` uses stock defaults (--transfers 4, single-stream per
# file), so one big checkpoint downloads on a single TCP connection to B2 and
# tops out ~60 MB/s. The flags below split each large file into parallel ranged
# GETs (download) / multipart PUTs (upload) and copy more files at once, which
# saturates a fast link. All are env-overridable.
#   B2_TRANSFERS         files copied in parallel                (default 8)
#   B2_CHECKERS          metadata checkers in parallel           (default 16)
#   B2_MT_STREAMS        parallel ranged GETs per large file     (default 8)
#   B2_MT_CUTOFF         min file size to split into streams     (default 128M)
#   B2_S3_CHUNK          multipart chunk size for uploads        (default 64M)
#   B2_S3_UP_CONCURRENCY parallel chunk uploads per file         (default 8)
B2_TRANSFERS="${B2_TRANSFERS:-8}"
B2_CHECKERS="${B2_CHECKERS:-16}"
B2_MT_STREAMS="${B2_MT_STREAMS:-8}"
B2_MT_CUTOFF="${B2_MT_CUTOFF:-128M}"
B2_S3_CHUNK="${B2_S3_CHUNK:-64M}"
B2_S3_UP_CONCURRENCY="${B2_S3_UP_CONCURRENCY:-8}"

# Common speed flags (both directions).
XFER_OPTS=(
  --fast-list
  --transfers "$B2_TRANSFERS"
  --checkers  "$B2_CHECKERS"
  --multi-thread-streams "$B2_MT_STREAMS"
  --multi-thread-cutoff  "$B2_MT_CUTOFF"
)
# Upload-only: bigger multipart chunks + parallel part uploads.
UP_OPTS=(
  --s3-chunk-size        "$B2_S3_CHUNK"
  --s3-upload-concurrency "$B2_S3_UP_CONCURRENCY"
)

need() { [ -n "${!1:-}" ] || { echo "b2_sync: missing env $1" >&2; exit 2; }; }

ensure_rclone() {
  command -v rclone >/dev/null 2>&1 && return
  echo "b2_sync: installing rclone…" >&2
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
        timeout 180 apt-get install -y -qq rclone >/dev/null 2>&1; }
}

config() {
  need B2_KEY_ID; need B2_APPLICATION_KEY; need B2_S3_ENDPOINT
  b2_guard_live_rclone_config "$RC"
  mkdir -p "$(dirname "$RC")"
  # preserve any other remotes already in rclone.conf; replace only [b2]
  local tmp; tmp="$(mktemp "${RC}.XXXXXX")"
  if [ -f "$RC" ]; then awk -v r="[$REMOTE]" '/^\[/{skip=($0==r)} !skip' "$RC" > "$tmp"; fi
  cat >> "$tmp" <<EOF
[$REMOTE]
type = s3
provider = Other
access_key_id = ${B2_KEY_ID}
secret_access_key = ${B2_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
# key is scoped to one bucket (no create/list-all rights) — don't probe/create it
no_check_bucket = true
EOF
  mv "$tmp" "$RC"
  chmod 600 "$RC"   # contains the B2 secret key
  # Option-1b scoped WRITE remote: when a prefix-restricted write key is present
  # (B2_WRITE_KEY_ID), add a second [b2w] remote and route pushes through it while
  # reads stay on the bucket-wide [b2]. Absent B2_WRITE_* => no [b2w]; wdest()
  # degrades to [b2] and behavior is exactly as before. See CREDENTIAL_LIFECYCLE.md.
  if [ -n "${B2_WRITE_KEY_ID:-}" ] && [ -n "${B2_WRITE_APPLICATION_KEY:-}" ]; then
    local tmpw; tmpw="$(mktemp "${RC}.XXXXXX")"
    awk '/^\[/{skip=($0=="[b2w]")} !skip' "$RC" > "$tmpw"
    cat >> "$tmpw" <<EOF
[b2w]
type = s3
provider = Other
access_key_id = ${B2_WRITE_KEY_ID}
secret_access_key = ${B2_WRITE_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
no_check_bucket = true
EOF
    mv "$tmpw" "$RC"; chmod 600 "$RC"
    echo "b2_sync: wrote scoped write remote 'b2w' (namePrefix-restricted key)"
  fi
  # PUBLISH remote: a training bundle's publish stage writes the named adapter to
  # checkpoints/<RUN_NAME>/, which the jobs/-scoped [b2w] key may NOT touch (one
  # namePrefix per B2 key). Its own key lands here as [b2p]. Absent B2_PUBLISH_*
  # there is no [b2p] and a publish-stage write fails loudly rather than
  # silently landing somewhere else — the submit-time write-scope preflight
  # (jobmeta.b2_write_preflight) is what stops that bundle before it is rented.
  if [ -n "${B2_PUBLISH_KEY_ID:-}" ] && [ -n "${B2_PUBLISH_APPLICATION_KEY:-}" ]; then
    local tmpp; tmpp="$(mktemp "${RC}.XXXXXX")"
    awk '/^\[/{skip=($0=="[b2p]")} !skip' "$RC" > "$tmpp"
    cat >> "$tmpp" <<EOF
[b2p]
type = s3
provider = Other
access_key_id = ${B2_PUBLISH_KEY_ID}
secret_access_key = ${B2_PUBLISH_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
no_check_bucket = true
EOF
    mv "$tmpp" "$RC"; chmod 600 "$RC"
    echo "b2_sync: wrote scoped publish remote 'b2p' (namePrefix-restricted key)"
  fi
  echo "b2_sync: wrote $RC (remote '$REMOTE' -> $B2_S3_ENDPOINT)"
  # add the read-only [b2eu] replica remote when EU creds are configured, so
  # region-aware pulls can reach it. No-op without EU creds.
  if declare -f b2_region_config >/dev/null 2>&1 && [ -n "${B2_KEY_ID_EU:-}" ]; then
    b2_region_config >/dev/null 2>&1 || true
  fi
}

dest() { echo "${REMOTE}:${B2_BUCKET:?set B2_BUCKET}/${1#/}"; }

# Write destination: the scoped [b2w] remote when a write key is configured,
# else the bucket-wide [b2] remote (identical to dest). Every push/watch target
# must be within the write key's namePrefix or B2 rejects it (403) — by design.
wdest() {
  local r="$REMOTE"; [ -n "${B2_WRITE_KEY_ID:-}" ] && r="b2w"
  echo "${r}:${B2_BUCKET:?set B2_BUCKET}/${1#/}"
}

# Read source for a subpath: the region-gated remote:bucket (US source, or the
# EU replica when the subpath is a static asset present in EU). Falls back to the
# plain US dest when the region helper is unavailable.
read_src() {
  local sub="${1#/}"
  if declare -f b2_region_read_remote >/dev/null 2>&1; then
    echo "$(b2_region_read_remote "$sub")/${sub}"
  else
    dest "$sub"
  fi
}

main() {
  local sub="${1:-}"; shift || true
  # `doctor` answers a question about a FILE — it must not install rclone, and
  # it must stay runnable on exactly the broken box its answer is about.
  if [ "$sub" = doctor ]; then doctor "$RC"; return; fi
  ensure_rclone
  # `config` IS the repair, so it is the one subcommand never blocked by the
  # damage it exists to fix; everything else refuses loudly rather than
  # emitting a DNS error that points the reader at the network.
  [ "$sub" = config ] || b2_require_healthy_rclone_config "$RC"
  # need the [b2] section (and [b2w] when a scoped write key is configured but
  # the remote is not yet written — e.g. a re-attach that added B2_WRITE_*).
  if ! grep -qs "^\[$REMOTE\]" "$RC" \
     || { [ -n "${B2_WRITE_KEY_ID:-}" ] && ! grep -qs '^\[b2w\]' "$RC"; } \
     || { [ -n "${B2_PUBLISH_KEY_ID:-}" ] && ! grep -qs '^\[b2p\]' "$RC"; }; then
    config >/dev/null
  fi
  case "$sub" in
    config) config ;;
    push)   rclone copy "${XFER_OPTS[@]}" "${UP_OPTS[@]}" -v "$1" "$(wdest "$2")" ;;
    pull)   rclone copy "${XFER_OPTS[@]}" -v "$(read_src "$1")" "$2" ;;
    ls)     rclone ls "$(dest "${1:-}")" ;;
    watch)
      local local_dir="$1" remote="$2" interval="${3:-300}"
      echo "b2_sync: watching $local_dir -> $(wdest "$remote") every ${interval}s"
      while true; do
        rclone copy "${XFER_OPTS[@]}" "${UP_OPTS[@]}" "$local_dir" "$(wdest "$remote")" 2>/dev/null \
          && echo "b2_sync: pushed $(date -u +%H:%M:%S)" || echo "b2_sync: push failed (retrying)"
        sleep "$interval"
      done ;;
    *) grep -E '^#( |$)' "$0" | sed 's/^# \?//'; exit 1 ;;
  esac
}
# run only when executed, not when sourced (callers may source for read_src/dest)
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then main "$@"; fi
