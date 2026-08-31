#!/usr/bin/env bash
# b2_transport.sh — the ONE place a box talks to B2 about a merged model dir.
#
#   b2_transport.sh has  <b2-key-prefix>              rc 0 iff the prefix holds a
#                                                     COMPLETE publish (PUSHED.json)
#   b2_transport.sh pull <b2-key-prefix> <dest-dir>
#   b2_transport.sh push <src-dir> <b2-key-prefix>    payload first, PUSHED.json LAST
#
# Hoisted from driftr3-v10-27b-gen/ (byte-identical in three bundles). Those
# copies stay; this one is for new consumers and is parametrized where they
# were hardcoded. Conventions: docs/architecture/MERGED_MODEL_ARTIFACTS.md.
#
# WRITE SCOPE. Every write goes under `checkpoints/` by default, and that is not
# a stylistic choice: a jobs box's B2 keys each carry exactly ONE `namePrefix`,
# so the grant is a property of the key and not of the caller's intent. The
# table is `jobmeta.B2_BOX_GRANTS` — read it there, do not restate it here; two
# copies of a grant model is how one of them goes stale. A merged-model cache is
# neither a job's results nor a new grant, so it lives under the prefix the
# publish key already holds. Writing anywhere else is a 403 discovered at the
# END of a 52 GiB upload.
#
# `MODELKIT_B2_WRITE_PREFIXES` overrides the allowed set (space-separated,
# trailing slash), for a caller whose box was minted a different grant. It can
# only ever move the refusal, never remove it: an empty or unset value falls
# back to `checkpoints/`, because a transport that accepts every prefix is not
# a scope check, it is a comment.
#
# `MODELKIT_B2_WRITE_REMOTE` (default `b2p`) names the rclone remote the push
# uses. The remote and the prefix are the SAME grant seen from two sides, so
# moving one without the other is the 403 again.
#
# NEVER `rclone copyto`. `copyto` HEADs the destination first, and B2 has
# hours-long windows where HeadObject on a not-yet-existing key returns 403, so
# the push silently fails and the box reports success (BOOT_OBSERVABILITY.md
# "Box-side log/result pushes"; memory `b2-copyto-headobject-403`; it has eaten a
# whole publish before). The fallback here is therefore list-based
# `rclone copy --include`, the same idiom the training bundles' publish stage
# uses, and `rclone rcat` for the single small marker object.
#
# TRANSPORT LADDER. b2x first (one static Go binary; parts derived from object
# size, so a 52 GiB dir is not silently capped at rclone's 9-effective-flows),
# rclone second. Every call site falls back, per b2x_boot.sh's contract: a box
# that cannot obtain b2x still works exactly as it did before.
#
# THE MARKER IS WRITTEN LAST and read by `has`. A restore that raced a push
# would otherwise pull a truncated 52 GiB dir, and the merge guard would have to
# be the thing that catches it. It WOULD catch it — but paying a 52 GiB download
# to discover a race we can exclude by ordering is not a design.
set -uo pipefail

MARKER="PUSHED.json"

log() { echo ">> b2t: $*" >&2; }

: "${B2_BUCKET:=}"
: "${MODELKIT_B2_WRITE_PREFIXES:=}"
: "${MODELKIT_B2_WRITE_REMOTE:=b2p}"

WRITE_PREFIXES="${MODELKIT_B2_WRITE_PREFIXES:-}"
[ -n "$WRITE_PREFIXES" ] || WRITE_PREFIXES="checkpoints/"

# --- b2x, if this box can get it --------------------------------------------
B2X=""
_ensure_b2x() {
  [ -n "$B2X" ] && return 0
  [ "${B2X_DISABLE:-0}" = "1" ] && return 1
  local c
  for c in "${B2X_INSTALL_DIR:-/workspace/bin}/b2x" /usr/local/bin/b2x \
           /workspace/eval/bin/b2x; do
    [ -x "$c" ] && "$c" version >/dev/null 2>&1 && { B2X="$c"; return 0; }
  done
  command -v b2x >/dev/null 2>&1 && b2x version >/dev/null 2>&1 && {
    B2X="$(command -v b2x)"; return 0; }
  # The bootstrap shim, wherever jobd put it. Sourcing it defines b2x_ensure,
  # which fetches the binary over the already-configured [b2] rclone remote.
  local s
  for s in "${JOBD_DIR:-/workspace/jobd}/b2x_boot.sh" \
           "${JOBD_DIR:-/workspace/jobd}/onstart/b2x_boot.sh" \
           /workspace/jobd/onstart/b2x_boot.sh; do
    if [ -f "$s" ]; then
      # shellcheck disable=SC1090
      . "$s" && b2x_ensure && [ -n "${B2X:-}" ] && return 0
    fi
  done
  B2X=""
  return 1
}

_need_bucket() {
  [ -n "$B2_BUCKET" ] || { log "B2_BUCKET unset — no B2 transport"; return 1; }
}

case "${1:-}" in

has)
  key="${2:?usage: b2_transport.sh has <prefix>}"
  _need_bucket || exit 1
  if _ensure_b2x; then
    "$B2X" stat "${key%/}/$MARKER" >/dev/null 2>&1 && exit 0
    # b2x present but the object is absent: that is an ANSWER, not a transport
    # failure, so do not re-ask over rclone. (An auth/config failure would have
    # failed `b2x version` above and we would not be here.)
    exit 1
  fi
  rclone lsf "b2:$B2_BUCKET/${key%/}/$MARKER" >/dev/null 2>&1 && exit 0
  exit 1
  ;;

pull)
  key="${2:?usage: b2_transport.sh pull <prefix> <dest>}"
  dest="${3:?usage: b2_transport.sh pull <prefix> <dest>}"
  _need_bucket || exit 1
  mkdir -p "$dest" || exit 1
  # The completion marker is EXCLUDED from the pull. It lives in the same prefix
  # as the payload (so `has` is one stat), but it is transport metadata, not
  # model content — landing it in the merged dir would add a file the merged-dir
  # fingerprint has never seen and turn every restore into an UNEXPECTED-file
  # failure. Same class of bug as jobd's `.complete` marker, which the base
  # gates all `--ignore`.
  if _ensure_b2x && "$B2X" pull --exclude "$MARKER" "${key%/}/" "$dest/"; then
    log "pulled ${key%/}/ -> $dest via b2x"
    exit 0
  fi
  log "b2x pull unavailable/failed — falling back to rclone copy"
  rclone copy --fast-list --transfers 16 --multi-thread-streams 16 \
      --multi-thread-cutoff 64M --exclude "$MARKER" \
      "b2:$B2_BUCKET/${key%/}/" "$dest/" || exit 1
  log "pulled ${key%/}/ -> $dest via rclone"
  exit 0
  ;;

push)
  src="${2:?usage: b2_transport.sh push <src-dir> <prefix>}"
  key="${3:?usage: b2_transport.sh push <src-dir> <prefix>}"
  _need_bucket || exit 1
  [ -d "$src" ] || { log "push: $src is not a directory"; exit 1; }
  in_scope=0
  for p in $WRITE_PREFIXES; do
    case "${key%/}/" in "$p"*) in_scope=1; break ;; esac
  done
  if [ "$in_scope" != 1 ]; then
    log "REFUSING to push to '$key': the box's write grant is namePrefix="
    log "$WRITE_PREFIXES (see jobmeta.B2_BOX_GRANTS). A push outside it is a"
    log "403 discovered at the END of a 52 GiB upload, not at the start."
    exit 2
  fi

  # PAYLOAD FIRST. The marker names the push complete and must not exist while
  # any payload byte is still missing.
  rm -f "$src/$MARKER"
  pushed=0
  if _ensure_b2x && "$B2X" push --exclude "$MARKER" "$src/" "${key%/}/"; then
    log "pushed payload via b2x"
    pushed=1
  else
    log "b2x push unavailable/failed — falling back to list-based rclone copy"
    # LIST-BASED `copy --include`, root-anchored, never `copyto`. Root anchoring
    # keeps a stray subdirectory out of the published prefix.
    inc=()
    while IFS= read -r f; do inc+=(--include "/$f"); done < <(
      cd "$src" && find . -maxdepth 1 -type f -printf '%f\n' | grep -v "^$MARKER\$")
    [ "${#inc[@]}" -gt 0 ] || { log "push: nothing to upload from $src"; exit 1; }
    rclone copy --fast-list "${inc[@]}" "$src" \
      "$MODELKIT_B2_WRITE_REMOTE:$B2_BUCKET/${key%/}/" \
      && { log "pushed payload via rclone"; pushed=1; }
  fi
  [ "$pushed" = 1 ] || { log "push FAILED — marker NOT written"; exit 1; }

  # READ BACK before claiming it. A push that cannot be listed is a push that
  # did not happen, and this is the cheap moment to find out.
  n_local=$(find "$src" -maxdepth 1 -type f ! -name "$MARKER" | wc -l)
  n_remote=$(rclone lsf --files-only "b2:$B2_BUCKET/${key%/}/" 2>/dev/null \
             | grep -vc "^$MARKER\$")
  log "read-back: $n_remote objects at ${key%/}/ vs $n_local local files"
  [ "$n_remote" -ge "$n_local" ] || { log "read-back SHORT — marker NOT written"; exit 1; }

  # MARKER LAST, via rcat (a streaming PUT that never HEADs the destination).
  printf '%s\n' "{\"complete\": true, \"files\": $n_local, \"ts_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    | rclone rcat "$MODELKIT_B2_WRITE_REMOTE:$B2_BUCKET/${key%/}/$MARKER" || {
      log "marker write FAILED — the prefix stays UNPUBLISHED, which is correct:"
      log "a payload nobody can see beats a marker that lies."; exit 1; }
  log "PUBLISHED ${key%/}/ ($n_local files + $MARKER)"
  exit 0
  ;;

*)
  grep -E '^#( |$)' "$0" | sed 's/^# \?//'
  exit 2
  ;;
esac
