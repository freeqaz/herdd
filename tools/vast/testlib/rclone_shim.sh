#!/usr/bin/env bash
# rclone_shim.sh — a fake `rclone` that maps `b2:<bucket>/<key>` onto files under
# $FAKE_BUCKET on local disk. THE local B2: no network, no credentials, no cost.
# Shared source of truth for the local job harnesses — install it on PATH as
# `rclone` (see test_jobd.py `_make_bucket`, rehearse.sh, and joblocal.py's
# LOCAL GPU LANE, which points BOTH executors — jobd.sh and jobmeta's
# `_default_runner` — at this one implementation). Ops handled: cat, rcat,
# copyto, copy, sync, lsf, deletefile, listremotes (extend here, NEVER fork a
# second copy — the local lane's whole correctness argument is that laptop side
# and box side share this file).
set -u
B="$FAKE_BUCKET"
map() {  # strip b2*:BUCKET/ -> $B/... ; leave local paths alone
  # The scoped remotes ([b2w] jobs/, [b2p] checkpoints/) map onto the SAME fake
  # bucket: the shim models the transport, not the entitlement. What a real box's
  # namePrefix scoping would reject is checked statically at submit instead
  # (jobmeta.b2_write_preflight), so a rehearsal exercising the publish stage
  # writes where the box would rather than inventing a directory named `b2p:…`.
  case "$1" in
    b2:*/*|b2w:*/*|b2p:*/*|b2eu:*/*) echo "$B/${1#*:*/}" ;;
    b2:*|b2w:*|b2p:*|b2eu:*)         echo "$B/" ;;
    *)                               echo "$1" ;;
  esac
}
op="$1"; shift
case "$op" in
  listremotes) echo "b2:"; exit 0 ;;
  rcat)
    # ATOMIC: write a sibling temp then rename. An in-place `cat > "$p"` leaves a
    # window where a concurrent reader sees a ZERO-BYTE object, which real B2
    # never exposes (an object appears whole or not at all). That window was
    # root-caused 2026-07-30 as the `JSONDecodeError: line 1 column 1` flake in
    # test_jobd.py / test_b2_mint_key.py — blamed on machine load, actually a
    # missing atomic write. The local GPU lane runs CONCURRENT jobs against this
    # shim (jobd heartbeats + events + checkpoint syncs from N runners), so the
    # window is hit routinely, not just under test load.
    p="$(map "$1")"; d="$(dirname "$p")"; mkdir -p "$d"
    t="$(mktemp "$d/.rclone_shim.XXXXXX")" || exit 1
    cat > "$t" && mv -f "$t" "$p" || { rm -f "$t"; exit 1; }
    exit 0 ;;
  cat)  p="$(map "$1")"; [ -f "$p" ] && { cat "$p"; exit 0; } || exit 1 ;;
  hashsum)
    # `hashsum <algo> <remote>` — the submit-time staged-asset staleness
    # preflight (jobmeta._b2_object_fingerprint) reads it. Against the real
    # remote this is a METADATA read (no download): our `b2:` is rclone
    # `type = s3` and serves md5/ETag, while a native-b2 remote serves sha1 —
    # which is why the preflight NEGOTIATES the algo instead of assuming one.
    # This shim answers all three. Output shape is rclone's: "<hash>  <name>".
    # An unknown algo exits non-zero, which is what drives the preflight's size
    # fallback — the same degradation a multipart object (whose ETag is a hash
    # of part hashes, so rclone reports no usable hash) produces.
    algo="$1"; shift; p="$(map "$1")"; [ -f "$p" ] || exit 1
    case "$(echo "$algo" | tr 'A-Z' 'a-z' | tr -d '-')" in
      sha1)   h="$(sha1sum "$p" | cut -d' ' -f1)" ;;
      sha256) h="$(sha256sum "$p" | cut -d' ' -f1)" ;;
      md5)    h="$(md5sum "$p" | cut -d' ' -f1)" ;;
      *) echo "shim: unsupported hash $algo" >&2; exit 2 ;;
    esac
    echo "$h  $(basename "$p")"; exit 0 ;;
  size)
    # `size [--json] <remote>` — used by jobmeta.measure_asset_bytes (the submit
    # disk advisory) and by the staleness preflight's no-hash fallback. Handles
    # both a single object and a prefix.
    json=0; target=""
    for a in "$@"; do case "$a" in --json) json=1 ;; --*) ;; *) target="$a" ;; esac; done
    p="$(map "$target")"
    if [ -f "$p" ]; then
      n=1; b="$(wc -c < "$p" | tr -d ' ')"
    elif [ -d "$p" ]; then
      n="$(find "$p" -type f 2>/dev/null | wc -l | tr -d ' ')"
      b="$(find "$p" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    else
      exit 1
    fi
    if [ "$json" = 1 ]; then echo "{\"count\":$n,\"bytes\":$b}"
    else echo "Total objects: $n"; echo "Total size: $b Bytes"; fi
    exit 0 ;;
  copyto)
    src="$1"; dst="$2"; s="$(map "$src")"; d="$(map "$dst")"
    [ -f "$s" ] || exit 1; mkdir -p "$(dirname "$d")"; cp "$s" "$d"; exit 0 ;;
  deletefile)
    # `job cancel` / `job retarget` remove a queue ticket with this. Real rclone
    # exits non-zero when the object is absent; match that so the CLI's
    # idempotency logic sees the same shape it does against B2.
    p="$(map "$1")"; [ -f "$p" ] || exit 1; rm -f "$p"; exit 0 ;;
  copy|sync)
    # copy|sync SRC/ DST [flags...] — value-carrying flags consume their next arg.
    # `sync` mirrors src->dst (deleting extraneous dst files); jobd's asset_pull +
    # the shipped eval-template use mode: sync. In the harness the dest starts
    # fresh, so copy-into is a faithful stand-in (nothing extraneous to prune).
    #
    # --include and --min-age are HONORED (they used to be silently dropped along
    # with every other flag, which quietly made every local harness OVER-COLLECT):
    #   --include  jobd publishes results with `copy --include <glob>… $run
    #              b2:…/results/`. Dropping the filter copied the ENTIRE job
    #              workdir into results/, so rehearse.sh's "every results glob
    #              matched something" assertion could pass on files no glob
    #              selected, and the local GPU lane would have shipped GB of
    #              intermediates into its bucket on every checkpoint interval.
    #   --min-age  the mid-run checkpoint sync passes it to skip files that are
    #              still being written. Dropping it let a half-written checkpoint
    #              reach the bucket and then be pulled BACK over a good one on
    #              resume. jobd already computes its own honest `files=` count for
    #              the event; this makes the bytes agree with it.
    #   --exclude  jobd's asset pull passes it for an asset's `receipt:` marker,
    #              and the b2_transport/serve-gen push+pull pair for PUSHED.json.
    #              Ignoring it made a harness UNDER-test the very exclusion it
    #              was there to prove: the marker landed anyway and only the
    #              belt-and-braces `rm` kept the tree clean.
    # The tuning flags stay ignored (no caller relies on them).
    args=(); inc=(); exc=(); minage=""; skip=0; prev=""
    for a in "$@"; do
      if [ "$skip" = 1 ]; then
        case "$prev" in
          --include) inc+=("$a") ;;
          --exclude) exc+=("$a") ;;
          --min-age) minage="$a" ;;
        esac
        skip=0; continue
      fi
      case "$a" in
        --min-age|--include|--exclude|--transfers|--checkers|--retries|\
        --multi-thread-streams|--multi-thread-cutoff|--stats|--stats-log-level)
          prev="$a"; skip=1 ;;
        --*) ;;
        *) args+=("$a") ;;
      esac
    done
    s="$(map "${args[0]}")"; d="$(map "${args[1]}")"
    mkdir -p "$d"; [ -d "$s" ] || exit 0
    # --min-age "45s"/"1m"/"2h"/"1d"/bare-seconds -> a mtime cutoff epoch.
    cutoff=""
    if [ -n "$minage" ]; then
      n="${minage%[smhd]}"; u="${minage#$n}"
      case "$u" in m) n=$(( n * 60 )) ;; h) n=$(( n * 3600 )) ;; d) n=$(( n * 86400 )) ;; esac
      cutoff=$(( $(date +%s) - n ))
    fi
    # rclone filter semantics, close enough for every caller we have: a leading
    # `/` anchors the pattern at the transfer root; without one it may also match
    # at any depth. `**` behaves as `*` under bash [[ ]] (which does not treat
    # `/` specially), which is what these globs mean.
    _match() {   # _match <rel> <pattern...> — rclone anchoring, close enough
      local rel="$1" p q; shift
      for p in "$@"; do
        q="${p#/}"
        [[ "$rel" == $q ]] && return 0
        [ "$q" = "$p" ] && [[ "$rel" == */$q ]] && return 0
      done
      return 1
    }
    keep() {
      [ "${#inc[@]}" -eq 0 ] && return 0
      _match "$1" "${inc[@]}"
    }
    # --exclude WINS over --include, as in rclone: the filters are one ordered
    # rule list and an exclusion is not overridable by a later include.
    drop() {
      [ "${#exc[@]}" -eq 0 ] && return 1
      _match "$1" "${exc[@]}"
    }
    while IFS= read -r rel; do
      [ -n "$rel" ] || continue
      drop "$rel" && continue
      keep "$rel" || continue
      if [ -n "$cutoff" ]; then
        mt="$(stat -c %Y "$s/$rel" 2>/dev/null || echo 0)"
        [ "$mt" -le "$cutoff" ] 2>/dev/null || continue
      fi
      mkdir -p "$d/$(dirname "$rel")" 2>/dev/null
      cp -p "$s/$rel" "$d/$rel" 2>/dev/null
    done < <(cd "$s" && find . -type f -printf '%P\n' 2>/dev/null)
    exit 0 ;;
  lsf)
    rec=0; dirs=0; target=""
    for a in "$@"; do case "$a" in
      -R) rec=1 ;; --dirs-only) dirs=1 ;; --*) ;; *) target="$a" ;;
    esac; done
    p="$(map "$target")"
    case "$target" in
      */) # directory listing
        [ -d "$p" ] || exit 0
        if [ "$rec" = 1 ]; then
          ( cd "$p" && find . -mindepth 1 $( [ "$dirs" = 1 ] && echo -type d ) \
              -printf '%P%y\n' 2>/dev/null | sed 's/f$//; s/d$/\//' )
        else
          # `find`, not `"$p"/*`: a glob silently SKIPS DOTFILES, and a merged
          # model dir's merge marker is one (`.v4_relayout_ok.json`,
          # `.g4_merge_ok.json`). Real rclone lists them, and
          # b2_transport.sh's push COUNTS this listing in its read-back to
          # decide whether to write PUSHED.json — so the glob made every
          # rehearsed publish report a short push that had in fact landed every
          # byte, and no amount of rehearsing could have caught a real one. The
          # -R branch above already used find; this makes the two agree. Sorted
          # to keep the glob's output order.
          ( cd "$p" && find . -mindepth 1 -maxdepth 1 \
              $( [ "$dirs" = 1 ] && echo -type d ) \
              -printf '%P%y\n' 2>/dev/null | sed 's/f$//; s/d$/\//' | sort )
        fi ;;
      *) [ -f "$p" ] && basename "$p" || exit 1 ;;
    esac
    exit 0 ;;
  *) echo "shim: unhandled op $op" >&2; exit 2 ;;
esac
