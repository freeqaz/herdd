#!/usr/bin/env bash
# preempt_trap.sh — box-side preemption trap for onstart/train.sh (SPOT_DESIGN §3.3).
#
# Externalized from train.sh PURELY for the 16 KiB inline-onstart wire cap (the same
# reason resume_pull.sh is a companion). train.sh sources this once, right after the
# b2: rclone remote is configured, so _preempt_trap closes over the caller's
# emit_event / $CKPT_DIR / $B2 / $RUN_ID (resolved at fire time, dynamic scope).
#
# BEST-EFFORT BONUS on top of the 180s periodic push (the PRIMARY defense): vast
# preemption is abrupt with no documented grace window, so this only narrows the loss
# window WHEN vast happens to deliver SIGTERM/SIGINT (docker stop). On that signal it:
#   1. emits ONE non-terminal 'preempted' event (a preempted run is NOT failed);
#   2. kicks ONE bounded final checkpoint flush — no --min-age (grab the newest bytes;
#      torn files are tolerated: resume_pull pulls newest-2 dirs and HF validates),
#      trailing-slash dest (rides out B2's flaky HEAD dest-check), --exclude STATUS,
#      wrapped in `timeout 45` so the death path can never hang;
#   3. exits.
# It is NOT armed on EXIT (normal completion must never fire it). Two guards keep it
# from mislabelling a run that already finished:
#   - $RC set  -> TRAIN_CMD has RETURNED (we're in the post-train artifact push /
#     eval grace-wait / teardown window). A preemption here is NOT data loss — the
#     work is done. Emit the REAL terminal (done/failed by RC) and exit RC, so
#     supervise stops instead of relaunching and re-training a finished run. This
#     mirrors run_durable.sh's `[ -n "${RC:-}" ]` guard; .run_terminal alone is
#     insufficient because it's written ~30m late (after the grace-wait).
#   - .run_terminal present -> a terminal was already emitted (e.g. MAX_HOURS
#     watchdog); no-op.
# Only when neither holds (genuinely mid-training) do we emit non-terminal
# 'preempted' + one bounded final flush. `trap -` disarms first so a second signal
# can't re-enter the handler.
# BEGIN preempt-local-save
# Ask the TRAINER for a fresh checkpoint on LOCAL DISK, before the slow B2 flush.
#
# The flush below can only ever push bytes that already exist. Killed 19 minutes
# into a 20-minute SAVE_STEPS window, those bytes are 19 minutes stale and no
# amount of flushing invents the missing progress. A local save is ~0.5-11s at
# our shape; ~1 GB to B2 is far slower — so we ask first and flush second. And
# since instance->instance salvage landed (tools/vast/salvage.py), a checkpoint
# that only reaches local disk is still recoverable, which is what makes writing
# one worth the seconds it costs.
#
# Signalling is by EXPLICIT PID, read from files each rank wrote at startup
# (onstart/preempt_save.py). Never `pkill -USR1 -f <pattern>` — that pattern is
# present in this shell's own argv and would match the trap itself — and never
# `kill -USR1 -<pgid>`, which would also hit unrelated children for which
# SIGUSR1's default action is TERMINATE.
#
# Waiting is on .preempt_save_complete, which rank 0 writes ONLY after every rank
# reported its own save finished. A checkpoint DIRECTORY proves nothing here:
# transformers 5.13.0 writes in place, and under DDP a non-zero rank creates the
# dir before rank 0 writes a byte. We snapshot the markers that already exist
# BEFORE signalling and wait for a NEW one, so a marker left by an earlier
# preemption in the same run cannot be mistaken for this save's completion.
#
# Every path returns 0: this runs on the death path, where a non-zero return or
# a hang costs more than the checkpoint is worth. Knobs: PREEMPT_SAVE_ENABLED=0
# to disable, PREEMPT_SAVE_WAIT_S to change the ceiling.
#
# EVERY exit reports (_preempt_save_report). See its comment: a safety net whose
# degraded mode looks exactly like its working mode is not a safety net, and this
# one degraded silently for its entire life.

# Report the outcome of a preempt-save attempt LOUDLY, and — where the lane gave
# us an emitter — as a B2 EVENT that makes its ABSENCE detectable off-box.
#
# WHY: every early return below used to be a `..` log line on a dying box plus
# `return 0`. So when `preempt_save.py` turned out to be in NO shipping manifest
# at all (eval-env/bake.sh's companion list, herdd's `_job_attach_files`,
# train.sh's companion pull — all three missing it; fixed 2026-08-06) the only
# trace anywhere was one line in an onstart log nobody reads, and the feature
# was dead from the day it landed without a single alarm. The whole point of a
# safety net is that you find out when it is not there.
#
# `complete` is the only non-`!!` success. `disabled` is a deliberate operator
# choice, so it is informational — but it still REPORTS, because "why did
# nothing happen?" is exactly the question this answers. Everything else is a
# DEFECT that wants a human.
#
# The emitter is pluggable because the two lanes emit differently (train.sh's
# `emit_event <kind> <msg>` vs jobd's per-job jobmeta emit). Undefined => the
# echo still happens. Never load-bearing: this runs on the death path, so it
# swallows everything and always returns 0.
_preempt_save_report() {
  local outcome="$1" detail="${2:-}"
  case "$outcome" in
    complete|disabled) echo ".. preempt-save: ${outcome}${detail:+ — $detail}" ;;
    *)                 echo "!! preempt-save: ${outcome}${detail:+ — $detail}" ;;
  esac
  if declare -F _preempt_save_emit >/dev/null 2>&1; then
    _preempt_save_emit "$outcome" "$detail" || true
  fi
  return 0
}

_preempt_local_save() {
  local piddir="${PREEMPT_SAVE_PIDDIR:-/workspace/.preempt_save_pids}"
  local wait_s="${PREEMPT_SAVE_WAIT_S:-30}"
  local ckpt="${CKPT_DIR:-}"
  [ "${PREEMPT_SAVE_ENABLED:-1}" = "1" ] || {
    _preempt_save_report disabled "PREEMPT_SAVE_ENABLED=0"; return 0; }
  [ -n "$ckpt" ] && [ -d "$ckpt" ] || {
    _preempt_save_report no_ckptdir "CKPT_DIR unset or absent (${ckpt:-<unset>})"
    return 0; }
  # THE failure that hid the missing module for this feature's whole life. Name
  # the actual causes, in the order worth checking, so the log line is a lead
  # and not a shrug.
  [ -d "$piddir" ] || {
    _preempt_save_report no_piddir \
      "no $piddir: the trainer never armed the handler. Check the run log for \
'preempt-save unavailable (ModuleNotFoundError...)' — preempt_save.py must be \
staged where the trainer can import it (eval-env companions for the run lane, \
the jobd bundle for the jobs lane)"
    return 0; }

  local before
  before=$(find "$ckpt" -maxdepth 4 -name .preempt_save_complete 2>/dev/null | sort)

  local n=0 pid f
  for f in "$piddir"/*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f" 2>/dev/null) || continue
    case "$pid" in ''|*[!0-9]*) continue ;; esac
    kill -0 "$pid" 2>/dev/null || continue
    kill -USR1 "$pid" 2>/dev/null && n=$((n+1))
  done
  [ "$n" -gt 0 ] || {
    _preempt_save_report no_live_pid \
      "pid dir $piddir has no live trainer pid (stale pids from a prior attempt?)"
    return 0; }
  echo ".. preempt-save: asked $n rank(s) for an immediate checkpoint (<=${wait_s}s)"

  local i=0 now fresh
  while [ "$i" -lt "$wait_s" ]; do
    now=$(find "$ckpt" -maxdepth 4 -name .preempt_save_complete 2>/dev/null | sort)
    # ADDED markers only, never "the list changed". `save_total_limit` rotation
    # runs at the end of every _save_checkpoint and can DELETE a checkpoint that
    # carried a marker from a previous process's forced save (visible after a
    # resume) — that also changes the list, and reporting it as success would
    # flush stale bytes under a green flag.
    fresh=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$now"))
    if [ -n "$fresh" ]; then
      _preempt_save_report complete \
        "a fresh all-rank-COMPLETE checkpoint landed locally after ${i}s ($(printf '%s' "$fresh" | tr '\n' ' '))"
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  _preempt_save_report timeout \
    "no COMPLETE checkpoint inside ${wait_s}s ($n rank(s) signalled) — flushing \
what exists. Any checkpoint dir without .preempt_save_complete may be TORN; do \
not resume from it without checking trainer_state.json"
  return 0
}
# END preempt-local-save

_preempt_trap() {
  trap - TERM INT
  [ -f /workspace/.run_terminal ] && exit 0
  if [ -n "${RC:-}" ]; then
    case "${RC}" in 0) _emit_terminal done ;; *) _emit_terminal failed "rc=${RC}" ;; esac
    echo "rc=${RC} $(date -u +%FT%TZ)" > /workspace/.run_terminal
    exit "${RC}"
  fi
  emit_event preempted || true
  _preempt_local_save || true
  # Two-writer fence (HANDOFF_DESIGN §4): a superseded box — one a newer epoch was
  # PROMOTED over — must not push its stale bytes onto the prefix the promoted
  # understudy now writes. Skip the BYTES only, never the events. Reuses the
  # caller's _handoff_epoch_stale (train.sh, promoted-keyed: stale only AFTER a
  # newer promotion, so the fence-park final flush — which happens pre-promotion —
  # still goes through). Fail-safe: function undefined (standalone source) or
  # HANDOFF_EPOCH unset => flush exactly as before.
  if declare -F _handoff_epoch_stale >/dev/null 2>&1 && _handoff_epoch_stale; then
    echo "!! preempt trap: handoff epoch ${HANDOFF_EPOCH:-?} superseded — skipping final checkpoint flush (bytes only; events still emitted)"
  else
    # DURABILITY, not speed. A multi-GB flush at stock rclone concurrency does
    # not finish inside 45 s on a per-flow-shaped host, so eviction was silently
    # losing state — this is the whole reason the transport work covers uploads.
    # b2x's --deadline additionally orders NEWEST FIRST and completes each object
    # before starting the next, so a budget that still runs out leaves the newest
    # checkpoints WHOLE on B2 rather than an arbitrary torn subset (multipart
    # uploads are atomic, so a truncated flush never publishes a torn object).
    # $B2X is the binary path b2x_ensure resolved in train.sh (dynamic scope,
    # like $CKPT_DIR/$B2/$RUN_ID above); empty when b2x is unavailable, in which
    # case this collapses to exactly the original rclone line. Invoked directly
    # rather than via the b2x_push wrapper because a shell FUNCTION would not
    # survive into `timeout`'s child.
    #
    # `.preempt_*` is EXCLUDED, and that exclusion is load-bearing. b2x orders
    # NEWEST FIRST, and .preempt_save_complete is by construction the newest file
    # in the checkpoint it certifies — so without this the deadline would upload
    # the completeness FLAG first and truncate before the 646 MB optimizer and
    # 323 MB adapter, publishing a green flag over weights that are not there.
    # That inverts the very prefix-completion property the newest-first order
    # exists to give. The marker is a LOCAL-DISK claim for salvage to read off
    # the dying box; it has no business on B2 ahead of the bytes.
    # Flags BEFORE the positionals: b2x parses with Go's flag package, which
    # stops at the first non-flag, so the old spelling exited 2 (usage) in
    # milliseconds and this flush has ALWAYS gone out over rclone.
    { [ -n "${B2X:-}" ] && timeout 45 "$B2X" push \
        --deadline 40s --exclude STATUS --exclude '.preempt_*' \
        "$CKPT_DIR" "$B2/checkpoints/${RUN_ID}/" 2>/dev/null; } \
      || timeout 45 rclone copy --fast-list --exclude STATUS --exclude '.preempt_*' "$CKPT_DIR" "$B2/checkpoints/${RUN_ID}/" 2>/dev/null || true
  fi
  # final_flush: the cutover fence a handoff understudy waits on (HANDOFF_DESIGN;
  # _handoff_run_signals). Emitted AFTER the bounded flush so the newest checkpoint
  # bytes are on B2 before the understudy delta-pulls. Mirrors jobd.sh's jobs-lane
  # emit. Cheap non-terminal signal — harmless on a normal (non-handoff) preemption
  # (the runmeta fold tolerates unknown events). NOTE: only fires when vast delivers
  # SIGTERM; a fence-park is a graceful docker stop (SIGTERM likely), but delivery is
  # [open] per SPOT_DESIGN §1 — verify run-lane --handoff cutover on a live box.
  emit_event final_flush || true
  exit 143
}
# PREEMPT_TRAP_NO_INSTALL=1 => define the functions but arm NOTHING. jobd.sh
# sources this file purely to reuse `_preempt_local_save` (the jobs lane has its
# OWN `_jobd_preempt` trap and must keep it); installing train.sh's trap there
# would silently REPLACE jobd's handler and lose every per-job `preempted` event
# and bounded flush. Unset on the run lane => arm exactly as before.
[ "${PREEMPT_TRAP_NO_INSTALL:-0}" = "1" ] || trap _preempt_trap TERM INT
