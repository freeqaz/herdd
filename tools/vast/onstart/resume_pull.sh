# onstart/resume_pull.sh — narrowed checkpoint resume-pull (SOURCED companion).
#
# Sourced by onstart/train.sh from B2 (eval-env/resume_pull.sh, staged by
# eval-env/bake.sh alongside the other companions) right before training
# resume. Externalized because train.sh rides Vast's 16 KiB inline-onstart cap.
#
# Why: the 180s sync loop is `rclone copy` (never deletes on B2) and no prune
# exists, so a long run's checkpoints/<RID>/ prefix holds every save ever made.
# The legacy resume pulled ALL of it — a 20-save run pulled ~20x the bytes HF
# resume needs, on a billed idle box, on every eviction/relaunch (the only
# recurring real-dollar item in the 2026-07-09 B2 efficiency audit; see
# tools/vast/B2_EFFICIENCY_PLAN_2026-07-09.md).
#
# What it does: per layout root (flat checkpoint-*/ AND arms/<name>/checkpoint-*/,
# the same dual layout the sync-loop step-parse globs), pull only the newest TWO
# checkpoint-<step> dirs. Two, not one: the newest can be a partial upload from
# a box that died mid-push; HF resume validation then falls back to the complete
# one. Then one cheap top-level-residue copy (adapter/tokenizer/config files)
# excluding the checkpoint dirs and STATUS.
#
# Contract (train.sh sources this; correctness beats efficiency):
#   * expects B2, RUN_ID, CKPT_DIR, RC_FAST[] from the caller's scope, and
#     b2x_pull the same way — train.sh sources b2x_boot.sh well before it
#     sources this companion, so the function is already in scope. It was never
#     called here, which left the eviction path (1-20 GB, on EVERY relaunch)
#     rclone-only. A no-op stub is defined by the shim when b2x is absent, so
#     the `||` fallbacks below are the pre-existing lines unchanged.
#   * NEVER calls exit (it would kill train.sh) and always returns 0
#   * zero parsed steps or lsf failure => verbatim legacy whole-prefix pull
#   * --exclude STATUS preserved everywhere; no B2 writes or deletes, reads only

resume_pull_narrowed() {
  local r s steps got=0 roots=("")
  # layout roots relative to checkpoints/<RID>/: flat "" + each arms/<name>/
  while IFS= read -r r; do roots+=("arms/${r%/}/"); done \
    < <(rclone lsf --dirs-only "$B2/checkpoints/${RUN_ID}/arms" 2>/dev/null)
  for r in "${roots[@]}"; do
    steps=$(rclone lsf --dirs-only "$B2/checkpoints/${RUN_ID}/${r}" 2>/dev/null \
            | sed -n 's#^checkpoint-\([0-9][0-9]*\)/$#\1#p' | sort -n | tail -2)
    [ -n "$steps" ] || continue
    # while-read, not `for s in $steps`: no reliance on word-splitting (bash
    # splits, zsh wouldn't — keep the companion shell-agnostic for harnesses)
    while IFS= read -r s; do
      [ -n "$s" ] || continue
      b2x_pull "$B2/checkpoints/${RUN_ID}/${r}checkpoint-${s}" \
        "$CKPT_DIR/${r}checkpoint-${s}" 2>/dev/null \
        || rclone copy "${RC_FAST[@]}" "$B2/checkpoints/${RUN_ID}/${r}checkpoint-${s}" \
        "$CKPT_DIR/${r}checkpoint-${s}" 2>/dev/null && got=1
    done <<EOF
$steps
EOF
  done
  if [ "$got" -eq 1 ]; then
    echo ">> resume_pull: narrowed pull (newest<=2 per root: ${roots[*]:-flat})"
    # top-level residue minus the checkpoint dirs and the live STATUS marker
    b2x_pull "$B2/checkpoints/${RUN_ID}" "$CKPT_DIR" \
      --exclude STATUS --exclude 'checkpoint-*/**' --exclude 'arms/*/checkpoint-*/**' 2>/dev/null \
      || rclone copy "${RC_FAST[@]}" --exclude STATUS --exclude 'checkpoint-*/**' \
      --exclude 'arms/*/checkpoint-*/**' \
      "$B2/checkpoints/${RUN_ID}" "$CKPT_DIR" 2>/dev/null || true
  else
    # first boot (empty prefix), lsf failure, or every narrowed copy failed:
    # the legacy whole-prefix pull, verbatim
    b2x_pull "$B2/checkpoints/${RUN_ID}" "$CKPT_DIR" --exclude STATUS 2>/dev/null \
      || rclone copy "${RC_FAST[@]}" --exclude STATUS \
      "$B2/checkpoints/${RUN_ID}" "$CKPT_DIR" 2>/dev/null || true
  fi
  return 0
}
resume_pull_narrowed
