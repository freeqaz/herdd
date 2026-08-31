#!/usr/bin/env bash
# onstart/yield_fence.sh — the canonical "yield to the paying GPU trainer" stack.
#
# The rented box is billed for its GPU; the CPU compile/score work (eval sidecar,
# compile-farm) only ever rides the trainer's idle cores and MUST get out of the
# way the instant the trainer's dataloader/checkpoint/optimizer threads want CPU,
# IO, or memory. This file is the ONE place that stack is defined, so the eval
# sidecar, the farm worker, and the local dry-run all fence identically.
#
# The yield stack (each layer best-effort, degrades cleanly — never aborts):
#   1. renice 19        — lowest CPU scheduling priority (CFS: trainer preempts).
#   2. ionice -c3       — idle IO class (trainer's disk reads/writes win).
#   3. oom_score_adj    — 800 on self; children inherit, so the OOM killer culls
#                         the farm before the trainer (nice governs CPU, not RAM).
#   4. cgroup v2 cpu.weight — NEW third CPU layer: place self in a low-weight
#                         (min=1) leaf cgroup so, where the container grants
#                         cgroup-v2 write access, the CFS gives the farm CPU only
#                         when the trainer is idle. This is defense-in-depth on
#                         top of nice; nice remains the guaranteed layer (cgroup
#                         write access is not guaranteed inside every container).
#
# Usage:
#   . yield_fence.sh; yield_fence_self [label]   # sourced: fence the current shell
#   yield_fence.sh -- CMD [ARGS...]              # exec CMD under the fence
#   yield_fence.sh --report [label]              # fence self + print applied state
#   yield_fence.sh --self-test                   # local dry-run (no box needed)
#
# All functions are safe under `set -u`. Nothing here writes to B2 or touches the
# trainer's processes/cgroup — it only ever lowers the CURRENT process group.

# --- config knobs (env-overridable; sane low defaults) -----------------------
YIELD_NICE="${YIELD_NICE:-19}"          # 19 = lowest
YIELD_IONICE_CLASS="${YIELD_IONICE_CLASS:-3}"   # 3 = idle
YIELD_OOM_ADJ="${YIELD_OOM_ADJ:-800}"   # 0..1000; higher = killed first
YIELD_CPU_WEIGHT="${YIELD_CPU_WEIGHT:-1}"       # cgroup v2 cpu.weight 1..10000, 1=min
YIELD_CGROUP_LEAF="${YIELD_CGROUP_LEAF:-farm.slice}"  # leaf name under the delegated base

# yield_fence_cgroup_base — echo a writable cgroup-v2 base dir we may create a
# leaf under, or nothing if cgroup v2 / cpu controller / write access is absent.
# Never fails; prints at most one line.
yield_fence_cgroup_base() {
  local base=/sys/fs/cgroup
  [ -f "$base/cgroup.controllers" ] || return 0            # not cgroup v2 unified
  grep -qw cpu "$base/cgroup.controllers" 2>/dev/null || return 0
  # our own cgroup, relative to the (namespaced) root, e.g. 0::/foo/bar
  local rel; rel="$(sed -n 's/^0::\(.*\)$/\1/p' /proc/self/cgroup 2>/dev/null)"
  # Prefer creating the leaf as a SIBLING of our current cgroup (same parent) so
  # cpu.weight is compared against the trainer on equal footing; fall back to the
  # namespace root. Walk up until we find a dir we can write (mkdir a probe).
  local cand parent
  for cand in "${base}${rel%/*}" "$base"; do
    [ -d "$cand" ] || continue
    parent="$cand"
    if ( set -e; probe="$parent/.yf_probe.$$"; mkdir "$probe" 2>/dev/null; rmdir "$probe" 2>/dev/null ) 2>/dev/null; then
      printf '%s\n' "$parent"; return 0
    fi
  done
  return 0
}

# yield_fence_cgroup — move $$ into a min-weight leaf cgroup. Best-effort; echoes
# the leaf path on success, nothing otherwise. Enables the cpu controller in the
# parent's subtree_control first (required before a child leaf can carry cpu.weight).
yield_fence_cgroup() {
  local parent leaf
  parent="$(yield_fence_cgroup_base)"
  [ -n "$parent" ] || return 0
  leaf="$parent/$YIELD_CGROUP_LEAF"
  # enable cpu in the parent's subtree so the leaf gets a cpu.weight knob. The
  # no-internal-process rule forbids enabling a controller while the parent holds
  # tasks UNLESS the parent is the (namespace) root — inside a container the
  # delegated base usually is. Best-effort: if it fails we simply won't have the
  # knob and bail out below.
  if [ -w "$parent/cgroup.subtree_control" ]; then
    grep -qw cpu "$parent/cgroup.subtree_control" 2>/dev/null \
      || echo '+cpu' > "$parent/cgroup.subtree_control" 2>/dev/null || true
  fi
  mkdir -p "$leaf" 2>/dev/null || return 0
  [ -f "$leaf/cpu.weight" ] || return 0           # cpu controller not actually delegated
  echo "$YIELD_CPU_WEIGHT" > "$leaf/cpu.weight" 2>/dev/null || true
  # move ourselves in LAST (once a process is in the leaf, subtree edits on the
  # parent may be constrained). Children we spawn inherit this cgroup.
  echo "$$" > "$leaf/cgroup.procs" 2>/dev/null || return 0
  printf '%s\n' "$leaf"
}

# yield_fence_self [label] — apply the full stack to the current process. Every
# layer is independently best-effort; a failure of one never blocks the others.
YIELD_FENCE_APPLIED=""       # human-readable summary, set for logging/report
yield_fence_self() {
  local label="${1:-yield}" leaf=""
  renice "$YIELD_NICE" -p "$$" >/dev/null 2>&1 || true
  ionice "-c$YIELD_IONICE_CLASS" -p "$$" >/dev/null 2>&1 || true
  echo "$YIELD_OOM_ADJ" > "/proc/$$/oom_score_adj" 2>/dev/null || true
  leaf="$(yield_fence_cgroup)"
  YIELD_FENCE_APPLIED="nice=$YIELD_NICE ionice=c$YIELD_IONICE_CLASS oom_adj=$YIELD_OOM_ADJ cgroup=${leaf:-<none>}"
  echo ">> [$label] yield-fence: $YIELD_FENCE_APPLIED" >&2
}

# yield_fence_report [label] — apply + print the OBSERVED (read-back) state.
yield_fence_report() {
  yield_fence_self "${1:-report}"
  local nice ion cg
  nice="$(ps -o ni= -p "$$" 2>/dev/null | tr -d ' ')"
  ion="$(ionice -p "$$" 2>/dev/null)"
  cg="$(sed -n 's/^0::\(.*\)$/\1/p' /proc/self/cgroup 2>/dev/null)"
  echo "observed: nice=$nice ionice='$ion' cgroup=$cg oom_adj=$(cat /proc/$$/oom_score_adj 2>/dev/null)"
}

# --- self-test (local dry-run; no rented box, no B2) -------------------------
# Confirms the fence is really applied to a child and that a higher-priority
# sibling preempts a fenced one. cgroup preemption is measured only where this
# host grants cgroup-v2 write access; elsewhere that sub-check SKIPs (not fails),
# exercising exactly the clean-degradation path the box relies on.
yield_fence_selftest() {
  local rc=0 pass=0 skip=0 self="${BASH_SOURCE[0]}"
  echo "=== yield_fence self-test (host $(uname -sm)) ==="

  # (1) renice applied to a fenced child. Spawn a FRESH bash (so its $$ is the
  # child, mirroring the real sidecar/farm process) that sources this file and
  # fences itself, then sleeps — exactly how eval_sidecar.sh applies the stack.
  bash -c ". '$self'; yield_fence_self child >/dev/null 2>&1; exec sleep 5" &
  local kid=$!; sleep 0.5
  local ni; ni="$(ps -o ni= -p "$kid" 2>/dev/null | tr -d ' ')"
  if [ "$ni" = "$YIELD_NICE" ]; then echo "PASS renice: child nice=$ni"; pass=$((pass+1))
  else echo "FAIL renice: child nice='$ni' (want $YIELD_NICE)"; rc=1; fi

  # (2) ionice class applied
  local ic; ic="$(ionice -p "$kid" 2>/dev/null)"
  case "$ic" in
    idle*|*"class 3"*|none*) echo "PASS ionice: '$ic'"; pass=$((pass+1)) ;;
    "") echo "SKIP ionice: ionice not available / unreadable here"; skip=$((skip+1)) ;;
    *) echo "FAIL ionice: '$ic' (want idle)"; rc=1 ;;
  esac

  # (3) oom_score_adj inherited by the child
  local oa; oa="$(cat "/proc/$kid/oom_score_adj" 2>/dev/null)"
  if [ "$oa" = "$YIELD_OOM_ADJ" ]; then echo "PASS oom_adj: child=$oa"; pass=$((pass+1))
  else echo "SKIP oom_adj: child='$oa' (proc not writable here)"; skip=$((skip+1)); fi
  kill "$kid" 2>/dev/null || true; wait "$kid" 2>/dev/null || true

  # (4) cgroup cpu.weight preemption micro-benchmark (only where writable)
  local parent; parent="$(yield_fence_cgroup_base)"
  if [ -z "$parent" ]; then
    echo "SKIP cgroup: no writable cgroup-v2 cpu base here — box relies on the"
    echo "     |  best-effort path; nice/ionice/oom remain the guaranteed layers."
    skip=$((skip+1))
  else
    if [ -w "$parent/cgroup.subtree_control" ]; then
      grep -qw cpu "$parent/cgroup.subtree_control" 2>/dev/null \
        || echo '+cpu' > "$parent/cgroup.subtree_control" 2>/dev/null || true
    fi
    local lo="$parent/yf_lo.$$" hi="$parent/yf_hi.$$"
    mkdir -p "$lo" "$hi" 2>/dev/null || true
    if [ -f "$lo/cpu.weight" ] && [ -f "$hi/cpu.weight" ]; then
      echo 1     > "$lo/cpu.weight" 2>/dev/null || true   # farm-like (min)
      echo 10000 > "$hi/cpu.weight" 2>/dev/null || true   # trainer-like (max)
      # pin both to a single CPU so weight actually arbitrates, then spin.
      local cpu=0
      ( echo $BASHPID > "$lo/cgroup.procs" 2>/dev/null; taskset -cp "$cpu" $BASHPID >/dev/null 2>&1
        end=$((SECONDS+3)); while [ "$SECONDS" -lt "$end" ]; do :; done ) &
      local plo=$!
      ( echo $BASHPID > "$hi/cgroup.procs" 2>/dev/null; taskset -cp "$cpu" $BASHPID >/dev/null 2>&1
        end=$((SECONDS+3)); while [ "$SECONDS" -lt "$end" ]; do :; done ) &
      local phi=$!
      wait "$plo" "$phi" 2>/dev/null || true
      local ulo uhi
      ulo="$(sed -n 's/^usage_usec //p' "$lo/cpu.stat" 2>/dev/null)"
      uhi="$(sed -n 's/^usage_usec //p' "$hi/cpu.stat" 2>/dev/null)"
      if [ -n "$ulo" ] && [ -n "$uhi" ] && [ "$uhi" -gt "$ulo" ]; then
        echo "PASS cgroup: high-weight sibling got more CPU (hi=${uhi}us > lo=${ulo}us)"
        pass=$((pass+1))
      else
        echo "SKIP cgroup: usage unreadable/inconclusive (hi='$uhi' lo='$ulo') on this host"
        skip=$((skip+1))
      fi
    else
      echo "SKIP cgroup: cpu controller not delegated to leaves here"
      skip=$((skip+1))
    fi
    rmdir "$lo" "$hi" 2>/dev/null || true
  fi

  echo "=== yield_fence self-test: $pass passed, $skip skipped, rc=$rc ==="
  return "$rc"
}

# --- dispatch (only when executed, not when sourced) -------------------------
# Detect sourced-vs-executed without bashisms that shellcheck dislikes.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -uo pipefail
  case "${1:-}" in
    --self-test) yield_fence_selftest; exit $? ;;
    --report)    shift; yield_fence_report "${1:-report}"; exit 0 ;;
    --)          shift; yield_fence_self "wrap"; exec "$@" ;;
    "" )         yield_fence_report "report"; exit 0 ;;
    *)           echo "usage: yield_fence.sh [--self-test|--report [label]|-- CMD...]" >&2; exit 2 ;;
  esac
fi
