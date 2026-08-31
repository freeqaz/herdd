#!/usr/bin/env bash
# onstart/jobd.sh — per-box job daemon for the B2-mediated job system (JOBS_DESIGN.md).
#
# WHAT (v2): polls b2:$B2_BUCKET/jobs/queue/$IID/ (~20s) and SCHEDULES tickets onto
# the box's GPUs: each job declares `needs.gpus` (N cards or "all"); jobd assigns
# free cards via CUDA_VISIBLE_DEVICES and runs jobs CONCURRENTLY until the box is
# saturated (strict FIFO — the oldest ticket that does not fit blocks younger ones,
# so a whole-box job can never starve). Jobs are INTERRUPTION-TOLERANT: a job
# interrupted by preemption/park/daemon-death is picked back up on the next boot
# (`resumed` event, checkpoint pull-back, JOB_RESTART_COUNT env), bounded by the
# ticket's max_restarts. Per job: claim -> bundle download by hash (local dedupe)
# -> fresh extract -> entrypoint with a hard timeout -> heartbeats -> results globs
# FIRST, log.txt, results.DONE.json LAST (farm_worker crash-safe marker-last rule)
# -> done/failed. A running job also watches jobs/<id>/CANCEL (written by `herdd
# job cancel`): on sight it kills the entrypoint tree and records terminal
# `cancelled` (non-resumable — distinct from an interrupted job, which resumes).
# The same poll watches jobs/<id>/CHECKPOINT_NOW (`herdd job flush`): on sight it
# fires ONE unfiltered checkpoint sync (trig=flush-now, whole declared glob, no
# --min-age), deletes the marker, and leaves the entrypoint running — a flush, not
# a stop. It is for a PRE-PARK / PRE-HANDOFF flush and CANNOT rescue an eviction:
# vast delivers no SIGTERM on a spot reclaim and the warning budget is single-digit
# seconds (measured 2026-08-26), far short of one marker poll. See
# _flush_marker_consume.
#
# WHY THIS SHAPE: generalizes onstart/farm_worker.sh (idempotent, resumable across
# box death via a result marker on B2) + runmeta's append-only event discipline
# (jobd.py emits the same immutable event envelope the laptop `herdd job` reads).
# It replaces the hand-rolled ssh+scp+nohup+tunnel operator surface (doc 109).
#
# DELIVERY: (a) baked into serve/train onstarts behind JOBD=1 (restarts with the
# box — onstart re-runs on every resume), or (b) pushed onto an existing box by
# `herdd job attach <IID>` (+ a /root/onstart.sh hook so it ALSO restarts on
# resume). A JOBD_STATUS marker mirrors SERVE_STATUS conventions. `herdd job
# supervise <IID>` pairs with this daemon for spot boxes: it defends/rescues the
# bid and re-attaches jobd; THIS side owns resuming the interrupted jobs.
#
# ENV CONTRACT (box provides IID + B2_*; the rest are optional / test hooks):
#   INSTANCE_ID / CONTAINER_ID   the vast instance id (queue prefix); JOBD_IID overrides
#   B2_BUCKET B2_KEY_ID B2_APPLICATION_KEY B2_S3_ENDPOINT [B2_REGION]   B2 transport
#   JOBD_ROOT           workspace root (default /workspace)
#   JOBD_POLL           poll interval seconds (default 20)
#   JOBD_HEARTBEAT_S    heartbeat interval seconds (default 60)
#   JOBD_CANCEL_POLL    running-job CANCEL-marker poll seconds (default 15)
#   JOBD_NO_CANCEL=1    disable the cooperative cancel-watch (tests)
#   JOBD_CPU_SLOTS      max concurrent gpus=0 jobs (default 2)
#   JOBD_ONCE=1         one claim pass, then DRAIN running jobs and exit (tests)
#   JOBD_MAX_JOBS=N     stop claiming after N spawns this boot (default 0 = unbounded)
#   JOBD_SKIP_GPU=1     skip the nvidia probe: box presents 0 GPUs (CPU boxes/tests)
#   JOBD_FAKE_GPUS      "idx:gb,idx:gb" fake inventory (scheduler tests, no nvidia)
#   JOBD_SKIP_B2CONFIG=1  do not run b2_sync.sh config (tests: rclone shim needs none)
#   JOBD_B2_SYNC        path to b2_sync.sh (default: beside this file, then ../)
#   BOX_IDENTITY_NONCE  cred-broker identity (buildout §2.6); absent => cred
#                       refresh is a no-op and behavior is unchanged
#   B2_KEY_EXPIRES_AT   epoch expiry of the shipped B2 key (refresh trigger)
#   JOBD_CRED_DIR       cred-refresh marker dir (default: beside this file)
#   JOBD_CRED_CLIENT    cred_client.py path override (tests)
#   JOBD_CRED_THROTTLE_S  min seconds between refresh ATTEMPTS (default 3600)
#   JOBD_CRED_TIMEOUT_S   hard cap on ONE cred_client run (default 900 — covers
#                       the full §2.5 transport ladder worst case with slack)
#   JOBD_ENV_FILE       jobd.env path override (default: beside this file)
#   --- checkpoint sync + BOX-DISK lifecycle (JOBS_DESIGN.md §"Checkpoint disk
#       lifecycle"). NOTHING here ever deletes anything on B2. ----------------
#   JOBD_CKPT_MIN_AGE   periodic-pass --min-age window (default 45s); the
#                       fire-on-arrival pass uses none (completeness is proven)
#   JOBD_CKPT_TAIL      live-append tail snapshot, task #110 (default 1; 0 off).
#                       A file appended faster than JOBD_CKPT_MIN_AGE can NEVER
#                       clear that window, so it was never shipped at all. This
#                       stages a copy cut at its LAST COMPLETE LINE — see
#                       jobmeta.ckpt_tail_snapshot for the admission rules
#   JOBD_CKPT_TAIL_MAX_MB  per-file cap on that snapshot (default 128)
#   JOBD_CKPT_WATCH_S   fire-on-arrival readdir interval (default 5; 0 disables,
#                       leaving only the JOB_CHECKPOINT_S timer)
#   JOBD_CKPT_SETTLE_S  quiescence window a checkpoint dir must clear before it
#                       counts as complete / prunable (default 5). 0 disables the
#                       whole quiescence test, which reduces "complete" to a bare
#                       trainer_state.json existence check — the fast path may
#                       then fire before a non-zero rank's rng_state_<i>.pth
#                       lands. Deletion stays read-back gated either way.
#   JOBD_CKPT_PRUNE=0   disable delete-after-sync (default on)
#   JOBD_CKPT_KEEP      local checkpoint dirs retained per layout root
#                       (default 2, CLAMPED to >=2 — resume_pull.sh's newest-2)
#   JOBD_CKPT_SCRUB=0   disable the end-of-run checkpoint scrub (default on)
#   JOBD_DISK_PRECHECK=0  disable the pre-staging free-space refusal (default on)
#   JOBD_DISK_HEADROOM_GB  working headroom added to a MEASURED asset requirement
#                       (default 5)
#   JOBD_MIN_FREE_GB    operator floor on free GB before staging (default 0 = the
#                       measured requirement alone decides). Env twin of the
#                       ticket-declared `needs.disk_gb` (JOB_NEEDS_DISK_GB).
#   JOBD_DISK_SIZE_TIMEOUT_S  cap on ONE `rclone size` sizing probe (default 120)
#
#   --- B2 transfer guards (see the block above asset_pull) --------------------
#   A slow or flaky host must surface as a NAMED failure, never a silent hang.
#   Two signals, matching the control-plane docker-pull watchdog: a wall-clock
#   ceiling and a bytes-per-second floor. Verdicts reach the event log as
#   `asset_stage_timeout:` / `asset_stage_slow:` rather than `asset_stage_failed:`.
#   JOBD_ASSET_GUARD=0    disable both guards (restores pre-2026-08-03 behavior)
#   JOBD_ASSET_MIN_MBPS       aggregate floor, MB/s (default 3; 0 disables)
#   JOBD_ASSET_MBPS_WINDOW_S  the floor must be under water this long (default 300)
#   JOBD_ASSET_SLACK_S        non-byte-bound allowance in the ceiling (default 300)
#   JOBD_ASSET_MIN_TIMEOUT_S  ceiling never shorter than this (default 900)
#   JOBD_ASSET_TIMEOUT_S      flat ceiling when the remote size is UNKNOWN (default 14400)
#   JOBD_ASSET_POLL_S         watcher tick (default 1)
#   JOBD_BUNDLE_TIMEOUT_S     cap on ONE bundle download (default 600)
#   JOBD_BUNDLE_RETRIES / JOBD_BUNDLE_BACKOFF   bundle retry rounds / step (3 / 5)
#
#   --- interrupted-transfer self-heal + bounded retry (defect #77) ------------
#   An rclone pull killed mid-flight leaves `<name>.<rand>.partial` temps and a
#   *.index.json that names shards which never landed. The bundle's own gate then
#   reports every shard MISSING, `die`s, and jobd used to take EVERY nonzero rc
#   terminal — so a transport hiccup burned the whole job. See the block above
#   `_interrupted_transfer_evidence` for the classifier, the skip-guard defeat,
#   and why this rides its OWN counter instead of max_restarts.
#   JOBD_TRANSFER_HEAL=0      disable classification + self-heal + retry entirely
#                             (restores the pre-2026-08-09 "every rc!=0 is
#                             terminal" behavior exactly)
#   JOBD_TRANSFER_RETRIES     retries after an interrupted-transfer verdict
#                             (default 1; CLAMPED to 2 — this is a hiccup budget,
#                             not a crash-loop budget)
#   JOBD_TRANSFER_BACKOFF_S   backoff before the retry lands (default 30)
#   JOBD_TRANSFER_MIN_FREE_GB free-GB floor under which an interrupted transfer is
#                             ruled `insufficient_disk:` and NEVER retried
#                             (default 5) — ENOSPC leaves partials that look
#                             exactly like an interrupt
#   JOBD_PARTIAL_STALE_S      quiescence window applied to DELETION in a shared
#                             root while ANOTHER job is running (default 120) —
#                             a partial written inside it belongs to a sibling
#                             arm's LIVE pull. Never applied to classification,
#                             and not applied at all when nothing else is running
#   JOBD_TRANSFER_SCAN_DIRS   extra ':'-separated scan roots (default: the job
#                             workdir + $ASSETS_DIR + $ROOT minus $JOBS_DIR)
#   JOBD_TMPFS_PROBE=0  skip the boot scratch probe's mount -t tmpfs attempt
#                       (the rest of the probe still runs; default on)
#   JOBD_BOOT_NONCE_FILE  container-boot nonce path (default
#                       /dev/shm/jobd_boot_nonce_<IID> — a tmpfs, so a box
#                       stop/start wipes it). Test seam for the resume-time
#                       preempt-vs-crash inference; see the boot-nonce block.
#   --- boot GEMM ceiling (host acceptance TELEMETRY; see gemm_probe) ---------
#   JOBD_GEMM_PROBE=0   disable the boot GEMM ceiling probe entirely (default on)
#   JOBD_GEMM_TIMEOUT_S hard wall-clock cap on the probe, outer bound (default
#                       120; gemm_probe's own --deadline-s is 90, so the shell
#                       timeout is a backstop, not the primary bound)
#   JOBD_GEMM_MAX_AGE_S re-probe only if the cached record is older than this
#                       (default 86400). A machine's ceiling does not change
#                       between a park and a resume; re-measuring every boot
#                       would pay 30s for a number we already have.
#   JOBD_GEMM_PY        interpreter for the probe (default: the venv named by
#                       /workspace/.train_env_activate, else python3)
#   --- shared Triton JIT cache (triton_cache.py; FAIL-OPEN at every step) ----
#   JOBD_TRITON_CACHE=0     disable the cross-box Triton cache entirely
#                           (default on: pull at boot, TRITON_CACHE_DIR export
#                           to GPU entrypoints, push after each GPU job)
#   JOBD_TRITON_CACHE_DIR   box-level cache dir (default $ROOT/triton-cache)
#   JOBD_TC_TIMEOUT_S       outer bound on one pull/push (default 240)
#   JOBD_TC_PY              interpreter for key detection (default: the venv
#                           named by /workspace/.train_env_activate, else
#                           python3 — same resolution as JOBD_GEMM_PY; without
#                           torch in it the key degrades to torchnone-*, which
#                           is a working but coarser cache identity)
set -uo pipefail

JOBD_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${JOBD_PYTHON:-python3}"
JH="$JOBD_DIR/jobd.py"
[ -f "$JH" ] || { echo "!! jobd: helper not found: $JH" >&2; exit 1; }
MP="$JOBD_DIR/metrics_probe.py"   # host-metrics probe (optional; GPU/cpu/net/disk)

# --- preempt-forced checkpoint primitive (SPOT_DESIGN §3.3) -------------------
# Source preempt_trap.sh for `_preempt_local_save` / `_preempt_save_report` ONLY.
# PREEMPT_TRAP_NO_INSTALL=1 is load-bearing: without it the source would arm
# train.sh's `_preempt_trap` on TERM/INT and REPLACE jobd's own `_jobd_preempt`,
# losing every per-job `preempted` event, breadcrumb and bounded flush. jobd keeps
# its trap and merely borrows the primitive (called from `_jobd_preempt`).
#
# Missing file => `_preempt_local_save` stays undefined and the preempt path skips
# it (the `declare -F` guard there), i.e. exactly the pre-2026-08-06 behaviour. A
# box on an older jobd bundle therefore degrades rather than breaking.
if [ -f "$JOBD_DIR/preempt_trap.sh" ]; then
  # shellcheck disable=SC1091
  PREEMPT_TRAP_NO_INSTALL=1 . "$JOBD_DIR/preempt_trap.sh" || true
fi
# The trainer imports preempt_save.py by probing a few flat dirs; tell it exactly
# where the bundle put it so the jobs lane never depends on cwd or a guess.
#
# AND copy it to $ROOT (/workspace), which is the compatibility half. The jobs
# lane stages the trainer from b2:.../runsets/<name>/, and THAT copy is pinned by
# a sha1 `tracks:` gate in each job-config — restaging it to pick up the new
# probe would invalidate every authorised bundle. It does not need restaging: the
# already-staged trainer ALREADY probes /workspace (it just never found anything
# there, because only the run lane's train.sh ever staged companions and the jobs
# lane does not run train.sh). Putting the module where the OLD probe already
# looks makes this work with the trainer that is on B2 today, while
# $PREEMPT_SAVE_DIR is the exact answer for newer ones.
#
# NB: spelled `${JOBD_ROOT:-/workspace}` and not `$ROOT` on purpose — $ROOT is
# defined ~40 lines BELOW this block, and `set -u` is on, so referencing it here
# would abort the daemon at boot on every box.
if [ -f "$JOBD_DIR/preempt_save.py" ]; then
  export PREEMPT_SAVE_DIR="$JOBD_DIR"
  _ps_root="${JOBD_ROOT:-/workspace}"
  [ "$JOBD_DIR" = "$_ps_root" ] \
    || cp -f "$JOBD_DIR/preempt_save.py" "$_ps_root/preempt_save.py" 2>/dev/null || true
  unset _ps_root
fi

IID="${JOBD_IID:-${INSTANCE_ID:-${CONTAINER_ID:-}}}"
[ -n "$IID" ] || { echo "!! jobd: no instance id (set INSTANCE_ID/CONTAINER_ID or JOBD_IID)" >&2; exit 1; }
: "${B2_BUCKET:?jobd: B2_BUCKET required}"
B2="b2:${B2_BUCKET}"                 # bucket-wide READS (assets, queue, tickets)
# Option-1b scoped WRITE remote: when the box carries a prefix-restricted write
# key (B2_WRITE_KEY_ID), jobd's writes (ALL under jobs/) go through [b2w]; reads
# stay on [b2]. Single-key box (no B2_WRITE_*) => B2W == B2 (unchanged). Keep
# every $B2W target under jobs/ or the scoped key 403s. See CREDENTIAL_LIFECYCLE.md.
if [ -n "${B2_WRITE_KEY_ID:-}" ]; then B2W="b2w:${B2_BUCKET}"; else B2W="$B2"; fi

# --- handoff epoch guard (HANDOFF_DESIGN §4, task T6; NO-OP off the handoff path) ---
# During a `--handoff` migration the driver stamps jobs/<JOB_ID>/handoff/<epoch>.json
# at each ARM (telemetry) and writes jobs/<JOB_ID>/handoff/promoted at CUTOVER; each
# box gets its own HANDOFF_EPOCH. A box must NOT push to a job's B2 results/ once a
# STRICTLY-NEWER epoch was PROMOTED over it (the understudy after cutover, or a parked
# husk that woke up stale) — that is the two-writer corruption §4 guards against.
# Keyed on the promoted marker's "epoch" field, NOT the max ARM-time <epoch>.json:
# write ownership transfers at PROMOTION, and ARM-keying silenced the still-canonical
# box for a second handoff's whole ARM->cutover window — permanently, after an ABORTED
# attempt (ARM markers are never cleaned up). Mirrors onstart/train.sh's run-lane
# predicate. FAIL-SAFE: HANDOFF_EPOCH unset (every normal job), a missing/unreadable
# promoted marker, or an unparsable epoch field => not stale => push as today.
HANDOFF_EPOCH="${HANDOFF_EPOCH:-}"
_handoff_epoch_stale() {   # _handoff_epoch_stale <jobid>; rc 0 == STALE (refuse push)
  [ -n "$HANDOFF_EPOCH" ] || return 1
  local jid="$1" pj pe
  pj=$(rclone cat "$B2/jobs/$jid/handoff/promoted" 2>/dev/null) || return 1
  pe=$(printf '%s' "$pj" | sed -n 's/.*"epoch":[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)
  [ -n "$pe" ] || return 1
  [ "$pe" -gt "$HANDOFF_EPOCH" ] 2>/dev/null
}
_handoff_stamp_epoch() {   # _handoff_stamp_epoch <jobid>; best-effort, no-op off handoff
  [ -n "$HANDOFF_EPOCH" ] || return 0
  printf '%s\n' "$HANDOFF_EPOCH" \
    | rclone rcat "$B2W/jobs/$1/handoff/owner_epoch" 2>/dev/null || true
}

ROOT="${JOBD_ROOT:-/workspace}"
CACHE_DIR="$ROOT/.job_cache"
JOBS_DIR="$ROOT/jobs"
STATE_DIR="$JOBS_DIR/.state"     # per-job attempts/terminal/running bookkeeping
POLL="${JOBD_POLL:-20}"
HB="${JOBD_HEARTBEAT_S:-60}"
MAXJ="${JOBD_MAX_JOBS:-0}"
CPU_SLOTS="${JOBD_CPU_SLOTS:-2}"
# Consecutive unreadable-ticket polls tolerated before poll_once trusts a job's
# results.DONE.json (see there). ~30 * POLL ≈ 10 min of blip absorbed.
DONE_UNKNOWN_MAX="${JOBD_DONE_UNKNOWN_MAX:-30}"
mkdir -p "$CACHE_DIR" "$JOBS_DIR" "$STATE_DIR"
# Clear any stale preempt marker from a PRIOR boot (disk survives a park, so the
# trap's marker would otherwise persist across a resume and wrongly demote a
# genuinely-crashing job to "resumable"). Written fresh by _jobd_preempt below.
PREEMPT_MARK="$STATE_DIR/.preempting"
rm -f "$PREEMPT_MARK"

# --- container-boot nonce: infer "the box went away" AFTER THE FACT -----------
# Vast delivers NO preempt signal — measured three independent ways (SPOT_DESIGN
# §1: 0 `preempted`/0 `final_flush` across 61 runs; HANDOFF_DESIGN O1 negative
# x3; the v11 job 2026-08-06, whose every post-eviction resume read
# "kind":"crash"). So the trap's `.preempted` breadcrumb almost never exists on
# a real eviction, every preempt-resume was classified a CRASH, and a healthy
# job on a contested spot market burned max_restarts (commonly 2) instead of
# JOBD_PREEMPT_CAP (20) — then died looking flaky rather than outbid (a live
# contributor to the 27B lane's livelock).
#
# The classification is therefore INFERRED AT RESUME TIME from what the box can
# actually observe: did the CONTAINER go down between the job's spawn and this
# resume?  A nonce minted once per container boot lives on a tmpfs
# (/dev/shm — wiped when vast stops/starts or the host reboots, preserved
# across a daemon restart inside a live container), and each spawn records the
# current nonce next to the job's other breadcrumbs on PERSISTENT disk
# ($STATE_DIR survives the park).  At resume:
#   nonce changed   -> the box was taken down under the job  -> PREEMPT
#   nonce unchanged -> the runner died on a LIVE box         -> crash
#   no data (old-bundle spawn, no tmpfs)                     -> crash (old behavior)
# The trap's breadcrumb, when a signal DID arrive, still wins (strongest
# evidence, and it also covers a same-container drain). See poll_once's
# dual-counter cap logic. JOBD_BOOT_NONCE_FILE is the test seam.
BOOT_NONCE_FILE="${JOBD_BOOT_NONCE_FILE:-/dev/shm/jobd_boot_nonce_${IID}}"
if [ ! -s "$BOOT_NONCE_FILE" ]; then
  printf '%s\n' "$(date +%s)-$$-${RANDOM}${RANDOM}" > "$BOOT_NONCE_FILE" 2>/dev/null || true
fi
BOOT_NONCE="$(cat "$BOOT_NONCE_FILE" 2>/dev/null || true)"
[ -n "$BOOT_NONCE" ] || echo ">> [jobd] WARNING: no writable boot-nonce path ($BOOT_NONCE_FILE) — preempt-vs-crash inference disabled, resumes classify as crash" >&2

# --- idle self-park (v2.1) ----------------------------------------------------
# The owner's worst case is a box that burns cash after the queue drains (or that
# never gets a job at all). Default: PARK (stop) ourselves once idle. GPU billing
# ends; disk stays warm for `herdd start` (onstart re-runs -> jobd comes back).
# Every terminal job already pushed its results to B2 before it went terminal, and
# we only ever park with ZERO running jobs, so nothing un-synced is lost.
#   JOBD_IDLE_PARK=0    opt out entirely (box runs forever — old behavior)
#   JOBD_IDLE_PARK_S    grace after the queue DRAINS before parking (default 600,
#                       the MULTI-job default; an explicit value always wins and
#                       is never auto-lowered — see JOBD_IDLE_PARK_S_SINGLE)
#   JOBD_IDLE_PARK_S_SINGLE  grace for a SINGLE-arm box (<=1 job ran here) when
#                       the operator did NOT pin JOBD_IDLE_PARK_S (default 120).
#                       A single-arm training box is done for good once its one
#                       job goes terminal (results already on B2), so it should
#                       reclaim GPU billing fast — no reason to sit 600s idle.
#                       120s (was 60) leaves room for a resume-then-submit to
#                       land its first ticket before the empty-queue timer parks
#                       the box out from under it (the freed on-demand machine is
#                       then taken by another renter). Pre-queueing the ticket
#                       still beats bumping this — jobd never parks while any
#                       pending ticket sits in the queue.
#                       (Spot caveat: parking sooner realizes a host reclaim
#                       sooner; safe here because a done single-arm job needs no
#                       resume — memory parked-spot-box-reclaimed.)
#   JOBD_NO_JOB_PARK_S  deadline when NO job ever arrived (default 3600)
IDLE_PARK="${JOBD_IDLE_PARK:-1}"
IDLE_PARK_S="${JOBD_IDLE_PARK_S:-600}"
# "1" iff the operator pinned the grace (env set + non-empty): then never
# auto-lower to the single-arm default, honor their value verbatim.
IDLE_PARK_S_EXPLICIT="${JOBD_IDLE_PARK_S:+1}"
IDLE_PARK_S_SINGLE="${JOBD_IDLE_PARK_S_SINGLE:-120}"
NO_JOB_PARK_S="${JOBD_NO_JOB_PARK_S:-3600}"
BOOT_TS="$(date +%s)"
LAST_BUSY_TS="$BOOT_TS"
EVER_BUSY=0
PARK_EMITTED=0

log() { echo ">> [jobd $IID] $*" >&2; }

# --- box self-control (self-park) ---------------------------------------------
# Vast injects a per-instance scoped credential (CONTAINER_API_KEY) that can
# manage ONLY this box; the laptop VASTAI_API_KEY is NEVER shipped here. This is
# the SAME mechanism onstart/train.sh + serve_vllm.sh use — we PARK (stop), never
# destroy, so results/disk survive. JOBD_PARK_CMD is a TEST seam: when set it
# stands in for the real curl park (no vast API, no key) so the fake-B2 harness
# can observe a park without a live box.
VAPI="${JOBD_VAPI:-https://console.vast.ai/api/v0/instances}"
_iid_key() {
  _PIID="${INSTANCE_ID:-${CONTAINER_ID:-$IID}}"
  _PKEY="${VASTAI_API_KEY:-${CONTAINER_API_KEY:-}}"
  [ -z "$_PKEY" ] && [ -f "$HOME/.vast_api_key" ] && _PKEY="$(cat "$HOME/.vast_api_key" 2>/dev/null)"
  [ -n "$_PIID" ] && [ -n "$_PKEY" ]
}
self_park() {
  if [ -n "${JOBD_PARK_CMD:-}" ]; then eval "$JOBD_PARK_CMD" || true; return 0; fi
  _iid_key || { log "no iid/self-control key — park by hand: herdd stop $IID"; return 1; }
  log "self-park instance $_PIID (resume: herdd start $_PIID; disk bills until destroyed)"
  curl -s --connect-timeout 10 --max-time 30 --retry 3 --retry-delay 5 -X PUT -H "Authorization: Bearer ${_PKEY}" -H 'Content-Type: application/json' \
    -d '{"state":"stopped"}' "$VAPI/${_PIID}/" >/dev/null 2>&1 || true
}

# --- cred-broker key refresh (cred-broker-buildout.md §2.6, C6) -----------------
# A box launched with a broker identity (BOX_IDENTITY_NONCE) rotates its B2 key
# IN PLACE before expiry by running cred_client.py (expected beside this file —
# staging it into the jobd bundle is herdd's job; absent => logged skip).
# The client rewrites the [b2]/[b2w] rclone remotes + jobd.env verify-then-swap.
# Markers, both under $CRED_DIR:
#   .cred_refresh_now    touched by the auth-failure greps -> immediate attempt
#   .cred_refresh_last   epoch of the last ATTEMPT (success or not) -> throttle
# No nonce (every pre-broker box) => maybe_refresh_creds is a cheap no-op and
# today's behavior is untouched. The B2-mediated lane needs no broker URL, so
# nonce presence is the ONLY gate. All best-effort: a refresh failure is logged
# and swallowed — it must NEVER crash the daemon. The client runs in the
# BACKGROUND under a hard `timeout`: its transport ladder (direct POST ->
# tailnet join -> B2 lane) can wedge on a dead network, and the poll loop must
# keep claiming/reaping jobs + parking regardless (a synchronous call here made
# the whole daemon externally indistinguishable from a dead box).
CRED_DIR="${JOBD_CRED_DIR:-$JOBD_DIR}"
CRED_CLIENT="${JOBD_CRED_CLIENT:-$JOBD_DIR/cred_client.py}"
CRED_THROTTLE_S="${JOBD_CRED_THROTTLE_S:-3600}"
CRED_TIMEOUT_S="${JOBD_CRED_TIMEOUT_S:-900}"
JOBD_ENV_FILE="${JOBD_ENV_FILE:-$JOBD_DIR/jobd.env}"
CRED_REFRESH_PID=""            # pid of the in-flight background refresh, if any
mkdir -p "$CRED_DIR" 2>/dev/null || true
cred_reload_env() {
  # Re-source the (cred_client-rewritten) jobd.env AND recompute the remotes.
  # Rotation can change the key SHAPE: a box launched on a single bucket-wide
  # key (B2W == "$B2") gets a scoped pair from the broker whose [b2] half is
  # READ-ONLY — keeping the launch-time B2W would 403 every jobd write until a
  # daemon restart. `.`-sourcing cannot UNSET vars the new jobd.env dropped, so
  # clear the write-pair vars first (covers the pair->single downgrade too).
  [ -f "$JOBD_ENV_FILE" ] || return 0
  unset B2_WRITE_KEY_ID B2_WRITE_APPLICATION_KEY
  . "$JOBD_ENV_FILE" 2>/dev/null || true
  B2="b2:${B2_BUCKET}"
  if [ -n "${B2_WRITE_KEY_ID:-}" ]; then B2W="b2w:${B2_BUCKET}"; else B2W="$B2"; fi
}
maybe_refresh_creds() {
  [ -n "${BOX_IDENTITY_NONCE:-}" ] || return 0     # pre-broker box: cheap no-op
  local now want=0 last=0 rc=""
  local flag="$CRED_DIR/.cred_refresh_now" mark="$CRED_DIR/.cred_refresh_last"
  local rcf="$CRED_DIR/.cred_refresh_rc"
  # refresh already in flight? NEVER block the poll loop on it — harvest when
  # done, otherwise come back next tick.
  if [ -n "$CRED_REFRESH_PID" ]; then
    kill -0 "$CRED_REFRESH_PID" 2>/dev/null && return 0
    wait "$CRED_REFRESH_PID" 2>/dev/null || true
    CRED_REFRESH_PID=""
    if [ -f "$rcf" ]; then read -r rc < "$rcf" 2>/dev/null || true; rm -f "$rcf"; fi
    if [ "$rc" = "0" ]; then
      rm -f "$flag" 2>/dev/null || true
      # rotated key installed: re-source jobd.env for the new B2_* + expiry and
      # recompute B2/B2W — the key shape may have flipped single<->pair (rclone
      # itself reads the rewritten rclone.conf, not these vars)
      cred_reload_env
      log "cred refresh OK (expires_at=${B2_KEY_EXPIRES_AT:--} write_remote=$B2W)"
    else
      log "cred refresh FAILED (rc=${rc:-timeout/killed}) — keeping current key; retry >=${CRED_THROTTLE_S}s (log: $CRED_DIR/cred_client.log)"
    fi
    return 0
  fi
  now="$(date +%s)"
  [ -f "$flag" ] && want=1
  if [ "$want" = 0 ]; then
    case "${B2_KEY_EXPIRES_AT:-}" in
      ''|*[!0-9]*) : ;;                            # unset/garbage: no expiry trigger
      *) [ "$now" -gt $(( B2_KEY_EXPIRES_AT - 86400 )) ] && want=1 ;;
    esac
  fi
  [ "$want" = 1 ] || return 0
  if [ -f "$mark" ]; then read -r last < "$mark" 2>/dev/null || true; fi
  case "$last" in ''|*[!0-9]*) last=0 ;; esac
  [ $(( now - last )) -ge "$CRED_THROTTLE_S" ] || return 0
  echo "$now" > "$mark" 2>/dev/null || true
  if [ ! -f "$CRED_CLIENT" ]; then
    log "cred refresh wanted (expires_at=${B2_KEY_EXPIRES_AT:--}) but no client at $CRED_CLIENT — skipping"
    return 0
  fi
  log "cred refresh: running cred_client in background (flag=$([ -f "$flag" ] && echo 1 || echo 0) expires_at=${B2_KEY_EXPIRES_AT:--} timeout=${CRED_TIMEOUT_S}s)"
  # BACKGROUND + hard timeout (see section comment). JOBD_DIR exported inline so
  # the client's jobd.env default matches ours; the client log stays on box disk
  # (never through B2 — the key may be dead). Exit code lands in $rcf for the
  # next tick's harvest; success is applied there (flag clear + env re-source).
  rm -f "$rcf" 2>/dev/null || true
  ( JOBD_DIR="$JOBD_DIR" timeout "$CRED_TIMEOUT_S" "$PY" "$CRED_CLIENT" \
      --jobd-env "$JOBD_ENV_FILE" >>"$CRED_DIR/cred_client.log" 2>&1
    echo $? > "$rcf" 2>/dev/null ) &
  CRED_REFRESH_PID=$!
  return 0
}

# --- single-instance guard (per ROOT): a second daemon (double attach, JOBD=1
# onstart + manual attach) would double-run jobs. flock is per-boot; the fd
# stays open for the daemon's lifetime.
exec 9>"$ROOT/.jobd.lock"
if ! flock -n 9; then
  log "another jobd already holds $ROOT/.jobd.lock — exiting (attach is idempotent)"
  exit 0
fi

# --- rclone remote config (idempotent; skippable in tests) --------------------
if [ "${JOBD_SKIP_B2CONFIG:-0}" != "1" ]; then
  B2SYNC="${JOBD_B2_SYNC:-}"
  if [ -z "$B2SYNC" ]; then
    for c in "$JOBD_DIR/b2_sync.sh" "$JOBD_DIR/../b2_sync.sh"; do
      [ -f "$c" ] && { B2SYNC="$c"; break; }
    done
  fi
  if [ -n "$B2SYNC" ] && ! rclone listremotes 2>/dev/null | grep -q '^b2:'; then
    bash "$B2SYNC" config >/dev/null 2>&1 || log "b2_sync config failed (continuing)"
  fi
fi

# --- b2x transport (optional; every use falls back to the rclone line) --------
# Sourced AFTER the rclone remote exists, because b2x_ensure's primary bootstrap
# rung fetches the binary over that remote. Absent/unfetchable => b2x_pull and
# b2x_push return 1 and every call site keeps its original rclone behavior.
for _c in "$JOBD_DIR/b2x_boot.sh" "$JOBD_DIR/onstart/b2x_boot.sh"; do
  [ -f "$_c" ] && { . "$_c"; break; }
done
command -v b2x_ensure >/dev/null 2>&1 || { b2x_pull() { return 1; }; b2x_push() { return 1; }; }

# >>> CKPT_LIFECYCLE_BEGIN (do not remove: test_jobd_ckpt_lifecycle.py sources
# the block between these two sentinels to unit-test the deletion logic in
# isolation, with `log`/`rclone`/`_handoff_epoch_stale` stubbed. Everything
# between them must therefore stay a pure function definition — no top-level
# side effects beyond the three CKPT_* result variables.)
# =============================================================================
# BOX-DISK CHECKPOINT LIFECYCLE  (JOBS_DESIGN.md §"Checkpoint disk lifecycle")
# =============================================================================
# Owner directive 2026-08-05: "have the box only retain a small number of
# checkpoints, or have it delete older checkpoints from the disk after they sync
# to b2 ... scrub checkpoints after a run is finished".
#
# MEASURED SHAPE (v7, Qwen2.5-Coder-7B, LoRA r32 all-linear): ONE checkpoint is
# ~0.98 GB (adapter_model.safetensors 323 MB fp32 + optimizer.pt 646 MB +
# tokenizer.json 11 MB). At SAVE_STEPS=10 over 156 steps that is ~15 GB PER ARM,
# and matrix.py runs its arms SERIALLY on ONE box — so arm 1's 15 GB sits on the
# disk for the whole of arm 2.
#
# WHY NOT SAVE_TOTAL_LIMIT: the trainer's own rotation would destroy the EARLY
# checkpoints the dose-curve / echo-collapse analysis needs (v4's dose curve was
# U-shaped). B2 must keep the FULL grid. Only the BOX disk is pruned here, and
# nothing in this block ever deletes anything on B2.
#
# THE SAFETY RULE, stated once: a local checkpoint dir is deleted ONLY after B2
# has been READ BACK and shown to hold that exact directory. An rclone/b2x exit
# code is not evidence. Anything that cannot be verified — a listing that errors,
# a size probe that will not parse, a name-set that differs by one file — is a
# REFUSAL to delete, never a permission. Fail safe, never fail open.

# _ckpt_step <path> — echo the trailing checkpoint-<N> step number; rc 1 if the
# basename is not a checkpoint dir. The ONE place the naming convention is
# parsed (trainer side: transformers PREFIX_CHECKPOINT_DIR).
_ckpt_step() {
  local b="${1##*/}" n
  case "$b" in checkpoint-*) n="${b#checkpoint-}" ;; *) return 1 ;; esac
  case "$n" in ''|*[!0-9]*) return 1 ;; esac
  printf '%s' "$n"
}

# _ckpt_quiescent <dir> — rc 0 iff no file under <dir> has been modified within
# the last JOBD_CKPT_SETTLE_S seconds (default 5; 0 disables the check). Cheap
# stand-in for "nobody is writing this any more".
_ckpt_quiescent() {
  local d="$1" settle="${JOBD_CKPT_SETTLE_S:-5}" now newest mt f
  [ -d "$d" ] || return 1
  case "$settle" in ''|*[!0-9]*) settle=5 ;; esac
  [ "$settle" -eq 0 ] && return 0
  now=$(date +%s 2>/dev/null || echo 0); newest=0
  while IFS= read -r f; do
    mt=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo "$now")
    [ "${mt:-0}" -gt "$newest" ] 2>/dev/null && newest="$mt"
  done < <(find "$d" -type f 2>/dev/null)
  [ "$newest" -gt 0 ] 2>/dev/null || return 1
  [ $(( now - newest )) -ge "$settle" ]
}

# _ckpt_write_complete <dir> — rc 0 iff the checkpoint dir is COMPLETE on disk.
#
# EVIDENCE (checked against transformers 5.13.0 — the version pinned by
# train-env/Dockerfile.base.t211 and every runsets/*/train.sh — not against
# folklore). `Trainer._save_checkpoint` in 5.13.0:
#   output_dir = os.path.join(run_dir, f"checkpoint-{global_step}")
#   self.save_model(output_dir, _internal_call=True)          # IN PLACE
# There is NO `tmp-checkpoint-<step>` staging dir and NO `os.rename` anywhere in
# 5.13.0's trainer.py (`grep`: 0 hits for each) — the 4.x staging-then-rename
# pattern was REMOVED upstream. So directory APPEARANCE is NOT a completion
# signal, and firing a sync on it would ship a torn checkpoint. Worse,
# `_save_rng_state` carries the comment "A process can arrive here before the
# process 0 has a chance to save the model, in which case output_dir may not yet
# exist" — under DDP a NON-ZERO rank can create the dir before rank 0 writes a
# single byte of the adapter.
#
# What IS ordered: rank 0 writes trainer_state.json (TRAINER_STATE_NAME) LAST —
# after save_model, _save_optimizer_and_scheduler, _save_scaler and
# _save_rng_state. That is the completion marker we use. The one write it does
# NOT order is a non-zero rank's rng_state_<i>.pth (each rank writes its own,
# unsynchronized), so the marker is paired with the quiescence window above.
_ckpt_write_complete() {
  local d="$1"
  [ -s "$d/trainer_state.json" ] || return 1
  _ckpt_quiescent "$d"
}

# _ckpt_b2_verified <local-dir> <b2-read-prefix> — rc 0 iff B2, READ BACK, holds
# exactly this directory: the same relative file-name set and the same total byte
# count. Two reads (`lsf -R` for names, `size --json` for bytes), both against the
# READ remote, so the answer is independent of which transport (b2x or rclone)
# did the push and of that push's exit code.
#
# Why not sha256 EVERYTHING: the publish-verify path already skips read-back above
# JOBD_PUBLISH_VERIFY_MAX_MB (64 MB) precisely because re-downloading model state
# to hash it costs more than it protects. A checkpoint is ~1 GB of
# optimizer/adapter state; hashing every one would re-download tens of GB per run.
# With an EXACT name-set match (not a superset) plus an exact total, a truncated
# remote file can only pass if some other file in the same directory GREW by the
# same amount — and these objects are write-once pushes of immutable local files,
# so nothing can grow.
#
# THIRD CHECK, added after adversarial review (2026-08-05): name set + byte total
# describe the SHAPE of a LoRA checkpoint, which is fixed by the run's config, not
# by its weights. Two different training lineages at the same step therefore
# verify against each other. Reachable in one place — two writers on one JOB_ID
# (a `job retarget` whose source box is still running, with HANDOFF_EPOCH unset,
# which makes the epoch fence a no-op). So when the local dir carries HF's
# trainer_state.json (a few KB, and its loss history uniquely identifies a
# lineage) its sha256 is read BACK from B2 and must match. Cheap next to a 1 GB
# directory, and it binds the verification to CONTENT rather than shape. A layout
# without that file (a non-HF checkpointer) falls back to name set + total.
#
# Any mismatch at all, and any check that cannot be performed => rc 1 => no delete.
_ckpt_b2_verified() {
  local ld="$1" pre="$2" ln lb rl rb lsha rsha
  [ -d "$ld" ] || return 1
  ln="$(cd "$ld" 2>/dev/null && find . -type f -printf '%P\n' 2>/dev/null | LC_ALL=C sort)"
  [ -n "$ln" ] || return 1                 # empty dir: nothing to verify -> refuse
  lb="$(cd "$ld" 2>/dev/null && find . -type f -printf '%s\n' 2>/dev/null \
        | awk '{s+=$1} END {print s+0}')"
  rl="$(rclone lsf -R --files-only "$pre/" 2>/dev/null | grep -v '/$' | LC_ALL=C sort)"
  if [ "$rl" != "$ln" ]; then
    local _nl _nr; _nl="$(printf '%s\n' "$ln" | grep -c .)"; _nr="$(printf '%s\n' "$rl" | grep -c .)"
    # Name the SUPERSET case separately: `rclone copy` never deletes, so a resume
    # at a different world size leaves e.g. rng_state_0..7 on B2 against a local
    # rng_state_0..3 and that dir can then NEVER be pruned. Correct (fail safe),
    # but an operator should not have to derive it from two counts.
    if [ "$_nr" -gt "$_nl" ] 2>/dev/null && [ -z "$(comm -23 <(printf '%s\n' "$ln") <(printf '%s\n' "$rl") 2>/dev/null)" ]; then
      log "checkpoint verify: remote is a strict SUPERSET for $ld vs $pre (local $_nl, remote $_nr — leftovers from an earlier attempt); prune WEDGED for this dir, NOT deleting"
    else
      log "checkpoint verify: name-set MISMATCH for $ld vs $pre (local $_nl files, remote $_nr) — NOT deleting"
    fi
    return 1
  fi
  rb="$(rclone size --json "$pre" 2>/dev/null \
        | sed -n 's/.*"bytes"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"
  # `-gt 0`, not just non-empty: against a real S3/B2 remote an ABSENT prefix is
  # not an error — `size --json` answers {"count":0,"bytes":0} and exits 0. The
  # name-set check above already catches that, but this keeps the size gate from
  # silently degrading to a no-op in production (both test shims model the absent
  # prefix as a non-zero exit, which real rclone does not).
  if [ -z "$rb" ] || [ "$rb" -le 0 ] 2>/dev/null; then
    log "checkpoint verify: size probe UNAVAILABLE or empty for $pre (got '${rb:-}') — NOT deleting"
    return 1
  fi
  if [ "$rb" != "$lb" ]; then
    log "checkpoint verify: byte-total MISMATCH for $ld vs $pre (local $lb, remote $rb) — NOT deleting"
    return 1
  fi
  if [ -s "$ld/trainer_state.json" ]; then
    lsha="$(sha256sum "$ld/trainer_state.json" 2>/dev/null | cut -d' ' -f1)"
    rsha="$(rclone cat "$pre/trainer_state.json" </dev/null 2>/dev/null \
            | sha256sum 2>/dev/null | cut -d' ' -f1)"
    if [ -z "$lsha" ] || [ -z "$rsha" ] || [ "$lsha" != "$rsha" ]; then
      log "checkpoint verify: trainer_state.json CONTENT mismatch/unreadable for $ld vs $pre (local ${lsha:0:12} remote ${rsha:0:12}) — NOT deleting"
      return 1
    fi
  fi
  return 0
}

# --- lever 4: PUBLISH-BY-MARKER (the atomicity seam) --------------------------
# B2/S3 has no atomic directory rename, so a multi-GB checkpoint reaches its
# final keys as N independent uploads and an eviction landing anywhere in that
# window leaves a directory that LOOKS like the newest checkpoint and is not one.
# Measured 2026-08-28, five times in one campaign (v16 rank ladder): a
# checkpoint-96 holding only trainer_state.json (2 objects, 57.7 KiB), a
# checkpoint-112 missing optimizer.pt, a checkpoint-176 at 5 objects / 2.6 GiB
# with the adapter absent. Each one was the numeric max, `--resume auto` selected
# it, and the run died. Nothing trained through a bad resume — the trainer's rc=3
# guard and transformers' own ValueError both fire — so this is an AVAILABILITY
# defect, and the fix must not weaken either guard.
#
# WHAT WE DO NOT DO: upload to a staging prefix and move on completion. A
# client-side `rclone move` on B2 is N copy+delete pairs and is itself
# interruptible, so it relocates the torn-directory window rather than closing
# it, and it would rewrite every path that keys on
# jobs/<id>/checkpoints/out/checkpoint-<N>/ (pull-back, prune verify, scrub,
# ckpt_retention, the published results glob).
#
# WHAT WE DO: leave the upload exactly where it is and make COMPLETION a separate,
# single, small, LAST write — one object, so it is as close to atomic as this
# substrate offers. It is published only after `_ckpt_b2_verified` has READ BACK
# the exact directory, which is the same evidence standard the delete gate uses.
#
# The marker is a SIBLING of the directory, not a member of it:
#     jobs/<id>/checkpoints/out/checkpoint-176/          <- the checkpoint
#     jobs/<id>/checkpoints/out/checkpoint-176.complete.json  <- the marker
# Inside the dir it would join the very name set and byte total that
# `_ckpt_b2_verified` compares against the LOCAL dir (which never holds it), and
# every subsequent verify would read a strict superset and wedge the prune.
# Beside it, nothing else in this file has to know it exists — and the resume
# pull-back's non-checkpoint leg brings it back for free.
#
# Its content is DETERMINISTIC (no timestamp): a resumed box re-pushes the file
# it pulled back to the same key, and an overwrite that changes bytes is an
# eventual-consistency window for no reason.
CKPT_COMPLETE_SUFFIX=".complete.json"

# _ckpt_marker_rel <checkpoint-rel> — the marker's path, relative to the run dir.
_ckpt_marker_rel() { printf '%s' "${1%/}$CKPT_COMPLETE_SUFFIX"; }

# _ckpt_names_complete — rc 0 iff the file names on stdin (one relative path per
# line) describe a RESUMABLE checkpoint. The legacy/no-marker fallback, and the
# ONE place the bash side spells the file-set contract; the python side spells
# the same one in train_proposer_lora.py `_checkpoint_completeness` and in
# ckpt_retention.py `incomplete_checkpoint_keys` (contract: CHECKPOINT_LIFECYCLE.md).
#
# Three requirements, each matching a shape an eviction actually produced:
#   trainer_state.json  — rank 0 writes it LAST, so its absence means torn.
#   optimizer + scheduler — transformers gates BOTH loads on one conjunction with
#     no else (v13-chain-v12); `.bnb_skipped` counterparts are OUR OWN deliberate
#     rename and excuse the absence.
#   weights — the checkpoint-176 shape had optimizer state and no adapter.
# Deliberately permissive about EXTRA files: a config we have not seen must read
# as complete, because every caller of this treats "not complete" as a reason to
# reach past the newest checkpoint.
_ckpt_names_complete() {
  awk '
    { n = $0; sub(/^.*\//, "", n); f[n] = 1 }
    END {
      if (!f["trainer_state.json"]) exit 1
      if (!(f["optimizer.pt"] || f["optimizer.bin"] \
            || f["optimizer.pt.bnb_skipped"] || f["optimizer.bin.bnb_skipped"])) exit 1
      if (!(f["scheduler.pt"] || f["scheduler.pt.bnb_skipped"])) exit 1
      if (!(f["adapter_model.safetensors"] || f["adapter_model.bin"] \
            || f["model.safetensors"] || f["pytorch_model.bin"] \
            || f["model.safetensors.index.json"] || f["pytorch_model.bin.index.json"])) exit 1
      exit 0
    }'
}

# _ckpt_marker_json <local-dir> <checkpoint-rel> — the marker body: the file list
# with per-file sizes, so a reader can check the claim instead of trusting it.
# Sorted, no timestamp => byte-identical on a re-push.
_ckpt_marker_json() {
  local d="$1" rel="$2" step body
  step="$(_ckpt_step "$rel")" || return 1
  body="$( (cd "$d" 2>/dev/null || exit 1
            find . -type f -printf '%P\t%s\n' 2>/dev/null | LC_ALL=C sort) \
           | awk -F'\t' -v step="$step" '
               function esc(s) { gsub(/\\/, "\\\\", s); gsub(/"/, "\\\"", s); return s }
               BEGIN { printf "{\"marker_version\": 1, \"step\": %d, \"files\": {", step }
               NF >= 2 { tot += $2; n += 1
                         if (n > 1) printf ", "
                         printf "\"%s\": %d", esc($1), $2 }
               END { printf "}, \"n_files\": %d, \"total_bytes\": %d}", n + 0, tot + 0
                     if (n + 0 == 0) exit 1 }' )" || return 1
  [ -n "$body" ] || return 1
  printf '%s\n' "$body"
}

# _ckpt_publish_marker <jobid> <run> <checkpoint-rel> — verify then publish.
# rc 1 on any refusal, and a refusal is always retryable: the next sync pass
# tries again, and until it succeeds the checkpoint is simply not yet published.
_ckpt_publish_marker() {
  local jobid="$1" run="$2" rel="$3" d="$run/$rel" body
  [ -d "$d" ] || return 1
  _ckpt_write_complete "$d" || return 1
  _ckpt_b2_verified "$d" "$B2/jobs/$jobid/checkpoints/$rel" || return 1
  body="$(_ckpt_marker_json "$d" "$rel")" || return 1
  printf '%s\n' "$body" \
    | rclone rcat "$B2W/jobs/$jobid/checkpoints/$(_ckpt_marker_rel "$rel")" 2>/dev/null
}

# _ckpt_mark_complete <jobid> <run> <matchlist> <markedfile> — publish a marker
# for every checkpoint dir this pass's glob expansion names that is complete on
# disk and not yet published. Must run BEFORE _ckpt_prune_synced: the marker is
# built from the LOCAL file list, which the prune is about to delete.
# JOBD_CKPT_MARK=0 disables (the resume side then falls back to the legacy
# file-set check, which is the pre-2026-08-28 behaviour).
CKPT_MARK_N=0; CKPT_MARK_LIST=""
_ckpt_mark_complete() {
  CKPT_MARK_N=0; CKPT_MARK_LIST=""
  [ "${JOBD_CKPT_MARK:-1}" = "1" ] || return 0
  local jobid="$1" run="$2" mf="$3" marked="$4" cands rel published
  cands="$(_ckpt_dirs_from_matchlist "$mf")"
  [ -n "$cands" ] || return 0
  [ -f "$marked" ] || : > "$marked"
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    grep -Fxq "$rel" "$marked" 2>/dev/null && continue
    _ckpt_safe_rel "$run" "$rel" || continue
    # Already published by an EARLIER BOX? Record and move on. A resumed box
    # pulls the checkpoint back with b2x's .b2x/state.json beside it, so the
    # local name set no longer equals the remote one and _ckpt_b2_verified would
    # refuse — correctly, and forever, logging a refusal every pass for a
    # checkpoint that was published hours ago on another box.
    published="$(rclone cat "$B2/jobs/$jobid/checkpoints/$(_ckpt_marker_rel "$rel")" \
                   </dev/null 2>/dev/null)"
    if [ -n "$published" ]; then
      printf '%s\n' "$rel" >> "$marked"
      continue
    fi
    if _ckpt_publish_marker "$jobid" "$run" "$rel"; then
      printf '%s\n' "$rel" >> "$marked"
      CKPT_MARK_N=$((CKPT_MARK_N + 1))
      CKPT_MARK_LIST="${CKPT_MARK_LIST:+$CKPT_MARK_LIST,}$rel"
      log "job $jobid: checkpoint $rel PUBLISHED — $(_ckpt_marker_rel "$rel") written last, after a read-back of the exact directory; --resume auto may now select it"
    else
      log "job $jobid: checkpoint $rel NOT published yet (B2 does not hold it exactly, or the marker write failed) — retried next pass; until then --resume auto reaches past it"
    fi
  done <<EOF
$cands
EOF
  return 0
}

# _ckpt_dirs_from_matchlist <file> — the checkpoint-<N> directories implied by a
# list of glob-matched RELATIVE file paths. Candidates come from the CHECKPOINT
# GLOB's expansion rather than a bare `find`, so a checkpoint-shaped directory
# that no `checkpoints:` glob selects is never a candidate. NOT a claim that the
# dir was shipped by THIS pass — a fire-on-arrival pass ships only its own new
# dirs while the prune still sees the whole expansion. The read-back in
# _ckpt_b2_verified is what proves durability; this only bounds the candidate set.
_ckpt_dirs_from_matchlist() {
  [ -s "$1" ] || return 0
  awk -F/ '{
    for (i = 1; i < NF; i++)
      if ($i ~ /^checkpoint-[0-9]+$/) {
        p = $1; for (j = 2; j <= i; j++) p = p "/" $j
        print p; break
      }
  }' "$1" | LC_ALL=C sort -u
}

# _ckpt_safe_rel <run> <rel> — rc 0 iff <rel> is a plain relative path naming a
# directory INSIDE <run>. Guards every recursive delete in this file.
_ckpt_safe_rel() {
  local run="$1" rel="$2"
  case "$rel" in ''|/*|.|..|*..*|*$'\n'*) return 1 ;; esac
  [ -d "$run/$rel" ] || return 1
  _ckpt_step "$rel" >/dev/null || return 1
  return 0
}

# _ckpt_du_bytes <dir>
_ckpt_du_bytes() {
  local k; k="$(du -sk "$1" 2>/dev/null | cut -f1)"
  case "$k" in ''|*[!0-9]*) k=0 ;; esac
  printf '%s' $(( k * 1024 ))
}

# --- bounded resume pull-back -------------------------------------------------
# _ckpt_latest_remote <b2-prefix> — echo the newest `checkpoint-N` path segment
# under the prefix (numeric max on N), or nothing if it cannot tell.
#
# WHY THIS EXISTS. The resume pull-back used to fetch the ENTIRE remote ladder.
# The trainer's own `--save-total-limit` bounds the ladder on the BOX DURING a
# run, but it does not touch B2 and the pull-back ignored it — so resume disk
# need grew LINEARLY with run length, and a long run became unresumable on the
# exact disk size that had been running it for hours.
#
# MEASURED 2026-08-07, in production: v9-gemma4 was evicted at step 135 of 156.
# Its replacement pulled back 27 checkpoints x ~1.55 GB = 41 GB of a 50 GB disk,
# leaving 9 GB against the 28 GB of assets it still needed. The job was reaped
# `INSUFFICIENT DISK` and the run stalled with every byte of its state intact.
# Nothing was corrupt and nothing was lost — it simply could not fit itself.
#
# Only the newest COMPLETE checkpoint is ever read: the trainer resumes from
# `_last_checkpoint_dir()`, which since 2026-08-28 walks step-descending and takes
# the first complete one. Every other is pure transfer cost and pure disk.
# JOBD_CKPT_PULL_KEEP raises the count if more than one is wanted.
#
# COMPLETENESS, not numeric max (2026-08-28). Bounding the pull-back to newest-1
# on 2026-08-07 silently retired the property CHECKPOINT_LIFECYCLE.md's newest-2
# floor exists to preserve — "the newest can be a partial upload from a box that
# died mid-push; HF resume validation then falls back to the complete one". With
# one dir pulled there IS no complete one to fall back to, which is how five v16
# restarts in one day cost ~16 steps each. Raising the count back to 2 would
# re-open the 41-GB disk incident above; asking B2 which dir is complete costs one
# extra LIST and pulls the RIGHT single directory.
#
# Two evidence tiers, in order — the same contract the trainer applies locally
# (CHECKPOINT_LIFECYCLE.md §"The completion contract"):
#   1. the sibling `<name>.complete.json` marker, written last by
#      _ckpt_mark_complete after a read-back;
#   2. legacy/no marker: the remote file set itself, via _ckpt_names_complete.
# FAIL-OPEN is preserved exactly: if NOTHING under the prefix reads as complete,
# fall back to the historical numeric max and say so. A pull-back that fetches
# too little is the expensive failure, and this must never introduce it.
#
# NOTE the sort: `sort -V`, never plain sort. Alphabetically `checkpoint-100`
# precedes `checkpoint-60`, which is how a "we only have checkpoint-90" false
# report was produced here on 2026-08-06.
_ckpt_latest_remote() {
  local prefix="$1" out dirs marks want got=0 res="" d
  out="$(rclone lsf --dirs-only "$prefix" 2>/dev/null)" || return 1
  [ -n "$out" ] || return 1
  dirs="$(printf '%s\n' "$out" | tr -d '/' | grep -E '^checkpoint-[0-9]+$' | sort -V)"
  [ -n "$dirs" ] || return 1
  want="${JOBD_CKPT_PULL_KEEP:-1}"
  case "$want" in ''|*[!0-9]*) want=1 ;; esac
  [ "$want" -lt 1 ] && want=1
  # One extra non-recursive LIST for the whole prefix, not one per candidate.
  marks="$(rclone lsf --files-only "$prefix" 2>/dev/null)"
  for d in $(printf '%s\n' "$dirs" | sort -Vr); do
    [ "$got" -ge "$want" ] && break
    if printf '%s\n' "$marks" | grep -Fxq "$d$CKPT_COMPLETE_SUFFIX"; then
      res="${res:+$res }$d"; got=$((got + 1)); continue
    fi
    if rclone lsf -R --files-only "$prefix$d/" 2>/dev/null | _ckpt_names_complete; then
      res="${res:+$res }$d"; got=$((got + 1)); continue
    fi
    log "checkpoint pull-back: SKIPPING remote $d — no completion marker and its remote file set is not resumable (a torn push from an evicted box); reaching past it"
  done
  if [ "$got" -eq 0 ]; then
    log "checkpoint pull-back: NO checkpoint under $prefix reads as complete — falling back to the numeric max (fail-open, unchanged behaviour). The trainer's own resume guard is the gate that must stop a bad resume."
    printf '%s\n' "$dirs" | tail -n "$want"
    return 0
  fi
  # UNQUOTED on purpose: `res` is a space-joined list of checkpoint-<N> names
  # (a name that could contain a space is not one this function ever produces).
  printf '%s\n' $res
}

# _ckpt_restore_complete <local-dir> <b2-read-prefix> — rc 0 iff every object B2
# holds under <prefix> landed locally as a NON-EMPTY file. The RESTORE-side
# counterpart to _ckpt_b2_verified, and it deliberately inverts two of that
# function's rules.
#
# WHY IT EXISTS (v13-chain-v12, 2026-08-17). A run resumed from checkpoint-112
# whose materialized copy was missing optimizer.pt and/or scheduler.pt.
# transformers 5.13.0 gates BOTH loads on one `optimizer.pt exists AND
# scheduler.pt exists` conjunction with NO else: if either is absent both loads
# are skipped SILENTLY — no warning, no log line, no return value — while
# trainer_state.json and the PEFT adapter load on separate unconditional paths.
# So global_step, the full log_history and the weights all come back and the
# resume LOOKS healthy. Measured damage over the 70 steps after the boundary: LR
# restarted from peak and the AdamW moments were rebuilt from zero, 4.15x the
# intended integrated LR for that segment. Which leg dropped the file (an
# incomplete push from the dying box, or this pull-back) is UNVERIFIED; this
# closes the READ half. Deletion was verified (_ckpt_b2_verified, prune+scrub)
# and restoration was not — that asymmetry was the whole hole.
#
# INVERSION 1 — SUBSET, not exact name set. Extra LOCAL files are expected and
# fine: b2x leaves .b2x/state.json (and .b2x-partial-* after a failed attempt)
# in the destination dir, and a same-box resume can still hold rng_state_<i>
# from a wider world size. Restoration only cares that everything remote
# ARRIVED, so an exact-name-set test here would be a permanent false alarm.
# INVERSION 2 — unverifiable is NOT a failure. The block header's "fail safe,
# never fail open" governs DELETES, where a refusal costs nothing. Nothing is
# deleted here: an unreadable listing returns 0 with a log line, because turning
# a LIST hiccup into an extra multi-GB pull on every rented box is a bigger loss
# than the miss it would catch.
#
# Presence + non-empty, NOT bytes: both transports write to a temp name and
# rename on completion (b2x promotes .b2x-partial-* only after every part lands;
# rclone's local backend is not --inplace), so a torn FINAL file is not a shape
# either of them produces. The shape that actually bit us is an absent one.
_ckpt_restore_complete() {
  local ld="$1" pre="$2" rl f miss=0 nr=0 shown=""
  [ -d "$ld" ] || { log "restore verify: $ld does not exist — nothing was restored"; return 1; }
  # Hard-timeout the listing, same reason as the disk-size probe: a wedged LIST
  # must not park a rented box before its entrypoint has started.
  rl="$(timeout "${JOBD_CKPT_VERIFY_TIMEOUT_S:-120}" \
        rclone lsf -R --files-only "$pre/" 2>/dev/null | grep -v '/$')"
  if [ -z "$rl" ]; then
    log "restore verify: remote listing UNAVAILABLE or empty for $pre — restore NOT verified (continuing)"
    return 0
  fi
  nr="$(printf '%s\n' "$rl" | grep -c .)"
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -s "$ld/$f" ] && continue
    miss=$((miss + 1))
    [ "$miss" -le 6 ] && shown="${shown:+$shown }$f"
  done <<EOF
$rl
EOF
  if [ "$miss" -gt 0 ]; then
    log "restore verify: $miss of $nr remote file(s) MISSING or EMPTY in $ld (from $pre): $shown"
    return 1
  fi
  log "restore verify: all $nr file(s) under $pre present in $ld"
  return 0
}

# --- lever 1: delete-after-sync ----------------------------------------------
# _ckpt_prune_synced <jobid> <run> <matchlist> — prune local checkpoint dirs that
# B2 verifiably already holds, KEEPING the newest JOBD_CKPT_KEEP (>=2) per layout
# root. Results land in CKPT_PRUNE_N / CKPT_PRUNE_BYTES / CKPT_PRUNE_LIST (stdout
# stays free for the log).
#
# Newest-2 is a HARD FLOOR, not a default: onstart/resume_pull.sh pulls the newest
# TWO checkpoint dirs per root precisely because "the newest can be a partial
# upload from a box that died mid-push; HF resume validation then falls back to
# the complete one". Keeping fewer than two locally would destroy the same
# survivability on the box side.
CKPT_PRUNE_N=0; CKPT_PRUNE_BYTES=0; CKPT_PRUNE_LIST=""
_ckpt_prune_synced() {
  CKPT_PRUNE_N=0; CKPT_PRUNE_BYTES=0; CKPT_PRUNE_LIST=""
  [ "${JOBD_CKPT_PRUNE:-1}" = "1" ] || return 0
  local jobid="$1" run="$2" mf="$3"
  local keep="${JOBD_CKPT_KEEP:-2}"
  case "$keep" in ''|*[!0-9]*) keep=2 ;; esac
  [ "$keep" -lt 2 ] && keep=2
  # TWO-WRITER FENCE (HANDOFF_DESIGN §4), re-checked immediately before any
  # delete: a superseded box must not prune on the strength of a prefix a newer
  # handoff epoch now owns. No-op (and no B2 read) off the handoff path.
  if _handoff_epoch_stale "$jobid"; then
    log "job $jobid: checkpoint prune REFUSED — handoff epoch $HANDOFF_EPOCH stale (a newer epoch owns jobs/$jobid)"
    return 0
  fi
  local cands victims rel d sz st par
  cands="$(_ckpt_dirs_from_matchlist "$mf")"
  [ -n "$cands" ] || return 0
  # Order by (layout root, step DESC) and drop the first $keep of each root.
  # STEP NUMBER, never mtime: a resume re-touches old dirs and would reorder them.
  victims="$(while IFS= read -r rel; do
               [ -n "$rel" ] || continue
               st="$(_ckpt_step "$rel")" || continue
               case "$rel" in */*) par="${rel%/*}" ;; *) par="." ;; esac
               printf '%s\t%s\t%s\n' "$par" "$st" "$rel"
             done <<EOF
$cands
EOF
             )"
  # NB $'\t' (bash ANSI-C quoting), never "$(printf '\t')" — command substitution
  # strips trailing whitespace and would hand `sort` an EMPTY delimiter.
  victims="$(printf '%s\n' "$victims" | LC_ALL=C sort -t$'\t' -k1,1 -k2,2nr \
             | awk -F'\t' -v keep="$keep" 'NF==3 { if (++c[$1] > keep) print $3 }')"
  [ -n "$victims" ] || return 0
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    _ckpt_safe_rel "$run" "$rel" || continue
    d="$run/$rel"
    if ! _ckpt_quiescent "$d"; then
      log "job $jobid: checkpoint prune SKIP $rel — still being written (settle window)"
      continue
    fi
    if ! _ckpt_b2_verified "$d" "$B2/jobs/$jobid/checkpoints/$rel"; then
      log "job $jobid: checkpoint prune SKIP $rel — NOT verified at jobs/$jobid/checkpoints/$rel; keeping the local copy"
      continue
    fi
    sz="$(_ckpt_du_bytes "$d")"
    if rm -rf "$d" 2>/dev/null; then
      CKPT_PRUNE_N=$((CKPT_PRUNE_N + 1)); CKPT_PRUNE_BYTES=$((CKPT_PRUNE_BYTES + sz))
      CKPT_PRUNE_LIST="${CKPT_PRUNE_LIST:+$CKPT_PRUNE_LIST,}$rel"
      log "job $jobid: checkpoint prune DELETED $rel (${sz}B) — name set + total bytes read back from jobs/$jobid/checkpoints/$rel; newest-$keep per root retained"
    else
      log "job $jobid: checkpoint prune FAILED to remove $rel (rm error) — left in place"
    fi
  done <<EOF
$victims
EOF
  return 0
}

# --- lever 2: end-of-run scrub ------------------------------------------------
# _ckpt_scrub_local <jobid> <run> <wdir> — after a run is finished AND its results
# are on B2, remove EVERY local checkpoint dir this job produced. Results land in
# CKPT_SCRUB_N / CKPT_SCRUB_BYTES / CKPT_SCRUB_LIST.
#
# ORDERING IS THE WHOLE POINT. The `results:` glob of every training bundle is
# `out/**`, which CONTAINS the checkpoint dirs — so a scrub running before the
# results push finished would delete the bytes being uploaded. This is therefore
# called exactly once, from the very end of run_job_body: after the results
# publish, after publish-verify, after the manifest / log.txt / results.DONE.json
# writes. The caller refuses to call it at all when publish-verify FAILED or when
# nothing was uploaded — though note that publish-verify SKIPS files over
# JOBD_PUBLISH_VERIFY_MAX_MB (64 MB), which is every file in a LoRA checkpoint, so
# that gate covers the small results and the READ-BACK BELOW is the only gate that
# covers the bytes actually being deleted.
#
# No newest-2 exemption here: the job is TERMINAL, and nothing local survives a
# re-claim anyway (`poll_once` rm -rf's the whole workdir, run_job_body rm -rf's
# the run dir, and the resume pull-back re-reads jobs/<id>/checkpoints/ off B2).
CKPT_SCRUB_N=0; CKPT_SCRUB_BYTES=0; CKPT_SCRUB_LIST=""
_ckpt_scrub_local() {
  CKPT_SCRUB_N=0; CKPT_SCRUB_BYTES=0; CKPT_SCRUB_LIST=""
  [ "${JOBD_CKPT_SCRUB:-1}" = "1" ] || return 0
  local jobid="$1" run="$2" wdir="$3"
  if _handoff_epoch_stale "$jobid"; then
    log "job $jobid: checkpoint scrub REFUSED — handoff epoch $HANDOFF_EPOCH stale"
    return 0
  fi
  local ml="$wdir/.ckpt_scrub.list" cands rel d sz
  cat "$wdir/.uploaded" "$wdir/.checkpoint.matched" 2>/dev/null > "$ml"
  cands="$(_ckpt_dirs_from_matchlist "$ml")"
  rm -f "$ml" 2>/dev/null || true
  [ -n "$cands" ] || return 0
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    _ckpt_safe_rel "$run" "$rel" || continue
    d="$run/$rel"
    # `wait $epid` returns when timeout's direct child exits — orphaned DDP ranks
    # and an in-flight checkpoint rclone can outlive it (the publish-verify block
    # says so out loud). Quiescence costs one stat pass and closes that window.
    if ! _ckpt_quiescent "$d"; then
      log "job $jobid: checkpoint scrub SKIP $rel — still being written after the entrypoint exited (orphaned rank or in-flight push)"
      continue
    fi
    # EITHER durable prefix is sufficient evidence: the mid-run sync writes
    # jobs/<id>/checkpoints/, the finalize publish writes jobs/<id>/results/.
    # (Resume reads only checkpoints/ — but see above: nothing local survives a
    # re-claim, so a results/-only dir costs a re-download, never data.)
    if ! _ckpt_b2_verified "$d" "$B2/jobs/$jobid/checkpoints/$rel" \
       && ! _ckpt_b2_verified "$d" "$B2/jobs/$jobid/results/$rel"; then
      log "job $jobid: checkpoint scrub SKIP $rel — not verified under checkpoints/ or results/; keeping the local copy"
      continue
    fi
    sz="$(_ckpt_du_bytes "$d")"
    if rm -rf "$d" 2>/dev/null; then
      CKPT_SCRUB_N=$((CKPT_SCRUB_N + 1)); CKPT_SCRUB_BYTES=$((CKPT_SCRUB_BYTES + sz))
      CKPT_SCRUB_LIST="${CKPT_SCRUB_LIST:+$CKPT_SCRUB_LIST,}$rel"
      log "job $jobid: checkpoint scrub DELETED $rel (${sz}B) — run finished, results published + verified, dir read back from B2"
    else
      log "job $jobid: checkpoint scrub FAILED to remove $rel (rm error) — left in place"
    fi
  done <<EOF
$cands
EOF
  return 0
}

# --- lever 3: fire the sync as soon as a checkpoint lands ---------------------
# _ckpt_new_ready <run> <seenfile> — echo the relative paths of checkpoint dirs
# that are COMPLETE on disk (see _ckpt_write_complete) and not yet in <seenfile>.
# `-prune` keeps the poll to a directory-name scan: it never descends INTO a
# checkpoint dir, and the per-file walk happens only for a dir we have not seen.
_ckpt_new_ready() {
  local run="$1" seen="$2" d rel
  [ -d "$run" ] || return 0
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    rel="${d#"$run"/}"          # quoted: $run is a PATH, never a glob pattern
    [ "$rel" = "$d" ] && continue
    grep -Fxq "$rel" "$seen" 2>/dev/null && continue
    _ckpt_write_complete "$d" || continue
    printf '%s\n' "$rel"
  done < <(find "$run" -type d -name 'checkpoint-[0-9]*' -prune -print 2>/dev/null)
}

# _ckpt_all_dirs <run> — every checkpoint-<N> dir under <run>, relative, whether
# or not it is complete. Used to SEED the fast path's seen-set.
_ckpt_all_dirs() {
  local run="$1" d rel
  [ -d "$run" ] || return 0
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    rel="${d#"$run"/}"
    [ "$rel" = "$d" ] && continue
    printf '%s\n' "$rel"
  done < <(find "$run" -type d -name 'checkpoint-[0-9]*' -prune -print 2>/dev/null)
}
# <<< CKPT_LIFECYCLE_END

# --- ensure zstd (bundles are tar.zst; jobd.py needs it to extract) -----------
if ! command -v zstd >/dev/null 2>&1 && ! "$PY" -c 'import zstandard' >/dev/null 2>&1; then
  log "zstd absent — installing"
  { timeout 120 apt-get update -qq >/dev/null 2>&1 || true; \
    timeout 180 apt-get install -y -qq zstd; } >/dev/null 2>&1 \
    || timeout 180 "$PY" -m pip install -q zstandard >/dev/null 2>&1 \
    || log "!! could not install zstd/zstandard — extraction will fail"
fi

# --- python-half health (FAILCLOSED_DESIGN) -----------------------------------
# jobd has two halves that fail INDEPENDENTLY: a bash/rclone half (poll, beacon,
# downloads, checkpoint pushes) and a python half (`$PY $JH` -> jobd.py ->
# jobmeta: every lifecycle EVENT, plus ticket parse and bundle extract). Box
# 47737955 (2026-08-13) ran 52 minutes with the first half perfectly healthy and
# the second 100% dead, and reported itself IDLE the whole time, because all ~13
# python call sites are `>/dev/null 2>&1 || true`.
#
# The rule (FAILCLOSED_DESIGN §2): telemetry may not kill a JOB; telemetry
# failure must end the box's right to BILL. Unobservable spend is waste by
# default. Three states, and the discriminator is the exit code, not a timer:
#
#   ok      — last python call succeeded.
#   broken  — CAPABILITY fault. rc==3 (jobd.py EXIT_STRUCTURAL: the interpreter
#             cannot import its own modules) is proof in ONE call, so it
#             escalates immediately with no counter. This class never recovers;
#             every subsequent call fails identically.
#   broken  — via the counter. Any OTHER nonzero rc is presumed TRANSPORT (a B2
#             500, a rate limit, a network blip) and is forgiven, but no longer
#             silently: PY_FAIL_STREAK consecutive failures spanning at least
#             PY_FAIL_WINDOW_S seconds is no longer a blip. Both conditions must
#             hold, so a fast retry burst cannot escalate.
#
# A single success anywhere resets the streak — the counter measures a standing
# condition, not a lifetime tally.
PY_HALF=ok                  # ok | broken
PY_HALF_REASON=""
PY_FAIL_STREAK=0
PY_FAIL_FIRST_TS=0
PY_BROKEN_TS=0              # epoch we first declared broken (arms the park clock)
PY_FAIL_STREAK_MAX="${JOBD_PY_FAIL_STREAK:-5}"
PY_FAIL_WINDOW_S="${JOBD_PY_FAIL_WINDOW_S:-600}"
# How long a box may keep billing after the python half is declared broken and
# it holds no running job. Short by design: this is the hole that cost $1.742 —
# a box that has never claimed work and cannot emit has nothing to protect, so
# it should die fast and cheap. A running job is NOT interrupted (see
# maybe_idle_park): mid-run the cost asymmetry inverts.
PY_BROKEN_PARK_S="${JOBD_PY_BROKEN_PARK_S:-300}"
JOBD_SELFTEST="${JOBD_SELFTEST:-1}"        # 0 disables the boot capability gate

# Deliberately NOT persisted across daemon restarts: a fresh boot re-runs the
# capability selftest, which is the authoritative answer. Keeping a stale
# breadcrumb would let one bad bundle condemn a box that has since been fixed
# (and would fire on a spot resume — see FAILCLOSED_DESIGN §7).
PY_BROKEN_FLAG="$STATE_DIR/.pyhalf_broken"
rm -f "$PY_BROKEN_FLAG" 2>/dev/null || true
_py_broken() {   # _py_broken <reason> — idempotent latch + loud, beaconed report
  [ "$PY_HALF" = "broken" ] && return 0
  PY_HALF=broken; PY_HALF_REASON="$1"; PY_BROKEN_TS="$(date +%s)"
  # CROSS-PROCESS LATCH. The heartbeat and checkpoint emitters run inside
  # backgrounded `( ... ) &` subshells, so a variable they set dies with them —
  # the daemon would never learn what its own watchers discovered. Drop a
  # breadcrumb the poll loop latches from (_py_breadcrumb_check). Local file, no
  # network: this path must not depend on anything that can be broken.
  [ -n "$PY_BROKEN_FLAG" ] && printf '%s\t%s\n' "$PY_BROKEN_TS" "$1" \
    > "$PY_BROKEN_FLAG" 2>/dev/null || true
  log "!! PYTHON HALF BROKEN: $1 — refusing to claim new work; no lifecycle"\
" events can be emitted from this box. Reporting on the rclone beacon."
  # The report MUST travel on a channel strictly below the one that failed
  # (FAILCLOSED_DESIGN §5). jobd_status is pure bash + rclone rcat and is the
  # half that demonstrably kept working through the incident.
  status_marker
}

pyhalf_selftest() {
  # THE BOOT CAPABILITY GATE. Runs before the box advertises itself as able to
  # take work. Offline and pure (jobd.py selftest touches no network), so the
  # only thing it can be red about is a fault that will never heal — which is
  # what licenses it to fail closed. Costs one interpreter start.
  [ "$JOBD_SELFTEST" = "1" ] || { log "python-half selftest DISABLED (JOBD_SELFTEST=0)"; return 0; }
  local out rc=0
  out="$("$PY" "$JH" selftest --instance-id "$IID" 2>&1)" || rc=$?
  if [ "$rc" = "0" ]; then
    log "python-half selftest OK"
    return 0
  fi
  _py_broken "boot selftest rc=$rc: $(printf '%s' "$out" | tr '\n\t' '  ' | tail -c 240)"
  return 1
}

beacon_tick() {
  # THE BEACON MUST BE PERIODIC. status_marker is otherwise event-driven only
  # (boot, spawn, reap, end-of-staging), which is a documented weakness in
  # herdd._jobd_heartbeat_epoch_soft and was half of two false ZOMBIE_NO_JOBD
  # alarms on 2026-08-07. On an IDLE box it means the marker is a BOOT STAMP
  # that ages forever, so "healthy and waiting for work" and "dead" produce the
  # identical observation. That ambiguity is why 47737955's genuine staleness
  # could not be acted on: the same signal is stale on every healthy idle box
  # too. A real heartbeat is the precondition for anyone downstream having teeth
  # (FAILCLOSED_DESIGN §8), not an optimisation.
  local now
  now="$(date +%s)"
  [ $(( now - LAST_STATUS_TS )) -ge "$STATUS_EVERY_S" ] || return 0
  status_marker
}

_py_breadcrumb_check() {   # latch a subshell's verdict into the daemon's state
  [ "$PY_HALF" = "broken" ] && return 0
  [ -n "$PY_BROKEN_FLAG" ] && [ -f "$PY_BROKEN_FLAG" ] || return 0
  local ts reason
  IFS=$'\t' read -r ts reason < "$PY_BROKEN_FLAG" 2>/dev/null || return 0
  PY_HALF=broken
  PY_HALF_REASON="${reason:-reported by a jobd watcher subshell}"
  case "$ts" in (*[!0-9]*|"") ts="$(date +%s)";; esac
  PY_BROKEN_TS="$ts"
  log "!! python half latched BROKEN from a watcher subshell: $PY_HALF_REASON"
  status_marker
}

_py_account() {   # _py_account <rc> — fold one python call's exit code into the health state
  local rc="$1" now
  if [ "$rc" = "0" ]; then
    PY_FAIL_STREAK=0; PY_FAIL_FIRST_TS=0
    return 0
  fi
  # rc 3 == jobd.py EXIT_STRUCTURAL: proven capability fault, no counter needed.
  if [ "$rc" = "3" ]; then
    _py_broken "jobd.py exited EXIT_STRUCTURAL (rc=3): python half cannot import its modules"
    return 0
  fi
  now="$(date +%s)"
  PY_FAIL_STREAK=$(( PY_FAIL_STREAK + 1 ))
  [ "$PY_FAIL_STREAK" = "1" ] && PY_FAIL_FIRST_TS="$now"
  if [ "$PY_FAIL_STREAK" -ge "$PY_FAIL_STREAK_MAX" ] \
     && [ $(( now - PY_FAIL_FIRST_TS )) -ge "$PY_FAIL_WINDOW_S" ]; then
    _py_broken "$PY_FAIL_STREAK consecutive python failures over $(( now - PY_FAIL_FIRST_TS ))s (last rc=$rc)"
  fi
  return 0
}

# _py_call <kind> <jobd.py args...> — the accounted replacement for the bare
# `"$PY" "$JH" ... || true`. Still never propagates a failure to the caller (the
# daemon must not die of a failed emit); the difference is that the failure is
# now COUNTED and can escalate. `kind` picks the accounting policy:
#
#   beacon — pure-observability calls whose silence IS the loss of observability
#            (emit, emit-box, heartbeat, checkpoint). These escalate.
#   report — calls that are THEMSELVES failure reports (stall_suspected,
#            checkpoint_sync_failed). Counted for the streak, but a failure to
#            report a failure must not compound into a second incident, so these
#            never arm the latch on their own; they only extend an existing
#            streak that a `beacon` call started.
#   dying  — the preempt-trap emits. The box is already going down under a hard
#            `timeout`; escalation is meaningless there and an extra beacon write
#            could stall a shutdown. Explicitly unaccounted.
_py_call() {
  local kind="$1"; shift
  local rc=0
  "$PY" "$JH" "$@" >/dev/null 2>&1 || rc=$?
  case "$kind" in
    beacon) _py_account "$rc" ;;
    report) [ "$rc" = "0" ] && { PY_FAIL_STREAK=0; PY_FAIL_FIRST_TS=0; } ;;
    dying)  : ;;
  esac
  return 0
}

LAST_STATUS_TS=0
# Beacon cadence. Well under GUARD_JOBD_STALE_S (600s) so a healthy idle box is
# never mistaken for a dead one, and cheap: one small rcat per interval.
STATUS_EVERY_S="${JOBD_STATUS_EVERY_S:-120}"
jobd_status() {   # coarse per-box heartbeat marker, SERVE_STATUS-shaped
  # `pyhalf=` is the field the incident needed and did not have: this marker was
  # the signal that LIED (it said IDLE while the python half was dead). It rides
  # at the END of the line — herdd._jobd_status_hb_epoch scans for the first
  # %FT%TZ-shaped token, so appending fields is format-safe for every reader.
  LAST_STATUS_TS="$(date +%s)"
  echo "$1 $(date -u +%FT%TZ)${2:+ $2} pyhalf=$PY_HALF${PY_HALF_REASON:+ pyreason=$(printf '%s' "$PY_HALF_REASON" | tr ' ' '_')}" \
    | rclone rcat "$B2W/jobs/nodes/$IID/JOBD_STATUS" 2>/dev/null || true
}

emit() {   # emit <job_id> <event> [extra jobd.py emit args...]
  local jid="$1" event="$2"; shift 2
  # exp_id/arm echo (matrix audit seam): JOB_EXP_ID/JOB_ARM are in dynamic
  # scope wherever a ticket has been prepared; empty/unset on plain jobs and on
  # the pre-parse paths. Values are validated slugs (no whitespace), so the
  # unquoted ${:+} expansion cannot mis-split.
  _py_call beacon emit "$jid" "$event" --instance-id "$IID" \
    ${JOB_EXP_ID:+--field "exp_id=$JOB_EXP_ID"} ${JOB_ARM:+--field "arm=$JOB_ARM"} \
    "$@"
}

# emit_report — same wire call as emit(), `report` accounting. For events that
# are themselves failure reports; see _py_call's kind table.
emit_report() {
  local jid="$1" event="$2"; shift 2
  _py_call report emit "$jid" "$event" --instance-id "$IID" \
    ${JOB_EXP_ID:+--field "exp_id=$JOB_EXP_ID"} ${JOB_ARM:+--field "arm=$JOB_ARM"} \
    "$@"
}

mark_terminal() {   # local terminal cache: this box never reconsiders the job
  echo "$2 $(date -u +%FT%TZ)" > "$STATE_DIR/$1.terminal" 2>/dev/null || true
}

# kill_tree <pid> [signal] — signal a pid AND its whole descendant tree (children
# first). Used by the cancel-watch to actually stop a running entrypoint's process
# tree (a trainer that spawned workers, etc.). PPID-recursive rather than a
# process-group kill so it can never catch the daemon's own group — the runner
# subshell is a background job in the daemon's group, not its own leader. `ps`
# fallback where pgrep is absent.
kill_tree() {
  local pid="$1" sig="${2:-TERM}" c
  for c in $(pgrep -P "$pid" 2>/dev/null || ps -o pid= --ppid "$pid" 2>/dev/null); do
    kill_tree "$c" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null || true
}
CANCEL_POLL="${JOBD_CANCEL_POLL:-15}"   # how often a running job checks its CANCEL marker

# >>> FLUSHNOW_BEGIN (do not remove: test_jobd_flush_now.py sources this block)
# CHECKPOINT_NOW — operator-initiated checkpoint flush. `herdd job flush <id>`
# rcats a marker at jobs/<id>/CHECKPOINT_NOW; this is the box half. It rides the
# CANCEL poll (same subshell, same lsf cadence) and drops a breadcrumb the
# checkpoint-sync loop picks up on its next watch tick, so the flush lands within
# ~CANCEL_POLL + JOBD_CKPT_WATCH_S of the write.
#
# WHAT IT IS FOR: a pre-park / pre-handoff flush — you are about to stop a box and
# want the newest bytes on B2 without waiting out JOB_CHECKPOINT_S.
# WHAT IT CANNOT DO: rescue an eviction. vast delivers no SIGTERM on a spot
# reclaim and the warning budget is single-digit seconds (measured 2026-08-26), so
# nothing that must first be NOTICED over B2 can run inside it. The preempt trap's
# final flush is the eviction path; this is not.
# It never signals the entrypoint — a flush is not a stop.
#
# CONSUME-AND-DELETE, at most once. B2 has no CAS, so a lost delete costs one
# extra flush and never a missed one. (The "nothing ever deletes anything on B2"
# rule in the checkpoint-lifecycle section is about JOB STATE; this deletes the
# operator's own trigger object and nothing else.)
_flush_marker_consume() {   # <jobid> <breadcrumb> -> 0 iff a flush was requested
  local jobid="$1" crumb="$2"
  rclone lsf "$B2/jobs/$jobid/CHECKPOINT_NOW" 2>/dev/null | grep -q . || return 1
  : > "$crumb" 2>/dev/null || true
  rclone deletefile "$B2W/jobs/$jobid/CHECKPOINT_NOW" 2>/dev/null \
    || log "job $jobid: CHECKPOINT_NOW consumed but the marker DELETE failed — a later poll may flush again (harmless)"
  log "job $jobid: CHECKPOINT_NOW seen — requesting an unfiltered checkpoint flush"
  return 0
}
# <<< FLUSHNOW_END

emit_box() {   # emit_box <event> [k=v ...] — per-box lifecycle stream (jobs/nodes/<IID>/events/)
  local event="$1"; shift
  local -a ff=(); local kv
  for kv in "$@"; do [ -n "$kv" ] && ff+=(--field "$kv"); done
  _py_call beacon emit-box "$IID" "$event" "${ff[@]}"
}

# --- GPU inventory (once at boot) ----------------------------------------------
# GPU_IDS[i] = nvidia device index, GPU_MEM[i] = whole GB. JOBD_SKIP_GPU (or no
# nvidia-smi) presents 0 cards; JOBD_FAKE_GPUS="0:32,1:32" fakes an inventory so
# the scheduler is testable without hardware.
#
# VRAM is ROUNDED TO NEAREST GB, not truncated. nvidia-smi reports MiB against a
# card marketed in decimal GB, so every round-number card lands just under its
# own nameplate: a "32 GB" RTX 5090 reports 32607 MiB = 31.84 GiB, a 24 GB 3090
# reports 24564 = 23.99, a 96 GB PRO 6000 reports 97887 = 95.59. Truncating made
# `31 -ge 32` false, so **no 32 GB card could ever satisfy `gpu_ram_gb: 32`** —
# and because an unschedulable ticket hits the strict-FIFO `break` below, it did
# so SILENTLY: the ticket stayed `submitted` forever while the box billed.
# (Live incident 2026-07-31, box 46347213: the v7 pair sat 13 min at $0.44/hr,
# jobd healthy and polling, no log line, no event.) Operators write the
# nameplate number, so round-to-nearest is what `gpu_ram_gb: 32` MEANS.
#
# JOBD_GPU_ALLOW="0,1" RESTRICTS the inventory to those device indices. Unlike
# JOBD_FAKE_GPUS this is not a fake — the indices and their probed VRAM are real;
# it is a scheduling ALLOWANCE, so a box (or, in the LOCAL GPU LANE, a
# workstation) can hand jobd a subset of its cards and keep the rest. Unset =>
# every probed card, i.e. today's behavior. See LOCAL_GPU_LANE.md.
GPU_IDS=(); GPU_MEM=()
if [ -n "${JOBD_FAKE_GPUS:-}" ]; then
  IFS=',' read -ra _fg <<< "$JOBD_FAKE_GPUS"
  for _g in "${_fg[@]}"; do
    GPU_IDS+=("${_g%%:*}"); GPU_MEM+=("${_g##*:}")
  done
elif [ "${JOBD_SKIP_GPU:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  while IFS=',' read -r _idx _mib; do
    _idx="${_idx// /}"; _mib="${_mib// /}"
    [ -n "$_idx" ] && [ -n "$_mib" ] || continue
    GPU_IDS+=("$_idx"); GPU_MEM+=($(( (_mib + 512) / 1024 )))
  done < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null)
fi
if [ -n "${JOBD_GPU_ALLOW:-}" ] && [ "${#GPU_IDS[@]}" -gt 0 ]; then
  _aids=(); _amem=()
  for _i in "${!GPU_IDS[@]}"; do
    case ",${JOBD_GPU_ALLOW}," in
      *",${GPU_IDS[$_i]},"*) _aids+=("${GPU_IDS[$_i]}"); _amem+=("${GPU_MEM[$_i]}") ;;
    esac
  done
  log "GPU allowance JOBD_GPU_ALLOW=$JOBD_GPU_ALLOW: ${#GPU_IDS[@]} probed -> ${#_aids[@]} usable"
  GPU_IDS=("${_aids[@]+"${_aids[@]}"}"); GPU_MEM=("${_amem[@]+"${_amem[@]}"}")
fi
NGPU=${#GPU_IDS[@]}

# --- effective CPU cores (cgroup quota, NOT nproc) ------------------------------
# MEASURED 2026-07-30 (BOX_SATURATION_AUDIT §5 rec 8 + §8): vast boxes hand the
# container a cpuset far wider than the CFS quota they actually allow. Box
# 46240842 showed `nproc` = 384 against `cpu.max 18432000 100000` = 184.32 cores
# (2.08x overstatement); box 46245045 showed 96 against 36.86 (2.6x). Exceeding a
# CFS quota is worse than plain contention — it stalls EVERY thread in the cgroup
# for the rest of each 100 ms period — so the number we export as CPU_CORES (read
# by autotune's dataloader sizing and by any SCORE_WORKERS=$CPU_CORES/N heuristic)
# must be the ALLOWANCE: min(nproc, floor(quota/period)).
# cgroup v2 = "<quota> <period>" in cpu.max ("max" = unlimited); v1 = the
# cfs_quota_us / cfs_period_us pair (-1 quota = unlimited). $JOBD_CGROUP_ROOT
# overrides the sysfs root so tests can inject a fake quota.
effective_cpu_cores() {   # -> allowed whole cores on stdout (>=1)
  local n q p cg quota=""
  n="$(nproc 2>/dev/null || echo 1)"
  [ "$n" -ge 1 ] 2>/dev/null || n=1
  cg="${JOBD_CGROUP_ROOT:-/sys/fs/cgroup}"
  q=""; p=""
  if [ -r "$cg/cpu.max" ]; then                       # cgroup v2
    read -r q p _ < "$cg/cpu.max" 2>/dev/null || true
  elif [ -r "$cg/cpu/cpu.cfs_quota_us" ] && [ -r "$cg/cpu/cpu.cfs_period_us" ]; then
    read -r q < "$cg/cpu/cpu.cfs_quota_us" 2>/dev/null || true   # cgroup v1
    read -r p < "$cg/cpu/cpu.cfs_period_us" 2>/dev/null || true
  fi
  # "max" (v2) / -1 (v1) / unparseable => no quota, keep nproc. A sub-1-core
  # allowance floors to 1, never to 0.
  if [ "${q:-max}" != "max" ] && [ "${q:-0}" -gt 0 ] 2>/dev/null \
     && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
    quota="$(( q / p ))"
    [ "$quota" -ge 1 ] || quota=1
  fi
  if [ -n "$quota" ] && [ "$quota" -lt "$n" ]; then
    echo "$quota"
  else
    echo "$n"
  fi
}

# --- scratch/filesystem probe (OBSERVABILITY ONLY — no policy, no behavior) -----
# The owner wants to eventually put low-value job scratch on RAM-backed storage:
# these boxes carry 64GB+ of RAM and barely touch it. THE MECHANISM THE IDEA
# ASSUMES IS WRONG on the one box anyone has actually looked at — inside a
# Docker container /tmp is normally the container's OVERLAY filesystem, i.e. the
# ALLOCATED (billed) disk rather than RAM, and that is exactly what the
# 2026-07-30 saturation audit measured (docs/plans/disk-sizing.md §6a: /tmp on
# overlayfs; the RAM-backed fs that DOES exist there is a 125 GB /dev/shm no
# script in this tree touches — elsewhere the Docker default is 64 MB).
# One box is not the fleet, and nothing in the daemon has ever recorded these
# facts, so a placement decision today would rest on a single hand-run `df`.
#
# So this MEASURES and CHANGES NOTHING. It records, once per boot, what actually
# backs /tmp, $ROOT and /dev/shm, how much RAM the box has, and whether this
# container is even permitted to mount a tmpfs — the evidence that decides
# whether the RAM-scratch idea is viable at all. Nothing here may ever gate a
# job. Every read degrades to the literal string "unknown" rather than failing
# daemon startup; an honest "unknown" is worth more than a guessed number,
# because a wrong fact here would be argued from later.
_kb_mb() {   # whole MB from a KiB count; non-numeric/absent -> "unknown"
  case "${1:-x}" in (*[!0-9]*|"") echo unknown ;; (*) echo $(( $1 / 1024 )) ;; esac
}

_fs_facts() {   # _fs_facts <path> -> "<fstype> <size_mb> <avail_mb>" (any field may be "unknown")
  local p="$1" t="" szkb="" frkb=""
  # `df -PT` gives type+size+avail in one shot: fs, TYPE, 1k-blocks, used, avail.
  read -r t szkb frkb <<< "$(df -PTk "$p" 2>/dev/null | awk 'END{if (NF>=5) print $2, $3, $5}')"
  # busybox df has no -T (and some images ship it): fall back to size/avail only,
  # then name the type from findmnt/proc. Any step may yield nothing -> unknown.
  [ -n "$szkb" ] || read -r szkb frkb <<< "$(df -Pk "$p" 2>/dev/null | awk 'END{if (NF>=4) print $2, $4}')"
  [ -n "$t" ] || t="$(findmnt -no FSTYPE --target "$p" 2>/dev/null | head -n1)"
  printf '%s %s %s\n' "${t:-unknown}" "$(_kb_mb "$szkb")" "$(_kb_mb "$frkb")"
}

_meminfo_mb() {   # _meminfo_mb <MemTotal|MemAvailable> -> whole MB or "unknown"
  local v
  v="$(awk -v k="$1:" '$1==k{print $2; exit}' /proc/meminfo 2>/dev/null)"
  _kb_mb "$v"
}

_cgroup_mem_mb() {   # _cgroup_mem_mb <max|current> -> whole MB | "max" | "unknown"
  # /proc/meminfo reports the HOST's memory, not this container's allowance —
  # the same overstatement trap effective_cpu_cores documents for nproc vs
  # cpu.max (measured 2.08-2.6x on real vast boxes). It matters here because
  # tmpfs pages are charged to the cgroup's memory limit: any future RAM-scratch
  # budget is bounded by THIS number, not by /dev/shm's nominal size, and
  # over-filling a tmpfs OOM-kills the job rather than costing disk. cgroup v2
  # first, then v1; "max"/absent/unparseable degrade honestly. $JOBD_CGROUP_ROOT
  # is the same sysfs test seam effective_cpu_cores uses.
  local cg="${JOBD_CGROUP_ROOT:-/sys/fs/cgroup}" v="" f
  case "$1" in
    max)     for f in "$cg/memory.max" "$cg/memory/memory.limit_in_bytes" ;do
               [ -r "$f" ] && { read -r v < "$f" 2>/dev/null; break; }; done ;;
    current) for f in "$cg/memory.current" "$cg/memory/memory.usage_in_bytes" ;do
               [ -r "$f" ] && { read -r v < "$f" 2>/dev/null; break; }; done ;;
  esac
  [ "${v:-}" = "max" ] && { echo max; return 0; }
  case "${v:-x}" in (*[!0-9]*|"") echo unknown; return 0 ;; esac
  # cgroup v1 spells "no limit" as a near-2^63 sentinel, not "max".
  [ "$v" -ge 9223372036854771712 ] 2>/dev/null && { echo max; return 0; }
  echo $(( v / 1048576 ))
}

_tmpfs_probe() {   # -> ok | denied | disabled | no_mount_cmd | no_tmpdir | mounted_no_umount
  # HONEST probe: actually try to mount a 1MB tmpfs and unmount it immediately —
  # `mount` needs CAP_SYS_ADMIN, which vast containers may or may not grant, and
  # no read of /proc answers "am I ALLOWED to". Bounded (timeout), disposable
  # (mktemp dir), and always cleaned up; JOBD_TMPFS_PROBE=0 turns it off for a
  # box where an operator does not want the daemon touching the mount table.
  [ "${JOBD_TMPFS_PROBE:-1}" = "1" ] || { echo disabled; return 0; }
  command -v mount >/dev/null 2>&1 && command -v umount >/dev/null 2>&1 \
    || { echo no_mount_cmd; return 0; }
  local d
  d="$(mktemp -d "${TMPDIR:-/tmp}/.jobd_tmpfs_probe.XXXXXX" 2>/dev/null)" \
    || { echo no_tmpdir; return 0; }
  if timeout 10 mount -t tmpfs -o size=1M jobd_probe "$d" >/dev/null 2>&1; then
    if timeout 10 umount "$d" >/dev/null 2>&1; then
      rmdir "$d" 2>/dev/null || true; echo ok
    else
      # Loud: a leaked mount is the one way this read-only probe could cost
      # anything. Never fatal, and the path is named so it can be cleaned by hand.
      log "!! tmpfs probe could not unmount $d — LEFT MOUNTED (set JOBD_TMPFS_PROBE=0 to disable)"
      echo mounted_no_umount
    fi
  else
    rmdir "$d" 2>/dev/null || true; echo denied
  fi
}

scratch_probe() {   # one structured box event + one log line, at boot. Never fatal.
  local tf ts_ tv wf ws_ wv sf ss_ sv mt ma cl cu tp
  read -r tf ts_ tv <<< "$(_fs_facts /tmp)"
  read -r wf ws_ wv <<< "$(_fs_facts "$ROOT")"
  read -r sf ss_ sv <<< "$(_fs_facts /dev/shm)"
  mt="$(_meminfo_mb MemTotal)"; ma="$(_meminfo_mb MemAvailable)"
  cl="$(_cgroup_mem_mb max)"; cu="$(_cgroup_mem_mb current)"
  tp="$(_tmpfs_probe)"
  # Field names are flat k=v so the estimator can fold them without a schema
  # change; jobd.py coerces integer-looking values to ints, so an "unknown"
  # stays a STRING and is never mistaken for a 0.
  emit_box scratch_probe \
    "tmp_fs=$tf" "tmp_size_mb=$ts_" "tmp_free_mb=$tv" \
    "workspace_path=$ROOT" "workspace_fs=$wf" "workspace_size_mb=$ws_" "workspace_free_mb=$wv" \
    "shm_fs=$sf" "shm_size_mb=$ss_" "shm_free_mb=$sv" \
    "mem_total_mb=$mt" "mem_avail_mb=$ma" \
    "cgroup_mem_limit_mb=$cl" "cgroup_mem_current_mb=$cu" "tmpfs_mount=$tp"
  log "scratch probe: /tmp=$tf(${tv}MB free of ${ts_}MB) $ROOT=$wf(${wv}MB free of ${ws_}MB) /dev/shm=$sf(${sv}MB free of ${ss_}MB) ram=${ma}MB avail of ${mt}MB cgroup_mem=${cu}/${cl}MB tmpfs_mount=$tp"
}

# --- DISK USAGE TELEMETRY (the half scratch_probe structurally cannot see) -----
# scratch_probe fires ONCE, at boot, so it records the ALLOCATION against an
# EMPTY disk and nothing else — it can never say what a job actually needed.
# That gap is why `--disk` has been argued from constants: disksize.py's
# BASE_OVERHEAD_GB and its (now archive-only) unpack peak had no measured
# allocated-vs-used distribution to check against, and an 18 GB phantom sat in
# the estimator long enough to make fleetd's replacement search refuse a rescue.
#
# This closes it with two cheap reads: a throttled df keeps a box high-water
# mark, and every job emits allocated-vs-used at its terminal, BEFORE the
# checkpoint scrub (i.e. at the job's real peak footprint, not after cleanup).
# TELEMETRY ONLY — nothing here gates, resizes, refuses or parks anything.
#
# The high-water mark is BOX-scoped, not job-scoped, and the field name says so:
# concurrent arms share one filesystem, so attributing a peak to one ticket
# would be a fiction. On the usual single-arm box the two coincide.
DISK_HW_FILE="$STATE_DIR/.disk_hw_mb"
DISK_HW_LAST=0

_disk_used_mb() {   # used MB on $ROOT, or "" when df could not answer
  local f s a; read -r f s a <<< "$(_fs_facts "$ROOT")"
  case "$s$a" in (*[!0-9]*|"") return 0 ;; esac
  echo $(( s - a ))
}

disk_hw_tick() {   # sample $ROOT into the box high-water mark. Never fatal.
  local now used hw
  now="$(date +%s)"
  [ $(( now - DISK_HW_LAST )) -ge "${JOBD_DISK_SAMPLE_S:-60}" ] || return 0
  DISK_HW_LAST="$now"
  used="$(_disk_used_mb)"; [ -n "$used" ] || return 0
  hw=0; [ -f "$DISK_HW_FILE" ] && read -r hw < "$DISK_HW_FILE" 2>/dev/null
  case "${hw:-x}" in (*[!0-9]*|"") hw=0 ;; esac
  [ "$used" -gt "$hw" ] 2>/dev/null && printf '%s\n' "$used" > "$DISK_HW_FILE"
  return 0
}

disk_usage_report() {   # disk_usage_report <jobid> <phase> [jobdir] — one box event
  local jobid="$1" phase="$2" jdir="${3:-}" f s a used hw wd
  read -r f s a <<< "$(_fs_facts "$ROOT")"
  used=unknown
  case "$s$a" in (*[!0-9]*|"") : ;; (*) used=$(( s - a )) ;; esac
  hw=unknown
  [ -f "$DISK_HW_FILE" ] && read -r hw < "$DISK_HW_FILE" 2>/dev/null
  case "${hw:-x}" in (*[!0-9]*|"") hw=unknown ;; esac
  # `used` is the whole container; `job_dir_mb` is this job's own bytes, which is
  # the figure disksize.py actually estimates. Bounded: a results tree the
  # manifest pass already walked is warm, but du must never delay a terminal.
  wd=unknown
  [ -n "$jdir" ] && [ -d "$jdir" ] \
    && wd="$(timeout 60 du -sm "$jdir" 2>/dev/null | awk 'END{if (NF) print $1}')"
  [ -n "$wd" ] || wd=unknown
  emit_box disk_usage "job=$jobid" "phase=$phase" \
    "workspace_fs=$f" "workspace_size_mb=$s" "workspace_free_mb=$a" \
    "workspace_used_mb=$used" "box_high_water_mb=$hw" "job_dir_mb=$wd"
  log "disk usage ($phase job=$jobid): ${used}MB used of ${s}MB (box high-water ${hw}MB, job dir ${wd}MB)"
  return 0
}

# --- boot GEMM ceiling (host-acceptance TELEMETRY) -----------------------------
# One structured box event + one B2 object, at boot. NEVER FATAL, and never a
# gate: nothing here rejects a box, re-rents, re-bids or aborts anything. See
# docs/plans/witness/perf/HOST_ACCEPTANCE_PROBE_2026-08-07.md §5 for why the policy
# half is deliberately unbuilt (a threshold needs a distribution; this makes it).
#
# WHY IT RUNS HERE, SYNCHRONOUSLY, BEFORE THE POLL LOOP. This is the only moment
# the GPU is provably idle. Backgrounding it would race the first claim, and
# gemm_probe.py's busy-GPU guard would then refuse forever on exactly the boxes
# we most want measured. The cost is bounded by JOBD_GEMM_TIMEOUT_S (default
# 120s) — under 0.6% of a 6 h rental against a measured 1.75-2.13x host spread —
# and the cached-record check means a park/resume pays it once, not every boot.
#
# FAIL-OPEN AT EVERY STEP: no probe file (older bundle), no train env, no torch,
# no GPU, a busy GPU, a wedged driver — each records a reason and returns 0.
# FLAT (job attach / jobd-boot bundle) first, then the repo layout one dir up —
# the same seam jobd.py uses for jobmeta/runmeta, so an in-tree jobd finds it too.
GP="$JOBD_DIR/gemm_probe.py"
[ -f "$GP" ] || GP="$(dirname "$JOBD_DIR")/gemm_probe.py"

gemm_probe() {
  [ "${JOBD_GEMM_PROBE:-1}" = "1" ] || return 0
  # A box presenting no cards BY POLICY (CPU boxes, the whole test suite, the
  # CPU-only rehearsal lane) has nothing to bench, and a FAKE inventory would
  # bench real silicon the operator said to leave alone. Both are hard skips
  # before anything is spawned.
  [ "${JOBD_SKIP_GPU:-0}" = "1" ] && return 0
  [ -n "${JOBD_FAKE_GPUS:-}" ] && return 0
  [ -f "$GP" ] || { log "gemm probe: gemm_probe.py absent (older jobd bundle) — skipping"; return 0; }

  # Cached record: a machine's GEMM ceiling does not change between a park and a
  # resume, and onstart re-runs on every resume. $ROOT survives a container
  # restart, so this is a per-BOX cache — NOT the store. The store is B2
  # (jobs/nodes/$IID/hostfacts/), written below; this file only decides whether
  # to spend 30s measuring again.
  local cache="$ROOT/.gemm_probe.json" maxage="${JOBD_GEMM_MAX_AGE_S:-86400}"
  local mtime age
  if [ -f "$cache" ]; then
    mtime="$(stat -c %Y "$cache" 2>/dev/null || echo 0)"
    age=$(( $(date -u +%s) - mtime ))
    if [ "$age" -ge 0 ] && [ "$age" -lt "$maxage" ] 2>/dev/null; then
      log "gemm probe: cached record ${age}s old (< ${maxage}s) — not re-measuring"
      return 0
    fi
  fi

  local fields="$STATE_DIR/.gemm_probe.fields" tmp="$cache.tmp"
  local logf="$JOBD_DIR/gemm_probe.log" to="${JOBD_GEMM_TIMEOUT_S:-120}"
  rm -f "$fields" "$tmp"
  # torch lives in the baked/rehydrated train env, not in the system python3.
  # Source the activate marker in a SUBSHELL ONLY: activating it in jobd's own
  # shell would change PATH/VIRTUAL_ENV for every entrypoint it later spawns,
  # which is a config change no job asked for.
  local act=""
  [ -f /workspace/.train_env_activate ] \
    && act="$(cat /workspace/.train_env_activate 2>/dev/null || true)"
  (
    if [ -z "${JOBD_GEMM_PY:-}" ] && [ -n "$act" ] && [ -f "$act" ]; then
      # shellcheck disable=SC1090
      . "$act" >/dev/null 2>&1 || true
    fi
    export JOBD_STATE_DIR="$STATE_DIR"
    # timeout is a BACKSTOP on top of gemm_probe's own --deadline-s: the python
    # side bounds the CUDA child, this bounds the python side (an import that
    # hangs on a wedged driver never reaches the child).
    timeout -k 10 "$to" "${JOBD_GEMM_PY:-python3}" "$GP" --quiet \
      --out "$tmp" --fields-out "$fields"
  ) >>"$logf" 2>&1
  local rc=$?

  if [ ! -s "$fields" ]; then
    # 124 == the shell timeout fired; anything else is a crash before the probe
    # could write. Both are recorded — a box we learned nothing about is itself
    # worth knowing, and silence would be indistinguishable from "not shipped".
    local why="probe_rc_$rc"
    [ "$rc" = 124 ] && why="jobd_timeout_${to}s"
    log "gemm probe: no record produced ($why) — continuing (see $logf)"
    emit_box gemm_probe "status=skipped_$why" "probe_timeout_s=$to"
    return 0
  fi

  # `|| [ -n "$line" ]` so a final line with no trailing newline is not dropped —
  # read returns nonzero at EOF and the last partial record would vanish.
  local -a ff=(); local line
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] && ff+=("$line")
    line=""
  done < "$fields"
  emit_box gemm_probe "${ff[@]}"
  log "gemm probe: $(tr '\n' ' ' < "$fields")"

  [ -s "$tmp" ] && mv -f "$tmp" "$cache" 2>/dev/null
  # Durable copy, keyed on the BOX and not on any job: jobs/nodes/<IID>/ is
  # jobd's per-box segment, and it is inside the `jobs/` namePrefix a split
  # box's scoped write key allows (CREDENTIAL_LIFECYCLE §2 — writing to a
  # hostfacts/ root would 403 on every such box). One immutable object per
  # measurement; the laptop's `hostfacts.py ingest` resolves the instance to a
  # machine_id and pins a by-machine copy while that mapping still exists.
  if [ -s "$cache" ]; then
    rclone rcat "$B2W/jobs/nodes/$IID/hostfacts/gemm-$(date -u +%Y%m%dT%H%M%SZ).json" \
      < "$cache" 2>>"$logf" \
      || log "gemm probe: B2 push failed — record kept locally at $cache"
  fi
  return 0
}

# --- harvested hostfacts DRAIN (the producer-agnostic other half) --------------
# gemm_probe above is a BENCHMARK jobd runs itself, so jobd owns the file and
# uploads it in the same breath. A HARVESTED fact — what a machine's cores are
# actually worth, counted off work we were already paying for — cannot work that
# way: it is produced by the JOB, mid-run, and only the job knows what it
# counted. See hostfacts.py's "the box-side DROP DIR" for the contract.
#
# So the job drops `<kind>-<ts>.json` in $DROP and jobd ships it. The producer
# needs no B2 key convention, no scoped-key rule, no rclone — three things that
# are wrong in an obvious way in every re-implementation and then 403 quietly.
#
# NEVER FATAL and never a gate, exactly like the probe it sits beside.
HOSTFACTS_DROP="${JOBD_HOSTFACTS_DROP:-$ROOT/hostfacts.d}"
hostfacts_drain() {
  [ "${JOBD_HOSTFACTS_DRAIN:-1}" = "1" ] || return 0
  [ -d "$HOSTFACTS_DROP" ] || return 0
  local f base sent="$HOSTFACTS_DROP/.sent" logf="$JOBD_DIR/hostfacts.log"
  for f in "$HOSTFACTS_DROP"/*.json; do
    # `*.json` unexpanded means an empty dir — the common case, every tick.
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    # `.partial` files are mid-write by contract (hostfacts.drop_record renames
    # into place), and the glob already excludes them. A zero-byte file is a
    # producer that died between create and write: skip it rather than PUT an
    # empty immutable object nobody can later distinguish from a real record.
    [ -s "$f" ] || { log "hostfacts: $base is empty — not uploading"; continue; }
    if rclone rcat "$B2W/jobs/nodes/$IID/hostfacts/$base" < "$f" 2>>"$logf"; then
      # Move rather than delete: a record that cost real work should survive a
      # bug in this function, and `.sent` is inside the drop dir so the glob
      # above (non-recursive, *.json) never re-reads it.
      mkdir -p "$sent" 2>/dev/null && mv -f "$f" "$sent/$base" 2>/dev/null \
        || rm -f "$f"
      log "hostfacts: uploaded $base"
    else
      # Left in place ON PURPOSE — the next tick retries. This is the one
      # difference from gemm_probe, which has a local cache to fall back on.
      log "hostfacts: B2 push failed for $base — kept for the next drain"
    fi
  done
  return 0
}

# --- boot CPU probe (host acceptance TELEMETRY; see cpu_probe.py) -------------
# The CPU mirror of gemm_probe, for the same reason: the selection path ranks
# CPU offers on `cores x GHz`, which cannot see IPC and cannot see whether a
# wide box scales at all. Measured 2026-08-24 the hostfacts store held 0 cpu
# records against 202 gemm — the difference being that gemm's producer is
# box-scoped and cpu's two were bundle-scoped and had never run.
#
# WHY HERE, SYNCHRONOUSLY. Same argument as gemm_probe: this is the only moment
# the box is provably unclaimed. cpu_probe.py refuses to measure a busy box, so
# backgrounding it would mean refusing forever on exactly the boxes worth
# measuring — and a probe landing mid-job is the co-tenant failure the CPU farm
# ruling (0a9f1926) was about.
#
# NO JOBD_SKIP_GPU / JOBD_FAKE_GPUS GUARD, DELIBERATELY. gemm_probe hard-skips
# a box presenting no cards because there is no silicon to bench. Here a box
# with no cards is the box we MOST want measured. This reads like a copy-paste
# omission; "fixing" it would blind the probe to the entire CPU-only fleet.
#
# It ships through the DROP DIR rather than rcat-ing its own record: the
# scoped-write-key rule is the thing that "is wrong in an obvious way in every
# re-implementation and then 403s quietly" (see the drain above), so there
# should be exactly one implementation of it and this is not it.
#
# Stdlib only, no train env, system python3: the boxes this matters most for
# have no torch and no GPU.
CP="$JOBD_DIR/cpu_probe.py"
[ -f "$CP" ] || CP="$(dirname "$JOBD_DIR")/cpu_probe.py"

cpu_probe() {
  [ "${JOBD_CPU_PROBE:-1}" = "1" ] || return 0
  # THE LOCAL LANE IS NOT A RENTED MACHINE. `job run-local` boots this same jobd
  # on the operator's own box with IID `local-<hostname>` (joblocal.py), and a
  # record filed from there enters the host scorecard and the fleet median as
  # though someone had rented a laptop. gemm_probe is spared this by accident —
  # run-local sets JOBD_FAKE_GPUS and it skips on that — which is precisely the
  # guard this probe drops on purpose, so it needs its own. It also keeps the
  # probe out of run-local's boot budget, which is seconds, not minutes.
  case "$IID" in
    local-*) log "cpu probe: local lane ($IID) — not a rented machine, skipping"
             return 0 ;;
  esac
  [ -f "$CP" ] || { log "cpu probe: cpu_probe.py absent (older jobd bundle) — skipping"; return 0; }

  # Per-BOX cache, NOT the store: core throughput does not change between a park
  # and a resume, and onstart re-runs on every resume. $ROOT survives a container
  # restart. The record itself goes to the drop dir; this only decides whether
  # to spend the time measuring again.
  local cache="$ROOT/.cpu_probe.json" maxage="${JOBD_CPU_MAX_AGE_S:-86400}"
  local mtime age
  if [ -f "$cache" ]; then
    mtime="$(stat -c %Y "$cache" 2>/dev/null || echo 0)"
    age=$(( $(date -u +%s) - mtime ))
    if [ "$age" -ge 0 ] && [ "$age" -lt "$maxage" ] 2>/dev/null; then
      log "cpu probe: cached record ${age}s old (< ${maxage}s) — not re-measuring"
      return 0
    fi
  fi

  local fields="$STATE_DIR/.cpu_probe.fields"
  local logf="$JOBD_DIR/cpu_probe.log" to="${JOBD_CPU_TIMEOUT_S:-240}"
  rm -f "$fields"
  (
    export JOBD_STATE_DIR="$STATE_DIR"
    export JOBD_HOSTFACTS_DROP="$HOSTFACTS_DROP"
    # BACKSTOP on top of the probe's own deadline: a fork that wedges on a
    # contended host never reaches the python-side bound.
    timeout -k 10 "$to" "${JOBD_CPU_PY:-python3}" "$CP" drop --fields
  ) >"$fields" 2>>"$logf"
  local rc=$?

  if [ ! -s "$fields" ]; then
    # 124 == the shell timeout fired; anything else crashed before writing.
    # Both are recorded: a box we learned nothing about is worth knowing, and
    # silence is indistinguishable from "never shipped".
    local why="probe_rc_$rc"
    [ "$rc" = 124 ] && why="jobd_timeout_${to}s"
    log "cpu probe: no record produced ($why) — continuing (see $logf)"
    emit_box cpu_probe "status=skipped_$why" "probe_timeout_s=$to"
    return 0
  fi

  # `|| [ -n "$line" ]` so a final line with no trailing newline is not dropped.
  local -a ff=(); local line
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] && ff+=("$line")
    line=""
  done < "$fields"
  emit_box cpu_probe "${ff[@]}"
  log "cpu probe: $(tr '\n' ' ' < "$fields")"
  date -u +%FT%TZ > "$cache" 2>/dev/null || true
  return 0
}

# --- shared Triton JIT cache (cross-box; tools/vast/triton_cache.py) ----------
# WHY: Triton JIT-compiles kernels on first use per box — measured 248s/222s of
# a cold short job's run phase (26-38% of wall clock; triton_cache.py's module
# doc has the table). The cache is keyed (torch, triton, sm) so it travels
# between boxes of the same arch; storage is R2 (R2_TC_* env) with B2 fallback.
# Re-keyed off `fla` on 2026-08-21 — a bake has no box to ask what fla is
# loaded, and Triton already hashes fla's kernels into its own entry names.
# FAIL-OPEN at every step: the tool itself exits 0 on every failure, and this
# wrapper backgrounds + bounds it, so a dead remote can never delay a claim or
# fail a job. Kill switch: JOBD_TRITON_CACHE=0.
TC_TOOL="$JOBD_DIR/triton_cache.py"
[ -f "$TC_TOOL" ] || TC_TOOL="$(dirname "$JOBD_DIR")/triton_cache.py"
TC_DIR="${JOBD_TRITON_CACHE_DIR:-$ROOT/triton-cache}"
TC_BASELINE_F="$ROOT/.triton_cache.baseline"   # entry count at last pull/push

_tc_py() {   # venv interpreter so `--detect` sees torch/fla (JOBD_GEMM_PY's twin)
  if [ -n "${JOBD_TC_PY:-}" ]; then echo "$JOBD_TC_PY"; return; fi
  echo "python3"
}

_tc_run() {  # _tc_run <out-json> <subcmd + args...> — bounded, venv-activated
  local out="$1"; shift
  local act="$ROOT/.train_env_activate" to="${JOBD_TC_TIMEOUT_S:-240}"
  if [ -z "${JOBD_TC_PY:-}" ] && [ -f "$act" ]; then
    timeout -k 10 "$to" bash -c '. "$1"; shift; exec python3 "$@"' _ "$act" \
      "$TC_TOOL" "$@" > "$out" 2>>"$ROOT/.triton_cache.log"
  else
    timeout -k 10 "$to" "$(_tc_py)" "$TC_TOOL" "$@" \
      > "$out" 2>>"$ROOT/.triton_cache.log"
  fi
}

triton_cache_boot_pull() {   # boot: populate $TC_DIR from the remote, in background
  [ "${JOBD_TRITON_CACHE:-1}" = "1" ] || return 0
  [ -f "$TC_TOOL" ] || { log "triton cache: tool absent (older bundle) — skipping"; return 0; }
  [ "${#GPU_IDS[@]}" -gt 0 ] 2>/dev/null || return 0   # CPU box: nothing to warm
  ( out="$ROOT/.triton_cache.pull.json"
    _tc_run "$out" pull --dest "$TC_DIR" --detect
    hit="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("hit"))' "$out" 2>/dev/null || echo "?")"
    ent="$("$PY" -c 'import json,sys;print((json.load(open(sys.argv[1])).get("dir") or {}).get("entries",0))' "$out" 2>/dev/null || echo 0)"
    echo "${ent:-0}" > "$TC_BASELINE_F" 2>/dev/null || true
    log "triton cache: boot pull hit=$hit entries=$ent ($(head -c 300 "$out" 2>/dev/null))"
    emit_box triton_cache "op=pull" "hit=$hit" "entries=$ent" ) &
  return 0
}

triton_cache_push_bg() {   # after a GPU job: publish growth, in background
  local jobid="$1"
  [ "${JOBD_TRITON_CACHE:-1}" = "1" ] || return 0
  [ -f "$TC_TOOL" ] && [ -d "$TC_DIR" ] || return 0
  ( out="$ROOT/.triton_cache.push.json"
    base="$(cat "$TC_BASELINE_F" 2>/dev/null || echo 0)"
    case "$base" in ''|*[!0-9]*) base=0 ;; esac
    # --update: replace the remote key only when this box compiled kernels
    # beyond the recorded baseline — keeps the tax one-time FLEET-wide while
    # never re-uploading identical bytes after every job.
    _tc_run "$out" push --src "$TC_DIR" --detect --update --baseline-entries "$base"
    pushed="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("pushed"))' "$out" 2>/dev/null || echo "?")"
    added="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("entries_added",0))' "$out" 2>/dev/null || echo 0)"
    if [ "$pushed" = "True" ]; then
      "$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("entries_packed",0))' "$out" \
        > "$TC_BASELINE_F" 2>/dev/null || true
    fi
    log "triton cache: job $jobid push pushed=$pushed entries_added=$added ($(head -c 300 "$out" 2>/dev/null))"
    emit_box triton_cache "op=push" "job=$jobid" "pushed=$pushed" "entries_added=$added" ) &
  return 0
}

# --- scheduler state ------------------------------------------------------------
# NOTE the explicit =() initializers: under bash 5.3 + set -u, a `declare -A M`
# without one leaves M UNSET, and ${#M[@]} / ${!M[@]} then abort the shell.
declare -A GPU_OWNER=()    # gpu index -> jobid
declare -A JOB_PID=()      # jobid -> pid of its runner subshell
declare -A JOB_GPU_SLOT=() # jobid -> "0,1" csv, or "-" for a CPU job.
# NOT named JOB_GPUS: run_job_body EXPORTS a scalar JOB_GPUS to the entrypoint
# (the documented contract — JOBS_DESIGN.md, AUTOTUNE_DESIGN.md). It runs in a
# subshell that INHERITS this scope, so while the two shared a name the
# `export JOB_GPUS="$gpus"` assigned to element [0] of the inherited
# ASSOCIATIVE ARRAY and exported NOTHING — every entrypoint on every box has
# been running without JOB_GPUS in its environment since v2. Found 2026-07-30
# by the first real run of the LOCAL GPU LANE (LOCAL_GPU_LANE.md).
CPU_RUNNING=0
SPAWNED=0

pick_gpus() {   # pick_gpus <n> <min_gb_per_card> -> csv on stdout, rc 1 if no fit
  local n="$1" ram="${2:-0}" out=() i idx
  for i in "${!GPU_IDS[@]}"; do
    idx="${GPU_IDS[$i]}"
    [ -n "${GPU_OWNER[$idx]:-}" ] && continue
    [ "${GPU_MEM[$i]}" -ge "$ram" ] 2>/dev/null || continue
    out+=("$idx")
    [ "${#out[@]}" -eq "$n" ] && break
  done
  [ "${#out[@]}" -eq "$n" ] || return 1
  local IFS=','; echo "${out[*]}"
}

status_marker() {
  if [ "${#JOB_PID[@]}" -gt 0 ]; then
    jobd_status "RUNNING ${#JOB_PID[@]}" "${!JOB_PID[*]}"
  else
    jobd_status IDLE
  fi
}

# --- boot-health beacon: surface the LIVE asset-pull rate on JOBD_STATUS -------
# A runner subshell drops a $STATE_DIR/<jobid>.staging marker ("<asset>\t<stats
# file>") around each live asset_pull (see asset_pull). While a box is staging a
# multi-GB asset it can be pre-`started` for many minutes; the boot-health
# monitor (`job supervise` / the workflow tick) needs a periodic mbps sample to
# condemn a per-flow-shaped host (the M2 45064080 incident) instead of burning
# the whole fixed boot deadline. staging_status parses the newest stats file's
# last one-line rate and appends `staging=<asset> mbps=<MB/s>` to the heartbeat.
# No stats yet (or the local-bucket test shim, which emits none) => the field is
# omitted rather than reported as 0 (a missing sample must not read as starved).
STAGING_SHOWN=0
_last_mbps() {   # _last_mbps <statsfile> -> decimal MB/s on stdout (empty if none)
  local f="$1" tok num unit
  [ -s "$f" ] || return 0
  tok="$(grep -aoE '[0-9]+(\.[0-9]+)? [KMGT]?i?B/s' "$f" 2>/dev/null | tail -n1)"
  [ -n "$tok" ] || return 0
  num="${tok%% *}"; unit="${tok##* }"
  case "$unit" in
    B/s)        awk -v n="$num" 'BEGIN{printf "%.1f", n/1048576}' ;;
    KiB/s|KB/s) awk -v n="$num" 'BEGIN{printf "%.1f", n/1024}' ;;
    MiB/s|MB/s) awk -v n="$num" 'BEGIN{printf "%.1f", n}' ;;
    GiB/s|GB/s) awk -v n="$num" 'BEGIN{printf "%.1f", n*1024}' ;;
    TiB/s|TB/s) awk -v n="$num" 'BEGIN{printf "%.1f", n*1048576}' ;;
  esac
}
staging_status() {
  local sm name statsf newest="" newest_mt=0 mt
  for sm in "$STATE_DIR"/*.staging; do
    [ -f "$sm" ] || continue
    mt="$(stat -c %Y "$sm" 2>/dev/null || stat -f %m "$sm" 2>/dev/null || echo 0)"
    if [ "${mt:-0}" -ge "$newest_mt" ]; then newest_mt="$mt"; newest="$sm"; fi
  done
  if [ -z "$newest" ]; then
    # staging finished since the last tick: reset the marker to the plain
    # RUNNING/IDLE line ONCE (status_marker is otherwise only event-driven).
    if [ "$STAGING_SHOWN" = "1" ]; then STAGING_SHOWN=0; status_marker; fi
    return 0
  fi
  IFS=$'\t' read -r name statsf < "$newest" 2>/dev/null || return 0
  local mbps=""; [ -n "$statsf" ] && mbps="$(_last_mbps "$statsf")"
  local extra="staging=${name:-?}"; [ -n "$mbps" ] && extra="$extra mbps=$mbps"
  STAGING_SHOWN=1
  if [ "${#JOB_PID[@]}" -gt 0 ]; then
    jobd_status "RUNNING ${#JOB_PID[@]}" "$extra ${!JOB_PID[*]}"
  else
    jobd_status STAGING "$extra"
  fi
}

reap() {   # free slots of finished runners; the runner itself published/emitted
  local jid changed=0
  for jid in "${!JOB_PID[@]}"; do
    kill -0 "${JOB_PID[$jid]}" 2>/dev/null && continue
    wait "${JOB_PID[$jid]}" 2>/dev/null || true    # no-op for adopted non-children
    local g gpus="${JOB_GPU_SLOT[$jid]:--}"
    if [ "$gpus" = "-" ]; then
      CPU_RUNNING=$(( CPU_RUNNING > 0 ? CPU_RUNNING - 1 : 0 ))
    else
      IFS=',' read -ra g <<< "$gpus"
      for i in "${g[@]}"; do [ -n "$i" ] && unset "GPU_OWNER[$i]"; done
    fi
    unset "JOB_PID[$jid]" "JOB_GPU_SLOT[$jid]"
    rm -f "$STATE_DIR/$jid.running" "$STATE_DIR/$jid.staging"
    changed=1
    log "reaped job $jid (gpus=$gpus)"
  done
  [ "$changed" = "1" ] && status_marker
  return 0
}

# reap_orphan_gpu_procs — best-effort clear of wedged GPU state at boot. After a
# torchrun/ranks SIGKILL (box preempt) a dead process can leave a GPU pinned at
# 100% util holding ~0 MiB (the 2026-07-12 box-44566398 zombie: cards at 100%/2
# MiB doing nothing, billing). On a fresh daemon boot NO job is adopted yet, so
# any compute-app still reparented to init (ppid=1) is an orphan from a prior
# run and safe to kill — a LIVE job's procs have a live runner ancestor. We never
# kill non-orphans, never touch GPUs under JOBD_SKIP_GPU/JOBD_FAKE_GPUS (tests),
# and only when the daemon adopted nothing (a normal restart with live runners is
# left alone). Opt out with JOBD_GPU_REAP=0.
reap_orphan_gpu_procs() {
  [ "${JOBD_GPU_REAP:-1}" = "1" ] || return 0
  [ "${JOBD_SKIP_GPU:-0}" = "1" ] && return 0
  [ -n "${JOBD_FAKE_GPUS:-}" ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  [ "${#JOB_PID[@]}" -eq 0 ] || return 0   # adopted live work -> do not touch GPUs
  local pids p ppid killed=0
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | tr -d ' ' | grep -E '^[0-9]+$' | sort -u)"
  [ -n "$pids" ] || return 0
  for p in $pids; do
    ppid="$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || echo "")"
    [ "$ppid" = "1" ] || continue          # only orphans reparented to init
    log "reaping orphaned GPU compute proc pid=$p (ppid=1, no live job) — wedged-context cleanup"
    kill -TERM "$p" 2>/dev/null || true; killed=1
  done
  if [ "$killed" = "1" ]; then
    sleep 2
    for p in $pids; do
      ppid="$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || echo "")"
      [ "$ppid" = "1" ] && kill -KILL "$p" 2>/dev/null || true
    done
  fi
}

adopt_running() {   # daemon restart while runners survived: re-own their slots
  local rf
  for rf in "$STATE_DIR"/*.running; do
    [ -f "$rf" ] || continue
    local jid pid gpus rest
    jid="$(basename "${rf%.running}")"
    read -r pid gpus rest < "$rf" || continue
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      JOB_PID["$jid"]="$pid"; JOB_GPU_SLOT["$jid"]="${gpus:--}"
      if [ "${gpus:--}" = "-" ]; then
        CPU_RUNNING=$((CPU_RUNNING+1))
      else
        local g; IFS=',' read -ra g <<< "$gpus"
        for i in "${g[@]}"; do [ -n "$i" ] && GPU_OWNER["$i"]="$jid"; done
      fi
      log "adopted running job $jid (pid=$pid gpus=${gpus:--})"
    else
      rm -f "$rf"    # stale: the job died with a previous daemon -> resume path
    fi
  done
}

# --- venv need (runs in the RUNNER so the source affects the entrypoint) -------
# CONTRACT (unchanged): stdout is EMPTY on success, or a one-line failure reason
# consumed by the caller as `reason="$(check_venv ...)"` (non-empty => job goes
# terminal `failed` BEFORE the entrypoint runs). Provisioning is therefore the
# REAL self-heal the design promised (JOBS_DESIGN.md) — when the env is ABSENT we
# invoke the matching provisioner rather than no-op'ing:
#   serve => job_serve.sh --build-venv (the FACTORED venv-build path; builds
#            $ROOT/serve without starting a server)
#   eval  => fetch_eval_env.sh (idempotent baked-env pull to $ROOT/eval)
# Provisioner PROGRESS must go to STDERR (the daemon log) — routing it to stdout
# would pollute `reason` and spuriously fail the job. Provisioner scripts resolve
# via env override, then PATH (test seam: a stub `job_serve.sh`/`fetch_eval_env.sh`
# on PATH is picked up), then the sibling onstart/ copy (the real box).
_venv_provisioner() {   # _venv_provisioner <script-name> <env-override-value>
  local name="$1" override="$2" p
  if [ -n "$override" ]; then echo "$override"; return; fi
  p="$(command -v "$name" 2>/dev/null || true)"
  [ -n "$p" ] || p="$JOBD_DIR/$name"
  echo "$p"
}

# _venv_provision <kind> <script> [args...] — run a provisioner under a
# per-kind, BOX-GLOBAL flock.
#
# The venv is box-global mutable state ($ROOT/serve, $ROOT/eval) and jobd runs
# jobs CONCURRENTLY (CPU_SLOTS, and multi-arm GPU packing), so N runners can
# reach the same ABSENT venv in the same second and all start provisioning it.
# MEASURED 2026-07-30 (frontier wave round 3): three concurrent jobs raced
# `job_serve.sh --build-venv` into /workspace/serve and the parallel pip
# self-upgrade corrupted pip itself — two of three jobs died with
# `ModuleNotFoundError: pip._internal`. The bundle-side fix (frontier-wave
# run.sh, 277f0a3c) only covers entrypoints that build the venv THEMSELVES;
# this is the same fix at the seam every `needs.venv:` job goes through.
#
# First holder provisions; the others block and then find the venv present
# (both provisioners are idempotent and fast-path on a warm install). `flock`
# absent (a slim image) => run unlocked rather than fail the job, which is the
# pre-2026-07-30 behaviour.
#
# BOUNDED, both halves (2026-08-20): the lock was box-global and untimed around
# an unbounded multi-GB pip, so ONE wedged provisioner stalled every job on the
# box forever. Timeout-and-fail instead: generous enough that a legitimate cold
# multi-GB install finishes, short enough that a wedge fails ONE ticket (through
# check_venv's normal reason => `failed` + mark_terminal) and leaves the box
# schedulable. Which bound tripped is named on stderr AND folded into the
# ticket's reason, because the two need different operator responses.
VENV_LOCK_WAIT_S="${JOBD_VENV_LOCK_WAIT_S:-1800}"          # 30 min behind a peer
VENV_PROVISION_TIMEOUT_S="${JOBD_VENV_PROVISION_TIMEOUT_S:-3600}"   # 60 min install
VENV_PROVISION_DETAIL=""
_venv_provision() {
  local kind="$1"; shift
  local rc=0
  VENV_PROVISION_DETAIL=""
  if command -v flock >/dev/null 2>&1; then
    # -E 125 separates "never got the lock" from the provisioner's own rc and
    # from `timeout`'s 124 ("the install itself blew its bound").
    flock -w "$VENV_LOCK_WAIT_S" -E 125 "$ROOT/.venv-$kind.lock" \
      timeout -k 30 "$VENV_PROVISION_TIMEOUT_S" "$@" || rc=$?
  else
    echo ">> check_venv: flock unavailable — provisioning $kind UNSERIALIZED" >&2
    timeout -k 30 "$VENV_PROVISION_TIMEOUT_S" "$@" || rc=$?
  fi
  case "$rc" in
    125) VENV_PROVISION_DETAIL="lock wait exceeded ${VENV_LOCK_WAIT_S}s — another job on this box holds $ROOT/.venv-$kind.lock" ;;
    124) VENV_PROVISION_DETAIL="install exceeded ${VENV_PROVISION_TIMEOUT_S}s and was killed" ;;
  esac
  [ -z "$VENV_PROVISION_DETAIL" ] \
    || echo ">> check_venv: $kind provisioning BOUND tripped — $VENV_PROVISION_DETAIL" >&2
  return "$rc"
}

# _eval_env_pin <job-env-file> — the ticket's EVAL_ENV_VER, or "".
#
# fetch_eval_env.sh reads EVAL_ENV_VER from ITS environment and otherwise
# resolves eval-env/LATEST, and check_venv runs it BEFORE the entrypoint
# subshell sources .job.env — so on a cold box a ticket pin used to document a
# choice it could not steer, and the box fetched LATEST. That is not a stale
# env, it is a DIFFERENT INSTRUMENT: a pinned bake deliberately does not
# advance LATEST, so an eval job then grades on a bake nobody named.
# Sourced in a subshell (the file is our own `export k=<shell-quoted>` output).
_eval_env_pin() {
  local envf="$1"
  [ -n "$envf" ] && [ -f "$envf" ] || return 0
  # shellcheck source=/dev/null
  ( set +u; . "$envf" >/dev/null 2>&1; printf '%s' "${EVAL_ENV_VER:-}" ) 2>/dev/null
}

check_venv() {
  local need_venv="$1" job_env_file="${2:-}"
  case "$need_venv" in
    none|"") : ;;
    eval)
      if [ -f "$ROOT/eval/env.sh" ]; then
        # shellcheck source=/dev/null
        source "$ROOT/eval/env.sh" 2>/dev/null || true
      else
        # ABSENT => self-provision the baked eval env (idempotent fast-path inside).
        local fe; fe="$(_venv_provisioner fetch_eval_env.sh "${JOBD_FETCH_EVAL_SH:-}")"
        local pin; pin="$(_eval_env_pin "$job_env_file")"
        local -a fecmd=(bash "$fe")
        if [ -n "$pin" ]; then
          fecmd=(env "EVAL_ENV_VER=$pin" bash "$fe")
          echo ">> check_venv: eval env absent — provisioning via $fe at the ticket pin EVAL_ENV_VER=$pin" >&2
        else
          echo ">> check_venv: eval env absent — provisioning via $fe (ticket names no EVAL_ENV_VER; the box env or eval-env/LATEST decides)" >&2
        fi
        if _venv_provision eval "${fecmd[@]}" >&2; then
          if [ -f "$ROOT/eval/env.sh" ]; then
            # shellcheck source=/dev/null
            source "$ROOT/eval/env.sh" 2>/dev/null || true
          elif ! "$PY" -c 'import upstream_monorepo' >/dev/null 2>&1; then
            echo "needs.venv=eval: provisioned but $ROOT/eval/env.sh still missing and upstream_monorepo unimportable"; return
          fi
        else
          echo "needs.venv=eval: fetch_eval_env.sh provisioning failed${VENV_PROVISION_DETAIL:+ ($VENV_PROVISION_DETAIL)} (see daemon log)"; return
        fi
      fi ;;
    serve)
      if [ -f "$ROOT/serve/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "$ROOT/serve/bin/activate" 2>/dev/null || true
      else
        # ABSENT => self-provision the serve venv via job_serve.sh's factored build.
        local js; js="$(_venv_provisioner job_serve.sh "${JOBD_JOB_SERVE_SH:-}")"
        echo ">> check_venv: serve venv absent — provisioning via $js --build-venv" >&2
        if _venv_provision serve bash "$js" --build-venv >&2; then
          if [ -f "$ROOT/serve/bin/activate" ]; then
            # shellcheck source=/dev/null
            source "$ROOT/serve/bin/activate" 2>/dev/null || true
          fi
        else
          echo "needs.venv=serve: job_serve.sh --build-venv provisioning failed${VENV_PROVISION_DETAIL:+ ($VENV_PROVISION_DETAIL)} (see daemon log)"; return
        fi
      fi ;;
    *) echo "needs.venv: unknown value '$need_venv'"; return ;;
  esac
  echo ""
}

# --- declarative asset staging (N4) --------------------------------------------
# Big inputs (base models, adapters, corpora) reach the box declaratively via the
# ticket's `assets:` list, replacing five hand-rolled b2pull() loops. ONE shared
# pull primitive: linear-backoff retries, trailing-slash dest (the B2 HEAD-flake
# rule — see the checkpoint-sync dest + train.sh:360), rclone stderr CAPTURED
# (auth-class failure => a loud named diagnostic, mirroring checkpoint_sync_failed).
# Stable cache: each asset lands under $ROOT/assets/<name> and survives park/resume
# and dedupes across matrix arms; a `.complete` byte-total marker (ported from
# ensure_base_model.sh) skips a re-pull. `require:` postconditions + an automatic
# index-aware weight-shard completeness check (ensure_base_model.sh:86-131) run
# post-pull. A non-optional asset failure takes the job terminal `failed` with a
# distinct `asset_stage_failed:<name>` reason BEFORE the entrypoint starts.
ASSETS_DIR="$ROOT/assets"

_dir_bytes() {   # total bytes of regular files under a dir (0 if absent/empty)
  local d="$1"
  [ -d "$d" ] || { echo 0; return 0; }
  find "$d" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{printf "%d\n", s+0}'
}

# --- free-space precheck (P4e) --------------------------------------------------
# WHY: staging pulled arbitrarily large assets with NO free-space check at all —
# the only `df` gate anywhere in this tree was farm_attach.sh's MIN_FREE_GB=20,
# and it guards a DIFFERENT lane. On a box whose --disk was sized too small,
# rclone dies partway through and the job goes terminal with
# `asset_stage_failed:<name>`: a TRANSPORT-shaped reason that never says "disk".
# The operator reads it as a flaky pull and retries onto the same undersized box,
# so the money bought no information. This precheck runs BEFORE the staging loop
# pulls a byte and refuses with its OWN reason (`insufficient_disk`) carrying the
# numbers, so the fix (a bigger --disk) is legible from the event alone.
#
# SAFETY INVARIANT — an UNREADABLE measurement never blocks the job. If `df` is
# missing, errors, or prints something we cannot parse we log it and PROCEED: a
# measurement we could not take is NOT evidence of a full disk (the same house
# rule as the reaper's "unreadable evidence never accelerates a destructive
# action"). Same for size: an asset whose byte count we cannot learn contributes
# ZERO to the requirement, so we only ever refuse on bytes we actually accounted
# for. A false refusal here strands a box that could have done the work — the
# exact failure mode this change exists to stop.

_free_kb() {   # _free_kb <path> -> free KiB on stdout; rc 1 (no output) = UNKNOWN
  local kb
  kb="$(df -Pk "$1" 2>/dev/null | awk 'END{if (NF>=4) print $4}')"
  case "${kb:-x}" in (*[!0-9]*|"") return 1 ;; esac
  echo "$kb"
}

_b2_size_bytes() {   # _b2_size_bytes <b2prefix> -> total bytes; rc 1 = UNKNOWN
  # `rclone size --json` is one LIST against the prefix (cheap next to the pull it
  # is about to authorize). Hard-timeout it: a wedged listing must not stall the
  # runner, it must fall through to UNKNOWN. The local test shim has no `size` op
  # and exits non-zero — which is exactly the UNKNOWN path, by design.
  local out b
  out="$(timeout "${JOBD_DISK_SIZE_TIMEOUT_S:-120}" rclone size --json "$B2/$1/" 2>/dev/null)" || return 1
  b="$(printf '%s' "$out" | sed -n 's/.*"bytes"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"
  case "${b:-x}" in (*[!0-9]*|"") return 1 ;; esac
  echo "$b"
}

# _asset_need_bytes <name> <b2prefix> — bytes this asset still has to LAND on
# disk. Echoes an integer (0 = already here, costs nothing); rc 1 = UNKNOWN.
# Mirrors the skip conditions in _stage_one_asset_body EXACTLY, so the precheck
# can never demand room for something staging will not pull:
#   - a pre-seeded local asset (LOCAL_GPU_LANE .local marker) is never pulled;
#   - a cache whose `.complete` byte-total still matches is reused verbatim, so a
#     RESUMED box is not asked to find room for bytes it already holds (that
#     would refuse work the box can obviously do);
#   - otherwise: remote size minus what is already in the cache (rclone copy
#     skips files it already has), floored at 0. Remote unreadable but a
#     `.complete` marker present => the marker's total minus what is on disk
#     (the interrupted-pull case, where the marker is the only size we have).
_asset_need_bytes() {
  local name="$1" b2p="$2"
  local cache="$ASSETS_DIR/$name" marker="$ASSETS_DIR/.$name.complete"
  [ -f "$ASSETS_DIR/.$name.local" ] && { echo 0; return 0; }
  local got want=0 rem remote
  got="$(_dir_bytes "$cache")"
  if [ -f "$marker" ]; then
    want="$(cat "$marker" 2>/dev/null || echo 0)"
    case "${want:-x}" in (*[!0-9]*|"") want=0 ;; esac
    if [ "$want" -gt 0 ] && [ "$got" -ge "$want" ] 2>/dev/null; then echo 0; return 0; fi
  fi
  if remote="$(_b2_size_bytes "$b2p")"; then
    rem=$(( remote - got )); [ "$rem" -gt 0 ] || rem=0
    echo "$rem"; return 0
  fi
  if [ "$want" -gt 0 ] 2>/dev/null; then
    rem=$(( want - got )); [ "$rem" -gt 0 ] || rem=0
    echo "$rem"; return 0
  fi
  return 1
}

# assets_disk_precheck <jobid> <specf> — rc 0 = proceed (it fits, or we could not
# measure); rc 1 = REFUSED, terminal `failed` already emitted (same contract as
# _stage_one_asset_body, so the caller just aborts before `started`).
#
# Deliberately OUTSIDE the per-asset stage lock and per-JOB, not per-box: N
# concurrent arms sharing one asset name each need the same X GB once (they share
# the cache), so each checking X independently is the right question, not N*X.
assets_disk_precheck() {
  local jobid="$1" specf="$2"
  [ "${JOBD_DISK_PRECHECK:-1}" = "1" ] || return 0
  local name b2p mode opt dest recp need
  local total=0 names="" unknown="" optional="" big=0 bigname=""
  while IFS=$'\t' read -r name b2p mode opt dest recp; do
    [ -n "$name" ] || continue
    # `optional: true` assets are EXCLUDED from the requirement. Staging already
    # tolerates their failure (logged + skipped, job runs on), so refusing the
    # whole job for want of room for one would be strictly worse than the
    # behavior it replaces — the job would have run fine without it.
    if [ "${opt:-0}" = "1" ]; then optional="${optional:+$optional,}$name"; continue; fi
    if need="$(_asset_need_bytes "$name" "$b2p")"; then
      [ "$need" -gt 0 ] 2>/dev/null || continue      # already staged: free
      total=$(( total + need ))
      names="${names:+$names,}$name"
      [ "$need" -gt "$big" ] && { big="$need"; bigname="$name"; }
    else
      unknown="${unknown:+$unknown,}$name"
    fi
  done < "$specf"

  # TICKET-DECLARED REQUIREMENT — WIRING HOOK (one line away from live).
  # `needs.disk_gb` already exists laptop-side (jobmeta.py validates it as a
  # positive number of GB — velvet P4), but onstart/jobd.py `prepare` does not
  # echo it yet. The moment it does — one
  # `f"JOB_NEEDS_DISK_GB={q(needs.get('disk_gb') or 0)}"` line in cmd_prepare's
  # `out` list, plus the name added to run_ticket's `local JOB_…` declaration so
  # a value cannot leak between jobs — this reads it with NO further change here.
  # JOBD_MIN_FREE_GB is the operator/env twin for a box-wide floor.
  # NOT read (deliberately): `needs.scratch_gb`. jobmeta.py defines it as a term
  # ADDED to the laptop-side --disk estimate, i.e. room the ENTRYPOINT will
  # write, not bytes staging must land. Its box-side term belongs next to this
  # line once someone owns that semantic; guessing it here could refuse a box
  # that would have run the job fine.
  local declared="${JOB_NEEDS_DISK_GB:-}"
  [ -n "$declared" ] || declared="${JOBD_MIN_FREE_GB:-0}"
  # disk_gb may legitimately be fractional (12.5) — round UP to whole GB rather
  # than discarding the whole declaration as unparseable.
  case "$declared" in
    *.*) case "${declared%%.*}" in
           (*[!0-9]*|"") declared=0 ;;
           (*) declared=$(( ${declared%%.*} + 1 )) ;;
         esac ;;
  esac
  case "$declared" in (*[!0-9]*|"") declared=0 ;; esac

  local headroom="${JOBD_DISK_HEADROOM_GB:-5}"
  case "$headroom" in (*[!0-9]*|"") headroom=5 ;; esac

  # Requirement in whole GiB, rounded UP (a partial GiB still has to fit), plus
  # working headroom — a stage is never the only thing writing (bundle extract,
  # rclone temp files, the entrypoint's first outputs). The declared figure is a
  # FLOOR, not a replacement: whichever is larger wins.
  local req_gb=0
  [ "$total" -gt 0 ] && req_gb=$(( (total + 1073741823) / 1073741824 + headroom ))
  [ "$declared" -gt "$req_gb" ] && req_gb="$declared"
  if [ "$req_gb" -le 0 ]; then
    [ -n "$unknown$optional" ] && log "job $jobid: disk precheck — no measurable requirement (${unknown:+unsized: $unknown }${optional:+optional-not-counted: $optional}); proceeding"
    return 0
  fi

  local free_kb
  if ! free_kb="$(_free_kb "$ASSETS_DIR")"; then
    # THE invariant: unreadable free space must NOT refuse the job.
    log "job $jobid: disk precheck SKIPPED — df unreadable for $ASSETS_DIR (would have wanted ~${req_gb}GB); proceeding"
    return 0
  fi
  local free_gb=$(( free_kb / 1048576 ))
  if [ "$free_gb" -ge "$req_gb" ] 2>/dev/null; then
    log "job $jobid: disk precheck ok — ${free_gb}GB free >= ${req_gb}GB required (assets: ${names:-none}${unknown:+; unsized: $unknown})"
    # SCRATCH: warn, never refuse. `needs.scratch_gb` is what the ENTRYPOINT
    # writes (a ninja build tree, N compile worktrees), not bytes staging has to
    # land — so it is deliberately NOT part of the refusal above: a box that
    # cannot fit the declared scratch may still run the job fine, and refusing
    # on an author's estimate would be worse than the silence it replaces.
    # But staying silent entirely is how a job stages cleanly and then dies with
    # a compiler ENOSPC nobody connects to disk. So leave the breadcrumb: it
    # costs nothing and it names the term. Hook mirrors JOB_NEEDS_DISK_GB —
    # cmd_prepare echoes neither yet.
    local scr="${JOB_NEEDS_SCRATCH_GB:-}"
    case "${scr%%.*}" in (*[!0-9]*|"") scr=0 ;; (*) scr="${scr%%.*}" ;; esac
    if [ "$scr" -gt 0 ] 2>/dev/null && [ $(( free_gb - req_gb )) -lt "$scr" ]; then
      log "job $jobid: !! disk WARNING — ${free_gb}GB free covers the ${req_gb}GB of assets but leaves $(( free_gb - req_gb ))GB for a declared ${scr}GB of scratch (needs.scratch_gb). Staging will succeed; the entrypoint may run out mid-build."
    fi
    return 0
  fi
  log "job $jobid: INSUFFICIENT DISK — ${free_gb}GB free < ${req_gb}GB required on $ASSETS_DIR (assets: ${names:-<declared only>}${unknown:+; unsized: $unknown})"
  emit "$jobid" failed \
    --field reason="insufficient_disk: ${free_gb}GB free < ${req_gb}GB required on $ASSETS_DIR (assets: ${names:-<declared only>})" \
    --field free_gb="$free_gb" --field required_gb="$req_gb" \
    --field assets="${names:-}" --field path="$ASSETS_DIR" \
    ${bigname:+--field largest="$bigname"} \
    ${unknown:+--field unsized="$unknown"} \
    ${optional:+--field optional_uncounted="$optional"}
  mark_terminal "$jobid" failed
  return 1
}

# --- transfer guards for the staging pull (owner scope-out 2026-08-02) ----------
# Stall DETECTION for a box is a control-plane job (fleetd) — a wedged box is
# exactly the one that will not answer a probe. What a TRANSFER must do for
# itself is bound its own runtime, and asset_pull had no bound of any kind: the
# rclone below could pull a 22 GB base model, and if the host went quiet the
# round never returned. rclone's own `--timeout 5m` (IO idle) bounds a single
# stalled read, but `--retries 3` x `--low-level-retries 10` on top of it means a
# flapping host can still burn hours, and NOTHING bounds a host that is merely
# crawling. So: a wall-clock ceiling and a bytes-per-second floor, the same two
# signals the control plane uses on the docker pull (_job_pull_watchdog_tick).
#
# NOT a health probe and not a box verdict — it kills ONE transfer and names why.
# The verdict reaches the caller in $ASSET_PULL_VERDICT so a timeout is emitted
# as a DIFFERENT terminal reason from a failure (`asset_stage_timeout` /
# `asset_stage_slow` vs `asset_stage_failed`), which is the whole point: the
# 2026-07 incident this pattern exists to prevent logged "pull/extract failed"
# with no cause and hid an InvalidAccessKeyId from a revoked key.
#
# Set JOBD_ASSET_GUARD=0 to restore the pre-guard behavior exactly.
ASSET_PULL_VERDICT=""

# _bytes_from_stats <statsfile> — cumulative bytes rclone reports having moved.
# Parses the `--stats-one-line` NOTICE ("Transferred: 1.234 GiB / 5.000 GiB, ...").
# Empty on no match, which the caller treats as "unmeasured", never as zero.
_bytes_from_stats() {
  local f="$1" tok
  [ -s "$f" ] || return 0
  tok="$(grep -aoE '[0-9]+(\.[0-9]+)? [KMGTP]?i?B / ' "$f" 2>/dev/null | tail -n1)"
  [ -n "$tok" ] || return 0
  printf '%s' "${tok% / }" | awk '
    { n=$1; u=$2; m=1
      if (u=="KiB"||u=="KB") m=1024
      else if (u=="MiB"||u=="MB") m=1048576
      else if (u=="GiB"||u=="GB") m=1073741824
      else if (u=="TiB"||u=="TB") m=1099511627776
      printf "%d\n", n*m }'
}

# _pull_progress_bytes <statsfile> <cache> — best available progress measure.
# The stats file first; on a client that writes none (the local test shim) fall
# back to bytes ON DISK, which is the honest thing anyway. An UNMEASURABLE
# transfer must never be condemned, so a failure here returns nothing and the
# rate check skips the tick rather than reading it as starved — the same house
# rule as the disk precheck's "unreadable evidence never accelerates a
# destructive action".
_pull_progress_bytes() {
  local b
  b="$(_bytes_from_stats "$1")"
  [ -n "$b" ] && { echo "$b"; return 0; }
  _dir_bytes "$2"
}

# _asset_ceiling_s <remote-bytes> — the wall-clock budget for ONE pull round.
# DERIVED, not constant: the same primitive stages a 40 MB tokenizer dir and a
# 22 GB monolith, and any constant generous enough for the second is useless
# against the first. bytes / floor-rate + slack, never below the minimum. At the
# 3 MB/s floor a 22 GB asset gets ~2h05m against a real-world pull of minutes
# (58-101 MB/s on 1 Gbps hosts, up to 1008 on a fat one — WEIGHTS_TRANSPORT_PLAN
# §2/§2b). An UNKNOWN remote size falls back to the flat JOBD_ASSET_TIMEOUT_S:
# we would rather bound it loosely than not at all.
_asset_ceiling_s() {
  local bytes="${1:-}" floor="${JOBD_ASSET_MIN_MBPS:-3}" slack="${JOBD_ASSET_SLACK_S:-300}"
  local minc="${JOBD_ASSET_MIN_TIMEOUT_S:-900}" c
  case "${bytes:-x}" in (*[!0-9]*|"") echo "${JOBD_ASSET_TIMEOUT_S:-14400}"; return 0 ;; esac
  c="$(awk -v b="$bytes" -v f="$floor" -v s="$slack" \
        'BEGIN{ if (f<=0) f=3; printf "%d", b/(f*1048576) + s }')"
  [ "${c:-0}" -lt "$minc" ] 2>/dev/null && c="$minc"
  echo "$c"
}

# _guarded_rclone_pull <statsfile> <cache> <ceiling_s> -- <rclone argv...>
# Runs ONE rclone pull under both guards. Echoes the verdict (ok|timeout|slow|
# failed) on stdout; returns rclone's own rc, or 124 (timeout convention) / 125
# (slow) when a guard killed it.
#
# The watcher waits on a PID WE LAUNCHED (`kill -0 "$pid"`) — never `pgrep -f`,
# which would match this very shell's own argv and spin forever (that footgun
# has stranded sessions on this box repeatedly). The loop is bounded by the
# ceiling itself, so it cannot outlive its budget even if the child never exits.
_guarded_rclone_pull() {
  local statsf="$1" cache="$2" ceiling="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local floor="${JOBD_ASSET_MIN_MBPS:-3}" window="${JOBD_ASSET_MBPS_WINDOW_S:-300}"
  # The tick governs how fast we notice the child EXITED, not how fast we judge
  # it — the rate is only ever evaluated on a full window. So it should be SHORT
  # at the start and lazy afterwards: a flat 1 s tick charges every asset on
  # every job a full second of dead time (measured: it took the jobd suite from
  # ~35 s to 227 s), while a multi-hour weights pull does not care whether we
  # look every 0.05 s or every second.
  #
  # So: start fine and back off to `poll`. GNU sleep takes fractions and
  # busybox's does not, so probe once rather than assume — this file runs on
  # whatever image the box happens to have.
  local poll="${JOBD_ASSET_POLL_S:-1}" tick=1 frac=0
  if sleep 0.05 2>/dev/null; then frac=1; tick=0.05; fi
  local t0 now pid rc anchor_t anchor_b cur rate
  t0="$(date +%s)"

  # The child's stdout must NOT reach ours: the caller reads our stdout as the
  # VERDICT, and a stray rclone line prepended to "timeout" would fail the
  # caller's `case` match and silently un-name the guard. Fold it into the stats
  # file rather than /dev/null — discarding transport output is exactly the
  # antipattern (`2>/dev/null` on the bundle pull) that hid an InvalidAccessKeyId,
  # and the auth grep below scans this whole file.
  "$@" >"$statsf" 2>&1 &
  pid=$!
  anchor_t="$t0"; anchor_b="$(_pull_progress_bytes "$statsf" "$cache")"

  while :; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      [ "$rc" -eq 0 ] && { echo ok; return 0; }
      echo failed; return "$rc"
    fi
    sleep "$tick"
    # back off toward the configured tick: 0.05 -> 0.1 -> 0.2 -> 0.4 -> 0.8 -> poll
    if [ "$frac" = 1 ]; then
      tick="$(awk -v t="$tick" -v p="$poll" 'BEGIN{ t=t*2; if (t>p) t=p; printf "%.2f", t }')"
    fi
    now="$(date +%s)"

    # (a) wall-clock ceiling
    if [ $(( now - t0 )) -ge "$ceiling" ] 2>/dev/null; then
      _kill_pull "$pid"; echo timeout; return 124
    fi

    # (b) throughput floor, over a FULL window and never on one sample. An
    # aggregate rate: hosts shape per TCP FLOW, so a single slow flow inside a
    # healthy transfer must not condemn (same reasoning as BOOT_MIN_MBPS).
    if [ "${floor%%.*}" != "0" ] && [ "$window" -gt 0 ] 2>/dev/null \
       && [ $(( now - anchor_t )) -ge "$window" ]; then
      cur="$(_pull_progress_bytes "$statsf" "$cache")"
      if [ -n "$cur" ] && [ -n "$anchor_b" ]; then
        rate="$(awk -v a="$anchor_b" -v c="$cur" -v s="$(( now - anchor_t ))" -v f="$floor" \
                 'BEGIN{ r=(c-a)/1048576/s; print (r<f) ? "under" : "over" }')"
        if [ "$rate" = "under" ]; then
          _kill_pull "$pid"; echo slow; return 125
        fi
      fi
      anchor_t="$now"; anchor_b="$cur"
    fi
  done
}

_kill_pull() {   # TERM, then KILL if it will not go; always reaped
  local pid="$1" i
  kill -TERM "$pid" 2>/dev/null || true
  for i in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# --- interrupted-transfer garbage: rclone's `*.partial` temps -------------------
# rclone downloads into `<name>.<rand>.partial` and RENAMES into place only once
# the file is whole. It NEVER resumes one: the next `rclone copy` opens a brand
# new temp with a fresh random suffix, so every killed round leaves one more
# corpse behind. That makes a leftover `.partial` two things at once —
#   (1) pure garbage (it costs disk and nothing will ever read it), and
#   (2) the ONLY on-disk trace that says "a transfer was interrupted HERE",
# which is exactly what `_interrupted_transfer_evidence` below reads.
#
# `.b2x-partial-<name>` is NOT the same thing and must never be swept blind: b2x
# preallocates a part-addressed file and RESUMES it from the `.b2x/` state db
# (b2x/state.go:191, :34). Deleting one turns a cheap resume into a full re-pull,
# or races a live transfer. See `_unclaimed_b2x_partials` for the narrow case
# where one is genuinely unowned.
#
# AGE FILTER (<min_age_s>, 0 = no filter): only ever a DELETION guard. In a root
# that belongs to one dead job — its workdir, or an asset cache held under the
# per-asset stage lock — there is no live writer by construction and the caller
# passes 0. In a root shared with a RUNNING sibling arm, a partial being written
# right now belongs to that arm's live pull, so only one older than the window is
# provably nobody's. Classification never applies the filter; see
# `_transfer_quiesce_s` for why the two want opposite thresholds.
#
# $JOBS_DIR is PRUNED throughout: another job's workdir is never ours to sweep,
# and a job scans its own workdir by passing it as the root (find starts there,
# so the prune cannot match).
# (Two spelled-out `find` calls rather than one built from an array: expanding an
# EMPTY array under `set -u` is a bash-4.3 error, and this file runs on whatever
# image the box happens to have.)
_scan_partials() {   # _scan_partials <root> [min_age_s] -> one path per line
  local root="$1" age="${2:-0}" cutoff
  [ -d "$root" ] || return 0
  if [ "${age:-0}" -gt 0 ] 2>/dev/null; then
    cutoff=$(( $(date +%s) - age ))
    find "$root" -path "$JOBS_DIR" -prune -o \
         -type f -name '*.partial' ! -newermt "@$cutoff" -print 2>/dev/null
  else
    find "$root" -path "$JOBS_DIR" -prune -o \
         -type f -name '*.partial' -print 2>/dev/null
  fi
}

# _find_b2x_partials <root> [min_age_s] — raw `.b2x-partial-*` candidates for the
# ownership filter in `_unclaimed_b2x_partials`. Same pruning + age rules.
_find_b2x_partials() {
  local root="$1" age="${2:-0}" cutoff
  [ -d "$root" ] || return 0
  if [ "${age:-0}" -gt 0 ] 2>/dev/null; then
    cutoff=$(( $(date +%s) - age ))
    find "$root" -path "$JOBS_DIR" -prune -o \
         -type f -name '.b2x-partial-*' ! -newermt "@$cutoff" -print 2>/dev/null
  else
    find "$root" -path "$JOBS_DIR" -prune -o \
         -type f -name '.b2x-partial-*' -print 2>/dev/null
  fi
}

_sweep_stale_partials() {   # _sweep_stale_partials <root> [min_age_s] -> count swept
  local root="$1" age="${2:-0}" f n=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    rm -f "$f" 2>/dev/null && n=$((n+1))
  done <<< "$(_scan_partials "$root" "$age")"
  echo "$n"
}

# asset_pull <jobid> <name> <b2prefix> <cache> <mode> [receipt] — the ONE shared
# primitive.
# Linear backoff (JOBD_ASSET_RETRIES rounds, JOBD_ASSET_BACKOFF s step); an
# auth-class failure is NOT transient, so it breaks out early + loud. Neither is
# a blown ceiling or a sub-floor rate: retrying a 2 h budget four more times
# burns 10 h to learn what the first round already established, so both break out
# and surface as their own verdict. rc 0 on a landed pull, 1 otherwise (stderr in
# $cerr for the caller's diagnostic; verdict in $ASSET_PULL_VERDICT).
asset_pull() {
  local jobid="$1" name="$2" b2p="$3" cache="$4" mode="$5" recp="${6:-}"
  local rounds="${JOBD_ASSET_RETRIES:-5}" step="${JOBD_ASSET_BACKOFF:-5}"
  local cerr="$cache/../.$name.pull.err"
  local statsf="$cache/../.$name.pull.stats"   # live rclone --stats lines (boot-health beacon)
  local smark="$STATE_DIR/$jobid.staging"      # the main loop reads this to heartbeat mbps
  local -a op=(copy); [ "$mode" = "sync" ] && op=(sync)
  # TUNED TRANSPORT (WEIGHTS_TRANSPORT_PLAN §TL;DR-3, matching onstart/train.sh's
  # RC_FAST 16/16/64M): stock rclone opens ~4 transfers / 4 streams, so on a
  # per-flow-shaped host (1-16 MB/s/flow, vast-per-flow-image-layering) the big
  # asset crawls; 16 transfers + 16 multi-thread ranged GETs saturate the NIC
  # instead. --stats writes a periodic one-line MB/s NOTICE to stderr that the
  # box-health beacon (JOBD_STATUS heartbeat + asset_throughput event) parses.
  # All env-overridable.
  local -a rc_fast=(--fast-list \
    --transfers "${JOBD_ASSET_TRANSFERS:-16}" \
    --multi-thread-streams "${JOBD_ASSET_STREAMS:-16}" \
    --multi-thread-cutoff 64M \
    --stats 30s --stats-one-line --stats-log-level NOTICE)
  # Keep the completeness marker OFF the box: it is transport metadata, not
  # content. Root-anchored so a payload file that happens to share the name
  # deeper in the tree still lands. (_stage_one_asset_body also rm's it after
  # the pull — `sync` never DELETES an excluded file it already wrote, so an
  # exclusion added to an existing cache needs the sweep to take effect.)
  [ -n "$recp" ] && [ "$recp" != "-" ] && rc_fast+=(--exclude "/$recp")
  [ -n "${JOBD_ASSET_PULLLOG:-}" ] && echo "$name" >> "$JOBD_ASSET_PULLLOG"
  # advertise the current staging asset + its stats file to the main-loop
  # heartbeat (staging_status); removed on every exit path below.
  printf '%s\t%s\n' "$name" "$statsf" > "$smark" 2>/dev/null || true

  # ONE sizing LIST for the whole retry loop, not one per round. Unknown size is
  # fine — _asset_ceiling_s falls back to the flat budget rather than to none.
  local guard="${JOBD_ASSET_GUARD:-1}" rbytes="" ceiling=0 verdict rc
  if [ "$guard" = "1" ]; then
    rbytes="$(_b2_size_bytes "$b2p" 2>/dev/null || true)"
    ceiling="$(_asset_ceiling_s "$rbytes")"
    log "job $jobid: asset '$name' pull guard — ceiling ${ceiling}s$([ -n "$rbytes" ] && echo " (${rbytes}B remote)"), floor ${JOBD_ASSET_MIN_MBPS:-3} MB/s over ${JOBD_ASSET_MBPS_WINDOW_S:-300}s"
  fi
  ASSET_PULL_VERDICT=""

  local i _swept
  for i in $(seq 1 "$rounds"); do
    # SWEEP FIRST, EVERY ROUND. rclone never resumes its `.partial` temps (see
    # _scan_partials), so anything left here is a corpse from a killed round —
    # this one's, a prior job's, or a prior boot's. Left alone they accumulate on
    # the box disk until the box is destroyed, and they are the same files the
    # interrupted-transfer classifier reads, so a stale one can also make a LATER
    # ordinary failure look like a transport interrupt.
    # SAFE UNDER CONCURRENCY BY PLACEMENT: every path into asset_pull runs inside
    # the per-asset stage lock (stage_one_asset flocks $ASSETS_DIR/.<name>.stage.lock
    # around _stage_one_asset_body), so no sibling arm can be mid-pull into THIS
    # cache while we delete here — which is why the sweep passes min_age 0 and
    # can afford to be unconditional.
    _swept="$(_sweep_stale_partials "$cache")"
    [ "${_swept:-0}" -gt 0 ] 2>/dev/null && \
      log "job $jobid: asset '$name' swept ${_swept} stale rclone .partial temp(s) before round $i/$rounds"
    : > "$cerr" 2>/dev/null || true
    : > "$statsf" 2>/dev/null || true
    # trailing-slash src AND dest: dodge rclone's flaky B2 HEAD dest-check.
    # rclone stderr carries BOTH the --stats NOTICE lines (the live beacon) and
    # any auth/error output; capture it to $statsf. On FAILURE mirror it into
    # $cerr so the auth-class diagnostic below is byte-for-byte the prior logic.
    # --- b2x first; the rclone line below is unchanged and is the fallback ----
    # This is the highest-volume transfer on the fleet: every base model,
    # dataset and corpus for every job comes through here, up to 22 GB. rclone's
    # effective concurrency is min(streams, ceil(size/64Mi)) because
    # --multi-thread-chunk-size is never set, so the tuned 16 above is really 16
    # flows at best; b2x plans parts from the object size and drains them from
    # one global queue.
    #
    # b2x carries the SAME two guards this function wraps rclone in, so the
    # attempt is bounded identically: --deadline is $ceiling, B2X_MIN_MBPS is
    # the same floor. Its exits 7/8 are therefore the same verdicts, and are
    # treated as terminal for the same stated reason the rclone ones are — a
    # blown budget or a starved host is a property of the HOST, not of this
    # attempt, and does not improve by being retried through a SLOWER client.
    # Any OTHER b2x failure falls through to rclone, so a b2x bug can never make
    # this site worse than it was. JOBD_ASSET_B2X=0 disables the attempt.
    verdict=""; rc=1
    if [ "${JOBD_ASSET_B2X:-1}" = "1" ] && command -v b2x_pull >/dev/null 2>&1; then
      B2X_MIN_MBPS="${JOBD_ASSET_MIN_MBPS:-3}" \
      B2X_MIN_MBPS_WINDOW_S="${JOBD_ASSET_MBPS_WINDOW_S:-300}" \
        b2x_pull "$B2/$b2p/" "$cache/" --deadline "${ceiling}s" 2>"$statsf"
      rc=$?
      case "$rc" in
        0) verdict=ok ;;
        7) verdict=timeout ;;
        8) verdict=slow ;;
        *) verdict="" ;;   # fall through to rclone
      esac
      [ -n "$verdict" ] || log "job $jobid: asset '$name' b2x exit $rc — retrying this round with rclone"
    fi
    if [ -n "$verdict" ]; then
      : # b2x reached a verdict (ok / timeout / slow); skip the rclone attempt
    elif [ "$guard" = "1" ]; then
      verdict="$(_guarded_rclone_pull "$statsf" "$cache" "$ceiling" -- \
                   rclone "${op[@]}" "${rc_fast[@]}" "$B2/$b2p/" "$cache/")"
      rc=$?
    else
      rclone "${op[@]}" "${rc_fast[@]}" "$B2/$b2p/" "$cache/" 2>"$statsf"
      rc=$?; verdict=$([ "$rc" -eq 0 ] && echo ok || echo failed)
    fi
    if [ "$rc" -eq 0 ]; then
      ASSET_PULL_VERDICT=ok
      rm -f "$cerr" "$smark" 2>/dev/null || true
      return 0
    fi
    cp -f "$statsf" "$cerr" 2>/dev/null || true
    # A guard verdict is a property of the HOST/budget, not of this attempt, so
    # it does not improve by being retried — surface it now, loudly and named.
    case "$verdict" in
      timeout|slow)
        local _gt; _gt="$(grep -av 'Transferred:' "$cerr" 2>/dev/null | tr '\n' ' ' | tail -c 200)"
        ASSET_PULL_VERDICT="$verdict"
        if [ "$verdict" = timeout ]; then
          log "job $jobid: asset '$name' pull TIMEOUT after ${ceiling}s (round $i/$rounds; JOBD_ASSET_TIMEOUT_S/_MIN_TIMEOUT_S): ${_gt:-<no rclone stderr>}"
        else
          log "job $jobid: asset '$name' pull SLOW HOST — under ${JOBD_ASSET_MIN_MBPS:-3} MB/s for a full ${JOBD_ASSET_MBPS_WINDOW_S:-300}s window (round $i/$rounds; JOBD_ASSET_MIN_MBPS): ${_gt:-<no rclone stderr>}"
        fi
        rm -f "$smark" 2>/dev/null || true
        return 1 ;;
    esac
    # drop the periodic --stats "Transferred:" lines from the log snippet so the
    # real error (not a stats line) is what surfaces; the auth grep below still
    # scans the WHOLE file, so detection is unchanged.
    local tail_; tail_="$(grep -av 'Transferred:' "$cerr" 2>/dev/null | tr '\n' ' ' | tail -c 200)"
    if grep -qiE 'InvalidAccessKeyId|SignatureDoesNotMatch|AccessDenied|Unauthorized|not valid| 403 ' "$cerr" 2>/dev/null; then
      log "job $jobid: asset '$name' pull B2 AUTH FAILURE (dead/rotated key): $tail_"
      : > "$CRED_DIR/.cred_refresh_now" 2>/dev/null || true   # broker refresh hint (§2.6)
      ASSET_PULL_VERDICT=auth
      rm -f "$smark" 2>/dev/null || true
      return 1     # non-transient — do not burn the remaining rounds
    fi
    log "job $jobid: asset '$name' pull failed (round $i/$rounds): $tail_"
    [ "$i" -lt "$rounds" ] && sleep $(( i * step ))
  done
  ASSET_PULL_VERDICT=failed
  rm -f "$smark" 2>/dev/null || true
  return 1
}

# =============================================================================
# COMPLETENESS RECEIPTS (`receipt:`) — the PUBLISHER's answer, not the box's
# =============================================================================
# A published prefix (a merged model, a relayout) carries a marker its publisher
# writes LAST, after every payload byte (tools/witness/jobs/*/b2_transport.sh
# `push`: payload, read-back, THEN `PUSHED.json` via rcat). Its presence is
# therefore the only cheap proof that a restore is not racing a push.
#
# THIS IS NOT `.complete`, and neither marker can do the other's job:
#   .complete  written by THIS box, a local byte total, answers "did I already
#              land these bytes?". Blind to a truncated PUBLISH — those bytes
#              pull cleanly and the total it records is the truncation's.
#   receipt    written by the PUBLISHER on B2, answers "is the remote whole?".
#              Blind to a truncated PULL — which is exactly what .complete, the
#              `require:` globs and the index shard check already cover.
# So the receipt gates the pull and the existing checks gate the landing.
#
# The marker is EXCLUDED from the staged dir. It shares the prefix with the
# payload (so the presence check is one read), but it is transport metadata:
# landing it adds a file the consumer's fingerprint has never seen and turns
# every restore into an unexpected-file failure — the same class of bug the
# base-model gates `--ignore` `.complete` for.

# asset_receipt_expect <b2prefix> <receipt> — echo the file count the receipt
# CLAIMS, or nothing. rc 1 = the prefix is NOT published (absent marker, or a
# body that explicitly says complete:false); rc 0 = published, or a marker we
# could not parse. UNPARSEABLE IS NOT A REFUSAL: a marker we cannot read is not
# evidence of an incomplete publish, the same house rule as the disk precheck's
# unreadable `df`. Only an ABSENT marker and an explicit denial refuse.
asset_receipt_expect() {
  local b2p="$1" recp="$2" body
  body="$(timeout "${JOBD_RECEIPT_TIMEOUT_S:-60}" rclone cat "$B2/$b2p/$recp" 2>/dev/null)" || return 1
  printf '%s' "$body" | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    assert isinstance(d, dict)
except Exception:
    sys.exit(0)                      # no claim; presence alone stands
if d.get("complete") is False:
    sys.exit(3)                      # an explicit denial IS a refusal
n = d.get("files")
if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
    print(n)
' 2>/dev/null
  # PIPESTATUS[1], not $?: the `printf` head of the pipe always succeeds, and
  # exit 3 is the parser's explicit "complete: false".
  [ "${PIPESTATUS[1]:-0}" = "3" ] && return 1
  return 0
}

# asset_count_files <cache> <receipt> — regular files staged, excluding the
# marker itself. Counted RECURSIVELY on purpose: the publisher counts its own
# top level, so a recursive count is >= its claim for a flat dir and strictly
# greater for a nested one. The caller refuses only on `staged -lt claimed`, so
# that direction can only ever forgive, never invent a refusal.
asset_count_files() {
  local cache="$1" recp="$2"
  [ -d "$cache" ] || { echo 0; return 0; }
  # `! -path`, not `! -name`: the exact receipt path, so a payload file that
  # happens to share the marker's basename deeper in the tree still counts.
  # Only an UNDER-count can produce a false refusal, so the filter must be as
  # narrow as possible. (The caller rm's the marker first, so this is belt.)
  find "$cache" -type f ! -path "$cache/$recp" 2>/dev/null | wc -l | tr -d ' '
}

# asset_check_require <cache> <wdir> <name> — postconditions after a pull. Echoes
# a one-line reason + returns 1 on the FIRST failure; silent rc 0 on success.
# (1) every `require:` glob must match at least one path; (2) automatic index-aware
# shard completeness — if a *.index.json exists, every shard it names must be
# present (ported from ensure_base_model.sh:86-131).
asset_check_require() {
  local cache="$1" wdir="$2" name="$3"
  local reqf="$wdir/.asset_require/$name" pat miss n
  if [ -s "$reqf" ]; then
    miss=""
    while IFS= read -r pat; do
      [ -n "$pat" ] || continue
      n=$( (shopt -s globstar nullglob dotglob; cd "$cache" 2>/dev/null || exit 0
            c=0; for f in $pat; do [ -e "$f" ] && c=$((c+1)); done; echo "$c") )
      [ "${n:-0}" -gt 0 ] || miss="$miss $pat"
    done < "$reqf"
    [ -z "$miss" ] || { echo "require globs unmatched:$miss"; return 1; }
  fi
  # index-aware weight-shard completeness (automatic; only when an index exists).
  # The walk itself lives in _index_missing_shards so the interrupted-transfer
  # self-heal can reuse it VERBATIM — the gate that condemns a truncated pull and
  # the heal that repairs it must never disagree about what "complete" means.
  local idx missing
  idx="$( (shopt -s nullglob globstar; cd "$cache" 2>/dev/null || exit 0
           for f in *.index.json **/*.index.json; do
             [ -f "$f" ] && { echo "$f"; break; }; done) )"
  if [ -n "$idx" ] && missing="$(_index_missing_shards "$cache/$idx")"; then
    echo "index shards missing: $missing"; return 1
  fi
  return 0
}

# _index_shard_names <index.json> — the shard filenames a HF-style *.index.json
# names, one per line. Silent + empty on an unreadable/unparseable file: a file
# we could not read is not evidence of anything (the same house rule as the disk
# precheck's unreadable `df`).
_index_shard_names() {
  "$PY" - "$1" <<'PY' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for s in sorted(set((d.get("weight_map") or {}).values())):
    print(s)
PY
}

# _index_missing_shards <index.json> — shard names the index DECLARES that are
# not on disk beside it, space-joined on stdout; rc 0 = at least one missing
# (i.e. this model dir is INCOMPLETE), rc 1 = complete, or no parseable index.
# Shard paths resolve relative to the index's own directory, which is what HF's
# weight_map means and what ensure_base_model.sh:86-131 did.
_index_missing_shards() {
  local idx="$1" idxdir shards sh missing=""
  [ -f "$idx" ] || return 1
  idxdir="$(dirname "$idx")"
  shards="$(_index_shard_names "$idx")"
  [ -n "$shards" ] || return 1
  while IFS= read -r sh; do
    [ -n "$sh" ] || continue
    [ -f "$idxdir/$sh" ] || missing="$missing $sh"
  done <<< "$shards"
  [ -n "$missing" ] || return 1
  printf '%s' "${missing# }"
  return 0
}

# =============================================================================
# INTERRUPTED TRANSFER: classify, self-heal, retry ONCE (defect #77, 2026-08-09)
# =============================================================================
# THE FAILURE. An rclone weights pull is killed mid-flight (host hiccup, a
# transfer guard firing, the box's network flapping). It leaves `<name>.<rand>
# .partial` temps and — the part that matters — a `*.index.json` that ALREADY
# LANDED and names shards that never did. The bundle's own completeness gate then
# reports every shard MISSING and `die`s; the bundle maps that to rc=4; jobd took
# EVERY nonzero rc terminal (mark_terminal), so the ticket was skipped forever.
# A whole job died of a transport hiccup, and no counter anywhere covered it:
# max_restarts only applies to NON-terminal exits, JOBD_PREEMPT_CAP to preempts,
# JOBD_ASSET_RETRIES to the declarative `assets:` lane only. This is the missing
# bounded retry for an ENTRYPOINT-step transfer failure.
#
# WHY A NAIVE RETRY IS WORTHLESS — and what makes this one work. Every bundle
# wraps its pull in a presence guard:
#     if [ ! -f "$dir/model.safetensors.index.json" ]; then ... rclone copy ...
# (driftr3 run.sh:383/449). The index is small and lands EARLY, so after an
# interrupted pull that guard is ARMED: retry #1 SKIPS the pull entirely, fails
# the same gate, and every further retry does the same thing forever. The retry
# is only worth anything because jobd SELF-HEALS first — when an index names
# shards that are absent we delete the orphan partials AND the index, which
# re-arms the bundle's own guard so the next attempt really re-pulls. rclone
# verifies and skips the shards that DID land, so the repeat pull is cheap.
#
# WHAT IT REFUSES TO DO.
#   * Classify on the rc integer. rc=4 is b2x's own "404, does not exist"
#     (b2x/main.go:46-54) and entrypoints use small rcs for everything; an
#     rc-only rule would retry every ordinary bug on the box. EVIDENCE ON THE
#     FILESYSTEM, or no retry — that is the whole discipline here.
#   * Retry into a full disk. ENOSPC leaves partials INDISTINGUISHABLE from an
#     interrupt, and a retry just buys the same failure twice. Insufficient free
#     space is its own terminal reason, checked BEFORE the heal deletes anything.
#   * Retry over corruption. A shard sitting at its FINAL path whose recorded
#     sha256 disagrees is mixed evidence — something renamed bad bytes into
#     place, and rclone may well skip re-fetching it. Terminal.
#   * Spend the crash budget. It rides its own $STATE_DIR/<jid>.transfer_retries
#     counter (cap 1, hard max 2), exactly as .preempts is separate from
#     .attempts — three outbids must not fail a healthy job, and neither must one
#     flaky pull.
#   * Outrank a preemption. The classifier sits STRICTLY AFTER run_job_body's
#     preempt block, which `return`s: a box stop also kills an rclone pull and
#     also leaves partials, and that death must bump .preempts, never this.

_transfer_retry_cap() {   # bounded, and bounded ABOVE by the code, not by config
  local c="${JOBD_TRANSFER_RETRIES:-1}"
  case "$c" in (*[!0-9]*|"") c=1 ;; esac
  [ "$c" -gt 2 ] && c=2
  echo "$c"
}

# _transfer_scan_roots <wdir> — the dirs a transfer could have left garbage in,
# one per line as "<root>\t<own>", where own=1 means the root belongs to THIS
# dead job alone and own=0 means it is shared with whatever else runs on the box.
#
#   $wdir      the job's OWN workdir. Its entrypoint is dead, so nothing under it
#              is being written and every partial there is provably ours.
#   $ROOT      the box workspace — SHARED. Bundles stage weights OUTSIDE the
#              workdir precisely so a resume does not re-pull 34 GB (driftr3's
#              /workspace/driftr3/models), and that is the exact case this fix
#              exists for, so $ROOT has to be in scope. $JOBS_DIR is PRUNED by
#              every scan: a sibling arm's workdir is not ours to judge or heal.
#   extras     JOBD_TRANSFER_SCAN_DIRS, ':'-separated, treated as shared.
# $ASSETS_DIR needs no separate entry — it is under $ROOT.
_transfer_scan_roots() {
  local wdir="$1" d
  [ -n "$wdir" ] && [ -d "$wdir" ] && printf '%s\t1\n' "$wdir"
  [ -d "$ROOT" ] && printf '%s\t0\n' "$ROOT"
  local IFS=':'
  for d in ${JOBD_TRANSFER_SCAN_DIRS:-}; do
    [ -n "$d" ] && [ -d "$d" ] && printf '%s\t0\n' "$d"
  done
  return 0
}

# _transfer_quiesce_s <jobid> <own> — how old a file in this root must be before
# we are willing to DELETE it. Evidence never uses this; deletion always does.
#
# THE TENSION, and why it resolves the way it does. A freshly-killed pull leaves
# a partial that is SECONDS old — so an age filter on EVIDENCE would blind the
# classifier to the exact failure it exists for. But a sibling arm's LIVE pull
# also leaves fresh partials, and deleting one of those (or, far worse, its
# index) breaks a job that is doing nothing wrong. Evidence and deletion want
# opposite thresholds, so they get different ones:
#   * classification reads every partial regardless of age. Over-firing costs at
#     most one extra attempt, and it is self-limiting — the heal removes the
#     partials, so the retry's own failure has no stale evidence to feed on.
#   * deletion is quiesced ONLY when a sibling job is actually running on this
#     box. With no other `.running` file there is no other writer by
#     construction, which is the common case (single arm, or every arm already
#     dead), and the heal is immediate and complete.
_transfer_quiesce_s() {
  local jobid="$1" own="${2:-0}" rf n=0
  [ "$own" = "1" ] && { echo 0; return 0; }        # our own dead workdir
  for rf in "$STATE_DIR"/*.running; do
    [ -f "$rf" ] || continue
    [ "$(basename "${rf%.running}")" = "$jobid" ] && continue
    n=$((n+1))
  done
  [ "$n" -gt 0 ] || { echo 0; return 0; }          # nobody else is writing
  local age="${JOBD_PARTIAL_STALE_S:-120}"
  case "$age" in (*[!0-9]*|"") age=120 ;; esac
  echo "$age"
}

# _unclaimed_b2x_partials <root> <min_age_s> — `.b2x-partial-<name>` files that
# NO b2x transfer will ever resume. Deliberately narrow, because the default
# reading of one of these is the OPPOSITE of garbage:
#   * a `.b2x/` state dir at or above the file means b2x holds resume state for
#     it (b2x/state.go:34) — that is a checkpoint-lane transfer that will pick
#     itself back up, and touching it converts a cheap resume into a full
#     re-pull;
#   * so we only count one sitting in a WEIGHTS-shaped directory — beside a
#     `*.index.json`, or inside the asset cache — with no state dir above it.
# When in doubt this yields nothing and the verdict falls back to `*.partial`,
# which rclone never resumes and which is therefore always safe to read.
_unclaimed_b2x_partials() {
  local root="$1" age="${2:-0}" f d claimed
  [ -d "$root" ] || return 0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    d="$(dirname "$f")"
    # weights-shaped? (an index beside it, or under the box asset cache)
    case "$d/" in
      "$ASSETS_DIR"/*) ;;
      *) ls "$d"/*.index.json >/dev/null 2>&1 || continue ;;
    esac
    # any b2x state dir between the file and the scan root => it is RESUME STATE
    claimed=0
    while :; do
      [ -d "$d/.b2x" ] && { claimed=1; break; }
      [ "$d" = "$root" ] || [ "$d" = "/" ] || [ -z "$d" ] && break
      d="$(dirname "$d")"
    done
    [ "$claimed" = "1" ] || printf '%s\n' "$f"
  done <<< "$(_find_b2x_partials "$root" "$age")"
  return 0
}

# _interrupted_transfer_evidence <jobid> <wdir> — rc 0 when the FILESYSTEM shows
# an interrupted transfer. Sets TRANSFER_EVIDENCE to a short human summary.
# Callers must have already established rc != 0; this function deliberately does
# not look at the rc at all, so it cannot be tempted into an rc-only rule.
TRANSFER_EVIDENCE=""
_interrupted_transfer_evidence() {
  local jobid="$1" wdir="$2" root own n_p=0 n_b=0 first="" c
  TRANSFER_EVIDENCE=""
  # age 0 everywhere: a pull killed one second ago is the case this exists for,
  # so evidence must not be quiesced (see _transfer_quiesce_s for the argument).
  while IFS=$'\t' read -r root own; do
    [ -n "$root" ] || continue
    while IFS= read -r c; do
      [ -n "$c" ] || continue
      n_p=$((n_p+1)); [ -n "$first" ] || first="$c"
    done <<< "$(_scan_partials "$root" 0)"
    while IFS= read -r c; do
      [ -n "$c" ] || continue
      n_b=$((n_b+1)); [ -n "$first" ] || first="$c"
    done <<< "$(_unclaimed_b2x_partials "$root" 0)"
  done <<< "$(_transfer_scan_roots "$wdir")"
  [ $(( n_p + n_b )) -gt 0 ] || return 1
  TRANSFER_EVIDENCE="${n_p} orphan .partial + ${n_b} unclaimed .b2x-partial (e.g. ${first##*/})"
  return 0
}

# _transfer_disk_exhausted <path> — rc 0 when free space is under the floor.
# UNREADABLE df is NOT evidence of a full disk (house rule: a measurement we
# could not take never accelerates a destructive verdict), so it returns rc 1 and
# the retry proceeds. Runs BEFORE the heal deletes anything, so the number it
# reports is the number the operator would see on the box.
TRANSFER_FREE_GB=""
_transfer_disk_exhausted() {
  local p="$1" floor="${JOBD_TRANSFER_MIN_FREE_GB:-5}" kb
  case "$floor" in (*[!0-9]*|"") floor=5 ;; esac
  TRANSFER_FREE_GB=""
  kb="$(_free_kb "$p")" || return 1
  TRANSFER_FREE_GB=$(( kb / 1048576 ))
  [ "$TRANSFER_FREE_GB" -lt "$floor" ] 2>/dev/null
}

# _transfer_corruption <wdir> — a file at its FINAL path whose RECORDED sha256
# disagrees with its bytes. That is mixed evidence, not an interrupt: something
# renamed bad bytes into place, and a re-pull may skip the file entirely (rclone
# compares size/mtime, not content). Terminal, never retried.
#
# We can only see this where the pull shipped a manifest — `SHA256SUMS` beside
# the files, or a `<file>.sha256` sidecar. With neither we say NOTHING: a check
# we could not run is not evidence of corruption, and inventing one here would
# strand jobs whose transports simply do not publish hashes.
TRANSFER_CORRUPTION=""
_transfer_corruption() {
  local root own f d want got name line
  TRANSFER_CORRUPTION=""
  while IFS=$'\t' read -r root own; do
    [ -n "$root" ] && [ -d "$root" ] || continue
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      case "$f" in
        *.sha256)
          [ -f "${f%.sha256}" ] || continue
          want="$(awk '{print $1; exit}' "$f" 2>/dev/null)"
          [ -n "$want" ] || continue
          got="$(sha256sum "${f%.sha256}" 2>/dev/null | cut -d' ' -f1)"
          [ -n "$got" ] && [ "$got" != "$want" ] && {
            TRANSFER_CORRUPTION="${f%.sha256}"; return 0; } ;;
        */SHA256SUMS)
          d="$(dirname "$f")"
          while IFS= read -r line; do
            want="${line%% *}"; name="${line##* }"
            [ -n "$want" ] && [ -n "$name" ] && [ -f "$d/$name" ] || continue
            got="$(sha256sum "$d/$name" 2>/dev/null | cut -d' ' -f1)"
            [ -n "$got" ] && [ "$got" != "$want" ] && {
              TRANSFER_CORRUPTION="$d/$name"; return 0; }
          done < "$f" ;;
      esac
    done <<< "$(find "$root" -path "$JOBS_DIR" -prune -o \
                  -type f \( -name '*.sha256' -o -name 'SHA256SUMS' \) -print 2>/dev/null)"
  done <<< "$(_transfer_scan_roots "$1")"
  return 1
}

# _heal_interrupted_transfer <jobid> <wdir> — THE SKIP-GUARD DEFEAT. Echoes a
# one-line summary of what it repaired.
#
# For every `*.index.json` under the scan roots whose declared shards are not all
# present, delete that directory's orphan partials AND the index itself. The
# index deletion is the whole point: the bundle's `if [ ! -f "$dir/…index.json" ]`
# guard is what makes a bare retry a no-op, and removing the index is what
# re-arms it. Any remaining orphan partials in the roots go too — rclone will
# never resume them, so they are garbage whether or not they sat beside an index.
#
# THE ONE THING IT WILL NOT TOUCH is a directory a sibling job may be pulling
# into right now: when another `.running` exists, a shared root's incomplete
# index is only healed if that directory has been QUIESCENT for
# JOBD_PARTIAL_STALE_S. An incomplete index with a live partial beside it is not
# a corpse, it is a download in progress, and deleting it would break a job that
# is doing nothing wrong. See _transfer_quiesce_s.
_heal_interrupted_transfer() {
  local jobid="$1" wdir="$2" root own age idx d missing n_idx=0 n_part=0 f c live
  while IFS=$'\t' read -r root own; do
    [ -n "$root" ] && [ -d "$root" ] || continue
    age="$(_transfer_quiesce_s "$jobid" "$own")"
    while IFS= read -r idx; do
      [ -n "$idx" ] || continue
      missing="$(_index_missing_shards "$idx")" || continue
      d="$(dirname "$idx")"
      if [ "${age:-0}" -gt 0 ] 2>/dev/null; then
        # is anything in this dir still being written? (a partial NEWER than the
        # quiescence window == a live transfer == hands off)
        live="$(find "$d" -maxdepth 1 -type f \
                  \( -name '*.partial' -o -name '.b2x-partial-*' \) \
                  -newermt "@$(( $(date +%s) - age ))" -print -quit 2>/dev/null)"
        if [ -n "$live" ]; then
          log "transfer heal: SKIP $d — a sibling job is running and $live was written inside the last ${age}s (live transfer, not a corpse)"
          continue
        fi
      fi
      log "transfer heal: $idx names shards that never landed ($missing) — dropping the index so the bundle's own pull guard re-arms"
      while IFS= read -r f; do
        [ -n "$f" ] || continue
        rm -f "$f" 2>/dev/null && n_part=$((n_part+1))
      done <<< "$(find "$d" -maxdepth 1 -type f \
                    \( -name '*.partial' -o -name '.b2x-partial-*' \) 2>/dev/null)"
      rm -f "$idx" 2>/dev/null && n_idx=$((n_idx+1))
    done <<< "$(find "$root" -path "$JOBS_DIR" -prune -o \
                  -type f -name '*.index.json' -print 2>/dev/null)"
    # whatever partials survive the per-index pass are still pure garbage
    c="$(_sweep_stale_partials "$root" "$age")"
    n_part=$(( n_part + ${c:-0} ))
  done <<< "$(_transfer_scan_roots "$wdir")"
  echo "${n_idx} stale index.json + ${n_part} partial(s) removed"
}

# _link_asset_dest <run> <cache> <dest> — symlink dest -> cache when dest is set
# and differs from the cache path (absolute dest is validated under /workspace;
# relative dest resolves inside the job workdir so the entrypoint's cwd finds it).
_link_asset_dest() {
  local run="$1" cache="$2" dest="$3" target
  [ -n "$dest" ] && [ "$dest" != "-" ] || return 0
  case "$dest" in
    /*) target="$dest" ;;
    *)  target="$run/$dest" ;;
  esac
  # SECURITY: never materialize a symlink outside the job sandbox ($ROOT,
  # default /workspace). The comment above long CLAIMED "validated under
  # /workspace" but the code took an absolute dest verbatim; enforce it now.
  # Reject a '..' segment (defeats the prefix check) and any target that does
  # not resolve under $ROOT. See CREDENTIAL_LIFECYCLE.md.
  case "/$target/" in
    */../*) log "asset dest '$dest' contains '..' — refusing to link"; return 1 ;;
  esac
  case "$target/" in
    "$ROOT"/*) : ;;
    *) log "asset dest '$dest' escapes $ROOT — refusing to link"; return 1 ;;
  esac
  [ "$target" = "$cache" ] && return 0
  # already pointing at this cache? do NOTHING. `rm -rf` + `ln` leaves a window
  # where the path does not exist, and with CONCURRENT jobs sharing an asset name
  # a peer's entrypoint may be loading a model through this very symlink while we
  # re-stage it (all three 2026-07-30 waves declared the same asset names/dests).
  if [ -L "$target" ] && [ "$(readlink "$target")" = "$cache" ]; then return 0; fi
  mkdir -p "$(dirname "$target")" 2>/dev/null || true
  rm -rf "$target" 2>/dev/null || true
  ln -sfn "$cache" "$target" 2>/dev/null || true
}

# stage_one_asset <...> — PER-ASSET-NAME lock around the staging body below.
#
# $ASSETS_DIR/<name> and its `.complete` marker are BOX-GLOBAL mutable state, and
# concurrent jobs routinely share asset names (all three 2026-07-30 frontier
# waves declared base / adapter-t / adapter-c). Unserialized, N runners each
# `rclone copy` the SAME multi-GB prefix into the SAME directory at once: N times
# the bandwidth, interleaved partial files, and a marker written from a byte
# total another writer is still changing. Serialized, the first holder pulls and
# the rest fall through the skip-if-complete fast path — which is also the only
# reason a 3-arm wave on a COLD box does not pull the base model three times.
#
# The lock is per NAME, taken one at a time (never held across another
# acquisition), so jobs staging different assets never wait on each other and
# there is no hold-and-wait cycle. flock absent => unserialized, as before.
stage_one_asset() {
  mkdir -p "$ASSETS_DIR"
  if command -v flock >/dev/null 2>&1; then
    ( flock 8; _stage_one_asset_body "$@" ) 8>"$ASSETS_DIR/.$4.stage.lock"
  else
    _stage_one_asset_body "$@"
  fi
}

# ... -> rc 0 staged (or optional-tolerated); rc 1 = HARD FAIL (the caller takes
# the job terminal). Emits the terminal `failed` itself so the reason is precise.
_stage_one_asset_body() {
  local jobid="$1" wdir="$2" run="$3" name="$4" b2p="$5" mode="$6" opt="$7" dest="$8"
  local recp="${9:-}"; [ "$recp" = "-" ] && recp=""
  local cache="$ASSETS_DIR/$name" marker="$ASSETS_DIR/.$name.complete"
  local localmark="$ASSETS_DIR/.$name.local"

  # PRE-SEEDED LOCAL ASSET (LOCAL_GPU_LANE.md). The operator pointed this asset's
  # cache at a directory that already exists on this machine — the base model in
  # ~/.cache/huggingface, an adapter in ~/checkpoints — via `job run-local
  # --asset NAME=DIR`, which writes $cache as a SYMLINK plus this marker.
  # We MUST NOT pull: the source is not a B2 mirror, and `rclone copy` would
  # write straight THROUGH the symlink into the operator's model cache. The
  # `.complete` byte-total path cannot express this either — _dir_bytes does not
  # follow the link, reads 0, and would re-pull every time.
  # Integrity is NOT weakened: `require:` postconditions + the index-aware shard
  # check + the `dest` link all still run below, against the real bytes, so a
  # wrong local path fails in exactly the place a truncated B2 pull does.
  # A rented box can never have this marker, so this is a strict no-op there.
  if [ -f "$localmark" ]; then
    log "job $jobid: asset '$name' PRE-SEEDED locally ($(readlink "$cache" 2>/dev/null || echo "$cache")) — skip pull"
    local lrerr
    if ! lrerr="$(asset_check_require "$cache" "$wdir" "$name")"; then
      if [ "$opt" = "1" ]; then
        log "job $jobid: OPTIONAL asset '$name' postcondition unmet ($lrerr) — continuing"
        return 0
      fi
      log "job $jobid: pre-seeded asset '$name' postcondition FAILED ($lrerr) — job cannot start"
      emit "$jobid" failed --field reason="asset_stage_failed:$name" \
        --field asset="$name" --field detail="$lrerr"
      mark_terminal "$jobid" failed
      return 1
    fi
    _link_asset_dest "$run" "$cache" "$dest"
    return 0
  fi
  mkdir -p "$cache"

  # skip-if-complete: a .complete byte-total marker whose count still matches the
  # cache => reuse it (survives park/resume; dedupes across arms). No re-pull.
  if [ -f "$marker" ]; then
    local want got
    want="$(cat "$marker" 2>/dev/null || echo 0)"
    got="$(_dir_bytes "$cache")"
    if [ -n "$want" ] && [ "$want" -gt 0 ] 2>/dev/null && [ "$got" -ge "$want" ] 2>/dev/null; then
      log "job $jobid: asset '$name' cache complete (${got}B) — skip pull"
      _link_asset_dest "$run" "$cache" "$dest"
      return 0
    fi
  fi

  # RECEIPT GATE — fail fast, BEFORE a byte moves. One small read answers "is
  # the remote prefix published at all?", so a restore racing a push costs one
  # LIST instead of a 52 GiB download plus whatever gate would have caught it
  # afterwards. Deliberately BELOW the skip paths above: an asset this box
  # already holds must not be made to depend on B2 being reachable.
  local _want_files=""
  if [ -n "$recp" ]; then
    if ! _want_files="$(asset_receipt_expect "$b2p" "$recp")"; then
      if [ "$opt" = "1" ]; then
        log "job $jobid: OPTIONAL asset '$name' has no completeness receipt ($recp) at $b2p — continuing without it"
        return 0
      fi
      log "job $jobid: asset '$name' NOT PUBLISHED — no completeness receipt '$recp' under $b2p (the publisher writes it LAST, so a prefix without one is truncated or still uploading) — job cannot start"
      emit "$jobid" failed --field reason="asset_receipt_missing:$name" \
        --field asset="$name" --field receipt="$recp" --field b2="$b2p"
      mark_terminal "$jobid" failed
      return 1
    fi
    log "job $jobid: asset '$name' receipt '$recp' present${_want_files:+ (claims ${_want_files} file(s))}"
  fi

  local _pull_t0; _pull_t0="$(date +%s)"
  if ! asset_pull "$jobid" "$name" "$b2p" "$cache" "$mode" "$recp"; then
    if [ "$opt" = "1" ]; then
      log "job $jobid: OPTIONAL asset '$name' pull failed (${ASSET_PULL_VERDICT:-failed}) — continuing without it"
      return 0
    fi
    # A TIMEOUT and a SLOW HOST get their own terminal reasons. `asset_stage_failed`
    # reads as "the pull is broken" and sends the operator to retry the same job on
    # the same shape of box; `asset_stage_timeout` / `asset_stage_slow` say the
    # transport was working and the HOST was not, which is a different action
    # (re-rent elsewhere) and a different lesson for hosts.py.
    local _reason="asset_stage_failed:$name"
    case "${ASSET_PULL_VERDICT:-}" in
      timeout) _reason="asset_stage_timeout:$name" ;;
      slow)    _reason="asset_stage_slow:$name" ;;
    esac
    log "job $jobid: asset '$name' pull FAILED (${ASSET_PULL_VERDICT:-failed}, non-optional) — job cannot start"
    emit "$jobid" failed --field reason="$_reason" --field asset="$name" \
      --field verdict="${ASSET_PULL_VERDICT:-failed}"
    mark_terminal "$jobid" failed
    return 1
  fi

  # Belt AND braces on the exclusion: --exclude keeps the marker off the wire,
  # this keeps it off the disk. They are not redundant — a cache staged before
  # the `receipt:` declaration existed, or by an rclone whose filter semantics
  # differ, can still hold one, and `sync` will not delete a file its own filter
  # hides from it. Runs BEFORE the count check so the marker never counts itself.
  [ -n "$recp" ] && rm -f "$cache/$recp" 2>/dev/null
  # COUNT CORROBORATION. The receipt's `files` is the publisher's own read-back
  # count; fewer landed here means the pull dropped something the require: globs
  # and the shard index are not watching. `-lt` (not `-ne`): a recursive local
  # count is >= a top-level published one by construction, so only a SHORTFALL
  # can fire. No claim in the body => no check, never a refusal.
  if [ -n "$recp" ] && [ -n "$_want_files" ] && [ "$_want_files" -gt 0 ] 2>/dev/null; then
    local _got_files; _got_files="$(asset_count_files "$cache" "$recp")"
    if [ "${_got_files:-0}" -lt "$_want_files" ] 2>/dev/null; then
      if [ "$opt" = "1" ]; then
        log "job $jobid: OPTIONAL asset '$name' staged ${_got_files} file(s), receipt claims ${_want_files} — continuing"
        return 0
      fi
      log "job $jobid: asset '$name' INCOMPLETE — staged ${_got_files} file(s), receipt '$recp' claims ${_want_files} — job cannot start"
      emit "$jobid" failed --field reason="asset_receipt_mismatch:$name" \
        --field asset="$name" --field receipt="$recp" \
        --field want_files="$_want_files" --field got_files="${_got_files:-0}"
      mark_terminal "$jobid" failed
      return 1
    fi
    log "job $jobid: asset '$name' receipt corroborated (${_got_files} >= ${_want_files} file(s))"
  fi

  local rerr
  if ! rerr="$(asset_check_require "$cache" "$wdir" "$name")"; then
    if [ "$opt" = "1" ]; then
      log "job $jobid: OPTIONAL asset '$name' postcondition unmet ($rerr) — continuing"
      return 0
    fi
    log "job $jobid: asset '$name' postcondition FAILED ($rerr) — job cannot start"
    emit "$jobid" failed --field reason="asset_stage_failed:$name" \
      --field asset="$name" --field detail="$rerr"
    mark_terminal "$jobid" failed
    return 1
  fi

  # record the byte-total marker (authoritative post-pull), then wire up dest.
  local _bytes; _bytes="$(_dir_bytes "$cache")"
  echo "$_bytes" > "$marker" 2>/dev/null || true
  _link_asset_dest "$run" "$cache" "$dest"
  # MEASURED asset-pull throughput, mirroring train.sh's pull_throughput fields
  # (bytes/secs/mbps) so hosts.py can host-score jobs/eval boxes the same way it
  # scores train boxes. Emitted on the PER-BOX stream (jobs/nodes/<IID>/events/)
  # keyed by IID — the join key hosts.py bridges from the launched event's
  # instance_id — because throughput is a host property, not a job property.
  # Integer MB/s (decimal, /1048576) matches the train lane's coarse record.
  local _secs=$(( $(date +%s) - _pull_t0 )); [ "$_secs" -lt 1 ] && _secs=1
  local _mbps=$(( _bytes / 1048576 / _secs ))
  emit_box asset_throughput "asset=$name" "bytes=$_bytes" "secs=$_secs" "mbps=$_mbps"
  log "job $jobid: asset '$name' staged -> $cache (${_bytes}B in ${_secs}s, ~${_mbps} MB/s)"
  return 0
}

# stage_assets <jobid> <wdir> <run> — stage every ticket asset BEFORE the
# entrypoint. rc 1 (a non-optional asset failed; terminal already emitted) tells
# the runner to abort before `started`.
stage_assets() {
  local jobid="$1" wdir="$2" run="$3" specf="$wdir/.assets.tsv"
  [ -s "$specf" ] || return 0
  mkdir -p "$ASSETS_DIR"
  # free-space precheck BEFORE anything is pulled: a box too small must fail with
  # `insufficient_disk` and the numbers, not with a transport-shaped
  # `asset_stage_failed` halfway through a multi-GB pull (P4e; see above).
  assets_disk_precheck "$jobid" "$specf" || return 1
  local name b2p mode opt dest recp
  while IFS=$'\t' read -r name b2p mode opt dest recp; do
    [ -n "$name" ] || continue
    stage_one_asset "$jobid" "$wdir" "$run" "$name" "$b2p" "${mode:-copy}" \
      "${opt:-0}" "${dest:-}" "${recp:-}" || return 1
  done < "$specf"
  return 0
}

# _prior_ckpt_event <jobid> — best-effort: does the job's B2 event history already
# hold a `checkpoint` or `resumed` event (i.e. SOME prior box synced state for this
# job)? This is the second retarget-continuation signal (HANDOFF_DESIGN §4): a job
# moved by `job retarget` onto a FRESH box has NO box-local .attempts/.preempts
# breadcrumb, so restart_count=0 and the pull-back would be skipped even though
# checkpoints exist on B2. The event TYPE lives INSIDE the JSON body (the object
# key is only ts-actor-nonce), so we must read bodies. Fully best-effort: any read
# failure returns rc 1 (no pull-back — the current, safe behavior); never crashes
# the runner. Returns on the first match; only ever called under JOB_CHECKPOINT_S>0.
_prior_ckpt_event() {
  local jobid="$1" keys k
  keys="$(rclone lsf "$B2/jobs/$jobid/events/" 2>/dev/null | grep '\.json$' || true)"
  [ -n "$keys" ] || return 1
  while IFS= read -r k; do
    [ -n "$k" ] || continue
    case "$(rclone cat "$B2/jobs/$jobid/events/$k" 2>/dev/null)" in
      *'"event":"checkpoint"'*|*'"event": "checkpoint"'*|\
      *'"event":"resumed"'*|*'"event": "resumed"'*) return 0 ;;
    esac
  done <<< "$keys"
  return 1
}

# ticket_requeue_ts <jobid> — echo the queue ticket's `requeued_ts`
# (jobmeta.REQUEUE_TICKET_MARK), or nothing. Stamped ONLY by `herdd job requeue`,
# which re-opens a TERMINAL-FAILED job under the same JOB_ID; poll_once uses it to
# override the results.DONE.json skip (see there for why that skip would otherwise
# swallow every requeue, and why this cannot loop).
#
# THREE answers, not two: a `?` means the ticket could not be READ or PARSED at
# all. The consumer latches `.terminal` on the empty answer and a latch is
# permanent, so folding "unknown" into "no requeue" let one transient B2 blip
# swallow an operator's `job requeue` on this box forever. Unknown must stay
# unknown; the caller then costs one extra read next poll instead.
ticket_requeue_ts() {
  local jobid="$1" _body _rc=0
  _body="$(rclone cat "$B2/jobs/queue/$IID/$jobid.json" 2>/dev/null)" || _rc=$?
  if [ "$_rc" != "0" ] || [ -z "$_body" ]; then echo "?"; return 0; fi
  printf '%s' "$_body" | "$PY" -c '
import json, sys
try:
    print(json.load(sys.stdin).get("requeued_ts") or "")
except Exception:
    print("?")
' 2>/dev/null || echo "?"
}

# --- runner body: everything after the scheduler said GO -----------------------
# Runs in a BACKGROUND subshell. Inherits the prepared JOB_* vars + wdir + gpu
# assignment from the spawning scope. Owns: bundle download/extract, venv check,
# checkpoint pull-back (resume), entrypoint + heartbeat + checkpoint loops,
# publish, terminal emit + local terminal cache, .running cleanup.
run_job_body() {
  local jobid="$1" wdir="$2" gpus="$3" restart_count="$4"

  # bundle download (content-addressed; local dedupe across repeated submits)
  #
  # This was the LEAST guarded B2 read on the box, despite being the one every
  # job passes through: a bare `copyto` with stderr sent to /dev/null, no
  # timeout, no retry, and a terminal reason ("bundle download failed") that
  # named no cause. That is precisely the shape of the incident this whole class
  # of work exists to prevent — a revoked key's InvalidAccessKeyId discarded with
  # 2>/dev/null, leaving an operator to guess. Bundles are small (MB-scale), so
  # a flat ceiling is defensible here where it would not be for an asset.
  local bfile="$CACHE_DIR/$JOB_BUNDLE_SHA.tar.zst"
  if [ ! -f "$bfile" ]; then
    local berr="$CACHE_DIR/.$JOB_BUNDLE_SHA.pull.err"
    local brounds="${JOBD_BUNDLE_RETRIES:-3}" bto="${JOBD_BUNDLE_TIMEOUT_S:-600}"
    local bi brc bok=0 breason="" btail=""
    for bi in $(seq 1 "$brounds"); do
      timeout "$bto" rclone copyto "$B2/jobs/bundles/$JOB_BUNDLE_SHA.tar.zst" \
        "$bfile" 2>"$berr"; brc=$?
      [ "$brc" -eq 0 ] && { bok=1; break; }
      btail="$(tr '\n' ' ' < "$berr" 2>/dev/null | tail -c 200)"
      if [ "$brc" -eq 124 ]; then
        # DISTINGUISHABLE from a failure: `timeout` reserves 124, and the reason
        # string says timeout so the event is legible without the log.
        breason="bundle download TIMEOUT after ${bto}s (JOBD_BUNDLE_TIMEOUT_S)"
        log "job $jobid: $breason (round $bi/$brounds, $JOB_BUNDLE_SHA)"
        break     # a blown ceiling will not improve on the next round
      fi
      if grep -qiE 'InvalidAccessKeyId|SignatureDoesNotMatch|AccessDenied|Unauthorized|not valid| 403 ' "$berr" 2>/dev/null; then
        breason="bundle download AUTH FAILURE (dead/rotated key): $btail"
        log "job $jobid: $breason"
        : > "$CRED_DIR/.cred_refresh_now" 2>/dev/null || true
        break     # non-transient — same rule as asset_pull
      fi
      breason="bundle download failed (rc=$brc): $btail"
      log "job $jobid: $breason (round $bi/$brounds)"
      [ "$bi" -lt "$brounds" ] && sleep $(( bi * ${JOBD_BUNDLE_BACKOFF:-5} ))
    done
    if [ "$bok" != 1 ]; then
      rm -f "$bfile" 2>/dev/null || true    # never leave a truncated cache entry
      emit "$jobid" failed --field reason="${breason:-bundle download failed}" \
        --field bundle_sha="$JOB_BUNDLE_SHA"
      mark_terminal "$jobid" failed; return
    fi
    rm -f "$berr" 2>/dev/null || true
  fi

  # fresh extract into the workdir (NEVER execute in the cache)
  local run="$wdir/work"; rm -rf "$run"; mkdir -p "$run"
  if ! "$PY" "$JH" extract "$bfile" "$run" --sha "$JOB_BUNDLE_SHA" >/dev/null 2>&1; then
    log "job $jobid: extract/sha-verify failed"
    rm -f "$bfile"     # drop a corrupt cache entry so a retry re-downloads
    emit "$jobid" failed --field reason="bundle extract/verify failed"
    mark_terminal "$jobid" failed; return
  fi

  # venv need (sources into THIS shell so the entrypoint inherits it)
  local reason; reason="$(check_venv "$JOB_NEEDS_VENV" "$wdir/.job.env")"
  if [ -n "$reason" ]; then
    log "job $jobid: unmet needs — $reason"
    emit "$jobid" failed --field reason="$reason"
    mark_terminal "$jobid" failed; return
  fi

  # resume: pull previously-synced state back so the entrypoint finds its
  # checkpoints at the same relative paths (HF --resume auto pattern). Only
  # jobs that opted into checkpointing have anything to pull; the DONE marker
  # can't exist here (terminal jobs never reach the runner).
  #
  # SOURCE = jobs/<id>/checkpoints/ (NOT results/). Mid-run checkpoints live under
  # their own prefix so the canonical jobs/<id>/results/ is written EXACTLY ONCE
  # (at finalize, as a NEW object → strong read-after-write; no overwrite window
  # for a downstream reader). The on-disk relative paths (e.g. results/gens_*.jsonl,
  # out/ckpt-*/) are preserved UNDER the prefix, so a pull-back into "$run/" lands
  # each file back at its original relative path.
  #
  # PULL-BACK PREDICATE (HANDOFF_DESIGN §4). Fire when JOB_CHECKPOINT_S>0 AND either:
  #   (a) restart_count>0 — a resume ON THIS BOX (crash/preempt). restart_count is
  #       attempts+preempts, read from BOX-LOCAL $STATE_DIR breadcrumbs, so it is 0
  #       on a FRESH box even when prior checkpoints exist on B2. This was the ONLY
  #       gate historically, which is why...
  #   (b) ...a `job retarget` CONTINUATION got silently restarted from scratch: the
  #       ticket moved here from another box, the local breadcrumbs never existed
  #       (restart_count=0), so (a) alone skipped the pull-back. Detect it from the
  #       ticket's `retargeted_from` marker OR a prior `checkpoint`/`resumed` event
  #       in the job's B2 history (best-effort probe; a failed read just falls back
  #       to (a)'s behavior). JOB_RETARGETED_FROM is prepared in poll_once.
  local do_pullback=0 pb_why=""
  if [ "${JOB_CHECKPOINT_S:-0}" -gt 0 ] 2>/dev/null; then
    if [ "$restart_count" -gt 0 ] 2>/dev/null; then
      do_pullback=1; pb_why="resume #$restart_count"
    elif [ -n "${JOB_RETARGETED_FROM:-}" ] && [ "${JOB_RETARGETED_FROM}" != "-" ]; then
      do_pullback=1; pb_why="retargeted from $JOB_RETARGETED_FROM (fresh box, restart_count=0)"
    elif _prior_ckpt_event "$jobid"; then
      do_pullback=1; pb_why="prior checkpoint/resumed event on B2 (fresh box, restart_count=0)"
    fi
  fi
  if [ "$do_pullback" = 1 ]; then
    log "job $jobid: $pb_why — pulling prior checkpoints back"
    # HIGHEST-IMPACT PULL ON THE BOX: this fires on EVERY spot preemption
    # resume, crash restart, and job retarget, and it was on STOCK rclone (4
    # flows). b2x also makes it idempotent — a resume on the SAME box already
    # holds most of these bytes, and re-fetching them was pure billed waste.
    # BOUNDED: fetch only the newest checkpoint(s), never the whole ladder. See
    # _ckpt_latest_remote for the 41-GB-of-a-50-GB-disk incident that motivated
    # this. Everything that is NOT a checkpoint dir (logs, corpus-identity.json,
    # markers) is still pulled in full — it is kilobytes and some of it is load
    # bearing for the resume.
    #
    # FAIL-OPEN: if the remote listing is unreadable, or names no checkpoint dir,
    # fall through to the historical whole-prefix pull. A resume that fetches too
    # much still resumes; a resume that fetches too little silently restarts from
    # scratch, which is the far more expensive failure and the one this must
    # never introduce.
    #
    # STILL FAIL-OPEN, now VERIFIED AND LOUD (v13-chain-v12, 2026-08-17). "Fetches
    # too little" turned out to have a third outcome, worse than either: a
    # checkpoint that arrives missing optimizer.pt/scheduler.pt does not restart
    # from scratch and does not fail — transformers skips BOTH loads silently and
    # trains on for 70 steps at 4.15x the intended integrated LR with zeroed Adam
    # moments (see _ckpt_restore_complete for the mechanism and the measurement).
    # So the pull is now checked against the remote listing and retried ONCE, and
    # a still-incomplete restore is logged loudly — but the DECISION stays
    # fail-open, unchanged: jobd runs on every rented box, a jobd bug is the
    # expensive kind, and the gate that must actually stop a bad resume is the
    # trainer's own pre-train check that optimizer.pt and scheduler.pt exist in
    # the checkpoint it is about to resume from. This is observation, not
    # enforcement.
    _keep="$(_ckpt_latest_remote "$B2/jobs/$jobid/checkpoints/out/" || true)"
    if [ -n "${_keep:-}" ]; then
      log "job $jobid: pull-back bounded to newest checkpoint(s): $(echo "$_keep" | tr '\n' ' ')"
      b2x_pull "$B2/jobs/$jobid/checkpoints/" "$run/" --exclude "out/checkpoint-*/**" \
        || rclone copy --fast-list "$B2/jobs/$jobid/checkpoints/" "$run/" \
             --exclude "out/checkpoint-*/**" 2>/dev/null \
        || log "job $jobid: non-checkpoint pull-back failed (continuing)"
      for _ck in $_keep; do
        _pb_ok=0
        # TWO attempts, hard-bounded (never a `while`): one retry is worth
        # exactly one round because both transports are idempotent RESUMES, not
        # restarts — b2x skips complete objects and refetches only the parts its
        # .b2x/state.json says are missing, and `rclone copy` refetches only what
        # is absent or short. A transient stall / deadline / throughput-floor
        # abort therefore does strictly better on the second pass, for the price
        # of one LIST. A file that is simply NOT ON B2 (a torn push from the
        # dying box) cannot be fixed here by any number of rounds; that case
        # leaves the loop loud instead of looping.
        for _pb_try in 1 2; do
          b2x_pull "$B2/jobs/$jobid/checkpoints/out/$_ck/" "$run/out/$_ck/" \
            || rclone copy --fast-list "$B2/jobs/$jobid/checkpoints/out/$_ck/" \
                 "$run/out/$_ck/" 2>/dev/null \
            || log "job $jobid: pull-back of $_ck FAILED on attempt $_pb_try (entrypoint may start fresh)"
          if _ckpt_restore_complete "$run/out/$_ck" "$B2/jobs/$jobid/checkpoints/out/$_ck"; then
            _pb_ok=1; break
          fi
          if [ "$_pb_try" = 1 ]; then
            log "job $jobid: restored $_ck is INCOMPLETE — retrying the pull once (idempotent resume, refetches only what is missing)"
          fi
        done
        if [ "$_pb_ok" != 1 ]; then
          log "job $jobid: !! RESUME STATE INCOMPLETE for $_ck after 2 pulls — PROCEEDING ANYWAY (fail-open, by design). If optimizer.pt or scheduler.pt is among the files named above, transformers will skip BOTH loads SILENTLY and train on a different recipe (v13-chain-v12, 2026-08-17); the trainer's own pre-train resume check is the gate meant to stop that."
        fi
      done
      unset _ck _pb_ok _pb_try
    else
      log "job $jobid: could not enumerate remote checkpoints — pulling whole prefix"
      b2x_pull "$B2/jobs/$jobid/checkpoints/" "$run/" \
        || rclone copy --fast-list "$B2/jobs/$jobid/checkpoints/" "$run/" 2>/dev/null \
        || log "job $jobid: checkpoint pull-back failed (entrypoint starts fresh)"
    fi
    unset _keep
    # CONTINUITY EVENT (Issue C): a box-LOCAL resume (restart_count>0) already
    # emitted `resumed` in poll_once, but a `job retarget` CONTINUATION lands on a
    # FRESH box (restart_count=0, cases (b)/(c) above) and emitted only
    # claimed+started — so a resume off ANOTHER box was invisible in the event log
    # (canary-job: understudy showed claimed+started, no `resumed`; continuity was
    # only readable from heartbeat tails). Emit `resumed` here for the fresh-box
    # continuation so fold_events counts it (last_resumed_ts) and `job status`
    # shows the pull-back. Reuse `resumed` (frozen EVENTS + already consumed) NOT a
    # new name (which the fold would orphan); kind=retarget + from_box carry the
    # provenance (from_box="-" on case (c), where the source box is unknown).
    if [ "$restart_count" -eq 0 ] 2>/dev/null; then
      emit "$jobid" resumed --field attempt=1 --field kind=retarget \
        --field from_box="${JOB_RETARGETED_FROM:--}"
    fi
  fi

  # declarative asset staging (N4): pull every ticket asset onto the box BEFORE
  # the entrypoint. A non-optional failure takes the job terminal `failed` (event
  # already emitted inside) — return here so `started` never fires.
  if ! stage_assets "$jobid" "$wdir" "$run"; then
    return
  fi

  # dispatch by extension: .py -> python, else bash
  local -a runcmd
  case "$JOB_ENTRYPOINT" in
    *.py) runcmd=("$PY" "$JOB_ENTRYPOINT") ;;
    *)    runcmd=(bash "$JOB_ENTRYPOINT") ;;
  esac

  local ncards=0
  [ "$gpus" != "-" ] && { local _g; IFS=',' read -ra _g <<< "$gpus"; ncards=${#_g[@]}; }
  # per-card VRAM (min GB across the assigned cards) + box CPU cores, for the
  # launch-shape planner (launch_plan.sh / autotune.py). JOB_GPU_COUNT alone is
  # not enough: the VRAM-safety grad-ckpt floor AND the quant-by-VRAM suggestion
  # both read JOB_GPU_RAM_GB, and autotune's dataloader-worker sizing reads
  # CPU_CORES — absent them the floor silently never fires on a real box. Empty
  # VRAM for a CPU job (gpus="-"); MIN (not max) is the OOM-safe choice on a
  # heterogeneous box. Reads the boot GPU inventory (GPU_IDS/GPU_MEM, whole GB).
  local job_gpu_ram_gb=""
  if [ "$gpus" != "-" ] && [ "$ncards" -gt 0 ]; then
    local _d _i _mem
    for _d in "${_g[@]}"; do
      for _i in "${!GPU_IDS[@]}"; do
        [ "${GPU_IDS[$_i]}" = "$_d" ] || continue
        _mem="${GPU_MEM[$_i]}"
        if [ -z "$job_gpu_ram_gb" ] || [ "$_mem" -lt "$job_gpu_ram_gb" ] 2>/dev/null; then
          job_gpu_ram_gb="$_mem"
        fi
        break
      done
    done
  fi
  # CPU_CORES = the cgroup ALLOWANCE, not the cpuset width (see
  # effective_cpu_cores: measured 2.08x/2.6x overstatements from nproc).
  local job_cpu_cores; job_cpu_cores="$(effective_cpu_cores)"
  log "job $jobid: start entrypoint='$JOB_ENTRYPOINT' timeout=${JOB_TIMEOUT_S}s gpus=$gpus attempt=$((restart_count+1))"
  emit "$jobid" started --field attempt="$((restart_count+1))"

  # Fire-on-arrival seed, taken HERE — BEFORE the entrypoint exists, never inside
  # the sync subshell. Anything already in the run dir at this instant came from
  # the resume pull-back (it is on B2 already), so firing on it would re-push GB
  # for nothing; anything that appears AFTER is the trainer's and must fire. Doing
  # this in the sync subshell raced the entrypoint and lost: the trainer's first
  # `mkdir out/checkpoint-N` beat the seed roughly half the time, seeded that dir
  # as already-seen, and the fast path then never fired for that checkpoint
  # (caught by test_jobd_checkpoint_sync_fires_on_arrival_not_on_the_timer, which
  # passed alone and failed inside the full suite — a sampled race, not a flake).
  _ckpt_all_dirs "$run" > "$wdir/.checkpoint.seen" 2>/dev/null || : > "$wdir/.checkpoint.seen"

  local logf="$wdir/log.txt"
  local start_s; start_s="$(date +%s)"
  # Declared HERE, not inside the publish-verify block, because the end-of-run
  # scrub gate reads it. bash `local` is function-scoped so the old placement
  # worked, but it made the gate silently depend on a declaration two hundred
  # lines away inside an `if` that may never run.
  local vfail=""
  (
    cd "$run" || exit 127
    set -a
    # shellcheck source=/dev/null
    . "$wdir/.job.env"
    export JOB_ID="$jobid" B2_BUCKET IID
    # GPU assignment: the scheduler's contract with the entrypoint. CPU jobs get
    # an EMPTY CUDA_VISIBLE_DEVICES (they must not grab cards owned by peers).
    # JOB_GPU_COUNT is what serve-style entrypoints feed to --dp/--tp math.
    if [ "$gpus" = "-" ]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES="$gpus"; fi
    export JOB_GPUS="$gpus" JOB_GPU_COUNT="$ncards" JOB_RESTART_COUNT="$restart_count"
    # per-card VRAM + box cores for the launch planner (see computation above).
    # Exported AFTER .job.env so these box FACTS win over any stale job-env value.
    # JOBD_STATE_DIR lets an entrypoint take a LIVE census of its siblings (one
    # `<jid>.running` per job jobd is currently running on this box) instead of
    # guessing how many peers it shares the CPU quota with — that census is the
    # divisor tools/witness/cpu_budget.py sizes its compile pool from.
    # JOBD_ROOT: make the box root an EXPLICIT part of the entrypoint contract.
    # A dozen bundles already spell `${JOBD_ROOT:-/workspace}` to find the asset
    # cache / the baked train env, and until now that only worked by ACCIDENT —
    # rehearse.sh and joblocal.py happen to pass JOBD_ROOT through jobd's own
    # environment, so a child inherited it. Nothing guaranteed it, and a caller
    # that set the root any other way would have silently sent every such bundle
    # back to a literal /workspace it cannot write. Exported here, with the other
    # box FACTS and AFTER .job.env for the same reason they are.
    # NO-OP ON A RENTED BOX: $ROOT is `${JOBD_ROOT:-/workspace}`, so the value
    # exported there is exactly /workspace and `${JOBD_ROOT:-/workspace}` in an
    # entrypoint expands to what it always did.
    export JOB_GPU_RAM_GB="$job_gpu_ram_gb" CPU_CORES="$job_cpu_cores" \
           JOBD_STATE_DIR="$STATE_DIR" JOBD_ROOT="$ROOT" \
           JOBD_HOSTFACTS_DROP="$HOSTFACTS_DROP"
    # Told, not guessed: a job that harvests its own work-rate must not have to
    # re-derive the drop path from JOBD_ROOT and agree with hostfacts.drop_dir()
    # by coincidence.
    # Shared Triton JIT cache dir (pre-populated from the remote at boot; see
    # triton_cache_boot_pull). ${VAR:-} keeps a job's OWN TRITON_CACHE_DIR from
    # .job.env — a bundle that wants isolation still gets it. CPU jobs get
    # nothing (no Triton, and the empty-string export would break Triton's
    # default resolution if a later job inherited it).
    if [ "$gpus" != "-" ] && [ "${JOBD_TRITON_CACHE:-1}" = "1" ]; then
      export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TC_DIR}"
    fi
    set +a
    timeout "$JOB_TIMEOUT_S" "${runcmd[@]}"
  ) >"$logf" 2>&1 &
  local epid=$!

  # heartbeat loop: emit a log tail (+ host_metrics, if the probe is present)
  # every HB seconds while the entrypoint runs. host_metrics is a compact
  # k:v line (GPU util/mem/pwr/temp/throttle + cpu/net/disk) — one --field, no
  # '=' in the value, so the K=V field parser splits it cleanly; folded into the
  # job view as last_metrics. Best-effort: a missing probe / no-GPU box just
  # omits it and the heartbeat still lands.
  # STALL DETECTION (JOBD_STALL_S, 0 disables). A wedged entrypoint writes
  # NOTHING and burns the full rental until some other timer notices. Measured
  # 2026-08-06 on job 20260806T082213-v11-...: a NCCL collective deadlocked on
  # its FIRST all-reduce and sat silent until torch's own watchdog fired at
  # 1800s — 30 minutes of paid silence, twice, on two different boxes. Nothing
  # in jobd looked at the fact that the log had stopped growing.
  #
  # So: if $logf has not grown for STALL_S while the entrypoint is still alive,
  # emit ONE `stall_suspected` event. It is an ALARM, never a kill — a long
  # legitimate quiet phase (model load, dataset tokenization) must not lose a
  # job, and only the operator/fleetd decides to act.
  #
  # $logf is jobd's OWN capture, created empty at entrypoint start, so it is
  # immune to the restored-log trap: a resumed box pulls the PREVIOUS attempt's
  # out/train.log back from B2 at its old size, and a detector watching the
  # bundle's log would read that frozen size as an instant stall.
  #
  # Default 600s: the longest legitimate silence measured on this fleet is the
  # ~5-7 min base-model load + tokenize phase, and the tightest real signal is
  # v11's 26.71 s/it (SAVE_STEPS 10 => a ~4.5 min checkpoint interval). 600s
  # clears both with headroom while still firing 3x sooner than torch's 1800s.
  local STALL_S="${JOBD_STALL_S:-600}"
  ( _sz_prev=""; _quiet=0; _fired=0
    while kill -0 "$epid" 2>/dev/null; do
      sleep "$HB"
      kill -0 "$epid" 2>/dev/null || break
      hm=""; [ -f "$MP" ] && hm="$("$PY" "$MP" fields 2>/dev/null || true)"
      # `beacon`: the per-job heartbeat is the strongest liveness signal any
      # outside observer has (herdd._job_liveness_epoch prefers it over the
      # JOBD_STATUS marker), so its silence IS the observability loss. Runs in
      # this backgrounded subshell — the verdict reaches the daemon through
      # _py_broken's breadcrumb, not through these variables.
      _py_call beacon emit "$jobid" heartbeat --instance-id "$IID" \
        ${JOB_EXP_ID:+--field "exp_id=$JOB_EXP_ID"} ${JOB_ARM:+--field "arm=$JOB_ARM"} \
        ${hm:+--field "host_metrics=$hm"} \
        --tail-file "$logf"
      # --- stall check (fully guarded; never affects the job) ---------------
      if [ "$STALL_S" -gt 0 ] 2>/dev/null; then
        _sz="$(stat -c %s "$logf" 2>/dev/null || echo 0)"
        if [ -n "$_sz_prev" ] && [ "$_sz" = "$_sz_prev" ]; then
          _quiet=$(( _quiet + HB ))
        else
          _quiet=0; _fired=0
        fi
        _sz_prev="$_sz"
        if [ "$_quiet" -ge "$STALL_S" ] && [ "$_fired" = 0 ]; then
          log "job $jobid: STALL SUSPECTED — $logf unchanged for ${_quiet}s (${_sz} B); entrypoint pid=$epid still alive"
          # `report`, not `beacon`: this event IS a failure report. A failure to
          # report a failure must not compound into a second incident, so it can
          # extend a streak but never arm the latch by itself.
          _py_call report emit "$jobid" stall_suspected --instance-id "$IID" \
            --field "quiet_s=$_quiet" --field "log_bytes=$_sz" \
            --field "threshold_s=$STALL_S" \
            ${hm:+--field "host_metrics=$hm"} \
            --tail-file "$logf"
          _fired=1
        fi
      fi
    done ) &
  local hbpid=$!

  # cancel-watch: `herdd job cancel` writes jobs/<id>/CANCEL and deletes the
  # queue ticket. The ticket-delete stops jobd RECLAIMING/resuming the job, but a
  # job already running here keeps going until this loop sees the marker and kills
  # the entrypoint's process tree. A `.cancelled` breadcrumb tells the post-wait
  # code to record a terminal `cancelled` instead of `failed`. Polls at
  # CANCEL_POLL (default 15s); one cheap `lsf` per interval, only while running.
  local cwpid=""
  if [ "${JOBD_NO_CANCEL:-0}" != "1" ]; then
    ( while kill -0 "$epid" 2>/dev/null; do
        sleep "$CANCEL_POLL"
        kill -0 "$epid" 2>/dev/null || break
        if rclone lsf "$B2/jobs/$jobid/CANCEL" 2>/dev/null | grep -q .; then
          : > "$wdir/.cancelled"
          log "job $jobid: CANCEL marker seen — killing entrypoint tree (pid=$epid)"
          kill_tree "$epid" TERM; sleep 3; kill_tree "$epid" KILL
          break
        fi
        # CHECKPOINT_NOW rides this poll rather than adding a second one: same
        # cadence, one more cheap lsf. It must NOT break the loop and must not
        # signal the entrypoint — a flush is not a stop. See _flush_marker_consume.
        _flush_marker_consume "$jobid" "$wdir/.checkpoint_now" || true
      done ) &
    cwpid=$!
  fi

  # mid-run checkpoint sync: every JOB_CHECKPOINT_S, ship the checkpoint globs
  # into the job's checkpoints/ prefix so a timeout/OOM/box-death loses at most
  # one interval (the 2026-07-10 LoRA job lost 3h of training to its own timeout
  # without this). --min-age skips files the entrypoint is mid-write on; the
  # trailing-slash dest skips rclone's flaky B2 HEAD dest-check (the S3 quirk
  # that ate base-reader-nanbeige-01's artifacts push).
  #
  # DEST = jobs/<id>/checkpoints/ (NOT results/). This is the write-side fix for the
  # four B2-eventual-consistency false-fails (2026-07-15/16/19/20): the checkpoint
  # loop used to overwrite jobs/<id>/results/<file> mid-run (for e2 paired-gen the
  # gens_*.jsonl are in BOTH the checkpoint and results globs), then finalize
  # overwrote the SAME key — and B2 OBJECT OVERWRITES are eventually consistent, so a
  # different client (the workflow controller's validate_generation_artifact) kept
  # reading the stale empty/partial version for ~90s+ after finalize. Routing the
  # recurring mid-run write to its own prefix makes finalize's write to results/ a
  # NEW object (strong read-after-write) — no client can ever read a stale results/.
  # The resume pull-back (above) reads this same checkpoints/ prefix back.
  #
  # ACCEPTED RISK (documented, not fixed): a HARD SIGKILL of the box (no docker-stop
  # SIGTERM, so neither this loop nor the preempt trap runs) loses at most one
  # JOB_CHECKPOINT_S interval of local progress. The --min-age window is a second,
  # narrower blind spot — files younger than JOBD_CKPT_MIN_AGE are skipped HERE; the
  # preempt trap's final flush uses NO --min-age precisely to close it on an orderly
  # stop.
  #
  # LOUD on failure (box 44566398): rclone's stderr is CAPTURED, not swallowed. A
  # dead/rotated B2 key silently FROZE this sync while compute ran on and stranded a
  # finished adapter. We now (a) NEVER kill the job for a sync failure — compute keeps
  # making local progress and every later tick retries; (b) drop a persistent
  # .checkpoint_sync_failed breadcrumb (survives even a fully-dead B2); (c) emit a
  # distinct, RATE-LIMITED `checkpoint_sync_failed` event (auth-class cause named)
  # so the gap is never silent for a returning operator or `job supervise`.
  #
  # FIRE-ON-ARRIVAL (owner 2026-08-05: "we should begin sync as soon as the
  # checkpoint hits the disk"). The 180 s timer was the dominant term in the
  # exposure window — box 46859541 lost 51 steps on 2026-08-05 because
  # checkpoint-50 was written 7 s AFTER a sync pass and the box died ~3 min later
  # with no SIGTERM, so the preempt trap's final flush never ran. The wait below
  # is therefore INTERRUPTIBLE: a JOBD_CKPT_WATCH_S (5 s) readdir looks for a
  # checkpoint-<N>/ dir that is COMPLETE on disk, and a hit fires the pass at once.
  # The periodic pass is RETAINED unchanged as the backstop — it still covers
  # non-checkpoint files, anything the watcher misses, and a retry after a failure.
  #
  # An early pass is scoped to exactly the newly-complete dirs and carries NO
  # --min-age: the age window is a heuristic for "not mid-write", and on this path
  # we have the real thing (see _ckpt_write_complete — trainer_state.json written
  # last by rank 0, plus a quiescence window for the unordered per-rank
  # rng_state_<i>.pth). Everything else keeps the age window on the periodic pass.
  local ckpid=""
  if [ "${JOB_CHECKPOINT_S:-0}" -gt 0 ] 2>/dev/null && [ -s "$wdir/.checkpoint.globs" ]; then
    log "job $jobid: checkpoint sync every ${JOB_CHECKPOINT_S}s (fire-on-arrival watch every ${JOBD_CKPT_WATCH_S:-5}s)"
    ( inc=()
      while IFS= read -r pat; do
        [ -n "$pat" ] && inc+=(--include "$pat")
      done < "$wdir/.checkpoint.globs"
      n=0; sfails=0; cerr="$wdir/.checkpoint_sync.err"
      cmatch="$wdir/.checkpoint.matched"      # this pass's glob expansion (prune input)
      # live-append tail snapshot (task #110): staging tree, cross-pass append
      # state, and its own stderr. All three live under $wdir, NEVER under $run —
      # a staging copy inside the run dir would be re-matched by the very globs
      # that produced it and published into results/.
      tstage="$wdir/.checkpoint.tail"
      tstate="$wdir/.checkpoint.tailstate"
      terr="$wdir/.checkpoint_tail.err"
      # dirs the fast path has already fired on. SEEDED BY THE CALLER, before the
      # entrypoint was launched (see the _ckpt_all_dirs call above the entrypoint
      # subshell) — seeding it here would race the trainer's first mkdir.
      seenf="$wdir/.checkpoint.seen"
      [ -f "$seenf" ] || : > "$seenf"
      # dirs whose completion marker is already on B2. Purely a cost cache — the
      # marker is idempotent, this just keeps the verify read-back to once per dir.
      markedf="$wdir/.checkpoint.marked"
      [ -f "$markedf" ] || : > "$markedf"
      # min-age window in seconds (the config form is always "<int>s"): a non-int
      # form disables the age filter (report the raw match count, the old behavior)
      # rather than crash the sync path.
      _minsec="${JOBD_CKPT_MIN_AGE:-45s}"; _minsec="${_minsec%s}"
      case "$_minsec" in ''|*[!0-9]*) _minsec="" ;; esac
      _tick="${JOBD_CKPT_WATCH_S:-5}"
      case "$_tick" in ''|*[!0-9]*) _tick=5 ;; esac
      while kill -0 "$epid" 2>/dev/null; do
        # --- interruptible wait: JOB_CHECKPOINT_S, cut short by a new checkpoint --
        # fast_all = every newly-complete dir the watcher saw (what gets marked
        # seen, so a dir this job does not ship cannot re-arm the watcher every
        # tick). fast = the subset the bundle's OWN checkpoint globs cover, which
        # is what the early pass is allowed to push.
        fast=""; fast_all=""
        if [ "$_tick" -le 0 ] || [ "$_tick" -ge "$JOB_CHECKPOINT_S" ]; then
          sleep "$JOB_CHECKPOINT_S"
        else
          _waited=0
          while [ "$_waited" -lt "$JOB_CHECKPOINT_S" ]; do
            sleep "$_tick"; _waited=$(( _waited + _tick ))
            kill -0 "$epid" 2>/dev/null || break
            # An operator flush cuts the wait short on the same tick the
            # fire-on-arrival watcher uses; the cancel-watch dropped the crumb.
            [ -f "$wdir/.checkpoint_now" ] && break
            fast_all="$(_ckpt_new_ready "$run" "$seenf")"
            [ -n "$fast_all" ] && break
          done
        fi
        kill -0 "$epid" 2>/dev/null || break
        # Consume the crumb HERE, above the empty-glob `continue`: a job whose
        # globs match nothing would otherwise re-arm the watch every single tick.
        flushnow=0
        if [ -f "$wdir/.checkpoint_now" ]; then flushnow=1; rm -f "$wdir/.checkpoint_now"; fi
        # HONEST FILE COUNT (Issue B; v1 canary silent-data-loss): rclone --min-age
        # skips files YOUNGER than the window, so `files=` must count only what
        # actually SHIPS — matches OLD ENOUGH to clear the window — not the raw glob
        # match. A job checkpointing FASTER than the window ships ZERO bytes every
        # pass, yet the pre-filter count reported files=<glob> anyway (17
        # "checkpoints", 0 bytes on B2, a later retarget "resumed" from nothing).
        # Count matches (`nmatch`, kept as `matched=` context) AND age-eligible
        # matches (`nship`, the honest `files=`); skip the pass entirely only when
        # NOTHING matches (rclone exits 0 on an empty match). The rclone copy below
        # still runs on nship==0 (a harmless 0-byte pass) — only the report changes.
        _now=$(date +%s 2>/dev/null || echo 0)
        nmatch=0; nship=0
        : > "$cmatch"
        while IFS= read -r _f; do
          [ -n "$_f" ] || continue
          nmatch=$((nmatch+1))
          printf '%s\n' "$_f" >> "$cmatch"
          if [ -z "$_minsec" ] || [ "$_minsec" = 0 ]; then nship=$((nship+1)); continue; fi
          # GNU stat first, BSD/busybox `stat -f %m` fallback (2026-07-18 review
          # P3: a non-GNU container made every file fall back to _now => age 0 =>
          # files=0 under-report while rclone's own --min-age still shipped bytes)
          _mt=$(stat -c %Y "$run/$_f" 2>/dev/null \
                || stat -f %m "$run/$_f" 2>/dev/null || echo "$_now")
          [ $(( _now - _mt )) -ge "$_minsec" ] 2>/dev/null && nship=$((nship+1)) || :
        done < <(shopt -s globstar nullglob dotglob; cd "$run" 2>/dev/null || exit 0
                 while IFS= read -r pat; do
                   [ -n "$pat" ] || continue
                   for f in $pat; do [ -f "$f" ] && printf '%s\n' "$f"; done
                 done < "$wdir/.checkpoint.globs")
        [ "${nmatch:-0}" -gt 0 ] || continue
        # --- what THIS pass ships -------------------------------------------------
        # periodic: the whole checkpoint glob, age-filtered (unchanged behavior).
        # fire-on-arrival: ONLY the newly-complete checkpoint dirs, no age filter
        # (completeness is proven, not estimated — see _ckpt_write_complete). Both
        # feed the same push/verify/prune below. `matched=` stays the honest whole-
        # glob count either way; `files=` is what this pass actually ships.
        # Intersect the watcher's hits with what the bundle DECLARED as checkpoints:
        # a `checkpoints:` glob that does not cover checkpoint-<N>/ must not have
        # those bytes shipped just because a directory-name watcher noticed them.
        if [ -n "$fast_all" ]; then
          fast="$(LC_ALL=C comm -12 <(printf '%s\n' "$fast_all" | LC_ALL=C sort) \
                    <(_ckpt_dirs_from_matchlist "$cmatch") 2>/dev/null)"
        fi
        pinc=("${inc[@]}"); page=(--min-age "${JOBD_CKPT_MIN_AGE:-45s}"); trig=periodic
        if [ -n "$fast" ]; then
          trig=new-checkpoint; pinc=(); page=(); nship=0
          while IFS= read -r _d; do
            [ -n "$_d" ] || continue
            pinc+=(--include "$_d/**")
            nship=$(( nship + $(find "$run/$_d" -type f 2>/dev/null | grep -c . || echo 0) ))
          done <<EOF
$fast
EOF
          log "job $jobid: checkpoint sync FIRED on new checkpoint(s): $(printf '%s' "$fast" | tr '\n' ' ')"
        fi
        # An operator flush is a SUPERSET of the fire-on-arrival arm — whole
        # declared glob, no age filter — so it wins when both land in one pass.
        # No age filter means a live append can ship with a torn final line; that
        # is the same trade the preempt trap's final flush makes (a torn file
        # beats no file — the resume pulls it back and the entrypoint revalidates).
        if [ "${flushnow:-0}" = 1 ]; then
          trig=flush-now; pinc=("${inc[@]}"); page=(); nship="$nmatch"
          log "job $jobid: checkpoint sync FIRED by operator CHECKPOINT_NOW — whole glob, no age filter"
        fi
        # two-writer fence (HANDOFF_DESIGN §4): once a newer handoff epoch owns this
        # job's B2 state, refuse the sync — the understudy is the writer now. No-op /
        # fail-safe off the handoff path (unset epoch => never stale).
        if _handoff_epoch_stale "$jobid"; then
          log "job $jobid: checkpoint sync REFUSED — handoff epoch $HANDOFF_EPOCH stale (a newer epoch owns jobs/$jobid); not overwriting the understudy"
          continue
        fi
        # --- LIVE-APPEND TAIL SNAPSHOT (task #110) --------------------------------
        # The age window above is the whole reason an actively-appended file is
        # never shipped: every append bumps its mtime, so a file written faster
        # than JOBD_CKPT_MIN_AGE can NEVER clear the window. Measured on job
        # 20260803T130435-frontier-wave-3a68: ten consecutive passes at
        # matched=16/files=15, the same in-flight gens_PAD.jsonl skipped every
        # time, 864 rows regenerated from scratch on resume.
        #
        # The window STAYS (a mid-write copy can carry a torn final line, and
        # corrupt data beats bounded lost compute in exactly no world). Instead
        # jobd.py stages a snapshot of each skipped file cut at its LAST COMPLETE
        # LINE, and only for files it can prove are append-only NDJSON — see
        # jobmeta.ckpt_tail_snapshot for the six admission rules. What reaches B2
        # is therefore always a whole number of complete records.
        #
        # Bounded: `rm -rf` before and after, so the stage costs at most one pass
        # of the live files (JOBD_CKPT_TAIL_MAX_MB each, default 128). Fail-soft:
        # a helper that errors prints 0 and the ordinary pass above is untouched.
        # JOBD_CKPT_TAIL=0 disables it entirely.
        # NOT on the flush arm: it has no age window, so nothing was skipped and
        # the whole live file already ships above. Staging a cut-at-last-complete-
        # line copy to the SAME key would overwrite that with a SHORTER one.
        ntail=0
        if [ "${JOBD_CKPT_TAIL:-1}" != "0" ] && [ -z "$fast" ] \
           && [ "${flushnow:-0}" != 1 ] \
           && [ -n "$_minsec" ] && [ "$_minsec" != 0 ]; then
          rm -rf "$tstage" 2>/dev/null || true
          mkdir -p "$tstage" 2>/dev/null || true
          ntail="$("$PY" "$JH" tail-snapshot --run "$run" --matchlist "$cmatch" \
                     --min-age "$_minsec" --state "$tstate" --stage "$tstage" \
                     --max-mb "${JOBD_CKPT_TAIL_MAX_MB:-128}" 2>>"$terr" \
                   | tail -n 1)"
          case "$ntail" in ''|*[!0-9]*) ntail=0 ;; esac
        fi
        : > "$cerr"
        # b2x preserves the failure text this block classifies below: its 403
        # message carries "AccessDenied", which the auth-failure grep matches.
        if b2x_push "$run" "$B2W/jobs/$jobid/checkpoints/" \
             ${page[@]+"${page[@]}"} "${pinc[@]}" 2>"$cerr" \
           || rclone copy --fast-list ${page[@]+"${page[@]}"} "${pinc[@]}" \
             "$run" "$B2W/jobs/$jobid/checkpoints/" 2>"$cerr"; then
          sfails=0; rm -f "$wdir/.checkpoint_sync_failed"; _handoff_stamp_epoch "$jobid"
          n=$((n+1))
          # Same destination prefix, same relative paths: once the file goes
          # quiet it ages past the window and the ordinary pass above overwrites
          # the snapshot with the complete file. A tail push that fails is LOGGED
          # and not counted in `files=`, but never flips the pass to failed — the
          # durable state the ordinary pass just shipped did land.
          if [ "${ntail:-0}" -gt 0 ]; then
            if b2x_push "$tstage" "$B2W/jobs/$jobid/checkpoints/" 2>>"$terr" \
               || rclone copy --fast-list "$tstage" \
                    "$B2W/jobs/$jobid/checkpoints/" 2>>"$terr"; then
              nship=$(( nship + ntail ))
              log "job $jobid: checkpoint tail-snapshot shipped $ntail live append file(s)"
            else
              log "job $jobid: checkpoint tail-snapshot push FAILED ($ntail file(s)) — see $terr"
              ntail=0
            fi
          fi
          rm -rf "$tstage" 2>/dev/null || true
          # PUBLISH-BY-MARKER, and it must precede the prune: the marker's file
          # list is built from the LOCAL directory the prune is about to delete.
          # Until this lands, the checkpoint exists on B2 but is not SELECTABLE —
          # which is the whole point: an eviction anywhere in the multi-GB upload
          # leaves a directory `--resume auto` reaches past instead of dying on.
          _ckpt_mark_complete "$jobid" "$run" "$cmatch" "$markedf"
          # DELETE-AFTER-SYNC (box disk only; never B2). Every deletion is gated on
          # a read-back of the exact directory from B2 — see _ckpt_prune_synced.
          _ckpt_prune_synced "$jobid" "$run" "$cmatch"
          # Accumulate what was pruned, for the CHECKPOINTS_PRUNED.json marker the
          # finalize writes. A pruned dir is gone from the run dir BEFORE the
          # finalize publish globs out/**, so it exists on B2 under checkpoints/
          # ONLY — never under results/. That is a real, invisible narrowing of the
          # published artifact (`herdd job pull` reads results/), and any
          # bucket-side retention sweep that treats "results/ is non-empty" as
          # "results/ is a superset of checkpoints/" would be wrong about this job.
          # The marker is how both find out. (A file, not a variable: this is a
          # background SUBSHELL and cannot write the parent's scope.)
          if [ -n "$CKPT_PRUNE_LIST" ]; then
            printf '%s\n' "$CKPT_PRUNE_LIST" | tr ',' '\n' >> "$wdir/.checkpoint.pruned"
            # Raise the marker on B2 AT THE MOMENT the invariant breaks, not only
            # at finalize: from here on results/ is no longer a superset of
            # checkpoints/, and a box that dies before finalize would otherwise
            # leave a pruned job with no marker at all. The finalize write below
            # replaces this with the complete list. Once per job (sentinel file).
            if [ ! -f "$wdir/.pruned_marker_sent" ]; then
              if printf '{"job_dirs": ["%s"], "prefix": "jobs/%s/checkpoints/", "partial": true}\n' \
                   "$(printf '%s' "$CKPT_PRUNE_LIST" | sed 's/,/", "/g')" "$jobid" \
                 | rclone rcat "$B2W/jobs/$jobid/CHECKPOINTS_PRUNED.json" 2>/dev/null; then
                : > "$wdir/.pruned_marker_sent"
              fi
            fi
          fi
          # `beacon`: a checkpoint event is independent proof-of-progress and one
          # of the three stamps _job_liveness_epoch folds. Note the bytes
          # themselves went up via rclone (the bash half) — this call only
          # ANNOUNCES them, which is exactly why its failure is observability
          # loss and not work loss, and why it escalates rather than aborts.
          _py_call beacon emit "$jobid" checkpoint --instance-id "$IID" \
            ${JOB_EXP_ID:+--field "exp_id=$JOB_EXP_ID"} ${JOB_ARM:+--field "arm=$JOB_ARM"} \
            --field n="$n" --field files="$nship" --field matched="$nmatch" \
            --field tail="$ntail" \
            --field trigger="$trig" --field pruned="$CKPT_PRUNE_N" \
            --field published="$CKPT_MARK_N" \
            --field pruned_bytes="$CKPT_PRUNE_BYTES" \
            ${CKPT_PRUNE_LIST:+--field "pruned_dirs=$CKPT_PRUNE_LIST"}
        else
          # SYNC FAILED — keep the job running (compute is local) and keep retrying.
          rm -rf "$tstage" 2>/dev/null || true
          sfails=$((sfails+1))
          _stail="$(tr '\n' ' ' < "$cerr" 2>/dev/null | tail -c 200)"
          if grep -qiE 'InvalidAccessKeyId|SignatureDoesNotMatch|AccessDenied|Unauthorized|not valid| 403 ' "$cerr" 2>/dev/null; then
            _sreason="B2 AUTH FAILURE (dead/rotated key): ${_stail}"
            : > "$CRED_DIR/.cred_refresh_now" 2>/dev/null || true   # broker refresh hint (§2.6)
          else
            _sreason="rclone sync error: ${_stail}"
          fi
          printf '%s consecutive=%s\n%s\n' "$(date -u +%FT%TZ)" "$sfails" "$_sreason" \
            > "$wdir/.checkpoint_sync_failed" 2>/dev/null || true
          log "job $jobid: checkpoint sync FAILED (#$sfails) — $_sreason"
          # rate-limit: emit on the 1st failure, then every JOBD_SYNC_FAIL_EVERY-th
          # consecutive one, so a long outage does not spam the event log.
          if [ $(( (sfails - 1) % ${JOBD_SYNC_FAIL_EVERY:-5} )) -eq 0 ]; then
            # `report`: another failure report — same argument as stall_suspected.
            _py_call report emit "$jobid" checkpoint_sync_failed --instance-id "$IID" \
              ${JOB_EXP_ID:+--field "exp_id=$JOB_EXP_ID"} ${JOB_ARM:+--field "arm=$JOB_ARM"} \
              --field consecutive="$sfails" --field reason="$_sreason"
          fi
        fi
        # Mark EVERY dir the watcher saw (fast_all, not just the shipped subset)
        # seen after the ATTEMPT, success or failure. Two reasons: a failing push
        # must not re-arm the watcher every 5 s, and a dir outside the checkpoint
        # globs would otherwise break the wait on every single tick forever. The
        # periodic backstop re-ships the whole glob, so nothing is stranded.
        if [ -n "$fast_all" ]; then printf '%s\n' "$fast_all" >> "$seenf"; fi
      done
      rm -f "$cerr" ) &
    ckpid=$!
  fi

  wait "$epid"; local rc=$?
  # kill_tree, not kill: each of these three is a `while …; do sleep N; …; done`
  # subshell, and killing only the subshell ORPHANS its in-flight `sleep`, which
  # keeps the job's inherited stdout/stderr open. Any consumer reading jobd's
  # output through a PIPE therefore blocks up to JOBD_HEARTBEAT_S (default 60s)
  # after the job has already finished — visible as `job run-local | tee` hanging
  # for a minute on a job that took two seconds (found 2026-07-30 by the local
  # lane's test suite, where every trivial job took exactly 60s).
  kill_tree "$hbpid" 2>/dev/null || true
  [ -n "$ckpid" ] && kill_tree "$ckpid" 2>/dev/null || true
  [ -n "$cwpid" ] && kill_tree "$cwpid" 2>/dev/null || true
  local dur=$(( $(date +%s) - start_s ))
  log "job $jobid: entrypoint exited rc=$rc dur=${dur}s"

  # --- box preempt/park in flight? -> INTERRUPTED, not failed -------------------
  # When vast stops the box (eviction, `supervise` budget-park, idle self-park),
  # the daemon's SIGTERM trap raises $PREEMPT_MARK and its entrypoint is killed
  # (usually a signal rc>=128). Such a death is an INTERRUPTION: leave the job
  # non-terminal (no failed emit, NO results.DONE.json, keep .running) so the next
  # boot RESUMES it (bounded by max_restarts), instead of stranding a run that may
  # have finished locally. rc==0 always wins (a genuine clean finish beats the
  # park race); a cancel is its own terminal path below. The trap and this runner
  # race off the same box-stop signal, so wait briefly for the marker on a
  # signal-kill before deciding — an ordinary app crash (rc<128) fails fast.
  if [ "$rc" -ne 0 ] && [ ! -f "$wdir/.cancelled" ]; then
    if [ "$rc" -ge 128 ]; then
      _pw=0
      while [ ! -f "$PREEMPT_MARK" ] && [ "$_pw" -lt 30 ]; do sleep 0.1; _pw=$((_pw+1)); done
    fi
    if [ -f "$PREEMPT_MARK" ]; then
      log "job $jobid: interrupted by box preempt/park (rc=$rc) — resumable, not failed"
      return   # leave .running + no terminal marker; the trap already emitted preempted
    fi
  fi

  # --- interrupted weights/asset transfer? -> SELF-HEAL + one bounded retry -----
  # POSITION IS LOAD-BEARING: STRICTLY AFTER the preempt block above, which
  # `return`s before ever reaching here. A box stop kills an in-flight rclone pull
  # too and leaves exactly the same `.partial` corpses behind — that death is a
  # PREEMPTION and must bump .preempts against JOBD_PREEMPT_CAP, never the
  # transfer budget. Putting this first would silently reclassify every eviction
  # that happened to interrupt a download.
  #
  # Full rationale — why evidence and not rc, why the index has to be deleted, why
  # a separate counter — is in the block above `_transfer_retry_cap`.
  local _transfer_fail_reason=""
  if [ "$rc" -ne 0 ] && [ ! -f "$wdir/.cancelled" ] && [ "${JOBD_TRANSFER_HEAL:-1}" = "1" ] \
     && _interrupted_transfer_evidence "$jobid" "$wdir"; then
    local _tr _trcap _healed
    _tr="$(cat "$STATE_DIR/$jobid.transfer_retries" 2>/dev/null || echo 0)"
    case "$_tr" in (*[!0-9]*|"") _tr=0 ;; esac
    _trcap="$(_transfer_retry_cap)"
    log "job $jobid: rc=$rc with interrupted-transfer evidence — $TRANSFER_EVIDENCE"
    if _transfer_disk_exhausted "$ROOT"; then
      # ENOSPC is indistinguishable from an interrupt by the partials alone, and a
      # retry into a full disk buys the same failure twice. Its own named reason,
      # carrying the number, so the fix (a bigger --disk) is legible from the event.
      _transfer_fail_reason="insufficient_disk: ${TRANSFER_FREE_GB}GB free on $ROOT after an interrupted transfer (< ${JOBD_TRANSFER_MIN_FREE_GB:-5}GB floor) — not retried"
      log "job $jobid: $_transfer_fail_reason"
    elif _transfer_corruption "$wdir"; then
      _transfer_fail_reason="transfer_corruption: $TRANSFER_CORRUPTION is at its final path with a sha256 that disagrees with its manifest — not an interrupt, not retried"
      log "job $jobid: $_transfer_fail_reason"
    elif [ "$_tr" -ge "$_trcap" ] 2>/dev/null; then
      _transfer_fail_reason="interrupted_transfer: $TRANSFER_EVIDENCE (retry budget spent: $_tr/$_trcap)"
      log "job $jobid: $_transfer_fail_reason"
    else
      _healed="$(_heal_interrupted_transfer "$jobid" "$wdir")"
      # The breadcrumb is what makes the next claim a TRANSFER-resume rather than
      # a crash-restart — the same shape as the trap's .preempted breadcrumb, and
      # consumed the same way (poll_once's dual-counter block).
      : > "$STATE_DIR/$jobid.transfer_retry" 2>/dev/null || true
      log "job $jobid: interrupted transfer healed ($_healed) — retry $((_tr+1))/$_trcap in ${JOBD_TRANSFER_BACKOFF_S:-30}s (does NOT spend max_restarts)"
      emit "$jobid" transfer_retry --field rc="$rc" \
        --field attempt="$((_tr+1))" --field cap="$_trcap" \
        --field evidence="$TRANSFER_EVIDENCE" --field healed="$_healed" \
        --tail-file "$logf"
      # Backoff, then hand the job back to the scheduler NON-TERMINAL: no failed
      # emit, no results.DONE.json (which would make poll_once skip it forever),
      # .running cleared so the next pass re-claims and re-runs the entrypoint.
      sleep "${JOBD_TRANSFER_BACKOFF_S:-30}"
      rm -f "$STATE_DIR/$jobid.running" "$STATE_DIR/$jobid.staging"
      return
    fi
  fi

  # --- publish: results globs FIRST, log.txt, results.DONE.json LAST ----------
  # List-based `rclone copy --include` + `rcat` ONLY — never `copyto`: its
  # per-key HeadObject intermittently 403s on B2 (2026-07-10 it silently ate an
  # entire publish while the rcat'd events landed fine). Retries per the same
  # incident's S3-flake lesson.
  : > "$wdir/.uploaded"; : > "$wdir/.matched"
  ( shopt -s globstar nullglob dotglob
    cd "$run" || exit 0
    inc=()
    while IFS= read -r pat; do
      [ -n "$pat" ] || continue
      inc+=(--include "$pat")
      for f in $pat; do [ -f "$f" ] && printf '%s\n' "$f" >> "$wdir/.matched"; done
    done < "$wdir/.results.globs"
    [ -s "$wdir/.matched" ] || exit 0
    for i in 1 2 3; do
      if b2x_push "$run" "$B2W/jobs/$jobid/results/" "${inc[@]}" 2>>"$wdir/.publish.err" \
         || rclone copy --fast-list "${inc[@]}" "$run" "$B2W/jobs/$jobid/results/" \
              2>>"$wdir/.publish.err"; then
        cp "$wdir/.matched" "$wdir/.uploaded"; exit 0
      fi
      sleep $(( i * 10 ))
    done
    log "job $jobid: results publish FAILED after retries (see .publish.err)" )

  # --- publish verify: DONE must mean READABLE, not merely PUT -----------------
  # DEFENSE-IN-DEPTH (kept after the write-side root-cause fix). The recurring
  # mid-run checkpoint sync no longer writes jobs/<id>/results/ (it goes to
  # jobs/<id>/checkpoints/), so this finalize copy is now the ONLY writer of each
  # results/ key — a NEW object, which B2 serves with strong read-after-write. That
  # eliminates the overwrite window that caused the four controller-side false-fails
  # (2026-07-15/16/19/20: validate_generation_artifact read sha-of-empty for a
  # declared arm even past its ~158s retry budget because the checkpoint loop had
  # already shipped an early empty/partial version to the SAME key and finalize's
  # overwrite stayed eventually-consistent for ~90s). We keep this verify anyway:
  # it now confirms the single finalize write landed and is readable, and still
  # guards the rare preempt-then-resume path (where the N1b results-glob preempt
  # flush can put an earlier version at results/ before the resumed run re-publishes).
  # B2 object OVERWRITES are eventually consistent: `rclone copy` returning 0
  # means the PUT landed, but a `cat` of an overwritten key can keep serving the
  # STALE version for minutes after the copy. The
  # DONE-written-last doctrine is only real if "written" means readable at its
  # final bytes: read every published result BACK from B2 and require its
  # sha256 to match the local file before results.DONE.json goes up. A settled
  # object confirms on the FIRST read (zero sleeps on the happy path). Files
  # above JOBD_PUBLISH_VERIFY_MAX_MB (default 64 — model checkpoints, which no
  # manifest validator cats) are skipped rather than re-downloaded. On the 2nd
  # consecutive mismatch the file is re-pushed once via rcat: `kill $ckpid`
  # cannot reach an IN-FLIGHT checkpoint rclone child, and if that orphan
  # finishes AFTER the final copy the newest B2 version is the stale one — a
  # re-push makes the newest version the correct bytes again, then the re-read
  # loop just waits out consistency. Budget exhaustion still writes DONE (a
  # finished job must reach a terminal state; same tolerance as a failed
  # publish) but logs + emits `publish_verify_failed` so the controller's own
  # stale-read backstop and the operator see it.
  if [ -s "$wdir/.uploaded" ]; then
    local vmax vleft vsleep vf vsz vlsha vrsha vmiss vok
    vmax=$(( ${JOBD_PUBLISH_VERIFY_MAX_MB:-64} * 1024 * 1024 ))
    vleft=${JOBD_PUBLISH_VERIFY_TIMEOUT_S:-300}
    vsleep=${JOBD_PUBLISH_VERIFY_BACKOFF_S:-2}
    vfail=""
    vok=""   # comma-joined results/-relative paths confirmed readable==local sha
    while IFS= read -r vf; do
      [ -n "$vf" ] && [ -f "$run/$vf" ] || continue
      vsz=$(stat -c %s "$run/$vf" 2>/dev/null || stat -f %z "$run/$vf" 2>/dev/null || echo 0)
      if [ "${vsz:-0}" -gt "$vmax" ]; then
        log "job $jobid: publish verify: SKIP $vf (${vsz}B > ${vmax}B cap)"
        continue
      fi
      vlsha=$(sha256sum "$run/$vf" 2>/dev/null | cut -d' ' -f1)
      [ -n "$vlsha" ] || continue
      vmiss=0
      while :; do
        vrsha=$(rclone cat "$B2/jobs/$jobid/results/$vf" </dev/null 2>/dev/null \
                  | sha256sum | cut -d' ' -f1)
        [ "$vrsha" = "$vlsha" ] && break
        vmiss=$((vmiss + 1))
        [ "$vmiss" -eq 2 ] && \
          rclone rcat "$B2W/jobs/$jobid/results/$vf" < "$run/$vf" 2>/dev/null || true
        if [ "$vleft" -le 0 ]; then vfail="$vf"; break; fi
        [ "$vsleep" -gt "$vleft" ] && vsleep=$vleft
        log "job $jobid: publish verify: $vf not readable at its published bytes yet (want ${vlsha:0:12} got ${vrsha:0:12}) — re-read in ${vsleep}s (${vleft}s left)"
        sleep "$vsleep"; vleft=$((vleft - vsleep))
        vsleep=$((vsleep * 2)); [ "$vsleep" -gt 30 ] && vsleep=30
      done
      [ -n "$vfail" ] && break
      vok="${vok:+$vok,}$vf"   # this file matched its local sha on read-back
    done < "$wdir/.uploaded"
    if [ -n "$vfail" ]; then
      log "job $jobid: publish verify FAILED — $vfail never matched its local sha within ${JOBD_PUBLISH_VERIFY_TIMEOUT_S:-300}s; writing DONE anyway (controller stale-read backstop + triage)"
      emit "$jobid" publish_verify_failed --field file="$vfail" \
        --field budget_s="${JOBD_PUBLISH_VERIFY_TIMEOUT_S:-300}"
    elif [ -n "$vok" ]; then
      # Positive signal: EVERY uploaded (non-oversized) result read back from B2
      # at its final local sha256. The controller's validate_generation_artifact
      # trusts this and SKIPS its redundant, cross-client-racy per-arm re-read
      # (the 5x false-fail). `files` lists the results/-relative paths verified —
      # the same suffix the manifest arm paths normalize to. Emitted BEFORE the
      # DONE marker, so it is present by the time the controller (keyed off DONE)
      # can look. Not emitted when the verify was skipped or .uploaded was empty.
      emit "$jobid" publish_verified --field files="$vok" \
        --field n_files="$(printf '%s' "$vok" | awk -F, '{print NF}')"
    fi
  fi

  # manifest (event payload) + DONE marker (full) in one python pass.
  #
  # `checkpoints_pruned` (added 2026-08-05 after adversarial review): the mid-run
  # delete-after-sync removes checkpoint dirs from the run dir BEFORE the finalize
  # publish globs `out/**`, so those dirs exist on B2 under jobs/<id>/checkpoints/
  # ONLY. `herdd job pull` and the DONE manifest read jobs/<id>/results/, which
  # therefore no longer holds the full dose-curve grid. That narrowing must not be
  # invisible: it is recorded here AND as its own small B2 object below, so a
  # human reading the manifest and a bucket-side retention sweep reasoning about
  # "is results/ a superset of checkpoints/" both get a straight answer.
  # `$IID` is passed so the marker can SAY WHICH ATTEMPT WROTE IT. The marker is
  # a mutable key at a stable name and `job requeue` clears nothing, so after a
  # re-open the previous attempt's marker sits at the live job's key and reads as
  # "finished" (false terminal, measured 2026-08-28 on
  # 20260828T064840-v16-r64-8c87 while it was 42% through training). `written_ts`
  # is what lets a reader date it against the re-open; `instance_id` is the
  # corroborating "and it was a different box". jobmeta.classify_done_marker
  # consumes both, and falls back to the B2 mtime for markers written before
  # this stamp existed — do not make either field required.
  "$PY" - "$run" "$wdir/.uploaded" "$rc" "$dur" "$wdir/.manifest.json" \
        "$wdir/results.DONE.json" "$wdir/.checkpoint.pruned" "$wdir/.pruned.json" \
        "${IID:-}" <<'PY'
import datetime, json, os, sys
run, listf, rc, dur, manf, donef, prunedf, prunedout, iid = sys.argv[1:10]
files = []
try:
    for line in open(listf):
        p = line.strip()
        if not p:
            continue
        fp = os.path.join(run, p)
        files.append({"path": p, "size": os.path.getsize(fp) if os.path.isfile(fp) else None})
except OSError:
    pass
pruned = []
try:
    pruned = sorted({ln.strip() for ln in open(prunedf) if ln.strip()})
except OSError:
    pass
json.dump(files, open(manf, "w"))
# runmeta.now_ts's format, spelled here because the box has no runmeta on $PATH
# in this pass: colon-free UTC with milliseconds, so it orders lexicographically
# against the event stream a reader compares it to.
_n = datetime.datetime.now(datetime.timezone.utc)
done = {"rc": int(rc), "duration_s": int(dur), "n_results": len(files),
        "written_ts": _n.strftime("%Y%m%dT%H%M%S") + "%03dZ" % (_n.microsecond // 1000),
        "results": files}
if iid:
    done["instance_id"] = iid
if pruned:
    done["checkpoints_pruned"] = pruned
    done["checkpoints_pruned_note"] = (
        "these checkpoint dirs were pruned from the box disk mid-run and are on B2 "
        "under jobs/<JOB_ID>/checkpoints/ ONLY, not under results/")
    json.dump({"job_dirs": pruned, "prefix": "jobs/<JOB_ID>/checkpoints/"},
              open(prunedout, "w"))
json.dump(done, open(donef, "w"))
PY
  # A standalone object, not just a field: a reader that has to parse the whole
  # DONE manifest to learn this would not bother, and `ckpt_retention.py` gates on
  # its mere PRESENCE. Written BEFORE results.DONE.json so the DONE-written-last
  # doctrine is preserved. RETRIED, and loud on exhaustion: this object is what
  # stops a bucket-side sweep from deleting the only copy of the pruned dirs, so
  # failing to place it silently is the one failure mode that must not be quiet.
  # (A partial marker was already raised mid-run, at the first prune — this
  # replaces it with the complete list.)
  if [ -s "$wdir/.pruned.json" ]; then
    _pm=0
    for _pi in 1 2 3; do
      rclone rcat "$B2W/jobs/$jobid/CHECKPOINTS_PRUNED.json" < "$wdir/.pruned.json" \
        2>/dev/null && { _pm=1; break; }
      sleep $(( _pi * 3 ))
    done
    if [ "$_pm" != 1 ]; then
      log "job $jobid: FAILED to write CHECKPOINTS_PRUNED.json after 3 tries — results/ is NOT a superset of checkpoints/ for this job and B2 does not say so; a bucket-side retention sweep must not treat this job as redundant (the list is also in results.DONE.json's checkpoints_pruned)"
      emit "$jobid" checkpoints_pruned_marker_failed \
        --field dirs="$(tr '\n' ',' < "$wdir/.checkpoint.pruned" 2>/dev/null | sed 's/,$//')"
    fi
  fi

  emit "$jobid" results_uploaded --results-json "$wdir/.manifest.json"
  # rcat streams a PUT (no HeadObject) — same transport the events use.
  rclone rcat "$B2W/jobs/$jobid/log.txt" < "$logf" 2>/dev/null || true
  # results.DONE.json LAST — its presence means every result already landed
  # AND (publish verify above) is readable on B2 at its final bytes.
  rclone rcat "$B2W/jobs/$jobid/results.DONE.json" < "$wdir/results.DONE.json" 2>/dev/null || true

  if [ -f "$wdir/.cancelled" ]; then
    # operator cancel: TERMINAL + non-resumable. Partial results/log already
    # published above (crash-safe), but NO clean-finish claim beyond that.
    emit "$jobid" cancelled --field rc="$rc" --field reason="cancelled by operator" \
      --tail-file "$logf"
    mark_terminal "$jobid" cancelled
  elif [ "$rc" -eq 0 ]; then
    emit "$jobid" done --field rc=0 --results-json "$wdir/.manifest.json"
    mark_terminal "$jobid" done
  else
    local why="rc=$rc"; [ "$rc" -eq 124 ] && why="timeout after ${JOB_TIMEOUT_S}s"
    # A transfer verdict that refused to retry (disk full, corruption, budget
    # spent) names ITSELF here, following the asset_stage_timeout: precedent: a
    # bare "rc=4" sends the operator to re-run the same job on the same shape of
    # box, which is how defect #77 stayed invisible for a whole campaign.
    [ -n "${_transfer_fail_reason:-}" ] && why="$_transfer_fail_reason"
    emit "$jobid" failed --field rc="$rc" --field reason="$why" --tail-file "$logf"
    mark_terminal "$jobid" failed
  fi

  # Allocated-vs-used, measured HERE because the scrub below is about to delete
  # the checkpoints — after it, the peak this job needed is unrecoverable.
  disk_usage_report "$jobid" terminal "$wdir"

  # Publish any Triton kernels this job compiled beyond the pulled baseline
  # (backgrounded + bounded; a failed job's kernels compiled fine and are just
  # as reusable, so this runs on every terminal path — the preempt/park path
  # returned far above and correctly skips it on a dying box).
  [ "$gpus" != "-" ] && triton_cache_push_bg "$jobid"

  # --- END-OF-RUN CHECKPOINT SCRUB (box disk only; never B2) --------------------
  # Owner 2026-08-05: "we need to scrub checkpoints after a run is finished and we
  # have extracted metadata. no strong reason to keep them around."
  #
  # THIS IS THE LAST THING run_job_body DOES, and the position is load-bearing, not
  # stylistic. Every training bundle's `results:` glob is `out/**` — which CONTAINS
  # the checkpoint dirs — so a scrub placed any earlier would delete the bytes the
  # publish is still uploading, or the files the manifest pass is still `stat`ing.
  # By here: results pushed (retried), publish-verified, manifest + log.txt +
  # results.DONE.json written, terminal event emitted.
  #
  # THREE GATES, all of which must hold:
  #   1. `.uploaded` non-empty  — the results push actually succeeded (it is written
  #      only on a successful copy). No push, no scrub.
  #   2. `vfail` empty          — publish verify did not fail. A run whose B2 state
  #      is already suspect keeps its local copy for triage.
  #   3. per-directory read-back inside _ckpt_scrub_local — the dir must be present
  #      on B2 at the same name set and byte total, under checkpoints/ or results/.
  # The PREEMPT path never reaches here (it `return`s far above), so an interrupted,
  # resumable job never gets scrubbed.
  if [ -s "$wdir/.uploaded" ] && [ -z "${vfail:-}" ]; then
    _ckpt_scrub_local "$jobid" "$run" "$wdir"
    if [ "$CKPT_SCRUB_N" -gt 0 ]; then
      log "job $jobid: checkpoint scrub freed ${CKPT_SCRUB_BYTES}B across $CKPT_SCRUB_N dir(s)"
      emit "$jobid" checkpoints_scrubbed --field n="$CKPT_SCRUB_N" \
        --field bytes="$CKPT_SCRUB_BYTES" --field dirs="$CKPT_SCRUB_LIST"
    fi
  elif [ "${JOBD_CKPT_SCRUB:-1}" = "1" ]; then
    log "job $jobid: checkpoint scrub SKIPPED — $([ -s "$wdir/.uploaded" ] && echo "publish verify failed (${vfail:-})" || echo "no results were uploaded")"
  fi

  rm -f "$STATE_DIR/$jobid.running"
}

# --- one poll pass: schedule any runnable tickets (strict FIFO) -----------------
poll_once() {
  reap
  _py_breadcrumb_check
  # FAIL CLOSED BEFORE CLAIMING (FAILCLOSED_DESIGN §4). Claiming a ticket we
  # cannot emit a single event for is strictly worse than not claiming it: the
  # ticket is consumed, the job may even run, and no observer can ever learn
  # what happened to it. Preflights fail closed; refuse and let maybe_idle_park
  # take the box down. Already-running jobs are untouched (reap above still
  # harvests them).
  if [ "$PY_HALF" = "broken" ]; then
    return 0
  fi
  local tickets
  tickets="$(rclone lsf "$B2/jobs/queue/$IID/" 2>/dev/null | grep '\.json$' | sort || true)"
  [ -n "$tickets" ] || return 0
  local t
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    local jobid="${t%.json}"
    [ -f "$STATE_DIR/$jobid.terminal" ] && continue          # done here before
    [ -n "${JOB_PID[$jobid]:-}" ] && continue                # running right now
    if [ "$MAXJ" != "0" ] && [ "$SPAWNED" -ge "$MAXJ" ]; then break; fi

    # remote terminal? results.DONE.json is written LAST by ANY box that ran
    # the job to completion (this box pre-cache, or a retargeted twin). Cache
    # the answer locally so known-finished jobs cost zero B2 reads per poll.
    local donekey
    donekey="$(rclone lsf "$B2/jobs/$jobid/results.DONE.json" 2>/dev/null || true)"
    if [ -n "$donekey" ]; then
      # ...UNLESS an operator requeued it (`herdd job requeue`). A run that dies
      # with rc!=0 publishes its partial results + results.DONE.json BEFORE it
      # emits `failed`, so the marker is present for EVERY infra-killed job — this
      # skip is what made a same-JOB_ID re-open impossible, and it would swallow
      # the requeue in silence. The re-minted ticket carries `requeued_ts`
      # (jobmeta.REQUEUE_TICKET_MARK); its presence is the explicit "run this
      # again" and outranks the prior attempt's marker.
      # Bounded WITHOUT extra state: this box has no $STATE_DIR terminal
      # breadcrumb for the job yet (checked above), and the moment this attempt
      # goes terminal `mark_terminal` writes one — so a requeue is honoured
      # exactly once per box and a re-failed job cannot loop. (`job requeue`
      # refuses to target the box that already failed the job, for that reason.)
      # One extra `rclone cat`, only on the rare DONE-marker-present path.
      local rq_ts
      rq_ts="$(ticket_requeue_ts "$jobid")"
      if [ "$rq_ts" = "?" ]; then
        # The read failed, so we do not KNOW whether an operator requeued this.
        # `mark_terminal` is permanent and unobservable from B2; a skipped poll
        # costs one interval. Never latch on ignorance — but BOUND the ignorance:
        # a persistently unreadable ticket re-read B2 every poll forever, so the
        # count of CONSECUTIVE unknowns persists in $STATE_DIR and falls through
        # to the DONE marker's own answer once it is clearly not a blip. Any
        # readable answer resets it, so a transient blip still costs one poll.
        local unkf="$STATE_DIR/$jobid.done_unknown" unk=0
        unk="$(cat "$unkf" 2>/dev/null || true)"
        case "$unk" in ''|*[!0-9]*) unk=0 ;; esac   # a torn/absent count is 0, never a hang
        unk=$((unk + 1))
        echo "$unk" > "$unkf" 2>/dev/null || true
        if [ "$unk" -ge "$DONE_UNKNOWN_MAX" ]; then
          log "job $jobid: ticket unreadable for $unk consecutive polls — latching remote-done on the DONE marker (a later requeue must re-mint on another box)"
          rm -f "$unkf"
          mark_terminal "$jobid" remote-done; continue
        fi
        log "job $jobid: results.DONE.json is present but the ticket could not be read this pass ($unk/$DONE_UNKNOWN_MAX) — not latching remote-done; retrying next poll"
        continue
      fi
      rm -f "$STATE_DIR/$jobid.done_unknown"
      if [ -n "$rq_ts" ]; then
        log "job $jobid: results.DONE.json from a prior attempt is present, but the ticket carries an operator REQUEUE ($rq_ts) — running it"
      else
        mark_terminal "$jobid" remote-done; continue
      fi
    fi

    # cancelled before we claimed? `herdd job cancel` deletes the ticket, so
    # normally it is already gone from this listing — but if the delete lagged
    # the listing (B2 read-after-write on a DIFFERENT key), the CANCEL marker
    # closes the race: never run it, mark it terminal locally so it is skipped
    # and never counted as pending. The CLI already wrote the terminal
    # `cancelled` event, so we do not re-emit.
    local cankey
    cankey="$(rclone lsf "$B2/jobs/$jobid/CANCEL" 2>/dev/null || true)"
    if [ -n "$cankey" ]; then
      log "job $jobid: CANCEL marker present before claim — not running"
      mark_terminal "$jobid" cancelled; continue
    fi

    # prior claim by this box (box-actor event key "<ts>-box_<IID>-<nonce>")?
    # v1 skipped such tickets forever; v2 treats claimed-but-unfinished as
    # INTERRUPTED and resumes. A lone cli `submitted` event is a fresh claim.
    local have resume=0
    have="$(rclone lsf "$B2/jobs/$jobid/events/" 2>/dev/null)"
    case "$have" in
      *"-box_${IID}-"*) resume=1 ;;
    esac

    # ticket + canonical config (needed for scheduling BEFORE spawn)
    local wdir="$JOBS_DIR/$jobid"
    rm -rf "$wdir"; mkdir -p "$wdir"
    if ! rclone copyto "$B2/jobs/queue/$IID/$jobid.json" "$wdir/.ticket.json" 2>/dev/null; then
      log "job $jobid: ticket download failed — skip this pass"; rm -rf "$wdir"; continue
    fi
    local JOB_ID JOB_NAME JOB_BUNDLE_SHA JOB_ENTRYPOINT JOB_TIMEOUT_S JOB_NEEDS_GPU \
          JOB_NEEDS_GPU_RAM_GB JOB_NEEDS_GPUS JOB_NEEDS_VENV JOB_N_RESULTS \
          JOB_CHECKPOINT_S JOB_MAX_RESTARTS JOB_EXP_ID JOB_ARM
    # `if ! eval "$(...)"` WAS A BLIND GUARD and this is the 14th silent call
    # site — the one that looked protected. Command substitution DISCARDS the
    # inner exit status, and `eval ""` returns 0, so when jobd.py died on stderr
    # and printed nothing, `eval` succeeded, the "ticket parse failed" branch
    # never ran, and every JOB_* var below silently became the empty string.
    # Verified 2026-08-14: `bash -c 'eval "$(python3 -c "import nope" 2>/dev/null)"; echo $?'`
    # prints 0. Capture first, CHECK the status, then eval.
    #
    # `local` on its own line: `local x="$(cmd)"` would make $? the exit status
    # of `local` (always 0) and re-introduce exactly the bug being fixed here.
    local _prep_out _prep_rc=0
    _prep_out="$("$PY" "$JH" prepare "$wdir/.ticket.json" \
                   --env-out "$wdir/.job.env" --results-out "$wdir/.results.globs" \
                   --checkpoints-out "$wdir/.checkpoint.globs" \
                   --assets-out "$wdir/.assets.tsv" \
                   --asset-require-dir "$wdir/.asset_require" 2>"$wdir/.prepare.err")" || _prep_rc=$?
    if [ "$_prep_rc" -ne 0 ] || [ -z "$_prep_out" ]; then
      # rc 3 (EXIT_STRUCTURAL) means the interpreter cannot run our code at all.
      # That is a BOX fault, not a ticket fault: failing the ticket would be a
      # lie that also destroys it for every future box. Leave it queued, declare
      # the python half broken, and let the fail-closed path take the box down.
      if [ "$_prep_rc" = "3" ]; then
        log "job $jobid: NOT a bad ticket — the python half cannot run (rc=3); leaving the ticket queued"
        _py_broken "ticket prepare exited EXIT_STRUCTURAL (rc=3)"
        rm -rf "$wdir"; break
      fi
      # prepare's stderr, kept (it used to go to /dev/null): an rc with no cause
      # is unattributable after the fact, and the 2026-08-19 retarget race cost a
      # night precisely because nobody could tell WHY rc=1. LAST line, not first:
      # a python child's last stderr line is the exception, the first is
      # "Traceback (most recent call last):". One line, so a traceback cannot
      # bloat an immutable event body.
      local _prep_err=""
      _prep_err="$(tail -n 1 "$wdir/.prepare.err" 2>/dev/null | tr -d '\r' | cut -c1-200)"
      # ABSENT is not MALFORMED — the same call the rc=3 branch above makes, for
      # the same reason. `job retarget` / fleetd's eviction replacement delete the
      # old queue pointer, and that delete can land between poll_once's LIST and
      # this prepare; condemning the ticket then destroys it for every future box
      # AND latches a breadcrumb that makes THIS box skip the JOB_ID forever.
      #
      # The discriminator is a re-LIST of the PARENT prefix — the same command
      # poll_once opened with — never a stat of the key itself: rclone reports a
      # missing object as a plain nonzero exit with an error on stderr, which is
      # indistinguishable from a transport failure. rc=0 with the key absent from
      # a listing that succeeded is PROOF the ticket was deleted. A nonzero list
      # is INCONCLUSIVE and falls through to the fail path below, exactly as
      # before — including the case where this box's queue prefix is now
      # completely empty, which B2 may itself answer nonzero.
      local _q_out _q_rc=0
      _q_out="$(rclone lsf "$B2/jobs/queue/$IID/" 2>/dev/null)" || _q_rc=$?
      if [ "$_q_rc" = "0" ] && ! printf '%s\n' "$_q_out" | grep -qxF "$jobid.json"; then
        log "job $jobid: ticket VANISHED from jobs/queue/$IID/ mid-pass (prepare rc=$_prep_rc${_prep_err:+: $_prep_err}) — moved out from under this poller, NOT a bad ticket; emitting nothing and leaving no breadcrumb"
        rm -rf "$wdir"; continue
      fi
      log "job $jobid: ticket parse failed (rc=$_prep_rc, ${#_prep_out} bytes of vars)${_prep_err:+ — $_prep_err}"
      emit "$jobid" failed --field reason="ticket parse failed (rc=$_prep_rc)${_prep_err:+: $_prep_err}"
      mark_terminal "$jobid" failed; continue
    fi
    eval "$_prep_out"

    # retarget-continuation marker: `job retarget` (cmd_job_retarget) stamps the
    # ticket with the SOURCE box id. On a fresh box the box-local restart-count
    # breadcrumbs do not exist (restart_count=0), so this marker is what tells
    # run_job_body to pull prior checkpoints back instead of restarting from
    # scratch (HANDOFF_DESIGN §4 gap). Read straight off the ticket (best-effort:
    # empty on a normal job — the runner's predicate then relies on restart_count
    # or the B2 event-history probe). A `local` here so run_job_body inherits it
    # via dynamic scope, same as the JOB_* vars above.
    local JOB_RETARGETED_FROM
    JOB_RETARGETED_FROM="$("$PY" - "$wdir/.ticket.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("retargeted_from") or "")
except Exception:
    pass
PY
)"

    # restart cap: THREE separate counters persist on local disk (a crash-loop is
    # bounded even when the event log is unreachable) so an infrastructure failure
    # does not exhaust a healthy job's crash budget (box 44566398 lineage: three
    # preempt-resumes used to burn the default max_restarts and fail a job that
    # never crashed):
    #   .attempts — fresh claim + genuine CRASH-restarts; cap = 1 + max_restarts.
    #   .preempts — PREEMPT-mediated resumes (the box was stopped by vast/park/
    #               supervise, whether or not a signal ever reached us);
    #               cap = JOBD_PREEMPT_CAP (default 20), a runaway backstop ONLY.
    #   .transfer_retries — the runner classified the failure as an INTERRUPTED
    #               TRANSFER, healed the half-pulled model dir and asked for one
    #               more go (defect #77); cap = _transfer_retry_cap (1, hard max
    #               2). Same argument as .preempts, different infrastructure: a
    #               flaky rclone pull is not the job crashing.
    # A resume is a PREEMPT-resume on either of two evidences, strongest first:
    #   (a) the trap's per-job .preempted breadcrumb (a signal DID arrive;
    #       consumed once, on claim below), or
    #   (b) the container-boot nonce recorded at the job's spawn differs from
    #       the current one — the BOX WENT DOWN between spawn and resume (vast
    #       eviction, park, host reboot: none of which deliver a signal here,
    #       see the boot-nonce block at the top). detect=boot_change.
    # BOTH preempt evidences OUTRANK the transfer breadcrumb: a box that died
    # during the retry backoff was evicted, and the eviction is what has to be
    # counted (the runner-side classifier is ordered the same way, after the
    # preempt block). Only then does the transfer breadcrumb decide.
    # A resume with the SAME nonce and no breadcrumb died on a LIVE box (e.g. OOM
    # took the runner) — a genuine crash, still counted against the crash cap; a
    # resume with no nonce data (spawned by an older bundle, or no writable tmpfs)
    # falls back to crash, the pre-inference behavior. rc 0 always wins (terminal
    # before we get here).
    local attempts preempts transfer_retries is_preempt=0 is_transfer=0
    local resume_detect="" _prior_nonce="" _trcap
    attempts="$(cat "$STATE_DIR/$jobid.attempts" 2>/dev/null || echo 0)"
    preempts="$(cat "$STATE_DIR/$jobid.preempts" 2>/dev/null || echo 0)"
    transfer_retries="$(cat "$STATE_DIR/$jobid.transfer_retries" 2>/dev/null || echo 0)"
    case "$attempts" in (*[!0-9]*|"") attempts=0;; esac
    case "$preempts" in (*[!0-9]*|"") preempts=0;; esac
    case "$transfer_retries" in (*[!0-9]*|"") transfer_retries=0;; esac
    if [ "$resume" = "1" ]; then
      if [ -f "$STATE_DIR/$jobid.preempted" ]; then
        is_preempt=1; resume_detect="trap"
      else
        _prior_nonce="$(cat "$STATE_DIR/$jobid.bootnonce" 2>/dev/null || true)"
        if [ -n "$_prior_nonce" ] && [ -n "$BOOT_NONCE" ] \
           && [ "$_prior_nonce" != "$BOOT_NONCE" ]; then
          is_preempt=1; resume_detect="boot_change"
        elif [ -f "$STATE_DIR/$jobid.transfer_retry" ]; then
          is_transfer=1; resume_detect="transfer"
        elif [ -n "$_prior_nonce" ] && [ "$_prior_nonce" = "$BOOT_NONCE" ]; then
          resume_detect="same_boot"      # died on a live box: a real crash
        else
          resume_detect="unknown"        # no nonce data: old-bundle spawn
        fi
      fi
    fi
    if [ "$is_preempt" = "1" ]; then
      if [ "$preempts" -ge "${JOBD_PREEMPT_CAP:-20}" ]; then
        log "job $jobid: preempt cap exceeded ($preempts preempt-resumes, cap=${JOBD_PREEMPT_CAP:-20})"
        emit "$jobid" failed \
          --field reason="preempt cap exceeded ($preempts preempt-resumes, cap=${JOBD_PREEMPT_CAP:-20})"
        mark_terminal "$jobid" failed; rm -f "$STATE_DIR/$jobid.preempted"; continue
      fi
    elif [ "$is_transfer" = "1" ]; then
      # Backstop only — the runner refuses to write the breadcrumb once the budget
      # is spent, so this fires only if one survived (a stale breadcrumb from a
      # boot that never got to claim). Terminal with the NAMED reason either way.
      _trcap="$(_transfer_retry_cap)"
      if [ "$transfer_retries" -ge "$_trcap" ] 2>/dev/null; then
        log "job $jobid: interrupted-transfer retry cap exceeded ($transfer_retries, cap=$_trcap)"
        emit "$jobid" failed \
          --field reason="interrupted_transfer: retry cap exceeded ($transfer_retries retries, cap=$_trcap)"
        mark_terminal "$jobid" failed; rm -f "$STATE_DIR/$jobid.transfer_retry"; continue
      fi
    else
      if [ "$attempts" -ge $(( 1 + ${JOB_MAX_RESTARTS:-0} )) ]; then
        log "job $jobid: restart cap exceeded ($attempts attempts, max_restarts=$JOB_MAX_RESTARTS)"
        emit "$jobid" failed \
          --field reason="restart cap exceeded ($attempts attempts, max_restarts=$JOB_MAX_RESTARTS)"
        mark_terminal "$jobid" failed; continue
      fi
    fi

    # scheduling: resolve the card request against the live inventory
    local want="$JOB_NEEDS_GPUS" gpus="-"
    [ "$want" = "all" ] && want="$NGPU"
    case "$want" in (*[!0-9]*|"") want=0;; esac
    if [ "${JOBD_SKIP_GPU:-0}" = "1" ] && [ "$NGPU" -eq 0 ]; then want=0; fi
    if [ "$want" -gt 0 ]; then
      if [ "$want" -gt "$NGPU" ]; then
        log "job $jobid: unmet needs — needs.gpus=$JOB_NEEDS_GPUS but box has $NGPU"
        emit "$jobid" failed --field reason="needs.gpus=$JOB_NEEDS_GPUS: box has $NGPU GPUs"
        mark_terminal "$jobid" failed; continue
      fi
      if ! gpus="$(pick_gpus "$want" "${JOB_NEEDS_GPU_RAM_GB:-0}")"; then
        # Distinguish TRANSIENT (cards busy — wait, FIFO) from IMPOSSIBLE (no
        # card on this box is big enough even when idle). Impossible used to
        # take the silent `break` too, so a ticket the box could NEVER run sat
        # `submitted` forever while the box billed — indistinguishable, to
        # every observer, from one merely waiting its turn. Treat it like the
        # needs.gpus branch above: say so, and fail the ticket.
        _biggest=0
        for _m in "${GPU_MEM[@]:-0}"; do
          [ "$_m" -gt "$_biggest" ] 2>/dev/null && _biggest="$_m"
        done
        if [ "${JOB_NEEDS_GPU_RAM_GB:-0}" -gt "$_biggest" ] 2>/dev/null; then
          log "job $jobid: unmet needs — needs.gpu_ram_gb=${JOB_NEEDS_GPU_RAM_GB} but the largest card is ${_biggest} GB"
          emit "$jobid" failed --field \
            reason="needs.gpu_ram_gb=${JOB_NEEDS_GPU_RAM_GB}: largest card is ${_biggest} GB"
          mark_terminal "$jobid" failed; continue
        fi
        # STRICT FIFO: the oldest ticket that does not fit blocks younger ones
        # (a whole-box job can never be starved by a stream of 1-GPU arms).
        rm -rf "$wdir"; break
      fi
    else
      if [ "$CPU_RUNNING" -ge "$CPU_SLOTS" ]; then rm -rf "$wdir"; break; fi
    fi

    # claim/resume + spawn. restart_count = total prior runs (crash + preempt) so
    # ANY resume triggers the checkpoint pull-back (JOB_RESTART_COUNT>0); a fresh
    # claim is 0. Increment ONLY the counter that applies: a preempt-resume bumps
    # .preempts and consumes its breadcrumb (leaving the crash budget untouched); a
    # fresh claim / crash-restart bumps .attempts.
    local restart_count=$(( attempts + preempts + transfer_retries ))
    local _kind=crash
    [ "$is_preempt" = "1" ] && _kind=preempt
    [ "$is_transfer" = "1" ] && _kind=transfer
    if [ "$resume" = "1" ]; then
      # `detect` says WHICH evidence classified this resume (trap | boot_change
      # | transfer | same_boot | unknown) — the audit trail for the budget it
      # drains.
      emit "$jobid" resumed --field attempt="$((restart_count+1))" \
        --field kind="$_kind" --field detect="${resume_detect:-unknown}"
    else
      emit "$jobid" claimed
    fi
    if [ "$is_preempt" = "1" ]; then
      echo $((preempts+1)) > "$STATE_DIR/$jobid.preempts"
      rm -f "$STATE_DIR/$jobid.preempted"          # consumed
    elif [ "$is_transfer" = "1" ]; then
      echo $((transfer_retries+1)) > "$STATE_DIR/$jobid.transfer_retries"
      rm -f "$STATE_DIR/$jobid.transfer_retry"     # consumed
    else
      echo $((attempts+1)) > "$STATE_DIR/$jobid.attempts"
    fi
    # Record WHICH container boot spawns this run (persistent disk) — the other
    # half of the resume-time preempt inference above. Rewritten every spawn so
    # the comparison is always against the run that actually died; removed when
    # the nonce is unavailable so a stale value can never masquerade as data.
    if [ -n "$BOOT_NONCE" ]; then
      printf '%s\n' "$BOOT_NONCE" > "$STATE_DIR/$jobid.bootnonce" 2>/dev/null || true
    else
      rm -f "$STATE_DIR/$jobid.bootnonce"
    fi
    # 9>&-: do NOT leak the daemon's flock fd into the runner — `timeout` puts
    # entrypoints in their own process group, so a leaked fd would keep the
    # lock held (blocking the next daemon boot) even after the daemon dies.
    run_job_body "$jobid" "$wdir" "$gpus" "$restart_count" 9>&- &
    local pid=$!
    JOB_PID["$jobid"]="$pid"; JOB_GPU_SLOT["$jobid"]="$gpus"
    if [ "$gpus" = "-" ]; then
      CPU_RUNNING=$((CPU_RUNNING+1))
    else
      local g; IFS=',' read -ra g <<< "$gpus"
      for i in "${g[@]}"; do [ -n "$i" ] && GPU_OWNER["$i"]="$jobid"; done
    fi
    # .running = the preemption trap's + adoption's view of live work:
    # "pid gpus wdir ckpt_s exp arm" (all space-free tokens).
    printf '%s %s %s %s %s %s\n' "$pid" "$gpus" "$wdir" "${JOB_CHECKPOINT_S:-0}" \
      "${JOB_EXP_ID:--}" "${JOB_ARM:--}" > "$STATE_DIR/$jobid.running"
    SPAWNED=$((SPAWNED+1))
    status_marker
  done <<< "$tickets"
  return 0
}

# --- preemption trap (SPOT_DESIGN §3.3) ---------------------------------------
# On an EXTERNAL SIGTERM/SIGINT to the daemon: for EVERY running job, emit ONE
# non-terminal `preempted` event (cheap, tells the fold + `job supervise` why the
# log went quiet — the job is NOT failed; jobd resumes it on the next boot), then
# one bounded final checkpoint-glob flush per checkpointing job. Events first,
# flushes second: vast's kill grace is undocumented, so the cheap writes must not
# queue behind the expensive ones. Does NOT fire for the internal `timeout` kill
# of an entrypoint (that never signals this shell) or an idle daemon (no .running
# files). `trap -` disarms first so a second signal can't re-enter.
# FAILCLOSED_DESIGN §6, the `dying` exemption: the three `$PY $JH emit` calls in
# this trap keep their bare `|| true` and are NOT accounted. The box is already
# going down under a hard `timeout 20`; declaring the python half broken here
# would buy nothing (there is no future work to refuse) and would cost a beacon
# write and a park attempt on a shutdown path whose grace period vast does not
# document. Escalation needs a future to protect, and this path has none.
_jobd_preempt() {
  trap - TERM INT
  # FIRST: raise the preempt marker so any live runner whose entrypoint the box
  # stop is ALSO killing does NOT record a terminal `failed` (the 2026-07-12
  # budget-park bug: supervise parked a box whose train job had just finished; the
  # SIGKILL'd torchrun made the runner mark the job `failed`-terminal -> skipped
  # forever on resume, the completed checkpoint-688 stranded). A single `: >` write
  # lands well before the runner's post-`wait` cleanup reaches its terminal branch.
  : > "$PREEMPT_MARK" 2>/dev/null || true
  local rf jid pid gpus wdir ckpt exp arm
  # cheap events + a PER-JOB preempt breadcrumb BEFORE the (expensive) flushes.
  # The .preempted breadcrumb persists across the resume boot (unlike the global
  # .preempting marker, which is cleared at boot) so the next claim can tell a
  # PREEMPT-resume (generous cap) from a CRASH-restart (max_restarts) — three
  # outbids must not burn a healthy job's crash budget (box 44566398 lineage).
  # See poll_once's dual-counter cap logic.
  for rf in "$STATE_DIR"/*.running; do
    [ -f "$rf" ] || continue
    jid="$(basename "${rf%.running}")"
    read -r pid gpus wdir ckpt exp arm < "$rf" || continue
    : > "$STATE_DIR/$jid.preempted" 2>/dev/null || true
    timeout 20 "$PY" "$JH" emit "$jid" preempted --instance-id "$IID" \
      $([ "${exp:--}" != "-" ] && printf -- '--field exp_id=%s' "$exp") \
      $([ "${arm:--}" != "-" ] && printf -- '--field arm=%s' "$arm") \
      >/dev/null 2>&1 || true
  done
  # THEN, BEFORE any flush: ask the TRAINER for one fresh COMPLETE checkpoint on
  # local disk (SPOT_DESIGN §3.3, `_preempt_local_save` in preempt_trap.sh).
  #
  # WHY THIS IS HERE AT ALL (2026-08-06): the flushes below can only ever push
  # bytes that ALREADY EXIST. Killed 19 minutes into a 20-minute SAVE_STEPS
  # window they are 19 minutes stale, and no amount of flushing invents the
  # missing progress. The run lane has asked the trainer first since the
  # primitive landed; the JOBS lane — which is where every production training
  # run actually happens (v7..v11) — never did. It had no SIGUSR1 path at all,
  # so the module could have been staged perfectly and still never fired.
  #
  # ONCE, not per job: the pid dir is BOX-GLOBAL (/workspace/.preempt_save_pids),
  # so a second call would re-signal the same ranks — and `agree()` is one-shot
  # per request, so the second ask is at best wasted seconds on the death path.
  # We use the first RUNNING job that declared checkpointing as the search root
  # ($wdir/work, under which the trainer's out/checkpoint-N/ lives), because that
  # is the tree whose new marker proves the save landed.
  #
  # Ordering: AFTER the `preempted` events (cheap writes must never queue behind
  # anything) and BEFORE the flushes, so the flush uploads the checkpoint we just
  # forced rather than the stale one. Bounded by PREEMPT_SAVE_WAIT_S and totally
  # best-effort — a box that cannot make a checkpoint still flushes what it has.
  if declare -F _preempt_local_save >/dev/null 2>&1; then
    for rf in "$STATE_DIR"/*.running; do
      [ -f "$rf" ] || continue
      jid="$(basename "${rf%.running}")"
      read -r pid gpus wdir ckpt exp arm < "$rf" || continue
      [ "${ckpt:-0}" -gt 0 ] 2>/dev/null || continue
      # Lane emitter for _preempt_save_report: every outcome (including a SKIP)
      # becomes a job event, so "the safety net did not run" is visible on B2
      # instead of dying in a box log. Closes over $jid by redefinition.
      _preempt_save_emit() {
        timeout 20 "$PY" "$JH" emit "$jid" preempt_save --instance-id "$IID" \
          --field result="$1" --field detail="${2:-}" >/dev/null 2>&1 || true
      }
      CKPT_DIR="$wdir/work" _preempt_local_save || true
      unset -f _preempt_save_emit
      break
    done
  fi
  # THEN the bounded flushes (per job): checkpoint globs, THEN results globs. Both
  # use NO --min-age (grab the freshest bytes — a torn file beats no file; the
  # resume pulls newest state back and the entrypoint re-validates), a trailing-slash
  # dest (rides out B2's flaky HEAD dest-check), each wrapped in `timeout` so the
  # death path can never hang. Checkpoint globs go to jobs/<id>/checkpoints/ (the same
  # prefix the mid-run sync + resume pull-back use — never results/, which stays a
  # single new-object write at finalize). Results globs are flushed too (N1b): a job
  # that had WRITTEN its final results but not yet published (publish runs post-`wait`,
  # which the box stop pre-empts) would otherwise strand them — checkpoint globs alone
  # do not cover a job that opted out of mid-run checkpointing; those DO go to results/
  # (they are final bytes, and the job resumes+re-publishes them, so publish-verify
  # still covers any resulting overwrite window on the rare preempt-then-resume path).
  for rf in "$STATE_DIR"/*.running; do
    [ -f "$rf" ] || continue
    jid="$(basename "${rf%.running}")"
    read -r pid gpus wdir ckpt exp arm < "$rf" || continue
    # two-writer fence (HANDOFF_DESIGN §4 interleave 2): a stale husk's final flush,
    # landing AFTER the understudy already pulled, would give it an older/torn base.
    # Skip the byte flushes when a newer epoch owns the job (the understudy is the
    # writer); fail-safe off the handoff path. final_flush below still emits (it is a
    # cheap non-terminal signal, not a write to the run's checkpoint state).
    if _handoff_epoch_stale "$jid"; then
      log "job $jid: preempt flush REFUSED — handoff epoch $HANDOFF_EPOCH stale (a newer epoch owns jobs/$jid)"
    else
      # DURABILITY, not speed: this is the eviction path, and a multi-GB flush
      # at stock rclone concurrency does not finish inside 45 s on a per-flow-
      # shaped host — state was being lost silently. b2x's --deadline also
      # orders NEWEST FIRST, so a budget that still runs out leaves the newest
      # checkpoints fully uploaded rather than an arbitrary torn subset.
      # The outer `timeout 45` stays as a belt-and-braces guard on the death path.
      if [ "${ckpt:-0}" -gt 0 ] 2>/dev/null && [ -s "$wdir/.checkpoint.globs" ]; then
        local cinc=() cpat
        # `.preempt_*` EXCLUDED, and the exclusion is load-bearing — the same
        # reason the run lane's trap excludes it (preempt_trap.sh). b2x orders
        # NEWEST FIRST, and `.preempt_save_complete` is BY CONSTRUCTION the newest
        # file in the checkpoint it certifies (rank 0 writes it last, after every
        # rank reported). Without this the deadline would upload the completeness
        # FLAG first and then truncate before the ~646 MB optimizer and ~323 MB
        # adapter — publishing a GREEN FLAG over weights that are not there, and
        # inverting the very prefix-completeness property newest-first exists to
        # give. This matters more now that the local save above actually produces
        # these markers on the jobs lane.
        #
        # The checkpoint BYTES still go to B2 in full; only the marker is held
        # back. The marker is a LOCAL-DISK claim, for salvage to read off the
        # dying box (salvage.py, instance->instance). B2 completeness comes from
        # multipart atomicity + b2x completing each object before starting the
        # next, so a truncated flush leaves the newest checkpoints WHOLE rather
        # than an arbitrary torn subset.
        #
        # Rule ORDER matters: rclone/b2x take first-match-wins, so the exclude
        # must precede the `--include` list or the includes would re-admit it.
        cinc+=(--exclude '.preempt_*')
        while IFS= read -r cpat; do [ -n "$cpat" ] && cinc+=(--include "$cpat"); done < "$wdir/.checkpoint.globs"
        b2x_push "$wdir/work" "$B2W/jobs/$jid/checkpoints/" --deadline 40s "${cinc[@]}" \
          || timeout 45 rclone copy --fast-list "${cinc[@]}" "$wdir/work" "$B2W/jobs/$jid/checkpoints/" 2>/dev/null || true
      fi
      if [ -s "$wdir/.results.globs" ]; then
        local rinc=() rpat
        while IFS= read -r rpat; do [ -n "$rpat" ] && rinc+=(--include "$rpat"); done < "$wdir/.results.globs"
        b2x_push "$wdir/work" "$B2W/jobs/$jid/results/" --deadline 40s "${rinc[@]}" \
          || timeout 45 rclone copy --fast-list "${rinc[@]}" "$wdir/work" "$B2W/jobs/$jid/results/" 2>/dev/null || true
      fi
    fi
    # final_flush: the cutover fence the handoff understudy waits on (HANDOFF_DESIGN
    # §4). Emitted per job AFTER its bounded preempt-time flushes above complete —
    # i.e. the box, on its way down, has pushed its last checkpoint/results for THIS
    # job to B2; the understudy write-enables only once it observes this. Non-terminal
    # (the job still resumes). Best-effort + timeout-bounded, like the `preempted`
    # emit above — a dying box must never hang or crash on it.
    timeout 20 "$PY" "$JH" emit "$jid" final_flush --instance-id "$IID" \
      $([ "${exp:--}" != "-" ] && printf -- '--field exp_id=%s' "$exp") \
      $([ "${arm:--}" != "-" ] && printf -- '--field arm=%s' "$arm") \
      >/dev/null 2>&1 || true
  done
  exit 143
}
trap _jobd_preempt TERM INT

# --- idle self-park check (v2.1) ----------------------------------------------
# Called once per poll loop AFTER poll_once. The box is "busy" if a job is
# running OR any queue ticket is not yet locally terminal; busy resets the idle
# clock. Idle past the grace -> emit parked_self + self-park (or, with no
# self-control key, emit `drained` once and let the laptop park). Never fires in
# JOBD_ONCE mode (that path exits before this loop).
maybe_idle_park() {
  [ "$IDLE_PARK" = "1" ] || return 0
  reap
  _py_breadcrumb_check
  if [ "${#JOB_PID[@]}" -gt 0 ]; then
    # A RUNNING JOB OUTRANKS A BROKEN PYTHON HALF (FAILCLOSED_DESIGN §6). The
    # compute is local and the checkpoint bytes travel by rclone, so real work
    # is still being produced and still landing on B2 — that is an independent
    # progress signal, and killing it to punish a dead event emitter would
    # destroy hours of GPU time to save minutes of it. Mid-run we fail OPEN; the
    # escalation is that the beacon now says `pyhalf=broken` out loud.
    LAST_BUSY_TS="$(date +%s)"; EVER_BUSY=1; return 0
  fi

  # FAIL CLOSED: python half broken and NOTHING running. This box cannot claim
  # work (poll_once refuses), cannot emit a single lifecycle event, and holds no
  # job worth protecting — it can only bill. This is the exact hole that cost
  # $1.742 on 47737955, and it is checked BEFORE the pending-ticket test on
  # purpose: see the note there.
  if [ "$PY_HALF" = "broken" ]; then
    local bnow bidle
    bnow="$(date +%s)"; bidle=$(( bnow - PY_BROKEN_TS ))
    if [ "$bidle" -ge "$PY_BROKEN_PARK_S" ] 2>/dev/null; then
      log "python half broken ${bidle}s with no running job — parking (reason=$PY_HALF_REASON)"
      jobd_status PARKING "reason=pyhalf_broken broken_s=$bidle"
      # NOT emit_box: the ONLY action on this path must not require the
      # subsystem that failed. self_park is curl, jobd_status is rclone; both
      # live strictly below the python half.
      self_park
      [ -n "${JOBD_PARK_CMD:-}" ] && exit 0
      return 0
    fi
    log "python half broken ${bidle}s (park at ${PY_BROKEN_PARK_S}s), no job running"
    return 0
  fi

  # pending work? any ticket without a local terminal marker.
  #
  # THE INVERSION THIS USED TO HAVE: a ticket the box could not parse never got
  # a terminal marker, so it read as "pending work" on EVERY poll, reset
  # LAST_BUSY_TS, and the idle clock could never reach any deadline. The box
  # that most needed to park was the one structurally incapable of parking. It
  # also falsified a written assumption in herdd._jobd_heartbeat_epoch_soft —
  # "jobd self-parks an idle box at JOBD_IDLE_PARK_S anyway, so the verdict and
  # the daemon's own intent agree" — precisely in the case that matters. The
  # broken-half branch above now runs FIRST, so pending tickets can no longer
  # hold a box that cannot act on them.
  local tickets t jid pending=0
  tickets="$(rclone lsf "$B2/jobs/queue/$IID/" 2>/dev/null | grep '\.json$' || true)"
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    jid="${t%.json}"
    [ -f "$STATE_DIR/$jid.terminal" ] && continue
    pending=1; break
  done <<< "$tickets"
  if [ "$pending" = "1" ]; then LAST_BUSY_TS="$(date +%s)"; EVER_BUSY=1; return 0; fi

  # tally terminal outcomes for the parked_self event (returning-agent surfacing).
  # ANY terminal marker means a job ran HERE — so the box is draining, not a box
  # that never saw a job (a fast job can finish between polls, never caught mid
  # flight by the reap above, which would otherwise misclassify the reason).
  local nd=0 nf=0 f st
  for f in "$STATE_DIR"/*.terminal; do
    [ -f "$f" ] || continue
    read -r st _ < "$f" 2>/dev/null || continue
    case "$st" in done|remote-done) nd=$((nd+1)) ;; failed) nf=$((nf+1)) ;; esac
  done
  [ $(( nd + nf )) -gt 0 ] && EVER_BUSY=1

  local now idle deadline reason
  now="$(date +%s)"; idle=$(( now - LAST_BUSY_TS ))
  if [ "$EVER_BUSY" = "1" ]; then
    reason=drained
    # single-arm box (<=1 terminal job ran here) + operator did NOT pin the
    # grace -> use the shorter single-arm grace so a done training box reclaims
    # GPU billing fast (higher uptime-utilization). >1 job (a multi-job/bakeoff
    # box) or an explicit JOBD_IDLE_PARK_S keeps the full multi-job grace.
    if [ -z "$IDLE_PARK_S_EXPLICIT" ] && [ $(( nd + nf )) -le 1 ]; then
      deadline="$IDLE_PARK_S_SINGLE"
    else
      deadline="$IDLE_PARK_S"
    fi
  else deadline="$NO_JOB_PARK_S"; reason=no_job; fi
  [ "$idle" -ge "$deadline" ] 2>/dev/null || return 0

  # genuinely cannot self-stop (no scoped key AND not a test) -> the fallback the
  # design names: emit ONE `drained` box event; the laptop supervise/CLI parks.
  if [ -z "${JOBD_PARK_CMD:-}" ] && ! _iid_key; then
    [ "$PARK_EMITTED" = "1" ] && return 0
    log "idle ${idle}s but no self-control key — emitting drained (laptop must park)"
    # CIRCULAR ERROR PATH, FIXED. This is the one action the design names for a
    # box that cannot stop itself, and it used to run SOLELY through emit_box —
    # i.e. the exact subsystem whose death is the thing most likely to have
    # stranded the box. Worse, PARK_EMITTED was latched unconditionally right
    # after, so a swallowed emit was recorded as a delivered one and never
    # retried. Write the fact on the bash/rclone channel FIRST (it is strictly
    # below the python half and cannot be taken out by the same fault), then
    # attempt the richer event, and only latch if the box could actually have
    # sent it.
    jobd_status DRAINED "reason=$reason idle_s=$idle n_done=$nd n_failed=$nf"
    if [ "$PY_HALF" = "broken" ]; then
      log "python half broken — the drained EVENT cannot be emitted; the DRAINED beacon above is the only report"
      PARK_EMITTED=1        # the beacon IS the report; do not spin retrying a dead emitter
      return 0
    fi
    emit_box drained "reason=$reason" "idle_s=$idle" "n_done=$nd" "n_failed=$nf"
    PARK_EMITTED=1
    return 0
  fi

  if [ "$PARK_EMITTED" != "1" ]; then
    log "idle ${idle}s >= ${deadline}s (ever_busy=$EVER_BUSY reason=$reason) — self-parking"
    jobd_status PARKING "reason=$reason idle=${idle}s"
    emit_box parked_self "reason=$reason" "idle_s=$idle" "n_done=$nd" "n_failed=$nf"
    PARK_EMITTED=1
  fi
  self_park
  # A successful real park stops the box (this shell dies mid-loop). If we're
  # still here it was the TEST seam (exit so the harness sees a clean stop) or a
  # park that did not take (return -> the next tick re-issues it, no re-emit).
  [ -n "${JOBD_PARK_CMD:-}" ] && exit 0
  return 0
}

log "jobd up (root=$ROOT poll=${POLL}s gpus=$NGPU cpu_slots=$CPU_SLOTS once=${JOBD_ONCE:-0} "\
"idle_park=$IDLE_PARK/${IDLE_PARK_S}s single=${IDLE_PARK_S_SINGLE}s${IDLE_PARK_S_EXPLICIT:+ (pinned)} no_job=${NO_JOB_PARK_S}s)"
pyhalf_selftest      # capability gate: FAIL CLOSED before advertising for work
scratch_probe        # read-only host facts (fs/RAM/tmpfs); no policy, never fatal
gemm_probe           # dense-bf16 GEMM ceiling; telemetry only, bounded, never fatal
cpu_probe            # core throughput + scaling; runs on GPU-less boxes too
hostfacts_drain      # ship anything a PREVIOUS container left undrained
triton_cache_boot_pull   # warm the shared Triton JIT cache; backgrounded, fail-open
adopt_running
reap_orphan_gpu_procs
status_marker
if [ "${JOBD_ONCE:-0}" = "1" ]; then
  poll_once
  # drain: tests (and one-shot ops) need "pass done" to mean "jobs done"
  while [ "${#JOB_PID[@]}" -gt 0 ]; do sleep 1; reap; disk_hw_tick; done
  hostfacts_drain   # after the jobs, or a one-shot run ships nothing it harvested
  exit 0
fi
while true; do
  poll_once
  staging_status      # heartbeat live asset-pull mbps while any job is staging
  beacon_tick         # periodic JOBD_STATUS so idle-and-fine != dead
  disk_hw_tick        # throttled df -> box disk high-water mark (telemetry only)
  hostfacts_drain     # ship any harvested host facts a job dropped; never fatal
  maybe_refresh_creds
  # MAXJ reached and everything drained -> exit (bounded-batch mode)
  if [ "$MAXJ" != "0" ] && [ "$SPAWNED" -ge "$MAXJ" ] && [ "${#JOB_PID[@]}" -eq 0 ]; then
    log "JOBD_MAX_JOBS=$MAXJ reached and drained — exiting"
    exit 0
  fi
  maybe_idle_park
  sleep "$POLL"
done
