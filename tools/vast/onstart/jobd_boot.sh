# --- jobd provision-time bootstrap (herdd launch --jobs) --------------------
# Prepended to the launch onstart so a box starts jobd AT BOOT — it begins
# polling its B2 queue (jobs/queue/<IID>/) the moment it is up, with no separate
# `herdd job attach` round-trip. Design: JOBS_DESIGN.md ("Provision-time jobd").
#
# WHY a B2 pull and not an inline daemon: jobd.sh + its helpers are ~60 KiB, far
# over Vast's 16 KiB onstart cap. So the LAPTOP stages the daemon files as a
# content-addressed tar to b2:$B2_BUCKET/jobs/jobd-boot/<sha>.tar and bakes only
# the <sha> into this tiny stanza (the rehydrate pattern). No secret literals
# live here: B2 creds arrive as container env (--env, injected by vast); this
# script only REFERENCES $B2_* by name. The daemon self-discovers its instance id
# from $INSTANCE_ID/$CONTAINER_ID (vast injects CONTAINER_ID; INSTANCE_ID is NOT
# always present — the fallback chain is live-verified necessary, box 44482324),
# so the queue path needs no id baked at launch — resolving the chicken-and-egg
# (IID unknown until the API returns). Plain tar (not tar.zst) so extraction
# needs no zstd at boot.
#
# WHY the setsid dance (live incident, box 44482324): vast's onstart runner
# reaps its PROCESS GROUP when the onstart script exits — a plain `( ... ) &`
# background subshell was killed mid rclone-install (one log line, then
# nothing: no /workspace/jobd, no daemon, the else-branch never ran). So the
# bootstrap body is written to a file and launched in its OWN SESSION (setsid;
# nohup fallback) with stdin/stdout/stderr fully DETACHED — the same discipline
# the body itself applies when spawning jobd. All bootstrap output lands in
# /workspace/jobd-boot.log (persistent, so a dead bootstrap is diagnosable
# without hunting the onstart log).
#
# Placeholders substituted by herdd at launch: @JOBD_BUNDLE_SHA@.
JOBD_BOOT_WS="${JOBD_BOOT_WS:-/workspace}"   # test seam: retarget staging + log
mkdir -p "$JOBD_BOOT_WS"
cat > "$JOBD_BOOT_WS/.jobd_boot.sh" <<'JOBD_BOOT_BODY'
#!/usr/bin/env bash
set -u
# handshake marker: by the time bash executes THIS line the worker is already in
# its own session (setsid ran pre-exec) — the stanza waits for the marker before
# returning, so an onstart-group reap right after exit can never catch us.
_WS="$(cd "$(dirname "$0")" && pwd)"
: > "$_WS/.jobd_boot.started" 2>/dev/null || true
_jlog() { echo ">> [jobd-boot] $(date -u +%FT%TZ) $*"; }
# JOBD_BOOT_DIR / JOBD_BOOT_NO_START are TEST seams (no-ops in prod): retarget
# the install dir and skip the final daemon spawn so the pull+extract+env-file
# path is exercisable under the fake-rclone harness.
JOBD_DIR="${JOBD_BOOT_DIR:-/workspace/jobd}"
# need B2 creds (passed as container env). Absent -> no queue transport; bail
# quietly so a non-jobs onstart is unaffected.
if [ -z "${B2_BUCKET:-}" ] || [ -z "${B2_KEY_ID:-}" ] || \
   [ -z "${B2_APPLICATION_KEY:-}" ] || [ -z "${B2_S3_ENDPOINT:-}" ]; then
  _jlog "B2_* env not set — skipping jobd autostart"
  exit 0
fi
# rclone (idempotent install; same fallbacks as onstart/train.sh). Baked into
# train:t214+ (train-env/Dockerfile), so this chain is the path for UPSTREAM
# images and older tags only — it stays, and stays bounded.
if ! command -v rclone >/dev/null 2>&1; then
  _jlog "installing rclone"
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
        timeout 180 apt-get install -y -qq rclone >/dev/null 2>&1; } || true
fi
if ! command -v rclone >/dev/null 2>&1; then
  _jlog "!! rclone install failed — no queue transport, no jobd"
  exit 1
fi
# b2: remote from env (secret lands in the on-box conf, never in this script)
# RCLONE_CONFIG: listremotes below reads it, so the writer must agree or the
# idempotency check never sees what was written. (Body is a VERBATIM onstart
# prelude against a 16 KiB cap — keep additions here terse.)
RCONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
mkdir -p "$(dirname "$RCONF")"
if ! rclone listremotes 2>/dev/null | grep -q '^b2:'; then
  cat >> "$RCONF" <<RCEOF
[b2]
type = s3
provider = Other
access_key_id = ${B2_KEY_ID}
secret_access_key = ${B2_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
no_check_bucket = true
RCEOF
  chmod 600 "$RCONF"
fi
# Option-1b scoped write remote: when the launcher shipped a prefix-restricted
# write key (B2_WRITE_KEY_ID), the RO [b2] above can read but not write — jobd's
# writes (all under jobs/) go through [b2w]. Absent B2_WRITE_* the box has one
# bucket-wide key and everything stays on [b2]. See CREDENTIAL_LIFECYCLE.md.
if [ -n "${B2_WRITE_KEY_ID:-}" ] && [ -n "${B2_WRITE_APPLICATION_KEY:-}" ] \
   && ! rclone listremotes 2>/dev/null | grep -q '^b2w:'; then
  cat >> "$RCONF" <<RCEOF
[b2w]
type = s3
provider = Other
access_key_id = ${B2_WRITE_KEY_ID}
secret_access_key = ${B2_WRITE_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
no_check_bucket = true
RCEOF
  chmod 600 "$RCONF"
fi
# PUBLISH remote: a training bundle publishes its named adapter to
# checkpoints/<RUN_NAME>/, which the jobs/-scoped [b2w] key may not write (a B2
# key carries ONE namePrefix). Its own scoped key becomes [b2p]. Absent
# B2_PUBLISH_* there is no [b2p] and a publish stage fails loudly — the
# submit-time write-scope preflight is what keeps such a bundle off the box.
# B2_PUBLISH_KEY_SCOPE_FIX_2026-08-05.md / CREDENTIAL_LIFECYCLE.md
if [ -n "${B2_PUBLISH_KEY_ID:-}" ] && [ -n "${B2_PUBLISH_APPLICATION_KEY:-}" ] \
   && ! rclone listremotes 2>/dev/null | grep -q '^b2p:'; then
  cat >> "$RCONF" <<RCEOF
[b2p]
type = s3
provider = Other
access_key_id = ${B2_PUBLISH_KEY_ID}
secret_access_key = ${B2_PUBLISH_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
no_check_bucket = true
RCEOF
  chmod 600 "$RCONF"
fi
# pull + extract the daemon bundle (content-addressed, plain tar). Retry with
# backoff so a transient B2 blip never leaves the box daemon-less, and CAPTURE
# rclone's stderr instead of swallowing it: the 2026-07-12 box-44566398 boot log
# read only "pull/extract failed — no jobd" with no cause, hiding an
# InvalidAccessKeyId (the box's ephemeral B2 key had been revoked mid-session by a
# colliding concurrent `launch --jobs`). Download to a file first (so the exit
# code + stderr are inspectable), then extract.
mkdir -p "$JOBD_DIR"
_btar="$JOBD_DIR/.jobd-bundle.tar"
_berr="$JOBD_DIR/.jobd-bundle.err"
_jobd_ok=0
for _try in 1 2 3 4 5; do
  : > "$_berr"
  if rclone cat "b2:${B2_BUCKET}/jobs/jobd-boot/@JOBD_BUNDLE_SHA@.tar" >"$_btar" 2>"$_berr" \
       && tar -x -C "$JOBD_DIR" -f "$_btar" 2>>"$_berr" && [ -f "$JOBD_DIR/jobd.sh" ]; then
    _jobd_ok=1; break
  fi
  # An auth failure (revoked / expired / wrong key) is NOT transient: stop
  # retrying and surface it LOUD. Every B2-dependent path — this pull, checkpoint
  # sync, heartbeats — is dead until the key is rotated from the laptop.
  if grep -qiE 'InvalidAccessKeyId|SignatureDoesNotMatch|AccessDenied|Unauthorized|not valid| 403 ' "$_berr" 2>/dev/null; then
    _jlog "!! B2 AUTH FAILURE pulling the daemon bundle — the box B2 key is DEAD (revoked/expired)"
    _jlog "!! $(tr '\n' ' ' < "$_berr" 2>/dev/null | tail -c 300)"
    _jlog "!! rotate the key from the laptop: herdd job attach ${INSTANCE_ID:-${CONTAINER_ID:-<IID>}}"
    break
  fi
  _jlog "daemon bundle pull/extract attempt ${_try}/5 failed — retrying"
  _jlog "   $(tr '\n' ' ' < "$_berr" 2>/dev/null | tail -c 200)"
  sleep $(( _try * 5 ))
done
rm -f "$_btar"
if [ "$_jobd_ok" = 1 ]; then
  chmod +x "$JOBD_DIR/jobd.sh" 2>/dev/null || true
  # env file for jobd + the resume hook (creds already in the environment;
  # write them so a resumed boot re-sources without re-reading launch env).
  # PRESERVE an existing jobd.env: onstart re-runs on every park/resume with the
  # ORIGINAL launch-time env, but `herdd job attach` (the rotation lane) and
  # cred_client.py rewrite jobd.env with FRESHER keys mid-session — regenerating
  # here would revert a rotated key to possibly-revoked launch creds (and a
  # past expiry). The launch env never changes post-launch, so an existing sane
  # jobd.env is always at least as fresh as what we would write. Regenerate only
  # when absent or torn (no B2_KEY_ID line — a boot killed mid-write).
  if [ -f "$JOBD_DIR/jobd.env" ] \
     && grep -q '^export B2_KEY_ID=' "$JOBD_DIR/jobd.env" 2>/dev/null; then
  _jlog "jobd.env already present (attach/cred_client may have rotated keys) — preserving"
  else
  {
    echo "export B2_BUCKET=${B2_BUCKET}"
    echo "export B2_KEY_ID=${B2_KEY_ID}"
    echo "export B2_APPLICATION_KEY=${B2_APPLICATION_KEY}"
    echo "export B2_S3_ENDPOINT=${B2_S3_ENDPOINT}"
    echo "export B2_REGION=${B2_REGION:-us-west-004}"
    # scoped write key (Option 1b) — present only when the launcher shipped one;
    # jobd/jobmeta route writes to [b2w] when B2_WRITE_KEY_ID is set.
    [ -n "${B2_WRITE_KEY_ID:-}" ]         && echo "export B2_WRITE_KEY_ID=${B2_WRITE_KEY_ID}"
    [ -n "${B2_WRITE_APPLICATION_KEY:-}" ] && echo "export B2_WRITE_APPLICATION_KEY=${B2_WRITE_APPLICATION_KEY}"
    # publish key (checkpoints/ grant) — an entrypoint's publish stage writes
    # through [b2p]; persisted so a resumed boot keeps the grant.
    [ -n "${B2_PUBLISH_KEY_ID:-}" ]         && echo "export B2_PUBLISH_KEY_ID=${B2_PUBLISH_KEY_ID}"
    [ -n "${B2_PUBLISH_APPLICATION_KEY:-}" ] && echo "export B2_PUBLISH_APPLICATION_KEY=${B2_PUBLISH_APPLICATION_KEY}"
    # cred-broker identity (cred-broker-buildout.md §2.1/§2.6): persisted so a
    # RESUMED boot re-sources them and jobd's maybe_refresh_creds + cred_client.py
    # can rotate the key in place pre-expiry. Absent on pre-broker launches ->
    # nothing written -> the refresh hook stays a no-op.
    [ -n "${B2_KEY_EXPIRES_AT:-}" ]   && echo "export B2_KEY_EXPIRES_AT=${B2_KEY_EXPIRES_AT}"
    [ -n "${CRED_BROKER_URL:-}" ]     && echo "export CRED_BROKER_URL=${CRED_BROKER_URL}"
    [ -n "${BOX_IDENTITY_NONCE:-}" ]  && echo "export BOX_IDENTITY_NONCE=${BOX_IDENTITY_NONCE}"
    [ -n "${CRED_ROLE:-}" ]           && echo "export CRED_ROLE=${CRED_ROLE}"
    [ -n "${TS_AUTHKEY:-}" ]          && echo "export TS_AUTHKEY=${TS_AUTHKEY}"
    echo "export INSTANCE_ID=${INSTANCE_ID:-${CONTAINER_ID:-}}"
    [ -n "${JOBD_IDLE_PARK:-}" ]    && echo "export JOBD_IDLE_PARK=${JOBD_IDLE_PARK}"
    [ -n "${JOBD_IDLE_PARK_S:-}" ]  && echo "export JOBD_IDLE_PARK_S=${JOBD_IDLE_PARK_S}"
    [ -n "${JOBD_NO_JOB_PARK_S:-}" ] && echo "export JOBD_NO_JOB_PARK_S=${JOBD_NO_JOB_PARK_S}"
  } > "$JOBD_DIR/jobd.env"
  chmod 600 "$JOBD_DIR/jobd.env"
  fi
  # persistence: onstart re-runs on every resume, but so does the launch stanza —
  # jobd's own flock makes a double start harmless (the second exits). Still
  # install the guarded hook so a bare container-restart (no onstart) revives
  # jobd too. Mirrors `herdd job attach`.
  if [ -f /root/onstart.sh ] && ! grep -q jobd-autostart /root/onstart.sh 2>/dev/null; then
    printf '%s\n' "[ -f $JOBD_DIR/jobd.env ] && (. $JOBD_DIR/jobd.env && nohup bash $JOBD_DIR/jobd.sh >>$JOBD_DIR/jobd.log 2>&1 &) # jobd-autostart (launch --jobs)" >> /root/onstart.sh
  fi
  if [ "${JOBD_BOOT_NO_START:-0}" = "1" ]; then
    _jlog "JOBD_BOOT_NO_START=1 — extracted, not starting jobd (test seam)"
  else
    _jlog "starting jobd (queue jobs/queue/${INSTANCE_ID:-${CONTAINER_ID:-?}}/)"
    ( . "$JOBD_DIR/jobd.env" && nohup bash "$JOBD_DIR/jobd.sh" >"$JOBD_DIR/jobd.log" 2>&1 & ) </dev/null
  fi
else
  _jlog "!! daemon bundle pull/extract failed after retries (jobs/jobd-boot/@JOBD_BUNDLE_SHA@.tar) — no jobd (cause above)"
fi
JOBD_BOOT_BODY
chmod +x "$JOBD_BOOT_WS/.jobd_boot.sh"
echo ">> [jobd-boot] launching detached (log: $JOBD_BOOT_WS/jobd-boot.log)"
rm -f "$JOBD_BOOT_WS/.jobd_boot.started"
if command -v setsid >/dev/null 2>&1; then
  setsid bash "$JOBD_BOOT_WS/.jobd_boot.sh" </dev/null >>"$JOBD_BOOT_WS/jobd-boot.log" 2>&1 &
else
  # weak fallback (nohup only blocks SIGHUP, not a group SIGKILL) — setsid is
  # in util-linux and present on every image we've booted
  nohup bash "$JOBD_BOOT_WS/.jobd_boot.sh" </dev/null >>"$JOBD_BOOT_WS/jobd-boot.log" 2>&1 &
fi
# handshake: don't proceed until the worker confirms it is in its OWN session
# (its first act is writing the marker, which happens post-setsid by
# construction). Without this, a group reap immediately after onstart exit
# could still kill the worker pre-setsid — the same race, one layer down.
_jbi=0
while [ ! -f "$JOBD_BOOT_WS/.jobd_boot.started" ] && [ "$_jbi" -lt 100 ]; do
  sleep 0.1; _jbi=$((_jbi+1))
done
[ -f "$JOBD_BOOT_WS/.jobd_boot.started" ] \
  || echo ">> [jobd-boot] !! worker did not confirm start within 10s (see $JOBD_BOOT_WS/jobd-boot.log)"
# --- end jobd provision-time bootstrap ----------------------------------------
