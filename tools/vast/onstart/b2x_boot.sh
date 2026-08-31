#!/usr/bin/env bash
# b2x_boot.sh — make the `b2x` transport available on a box, from any starting state.
#
# Source this and call `b2x_ensure`. It sets $B2X to an executable path and
# returns 0, or returns 1 — in which case the CALLER MUST fall back to its
# original rclone line. Every migrated call site is written that way, so a box
# that cannot obtain b2x still works exactly as it did before. That matters:
# `herdd ls` flags STALE-IMAGE precisely because parked boxes keep old images
# for a long time, and a migration that bricks them is a failure.
#
# THE DISTRIBUTION LADDER (first hit wins):
#   1. already present     — baked into the train image, or bootstrapped earlier
#                            on this box (survives a park/resume: same disk).
#                            Accepted only if it is the version this shim wants
#                            (B2X_REQUIRE_VERSION, stamped by publish.sh); a
#                            stale one is DEMOTED to a fallback, not accepted,
#                            or an image that bakes b2x would pin the fleet to
#                            bake-time forever and rung 2 could never reach it.
#   2. rclone from B2      — the primary bootstrap. This is NOT the chicken-and-
#                            egg problem it looks like: we need rclone for ONE
#                            6.5 MB fetch, not for the multi-GB path. A single
#                            stock flow moves 6.5 MB in well under a second even
#                            on a shaped host. Every existing box already has a
#                            configured [b2] remote (b2_sync.sh config runs at
#                            boot), so this works on boxes in the field TODAY,
#                            with no image rebake and no relaunch.
#   3. python3 + SigV4     — for a box with credentials but no rclone. python3
#                            is already a hard dependency of jobd/eval_sidecar/
#                            fetch_eval_env, so this adds no new requirement.
#   4. give up             — return 1; caller keeps its rclone behavior.
#
# Backblaze's NATIVE b2_authorize_account API is NOT usable here (verified
# 2026-08-01: both v2 and v3 return `bad_request: not currently supported on API
# version number N` for our application keys), which is why there is no
# curl-only rung. The bucket is private, so an unauthenticated curl is out too.
#
# Env: B2_BUCKET, B2_KEY_ID, B2_APPLICATION_KEY, B2_S3_ENDPOINT, [B2_REGION]
#      B2X_VERSION  pin a version (default: B2X_REQUIRE_VERSION below, else
#                   whatever tools/b2x/LATEST says)
#      B2X_DISABLE=1  force every caller onto its rclone fallback (kill switch)
#
# B2X_REQUIRE_VERSION is EMPTY in the repo copy and STAMPED by publish.sh into
# the copy it uploads, so the published shim and the published binary always
# travel as a matched pair. Empty = the historical accept-anything behavior.
B2X_REQUIRE_VERSION="${B2X_REQUIRE_VERSION:-}"

B2X_INSTALL_DIR="${B2X_INSTALL_DIR:-/workspace/bin}"
B2X=""

# Where this shim lives, resolved AT SOURCE TIME. cdn_pull.py is staged beside
# it by every lane that ships it (jobd bundle, eval-env companions, train.sh's
# and serve_vllm.sh's boot fetch), so this is the first place to look for it.
_B2X_SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"

_b2x_log() { echo ">> b2x_boot: $*" >&2; }

# _b2x_bounded <secs> <cmd...> — same guard rung 2 uses, hoisted so the CDN tier
# can bound its own fetches. A coreutils-less image runs unguarded, as before.
_b2x_bounded() {
  local s="$1"; shift
  if command -v timeout >/dev/null 2>&1; then timeout "$s" "$@"; else "$@"; fi
}

# --- transport tally ---------------------------------------------------------
# One append per transfer, so ABSENCE is detectable: a box whose tally has no
# `ok` line never used b2x, and before this existed that was invisible from
# off-box. b2x_tally_summary is what a caller folds into a boot_mark/heartbeat.
#
# Appends are single short printf's to an O_APPEND fd, which Linux keeps atomic
# under PIPE_BUF — jobd runs matrix arms concurrently, so that property is load
# bearing, not incidental. Every write is best-effort: a tally that cannot be
# written must never fail the transfer it is describing.
B2X_TALLY="${B2X_TALLY:-/workspace/.b2x_tally}"

_b2x_tally() {   # _b2x_tally <cdn|ok|fallback|disabled|unavailable> <bytes>
  printf '%s\t%s\t%s\n' "$1" "${2:-0}" "$(date -u +%Y%m%dT%H%M%SZ)" \
    >> "$B2X_TALLY" 2>/dev/null || true
}

# `cdn` leads because it is a TRANSPORT verdict like `ok`, not a failure count:
# a base-model pull served entirely from the edge never reaches b2x at all, so
# without its own counter it would read as a box that simply did no transfers.
b2x_tally_summary() {   # -> "cdn=C ok=N fallback=M disabled=K unavailable=J bytes=B"
  [ -f "$B2X_TALLY" ] || { echo "cdn=0 ok=0 fallback=0 disabled=0 unavailable=0 bytes=0"; return; }
  awk -F'\t' '{n[$1]++; b+=$2}
    END{printf "cdn=%d ok=%d fallback=%d disabled=%d unavailable=%d bytes=%d",
        n["cdn"], n["ok"], n["fallback"], n["disabled"], n["unavailable"], b}' \
    "$B2X_TALLY" 2>/dev/null || echo "cdn=? ok=? fallback=? disabled=? unavailable=? bytes=?"
}

# _b2x_keep_fallback <path|""> — the ladder could not install the version this
# shim wants. A STALE b2x is still enormously better than no b2x (the caller's
# alternative is a single-flow rclone), so an upgrade that cannot happen must
# never cost the box its transport. Returns 0 with $B2X set, or 1 if there was
# nothing to fall back to.
_b2x_keep_fallback() {
  [ -n "$1" ] || return 1
  B2X="$1"
  _b2x_log "upgrade unavailable — keeping present b2x $("$1" version 2>/dev/null | tr -d '[:space:]') at $1"
  return 0
}

# _b2x_verify <path> <expected-sha256|""> -> 0 if the file runs and matches
_b2x_verify() {
  local p="$1" want="$2" got
  [ -s "$p" ] || return 1
  chmod +x "$p" 2>/dev/null || true
  if [ -n "$want" ]; then
    got="$(sha256sum "$p" 2>/dev/null | awk '{print $1}')"
    [ "$got" = "$want" ] || { _b2x_log "sha256 mismatch on $p (want $want got ${got:-none})"; return 1; }
  fi
  "$p" version >/dev/null 2>&1
}

b2x_ensure() {
  if [ "${B2X_DISABLE:-0}" = "1" ]; then
    # Announce the kill switch ONCE. A silent `return 1` here is indistinguishable
    # from a box that simply never had b2x, which is the shape that hid the jobs
    # lane shipping no shim at all for its whole existence.
    [ -n "${_B2X_DISABLE_SAID:-}" ] || { _B2X_DISABLE_SAID=1
      _b2x_log "B2X_DISABLE=1 — every call site will use its rclone fallback"
      _b2x_tally disabled 0; }
    return 1
  fi

  # The version this shim wants. publish.sh STAMPS the copy it uploads to B2
  # with the version it published alongside, so shim and binary always travel
  # as a pair; the repo copy leaves it empty, which is exactly the old
  # accept-anything behavior. B2X_VERSION (an explicit operator pin) wins.
  local reqver="${B2X_VERSION:-${B2X_REQUIRE_VERSION:-}}"

  # --- rung 1: already present ------------------------------------------------
  # VERSION-AWARE, and it has to be. A present binary used to win outright,
  # which quietly made the whole ladder unreachable on any box whose IMAGE
  # bakes /usr/local/bin/b2x: train-t211-latest does, so every fresh box kept
  # the binary frozen at bake time and no publish could ever reach it. That
  # defeats the stated point of rung 2 ("no image rebake and no relaunch") and
  # made a 6.5 MB binary swap cost a full image rebake.
  #
  # Staleness only DEMOTES a candidate, it never discards it: `fallback` holds
  # the best present binary and every failure path below returns to it. A box
  # that cannot reach B2 therefore behaves exactly as it does today.
  local cand fallback=""
  for cand in "${B2X_INSTALL_DIR}/b2x" /usr/local/bin/b2x /workspace/eval/bin/b2x; do
    if _b2x_verify "$cand" ""; then
      if [ -z "$reqver" ] || [ "$("$cand" version 2>/dev/null | tr -d '[:space:]')" = "$reqver" ]; then
        B2X="$cand"; return 0
      fi
      [ -n "$fallback" ] || fallback="$cand"
    fi
  done
  if command -v b2x >/dev/null 2>&1 && b2x version >/dev/null 2>&1; then
    cand="$(command -v b2x)"
    if [ -z "$reqver" ] || [ "$("$cand" version 2>/dev/null | tr -d '[:space:]')" = "$reqver" ]; then
      B2X="$cand"; return 0
    fi
    [ -n "$fallback" ] || fallback="$cand"
  fi
  [ -z "$fallback" ] || _b2x_log "present b2x ($fallback) is not ${reqver} — trying to upgrade"

  # Credentials are required for every remaining rung.
  [ -n "${B2_BUCKET:-}" ] && [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ] || {
    _b2x_log "no B2 credentials in env — caller falls back to rclone"
    _b2x_keep_fallback "$fallback"; return $?
  }

  mkdir -p "$B2X_INSTALL_DIR" 2>/dev/null || { _b2x_keep_fallback "$fallback"; return $?; }
  local tmp="${B2X_INSTALL_DIR}/.b2x.dl.$$"
  # An exact required version is also what we FETCH, which saves reading LATEST.
  local ver="$reqver" want=""

  # --- rung 2: rclone (the primary bootstrap) --------------------------------
  # TIME-BOXED. This rung runs BEFORE the box has any transport, so a hang here
  # hangs the boot itself with no guard downstream to catch it — and it is the
  # cheapest possible thing to bound: a LATEST pointer, a sha256 line, and a
  # 6.5 MB binary. Even one stock TCP flow at the BOTTOM of the observed
  # per-flow shaping band (~1 MB/s) moves 6.5 MB in about 7 s, so
  # B2X_BOOT_TIMEOUT_S=120 is ~17x the slow case and cannot false-fire. Failing
  # this rung is cheap and correct: the ladder simply falls through to python3
  # and then to "caller keeps rclone".
  local _bt="${B2X_BOOT_TIMEOUT_S:-120}"
  command -v timeout >/dev/null 2>&1 || _bt=""    # coreutils-less image: unguarded, as before
  _b2x_to() { if [ -n "$_bt" ]; then timeout "$_bt" "$@"; else "$@"; fi; }
  if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q '^b2:'; then
    local B2="b2:${B2_BUCKET}"
    [ -n "$ver" ] || ver="$(_b2x_to rclone cat "$B2/tools/b2x/LATEST" 2>/dev/null | tr -d '[:space:]')"
    if [ -n "$ver" ]; then
      want="$(_b2x_to rclone cat "$B2/tools/b2x/b2x-${ver}-linux-amd64.sha256" 2>/dev/null | awk '{print $1}')"
      if _b2x_to rclone copyto "$B2/tools/b2x/b2x-${ver}-linux-amd64" "$tmp" 2>/dev/null && _b2x_verify "$tmp" "$want"; then
        mv -f "$tmp" "${B2X_INSTALL_DIR}/b2x" && B2X="${B2X_INSTALL_DIR}/b2x"
        _b2x_log "installed b2x ${ver} via rclone -> $B2X"
        return 0
      fi
    fi
    rm -f "$tmp"
  fi

  # --- rung 3: python3 + SigV4 -----------------------------------------------
  local PY; PY="$(command -v python3 || true)"
  if [ -n "$PY" ]; then
    if "$PY" - "$tmp" "$ver" <<'PYEOF' 2>/dev/null && _b2x_verify "$tmp" ""; then
import datetime, hashlib, hmac, os, sys, urllib.request

dest, ver = sys.argv[1], (sys.argv[2] or "")
bucket   = os.environ["B2_BUCKET"]
key_id   = os.environ["B2_KEY_ID"]
secret   = os.environ["B2_APPLICATION_KEY"]
region   = os.environ.get("B2_REGION") or "us-west-004"
endpoint = os.environ.get("B2_S3_ENDPOINT") or f"https://s3.{region}.backblazeb2.com"
host     = endpoint.split("://", 1)[1].rstrip("/")

def get(path):
    """Signed GET of one object. Mirrors tools/vast/b2x/sigv4.go."""
    now   = datetime.datetime.now(datetime.timezone.utc)
    amz   = now.strftime("%Y%m%dT%H%M%SZ")
    stamp = now.strftime("%Y%m%d")
    empty = hashlib.sha256(b"").hexdigest()
    cr = "\n".join([
        "GET", path, "",
        f"host:{host}\nx-amz-content-sha256:{empty}\nx-amz-date:{amz}\n",
        "host;x-amz-content-sha256;x-amz-date", empty,
    ])
    scope = f"{stamp}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope, hashlib.sha256(cr.encode()).hexdigest()])
    k = ("AWS4" + secret).encode()
    for part in (stamp, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(endpoint.rstrip("/") + path, headers={
        "Host": host, "x-amz-date": amz, "x-amz-content-sha256": empty,
        "Authorization": ("AWS4-HMAC-SHA256 "
                          f"Credential={key_id}/{scope}, "
                          "SignedHeaders=host;x-amz-content-sha256;x-amz-date, "
                          f"Signature={sig}"),
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

if not ver:
    ver = get(f"/{bucket}/tools/b2x/LATEST").decode().strip()
blob = get(f"/{bucket}/tools/b2x/b2x-{ver}-linux-amd64")
try:
    want = get(f"/{bucket}/tools/b2x/b2x-{ver}-linux-amd64.sha256").decode().split()[0]
    if hashlib.sha256(blob).hexdigest() != want:
        sys.exit("b2x bootstrap: sha256 mismatch")
except Exception as e:
    if "mismatch" in str(e):
        raise
with open(dest, "wb") as f:
    f.write(blob)
PYEOF
      mv -f "$tmp" "${B2X_INSTALL_DIR}/b2x" && B2X="${B2X_INSTALL_DIR}/b2x"
      _b2x_log "installed b2x via python3/sigv4 -> $B2X"
      return 0
    fi
    rm -f "$tmp"
  fi

  _b2x_keep_fallback "$fallback" && return 0
  _b2x_log "could not obtain b2x — caller falls back to rclone"
  return 1
}

# --- CDN tier: base models off the Cloudflare-fronted public mirror ----------
#
# Rung 0 of the PULL ladder, and it lives INSIDE b2x_pull so every existing call
# site gets it with no call-site edit at all (train.sh's base pull, serve_vllm's
# MODEL_B2, jobd's asset pull, the runsets' selftest base, serve_eval_fleet).
#
# It engages only when ALL of these hold, and returns 1 — silently, leaving the
# b2x -> rclone ladder below completely unchanged — whenever any of them does not:
#   * B2_CDN_HOST / B2_CDN_BUCKET / B2_CDN_PREFIX are all in the box env;
#   * the source is under `base-models/`. That prefix is the ONLY thing mirrored
#     (public upstream weights); checkpoints, artifacts and eval-env must fall
#     straight through, and the WAF refuses them anyway;
#   * cdn_pull.py and python3 are both on this box.
# Every later failure — a 404 on CDN_MANIFEST.json because the model is not
# mirrored yet, one bad chunk, any nonzero exit — logs a reason and returns 1.
# FAIL OPEN, never closed: the CDN can only ever make a pull faster, never make
# one impossible.
#
# ATOMICITY — THE ONE INVARIANT A FUTURE EDIT COULD SILENTLY BREAK.
# cdn_pull.py PREALLOCATES every destination file at full size before it fetches
# a single chunk (that is what lets N workers write N offsets at once). So a
# pull that dies half way leaves FULL-SIZE FILES FULL OF HOLES — and both
# fallbacks would then decide there is nothing left to do: b2x preallocates the
# same way, and rclone's default size-and-modtime compare sees the size match
# and SKIPS the object. The box would go on to train or serve on zero-filled
# weights with every transfer in the chain reporting success.
# That is why the CDN pull lands in a TEMP SIBLING DIRECTORY and its files are
# renamed into the real destination only after cdn_pull.py has exited 0 with
# every chunk verified against the manifest sha1. NEVER point cdn_pull.py
# straight at the caller's dst.

# _b2x_cdn_move <srcdir> <dstdir> — rename every file across; same filesystem
# (the temp dir is a sibling of dst) so each mv is a rename, not a copy.
# A failure part way leaves the destination holding only COMPLETE, sha1-verified
# files plus whatever was already there, which is precisely the state the b2x and
# rclone fallbacks are safe to resume from.
_b2x_cdn_move() {
  local s="$1" d="$2" lst f rc=0
  lst="$(mktemp 2>/dev/null)" || return 1
  ( cd "$s" && find . -type f -print ) > "$lst" 2>/dev/null || { rm -f "$lst"; return 1; }
  [ -s "$lst" ] || { rm -f "$lst"; return 1; }
  while IFS= read -r f; do
    f="${f#./}"
    [ -n "$f" ] || continue
    case "$f" in */*) mkdir -p "$d/${f%/*}" 2>/dev/null || rc=1 ;; esac
    mv -f "$s/$f" "$d/$f" 2>/dev/null || rc=1
  done < "$lst"
  rm -f "$lst"
  return $rc
}

# _b2x_cdn_script — echo a usable cdn_pull.py, or return 1.
_b2x_cdn_script() {
  local c
  for c in "${B2X_CDN_PULL:-}" \
           "${_B2X_SHIM_DIR:-}/cdn_pull.py" \
           "${_B2X_SHIM_DIR:-}/../cdn_pull.py" \
           "${JOBD_DIR:-/workspace/jobd}/cdn_pull.py" \
           "${JOBD_DIR:-/workspace/jobd}/onstart/cdn_pull.py" \
           /workspace/cdn_pull.py \
           "${B2X_INSTALL_DIR}/cdn_pull.py" \
           /workspace/eval/upstream-monorepo/tools/vast/cdn_pull.py; do
    if [ -n "$c" ] && [ -s "$c" ]; then echo "$c"; return 0; fi
  done
  # Last resort, and the same argument as the b2x ladder's rung 2: one small
  # object over the already-configured [b2] remote reaches every box in the
  # field today, with no image rebake and no relaunch. Bounded, and attempted
  # at most once per shell so an absent object costs one timeout, not one per pull.
  [ -z "${_B2X_CDN_FETCHED:-}" ] || return 1
  _B2X_CDN_FETCHED=1
  command -v rclone >/dev/null 2>&1 || return 1
  [ -n "${B2_BUCKET:-}" ] || return 1
  rclone listremotes 2>/dev/null | grep -q '^b2:' || return 1
  mkdir -p "$B2X_INSTALL_DIR" 2>/dev/null || return 1
  local dl="${B2X_INSTALL_DIR}/.cdn_pull.dl.$$"
  if _b2x_bounded 60 rclone copyto "b2:${B2_BUCKET}/eval-env/cdn_pull.py" "$dl" 2>/dev/null \
     && [ -s "$dl" ] && mv -f "$dl" "${B2X_INSTALL_DIR}/cdn_pull.py"; then
    _b2x_log "installed cdn_pull.py via rclone -> ${B2X_INSTALL_DIR}/cdn_pull.py"
    echo "${B2X_INSTALL_DIR}/cdn_pull.py"; return 0
  fi
  rm -f "$dl" 2>/dev/null
  return 1
}

# _b2x_cdn_pull <b2-src> <local-dst> [b2x args...] -> 0 only if the CDN served
# every byte and every chunk verified.
_b2x_cdn_pull() {
  local src="$1" dst="$2"; shift 2
  [ -n "$src" ] && [ -n "$dst" ] || return 1

  # --- src -> (model, only) --------------------------------------------------
  # Shapes in the field: `b2:<bucket>/base-models/<slug>` (train.sh, serve_vllm,
  # the runsets), `<bucket>/base-models/...`, a bare `base-models/...`, and any
  # of them with a trailing slash (jobd's asset pull). At most ONE component may
  # precede base-models/ — the bucket. Anything deeper is not a mirror path, and
  # an unmappable shape falls through rather than being guessed at.
  local p="${src#b2:}"; p="${p%/}"
  case "$p" in
    base-models/*) : ;;
    */base-models/*)
      p="${p#*/}"
      case "$p" in base-models/*) : ;; *) return 1 ;; esac ;;
    *) return 1 ;;
  esac
  local rel="${p#base-models/}"
  [ -n "$rel" ] || return 1
  local model="${rel%%/*}" only=""
  case "$rel" in */*) only="${rel#*/}" ;; esac
  [ -n "$model" ] || return 1

  # B2X_DISABLE is documented as "force every caller onto its rclone fallback",
  # and a CDN-served pull is not that — so the file-wide kill switch kills this
  # rung too. B2X_CDN_DISABLE turns off ONLY the CDN and leaves b2x running.
  # An explicit `if`, not `[ … ] && return 1`: the trailing-&& form leaves the
  # statement's status at 1 when the test is false, which is a `set -e` exit in
  # a sourced-into-anything shim.
  if [ "${B2X_CDN_DISABLE:-0}" = "1" ] || [ "${B2X_DISABLE:-0}" = "1" ]; then
    return 1
  fi
  if [ -z "${B2_CDN_HOST:-}" ] || [ -z "${B2_CDN_BUCKET:-}" ] || [ -z "${B2_CDN_PREFIX:-}" ]; then
    # Announce ONCE. Silence here is indistinguishable from "the CDN served it",
    # which is the shape that hid the jobs lane shipping no b2x shim at all.
    [ -n "${_B2X_CDN_UNSET_SAID:-}" ] || { _B2X_CDN_UNSET_SAID=1
      _b2x_log "cdn miss -> b2x (B2_CDN_HOST/BUCKET/PREFIX not in the box env)"; }
    return 1
  fi

  # --- extra args ------------------------------------------------------------
  # --deadline bounds the attempt (jobd's asset pull passes its own ceiling) and
  # --stats-env is answered below with the CDN's figures in the B2X_* shape the
  # caller expects. ANY other flag (--exclude, --min-age, ...) has no manifest
  # equivalent, so we fall through rather than silently drop a filter the caller
  # asked for.
  local a want="" deadline="" callers_se=""
  for a in "$@"; do
    case "$want" in
      deadline) deadline="${a%s}"; want=""; continue ;;
      stats)    callers_se="$a";   want=""; continue ;;
    esac
    case "$a" in
      --deadline)  want=deadline ;;
      --stats-env) want=stats ;;
      *) return 1 ;;
    esac
  done
  [ -z "$want" ] || return 1

  local PY; PY="$(command -v python3 2>/dev/null || true)"
  [ -n "$PY" ] || { _b2x_log "cdn miss -> b2x (no python3)"; return 1; }
  local script; script="$(_b2x_cdn_script)" || {
    _b2x_log "cdn miss -> b2x (cdn_pull.py not on this box)"; return 1; }

  mkdir -p "$(dirname "$dst")" 2>/dev/null || return 1
  local tmpd="${dst%/}.cdn_tmp.$$"
  rm -rf "$tmpd" 2>/dev/null
  mkdir -p "$tmpd" 2>/dev/null || return 1
  local se errf
  se="$(mktemp 2>/dev/null)" && errf="$(mktemp 2>/dev/null)" \
    || { rm -rf "$tmpd"; rm -f "$se" 2>/dev/null; return 1; }

  local args=(--model "$model" --dest "$tmpd" --stats-env "$se")
  [ -z "$only" ] || args+=(--only "$only")
  [ -z "${B2X_CDN_CONCURRENCY:-}" ] || args+=(--concurrency "$B2X_CDN_CONCURRENCY")

  local rc=0
  if [ -n "$deadline" ]; then
    _b2x_bounded "$deadline" "$PY" "$script" "${args[@]}" >/dev/null 2>"$errf"; rc=$?
  else
    "$PY" "$script" "${args[@]}" >/dev/null 2>"$errf"; rc=$?
  fi

  CDN_BYTES=""; CDN_SECS=""; CDN_MBPS=""; CDN_CHUNKS=""; CDN_FAILED=""
  if [ -s "$se" ]; then . "$se" 2>/dev/null || true; fi

  # An absent stats file counts as failure: cdn_pull writes it before it reports,
  # so "no stats" means it died early (a 404 manifest is exactly that shape).
  if [ "$rc" -ne 0 ] || [ "${CDN_FAILED:-1}" != "0" ]; then
    _b2x_log "cdn miss -> b2x (${model}: exit ${rc}, ${CDN_FAILED:-?} chunk failures; $(tail -n 1 "$errf" 2>/dev/null))"
    rm -rf "$tmpd" 2>/dev/null; rm -f "$se" "$errf" 2>/dev/null
    return 1
  fi

  local moved=0
  if [ -n "$only" ]; then
    # --only is a SUBSTRING match, so take the exact relative path and nothing else.
    if [ -s "$tmpd/$only" ] && mv -f "$tmpd/$only" "$dst" 2>/dev/null; then moved=1; fi
  else
    if mkdir -p "$dst" 2>/dev/null && _b2x_cdn_move "$tmpd" "$dst"; then moved=1; fi
  fi
  rm -rf "$tmpd" 2>/dev/null
  if [ "$moved" != 1 ]; then
    _b2x_log "cdn miss -> b2x (${model}: verified files would not move into ${dst})"
    rm -f "$se" "$errf" 2>/dev/null
    return 1
  fi
  rm -f "$se" "$errf" 2>/dev/null

  B2X_LAST_BYTES="${CDN_BYTES:-}"; B2X_LAST_SECS="${CDN_SECS:-}"
  B2X_LAST_MBPS="${CDN_MBPS:-}";   B2X_LAST_STREAMS="${B2X_CDN_CONCURRENCY:-}"
  B2X_LAST_VERDICT=ok;             B2X_LAST_TRANSPORT=cdn
  export B2X_LAST_BYTES B2X_LAST_SECS B2X_LAST_MBPS B2X_LAST_STREAMS \
         B2X_LAST_VERDICT B2X_LAST_TRANSPORT
  if [ -n "$callers_se" ]; then
    { printf 'B2X_BYTES=%s\n' "${CDN_BYTES:-0}"
      printf 'B2X_SECS=%s\n'  "${CDN_SECS:-0}"
      printf 'B2X_MBPS=%s\n'  "${CDN_MBPS:-0}"
      printf 'B2X_VERDICT=ok\n'
      printf 'B2X_TRANSPORT=cdn\n'; } > "$callers_se" 2>/dev/null || true
  fi
  _b2x_log "pull OK via cdn (${CDN_CHUNKS:-?} chunks, ${CDN_MBPS:-?} MB/s): ${model} -> ${dst}"
  _b2x_tally cdn "${CDN_BYTES:-0}"
  return 0
}

# --- call-site wrappers ------------------------------------------------------
#
# Every migrated site is written as ONE line:
#
#     b2x_pull "$B2/jobs/$id/checkpoints/" "$run/" || <the original rclone line>
#
# so the fallback is the pre-existing code, visible and unchanged, right there
# next to the new path. If b2x is unavailable OR its transfer fails for any
# reason, the site behaves exactly as it did before the migration. That is
# deliberately conservative: a fallback can never make a site worse than today,
# and masking a b2x bug is prevented by logging every fallback loudly (the line
# lands in onstart.log, which is pushed to B2 every 45 s).
#
# Paths are passed through verbatim, including the "b2:$B2_BUCKET/..." spelling
# the existing $B2 variables carry — b2x normalizes it — so a migrated line
# keeps using the same variable it always did.

# _b2x_run <pull|push> <src> <dst> [args...] — the shared body of both wrappers.
#
# Logs the SUCCESS path as well as the failure one. Only logging failures reads
# as "quiet = healthy", but it is equally the signature of a shim that was never
# defined, and telling those apart after the fact was impossible until now. On
# success it also exports B2X_LAST_{BYTES,SECS,MBPS,STREAMS,VERDICT} so a caller
# can put the real transport figures on its own heartbeat rather than regexing
# rclone-shaped prose back out of a stats file.
#
# B2X_LAST_STREAMS is the concurrency b2x ACTUALLY used, which is the only
# witness that a B2X_CONCURRENCY override took effect — the env var is parsed
# with a positive-int guard that silently keeps the computed default otherwise.
_b2x_run() {
  local op="$1" src="$2" dst="$3"; shift 3
  local se="" ours=0 rc=0 a prev=""
  # Reuse the caller's --stats-env when it brought one (a site that wants the
  # file at a known path keeps it); otherwise add our own and clean it up. Read
  # it either way, so the log line carries real figures in both shapes.
  for a in "$@"; do [ "$prev" = "--stats-env" ] && se="$a"; prev="$a"; done
  if [ -z "$se" ]; then
    se="$(mktemp 2>/dev/null)" && ours=1 && set -- "$@" --stats-env "$se"
  fi

  # FLAGS BEFORE POSITIONALS, always. b2x parses with Go's flag package
  # (fs.Parse(os.Args[2:])), which STOPS at the first non-flag argument — so
  # `b2x pull SRC DST --exclude X` leaves four positionals, fails the arity
  # check, prints usage and exits 2 WITHOUT ATTEMPTING THE TRANSFER. Every
  # caller then falls back to rclone and succeeds, so the site looks fine and
  # is simply never using b2x. Measured against the real binary 2026-08-25:
  # flags-after => exit 2, flags-before => the transfer runs.
  # Callers keep the natural `b2x_pull <src> <dst> [flags...]` shape; the
  # reordering happens here so no call site has to remember.
  "$B2X" "$op" "$@" "$src" "$dst"
  rc=$?

  B2X_LAST_BYTES=""; B2X_LAST_SECS=""; B2X_LAST_MBPS=""
  B2X_LAST_STREAMS=""; B2X_LAST_VERDICT=""
  # Which transport actually moved the bytes. The CDN tier sets this to `cdn`;
  # a caller putting figures on a heartbeat wants to know WHICH ladder rung they
  # came from, and MB/s alone cannot say.
  B2X_LAST_TRANSPORT=b2x; export B2X_LAST_TRANSPORT
  if [ -n "$se" ] && [ -s "$se" ]; then
    # Written by b2x itself (stats.go), KEY=VALUE, no interpolation.
    . "$se" 2>/dev/null || true
    B2X_LAST_BYTES="${B2X_BYTES:-}";     B2X_LAST_SECS="${B2X_SECS:-}"
    B2X_LAST_MBPS="${B2X_MBPS:-}";       B2X_LAST_STREAMS="${B2X_STREAMS:-}"
    B2X_LAST_VERDICT="${B2X_VERDICT:-}"
    export B2X_LAST_BYTES B2X_LAST_SECS B2X_LAST_MBPS B2X_LAST_STREAMS B2X_LAST_VERDICT
  fi
  [ "$ours" -eq 1 ] && rm -f "$se" 2>/dev/null

  if [ "$rc" -eq 0 ]; then
    _b2x_log "$op OK via b2x: ${B2X_LAST_BYTES:-?}B in ${B2X_LAST_SECS:-?}s (${B2X_LAST_MBPS:-?} MB/s, ${B2X_LAST_STREAMS:-?} streams)"
    _b2x_tally ok "${B2X_LAST_BYTES:-0}"
    return 0
  fi
  _b2x_log "$op FAILED (exit $rc) for ${src} — falling back to rclone"
  _b2x_tally fallback "${B2X_LAST_BYTES:-0}"
  return 1
}

b2x_pull() {   # b2x_pull <b2-src> <local-dst> [extra b2x args...]
  # Rung 0: the CDN mirror. base-models/ only, and fail-open — a miss logs its
  # reason and returns here, leaving everything below byte-for-byte unchanged.
  if _b2x_cdn_pull "$@"; then return 0; fi
  b2x_ensure || { _b2x_tally unavailable 0; return 1; }
  local src="$1" dst="$2"; shift 2
  _b2x_run pull "$src" "$dst" "$@"
}

b2x_push() {   # b2x_push <local-src> <b2-dst> [extra b2x args...]
  b2x_ensure || { _b2x_tally unavailable 0; return 1; }
  local src="$1" dst="$2"; shift 2
  _b2x_run push "$src" "$dst" "$@"
}
