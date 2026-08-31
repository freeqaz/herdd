#!/usr/bin/env bash
# b2_region.sh — region-aware B2 read selection (US source vs EU replica).
#
# WHY: example-runs-bucket (us-west-004) is the WRITABLE source; example-runs-mirror
# (eu-central-003) is a READ-ONLY cross-account replica. A box in/near the EU
# pays a ~3.7x RTT tax reaching us-west (measured 2026-07-11: 37ms vs 135ms ICMP;
# 125ms vs 419ms TLS setup), which hammers the latency-bound weight-verify
# preamble (rclone lsf -R / cat index / cat .complete / size) and the boot
# control plane. Reading static assets from the nearest replica removes that tax
# and is what makes cheap EU/Asia hosts bootable at US-like speed. It does NOT
# help the docker image pull (that's GitLab's CDN, not B2).
#
# SAFETY MODEL — three invariants, so this is strictly additive over the US path:
#   1. WRITES ALWAYS GO TO US SOURCE. The EU key is read-only and the minted US
#      key is US-scoped, so a write physically cannot land in EU. Callers keep
#      using `b2:$B2_BUCKET` for pushes/checkpoints/events unchanged.
#   2. EU IS CHOSEN PER-ASSET ONLY IF THE ASSET EXISTS THERE. B2 replication is
#      forward-only + async, so the replica can be empty or lagging. Every read
#      is gated by a 1-RTT existence check; a miss falls back to US. This makes
#      the system correct at any replication state and auto-adopts EU as assets
#      replicate — no flag day.
#   3. NO EU CREDS / NO EU ENDPOINT  => every function degrades to exactly the US
#      behavior that shipped before this file.
#
# Env consumed (from .env; see canonical UPPERCASE names):
#   B2_KEY_ID/B2_APPLICATION_KEY/B2_BUCKET/B2_S3_ENDPOINT/B2_REGION   US source
#   B2_KEY_ID_EU/B2_APPLICATION_KEY_EU/B2_BUCKET_EU/B2_S3_ENDPOINT_EU/B2_REGION_EU
#                                                                     EU replica
#   B2_REGION_MODE = auto|us|eu   (default auto: probe; us/eu force a region)
#
# Sourceable (functions) OR run as CLI:
#   b2_region.sh config                 # write [b2] (US) + [b2eu] (EU) rclone remotes
#   b2_region.sh probe                  # print "us"|"eu" by TLS-connect RTT (+timings on stderr)
#   b2_region.sh pick                   # print the selected read region (honors B2_REGION_MODE)
#   b2_region.sh read-remote <subpath>  # print the gated "remote:bucket" for one asset
#
# NOTE: no `set -e` at file scope — this is meant to be SOURCED; shell options
# stay the caller's. The CLI block below sets strict mode for standalone runs.

_b2r_rc() { echo "${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"; }

# This file writes [b2] too, so it needs b2_sync.sh's live-config guard — a
# guard on only one of the two writers leaves the hole open. b2_sync.sh sources
# this file, so its definition normally wins; the copy below is for a standalone
# `b2_region.sh config`. The two must stay in step; the WHY and the incident are
# at b2_sync.sh's definition, and `test_b2_conf_guard.py` drives BOTH.
if ! declare -f b2_guard_live_rclone_config >/dev/null 2>&1; then
  b2_guard_live_rclone_config() {
    local rc="$1" why="" host
    [ "$rc" = "${HOME:-/nonexistent-home}/.config/rclone/rclone.conf" ] || return 0
    host="${B2_S3_ENDPOINT:-}"; host="${host#*://}"; host="${host%%/*}"; host="${host%%:*}"
    case "$host" in
      *.invalid|*.test|*.example|*.localhost|invalid|test|example|localhost|\
      example.com|example.net|example.org|*.example.com|*.example.net|*.example.org)
        why="B2_S3_ENDPOINT host '$host' is an RFC 2606/6761 reserved name" ;;
    esac
    if [ -z "$why" ] && [ -n "${PYTEST_CURRENT_TEST:-}" ]; then
      why="running under pytest (PYTEST_CURRENT_TEST=${PYTEST_CURRENT_TEST%% *})"
    fi
    [ -n "$why" ] || return 0
    echo "b2_region: REFUSING to write the live rclone config $rc" >&2
    echo "b2_region:   reason: $why" >&2
    echo "b2_region:   That [b2] remote is shared by every session and by fleetd." >&2
    echo "b2_region:   Point RCLONE_CONFIG at a temp path (or set HOME) and re-run." >&2
    exit 3
  }
fi
_b2r_host() { echo "${1#http*://}" | sed 's#/.*##'; }   # https://h/x -> h
_b2r_have_eu() { [ -n "${B2_KEY_ID_EU:-}" ] && [ -n "${B2_APPLICATION_KEY_EU:-}" ] \
                 && [ -n "${B2_S3_ENDPOINT_EU:-}" ] && [ -n "${B2_BUCKET_EU:-}" ]; }

# Write both rclone remotes, preserving any others. [b2] = US (write+read
# fallback, unchanged name so every existing caller keeps working); [b2eu] = EU
# read-only, added only when EU creds are present.
b2_region_config() {
  : "${B2_KEY_ID:?}"; : "${B2_APPLICATION_KEY:?}"; : "${B2_S3_ENDPOINT:?}"
  local rc; rc="$(_b2r_rc)"
  b2_guard_live_rclone_config "$rc"
  mkdir -p "$(dirname "$rc")"
  local tmp; tmp="$(mktemp "${rc}.XXXXXX")"
  # drop existing [b2] and [b2eu] sections, keep the rest
  if [ -f "$rc" ]; then
    awk 'BEGIN{skip=0} /^\[/{skip=($0=="[b2]"||$0=="[b2eu]")} !skip' "$rc" > "$tmp"
  fi
  cat >> "$tmp" <<EOF
[b2]
type = s3
provider = Other
access_key_id = ${B2_KEY_ID}
secret_access_key = ${B2_APPLICATION_KEY}
endpoint = ${B2_S3_ENDPOINT}
region = ${B2_REGION:-us-west-004}
acl = private
no_check_bucket = true
EOF
  if _b2r_have_eu; then
    cat >> "$tmp" <<EOF
[b2eu]
type = s3
provider = Other
access_key_id = ${B2_KEY_ID_EU}
secret_access_key = ${B2_APPLICATION_KEY_EU}
endpoint = ${B2_S3_ENDPOINT_EU}
region = ${B2_REGION_EU:-eu-central-003}
acl = private
no_check_bucket = true
EOF
  fi
  mv "$tmp" "$rc"; chmod 600 "$rc"
  echo "b2_region: wrote $rc (b2=US$( _b2r_have_eu && echo ', b2eu=EU' ))" >&2
}

# Median TLS-appconnect time (2 RTT) to an endpoint host, in seconds. Object-
# independent: works even when the replica is empty. Prints 999 on failure.
_b2r_rtt() {
  local url="$1" s; local -a v=()
  for _ in 1 2 3; do
    s="$(curl -s -o /dev/null -w '%{time_appconnect}' --max-time 6 "$url" 2>/dev/null || echo 999)"
    [ -n "$s" ] || s=999; v+=("$s")
  done
  printf '%s\n' "${v[@]}" | sort -n | sed -n 2p
}

# Print "us" or "eu": the faster region by TLS-connect RTT. No EU => "us".
# Any probe failure biases to the reachable one; both fail => "us".
b2_region_probe() {
  _b2r_have_eu || { echo us; return; }
  local us eu
  us="$(_b2r_rtt "${B2_S3_ENDPOINT}/")"
  eu="$(_b2r_rtt "${B2_S3_ENDPOINT_EU}/")"
  echo "b2_region: probe us=${us}s eu=${eu}s" >&2
  awk -v u="$us" -v e="$eu" 'BEGIN{ if (e+0>0 && e+0<u+0) print "eu"; else print "us" }'
}

# The selected read region, honoring B2_REGION_MODE (auto|us|eu). auto => probe.
b2_region_pick() {
  case "${B2_REGION_MODE:-auto}" in
    us) echo us ;;
    eu) _b2r_have_eu && echo eu || echo us ;;
    *)  b2_region_probe ;;
  esac
}

# Static-asset prefixes eligible for EU reads. EVERYTHING ELSE (checkpoints/,
# events/, artifacts/, jobs/, runs/ — fresh/mutable run state) is forced to US:
# B2 replication is async, so an EU copy of mutable state can be stale/partial.
# Space-separated, env-overridable.
B2_EU_READ_PREFIXES="${B2_EU_READ_PREFIXES:-base-models/ train-env/}"

_b2r_prefix_ok() {
  local sub="${1#/}" p
  for p in $B2_EU_READ_PREFIXES; do case "$sub/" in "$p"*) return 0;; esac; done
  return 1
}

# Per-asset gate: echo the "remote:bucket" prefix a read of <subpath> should use.
# EU only when: subpath is a static-asset prefix AND selected region is eu AND
# <subpath> actually exists in EU; otherwise US. B2_READ_REGION (us|eu, from a
# prior pick) is honored if exported, else picked here.
b2_region_read_remote() {
  local sub="${1:-}" region="${B2_READ_REGION:-}"
  if ! _b2r_prefix_ok "$sub"; then echo "b2:${B2_BUCKET}"; return; fi
  [ -n "$region" ] || region="$(b2_region_pick)"
  if [ "$region" = eu ] && _b2r_have_eu; then
    # 1-RTT existence probe; any hit (file or dir) qualifies.
    if rclone lsf "b2eu:${B2_BUCKET_EU}/${sub#/}" 2>/dev/null | grep -q .; then
      echo "b2eu:${B2_BUCKET_EU}"; return
    fi
    echo "b2_region: '$sub' absent in EU replica -> US" >&2
  fi
  echo "b2:${B2_BUCKET}"
}

# ---- CLI --------------------------------------------------------------------
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
  set -euo pipefail
  # standalone: load .env if the canonical vars aren't already in env
  if [ -z "${B2_BUCKET:-}" ]; then
    for d in . .. ../.. ../../..; do
      [ -f "$d/.env" ] && { set -a; . "$d/.env"; set +a; break; }
    done
  fi
  cmd="${1:-}"; shift || true
  case "$cmd" in
    config)      b2_region_config ;;
    probe)       b2_region_probe ;;
    pick)        b2_region_pick ;;
    read-remote) b2_region_read_remote "${1:-}" ;;
    *) sed -n '2,40p' "$0"; exit 1 ;;
  esac
fi
