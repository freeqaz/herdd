#!/usr/bin/env bash
# onstart/tailnet_join.sh — idempotent tailnet join for vast boxes (C3, §2.5).
#
# Downloads the STATIC tailscale tarball (pinned version, sha256-verified —
# override TS_VERSION/TS_SHA256 TOGETHER), starts tailscaled in
# userspace-networking mode with a SOCKS5 proxy on localhost:1055 (containers
# have no /dev/net/tun), then `tailscale up` with $TS_AUTHKEY. Ephemerality
# comes from the auth key being minted ephemeral+tagged workstation-side, not
# from anything here. Exits 0 fast when already joined. NEVER echoes the auth
# key (no set -x, key never interpolated into a log line).
set -u
_tlog() { echo ">> [tailnet-join] $(date -u +%FT%TZ) $*" >&2; }

TS_VERSION="${TS_VERSION:-1.98.2}"
TS_SHA256="${TS_SHA256:-85c2fdeacebebfd5afc6b6aea1b9522583d1fb5159c23fdc5bd83e98137efb1c}"
TS_STATEDIR="${TS_STATEDIR:-/workspace/.tailscale}"
TS_BIN="${TS_BIN:-$TS_STATEDIR/bin}"
TS_SOCKS_PORT="${TS_SOCKS_PORT:-1055}"
TS_SOCK=/var/run/tailscale/tailscaled.sock

_ts() {  # prefer our static install; fall back to a preinstalled tailscale
  if [ -x "$TS_BIN/tailscale" ]; then "$TS_BIN/tailscale" "$@"; else tailscale "$@"; fi
}

# fast path: a CLI exists and the node is already up -> nothing to do
if { [ -x "$TS_BIN/tailscale" ] || command -v tailscale >/dev/null 2>&1; } \
    && _ts status >/dev/null 2>&1; then
  _tlog "already joined — nothing to do"
  exit 0
fi

[ -n "${TS_AUTHKEY:-}" ] || { _tlog "!! TS_AUTHKEY not set"; exit 1; }

mkdir -p "$TS_STATEDIR" "$TS_BIN"
if [ ! -x "$TS_BIN/tailscaled" ] || [ ! -x "$TS_BIN/tailscale" ]; then
  _tgz="/tmp/tailscale_${TS_VERSION}.tgz"
  _tdir="/tmp/tailscale_${TS_VERSION}_amd64"
  _tlog "downloading static tailscale ${TS_VERSION}"
  curl -fsSL --connect-timeout 10 --max-time 120 -o "$_tgz" \
      "https://pkgs.tailscale.com/stable/tailscale_${TS_VERSION}_amd64.tgz" \
    || { _tlog "!! download failed"; exit 1; }
  if ! echo "${TS_SHA256}  ${_tgz}" | sha256sum -c - >/dev/null 2>&1; then
    _tlog "!! sha256 MISMATCH on tailscale tarball — refusing to run it"
    rm -f "$_tgz"
    exit 1
  fi
  tar -xzf "$_tgz" -C /tmp || { _tlog "!! extract failed"; rm -f "$_tgz"; exit 1; }
  cp "$_tdir/tailscale" "$_tdir/tailscaled" "$TS_BIN/" \
    || { _tlog "!! install copy failed"; exit 1; }
  chmod +x "$TS_BIN/tailscale" "$TS_BIN/tailscaled"
  rm -rf "$_tgz" "$_tdir"
fi

# tailscaled: userspace networking + SOCKS5 (no tun device in the container)
if ! pgrep -x tailscaled >/dev/null 2>&1; then
  _tlog "starting tailscaled (userspace-networking, socks5 localhost:${TS_SOCKS_PORT})"
  mkdir -p /var/run/tailscale
  nohup "$TS_BIN/tailscaled" \
      --tun=userspace-networking \
      --socks5-server="localhost:${TS_SOCKS_PORT}" \
      --statedir "$TS_STATEDIR" \
      >>"$TS_STATEDIR/tailscaled.log" 2>&1 &
fi
_i=0
while [ ! -S "$TS_SOCK" ] && [ "$_i" -lt 30 ]; do sleep 1; _i=$((_i+1)); done
[ -S "$TS_SOCK" ] || { _tlog "!! tailscaled socket never appeared (see $TS_STATEDIR/tailscaled.log)"; exit 1; }

_HOST="vast-${JOBD_IID:-${INSTANCE_ID:-${CONTAINER_ID:-box}}}"
_tlog "tailscale up (hostname ${_HOST})"
if ! _ts up --auth-key="$TS_AUTHKEY" --hostname="$_HOST" --timeout 60s >/dev/null 2>&1; then
  _tlog "!! tailscale up failed (see $TS_STATEDIR/tailscaled.log)"
  exit 1
fi
_i=0
while [ "$_i" -lt 60 ]; do
  if _ts status >/dev/null 2>&1; then _tlog "joined tailnet as ${_HOST}"; exit 0; fi
  sleep 1; _i=$((_i+1))
done
_tlog "!! node not up within 60s of tailscale up"
exit 1
