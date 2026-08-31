#!/usr/bin/env bash
# ensure_base_model.sh — guarantee a base model is present-and-COMPLETE on B2 so
# a launched Vast box NEVER fetches a base model from HuggingFace on-box.
#
# WHY: on these boxes the HF Xet CAS client (Rust) hits `tls handshake eof` /
# UnexpectedEof and then DEADLOCKS (100 threads parked on futex_wait_queue_me) —
# the process never retries or errors, it just hangs and bills for nothing (this
# wedged modelzoo-reader-01 for 45 min). Meanwhile box<->B2 is rock solid. So
# the contract is: the box only ever pulls a base from B2. This helper, run on
# the launcher machine BEFORE a box is started, makes that safe by seeding the
# model HF->B2 locally if it isn't already there, and it FAILS CLOSED otherwise.
#
# Maps an HF id -> a deterministic B2 slug under base-models/<slug>/, checks for
# a COMPLETE copy (config + tokenizer + all weight shards; index-aware; plus an
# optional .complete byte marker this script writes), and if absent/incomplete
# materialises the model locally via huggingface_hub.snapshot_download (which
# reuses the local HF cache if present, else downloads from HF) and rclone-copies
# it to B2. Idempotent + safe to re-run (rclone skips already-present files).
#
# STDOUT CONTRACT: on success prints EXACTLY the B2 subpath (base-models/<slug>)
# and nothing else. ALL human/progress logs go to STDERR. Never prints tokens.
# --print-bytes EXTENDS that contract to "<subpath>\t<bytes>" on ONE line — it
# is opt-in precisely because callers do a bare `stdout.strip()` (herdd.py's
# base gate does), which a second line or an unexpected field would break.
#
# Usage:
#   ensure_base_model.sh <hf-id> [--slug SLUG] [--check-only] [--allow-hf-seed]
#                                [--print-bytes]
#
#   # gate + seed if needed, capture the subpath the box should pull:
#   SUB=$(tools/vast/ensure_base_model.sh Qwen/Qwen2.5-Coder-7B-Instruct)
#
#   # presence check only — no seed, no upload (exit 0 present / non-0 absent):
#   tools/vast/ensure_base_model.sh Qwen/Qwen2.5-Coder-1.5B-Instruct --check-only
#
#   # also report the model's byte total, so a caller can SIZE the box's disk
#   # from the config instead of hand-typing --disk:
#   IFS=$'\t' read -r SUB BYTES < <(tools/vast/ensure_base_model.sh "$ID" --print-bytes)
#
# Exit: 0 = present-and-complete on B2 (subpath printed); non-zero = not present
# (and, in --check-only, NOT seeded). Seeding requires huggingface_hub + creds.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
log() { printf '%s\n' "$*" >&2; }
die() { printf '!! %s\n' "$*" >&2; exit 1; }

HF_ID=""; SLUG=""; CHECK_ONLY=0; ALLOW_HF_SEED=1; PRINT_BYTES=0
while [ $# -gt 0 ]; do case "$1" in
  --slug) SLUG="$2"; shift 2;;
  --check-only|--dry-run) CHECK_ONLY=1; shift;;
  --allow-hf-seed) ALLOW_HF_SEED=1; shift;;      # (default) permit HF->B2 seed
  --no-hf-seed) ALLOW_HF_SEED=0; shift;;         # gate only; never touch HF
  --print-bytes) PRINT_BYTES=1; shift;;          # stdout -> "<subpath>\t<bytes>"
  -*) die "unknown arg $1";;
  *) [ -z "$HF_ID" ] && HF_ID="$1" || die "unexpected extra arg $1"; shift;;
esac; done
[ -n "$HF_ID" ] || die "usage: ensure_base_model.sh <hf-id> [--slug S] [--check-only]"

# --- credentials + rclone remote (self-configure for standalone use) ---------
# When called from `herdd train` the env + remote are already set up; do it
# here too so the script is usable on its own. Creds via env only, never logged.
if [ -z "${B2_BUCKET:-}" ] || [ -z "${B2_KEY_ID:-}" ]; then
  ENVF="$(cd "$HERE/../.." && pwd)/.env"
  [ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
fi
: "${B2_BUCKET:?B2_BUCKET required (source ./.env)}"
: "${B2_KEY_ID:?B2 creds required (source ./.env)}"
: "${B2_APPLICATION_KEY:?B2 creds required (source ./.env)}"
RCONF="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"   # rclone reads it too
# A POISONED [b2] (reserved endpoint, e.g. the 2026-08-22 fixture clobber) still
# matches a presence grep, so probing for a stanza that EXISTS left the damage in
# place and every pull failed DNS. `doctor` answers "usable?", not "present?".
bash "$HERE/b2_sync.sh" doctor >/dev/null 2>&1 || bash "$HERE/b2_sync.sh" config >&2

# --- HF id -> canonical B2 slug ----------------------------------------------
# Hand-picked slugs for the models this repo trains on (match the user's B2
# convention: base-models/<slug>/); deterministic lowercase+'/'->'-' fallback
# for anything else. Same function is used by the seed step and the launch gate,
# so they agree by construction regardless of the exact string.
canonical_slug() {
  case "$1" in
    Qwen/Qwen2.5-Coder-7B-Instruct)   echo "qwen25-coder-7b-instruct";   return;;
    Qwen/Qwen2.5-Coder-1.5B-Instruct) echo "qwen25-coder-1.5b-instruct"; return;;
    # qwen3.5 family — the fallback (lower + '/'->'-') would yield
    # "qwen-qwen3.5-4b", but seed_bakeoff_bases.sh / models_fixup already staged
    # these under the dotless "qwen35-*" slug. Pin the hand-picked slug so the
    # launch gate and the seed step agree by construction (see reasoning-sft-4b).
    Qwen/Qwen3.5-4B)                  echo "qwen35-4b";                  return;;
    Qwen/Qwen3.5-2B)                  echo "qwen35-2b";                  return;;
    Qwen/Qwen3.5-9B)                  echo "qwen35-9b";                  return;;
    # qwen3.6 — same dotless convention. The FP8 siblings were staged by hand as
    # base-models/qwen36-27b-fp8 / -35b-a3b-fp8, so the bf16 bases have to match
    # or the launch gate looks in a slug the seed step never wrote. NOTE the FP8
    # copies are SERVING artifacts and are NOT trainable (finegrained-fp8's
    # is_trainable is hard-False and Trainer raises) — a training run must name
    # the bf16 id here, never the -FP8 one.
    Qwen/Qwen3.6-27B)                 echo "qwen36-27b";                 return;;
    Qwen/Qwen3.6-35B-A3B)             echo "qwen36-35b-a3b";             return;;
    # qwen3.8 — same dotless convention (the fallback would yield
    # "qwen-qwen3.8-27b"). Multimodal: the bf16 repo carries a vision tower and
    # image/video preprocessor configs, all covered by the allow-patterns below.
    # Its -FP8 sibling is a SERVING artifact and is not trainable — see the
    # qwen3.6 note above; a training run must name the bf16 id here.
    Qwen/Qwen3.8-27B)                 echo "qwen38-27b";                 return;;
    # Staged 2026-08-26 THROUGH this script (the qwen3.6 FP8 siblings above were
    # hand-staged, which is why they have no case here). Pinned rather than left
    # to the fallback, which would yield "qwen-qwen3.8-27b-fp8" and put the
    # launch gate in a slug the seed step never wrote. Registry entry +
    # per-file pins: tools/vast/modelkit/registry/qwen38-27b-fp8.json.
    Qwen/Qwen3.8-27B-FP8)             echo "qwen38-27b-fp8";             return;;
    # non-qwen bases — same convention as the hand-staged siblings already on
    # B2 (drop the org, drop the family-version dot, lowercase): the fallback
    # would keep the org and yield "google-gemma-4-31b" /
    # "meta-models-muse-glimmer-30b", which no seed step ever wrote.
    # gemma4-31b sits alongside the extracted gemma4-12b-text.
    google/gemma-4-31B)               echo "gemma4-31b";                 return;;
    meta-models/Muse-Glimmer-30B)     echo "muse-glimmer-30b";           return;;
  esac
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr '/' '-'
}
[ -n "$SLUG" ] || SLUG="$(canonical_slug "$HF_ID")"
SUB="base-models/$SLUG"
REMOTE="b2:$B2_BUCKET/$SUB"

# --- completeness check ------------------------------------------------------
# COMPLETE = config.json + a tokenizer file + at least one weight file, AND (if
# a *.index.json shard map exists) every referenced shard present, AND (if a
# .complete byte-marker exists) the recorded byte total matches remote size.
# The marker is belt-and-suspenders: pre-existing entries (e.g. gemma4-12b-text)
# have none and still validate on file-class presence.
check_complete() {
  local listing
  listing="$(rclone lsf --files-only -R "$REMOTE" 2>/dev/null || true)"
  [ -n "$listing" ] || { log ">> B2 $SUB: ABSENT (no objects)"; return 1; }
  _have()     { printf '%s\n' "$listing" | grep -Fxq "$1"; }
  _have_ext() { printf '%s\n' "$listing" | grep -Eq "$1"; }

  _have 'config.json' || { log ">> B2 $SUB: INCOMPLETE (no config.json)"; return 1; }
  if ! { _have 'tokenizer.json' || _have 'tokenizer.model' || _have 'tokenizer_config.json'; }; then
    log ">> B2 $SUB: INCOMPLETE (no tokenizer file)"; return 1
  fi
  local wext='(\.safetensors|\.bin|\.pt|\.pth)$'
  _have_ext "$wext" || { log ">> B2 $SUB: INCOMPLETE (no weight file)"; return 1; }

  # sharded models: verify every shard named in the index is present.
  local idx
  idx="$(printf '%s\n' "$listing" | grep -E '\.index\.json$' | head -n1 || true)"
  if [ -n "$idx" ]; then
    local shards missing
    shards="$(rclone cat "$REMOTE/$idx" 2>/dev/null \
      | python3 -c 'import json,sys;d=json.load(sys.stdin);print("\n".join(sorted(set((d.get("weight_map") or {}).values()))))' 2>/dev/null || true)"
    if [ -n "$shards" ]; then
      missing=""
      while IFS= read -r sh; do
        [ -z "$sh" ] && continue
        _have "$sh" || missing="$missing $sh"
      done <<< "$shards"
      [ -z "$missing" ] || { log ">> B2 $SUB: INCOMPLETE (index shards missing:$missing)"; return 1; }
    fi
  fi

  # optional .complete byte marker: if present, recorded bytes must match remote.
  if _have '.complete'; then
    local want got
    want="$(rclone cat "$REMOTE/.complete" 2>/dev/null \
      | python3 -c 'import json,sys;print(json.load(sys.stdin).get("bytes",""))' 2>/dev/null || true)"
    got="$(rclone size --json "$REMOTE" 2>/dev/null \
      | python3 -c 'import json,sys;print(json.load(sys.stdin).get("bytes",""))' 2>/dev/null || true)"
    # remote size includes the marker itself; allow small slack.
    if [ -n "$want" ] && [ -n "$got" ] && [ "$got" -lt "$want" ] 2>/dev/null; then
      log ">> B2 $SUB: INCOMPLETE (.complete wants ${want}B, remote has ${got}B)"; return 1
    fi
  fi
  log ">> B2 $SUB: COMPLETE"
  return 0
}

# Byte total of the model on B2. Only called under --print-bytes, so the extra
# LIST costs nothing on the default path.
#
# `rclone size` is authoritative over the `.complete` marker: the marker records
# what was uploaded at seed time and check_complete deliberately allows the
# remote to EXCEED it (the marker adds its own bytes), so the marker is a lower
# bound while size is the current truth. The marker is the fallback for the case
# where a LIST fails but the model is otherwise known-complete.
#
# Prints nothing (empty) when neither source answers — a caller sizing a disk
# must be able to tell "unmeasured" from a real number, never silently get 0.
remote_bytes() {
  local b
  b="$(rclone size --json "$REMOTE" 2>/dev/null \
    | python3 -c 'import json,sys;print(json.load(sys.stdin).get("bytes",""))' 2>/dev/null || true)"
  if [ -z "$b" ] || ! [ "$b" -gt 0 ] 2>/dev/null; then
    b="$(rclone cat "$REMOTE/.complete" 2>/dev/null \
      | python3 -c 'import json,sys;print(json.load(sys.stdin).get("bytes",""))' 2>/dev/null || true)"
  fi
  [ -n "$b" ] && [ "$b" -gt 0 ] 2>/dev/null && printf '%s' "$b" || printf ''
}

# STDOUT contract: the subpath alone, or "<subpath>\t<bytes>" under
# --print-bytes. Success only.
emit_stdout() {
  if [ "$PRINT_BYTES" -eq 1 ]; then
    printf '%s\t%s\n' "$SUB" "$(remote_bytes)"
  else
    printf '%s\n' "$SUB"
  fi
}

if check_complete; then
  emit_stdout
  exit 0
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  log ">> --check-only: $HF_ID is NOT present-and-complete on B2 ($SUB) — would seed HF->B2"
  exit 4
fi
[ "$ALLOW_HF_SEED" -eq 1 ] || die "$HF_ID absent on B2 and --no-hf-seed given (fail closed)"

# --- seed: materialise locally (HF cache or HF download), then rclone -> B2 ---
log ">> seeding $HF_ID -> $REMOTE (snapshot_download then rclone copy)"
SRC="$(HF_HUB_ENABLE_HF_TRANSFER=0 python3 - "$HF_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download
# reuses the local HF cache if the snapshot is already there (no re-download);
# else downloads from HF. We only need serving files.
p = snapshot_download(
    repo_id=sys.argv[1],
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model",
                    "tokenizer*", "merges.txt", "vocab.json", "*.txt",
                    # transformers>=5 ships the chat template as a separate
                    # chat_template.jinja — dropping it 400s every vLLM chat
                    # request (LFM2.5 bakeoff-02 failure)
                    "*.jinja",
                    # trust_remote_code archs (nemotron-nano-9b-v2) ship their
                    # modeling/configuration code as *.py in the repo — without
                    # it transformers raises "does not appear to have a file
                    # named configuration_*.py" (bakeoff-03 nemotron skip)
                    "*.py"],
)
print(p)
PY
)" || die "snapshot_download failed for $HF_ID (huggingface_hub / HF_TOKEN?)"
[ -n "$SRC" ] && [ -d "$SRC" ] || die "snapshot_download returned no dir for $HF_ID"
log ">> local snapshot: (resolved) — uploading serving files to B2"

# prefer safetensors: if present locally, don't upload legacy *.bin duplicates.
# The include list MUST stay in sync with snapshot_download's allow_patterns
# above. It drifted once and the drift was silent: the download step was fixed
# to fetch `*.jinja` and `*.py` (chat template; trust_remote_code modeling
# files) but THIS list was not, so both were downloaded locally and then never
# uploaded — the B2 copy was "complete" by check_complete's file-class test
# while missing exactly the file whose absence 400s every vLLM chat request.
# Qwen3.6-27B ships a chat_template.jinja, so this is not hypothetical.
INC=(--include '*.safetensors' --include '*.json' --include 'tokenizer*'
     --include '*.model' --include 'merges.txt' --include 'vocab.json' --include '*.txt'
     --include '*.jinja' --include '*.py')
if [ -z "$(find -L "$SRC" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
  INC+=(--include '*.bin' --include '*.pt' --include '*.pth')
fi
# --copy-links: follow the cache's blob symlinks so real weights upload.
# transfers/checkers capped modestly (be a good local citizen).
rclone copy --copy-links -v --transfers 4 --checkers 8 \
  --s3-chunk-size 64M --s3-upload-concurrency 4 \
  "${INC[@]}" "$SRC" "$REMOTE" >&2 || die "rclone copy to $REMOTE failed"

# write the .complete byte marker from the REMOTE listing (authoritative).
BYTES="$(rclone size --json "$REMOTE" 2>/dev/null \
  | python3 -c 'import json,sys;print(json.load(sys.stdin).get("bytes",0))' 2>/dev/null || echo 0)"
python3 - "$HF_ID" "$SLUG" "$BYTES" > /tmp/.ensure_marker.$$ <<'PY'
import json, sys, datetime
hf, slug, byts = sys.argv[1], sys.argv[2], int(sys.argv[3] or 0)
json.dump({"hf_id": hf, "slug": slug, "bytes": byts,
           "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
          sys.stdout)
PY
rclone rcat "$REMOTE/.complete" < /tmp/.ensure_marker.$$ >&2 || log ">> WARN: marker write failed (non-fatal)"
rm -f /tmp/.ensure_marker.$$

# re-verify before declaring success (fail closed on a bad upload).
check_complete || die "post-seed verification FAILED for $SUB (upload incomplete)"
emit_stdout
