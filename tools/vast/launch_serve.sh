#!/usr/bin/env bash
# launch_serve.sh — one command to launch a vLLM OpenAI-compatible SERVE box on
# vast for a paired eval (the CPU driver is tools/pipeline/run_paired_eval.sh).
# Sibling of launch_train.sh: same .env sourcing, same comment-strip-the-onstart
# wire, same --no-hf-token env-forward, plus a SERVE_STATUS B2 marker so a wedged
# pull is diagnosable from the marker instead of a hand `ssh curl` loop.
#
# Pre-req: the base model (and any adapters / chat template) already staged on B2
# (the on-box HF Xet client deadlocks — B2 is the pin). Then:
#   launch_serve.sh --model b2:base-models/qwen3-8b --gpu 5090 --disk 60 \
#     --max-hours 4 --kv-cache-dtype fp8
#   serve_ready.sh <SERVE_ID>            # prints LLM_BASE_URL for run_paired_eval.sh
#
# THREE-PLACE ADAPTER-NAME INVARIANT (the silent-no-op guard, [[qwen36-27b-sft-plan]]):
#   LORA_SPECS keys  ==  run_paired_eval.sh MODELS tokens  ==  /v1/models ids.
#   serve_ready.sh --expect-models is the gate; a mismatch scores base weights.
#
# QUANT POLICY (ROADMAP_CONFIRM §6 amendment 2): paired evals serve fp / fp8-KV
#   ONLY. Default is NO --quantization; --kv-cache-dtype fp8 is the sanctioned
#   memory fallback. An explicit --quantization prints a stack-specific-label
#   warning (the bnb-4bit floor does not transfer to the fp instrument).
#
# Flags (flag -> serve_vllm env / herdd arg):
#   --model M            b2:<sub> -> MODEL_B2 (box pulls base from B2); else MODEL_ID=<hf id>
#   --model-artifact S   resolve the model from the modelkit REGISTRY
#                        (tools/vast/modelkit/registry/<S>.json; see
#                        MERGED_MODEL_ARTIFACTS.md). Fills MODEL_B2, SERVED_NAME,
#                        MAX_LEN, --dtype, --tp and the --gpu-ram floor, verifies
#                        the artifact PRE-SPEND on B2, and ships the on-box
#                        IDENTITY EXPECTATION so the box proves the bytes it
#                        pulled are that artifact before vLLM ever starts.
#                        Explicit flags win — EXCEPT --model, which is mutually
#                        exclusive: an expectation for artifact X against model Y
#                        can only fail, and it fails AFTER the box is paid for.
#                        An artifact with no identity pin is refused, not served
#                        ungated (--model b2:<root> is still the ungated door).
#   --dtype D            SERVE_DTYPE / vLLM --dtype. UNSET = flag not emitted and
#                        vLLM's own default stands (every serve banked to date).
#   --served-name N      SERVED_NAME (default: basename of model / b2 slug)
#   --lora "n=b2sub,..." LORA_SPECS (adapter name=b2subpath pairs)
#   --max-lora-rank R    MAX_LORA_RANK (default 32)
#   --dp N|auto          SERVE_DP: vLLM-native data-parallel engines in ONE server on
#                        :8000 (auto = GPU count / --tp). THE multi-GPU saturation
#                        path — queue-aware internal LB, no HAProxy. Rent the cards
#                        with --num-gpus to match (dp x tp GPUs).
#   --tp N               SERVE_TP: --tensor-parallel-size per engine (default 1).
#                        --dp 2 --tp 2 on a 4-GPU box = the shape for a model whose
#                        bf16 weights don't fit one card with KV cache.
#   --replicas N|auto    SERVE_REPLICAS (default 1; auto = GPU count behind HAProxy).
#                        LEGACY fallback to --dp; mutually exclusive with --dp/--tp
#   --max-len N          MAX_LEN / --max-model-len (default 16384)
#   --gpu-util F         GPU_UTIL (default 0.90; 0.95 + a 2nd job OOMs)
#   --kv-cache-dtype D   KV_CACHE_DTYPE (the doc-95 fp8-KV memory fallback)
#   --max-num-seqs N     MAX_NUM_SEQS / --max-num-seqs (decode batch width).
#                        OMITTED BY DEFAULT — vLLM's card-dependent default stands
#                        (256 <70 GiB / 1024 >=70 GiB), which is what every serve
#                        banked before 2026-08-09 ran at. Width is a MEASURED
#                        serving-path term (-3 solves on v7), so setting it on a
#                        comparand is a new anchor, not a tuning tweak.
#   --max-num-batched-tokens N  MAX_NUM_BATCHED_TOKENS / --max-num-batched-tokens
#                        (prefill token budget per step). UNLIKE width, omitting
#                        this no longer omits the flag: since 2026-08-24
#                        serve_vllm.sh emits vLLM's OWN HTTP-path resolution
#                        (2048 <70 GiB / 8192 >=70 GiB) so the value is recorded
#                        rather than merely inferable. Same behaviour, now
#                        greppable. Pass `none` to suppress the flag entirely.
#   --chat-template SUB  CHAT_TEMPLATE_B2 (.jinja for template-less bases like gemma-4)
#   --trust-remote-code  TRUST_REMOTE_CODE=1 (custom-arch bases)
#   --no-prefix-caching  SERVE_PREFIX_CACHING=0. Prefix caching is ON BY DEFAULT
#                        for throughput, but it IS a comparand term (the
#                        output-identical claim was refuted 2026-08-24). Pass
#                        this to hold a frozen comparison on the OFF cohort, or
#                        for a model that cannot support it.
#   --mtp auto|1|0       SERVE_MTP. DEFAULT `auto` since 2026-08-27 (owner
#                        directive): MTP speculative decoding is ON whenever the
#                        checkpoint ships an MTP head, INCLUDING with --lora.
#                        Measured +205% output tok/s at k=1/9/20 (Qwen3.5-9B,
#                        v14 LoRA r64 unmerged, eval-format prompts, RTX PRO
#                        6000, vLLM 0.27.1.post1+fork.gfb8e9ed57 — run of record
#                        <upstream-bench>/archive/runs/2026-08-27-v14-lora-mtp/).
#                        `auto` fails CLOSED: no head on disk => off.
#                        **PASS `--mtp 0` IF EITHER APPLIES:**
#                          (a) your requests set `min_p` or `logit_bias` — the
#                              engine REFUSES each such request (HTTP 400)
#                              under spec decode, i.e. it breaks mid-run.
#                              Preflight guard: tools/vast/serve_sampling_guard.py.
#                          (b) this serve is a term in a FROZEN comparison —
#                              output is not bitwise stable across the arms, so
#                              MTP is a cohort term
#                              (docs/plans/witness/MTP_SERVE_DEFAULT_COHORT_2026-08-27.md).
#                        The pre-2026-08-27 anchor (-2.3% at k=20 on a shared-
#                        prefix 192-token T=0.6 workload) still reproduces on
#                        ITS workload; the discriminator is ACCEPTANCE, not
#                        concurrency. Assert the ENGINE, never the flag:
#                        `speculative_config=` in the banner / `vllm:spec_decode_*`
#                        on /metrics.
#   --mtp-num-spec N     SERVE_MTP_NUM_SPEC draft depth (DEFAULT 5 since
#                        2026-08-27 — n=1 buys only ~+45% of the same workload's
#                        ~+205%). The 1-layer head is reused autoregressively,
#                        not clamped. Depth is workload-shaped: sweep it at the
#                        real concurrency before pinning something else.
#   --vllm-extra "ARGS"  VLLM_EXTRA_ARGS (whitespace-split into vllm argv on the box)
#   --quantization Q     QUANTIZATION + LOUD warning (fp/fp8-kv is the paired-eval policy)
#   --bakeoff SLUG       resolve the base row from --models-json (default the seeded roster)
#   --models-json P      bakeoff manifest (default tools/vast/runsets/base-bakeoff/models.json)
#   --max-hours N        MAX_HOURS watchdog self-destruct (default 12; 0 off)
#   --cpu-farm           opt IN to the co-tenant CPU saturate-farm (CPU_FARM=1).
#                        DEAD FEATURE, default-OFF (owner ruling 2026-08-21): its
#                        rb3-objcache grew to 69 GB and took a live serving box to
#                        110/110 GB. Only pass this if you are babysitting the disk.
#   --no-cpu-farm        accepted for back-compat; now redundant (OFF is the default)
#   --serve-id ID        SERVE_ID + vast label serve:<ID> (default minted)
#   --gpu G              gpu alias (default 5090)          --num-gpus N (default 1)
#   --gpu-ram GB         min GPU RAM (default 0 = any)
#   --disk GB            container disk (default: AUTO — measured model bytes ->
#                        disksize.serve_disk_gb; static 60 only when unmeasurable)
#   --geo CC             restrict to country (repeatable). NO DEFAULT since
#                        2026-08-05 — the old `--geo US` default was a proxy for
#                        "a host that can pull the image fast", and --inet-down
#                        below measures that directly (owner directive; see
#                        docs/plans/witness/g2_push/FLEETD_AUTOREPLACE_2026-08-05.md)
#   --inet-down Mbps     min download (default: herdd's LAUNCH_INET_DOWN_MBPS
#                        knob, 1000 — slow image pulls dominate slow boots; 0 disables)
#   --exclude-machine ID never pick this machine_id (repeatable; the boot-SLA
#                        relaunch path uses it for host rotation)
#   --sla-kills N        INTERNAL: boot-SLA kill count carried across relaunches
#                        (set by the fleetd serve watch when it re-fires this script)
#   --machine ID         restrict auto-pick to a machine_id (repeatable)
#   --host ID            restrict auto-pick to a host_id (repeatable)
#   --offer ID           pin an explicit vast offer (skips auto-pick -> triggers F-2b)
#   --cuda V             min host cuda_max_good (default 12.8; 0 disables the guard).
#                        Same floor as herdd's `--cuda`
#                        (config.LAUNCH_CUDA_MAX_GOOD): the image is unified
#                        train+serve and cu129, so both lanes rent at the
#                        CUDA-12 family floor
#   --type bid|ondemand  instance type (default bid = spot, autobid, fleetd serve
#                        watch registered post-launch. ondemand is the explicit
#                        exception for of-record paired windows only — runbook §1)
#   --budget USD         fleetd serve-watch spend cap (default: derived from the
#                        actual $/hr x MAX_HOURS x 1.6 rescue margin)
#   --price P            bid $/hr (expert escape hatch; default = herdd autobid)
#   --image I            docker image (default: the unified train+serve image on our
#                        R2-backed registry — see IMAGE= below for the live pin;
#                        REGISTRY_AUTH_SECRET mints the pull token automatically)
#   --public-port        also map :8000 on the public IP (default off — tunnel-only; key always set)
#   --api-key-file P     bearer file (default out/serve_api_key.txt; generated 0600 if absent)
#   --dry-run            print env + wire size + exact herdd argv; NO marker, NO spend
#   --wait-ready         exec serve_ready.sh <SERVE_ID> after launch
#
# FLIPPING a live box to a model merged ON it (not a flag here — the launch env is
# immutable for the instance's life, so the flip has to live somewhere a boot
# re-reads): tools/vast/serve_flip.sh writes /workspace/serve_model_override.json
# and/or the per-serve B2 object, both of which onstart/serve_vllm.sh resolves
# before any pull, on every start. MERGED_MODEL_ARTIFACTS.md §7b.
#
# ATTACH MODE (serve on an EXISTING running/parked box instead of launching one —
# for "resume the parked serve box, don't mint a new one"; audit doc 109 §4 G1/G4):
#   --on-box IID         attach: ssh-run the SAME onstart/serve_vllm.sh payload on
#                        instance IID (no `herdd launch`); relabels it serve:<ID>,
#                        writes the identical SERVE_STATUS marker, so serve_ready.sh
#                        <ID> (or --base-url) works unchanged. All --model/--lora/
#                        --served-name/... flags above are reused as-is.
#   --resume             with --on-box: `herdd start` the box first if parked/exited
#   --restart            with --on-box: pkill any existing vllm/haproxy on the box first
#   --box-env PATH       with --on-box: single-source-of-truth env file recording the
#                        live IID + ssh endpoint (default out/box.env — fixes G4's
#                        stale-IID problem)
#
# Exit codes: 0 ok · 1 abort (creds / F-1a / F-2b / attach) · 2 usage ·
#             8 --model-artifact POLICY refusal (--lora against lora_forbidden) ·
#             9 --model-artifact PRE-SPEND refusal (artifact absent/partial on B2,
#               or carrying no identity pin). Nothing is spent on 8 or 9.
set -euo pipefail
# A `set -e` death here is silent, and callers pipe this to `tail` (which returns
# ITS own 0), so a mid-attach abort read as success. Make the failure the LAST
# thing on stdout+stderr, where a tail cannot miss it. rc is set in the trap body.
# shellcheck disable=SC2154
# The frozen identity expectation is a temp file on every path that composes one
# (staged to B2 / pushed over ssh, then done with), so reap it HERE rather than
# at each of the six exits — including the refusals, which are the exits a
# hand-placed `rm` gets forgotten on.
trap 'rc=$?; rm -f "${IDENT_EXPECT_FILE:-}" 2>/dev/null; [ "$rc" -ne 0 ] && { echo "!! launch_serve.sh ABORTED rc=$rc — nothing below this line ran." >&2; echo "!! launch_serve.sh ABORTED rc=$rc"; }' EXIT
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# _LAUNCH_SERVE_ENV: dev-only override of the sourced env file (same convention
# as _LAUNCH_SERVE_ONSTART below). A test filtering os.environ still inherits
# real B2_* through this line, which silently swaps which branch it exercises.
ENV="${_LAUNCH_SERVE_ENV:-$REPO_ROOT/.env}"; [ -f "$ENV" ] && set -a && . "$ENV" && set +a
VCTL=(python3 "$HERE/herdd.py")

usage() { grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'; exit "${1:-2}"; }

# Verbatim argv snapshot for the boot-SLA relaunch spec (written post-launch):
# the fleetd serve watch re-fires exactly this invocation (minus the volatile
# --serve-id/--exclude-machine/--sla-kills/--wait-ready, re-derived per kill)
# when the box misses its come-online deadline. Owner directive 2026-08-03.
ORIG_ARGV=("$@")

# --- defaults -----------------------------------------------------------------
MODEL_ID=""; MODEL_B2=""; SERVED_NAME=""; LORA_SPECS=""; MAX_LORA_RANK=32
# --model-artifact: empty = no registry involvement and NO new requirements, so
# every `--model b2:...` caller keeps today's behaviour exactly. MODEL_SET
# records whether --model was passed at all (MODEL_B2/MODEL_ID alone cannot say,
# since the artifact resolver writes MODEL_B2 too).
MODEL_ARTIFACT=""; MODEL_SET=0; SERVE_DTYPE=""
ARTIFACT_BYTES=""; IDENT_EXPECT_FILE=""
SERVE_REPLICAS=1; SERVE_DP=1; SERVE_TP=1; DP_SET=0
MAX_LEN=""; GPU_UTIL=0.90; KV_CACHE_DTYPE=""; CHAT_TEMPLATE_B2=""
# Empty = the flag is NEVER forwarded and vLLM's own default stands. Do not give
# either of these a value here: they are of-record eval serving-path terms.
MAX_NUM_SEQS=""; MAX_NUM_BATCHED_TOKENS=""
TRUST_REMOTE_CODE=""; VLLM_EXTRA_ARGS=""; QUANTIZATION=""; MAX_HOURS=12
# Empty = "caller said nothing" for all three; the ON defaults for prefix caching
# and (since 2026-08-27) MTP live in serve_vllm.sh so the attach path and the
# launch path cannot disagree. Do NOT give SERVE_MTP a value here: `--mtp 0`
# must stay distinguishable from "unset", and a literal here would be a second
# place the default lives.
SERVE_PREFIX_CACHING=""; SERVE_MTP=""; SERVE_MTP_NUM_SPEC=""
BAKEOFF=""; MODELS_JSON="$HERE/runsets/base-bakeoff/models.json"
CPU_FARM_ON=0   # dead feature, opt-IN only (owner ruling 2026-08-21)
# DISK="" = auto-size from measured model bytes (disksize.serve_disk_gb; falls
# back to vastconf.DISK_DEFAULT_SERVE_GB=60 when the model is unmeasurable).
# TYPE=bid = spot, autobid, fleetd-managed (owner ruling 2026-08-02: the whole
# pipeline is spot + fleetd; on-demand is the explicit exception, not a default).
# INET_DOWN="" = defer to herdd's LAUNCH_INET_DOWN_MBPS knob (default 1000,
# owner directive 2026-08-03 — the 39-min pull on an 805 Mb/s host — relaxed
# from 2000 the same day). GEO=() = GLOBAL (owner directive 2026-08-05): the
# bandwidth gate replaced the US pin, and the 600s boot SLA + pull watchdog stay
# the backstop for a fast-advertised host that turns out slow.
SERVE_ID=""; GPU=5090; NGPU=1; GPU_RAM=0; DISK=""; GEO=(); INET_DOWN=""
MACHINES=(); HOSTS=(); OFFER=""; CUDA_MIN=12.8; TYPE=bid; PRICE=""; BUDGET=""
EXCLUDES=(); SLA_KILLS=0
# Default serve image: the UNIFIED train+serve lane (2026-08-01 promote, was
# serve210v-20260712; flipped t211 -> t212 on 2026-08-08, t212 -> t213 on
# 2026-08-16). torch 2.13.0+cu129 +
# OUR vLLM fork baked in the base — the fork commit and version of record are in
# ONE place, tools/vast/train-env/VLLM_PIN, and are deliberately NOT restated
# here: three different commits were being called "our fork" across live docs
# until 2026-08-10 and this line carried one of the stale ones. It serves any
# model stock vllm can PLUS the extracted
# gemma4-unified base — and it is the SAME image `herdd train` launches, so a box
# can train and then serve with no container swap (validated 2026-08-01: real serve
# of gemma4-12b-text answering completions, train venv importing under CUDA while
# vLLM held the GPU). Override with --image. There is NO rollback tag: serve210v
# was deleted from the registry 2026-08-02 and e2-paired was re-pointed to t211
# the same day, so it is no longer a frozen paired pin either. Rollback is by
# REBUILD -- Dockerfile.serve210 + `bake.sh serve210` with an explicit SERVE_BASE.
# Must equal herdd.yaml default_image — test_rehearse.py pins every copy.
IMAGE="registry.example.com/train:latest"; PUBLIC_PORT=""; API_KEY_FILE="$REPO_ROOT/out/serve_api_key.txt"
DRY_RUN=0; WAIT_READY=0
ON_BOX=""; RESUME=0; RESTART=0; BOX_ENV="$REPO_ROOT/out/box.env"

# --- arg loop (launch_train.sh:98-134 style) ----------------------------------
while [ $# -gt 0 ]; do case "$1" in
  --model) case "$2" in b2:*) MODEL_B2="${2#b2:}";; *) MODEL_ID="$2";; esac; MODEL_SET=1; shift 2;;
  --model-artifact) MODEL_ARTIFACT="$2"; shift 2;;
  --dtype) SERVE_DTYPE="$2"; shift 2;;
  --served-name) SERVED_NAME="$2"; shift 2;;
  --lora) LORA_SPECS="$2"; shift 2;;
  --max-lora-rank) MAX_LORA_RANK="$2"; shift 2;;
  --replicas) SERVE_REPLICAS="$2"; shift 2;;
  --dp) SERVE_DP="$2"; DP_SET=1; shift 2;;
  --tp) SERVE_TP="$2"; shift 2;;
  --max-len) MAX_LEN="$2"; shift 2;;
  --gpu-util) GPU_UTIL="$2"; shift 2;;
  --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2;;
  --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2;;
  --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2;;
  --chat-template) CHAT_TEMPLATE_B2="$2"; shift 2;;
  --trust-remote-code) TRUST_REMOTE_CODE=1; shift;;
  --no-prefix-caching) SERVE_PREFIX_CACHING=0; shift;;
  --mtp) SERVE_MTP="$2"; shift 2;;
  --mtp-num-spec) SERVE_MTP_NUM_SPEC="$2"; shift 2;;
  --vllm-extra) VLLM_EXTRA_ARGS="$2"; shift 2;;
  --quantization) QUANTIZATION="$2"; shift 2;;
  --bakeoff) BAKEOFF="$2"; shift 2;;
  --models-json) MODELS_JSON="$2"; shift 2;;
  --max-hours) MAX_HOURS="$2"; shift 2;;
  --cpu-farm) CPU_FARM_ON=1; shift;;
  --no-cpu-farm) CPU_FARM_ON=0; shift;;   # back-compat no-op: OFF is the default
  --serve-id) SERVE_ID="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --num-gpus) NGPU="$2"; shift 2;;
  --gpu-ram) GPU_RAM="$2"; shift 2;;
  --disk) DISK="$2"; shift 2;;
  --geo) GEO+=("$2"); shift 2;;
  --inet-down) INET_DOWN="$2"; shift 2;;
  --exclude-machine) EXCLUDES+=("$2"); shift 2;;
  --sla-kills) SLA_KILLS="$2"; shift 2;;
  --machine) MACHINES+=("$2"); shift 2;;
  --host) HOSTS+=("$2"); shift 2;;
  --offer) OFFER="$2"; shift 2;;
  --cuda) CUDA_MIN="$2"; shift 2;;
  --type) TYPE="$2"; shift 2;;
  --budget) BUDGET="$2"; shift 2;;
  --price) PRICE="$2"; shift 2;;
  --image) IMAGE="$2"; shift 2;;
  --public-port) PUBLIC_PORT=1; shift;;
  --api-key-file) API_KEY_FILE="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  --wait-ready) WAIT_READY=1; shift;;
  --on-box) ON_BOX="$2"; shift 2;;
  --resume) RESUME=1; shift;;
  --restart) RESTART=1; shift;;
  --box-env) BOX_ENV="$2"; shift 2;;
  -h|--help) usage 0;;
  *) echo "!! unknown arg $1" >&2; usage 2;;
esac; done

# --- bakeoff row: fill the base fields from the manifest (explicit flags win) --
# Maps the slug row: b2_subpath->MODEL_B2, served_name->SERVED_NAME,
# max_model_len->MAX_LEN, quant->QUANTIZATION (bf16->unset), trust_remote_code,
# special_flags->VLLM_EXTRA_ARGS. Does NOT read gpu_mem_util/port/gpu (those are
# the multi-model-one-box harness fields); GPU_UTIL stays single-tenant 0.90.
resolve_bakeoff_row() {
  local slug="$1" json="$2"
  [ -f "$json" ] || { echo "!! --bakeoff: models-json '$json' not found — pass base flags manually (--model b2:... --served-name ... --max-len ...)" >&2; exit 2; }
  python3 - "$slug" "$json" <<'PY'
import json, shlex, sys
slug, path = sys.argv[1], sys.argv[2]
doc = json.load(open(path))
rows = [r for r in doc.get("models", []) if r.get("slug") == slug]
if not rows:
    have = ",".join(r.get("slug", "?") for r in doc.get("models", []))
    sys.exit(f"!! --bakeoff slug '{slug}' not in {path} (have: {have})")
r = rows[0]
def emit(k, v): print(f"{k}={shlex.quote(str(v))}")
emit("BK_MODEL_B2", r["b2_subpath"])
emit("BK_SERVED_NAME", r.get("served_name", slug))
emit("BK_MAX_LEN", r.get("max_model_len", 16384))
q = r.get("quant", "bf16")
emit("BK_QUANT", "" if q in ("bf16", "", None) else q)       # bf16 => no --quantization
emit("BK_TRC", "1" if r.get("trust_remote_code") else "")
sf = r.get("special_flags", [])
emit("BK_VLLM_EXTRA", " ".join(sf) if isinstance(sf, list) else str(sf))
PY
}
if [ -n "$BAKEOFF" ]; then
  _bk="$(resolve_bakeoff_row "$BAKEOFF" "$MODELS_JSON")"; eval "$_bk"
  [ -z "$MODEL_B2" ]      && MODEL_B2="${BK_MODEL_B2:-}"
  [ -z "$SERVED_NAME" ]   && SERVED_NAME="${BK_SERVED_NAME:-}"
  [ -z "$MAX_LEN" ]       && MAX_LEN="${BK_MAX_LEN:-}"
  [ -z "$QUANTIZATION" ]  && QUANTIZATION="${BK_QUANT:-}"
  [ -z "$TRUST_REMOTE_CODE" ] && TRUST_REMOTE_CODE="${BK_TRC:-}"
  [ -z "$VLLM_EXTRA_ARGS" ]   && VLLM_EXTRA_ARGS="${BK_VLLM_EXTRA:-}"
fi

# --- registry artifact: fill the model fields from the COMMITTED registry -----
# Same precedence rule as the bakeoff row above (an explicit flag already set
# wins), with ONE exception that is not stylistic: `--model` is mutually
# exclusive with `--model-artifact`. Letting `--model` win there would ship an
# identity expectation for artifact X alongside a pull of model Y — which either
# fails the on-box gate AFTER the box is paid for, or (if the gate is ever
# skipped) serves Y under X's name, which is the exact 2026-08-21 failure this
# flag exists to close.
if [ -n "$MODEL_ARTIFACT" ]; then
  [ -z "$BAKEOFF" ] || { echo "!! --model-artifact and --bakeoff are two model resolvers for one serve — pick one" >&2; exit 2; }
  if [ "$MODEL_SET" = "1" ]; then
    echo "!! --model-artifact '$MODEL_ARTIFACT' with an explicit --model: refusing." >&2
    echo "!!   Every other flag wins over the registry; this one cannot. The identity" >&2
    echo "!!   expectation shipped to the box names the ARTIFACT, so a --model pointing" >&2
    echo "!!   anywhere else is a gate failure paid for at rented-box prices." >&2
    exit 2
  fi
  _ar="$(python3 "$HERE/serve_artifact.py" resolve "$MODEL_ARTIFACT")" || exit 2
  eval "$_ar"
  MODEL_B2="$AR_MODEL_B2"
  [ -z "$SERVED_NAME" ]  && SERVED_NAME="$AR_SERVED_NAME"
  [ -z "$MAX_LEN" ]      && MAX_LEN="${AR_MAX_LEN:-}"
  [ -z "$SERVE_DTYPE" ]  && SERVE_DTYPE="${AR_DTYPE:-}"
  { [ "$SERVE_TP" = "1" ] && [ -n "${AR_TP:-}" ]; } && SERVE_TP="$AR_TP"
  # min_vram_gb is a FLOOR on host selection, not a serving knob: a 27B bf16
  # merged model rented onto a 24 GB card fails at engine init, after the pull.
  case "$GPU_RAM" in 0|0.0|"") [ -n "${AR_MIN_VRAM_GB:-}" ] && GPU_RAM="$AR_MIN_VRAM_GB";; esac
  # Mounting an adapter over a MERGED dir applies it a SECOND time — the wrong
  # weights served under the right name, with every readiness gate green. Hard
  # refusal because nothing downstream can observe the double-apply.
  if [ -n "$LORA_SPECS" ] && [ -n "${AR_LORA_FORBIDDEN:-}" ]; then
    echo "!! --model-artifact '$MODEL_ARTIFACT' declares lora_forbidden, and --lora was passed: refusing." >&2
    echo "!!   The adapter is ALREADY MERGED into these weights. Mounting it again applies it" >&2
    echo "!!   twice: the server boots, /v1/models lists what you asked for, and the eval" >&2
    echo "!!   scores a model nobody trained. Drop --lora, or serve the base with" >&2
    echo "!!   --model b2:<base> and --lora." >&2
    exit 8
  fi
  echo ">> artifact '$MODEL_ARTIFACT' (kind=$AR_KIND, grade-$AR_GRADE identity): model=b2:$MODEL_B2 served=$SERVED_NAME" >&2
fi

# --- post-resolution defaults + validation ------------------------------------
[ -z "$MAX_LEN" ] && MAX_LEN=16384
if [ -z "$MODEL_ID" ] && [ -z "$MODEL_B2" ]; then
  echo "!! --model (or --bakeoff) required" >&2; usage 2
fi
# one multi-GPU mode at a time (mirrors the serve_vllm.sh on-box guard, but fails
# here pre-spend instead of on the billed box)
if [ "$SERVE_REPLICAS" != "1" ] && { [ "$SERVE_DP" != "1" ] || [ "$SERVE_TP" != "1" ]; }; then
  echo "!! --replicas is mutually exclusive with --dp/--tp — pick native DP (preferred) or HAProxy replicas" >&2; exit 2
fi
# no explicit multi-GPU shape: the box-side default is now dp=auto (saturate) —
# say so, and how to opt out. Explicit --dp 1 is forwarded and wins.
if [ "$NGPU" -gt 1 ] && [ "$DP_SET" = "0" ] && [ "$SERVE_REPLICAS" = "1" ] && [ "$SERVE_TP" = "1" ]; then
  echo ">> note: --num-gpus $NGPU with no --dp/--tp — box defaults to dp=auto ($NGPU engines, one endpoint); pass --dp 1 for a single engine" >&2
fi
# SERVED_NAME must be forwarded for the MODEL_B2 path: serve_vllm sets
# MODEL_ID=/workspace/base-model, so its basename fallback would be "base-model".
if [ -z "$SERVED_NAME" ]; then
  if [ -n "$MODEL_B2" ]; then SERVED_NAME="$(basename "$MODEL_B2")"
  else SERVED_NAME="$(basename "$MODEL_ID")"; fi
fi

# --- MTP: say the RESOLVED state out loud, before anything is spent -----------
# The checkpoint is still on B2 here, so this wrapper cannot detect the head —
# `auto` is settled on the box against the pulled bytes, and the ONLY authority
# on what actually ran is the engine (`speculative_config=` in the banner /
# `vllm:spec_decode_*` on /metrics). Printing intent is still worth its lines:
# the default flipped ON on 2026-08-27 and a caller who does not know that will
# not think to opt out of a cohort term.
case "${SERVE_MTP:-auto}" in
  0|off)
    echo ">> MTP: OFF (--mtp 0) — min_p/logit_bias honoured; this serve is on the pre-2026-08-27 OFF cohort." >&2 ;;
  1|on)
    echo ">> MTP: FORCED ON, n=${SERVE_MTP_NUM_SPEC:-5} — emitted even if no MTP head is detected on the box." >&2 ;;
  *)
    echo ">> MTP: auto (DEFAULT since 2026-08-27) — ON on the box iff the checkpoint ships an MTP head, n=${SERVE_MTP_NUM_SPEC:-5}${LORA_SPECS:+, LoRA notwithstanding}." >&2
    echo ">>   Pass --mtp 0 if this serve sets min_p/logit_bias (requests carrying them are REFUSED under spec decode)" >&2
    echo ">>   or is a term in a FROZEN comparison (output is not bitwise stable — cohort term)." >&2 ;;
esac

# --- mint + validate SERVE_ID (mirrors launch_train.sh:148 RUN_ID regex) ------
mint_serve_id() { printf 'serve-%s-%s' "$(date -u +%y%m%d-%H%M)" "$(openssl rand -hex 2)"; }
[ -z "$SERVE_ID" ] && SERVE_ID="$(mint_serve_id)"
if ! printf '%s' "$SERVE_ID" | grep -Eq '^[A-Za-z0-9._-]{1,64}$'; then
  echo "!! invalid --serve-id '$SERVE_ID': must match ^[A-Za-z0-9._-]{1,64}\$ (letters/digits/._- , 1-64 chars)" >&2
  exit 2
fi

# --- B2 requirement / marker gate ---------------------------------------------
# B2 creds are HARD-required when the box must pull from B2 (base/adapter/template);
# otherwise a SERVE_STATUS marker still needs them, so degrade to a log-only serve
# (no marker) with a warning rather than aborting a pure-HF-id serve.
MARKER=0
if [ -n "$MODEL_B2" ] || [ -n "$LORA_SPECS" ] || [ -n "$CHAT_TEMPLATE_B2" ]; then
  : "${B2_BUCKET:?MODEL_B2/LORA_SPECS/CHAT_TEMPLATE_B2 set but B2_BUCKET missing}"
  : "${B2_KEY_ID:?B2_KEY_ID missing}"; : "${B2_APPLICATION_KEY:?B2_APPLICATION_KEY missing}"
  : "${B2_S3_ENDPOINT:?B2_S3_ENDPOINT missing}"
  MARKER=1
elif [ -n "${B2_BUCKET:-}" ] && [ -n "${B2_KEY_ID:-}" ] \
  && [ -n "${B2_APPLICATION_KEY:-}" ] && [ -n "${B2_S3_ENDPOINT:-}" ]; then
  # all four needed by b2_sync.sh config + rclone rcat below (mirror serve_vllm's precheck)
  MARKER=1
else
  echo ">> note: no B2 creds — SERVE_STATUS marker DISABLED (log-only serve; serve_ready.sh --base-url still works)" >&2
fi

# --- --model-artifact PRE-SPEND gate + frozen identity expectation ------------
# Fail-closed before renting or attaching, in the shape of the base-model gate
# (vastlib/cli/train.py) — the same argument, one lane over: a box that boots to
# discover its model is half-published has already been paid for.
#
# TWO ARTIFACTS, TWO PURPOSES, and conflating them is how a gate stops gating:
#   * the B2 check protects MONEY. It reads a REMOTE listing and can say nothing
#     about what the box will actually receive.
#   * the expectation protects IDENTITY. It is composed from the COMMITTED
#     registry, never from the guards published beside the weights — otherwise
#     B2 corroborates B2 and a renamed or re-published prefix agrees with itself.
# It runs under --dry-run too: a pre-spend gate that only arms when you are
# spending cannot be exercised without spending.
if [ -n "$MODEL_ARTIFACT" ]; then
  [ "$MARKER" -eq 1 ] || { echo "!! --model-artifact needs B2 creds (the artifact lives on B2 and the gate reads it) — source tools/vast/.env" >&2; exit 9; }
  # Freeze the expectation FIRST: an artifact with no identity pin is refused
  # here, before a listing is even fetched, because there would be nothing for
  # the box to check itself against.
  IDENT_EXPECT_FILE="${TMPDIR:-/tmp}/serve.identity.$$.json"
  python3 "$HERE/serve_artifact.py" expect "$MODEL_ARTIFACT" --out "$IDENT_EXPECT_FILE" \
    || { rm -f "$IDENT_EXPECT_FILE"; exit 9; }
  bash "$HERE/b2_sync.sh" config >/dev/null 2>&1 || true
  ARTIFACT_BYTES="$(python3 "$HERE/serve_artifact.py" gate "$MODEL_ARTIFACT" --bucket "$B2_BUCKET")" \
    || { rm -f "$IDENT_EXPECT_FILE"; exit 9; }
  echo ">> artifact-gate: '$MODEL_ARTIFACT' PASSED pre-spend (identity expectation frozen: grade-$AR_GRADE, ${AR_N_FILES:-?} files)" >&2
  [ -n "${AR_CONTENT_SHA:-}" ] || echo ">> artifact-gate: NOTE grade-B pin is NULL for '$MODEL_ARTIFACT' (UNMEASURED, not clean) — the on-box gate will check the file set and sizes, NOT the bytes. Mint it with modelkit/gate_dir.py --emit on a box that already holds a verified copy (\$0)." >&2
fi

# --- the on-box identity payload: what gets staged, in ONE place --------------
# `<src>|<name-on-box>` per line. The gate code CANNOT ride the onstart wire
# (16 KiB cap; merged_fingerprint.py alone is ~11.7 KiB), so it travels the same
# per-SERVE prefix as parse_vllm_mem.py — b2:<bucket>/serve/<SERVE_ID>/<name> on
# the launch path, an ssh push to /workspace on the attach path. serve_vllm.sh
# resolves either.
#
# dirhash.py rides ONLY when a grade-B pin exists. Staging it unconditionally
# would put a tool on the box for a check no expectation asks for; leaving it
# out when grade B IS pinned makes the gate refuse (cannot check), which is the
# correct direction for both.
ident_assets() {
  [ -n "$IDENT_EXPECT_FILE" ] || return 0
  echo "$IDENT_EXPECT_FILE|identity_expect.json"
  echo "$HERE/serve_identity_gate.py|serve_identity_gate.py"
  echo "$HERE/modelkit/merged_fingerprint.py|merged_fingerprint.py"
  [ -n "${AR_CONTENT_SHA:-}" ] && echo "$HERE/modelkit/dirhash.py|dirhash.py"
  return 0
}

# --- quant policy warning (ROADMAP_CONFIRM §6 amendment 2) --------------------
if [ -n "$QUANTIZATION" ]; then
  echo "!! WARNING: --quantization=$QUANTIZATION — paired-eval policy (ROADMAP_CONFIRM §6 amendment 2) serves fp / fp8-KV ONLY." >&2
  echo "!!   A bnb-4bit floor does NOT transfer to the fp-vLLM instrument. Legitimate only for the bakeoff bnb arm," >&2
  echo "!!   and then the readout MUST be labeled stack-specific. Proceeding." >&2
fi

# --- api key: generate-or-read (must be the SAME file run_paired_eval reads) ---
ensure_api_key() {
  mkdir -p "$(dirname "$API_KEY_FILE")"
  if [ -s "$API_KEY_FILE" ]; then
    API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"
  else
    API_KEY="$(openssl rand -hex 24)"
    printf '%s\n' "$API_KEY" > "$API_KEY_FILE"; chmod 600 "$API_KEY_FILE"
    echo ">> minted serve api key -> $API_KEY_FILE (0600)"
  fi
}
ensure_api_key

# --- F-2b: pinned-offer CUDA guard (wrapper-side; upstream into herdd later) -
# --offer short-circuits herdd's auto-pick, so its search-side --cuda filter
# NEVER runs (herdd.py cmd_launch) — the exact Error-804 lottery that burned two
# windows. Look the offer up wire-side and hard-fail if its host cuda_max_good is
# below CUDA_MIN. (--machine/--host stay on the auto-pick path, so herdd's own
# --cuda filter covers them — see the search-arg assembly below.) Upstream into
# herdd cmd_launch once the peer-edit window on herdd.py closes — see
# EVALS_RUNBOOK "Deferred to herdd".
cuda_enabled() { [ "$(printf '%s' "$CUDA_MIN" | tr -d '0.')" != "" ]; }
pinned_offer_cuda_check() {
  local offer="$1" cmin="$2"
  python3 - "$offer" "$cmin" "$HERE" <<'PY'
import sys
offer_id, cuda_min, here = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
sys.path.insert(0, here)
import herdd  # READ-ONLY import of the (peer-dirty) module; no state mutated
data = herdd.request("POST", "v0/bundles/", {"id": {"eq": offer_id}, "limit": 1})
offers = data.get("offers", []) if isinstance(data, dict) else (data or [])
if not offers:
    sys.exit(f"!! F-2b: offer {offer_id} not found in v0/bundles — refusing (pass --cuda 0 to bypass the CUDA guard)")
cg = offers[0].get("cuda_max_good")
if cg is None:
    sys.exit(f"!! F-2b: offer {offer_id} has no cuda_max_good — refusing (--cuda 0 to bypass)")
if float(cg) < cuda_min:
    sys.exit(f"!! F-2b: offer {offer_id} cuda_max_good={cg} < required {cuda_min} — Error-804 risk, refusing (--cuda 0 to bypass)")
print(f">> F-2b: offer {offer_id} cuda_max_good={cg} >= {cuda_min} OK")
PY
}

# --- env assembly (forwarded into the container) ------------------------------
# Boxes get an ephemeral no-delete key minted per serve — else the standing
# B2_BOX_* / ops single key (docs/plans/keyless-b2-ingest.md) under the
# unchanged on-box names; the full-capability ops pair never leaves the
# workstation. The mint SHAPE depends on the CPU farm (DEFAULT OFF since the
# 2026-08-21 ruling), and the default is now the TIGHTER of the two:
#   farm OFF (default): bucket-wide-read + serve/-scoped-write PAIR
#     (docs/plans/cred-broker-buildout.md §2.7) — every writer on the box
#     (SERVE_STATUS + METRICS) lives under serve/, so the scope is exact.
#   farm ON (--cpu-farm only): single bucket-wide read+write key (the pre-pair
#     shape) — the eval sidecar writes evals/<RUN_ID>/ through [b2]
#     (onstart/eval_sidecar.sh), a B2 key carries ONE namePrefix, and serve/
#     vs evals/ share no parent — a serve/-scoped pair would silently strand
#     all farm output behind the sidecar's '|| true' writes. Widening the key
#     is now part of the price of opting into the farm, never a default.
# Serve keys get a long TTL (default 168h) because serve boxes park/resume for
# days; re-running --on-box rotates a fresh key onto a stale box (serve_vllm.sh
# rewrites the [b2]/[b2w] remotes from env on every run).
envs=(--env "SERVE_ID=$SERVE_ID" --env "VLLM_API_KEY=$API_KEY" --env "MAX_HOURS=$MAX_HOURS")
if [ "$MARKER" -eq 1 ]; then
  SHIP_B2_KEY_ID=""; SHIP_B2_APPLICATION_KEY=""
  SHIP_B2_WRITE_KEY_ID=""; SHIP_B2_WRITE_APPLICATION_KEY=""; SHIP_B2_EXPIRES_AT=""
  SERVE_KEY_HOURS="${B2_EPHEMERAL_HOURS:-168}"
  if [ "$DRY_RUN" -eq 0 ] && [ -n "${B2_MINTER_KEY_ID:-}" ] && [ -n "${B2_MINTER_APPLICATION_KEY:-}" ]; then
    if [ "$CPU_FARM_ON" = "1" ]; then
      # farm ON (--cpu-farm): the sidecar writes evals/ via [b2] — see header above
      mint_cmd=(python3 "$HERE/b2_mint_key.py" mint --name "serve-$SERVE_ID"
                --hours "$SERVE_KEY_HOURS")
    else
      # farm OFF (default): all on-box writers live under serve/ — pair is exact
      mint_cmd=(python3 "$HERE/b2_mint_key.py" mint-pair --name "serve-$SERVE_ID"
                --write-prefix serve/ --hours "$SERVE_KEY_HOURS")
    fi
    # PARSE (never eval) the export lines: eval would clobber this shell's own
    # ops B2_KEY_ID/B2_APPLICATION_KEY, which the fallback ladder below reads.
    if mint_out=$("${mint_cmd[@]}"); then
      SHIP_B2_KEY_ID=$(sed -n 's/^export B2_KEY_ID=//p' <<<"$mint_out")
      SHIP_B2_APPLICATION_KEY=$(sed -n 's/^export B2_APPLICATION_KEY=//p' <<<"$mint_out")
      # the single-key `mint` prints no B2_WRITE_* lines — these stay empty there
      SHIP_B2_WRITE_KEY_ID=$(sed -n 's/^export B2_WRITE_KEY_ID=//p' <<<"$mint_out")
      SHIP_B2_WRITE_APPLICATION_KEY=$(sed -n 's/^export B2_WRITE_APPLICATION_KEY=//p' <<<"$mint_out")
      SHIP_B2_EXPIRES_AT=$(python3 -c "import time; print(int(time.time() + float('$SERVE_KEY_HOURS') * 3600))")
    else
      echo "!! WARNING: ephemeral B2 key mint failed — falling back to the standing box key" >&2
    fi
  fi
  if [ -z "$SHIP_B2_KEY_ID" ]; then
    # single-key fallback: today's exact env set — no B2_WRITE_*, no expiry
    SHIP_B2_WRITE_KEY_ID=""; SHIP_B2_WRITE_APPLICATION_KEY=""; SHIP_B2_EXPIRES_AT=""
    if [ -n "${B2_BOX_KEY_ID:-}" ] && [ -n "${B2_BOX_APPLICATION_KEY:-}" ]; then
      SHIP_B2_KEY_ID="$B2_BOX_KEY_ID"; SHIP_B2_APPLICATION_KEY="$B2_BOX_APPLICATION_KEY"
    else
      echo "!! WARNING: B2_BOX_KEY_ID/B2_BOX_APPLICATION_KEY unset — shipping the full-capability ops key to a rented box (see docs/plans/keyless-b2-ingest.md)" >&2
      SHIP_B2_KEY_ID="$B2_KEY_ID"; SHIP_B2_APPLICATION_KEY="$B2_APPLICATION_KEY"
    fi
  fi
  envs+=(--env "B2_KEY_ID=$SHIP_B2_KEY_ID" --env "B2_APPLICATION_KEY=$SHIP_B2_APPLICATION_KEY"
         --env "B2_BUCKET=$B2_BUCKET" --env "B2_S3_ENDPOINT=$B2_S3_ENDPOINT"
         --env "B2_REGION=${B2_REGION:-us-west-004}")
  if [ -n "$SHIP_B2_EXPIRES_AT" ]; then
    # any minted key (pair OR single): expiry + role feed the cred-broker
    # refresh lane (§2.1 wire names).
    envs+=(--env "B2_KEY_EXPIRES_AT=$SHIP_B2_EXPIRES_AT" --env "CRED_ROLE=serve")
  fi
  if [ -n "$SHIP_B2_WRITE_KEY_ID" ]; then
    # scoped-pair path only: serve_vllm.sh routes its serve/ writes via [b2w]
    envs+=(--env "B2_WRITE_KEY_ID=$SHIP_B2_WRITE_KEY_ID"
           --env "B2_WRITE_APPLICATION_KEY=$SHIP_B2_WRITE_APPLICATION_KEY")
  fi
fi
# CDN weights mirror (b2x_boot.sh's rung-0 CDN tier). ALL THREE or none — the
# tier refuses to engage on a partial set, so shipping two of them is pure cost.
# A serve box makes the biggest base-model pull on the fleet, which is exactly
# where the edge is worth having. B2_CDN_PREFIX is a URL-bearer secret: it is
# masked in the two dry-run printers below and must never be echoed elsewhere.
if [ -n "${B2_CDN_HOST:-}" ] && [ -n "${B2_CDN_BUCKET:-}" ] && [ -n "${B2_CDN_PREFIX:-}" ]; then
  envs+=(--env "B2_CDN_HOST=$B2_CDN_HOST" --env "B2_CDN_BUCKET=$B2_CDN_BUCKET"
         --env "B2_CDN_PREFIX=$B2_CDN_PREFIX")
fi
[ -n "$MODEL_B2" ] && envs+=(--env "MODEL_B2=$MODEL_B2") || envs+=(--env "MODEL_ID=$MODEL_ID")
envs+=(--env "SERVED_NAME=$SERVED_NAME" --env "MAX_LEN=$MAX_LEN" --env "GPU_UTIL=$GPU_UTIL"
       --env "SERVE_REPLICAS=$SERVE_REPLICAS")
{ [ "$DP_SET" = "1" ] || [ "$SERVE_DP" != "1" ]; } && envs+=(--env "SERVE_DP=$SERVE_DP")
[ "$SERVE_TP" != "1" ] && envs+=(--env "SERVE_TP=$SERVE_TP")
[ -n "$LORA_SPECS" ]       && envs+=(--env "LORA_SPECS=$LORA_SPECS" --env "MAX_LORA_RANK=$MAX_LORA_RANK")
[ -n "$KV_CACHE_DTYPE" ]   && envs+=(--env "KV_CACHE_DTYPE=$KV_CACHE_DTYPE")
[ -n "$SERVE_DTYPE" ]      && envs+=(--env "SERVE_DTYPE=$SERVE_DTYPE")
# The box's on-box identity gate is ARMED by this one variable, and its absence
# is what makes every pre-artifact caller behave exactly as before. Set means
# "an expectation was shipped for this serve": a box that cannot then find it
# FAILS rather than skipping, because a gate that vanishes on a transient B2
# read is a gate you cannot rely on having run.
[ -n "$MODEL_ARTIFACT" ]   && envs+=(--env "SERVE_IDENT_REQUIRED=1"
                                     --env "SERVE_IDENT_ARTIFACT=$MODEL_ARTIFACT")
# Unset => the env var never reaches the box => serve_vllm.sh emits no flag =>
# vLLM's default. One `[ -n ... ]` guard is the whole comparability argument.
[ -n "$MAX_NUM_SEQS" ]     && envs+=(--env "MAX_NUM_SEQS=$MAX_NUM_SEQS")
[ -n "$MAX_NUM_BATCHED_TOKENS" ] && envs+=(--env "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS")
[ -n "$CHAT_TEMPLATE_B2" ] && envs+=(--env "CHAT_TEMPLATE_B2=$CHAT_TEMPLATE_B2")
[ -n "$QUANTIZATION" ]     && envs+=(--env "QUANTIZATION=$QUANTIZATION")
[ -n "$TRUST_REMOTE_CODE" ] && envs+=(--env "TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE")
# Prefix caching is OPT-OUT, so the guard is inverted against every knob above:
# only the DISABLE reaches the box. Unset => serve_vllm.sh's own :-1 default
# turns it on, which is the point (owner 2026-08-22).
[ -n "$SERVE_PREFIX_CACHING" ] && envs+=(--env "SERVE_PREFIX_CACHING=$SERVE_PREFIX_CACHING")
[ -n "$SERVE_MTP" ]        && envs+=(--env "SERVE_MTP=$SERVE_MTP")
[ -n "$SERVE_MTP_NUM_SPEC" ] && envs+=(--env "SERVE_MTP_NUM_SPEC=$SERVE_MTP_NUM_SPEC")
[ -n "$VLLM_EXTRA_ARGS" ]  && envs+=(--env "VLLM_EXTRA_ARGS=$VLLM_EXTRA_ARGS")
[ -n "${HF_TOKEN:-}" ]     && envs+=(--env "HF_TOKEN=$HF_TOKEN")   # forwarded via --no-hf-token path
# Always explicit: attach mode can land on a box carrying a pre-2026-08-21
# serve_vllm.sh whose in-script default is still ON.
envs+=(--env "CPU_FARM=$CPU_FARM_ON")

# --- G1/G4: ATTACH MODE — serve on an EXISTING box (no herdd launch) ---------
# Reuses ALL of the flag/bakeoff/env assembly above; only the delivery differs:
# instead of wiring serve_vllm.sh as a launch-time onstart, we ssh the SAME
# onstart/serve_vllm.sh onto a running (or just-resumed) box and run it detached.
# serve_vllm.sh writes the SAME SERVE_STATUS marker (SERVE_ID + B2_*), so the rest
# of the bundle (serve_ready.sh) keys on it unchanged.
box_status() {
  "${VCTL[@]}" show "$1" 2>/dev/null \
    | python3 -c 'import json,sys;print((json.load(sys.stdin).get("actual_status") or "").lower())' 2>/dev/null \
    || echo ""
}

write_box_env() {   # G4: single source of truth for "which box is this run serving on"
  local iid="$1" host="$2" port="$3"
  mkdir -p "$(dirname "$BOX_ENV")"
  {
    echo "# serve_on_box (launch_serve.sh --on-box) — $(date -u +%FT%TZ)"
    echo "IID=$iid"
    echo "SERVE_ID=$SERVE_ID"
    echo "SERVED_NAME=$SERVED_NAME"
    echo "SSH_HOST=$host"
    echo "SSH_PORT=$port"
  } > "$BOX_ENV"
}

attach_serve() {
  local iid="$ON_BOX"
  local src="${_LAUNCH_SERVE_ONSTART:-$HERE/onstart/serve_vllm.sh}"
  [ -f "$src" ] || { echo "!! attach: serve payload not found: $src" >&2; exit 1; }
  echo ">> [on-box] attach serve '$SERVE_ID' to instance $iid (served=$SERVED_NAME, model=${MODEL_B2:-$MODEL_ID})"

  # (a) resume the box if parked/exited (or forced with --resume) --------------
  local st; st="$(box_status "$iid")"
  if [ "$st" != "running" ]; then
    if [ "$RESUME" -eq 1 ] || [ "$st" = "exited" ] || [ "$st" = "stopped" ] || [ -z "$st" ]; then
      echo ">> [on-box] box status='${st:-unknown}' — resuming (herdd start --wait 600 --retry 900)"
      "${VCTL[@]}" start "$iid" --wait 600 --retry 900
    else
      echo "!! [on-box] instance $iid status='$st' (not running) — pass --resume to start it" >&2; exit 1
    fi
  fi

  # (b) resolve the FRESH ssh endpoint (host:port CHANGE across a park) ---------
  local ep host port
  ep="$("${VCTL[@]}" ssh "$iid" --print 2>/dev/null)" \
    || { echo "!! [on-box] no ssh endpoint for $iid (still booting?)" >&2; exit 1; }
  port="$(printf '%s\n' "$ep" | sed -n 's/.*-p \([0-9][0-9]*\).*/\1/p')"
  host="$(printf '%s\n' "$ep" | sed -n 's/.*root@\([^ ]*\).*/\1/p')"
  [ -n "$host" ] && [ -n "$port" ] \
    || { echo "!! [on-box] could not parse ssh endpoint from: $ep" >&2; exit 1; }
  local SSH=(ssh -p "$port" "root@$host"
             -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR
             -o ServerAliveInterval=30)
  echo ">> [on-box] ssh endpoint: root@$host:$port"

  # (c) relabel serve:<ID> so serve_ready.sh's full flow (marker+auto-tunnel) and
  #     park/resume identity resolve this box (verify-only --base-url also works).
  "${VCTL[@]}" label "$iid" "serve:$SERVE_ID" >/dev/null 2>&1 \
    || echo ">> [on-box] note: relabel to serve:$SERVE_ID failed (serve_ready --base-url still works)" >&2

  # (d) optional clear of a stale serve holding :8000 (opt-in; only this box) ---
  # `[v]llm serve` compiles to the same regex but is NOT the literal in the remote
  # shell's own argv — a bare pattern made pkill kill its own ssh shell (rc 143).
  if [ "$RESTART" -eq 1 ]; then
    echo ">> [on-box] --restart: clearing any existing vllm/haproxy on $iid"
    local _krc=0
    "${SSH[@]}" "pkill -f '[v]llm serve' >/dev/null 2>&1 || true; pkill -x haproxy >/dev/null 2>&1 || true; sleep 2" \
      || _krc=$?
    if [ "$_krc" -ne 0 ]; then
      echo "!! [on-box] --restart: remote kill step FAILED rc=$_krc (ssh/transport, not a no-match)." >&2
      echo "!!   Aborting BEFORE the payload is staged — the box may still hold the old serve on :8000." >&2
      exit "$_krc"
    fi
  fi

  # (e) LAUNCHED marker before we kick the serve (mirrors the launch path) ------
  # Best-effort: never abort on it. Under `set -o pipefail` an rclone that exits
  # before draining stdin makes `echo` take SIGPIPE, so the pipeline returns 141
  # and `set -e` would kill the run — mid-flight, after keys are already minted.
  if [ "$MARKER" -eq 1 ]; then
    bash "$HERE/b2_sync.sh" config >/dev/null
    echo "LAUNCHED $(date -u +%FT%TZ)" | rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/SERVE_STATUS" \
      || echo ">> note: SERVE_STATUS marker write failed (non-fatal)" >&2
  fi

  # (f) push the env file (identical K=V pairs the launch path forwards as --env)
  #     + the SAME serve payload, then run it detached (survives the ssh session).
  local remote_env=/root/serve_${SERVE_ID}.env remote_serve=/root/serve_${SERVE_ID}.sh
  {
    echo "# serve_on_box env — $(date -u +%FT%TZ)"
    local e k v
    for e in "${envs[@]}"; do
      [ "$e" = "--env" ] && continue
      k="${e%%=*}"; v="${e#*=}"
      printf 'export %s=%q\n' "$k" "$v"
    done
    # THIS attach's --model is authoritative: clear the counterpart var. An
    # earlier launch's MODEL_B2 persists in /etc/environment and the ssh session
    # inherits it, so an unclear MODEL_ID attach served BASE weights under the
    # new --served-name with every readiness gate green (2026-08-21).
    [ -n "$MODEL_B2" ] || echo "export MODEL_B2=''"
    [ -n "$MODEL_ID" ] || echo "export MODEL_ID=''"
  } | "${SSH[@]}" "cat > $remote_env && chmod 600 $remote_env"
  "${SSH[@]}" "cat > $remote_serve && chmod +x $remote_serve" < "$src"
  # serve_summary.json capture: serve_vllm.sh looks for the parser at
  # /workspace/parse_vllm_mem.py (last local rung of its resolution ladder).
  # Best-effort — a box without it just logs "parser not on this box".
  if [ -f "$HERE/parse_vllm_mem.py" ]; then
    "${SSH[@]}" "mkdir -p /workspace && cat > /workspace/parse_vllm_mem.py" \
      < "$HERE/parse_vllm_mem.py" \
      || echo ">> [on-box] note: parse_vllm_mem.py push failed (no serve_summary.json)" >&2
  fi
  # identity payload — HARD, unlike the parser above: this attach shipped
  # SERVE_IDENT_REQUIRED=1, so a box that cannot find the gate will refuse to
  # serve. Failing to stage it here means failing on the box instead, minutes
  # and one 52 GiB pull later.
  while IFS='|' read -r _src _name; do
    [ -n "$_src" ] || continue
    [ -f "$_src" ] || { echo "!! [on-box] identity payload missing: $_src" >&2; exit 1; }
    "${SSH[@]}" "mkdir -p /workspace && cat > /workspace/$_name" < "$_src" \
      || { echo "!! [on-box] staging /workspace/$_name FAILED — the box would refuse to serve; aborting" >&2; exit 1; }
    echo ">> [on-box] staged /workspace/$_name"
  done < <(ident_assets)
  echo ">> [on-box] staged $remote_serve + $remote_env; starting serve (detached)"
  "${SSH[@]}" ". $remote_env && nohup bash $remote_serve >/root/serve_${SERVE_ID}.log 2>&1 </dev/null & disown; echo \"   started serve pid \$!\""

  # (g) box.env — single source of truth (G4) ---------------------------------
  write_box_env "$iid" "$host" "$port"

  # --- next steps + teardown nag ---------------------------------------------
  echo ">> [on-box] wrote $BOX_ENV (IID=$iid SERVE_ID=$SERVE_ID SSH=root@$host:$port)"
  echo ">> serve marker: b2:${B2_BUCKET:-<none>}/serve/$SERVE_ID/SERVE_STATUS"
  echo ">>   ready   : $HERE/serve_ready.sh $SERVE_ID   (polls marker -> tunnels -> prints LLM_BASE_URL)"
  echo ">>   OR gate a pre-tunnelled/local serve: $HERE/serve_ready.sh --base-url http://127.0.0.1:<port>/v1 --expect-models $SERVED_NAME"
  echo ">>   tunnel  : ${VCTL[*]} tunnel $iid --local 28087 --remote 8000 [--background]"
  echo ">>   PARK-BACK (not destroy) when done: ${VCTL[*]} stop $iid --wait 120"
  if [ "$WAIT_READY" -eq 1 ]; then exec "$HERE/serve_ready.sh" "$SERVE_ID"; fi
}

if [ -n "$ON_BOX" ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo ">> [dry-run/on-box] would attach serve '$SERVE_ID' to instance $ON_BOX"
    echo ">> [dry-run/on-box] served=$SERVED_NAME  model=${MODEL_B2:-$MODEL_ID}  marker=$([ "$MARKER" -eq 1 ] && echo on || echo off)  resume=$RESUME restart=$RESTART"
    echo ">> [dry-run/on-box] box.env target: $BOX_ENV"
    echo ">> [dry-run/on-box] serve payload : ${_LAUNCH_SERVE_ONSTART:-$HERE/onstart/serve_vllm.sh}"
    echo ">> [dry-run/on-box] env pushed to box (secrets masked):"
    for e in "${envs[@]}"; do
      case "$e" in
        --env) continue;;
        VLLM_API_KEY=*|B2_APPLICATION_KEY=*|B2_KEY_ID=*|B2_WRITE_APPLICATION_KEY=*|B2_WRITE_KEY_ID=*|HF_TOKEN=*|B2_CDN_PREFIX=*) echo "     ${e%%=*}=****";;
        *) echo "     $e";;
      esac
    done
    # the counterpart clear the real attach writes — visible here so a dry run
    # proves the stale-MODEL_B2 override is closed without spending a box.
    [ -n "$MODEL_B2" ] || echo "     MODEL_B2=   (cleared: this attach's --model is MODEL_ID)"
    [ -n "$MODEL_ID" ] || echo "     MODEL_ID=   (cleared: this attach's --model is MODEL_B2)"
    while IFS='|' read -r _src _name; do
      [ -n "$_src" ] || continue
      echo ">> [dry-run/on-box] would stage $(wc -c < "$_src" 2>/dev/null || echo '?')B -> /workspace/$_name"
    done < <(ident_assets)
    echo ">> [dry-run/on-box] NO herdd start/ssh/label, NO marker, NO spend"
    exit 0
  fi
  attach_serve
  exit 0
fi

# --- search / price args (mirrors launch_train.sh:268-290) --------------------
# Spot is the DEFAULT (owner ruling 2026-08-02): herdd autobids 1.2x the live
# floor, and the post-launch `fleet watch --profile serve` runs the same
# defend/rescue bid ladder as jobs boxes (herdd serve_mode), so an outbid is
# an automated rescue, not a lost window. The one legitimate on-demand case is
# an OF-RECORD PAIRED WINDOW (a preemption mid-window invalidates the pairing);
# say so with an explicit --type ondemand.
if [ "$TYPE" = "bid" ]; then
  price_args=(--type bid); [ -n "$PRICE" ] && price_args+=(--price "$PRICE")
  if [ "$MARKER" -ne 1 ]; then
    # Without the SERVE_STATUS marker the MAX_HOURS watchdog park is
    # indistinguishable from an eviction (herdd._serve_self_park_soft reads
    # the marker) — fleetd would rescue-resume the box against its own
    # watchdog. Fail closed pre-spend rather than launch that fight.
    echo "!! spot serve needs B2 creds (SERVE_STATUS marker carries the watchdog's" >&2
    echo "!!   SELF_PARKED signal that tells fleetd a park from an eviction)." >&2
    echo "!!   Source tools/vast/.env, or pass --type ondemand explicitly." >&2
    exit 1
  fi
else
  price_args=(--type ondemand)
  echo ">> note: --type ondemand — the explicit exception (of-record paired windows); spot+fleetd is the default posture" >&2
  [ -n "$PRICE" ] && echo ">> note: --price ignored without --type bid (on-demand pays the listed rate)" >&2
fi
search_args=()
[ -n "$INET_DOWN" ] && search_args+=(--inet-down "$INET_DOWN")
for g in "${GEO[@]:-}"; do [ -n "$g" ] && search_args+=(--geo "$g"); done
# NO geo default (owner directive 2026-08-05, superseding the 2026-07-20 US pin
# recorded in tools/vast/workflow.py's ResourceProfile.geo comment): the search
# is global and gated on advertised bandwidth (--inet-down / the
# LAUNCH_INET_DOWN_MBPS knob) plus the CUDA floor. This also fixes the implicit
# re-pin on every boot-SLA relaunch: write_sla_spec saves ORIG_ARGV, which
# normally carries no --geo, so a re-fire used to re-derive `--geo US` here.
case "$GPU_RAM" in 0|0.0|"") ;; *) search_args+=(--gpu-ram "$GPU_RAM");; esac
[ -n "$OFFER" ] && search_args+=(--offer "$OFFER")
for m in "${MACHINES[@]:-}"; do [ -n "$m" ] && search_args+=(--machine "$m"); done
for m in "${EXCLUDES[@]:-}"; do [ -n "$m" ] && search_args+=(--exclude-machine "$m"); done
for h in "${HOSTS[@]:-}"; do [ -n "$h" ] && search_args+=(--host "$h"); done
# min-CUDA guard: forward to auto-pick UNLESS --offer pins the host (then F-2b runs
# wrapper-side, since --offer skips the search filter entirely).
if [ -z "$OFFER" ] && cuda_enabled; then search_args+=(--cuda "$CUDA_MIN"); fi
port_args=(); [ -n "$PUBLIC_PORT" ] && port_args=(--port 8000)

# --- container disk: measured, not hand-typed (owner ruling 2026-08-02) -------
# AUTO unless --disk was passed: disksize.serve_disk_gb over the MEASURED model
# bytes (rclone size of the b2: subpath). An HF-id model is unmeasurable
# pre-launch -> static vastconf.DISK_DEFAULT_SERVE_GB with a loud UNMEASURED
# note (never silently confident). LoRAs are counted 3GB each, coarse but
# bounded. Storage bills on the ALLOCATED size — see disksize.py's history.
if [ -z "$DISK" ]; then
  MODEL_BYTES=0
  # The artifact gate already measured this prefix pre-spend; reusing its number
  # is one fewer LIST of a 27-object / 52 GiB prefix, and it cannot disagree with
  # what the gate accepted.
  case "${ARTIFACT_BYTES:-0}" in ''|0|*[!0-9]*) ;; *) MODEL_BYTES="$ARTIFACT_BYTES";; esac
  if [ "$MODEL_BYTES" = "0" ] && [ -n "$MODEL_B2" ] && [ "$MARKER" -eq 1 ]; then
    bash "$HERE/b2_sync.sh" config >/dev/null 2>&1 || true
    MODEL_BYTES=$(rclone size --json "b2:$B2_BUCKET/$MODEL_B2" 2>/dev/null \
      | python3 -c 'import json,sys; print(json.load(sys.stdin).get("bytes", 0))' \
      2>/dev/null || echo 0)
  fi
  N_LORA=0
  [ -n "$LORA_SPECS" ] && N_LORA=$(( $(printf '%s' "$LORA_SPECS" | tr -cd ',' | wc -c) + 1 ))
  read -r DISK DISK_NOTE < <(MODEL_BYTES="$MODEL_BYTES" N_LORA="$N_LORA" \
    SERVE_TOOLS_DIR="$HERE" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["SERVE_TOOLS_DIR"])
import disksize, vastconf
gb, b = disksize.serve_disk_gb(int(os.environ.get("MODEL_BYTES") or 0),
                               extra_gb=3.0 * int(os.environ.get("N_LORA") or 0))
if b["complete"]:
    print(int(gb), f"auto: model {b['model_gb']}GB x{b['model_factor']} "
                   f"+ {b['base_overhead_gb']}GB overhead"
                   + (f" + {b['extra_gb']}GB lora" if b["extra_gb"] else ""))
else:
    print(vastconf.DISK_DEFAULT_SERVE_GB,
          "UNMEASURED model (HF id / no B2 read) — static default; "
          "pass --disk to size it yourself")
PY
)
  echo ">> disk: ${DISK}GB (${DISK_NOTE})"
fi

# --- onstart wire: inline if it fits, else B2 boot-pull (train_boot pattern) ---
# Vast caps the inline onstart at 16 KiB (`len(args)`). serve_vllm.sh is heavily
# documented AND large: its STRIPPED wire is ~16.5 KiB, OVER the cap. So — mirroring
# `herdd train` (onstart/train_boot.sh) and jobd (onstart/jobd_boot.sh) — when the
# stripped wire won't fit we stage the FULL serve_vllm.sh to
# b2:$B2_BUCKET/serve/$SERVE_ID/serve_main.sh and ship only the tiny
# onstart/serve_boot.sh (~2 KiB) that pulls+execs it at boot (removing the ceiling
# permanently). When the wire DOES fit (a slimmer payload, or the
# _LAUNCH_SERVE_ONSTART dev stub) we ship it inline, unchanged — no B2 round-trip.
# The B2 boot-pull needs the marker creds; a no-B2 serve that overflows the cap
# hard-fails (attach with --on-box, or re-slim serve_vllm.sh). Strip is heredoc-
# aware (rclone.conf/haproxy.cfg <<EOF blocks pass through); the source stays
# fully commented, only the WIRE is stripped.
# _LAUNCH_SERVE_ONSTART: dev-only override of the onstart source (F-1a lint test).
ONSTART_SRC="${_LAUNCH_SERVE_ONSTART:-$HERE/onstart/serve_vllm.sh}"
SERVE_BOOT_SRC="$HERE/onstart/serve_boot.sh"
ONSTART_CAP=15872

strip_onstart_wire() {   # $1=source path -> stripped wire on stdout (heredoc-aware)
  python3 - "$1" <<'PY'
import sys, re
hd = None
for ln in open(sys.argv[1]):
    if hd is None:                                   # outside a heredoc body
        s = ln.strip()
        if s == "" or (s.startswith("#") and not s.startswith("#!")):
            continue                                  # drop blank / full-line comment
        sys.stdout.write(s + "\n")                   # drop leading indent too
        m = re.search(r"""<<-?\s*['"]?([A-Za-z_]\w*)""", ln)
        if m:
            hd = m.group(1)                           # entering a heredoc: keep everything
    else:
        sys.stdout.write(ln)
        if ln.strip() == hd:
            hd = None
PY
}

# --- F-1a: onstart size budget (pre-spend; also runs in --dry-run) ------------
# total = stripped wire + injected ssh pubkey (--ssh) + ~128B slack for vast's own
# prepends. We pass --no-hf-token so there is no 1183B hf_login prelude. Both wire
# modes (inline serve_vllm / B2-boot serve_boot) are linted against this budget.
PUBKEY=""
for k in ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do [ -f "$k" ] && { PUBKEY="$k"; break; }; done
PUB_BYTES=0; [ -n "$PUBKEY" ] && PUB_BYTES=$(wc -c < "$PUBKEY")

ONSTART_WIRE="${TMPDIR:-/tmp}/serve.onstart.$$.sh"
WIRE_MODE=inline
STAGE_SERVE_MAIN=0
strip_onstart_wire "$ONSTART_SRC" > "$ONSTART_WIRE"
WIRE_BYTES=$(wc -c < "$ONSTART_WIRE")
TOTAL_BYTES=$(( WIRE_BYTES + PUB_BYTES + 128 ))

if [ "$TOTAL_BYTES" -gt "$ONSTART_CAP" ]; then
  # inline overflows — switch to the B2 boot-pull wire (needs the marker creds).
  if [ "$MARKER" -ne 1 ]; then
    rm -f "$ONSTART_WIRE"
    echo "!! F-1a: onstart wire ${WIRE_BYTES}B + pubkey ${PUB_BYTES}B + 128 = ${TOTAL_BYTES}B > ${ONSTART_CAP} cap," >&2
    echo "!!   and no B2 creds to stage the payload for a boot-pull wire (B2_* absent)." >&2
    echo "!!   Source tools/vast/.env for B2 creds, attach with --on-box, or re-slim serve_vllm.sh." >&2
    exit 1
  fi
  [ -f "$SERVE_BOOT_SRC" ] || { rm -f "$ONSTART_WIRE"; echo "!! F-1a: wire over cap but $SERVE_BOOT_SRC missing — cannot build the boot-pull wire" >&2; exit 1; }
  echo ">> onstart wire ${TOTAL_BYTES}B > ${ONSTART_CAP} cap — staging serve_vllm.sh to B2 + shipping onstart/serve_boot.sh (train_boot pattern)" >&2
  WIRE_MODE=b2boot
  STAGE_SERVE_MAIN=1
  strip_onstart_wire "$SERVE_BOOT_SRC" > "$ONSTART_WIRE"
  WIRE_BYTES=$(wc -c < "$ONSTART_WIRE")
  TOTAL_BYTES=$(( WIRE_BYTES + PUB_BYTES + 128 ))
  if [ "$TOTAL_BYTES" -gt "$ONSTART_CAP" ]; then
    rm -f "$ONSTART_WIRE"
    echo "!! F-1a: BOOT wire ${WIRE_BYTES}B + pubkey ${PUB_BYTES}B + 128 = ${TOTAL_BYTES}B > ${ONSTART_CAP} cap — serve_boot.sh unexpectedly large" >&2
    exit 1
  fi
fi

# --- assemble the exact herdd launch argv (dry-run prints it, live runs it) -
launch_argv=("${VCTL[@]}" launch --gpu "$GPU" --num-gpus "$NGPU"
  "${price_args[@]}" "${search_args[@]}"
  --label "serve:$SERVE_ID" --no-hf-token
  --image "$IMAGE" --disk "$DISK" --ssh
  "${port_args[@]}"
  --onstart "$ONSTART_WIRE" "${envs[@]}")

# --- --dry-run: print env (secrets masked) + wire size + argv; NO marker/spend -
if [ "$DRY_RUN" -eq 1 ]; then
  echo ">> [dry-run] SERVE_ID=$SERVE_ID  served=$SERVED_NAME  marker=$([ "$MARKER" -eq 1 ] && echo on || echo off)"
  echo ">> [dry-run] instance env:"
  for e in "${envs[@]}"; do
    case "$e" in
      --env) continue;;
      VLLM_API_KEY=*|B2_APPLICATION_KEY=*|B2_KEY_ID=*|B2_WRITE_APPLICATION_KEY=*|B2_WRITE_KEY_ID=*|HF_TOKEN=*|B2_CDN_PREFIX=*) echo "     ${e%%=*}=****";;
      *) echo "     $e";;
    esac
  done
  if [ "$WIRE_MODE" = b2boot ]; then
    echo ">> [dry-run] onstart wire: B2-BOOTSTRAP serve_boot.sh ${WIRE_BYTES}B (+pubkey ${PUB_BYTES}B +128 = ${TOTAL_BYTES}B < ${ONSTART_CAP} OK)"
    echo ">> [dry-run] would stage serve_vllm.sh ($(wc -c < "$ONSTART_SRC")B) -> b2:${B2_BUCKET:-<none>}/serve/$SERVE_ID/serve_main.sh"
  else
    echo ">> [dry-run] onstart wire: INLINE ${WIRE_BYTES}B stripped (+pubkey ${PUB_BYTES}B +128 = ${TOTAL_BYTES}B < ${ONSTART_CAP} OK)"
  fi
  if [ "$MARKER" -eq 1 ] && [ -f "$HERE/parse_vllm_mem.py" ]; then
    echo ">> [dry-run] would stage parse_vllm_mem.py ($(wc -c < "$HERE/parse_vllm_mem.py")B) -> b2:${B2_BUCKET:-<none>}/serve/$SERVE_ID/parse_vllm_mem.py (serve_summary.json capture)"
  fi
  while IFS='|' read -r _src _name; do
    [ -n "$_src" ] || continue
    echo ">> [dry-run] would stage $_name ($(wc -c < "$_src" 2>/dev/null || echo '?')B) -> b2:${B2_BUCKET:-<none>}/serve/$SERVE_ID/$_name (on-box identity gate)"
  done < <(ident_assets)
  echo ">> [dry-run] herdd argv:"
  for a in "${launch_argv[@]}"; do
    case "$a" in
      VLLM_API_KEY=*|B2_APPLICATION_KEY=*|B2_KEY_ID=*|B2_WRITE_APPLICATION_KEY=*|B2_WRITE_KEY_ID=*|HF_TOKEN=*|B2_CDN_PREFIX=*) printf '     %q ' "${a%%=*}=****";;
      *) printf '     %q ' "$a";;
    esac
  done; echo
  rm -f "$ONSTART_WIRE"
  exit 0
fi

# --- F-2b guard (live path only, pre-spend) -----------------------------------
if [ -n "$OFFER" ] && cuda_enabled; then pinned_offer_cuda_check "$OFFER" "$CUDA_MIN"; fi

# --- LAUNCHED marker (proves B2 creds before money; launch_train.sh:335 analog) -
# + per-SERVE payload staging for the B2 boot-pull wire: the FULL (unstripped)
# serve_vllm.sh lands at serve/<SERVE_ID>/serve_main.sh for onstart/serve_boot.sh
# to pull+exec. hard (set -e): a failed stage aborts before spend — a booted box
# would otherwise pull a missing server. STAGE_SERVE_MAIN is only ever 1 with
# MARKER=1 (the over-cap branch requires B2 creds), so this is inside the guard.
if [ "$MARKER" -eq 1 ]; then
  bash "$HERE/b2_sync.sh" config >/dev/null
  # best-effort marker — see the --on-box copy above for why this must not abort
  echo "LAUNCHED $(date -u +%FT%TZ)" | rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/SERVE_STATUS" \
    || echo ">> note: SERVE_STATUS marker write failed (non-fatal)" >&2
  if [ "$STAGE_SERVE_MAIN" -eq 1 ]; then
    echo ">> staging serve_vllm.sh ($(wc -c < "$ONSTART_SRC")B) -> b2:$B2_BUCKET/serve/$SERVE_ID/serve_main.sh" >&2
    rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/serve_main.sh" < "$ONSTART_SRC"
  fi
  # serve_summary.json capture (V10 sec-7 gap): a launched serve box has no repo
  # checkout, so the memory-profile parser rides the SAME per-SERVE prefix as
  # serve_main.sh and serve_vllm.sh pulls it from there. SOFT, unlike the server
  # stage above: a missing parser costs an observation, not the serve.
  if [ -f "$HERE/parse_vllm_mem.py" ]; then
    rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/parse_vllm_mem.py" < "$HERE/parse_vllm_mem.py" \
      || echo ">> note: parse_vllm_mem.py stage failed (no serve_summary.json for this run)" >&2
  fi
  # identity payload — HARD (set -e), like serve_main.sh and unlike the parser:
  # the box was shipped SERVE_IDENT_REQUIRED=1 and will refuse to serve without
  # these. A stage that fails here costs nothing; one that fails silently costs
  # a rented box and a 52 GiB pull before the refusal lands.
  while IFS='|' read -r _src _name; do
    [ -n "$_src" ] || continue
    echo ">> staging $_name ($(wc -c < "$_src")B) -> b2:$B2_BUCKET/serve/$SERVE_ID/$_name" >&2
    rclone rcat "b2:$B2_BUCKET/serve/$SERVE_ID/$_name" < "$_src"
  done < <(ident_assets)
fi

echo ">> launching $NGPU x $GPU ($TYPE) serve '$SERVE_ID' (model=${MODEL_B2:-$MODEL_ID}, served=$SERVED_NAME)"
out="$("${launch_argv[@]}")"; rm -f "$ONSTART_WIRE"
echo "$out"
ID=$(echo "$out" | sed -n 's/^launched instance \([0-9]*\).*/\1/p')
[ -z "$ID" ] && { echo "!! could not parse instance id; check output above" >&2; exit 1; }

# --- best-effort runmeta launched event (role=serve; launch_train.sh:399-422) --
DPH_VAL=""
[ "$TYPE" = "bid" ] && DPH_VAL="$PRICE"
[ -z "$DPH_VAL" ] && DPH_VAL=$(printf '%s\n' "$out" | sed -n 's/^picked:.*dph=\$\([0-9.][0-9.]*\).*/\1/p' | head -n1)
meta_fields=(--field "role=serve" --field "instance_id=$ID" --field "gpu=$GPU")
[ -n "$DPH_VAL" ] && meta_fields+=(--field "dph=$DPH_VAL")
python3 "$HERE/runmeta.py" emit "$SERVE_ID" launched "${meta_fields[@]}" >/dev/null 2>&1 \
  || echo ">> note: runmeta serve launched-event emit failed (non-fatal)" >&2

# --- fleetd serve watch: the box is MANAGED from launch (2026-08-02) ----------
# The serve profile runs the same defend/rescue bid ladder as jobs boxes
# (herdd serve_mode), so a spot serve box's outbid is an automated rescue and
# its spend is hard-capped. Budget default: actual $/hr x MAX_HOURS x 1.6
# (rescue bids raise the rate; 1.6 covers the ladder's headroom to the
# on-demand-anchored ceiling). A failed registration is LOUD, not fatal — the
# box still serves, but nothing caps it until the printed command is run.
if [ -z "$BUDGET" ]; then
  BUDGET=$(python3 - "$DPH_VAL" "$MAX_HOURS" <<'PY'
import sys
try:
    dph = float(sys.argv[1])
except (ValueError, IndexError):
    dph = 0.0
hours = float(sys.argv[2] or 12) or 24        # MAX_HOURS=0 (off) -> cap at 24h
print(round(dph * hours * 1.6, 2) if dph > 0 else 10.0)
PY
)
fi
#
# The watch also carries WHAT this box is supposed to be serving (P3). Same
# resolution that composed identity_expect.json a few hundred lines up — one
# `serve_artifact.py resolve` read, so the pin fleetd holds and the expectation
# the BOX is gated on cannot come from two readings of the registry that
# disagree. The daemon re-verifies the sha12 against the committed registry at
# registration anyway; this passes it because it is already in hand.
# Without --model-artifact there is no pin, the flags are omitted, and the
# registration is byte-identical to what it has always been.
WATCH_ARGS=(fleet watch "$ID" --profile serve --budget "$BUDGET")
if [ -n "$MODEL_ARTIFACT" ] && [ -n "${AR_FINGERPRINT:-}" ]; then
  WATCH_ARGS+=(--artifact "$MODEL_ARTIFACT" --expect-ident "${AR_FINGERPRINT:0:12}")
fi
if "${VCTL[@]}" "${WATCH_ARGS[@]}" >/dev/null 2>&1; then
  echo ">> fleetd: serve watch registered (bid defend/rescue + spend cap \$$BUDGET)"
  [ -n "$MODEL_ARTIFACT" ] && [ -n "${AR_FINGERPRINT:-}" ] \
    && echo ">> fleetd: identity-pinned to '$MODEL_ARTIFACT' (ident ${AR_FINGERPRINT:0:12}) — fleetd alarms and parks if this box ever verifies a different one"
else
  echo "!! fleetd serve-watch registration FAILED — the box is UNMANAGED (no bid ladder, no spend cap, no identity check)." >&2
  echo "!!   register it: ${VCTL[*]} ${WATCH_ARGS[*]}" >&2
fi

# --- boot-SLA relaunch spec (owner directive 2026-08-03) ----------------------
# The serve watch enforces a come-online SLA (BOOT_SLA_S, default 600s): a box
# whose SERVE_STATUS still reads the workstation's LAUNCHED token past the
# deadline is destroyed and THIS launch is re-fired on a different host
# (herdd._serve_boot_sla_condemn). That needs the launch to be reproducible,
# so save the original argv (minus the volatile flags, re-derived per kill)
# keyed by IID under ~/.local/state/herdd/serve-relaunch/. A pinned launch
# (--offer/--machine/--host) is an operator host choice: no spec, SLA
# enforcement stays alarm-only for it. Spec-write failure is a note, not an
# abort — the box serves either way; only the SLA loses its re-fire.
write_sla_spec() {
  local iid="$1" dir="${XDG_STATE_HOME:-$HOME/.local/state}/herdd/serve-relaunch"
  mkdir -p "$dir"
  SLA_IID="$iid" SLA_SERVE_ID="$SERVE_ID" SLA_KILLS="$SLA_KILLS" SLA_DIR="$dir" \
    python3 - "${ORIG_ARGV[@]}" <<'PY'
import json, os, sys
# ALLOW-LIST BY OMISSION, and that is the property to preserve: every flag not
# named here rides through verbatim, so `--model-artifact <slug>` round-trips
# and a condemn-and-relaunch RE-RESOLVES the registry, RE-RUNS the pre-spend
# gate and RE-VERIFIES the identity on the new host. Adding it to this set
# would silently downgrade every relaunched serve to an ungated one.
skip_valued = {"--serve-id", "--exclude-machine", "--sla-kills"}
argv, excl = [], []
it = iter(sys.argv[1:])
for tok in it:
    if tok in skip_valued:
        val = next(it, None)
        if tok == "--exclude-machine" and val is not None:
            excl.append(int(val))
        continue
    if tok == "--wait-ready":
        continue                       # a daemon re-fire must not block on ready
    argv.append(tok)
spec = {"script": "launch_serve.sh",
        "serve_id": os.environ["SLA_SERVE_ID"],
        "argv": argv,
        "exclude_machines": sorted(set(excl)),
        "sla_kills": int(os.environ.get("SLA_KILLS") or 0)}
path = os.path.join(os.environ["SLA_DIR"], os.environ["SLA_IID"] + ".json")
with open(path, "w") as fh:
    json.dump(spec, fh, indent=1)
print(f">> boot-SLA relaunch spec: {path} (kills={spec['sla_kills']},"
      f" excluded={spec['exclude_machines']})")
PY
}
if [ -z "$OFFER" ] && [ ${#MACHINES[@]} -eq 0 ] && [ ${#HOSTS[@]} -eq 0 ]; then
  write_sla_spec "$ID" \
    || echo ">> note: boot-SLA spec write failed — SLA enforcement is alarm-only for this box" >&2
else
  echo ">> note: pinned launch (--offer/--machine/--host) — no boot-SLA relaunch spec (the pin is an operator host choice; SLA enforcement off)" >&2
fi

# --- next steps + cost/teardown nag -------------------------------------------
echo ">> instance $ID launched. serve marker: b2:${B2_BUCKET:-<none>}/serve/$SERVE_ID/SERVE_STATUS"
echo ">>   tunnel  : ${VCTL[*]} tunnel $ID --local 28087 --remote 8000"
echo ">>   ready   : $HERE/serve_ready.sh $SERVE_ID   (polls the marker -> prints LLM_BASE_URL)"
[ -n "$DPH_VAL" ] && echo ">>   cost    : serve box $ID at \$$DPH_VAL/hr"
echo ">>   TEARDOWN: serve boxes have NO auto-teardown beyond MAX_HOURS=$MAX_HOURS — destroy after the eval:"
echo ">>             ${VCTL[*]} destroy $ID -y"

# --- --wait-ready: hand straight off to serve_ready.sh ------------------------
if [ "$WAIT_READY" -eq 1 ]; then exec "$HERE/serve_ready.sh" "$SERVE_ID"; fi
