#!/usr/bin/env bash
# launch_jobs_box.sh — one command to rent a box, put a jobs-v2 bundle on it, and
# hand it to fleetd. The TRAIN/BATCH analog of launch_serve.sh.
#
# Closes the gap JOBS_DESIGN.md "Non-goals (v1)" names as a TODO: `job submit`
# and `jobmatrix submit` both REQUIRE `--box <IID>`, so every bundle launch was
# three hand-run commands with two easy ways to lose money —
#
#     herdd launch --jobs …                 # rent, note the IID by eye
#     jobmatrix submit <dir> --box <IID>      # ...and if THIS refuses, you have
#                                             #    already rented the box
#     herdd fleet watch <IID> --profile jobs --budget N   # ...if you remember
#
# — the two failure modes being (a) a bundle that refuses its own preflight
# AFTER the meter started, and (b) a forgotten `fleet watch`, which leaves an
# unwatched box billing until fleetd's price-tiered grace fuse parks it (5 min
# at >= $2/hr, 30 min below — FLEETD_DESIGN.md §3, and only if it shows no
# workload evidence). This script does the three in the ONE order that is safe.
#
# ORDER OF OPERATIONS (each step exists to protect the next):
#   1. PREFLIGHT THE BUNDLE WITH NO BOX. A `--dry-run` submit against a
#      placeholder box id validates the config, builds + hashes the bundle, and
#      runs the `tracks:` provenance check — the one that refuses when a staged
#      B2 object no longer matches the repo file it mirrors. That refusal is
#      designed to fire "before any spend", but in the hand-run order above it
#      fires one step too late. Here it is the FIRST thing that happens, and a
#      non-zero exit means we never call the vast API at all.
#   2. RENT, with `--jobs` (jobd starts at boot, no attach round-trip) and
#      `--fleet-watch` (registers a `bare` watch the instant the box exists, so
#      the launch -> watch gap is never open).
#   3. SUBMIT IMMEDIATELY, without waiting for the box to be ready. Tickets live
#      in B2, not on the box, so queueing them early is not just harmless, it is
#      the documented race-free pattern: "jobd never parks while any pending
#      ticket sits in the queue" (JOBS_DESIGN.md). Waiting for `running` first
#      would open the idle-park window this closes.
#   4. UPGRADE THE WATCH to `--profile jobs --budget N --standing` — the
#      spend-capable ladder (defend/rescue bid, re-attach jobd on resume, park
#      on queue drain, park + alarm on budget breach), plus `--standing`
#      (FLEETD_DESIGN.md §4a-i; default here since 2026-08-20, FLEET_REVIEW
#      item 2 — 119 journaled LAPSED cycles where a drained watch just ENDED,
#      leaving the ceiling behind with no armed ladder until someone
#      re-registered). Standing changes what happens AFTER the queue drains,
#      not the drain itself: the box still drain-parks exactly as before, but
#      the watch record survives dormant-but-armed, and the next `job submit`
#      to this box re-arms it under the same cumulative budget. Opt out with
#      STANDING_WATCH=0 (env knob, not a flag — see its declaration below).
#      This step is still LAST on purpose: a `jobs` watch parks the box the
#      moment every ticket it can see is terminal, so arming it before the
#      tickets exist parks a box you just rented (box 46648873, 2026-08-03 —
#      it read as a budget trip and was not).
#
# Usage:
#   launch_jobs_box.sh <bundle-dir> [flags]
#   launch_jobs_box.sh tools/witness/jobs/v7-longctx-train --budget 6
#
# Flags:
#   --budget USD       fleetd hard spend cap; breach = PARK + alarm, never
#                      destroy (default 5). This is the real cost control —
#                      job `timeout_s` is a hang detector, not a wallet.
#   --gpu G            gpu alias (default h100 — sm_90, the architecture our
#                      pinned-FA2 bundles actually have cubins for; see THE
#                      DEFAULT CARD below). A bundle declaring needs.cc_allow
#                      DROPS this default — see THE ARCHITECTURE ALLOWLIST.
#   --num-gpus N       cards to rent (default 1 — owner ruling 2026-08-13,
#                      was 2). One card is the conservative default: it is what
#                      most bundles declare in `needs.gpus`, a second card is a
#                      real cost line, and the scaling is sublinear anyway
#                      (~1.27x speed for ~2.07x money at W=2, V7_PERF_LEVERS).
#                      Renting one too few is a cheap re-launch; renting one too
#                      many bills silently for the life of the box. A whole-box
#                      bundle (`needs.gpus: "all"`) DDPs over every card, so
#                      pass the count you want explicitly there; see the
#                      GRAD_ACCUM divisibility note below.
#   --gpu-ram GB       min GB per card (default: from the bundle's needs). This
#                      is a SEARCH FILTER — it picks an offer. It is NOT the
#                      contract the job runs under: `needs.gpu_ram_gb` is, and
#                      jobd re-checks it ON the box. Passing a value BELOW the
#                      bundle's need is refused here (see --force-gpu-ram).
#   --force-gpu-ram    allow --gpu-ram below the bundle's needs.gpu_ram_gb. Only
#                      correct when you know the declaration is wrong; the box
#                      you rent will still refuse the ticket unless the bundle
#                      is edited too.
#   --host-ram GB      min HOST RAM (default: from the bundle's
#                      needs.host_ram_gb). The axis CPU-shaped work is really
#                      sized by — a bf16 CPU merge holds the whole base
#                      resident. Search filter only: jobd has no host-RAM gate,
#                      so unlike --gpu-ram there is no ticket contract behind it
#                      and an explicit value simply wins.
#   --cpu-cores N      min effective cores (default: from needs.cpu_cores).
#                      cpu_cores_effective, i.e. the SLICE — an offer's raw
#                      cpu_cores is the whole machine's count.
#   --max-dph USD      refuse offers above this $/hr
#   --geo CC           restrict to country (repeatable; default unrestricted)
#   --type bid|ondemand   default ondemand — see the note under "SPOT"
#   --disk GB          container disk (default: AUTO from the bundle's assets).
#                      An explicit value BELOW the bundle's own estimate is
#                      refused here, before the rent (see --force-disk).
#   --force-disk       allow --disk below the estimate. Correct only when you
#                      know the derivation is wrong; the durable fix is
#                      needs.scratch_gb (or needs.disk_gb) in the bundle.
#   --only GLOB        matrix only: submit a subset of arms
#   --allow-stale-assets  run the bytes STAGED ON B2 even though a repo file
#                      they mirror has moved. Not a nuisance override: when a
#                      new arm must be trainer-identical to arms already
#                      trained, the staged blob IS the correct one, and
#                      re-staging both breaks that parity and writes shared B2
#                      state a peer session may be mid-run against (doc 49
#                      amendment 9's own reasoning). Say WHY in the run record.
#   --no-asset-check   skip the provenance preflight entirely — prefer
#                      --allow-stale-assets, which still reports the drift.
#                      Passing both flags is an error, not a last-wins.
#   --env K=V          submit-time pin folded onto the bundle's `env:`
#                      (repeatable, last wins). Single-arm bundles only — for a
#                      matrix, an arm's env belongs in matrix.py. Use it to run
#                      a bundle in a variant without minting a new bundle id,
#                      e.g. the padfree fit probe alone:
#                        --env FIT_PROBE_ONLY=1
#                      NOTE the pins live in the TICKET, not the bundle, so a
#                      later `job resubmit --reconstruct` must be given them
#                      again (herdd.py says so at submit time).
#   --artifact PREFIX=SLUG
#                      export one modelkit-registry artifact as submit-time env
#                      (repeatable): <PREFIX>_B2 plus its identity and serve
#                      facts, so a bundle's `${PREFIX_B2}` asset prefix comes
#                      from the COMMITTED registry instead of a hand-typed B2
#                      path (ASSET_PARAMETERIZATION.md). Without it the only
#                      way through this script was the raw --env escape hatch,
#                      i.e. exactly the typed prefix the registry exists to
#                      remove. Single-arm bundles only, for the same reason
#                      --env is: `jobmatrix.py submit` has no --artifact.
#   --image REF        override the box image (default: herdd.yaml)
#   --offer ID         pin an explicit offer (skips auto-pick)
#   --cc-allow LIST    forwarded to `herdd launch --cc-allow`: restrict offers
#                      to these compute capabilities AND stamp LAUNCH_CC_ALLOW
#                      on the box so eviction replacements inherit it (e.g. 90).
#                      DEFAULT: the bundle's own `needs.cc_allow`. Passing this
#                      OVERRIDES a bundle declaration, and says so out loud.
#   --eval-env-ver V   [needs.venv: eval] pin the baked eval-env on the BOX.
#                      Default: the bundle's own `env: EVAL_ENV_VER`. See
#                      "THE EVAL-ENV PIN" below — you rarely pass this.
#   --keep             fleetd: do NOT park the box when the queue drains
#   --dry-run          do step 1 and print the plan; rent nothing
#
# THE EVAL-ENV PIN. A `needs.venv: eval` bundle grades against the pre-baked
# eval env, and WHICH bake it gets is decided at BOOT, by jobd's `check_venv
# eval` -> `onstart/fetch_eval_env.sh`, which resolves `${EVAL_ENV_VER:-}` from
# the container env and otherwise falls back to `rclone cat eval-env/LATEST`.
# jobd sources the job's own `.job.env` in the ENTRYPOINT subshell, which runs
# AFTER that — so a bundle-level or `--env` pin reaches run.sh (where it is
# compared, and recorded in the artifacts) but CANNOT steer the fetch. Only the
# box launch env can, and this script previously had no way to set one.
#
# That is not a hypothetical gap. LATEST is a moving pointer and a deliberately
# pinned bake does not advance it: on 2026-08-09 `eval-env/LATEST` was
# 20260807-0503-84d35a08 while q6-round1-evals pinned 20260806-2152-76cd109a,
# so a box rented by this script provisioned a DIFFERENT rb3-xenon tree than
# the one the fixture's byte spans were resolved against — and the job died
# rc 6 at S0.b2/S0.c after paying for boot and a ~15 GB base-model pull.
#
# So: the pin is read out of the bundle (never retyped), injected as
# `herdd launch --env EVAL_ENV_VER=<ver>`, and then READ BACK OFF THE
# INSTANCE by the submit's own M4 gate under `--require-box-eval-pin`, which
# refuses if it did not land. Fail-closed at both ends — a bundle that needs a
# pin and cannot get one is refused BEFORE anything is rented.
#
# THAT COVERS THE BOX THIS SCRIPT RENTS, AND ONLY THAT BOX. A box rented on
# your behalf LATER — fleetd's eviction replacement, a handoff understudy — is
# launched by `_launch_job_replacement` / `_launch_job_understudy`, which until
# 2026-08-16 handed `_do_launch` an empty env and so reopened exactly this hole
# one hop downstream: box 47887414 was rented WITH the pin, outbid before it
# ran, and its replacement provisioned eval-env/LATEST while the queued job
# pinned a deliberately non-latest bake. Both E3 legs died rc 6 on
# env_identity's content gate. Those lanes now inherit the pin the same way
# they already inherit the image and the disk (herdd.py
# `INHERITED_LAUNCH_ENV_KEYS`), so the fail-closed claim above survives a
# rehost. The content gate remains the backstop, not the only stop.
#
# THE ARCHITECTURE ALLOWLIST. A bundle whose kernels only exist for some
# silicon says so in `needs.cc_allow: [80, 86, 89, 90]` (JOBS_CONFIG.md). This
# script reads it and passes `--cc-allow`, which narrows the offer search AND
# stamps LAUNCH_CC_ALLOW so every eviction replacement inherits the constraint —
# the link that was missing while the constraint was something a human had to
# remember to type, on every launch, about a property of the bundle. It failed
# that way three times (k4 2026-08-17, pk2's launch and both of its eviction
# replacements 2026-08-18/19).
#
# A declared allowlist also DROPS THE DEFAULT CARD NAME, for the same reason
# --machine does: a --gpu filter is an AND with the allowlist, so defaulting to
# a card outside it intersects to zero offers and reports "no offers match
# filters" — a thin-market symptom for a self-contradicting search. With the
# default dropped, cc_allow + the preferred-GPU tier policy + price pick the
# card. An EXPLICIT --gpu still wins, and is checked against the allowlist
# BEFORE the search: a contradiction refuses here, for $0.
#
# THE DEFAULT CARD. `--gpu h100` since 2026-08-19, was `rtxpro6000` (sm_120).
# (a) The baked flash_attn ships no sm_120 cubin, so the default rental was an
#     architecture our pinned-FA2 bundles cannot run on — three witnessed
#     misfires, each burning a ~19 GB weight pull and an eviction budget.
# (b) `herdd train` already defaults to h100 (herdd.yaml), so the two
#     launchers disagreed about the same question; now they agree.
# It is NOT a price argument, and the honest note is that it costs money: a live
# bid search on 2026-08-19 had RTX PRO 6000 CHEAPER than H100 — $0.4067/hr
# (cheapest of 20 offers) vs $1.4692/hr (of 13) — with more VRAM (96 vs 80 GB).
# Those are SEARCH REFERENCE prices and realized spot runs lower (pk5's H100 NVL
# bills $0.5333/hr against that $1.4692 reference), but the direction stands.
# So this trades money for compatibility: arch-tolerant work should pass
# `--gpu rtxpro6000` explicitly and take the cheaper, larger card.
#
# SPOT. `--type ondemand` is this script's default, which INVERTS `herdd
# launch` (bid) and `launch_serve.sh` (bid). That is the documented exception,
# not a new opinion — README.md, "Launch": "Spot + a fleetd watch is the posture
# for every managed box … Reserve --type ondemand for the explicit exceptions:
# an of-record paired window that a preemption would invalidate."
#
# A jobs box IS spot-capable (tickets checkpoint, jobd resumes, fleetd's jobs
# ladder defends and rescues the bid), so the default is about what the bundle
# MEASURES, not what it can survive: the first consumer here is a paired T-vs-C
# ablation whose whole claim rests on both arms seeing the same conditions, and
# a mid-arm eviction perturbs exactly one of them. Pass `--type bid` for
# throughput work, long runs, or anything where a restart is merely slower
# rather than confounding.
#
# GRAD_ACCUM DIVISIBILITY. A bundle with `needs.gpus: "all"` + `MODE: autotune`
# divides its authored GRAD_ACCUM by the card count to hold the effective batch
# invariant, and REFUSES (exit 12) when it does not divide. So --num-gpus should
# divide the bundle's GRAD_ACCUM: 1/2/4/8 for the usual 32. This script warns
# when it can read both numbers and they disagree; it does not silently adjust.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VCTL=(python3 "$HERE/herdd.py")

# GPU default: h100 (sm_90). See "THE DEFAULT CARD" above — compatibility, not
# price; rtxpro6000 was cheaper and larger, and is the right explicit choice for
# anything arch-tolerant.
BUNDLE=""; BUDGET=5; GPU=h100; GPU_EXPLICIT=0; NUM_GPUS=1; GPU_RAM=""; FORCE_GPU_RAM=0; FORCE_DISK=0; MAX_DPH=""
HOST_RAM=""; CPU_CORES=""
TYPE=ondemand; DISK=""; ONLY=""; IMAGE=""; OFFER=""; MACHINE=""; KEEP=0; DRY=0; CC_ALLOW=""
ASSET_POLICY=""; EEV_FLAG=""; EEV_PIN=""; EEV_SRC=""
GEOS=()
ENVS=()
ARTIFACTS=()

# STANDING_WATCH=0 opts the step-4 watch out of --standing (FLEETD_DESIGN.md
# §4a-i). Default ON: FLEET_REVIEW_2026-08-20 item 2 found 119 journaled
# LAPSED cycles from watches that just ended on queue drain. Env knob, not a
# flag, matching LAUNCH_JOBS_NO_ENV_FILE's style below.
WATCH_STANDING_ARGS=()
[ "${STANDING_WATCH:-1}" != "0" ] && WATCH_STANDING_ARGS=(--standing)

die() { echo "!! $*" >&2; exit 1; }
note() { echo ">> $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --budget) BUDGET="$2"; shift 2;;
    --gpu) GPU="$2"; GPU_EXPLICIT=1; shift 2;;
    --num-gpus) NUM_GPUS="$2"; shift 2;;
    --gpu-ram) GPU_RAM="$2"; shift 2;;
    --force-gpu-ram) FORCE_GPU_RAM=1; shift;;
    --host-ram) HOST_RAM="$2"; shift 2;;
    --cpu-cores) CPU_CORES="$2"; shift 2;;
    --max-dph) MAX_DPH="$2"; shift 2;;
    --geo) GEOS+=("$2"); shift 2;;
    --type) TYPE="$2"; shift 2;;
    --disk) DISK="$2"; shift 2;;
    --force-disk) FORCE_DISK=1; shift;;
    --only) ONLY="$2"; shift 2;;
    # Mutually exclusive, and an explicit error rather than last-wins: the two
    # flags mean OPPOSITE things about the provenance check (run it and accept
    # the drift vs never run it), so silently keeping whichever came second
    # launches under a policy the operator did not pick.
    --allow-stale-assets|--no-asset-check)
      if [ -n "$ASSET_POLICY" ] && [ "$ASSET_POLICY" != "$1" ]; then
        die "$ASSET_POLICY and $1 are mutually exclusive: --allow-stale-assets still
   runs the provenance check (and reports the drift), --no-asset-check skips it
   entirely. Pick one."
      fi
      ASSET_POLICY="$1"; shift;;
    --env) ENVS+=("$2"); shift 2;;
    --artifact) ARTIFACTS+=("$2"); shift 2;;
    --cc-allow) CC_ALLOW="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --offer) OFFER="$2"; shift 2;;
    --machine) MACHINE="$2"; shift 2;;
    --eval-env-ver) EEV_FLAG="$2"; shift 2;;
    --keep) KEEP=1; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,175p' "${BASH_SOURCE[0]}"; exit 0;;
    -*) die "unknown flag $1";;
    *) [ -z "$BUNDLE" ] || die "one bundle dir, got '$BUNDLE' and '$1'"
       BUNDLE="$1"; shift;;
  esac
done

[ -n "$BUNDLE" ] || die "usage: launch_jobs_box.sh <bundle-dir> [flags]"
[ -d "$BUNDLE" ] || die "no such bundle dir: $BUNDLE"
case "$TYPE" in ondemand|bid) :;; *) die "--type must be ondemand|bid";; esac

# `set -a` matters: a bare `. .env` sets SHELL variables, which the python
# subprocesses below never see — the tracks preflight then reports "B2_BUCKET
# not set … proceeding without the check", i.e. it silently degrades to no
# provenance check at all. Export or it did not happen.
#
# An already-exported environment WINS over the file — sourcing unconditionally
# would clobber a caller who deliberately set these (tests, CI, a second
# bucket), and silently pointing a launch at the wrong bucket is the worst kind
# of quiet.
# LAUNCH_JOBS_NO_ENV_FILE=1 skips the file entirely (CI / tests / a caller who
# supplies credentials some other way). The path is script-relative, so `cd`
# alone cannot opt out.
if [ -f "$REPO/.env" ] && [ "${LAUNCH_JOBS_NO_ENV_FILE:-0}" != "1" ] \
   && { [ -z "${VASTAI_API_KEY:-}" ] || [ -z "${B2_BUCKET:-}" ]; }; then
  set -a; . "$REPO/.env"; set +a
fi
[ -n "${VASTAI_API_KEY:-}" ] || die "VASTAI_API_KEY unset (source the repo .env)"
[ -n "${B2_BUCKET:-}" ] || die "B2_BUCKET unset — the tracks preflight would silently skip"

# --- which submit surface does this bundle use? ------------------------------
if [ -f "$BUNDLE/matrix.py" ]; then
  # `jobmatrix.py submit` has no --env: a matrix arm's env comes from matrix.py,
  # which is the point of a matrix. Refuse rather than drop the flag — a
  # silently-ignored --env launches the wrong experiment on a rented box.
  [ ${#ENVS[@]} -eq 0 ] || die "--env applies to a single-arm bundle; this one is
   a matrix — put the value in matrix.py, which is where an arm's env belongs."
  # Same refusal, same reason: --artifact composes ENV, and jobmatrix.py submit
  # has no --artifact either, so accepting it would drop the registry-composed
  # prefixes and launch the arms against whatever the bundle's `env:` says.
  [ ${#ARTIFACTS[@]} -eq 0 ] || die "--artifact applies to a single-arm bundle; this
   one is a matrix — an arm's artifact env belongs in matrix.py."
  SUBMIT=(python3 "$HERE/jobmatrix.py" submit "$BUNDLE")
  [ -n "$ONLY" ] && SUBMIT+=(--only "$ONLY")
  [ -n "$ASSET_POLICY" ] && SUBMIT+=("$ASSET_POLICY")
  SURFACE="matrix"
elif [ -f "$BUNDLE/job-config.yaml" ] || [ -f "$BUNDLE/job-config.json" ]; then
  [ -z "$ONLY" ] || die "--only applies to a matrix bundle; this one is single-arm"
  SUBMIT=("${VCTL[@]}" job submit "$BUNDLE")
  [ -n "$ASSET_POLICY" ] && SUBMIT+=("$ASSET_POLICY")
  # --env pins fold onto the bundle's `env:` at submit and travel in the TICKET,
  # not in the bundle — so they are part of both the step-1 preflight and the
  # real submit, and `${SUBMIT[*]}` under --dry-run prints them. Without this
  # passthrough the only way to run a bundle in a submit-time variant was to
  # rent by hand and call `herdd job submit --env` afterwards, i.e. to give up
  # the one ordering this script exists to enforce.
  # --artifact BEFORE --env, matching `job submit`'s own fold order: the
  # registry composes first and a raw --env of the same key deliberately wins.
  for _pv in ${ARTIFACTS[@]+"${ARTIFACTS[@]}"}; do SUBMIT+=(--artifact "$_pv"); done
  for _kv in ${ENVS[@]+"${ENVS[@]}"}; do SUBMIT+=(--env "$_kv"); done
  SURFACE="single"
else
  die "$BUNDLE has neither matrix.py nor job-config.yaml"
fi
note "bundle: $BUNDLE ($SURFACE)"

# --- STEP 1a: local checks first (no network, no API, instant) ---------------
# Ordered cheapest-first deliberately: a card count that cannot work is worth
# knowing before we spend even a B2 round-trip on it, and it keeps these checks
# testable without credentials.
#
# --- read the bundle's own needs, to size the box honestly --------------------
# NOTE the deliberate absence of a `try/except: pass` here. An earlier draft
# swallowed the exception and fell through to defaults, which meant a typo in
# this block presented as "the bundle asks for nothing in particular" — the box
# got hand-typed sizing and the DDP divisibility check silently never ran. A
# bundle we cannot read is a STOP, not a shrug.
read -r NEED_RAM NEED_GPUS NEED_GA NEED_VENV NEED_EEV NEED_CC \
        NEED_HOST_RAM NEED_CORES < <(
  BUNDLE_DIR="$BUNDLE" VAST_TOOLS="$HERE" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["VAST_TOOLS"])
import jobmeta
cfg = jobmeta.load_job_config(os.environ["BUNDLE_DIR"])
needs = cfg.get("needs") or {}
env = cfg.get("env") or {}
# One line, eight fields, "-" for absent — `read -r` cannot represent an empty
# field positionally. None of these values may contain whitespace; an env
# version is a bake id, `needs.venv` is a keyword, and the sm allowlist is
# comma-joined ("80,86,89,90") because that is also the `--cc-allow` spelling.
# An EMPTY cc_allow prints "-": absent and empty both mean UNCONSTRAINED, and
# the one thing this must never become is an allowlist of nothing.
raw_cc = needs.get("cc_allow") or []
if isinstance(raw_cc, (str, bytes)):
    raise SystemExit("needs.cc_allow must be a list of sm levels, e.g. [80, 90]")
cc = []
for s in raw_cc:
    t = str(s).strip().lower().removeprefix("sm_").removeprefix("sm")
    if not t.isdigit():
        raise SystemExit(f"needs.cc_allow entry {s!r} is not an sm level")
    cc.append(t)
print(needs.get("gpu_ram_gb") or "-", needs.get("gpus") or "-",
      env.get("GRAD_ACCUM") or "-", needs.get("venv") or "-",
      str(env.get("EVAL_ENV_VER") or "-").strip() or "-",
      ",".join(cc) or "-",
      needs.get("host_ram_gb") or "-", needs.get("cpu_cores") or "-")
PY
) || die "could not read $BUNDLE's job-config — refusing to size a box by guess"
# --gpu-ram vs needs.gpu_ram_gb: a filter is not a contract.
#
# Until 2026-08-14 an explicit --gpu-ram simply WON — no comparison, no warning
# — so `--gpu-ram 23` against a bundle declaring 30 rented a 24 GB card whose
# ticket jobd was guaranteed to refuse (it fails a ticket when the largest card
# is smaller than needs.gpu_ram_gb, and a 4090's 24082 MiB rounds to 24). The
# box booted, jobd started, the ticket never ran, and the meter did.
#
# The two numbers are NOT the same knob:
#   --gpu-ram          OFFER SELECTION. A vast search filter, workstation-side,
#                      spent once at rent time.
#   needs.gpu_ram_gb   the TICKET's contract. jobd re-checks it against the
#                      cards it actually finds and REFUSES the job on a box that
#                      does not meet it. Lowering the filter cannot lower this.
# So a --gpu-ram below the need does not buy a cheaper run — it buys a box that
# will not run the job. Refuse BEFORE the rent, which is the only place it is
# still free. Over-provisioning (--gpu-ram above the need) stays allowed: that
# is a legitimate "give me headroom" and the ticket is satisfied either way.
if [ -n "$GPU_RAM" ] && [ "$NEED_RAM" != "-" ] \
   && awk -v a="$GPU_RAM" -v b="$NEED_RAM" 'BEGIN{exit !(a+0 < b+0)}'; then
  if [ "$FORCE_GPU_RAM" = "1" ]; then
    note "--force-gpu-ram: renting for $GPU_RAM GB/card while $BUNDLE declares
   needs.gpu_ram_gb: $NEED_RAM. jobd will still refuse the ticket on a card
   below $NEED_RAM — you are buying a box, not a run."
  else
    die "--gpu-ram $GPU_RAM is BELOW $BUNDLE's needs.gpu_ram_gb: $NEED_RAM.
   --gpu-ram only selects the OFFER; needs.gpu_ram_gb is the TICKET's contract,
   and jobd re-checks it ON the box — it fails the ticket when the largest card
   is smaller than the declared need. Renting under it gets you a paid box whose
   job never starts (and jobd's refusal is quiet: the box just looks idle).
   Pick one:
     - drop --gpu-ram          use the bundle's own $NEED_RAM (the default)
     - fix the bundle          if $NEED_RAM is wrong, edit job-config.yaml's
                               needs.gpu_ram_gb — and make the config that fits
                               (e.g. QUANT/MAX_SEQ) fit too, or you have only
                               moved the false fit into the bundle
     - --force-gpu-ram         rent anyway, knowing the ticket will be refused"
  fi
fi
[ -z "$GPU_RAM" ] && [ "$NEED_RAM" != "-" ] && GPU_RAM="$NEED_RAM"

# --- CPU shape: the bundle's own floors become search filters ----------------
# The CPU-side twin of the block above, minus its refusal. These two are pure
# OFFER SELECTION with no ticket contract behind them — jobd has no host-RAM or
# core gate to refuse against — so an explicit flag simply wins and there is
# nothing to compare it to. If a box-side gate is ever added, the "a filter is
# not a contract" guard above is the shape to copy.
[ -z "$HOST_RAM" ] && [ "$NEED_HOST_RAM" != "-" ] && HOST_RAM="$NEED_HOST_RAM"
[ -z "$CPU_CORES" ] && [ "$NEED_CORES" != "-" ] && CPU_CORES="$NEED_CORES"

# --- the architecture allowlist: the BUNDLE declares it, the flag overrides ---
# See "THE ARCHITECTURE ALLOWLIST" in the header. The bundle is the durable
# statement (it is a property of the workload's kernels, not of one launch), so
# it is the DEFAULT here; an explicit --cc-allow wins, and says so, because
# silently overriding a declared constraint is the exact failure this exists to
# stop.
CC_SRC=""
if [ -n "$CC_ALLOW" ]; then
  CC_SRC="--cc-allow"
  if [ "$NEED_CC" != "-" ] && [ "$CC_ALLOW" != "$NEED_CC" ]; then
    note "--cc-allow $CC_ALLOW OVERRIDES $BUNDLE's declared needs.cc_allow:
   $NEED_CC. The box (and every eviction replacement of it) is held to YOUR
   list, not the bundle's. If the bundle's declaration is wrong, fix it there —
   a flag lasts one launch, the declaration lasts every relaunch."
  fi
elif [ "$NEED_CC" != "-" ]; then
  CC_ALLOW="$NEED_CC"; CC_SRC="bundle needs.cc_allow"
fi
if [ -n "$CC_ALLOW" ]; then
  note "arch allowlist: sm $CC_ALLOW (from $CC_SRC) -> --cc-allow + LAUNCH_CC_ALLOW"
  # An EXPLICIT --gpu is an AND with the allowlist. When the named card's
  # silicon is outside the list the search cannot return anything, and vast
  # reports that as an empty market — so refuse HERE, where the operator can
  # still see which two filters disagree, rather than after a confusing search.
  # Unknown card names do NOT refuse: this table cannot know every SKU, and a
  # gate that fires on ignorance is one nobody reads.
  if [ "$GPU_EXPLICIT" = 1 ]; then
    CC_CONFLICT="$(VAST_TOOLS="$HERE" GPU_ALIAS="$GPU" CC="$CC_ALLOW" python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["VAST_TOOLS"])
from vastlib.market import offers
bad = offers.gpu_alias_conflicts(os.environ["GPU_ALIAS"],
                                 offers.parse_cc_allow(os.environ["CC"]))
print(",".join(f"sm_{s}" for s in bad))
PY
    )" || die "could not check --gpu $GPU against the sm allowlist $CC_ALLOW"
    [ -z "$CC_CONFLICT" ] || die "--gpu $GPU is $CC_CONFLICT, which the active sm
   allowlist ($CC_ALLOW, from $CC_SRC) EXCLUDES. The two filters are an AND, so
   the offer search would return nothing and report it as an empty market.
   Pick one:
     - drop --gpu               let the allowlist and the price policy choose
     - name an in-list card     e.g. --gpu a100 / --gpu h100 for sm 80/90
     - fix needs.cc_allow       if the bundle's declaration is what is wrong"
  fi
fi

# --- STEP 1a-bis: resolve the eval-env pin (see "THE EVAL-ENV PIN" above) -----
# Only `needs.venv: eval` bundles have a fetch to steer. For everything else
# this is silent and injects nothing.
if [ "$NEED_VENV" = "eval" ]; then
  if [ -n "$EEV_FLAG" ]; then
    if [ "$NEED_EEV" != "-" ] && [ "$NEED_EEV" != "$EEV_FLAG" ]; then
      die "--eval-env-ver $EEV_FLAG conflicts with the bundle's own pin $NEED_EEV.
   The box would fetch '$EEV_FLAG' while the job asserts '$NEED_EEV', and the
   submit's M4 gate refuses that combination anyway (EVAL_ENV_VER CONFLICT) —
   after the box is rented. Drop the flag to use the bundle's pin, or change
   the bundle if the pin is what moved."
    fi
    EEV_PIN="$EEV_FLAG"; EEV_SRC="--eval-env-ver"
  elif [ "$NEED_EEV" != "-" ]; then
    EEV_PIN="$NEED_EEV"; EEV_SRC="bundle job-config env:"
    EEV_SAME_AS_BUNDLE=1
  else
    # No pin anywhere. `herdd job submit`'s M4 gate refuses this too, but it
    # refuses in STEP 3 — one step after the rent. Refuse here, for $0.
    die "$BUNDLE declares needs.venv: eval but names no EVAL_ENV_VER, and none
   was passed. An unpinned box resolves eval-env/LATEST at boot, which CAN be
   older (or newer) than the env the bundle was preflighted against — that is
   how wave A graded its FLOOR on pre-fix code. Name the version:
     list:  rclone lsf b2:\$B2_BUCKET/eval-env/
     then:  $0 $BUNDLE --eval-env-ver <ver>
   or put it in the bundle's env: block, which is the durable form."
  fi
  # The SAME string in both places, derived from ONE variable. The box env is
  # what steers fetch_eval_env.sh; the job env is what run.sh asserts against
  # the loaded manifest and records in the artifacts. jobmeta accepts either as
  # a pin and REFUSES when they disagree, so deriving both from EEV_PIN makes
  # that conflict unrepresentable rather than merely detected.
  #
  # It is also what makes --eval-env-ver usable at all on a bundle that carries
  # no `env:` pin: STEP 1b preflights against placeholder box 0, whose launch
  # env is unreadable by construction, so without this the M4 gate would see no
  # pin anywhere and refuse a launch that was in fact correctly pinned.
  if [ "${EEV_SAME_AS_BUNDLE:-0}" != "1" ]; then
    SUBMIT+=(--env "EVAL_ENV_VER=$EEV_PIN")
  fi
  note "eval-env pin: $EEV_PIN (from $EEV_SRC) -> box env EVAL_ENV_VER"
fi

# Whole-box bundle? Then the card count is the DDP world size, and the planner
# refuses a count that does not divide GRAD_ACCUM. Warn loudly rather than
# quietly re-picking --num-gpus for the operator.
if [ "$NEED_GPUS" = "all" ] && [ "$NEED_GA" != "-" ]; then
  if [ $(( NEED_GA % NUM_GPUS )) -ne 0 ]; then
    die "--num-gpus $NUM_GPUS does not divide the bundle's GRAD_ACCUM $NEED_GA.
   This bundle claims every card (needs.gpus: all), so the card count IS the DDP
   world size, and launch_plan.sh will exit 12 on the box rather than round the
   effective batch. Pick a card count that divides $NEED_GA."
  fi
  note "shape: whole-box, DDP over $NUM_GPUS cards (grad-accum ${NEED_GA} -> $(( NEED_GA / NUM_GPUS ))/rank, eff-batch held)"
fi

# --- STEP 1a-bis: every selected arm's DATA_FILE must be present AND match ---
# The bundler ships whatever is in data/ and says nothing about what is not.
# That is how a worktree-submitted bundle shipped NO corpora on 2026-08-06:
# every preflight below passed, the box rented, and the arm died at rc=13
# (doc 79 §6, ~$0.71). The corpora are gitignored by design — they are copied
# in at submit time — so "present in git" is no evidence at all, and the only
# honest check is on disk, here, before anything is rented.
#
# BOTH surfaces. A single-arm bundle declares DATA_FILE/EXPECT_SHA256 in
# job-config.yaml's env rather than in matrix.py variants, and its data/ is
# gitignored the same way — an unguarded single-arm launch has exactly the same
# rent-then-rc=13 failure shape. The config is read through jobmeta's loader
# (the same one `job submit`/`run-local` use), never hand-parsed.
#
# Measured on 2026-08-08: with the corpus present the bundle hashed 3,392,186 B,
# without it 2,814,005 B. Both built. Only the byte count differed, which is
# why this is a check and not something an operator is expected to notice.
CORPUS_NOTE="$(BUNDLE_DIR="$BUNDLE" VAST_TOOLS="$HERE" ONLY_GLOB="$ONLY" python3 - <<'PY'
import fnmatch, hashlib, os, sys
sys.path.insert(0, os.environ["VAST_TOOLS"])
d, only = os.environ["BUNDLE_DIR"], os.environ.get("ONLY_GLOB") or "*"


def arms():
    if os.path.isfile(os.path.join(d, "matrix.py")):
        import jobmatrix
        for arm in jobmatrix.expand(jobmatrix.load_experiment(d)):
            yield arm.name, (arm.env or {})
    else:
        import jobmeta
        cfg = jobmeta.load_job_config(d)
        yield str(cfg.get("name") or "single-arm"), (cfg.get("env") or {})


bad = []
matched = sha_ok = presence_only = no_data_file = 0
for name, env in arms():
    if not fnmatch.fnmatch(name, only):
        continue
    matched += 1
    data_file = env.get("DATA_FILE")
    if not data_file:
        no_data_file += 1
        continue
    p = os.path.join(d, "data", data_file)
    if not os.path.exists(p):
        bad.append(f"{name}: data/{data_file} is ABSENT — the bundle would ship "
                   f"without it and the arm would die on the box")
        continue
    want = env.get("EXPECT_SHA256")
    if not want:
        presence_only += 1
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        bad.append(f"{name}: data/{data_file} sha256 {h.hexdigest()[:16]}… "
                   f"but the arm pins {want[:16]}…")
    else:
        sha_ok += 1
for line in bad:
    print("!! " + line, file=sys.stderr)
if bad:
    sys.exit(1)
# The success note must say what was actually checked — "sha-matched" when some
# arms carry no EXPECT_SHA256, or when the glob matched nothing, is a lie.
if matched == 0:
    print(f"0 arms matched '{only}' — nothing checked (the dry-run submit "
          "below still refuses an empty selection)")
else:
    parts = []
    if sha_ok:
        parts.append(f"{sha_ok} sha-matched")
    if presence_only:
        parts.append(f"{presence_only} present, no EXPECT_SHA256 (presence-only)")
    if no_data_file:
        parts.append(f"{no_data_file} with no DATA_FILE (skipped)")
    print(f"{matched} arm(s) for '{only}': " + ", ".join(parts))
PY
)" || die "arm corpus check FAILED — nothing rented."
note "arm corpora: $CORPUS_NOTE"

# --- STEP 1b: preflight the bundle with NO BOX -------------------------------
# Box id 0 is a placeholder: --dry-run writes nothing and never calls the vast
# API, so this validates + hashes the bundle and runs the tracks provenance
# check while we still owe nobody any money.
note "preflight (no box rented yet — a refusal here costs \$0)"
if ! "${SUBMIT[@]}" --box 0 --dry-run; then
  die "bundle preflight FAILED — nothing rented. Fix the bundle (a tracks
   mismatch means re-running the runset's build.sh to restage B2), then retry."
fi

# --- container disk: measured from the bundle, not hand-typed ----------------
# Storage bills on the ALLOCATED size, so an idle over-allocation is a real
# standing cost (measured 2026-07-30 at $2.13-4.62/day/box). An asset we cannot
# size is reported UNKNOWN rather than counted as zero.
#
# THE ESTIMATE IS COMPUTED EVEN WHEN --disk IS EXPLICIT, and that is the point
# of this block since 2026-08-25. Until then an explicit --disk skipped the
# estimator entirely: `--disk 40` against a bundle whose merge needs 80 rented
# the box, pulled the ~19 GB image, pulled the base, booted vLLM, passed a 12/12
# positive control and THEN died rc 5 on the bundle's own pre-merge disk guard.
# Every one of those steps costs money; the comparison costs an rclone size.
# Same shape as the --gpu-ram gate above, same escape hatch (--force-disk).
read -r DISK_AUTO DISK_EST DISK_NOTE < <(
  BUNDLE_DIR="$BUNDLE" VAST_TOOLS="$HERE" B2_BUCKET="$B2_BUCKET" python3 - <<'PY'
import json, os, subprocess, sys
sys.path.insert(0, os.environ["VAST_TOOLS"])
import disksize, jobmeta
d, bucket = os.environ["BUNDLE_DIR"], os.environ["B2_BUCKET"]
FALLBACK = 80
sizes = {}
cfg = jobmeta.load_job_config(d)      # raises -> the shell `||` below refuses
for a in cfg.get("assets") or []:
    sub = a.get("b2")
    if not sub:
        continue
    try:
        out = subprocess.run(["rclone", "size", "--json", f"b2:{bucket}/{sub}"],
                             capture_output=True, text=True, timeout=120)
        if out.returncode == 0:
            sizes[a["name"]] = int(json.loads(out.stdout).get("bytes") or 0)
    except Exception:
        pass
gb, b = disksize.estimate_disk_gb(cfg, sizes)
unk = b.get("unknown_assets") or []
# Two numbers, deliberately different:
#   AUTO  what to rent when nobody said. FALLBACK covers an unsized asset,
#         where the derivation is missing a term we KNOW exists.
#   EST   the derivation itself — a true lower bound in every branch, and so
#         the only figure it is sound to refuse an explicit --disk against.
#         Refusing against FALLBACK would be refusing against a guess.
auto = max(int(gb), FALLBACK) if unk else int(gb)
# The label names what the number SAW. "measured-from-assets" was true and read
# as complete; a reader cannot act on a number without knowing its blind spots.
if b.get("declared_disk_gb"):
    label = "declared by needs.disk_gb"
elif b.get("scratch_gb"):
    label = "assets+venv+overhead+scratch"
else:
    label = "assets+venv+overhead; blind to what the entrypoint writes (no needs.scratch_gb)"
if unk:
    label += "; UNSIZED: " + ",".join(unk)
print(auto, int(gb), label)
PY
) || die "could not size the container disk from $BUNDLE — pass --disk"

if [ -z "$DISK" ]; then
  DISK="$DISK_AUTO"
  note "disk: ${DISK}GB (${DISK_NOTE})"
elif awk -v a="$DISK" -v b="$DISK_EST" 'BEGIN{exit !(b+0 > 0 && a+0 < b+0)}'; then
  if [ "$FORCE_DISK" = "1" ]; then
    note "--force-disk: renting ${DISK}GB against a ${DISK_EST}GB estimate
   (${DISK_NOTE}). Nothing on the box will resize a disk mid-run."
  else
    die "--disk $DISK is BELOW $BUNDLE's own estimate of ${DISK_EST}GB
   (${DISK_NOTE}). What that estimate saw is in the label, and it is a LOWER
   BOUND on every branch — with no needs.scratch_gb it cannot even see what the
   entrypoint writes. Renting under it buys a box that dies partway through,
   after the image pull, the asset pull and however much of the run got there.
   Pick one:
     - drop --disk             use the bundle's own ${DISK_AUTO}GB (the default)
     - fix the bundle          if the estimate is too high, edit
                               job-config.yaml — needs.scratch_gb ADDS to the
                               derivation, needs.disk_gb REPLACES it
     - --force-disk            rent under it anyway"
  fi
else
  note "disk: ${DISK}GB (explicit; estimate ${DISK_EST}GB — ${DISK_NOTE})"
fi

# --- assemble the launch -----------------------------------------------------
LAUNCH=("${VCTL[@]}" launch --jobs --fleet-watch
        --num-gpus "$NUM_GPUS" --disk "$DISK" --type "$TYPE" --ssh)
# --machine SEARCHES restricted to one machine instead of pinning an offer id.
# It exists because vast's offer `id` filter returns zero rows for live offers
# in every view, so `--offer` cannot be auto-priced and dies at the bid floor
# (autobid is the design; hand-pricing is not the fix). `--machine` keeps the
# server-side filters enforced AND keeps same-host comparability, which is the
# only reason to pin at all. Mutually exclusive with --offer.
if [ -n "$MACHINE" ] && [ -n "$OFFER" ]; then
  echo "error: --machine and --offer are mutually exclusive" >&2; exit 2
fi
# A named --gpu ON TOP of --machine is an AND, not a hint: the two filters
# intersect to nothing whenever the machine holds a different card, which is
# the normal case when the machine was chosen for CPU cores or VRAM rather
# than by card name. Naming the machine already narrows the search to one host,
# so let its card be whatever it is unless the caller said --gpu explicitly.
#
# An active sm allowlist narrows the search the same way, and the DEFAULT card
# name on top of it is the same AND-to-nothing: the shipped default is h100
# (sm_90), so a bundle declaring `cc_allow: [80, 86, 89]` would search for an
# H100 that the allowlist then drops — zero offers, reported as an empty market.
# So a declared/passed allowlist DROPS the default name and lets cc_allow + the
# preferred-GPU tier policy + price pick the card. Unlike the --machine case
# this does NOT pass --any-gpu: --machine is an operator hardware pin, an
# allowlist is not, so the default preferred-GPU policy stays on and keeps
# auto-pick on cards we actually run work on. An explicit --gpu still wins, and
# was already checked against the allowlist above.
if   [ -n "$MACHINE" ]; then
  LAUNCH+=(--machine "$MACHINE")
  if [ "$GPU_EXPLICIT" = 1 ]; then LAUNCH+=(--gpu "$GPU"); else LAUNCH+=(--any-gpu); fi
elif [ -n "$OFFER" ];   then LAUNCH+=(--offer "$OFFER")
elif [ -n "$CC_ALLOW" ] && [ "$GPU_EXPLICIT" != 1 ]; then
  note "no --gpu: the sm allowlist ($CC_ALLOW) picks the architecture, price
   picks the card (the default --gpu $GPU would AND with it)."
else                         LAUNCH+=(--gpu "$GPU"); fi
[ -n "$GPU_RAM" ] && LAUNCH+=(--gpu-ram "$GPU_RAM")
[ -n "$HOST_RAM" ] && LAUNCH+=(--host-ram "$HOST_RAM")
[ -n "$CPU_CORES" ] && LAUNCH+=(--cpu-cores "$CPU_CORES")
# Stamped as LAUNCH_CC_ALLOW on the box, so a fleetd eviction replacement stays
# inside the same sm allowlist (a compiled-for-one-arch cubin makes this
# tighter than the runtime attention gate can express).
[ -n "$CC_ALLOW" ] && LAUNCH+=(--cc-allow "$CC_ALLOW")
[ -n "$MAX_DPH" ] && LAUNCH+=(--max-dph "$MAX_DPH")
[ -n "$IMAGE" ] && LAUNCH+=(--image "$IMAGE")
for g in ${GEOS+"${GEOS[@]}"}; do LAUNCH+=(--geo "$g"); done
# The pin the box boots on. `launch --env K=V` lands it in vast's extra_env,
# which is the container env fetch_eval_env.sh reads at provision time.
[ -n "$EEV_PIN" ] && LAUNCH+=(--env "EVAL_ENV_VER=$EEV_PIN")

# ...and the assertion that it actually got there. The M4 gate re-reads the
# instance's extra_env at submit; --require-box-eval-pin turns its job-pin-only
# NOTE into a refusal, which is the correct verdict HERE (we rented this box
# seconds ago, so it is cold and nothing has provisioned /workspace/eval yet)
# and would be wrong for a hand-resubmit onto a warm box.
#
# jobmatrix has no EVAL_ENV_VER gate of its own, so the readback is only
# available on the single-job surface. The INJECTION above covers both; say so
# rather than implying a check that did not run.
SUBMIT_STRICT=()
if [ -n "$EEV_PIN" ]; then
  if [ "$SURFACE" = "single" ]; then
    SUBMIT_STRICT=(--require-box-eval-pin)
  else
    note "note: the eval-env pin is injected, but jobmatrix submit has no M4
   gate, so it is NOT read back off the instance. Confirm by hand if it matters."
  fi
fi

if [ "$DRY" -eq 1 ]; then
  note "[dry-run] would launch: ${LAUNCH[*]}"
  note "[dry-run] would submit: ${SUBMIT[*]} --box <IID> ${SUBMIT_STRICT[*]-}"
  note "[dry-run] would watch : ${VCTL[*]} fleet watch <IID> --profile jobs --budget $BUDGET ${WATCH_STANDING_ARGS[*]+"${WATCH_STANDING_ARGS[*]}"}"
  exit 0
fi

# --- STEP 2: rent ------------------------------------------------------------
# HERDD_WATCH_HINT=0: `launch --jobs` prints a "that watch is BARE, arm the
# ladder" nudge for the hand-run path. Step 4 below IS that command, so here the
# nudge would be an alarm firing on the recommended workflow.
note "renting: ${LAUNCH[*]}"
LAUNCH_OUT="$(HERDD_WATCH_HINT=0 "${LAUNCH[@]}" 2>&1 | tee /dev/stderr)"
IID="$(printf '%s\n' "$LAUNCH_OUT" | sed -n 's/^launched instance \([0-9][0-9]*\).*/\1/p' | tail -1)"
[ -n "$IID" ] || die "could not parse an instance id out of the launch output.
   The box MAY be running — check '${VCTL[*]} ls' before retrying, and destroy
   any box this script cannot name."
note "instance $IID (bare fleet watch registered at launch)"

# From here on a failure leaves a RENTED box, so every exit path has to say so.
trap 'echo "!! failed after renting $IID — it is still billing." >&2;
      echo "!!   inspect: ${VCTL[*]} show $IID" >&2;
      echo "!!   park:    ${VCTL[*]} stop $IID" >&2;
      echo "!!   destroy: ${VCTL[*]} destroy $IID" >&2' ERR

# --- STEP 3: submit (do NOT wait for ready — see the header) -----------------
note "submitting tickets to $IID"
"${SUBMIT[@]}" --box "$IID" ${SUBMIT_STRICT[@]+"${SUBMIT_STRICT[@]}"}

# --- STEP 4: upgrade the watch to the spend-capable jobs ladder --------------
WATCH=("${VCTL[@]}" fleet watch "$IID" --profile jobs --budget "$BUDGET"
       "${WATCH_STANDING_ARGS[@]+"${WATCH_STANDING_ARGS[@]}"}")
[ "$KEEP" -eq 1 ] && WATCH+=(--keep)
if "${WATCH[@]}"; then
  note "fleetd: watching $IID (profile=jobs budget=\$$BUDGET$([ "${#WATCH_STANDING_ARGS[@]}" -gt 0 ] && echo " standing"))"
else
  echo "!! fleetd did NOT take the watch — the box is rented and UNGOVERNED." >&2
  echo "!!   register it: ${WATCH[*]}" >&2
  echo "!!   or park it:  ${VCTL[*]} stop $IID" >&2
  exit 1
fi
trap - ERR

cat <<EOF

box $IID is rented, queued, and fleet-watched.
  progress : ${VCTL[*]} job ls --box $IID
  arms     : python3 $HERE/jobmatrix.py status <EXP_ID>
  spend    : ${VCTL[*]} fleet status
  boot     : ${VCTL[*]} wait $IID --state running --timeout 900
fleetd parks it when the queue drains (or on the \$$BUDGET cap). It does NOT
destroy — a parked box still bills its allocated disk, so run
'${VCTL[*]} ls' when you are done and destroy what you no longer need.
EOF
