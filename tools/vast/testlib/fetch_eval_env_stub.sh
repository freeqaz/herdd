#!/usr/bin/env bash
# fetch_eval_env_stub.sh — rehearsal stand-in for onstart/fetch_eval_env.sh.
#
# jobd's check_venv self-provisions `needs.venv: eval` by running
# fetch_eval_env.sh, which pulls the multi-GB baked eval env (game trees + wibo +
# MSVC-PPC cl + mwcceppc + dtk + objdiff-cli) from B2 with real credentials. In a
# LOCAL rehearsal there are no credentials and no baked env, so every `venv: eval`
# bundle used to die pre-entrypoint with
# `needs.venv=eval: fetch_eval_env.sh provisioning failed` — the bundle's own
# control flow never ran, which is the entire point of rehearsing.
#
# This stub creates the ONE thing check_venv looks for ($JOBD_ROOT/eval/env.sh,
# empty) and nothing else. It is wired in ONLY by `rehearse.sh --stub-eval-env`
# (via JOBD_FETCH_EVAL_SH, the documented test seam) — never on a box.
#
# WHAT THIS DOES NOT CERTIFY (pin, don't simulate — same rule as --stub-vllm):
# there is no game tree, no compiler, no objdiff-cli and no ninja behind it. An
# entrypoint that would touch the real tree must take its own DRY_RUN path in a
# rehearsal; the live env is proven by the on-box S0 gates (warm build + scorer
# smoke), not here.
set -euo pipefail

ROOT="${JOBD_ROOT:-/workspace}"
mkdir -p "$ROOT/eval"
cat >"$ROOT/eval/env.sh" <<EOF
# REHEARSAL STUB eval env — written by testlib/fetch_eval_env_stub.sh.
# No toolchain, no game trees. Never present on a real box.
export HERDD_EVAL_ENV_STUB=1
EOF
echo ">> [stub] wrote $ROOT/eval/env.sh (rehearsal only — no toolchain, no trees)" >&2
