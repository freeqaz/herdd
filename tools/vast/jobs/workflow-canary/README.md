**Hub:** [vast tooling](../../README.md) › [jobs](../../INDEX_JOBS.md) — the real-GPU canary that the of-record image still runs

# workflow-canary (M4-T2)

A bounded, real-GPU canary bundle. It certifies that the **of-record image**
actually runs on a GPU box before a long workflow stage commits real spend, and
emits a **reusable receipt** the control plane stores (keyed, TTL-bounded).

## What it does on the box

1. Activates the baked of-record train env (`needs.venv: none`).
2. Checkpoint round-trip: writes `out/ckpt/roundtrip.txt` and reads it back
   (jobd also flushes that glob to B2 every `checkpoint_s`).
3. GPU model + version probe: `nvidia-smi` name, and `torch` / `transformers` /
   `bitsandbytes` / `vllm` versions + CUDA, via a python probe.
4. One real torch **GPU step** (a `256×256` cuda matmul) — proves the image's
   torch+CUDA actually compute. Fail-closed (`rc=21`) if no working GPU step,
   unless `CANARY_ALLOW_NO_GPU=1` (rehearsal on the CPU-only box image).
5. Writes `out/canary-receipt.json` **last** (written-last doctrine — it is not
   a `checkpoints:` glob). jobd uploads `out/**` to `jobs/<JOB_ID>/results/`.

## Receipt

`kind: workflow-canary-receipt`. Body carries the composite `key`, its
`components` (`image_digest, jobd_sha, model_manifest_sha, adapter_manifest_sha,
recipe_sha`), `gpu_model`, `versions`, `gpu_step_ok`, `checkpoint_roundtrip_ok`,
timestamps, `rc`, `ttl_s`, and `expires_ts`.

The **control plane** (`workflowctl.py`) computes the key + components, passes
them in via `--env CANARY_*`, then reads the receipt back from results and
re-stores it at `workflow-canary/receipts/<key>.json`. A workflow's online plan
refuses to spend without a valid, unexpired receipt for its exact key
(`failure_class: CANARY_MISSING | CANARY_EXPIRED`).

## Fidelity (v1)

Real image + version-probe + GPU step + checkpoint round-trip. It does **not**
stage model/adapter weights or run a full gen/train step — that is M5-T3. The
model/adapter/recipe SHA components describe the *stage the receipt gates*, not
what this canary executed.

## Rehearse (CPU-only, no spend)

```bash
tools/vast/rehearse.sh tools/vast/jobs/workflow-canary --image
# rehearsal passes CANARY_ALLOW_NO_GPU=1 so the missing GPU is non-fatal
```
