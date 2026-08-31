#!/usr/bin/env bash
# Toy EVAL-shaped entrypoint (rehearsal fixture, N6). Proves the full local lane
# end-to-end WITHOUT a GPU or B2:
#   - the declarative `assets:` block staged the LoRA adapter at ./adapter (a
#     dest symlink into the box asset cache) — we read it to prove staging ran
#     BEFORE the entrypoint;
#   - the stub vLLM (rehearse --stub-vllm) answers /v1/models, /v1/completions,
#     and /v1/chat/completions;
#   - we emit an NDJSON `probe` row per served model (a `checkpoints:` glob) plus
#     a summary.json — both under the `results:` glob, so they land on the bucket.
# A real reader-eval entrypoint has this exact shape (serve gate -> probe ->
# grade); here the grading is trivial so the rehearsal stays deterministic.
set -euo pipefail
: "${STUB_ENDPOINT:?rehearse must run with --stub-vllm — STUB_ENDPOINT is unset}"
mkdir -p out

# 1) asset staged? read the adapter the assets: block placed at ./adapter. If the
#    asset never staged this path is missing and the job fails loudly here.
adapter_name="$(python3 -c 'import json; print(json.load(open("adapter/adapter_config.json"))["base_model"])')"

# 2) serve reachable + advertised models
curl -fsS "$STUB_ENDPOINT/v1/models" > out/models.json

# 3) probe each model over BOTH completion surfaces; one NDJSON row each
: > out/probe.ndjson
IFS=',' read -ra MODELS <<< "${STUB_MODELS:-stub-base,reader}"
n=0
for m in "${MODELS[@]}"; do
  [ -n "$m" ] || continue
  comp="$(curl -fsS -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"prompt\":\"ping\",\"max_tokens\":1}" \
    "$STUB_ENDPOINT/v1/completions")"
  chat="$(curl -fsS -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"weigh in\"}]}" \
    "$STUB_ENDPOINT/v1/chat/completions")"
  python3 - "$m" "$adapter_name" "$comp" "$chat" >> out/probe.ndjson <<'PY'
import json, sys
model, adapter, comp, chat = sys.argv[1:5]
text = json.loads(comp)["choices"][0].get("text", "")
content = json.loads(chat)["choices"][0]["message"]["content"]
json.dump({"model": model, "adapter": adapter,
           "completion": text, "chat": content}, sys.stdout)
sys.stdout.write("\n")
PY
  n=$((n + 1))
done

# 4) summary
python3 - "$n" "$adapter_name" <<'PY' > out/summary.json
import json, sys
json.dump({"n_models": int(sys.argv[1]), "adapter": sys.argv[2], "ok": True},
          sys.stdout)
PY
echo ">> eval-toy: probed $n model(s) with adapter=$adapter_name"
