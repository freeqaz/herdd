#!/usr/bin/env python3
"""eval_stub_server — a tiny OpenAI-compatible stub for DRY_RUN / local rehearsal
of the modelzoo-reader on-box eval. Stdlib only; NO model, NO GPU, NO network out.

It exists so `train.sh` under DRY_RUN=1 (and `rehearse.sh --stub-vllm`) can
exercise the ENTIRE eval flow — readiness poll (serve_ready.sh), the 1-token
completion probe, the divergence gate (divergence_smoke.sh), p0a_run_probe.py
transport + NDJSON, p0a_grade.py summary — with a stubbed "server" in place of
vLLM. The answer is NOT meant to be correct — DRY_RUN validates plumbing, not
capability.

DIVERGENT per-model output (key property, added 2026-07-12): responses are
SEEDED BY MODEL ID, so base vs adapter answers differ reproducibly. The old
stub returned byte-identical answers for every model, which silently defeated
the divergence gate (divergence_smoke.sh compares base vs LoRA output on the
same endpoint — identical text == "silent no-op adapter", the [[qwen36-27b-sft-
plan]] failure). Here the served answer carries a per-model `seed` field and
per-model prediction values, so the gate bites in rehearsal exactly as it would
against real weights. Determinism: same (model, prompt) -> same bytes, always.

Endpoints:
  GET  /health                 -> {"status": "ok"}                (liveness)
  GET  /v1/models              -> {"data":[{"id": <served-model>}, ...]}
  POST /v1/chat/completions    -> one choice; message.content is a JSON-object
                                  string (F1 predictions), prediction count read
                                  from the prompt's `input[i] =` lines. Supports
                                  stream:true (minimal SSE: a few delta chunks +
                                  [DONE]).
  POST /v1/completions         -> one choice with a `text` field (serve_ready.sh's
                                  1-token probe uses this). Supports stream:true.

Auth: if --api-key is set, requests without a matching Bearer token get 401
(mirrors serve_vllm.sh's --api-key, so the eval's auth path is exercised too).
/health is unauthenticated (probes hit it before a token is configured).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_INPUT_RE = re.compile(r"input\[\d+\]\s*=", re.MULTILINE)
_BOOL_HINT = "boolean output (0 or 1)"


def _model_seed(model: str) -> int:
    """Stable 32-bit seed from a model id — the divergence source. Different
    served model ids (base vs adapter) yield different seeds -> different bytes."""
    return int(hashlib.sha256((model or "stub").encode("utf-8")).hexdigest()[:8], 16)


def _answer_for(model: str, prompt: str) -> str:
    """The JSON-object answer string, DIVERGENT by model id. Prediction COUNT is
    read from the prompt (grader well-formedness); prediction VALUES and the
    trailing `seed` field vary by model so divergence_smoke.sh sees base != LoRA
    even when the divergence prompt carries zero `input[i] =` lines."""
    n_preds = len(_INPUT_RE.findall(prompt))
    is_bool = _BOOL_HINT in prompt
    seed = _model_seed(model)
    preds: list = []
    for i in range(n_preds):
        if is_bool:
            preds.append((seed >> (i % 31)) & 1)
        else:
            preds.append(round(((seed >> (i % 24)) % 1000) / 100.0, 2))
    # `seed` guarantees divergence even for n_preds == 0 (bare divergence prompt).
    return json.dumps({"family": "F1", "predictions": preds, "seed": seed})


class _Handler(BaseHTTPRequestHandler):
    api_key = ""
    served_models = ("stub-base", "reader")

    def log_message(self, fmt, *args):  # quiet
        pass

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {self.api_key}"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_sse(self, chunks: list) -> None:
        """Minimal SSE: one `data: <json>` line per chunk, then `data: [DONE]`.
        Each chunk is an already-shaped OpenAI streaming object."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for ch in chunks:
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except Exception:  # noqa: BLE001
            return {}

    def do_GET(self):  # noqa: N802
        p = self.path.rstrip("/")
        if p.endswith("/health"):
            self._send(200, {"status": "ok"})
            return
        if p.endswith("/v1/models") or p.endswith("/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": m, "object": "model"}
                                      for m in self.served_models]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        p = self.path.rstrip("/")
        is_chat = p.endswith("/chat/completions")
        is_comp = p.endswith("/completions") and not is_chat
        if not (is_chat or is_comp):
            self._send(404, {"error": "not found"})
            return
        if not self._authorized():
            self._send(401, {"error": "invalid api key"})
            return
        req = self._read_json()
        model = req.get("model", "stub")
        stream = bool(req.get("stream"))

        if is_chat:
            prompt = ""
            for msg in req.get("messages", []) or []:
                if msg.get("role") == "user":
                    prompt = str(msg.get("content", ""))
            content = _answer_for(model, prompt)
            if stream:
                self._send_sse(self._chat_stream(model, content))
                return
            self._send(200, {
                "id": "stub-0",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": max(1, len(prompt) // 4),
                          "completion_tokens": 8, "cost": 0.0},
            })
            return

        # legacy /v1/completions — serve_ready.sh's 1-token probe (prompt="ping",
        # max_tokens=1). Returns `text`, seeded by model like chat.
        prompt = str(req.get("prompt", ""))
        text = _answer_for(model, prompt)
        if stream:
            self._send_sse(self._comp_stream(model, text))
            return
        self._send(200, {
            "id": "stub-0",
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "text": text}],
            "usage": {"prompt_tokens": max(1, len(prompt) // 4),
                      "completion_tokens": 8, "cost": 0.0},
        })

    @staticmethod
    def _split3(s: str) -> list:
        """Split a string into up to 3 non-empty pieces for a few SSE chunks."""
        if not s:
            return [""]
        k = max(1, len(s) // 3)
        return [x for x in (s[:k], s[k:2 * k], s[2 * k:]) if x] or [s]

    def _chat_stream(self, model: str, content: str) -> list:
        pieces = self._split3(content)
        chunks = [{
            "id": "stub-0", "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        }]
        for pc in pieces:
            chunks.append({
                "id": "stub-0", "object": "chat.completion.chunk", "model": model,
                "choices": [{"index": 0, "delta": {"content": pc}}],
            })
        chunks.append({
            "id": "stub-0", "object": "chat.completion.chunk", "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
        return chunks

    def _comp_stream(self, model: str, text: str) -> list:
        chunks = []
        for pc in self._split3(text):
            chunks.append({
                "id": "stub-0", "object": "text_completion", "model": model,
                "choices": [{"index": 0, "text": pc, "finish_reason": None}],
            })
        chunks.append({
            "id": "stub-0", "object": "text_completion", "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        })
        return chunks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible stub for DRY_RUN eval")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--models", default="stub-base,reader",
                    help="comma-separated served model ids advertised on /v1/models")
    args = ap.parse_args(argv)
    _Handler.api_key = args.api_key
    _Handler.served_models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[stub] serving {args.host}:{args.port} models={_Handler.served_models} "
          f"auth={'on' if args.api_key else 'off'}", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
