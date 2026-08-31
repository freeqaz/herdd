"""Functional smoke for onstart/eval_job_lib.sh — the sourceable eval-phase lib.

No GPU, no B2, no real vLLM: the stdlib stub OpenAI server
(runsets/modelzoo-reader/eval_stub_server.py, which already emits DIVERGENT
per-model output) stands in for the serve endpoint, and a tiny bash driver
sources the lib and calls `ejl_gate`. Exercises the readiness gate's three
checks: all-expected-models-present, the 1-token probe, and the optional
divergence guard. Skipped when bash/curl are unavailable.
"""
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(_HERE, "onstart", "eval_job_lib.sh")
STUB = os.path.join(_HERE, "runsets", "modelzoo-reader", "eval_stub_server.py")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(LIB) and os.path.exists(STUB)
         and shutil.which("bash") and shutil.which("curl")),
    reason="needs bash + curl + the stub server")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Stub:
    def __init__(self, models, port):
        self.proc = subprocess.Popen(
            [sys.executable, STUB, "--port", str(port), "--models", models],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.port = port
        # wait for the port to answer
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("stub server never came up")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _run_gate(port, expect, env=None):
    base = f"http://127.0.0.1:{port}/v1"
    e = dict(os.environ)
    e["EJL_BASE_URL"] = base
    e["EJL_EXPECT_MODELS"] = expect
    e["EJL_GATE_TIMEOUT"] = "6"
    if env:
        e.update(env)
    return subprocess.run(
        ["bash", "-c", f'source {LIB}; ejl_gate; exit $?'],
        env=e, capture_output=True, text=True, timeout=60)


def test_ejl_gate_passes_when_all_models_present():
    stub = _Stub("nanbeige-base,reader", _free_port())
    try:
        r = _run_gate(stub.port, "nanbeige-base,reader")
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
        assert "[ejl]" in r.stdout and "PASSED" in r.stdout
    finally:
        stub.close()


def test_ejl_gate_fails_7_when_model_missing():
    stub = _Stub("nanbeige-base", stub_port := _free_port())
    try:
        r = _run_gate(stub_port, "nanbeige-base,reader")   # reader absent
        assert r.returncode == 7, f"rc={r.returncode} stdout={r.stdout} stderr={r.stderr}"
        assert "never all present" in r.stderr
    finally:
        stub.close()


def test_ejl_gate_divergence_passes_on_distinct_models():
    stub = _Stub("nanbeige-base,reader", _free_port())
    try:
        r = _run_gate(stub.port, "nanbeige-base,reader",
                      env={"EJL_DIVERGENCE": "nanbeige-base=reader"})
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
        assert "divergence OK" in r.stdout
    finally:
        stub.close()


def test_ejl_gate_divergence_fails_6_on_identical_ids():
    # same id both sides => identical greedy output => silent-no-op signature
    stub = _Stub("nanbeige-base,reader", _free_port())
    try:
        r = _run_gate(stub.port, "nanbeige-base",
                      env={"EJL_DIVERGENCE": "nanbeige-base=nanbeige-base"})
        assert r.returncode == 6, f"rc={r.returncode} stdout={r.stdout} stderr={r.stderr}"
        assert "DIVERGENCE FAIL" in r.stderr
    finally:
        stub.close()
