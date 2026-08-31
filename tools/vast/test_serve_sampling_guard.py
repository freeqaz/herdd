#!/usr/bin/env python3
"""`min_p` / `logit_bias` must never be sent to a speculating server.

Two halves, and neither is sufficient alone:

* the RUNTIME guard (`serve_sampling_guard.py`) — asserts the engine via
  `/metrics`, refuses on unknown;
* the TRACKED-SOURCE gate below — a runtime helper nobody calls guards nothing,
  so any tracked file that puts one of those parameters into a sampling payload
  must reference the guard.

The source gate has **zero hits in the repo today**, which is the whole reason
it is written as a gate and not as a doc: `min_p` is a lever this project has
decided to use and has not yet wired, and the moment somebody wires it against
a default-ON-MTP serve the loss is invisible. A gate that has never fired is
indistinguishable from a broken one, so `test_source_gate_can_actually_fail`
feeds it a synthetic file and requires a hit.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import serve_sampling_guard as G

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# The runtime discriminator: assert the engine, not the flag
# --------------------------------------------------------------------------- #

_ON = """\
# HELP vllm:spec_decode_num_accepted_tokens_total Accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="qwen35-9b"} 41234.0
vllm:spec_decode_num_draft_tokens_total{model_name="qwen35-9b"} 44100.0
vllm:num_requests_running{model_name="qwen35-9b"} 0.0
"""

_OFF = """\
# HELP vllm:num_requests_running Number running.
vllm:num_requests_running{model_name="qwen35-9b"} 0.0
vllm:prompt_tokens_total{model_name="qwen35-9b"} 9001.0
"""


def test_spec_series_are_the_discriminator():
    assert G.spec_decode_series(_ON) == [
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
    ]
    assert G.spec_decode_series(_OFF) == []


def test_a_zeroed_counter_still_counts_as_engaged():
    """The series EXISTING is the signal. A window in which nothing was drafted
    reports zeros, and reading that as 'MTP off' would wave through exactly the
    idle-server case a pre-flight check runs in."""
    zeroed = _ON.replace("41234.0", "0.0").replace("44100.0", "0.0")
    assert G.spec_decode_series(zeroed)


def test_metrics_url_strips_the_openai_prefix():
    """`/metrics` is at the server root; callers hold a `.../v1` base URL."""
    assert G._metrics_url("http://h:8000/v1") == "http://h:8000/metrics"
    assert G._metrics_url("http://h:8000/") == "http://h:8000/metrics"


# --------------------------------------------------------------------------- #
# The refusal
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("params", [
    {},
    {"temperature": 0.6, "top_p": 0.95},
    {"min_p": 0.0},          # vLLM's own default — indistinguishable from unset
    {"logit_bias": {}},
])
def test_inert_params_never_refuse(params, monkeypatch):
    monkeypatch.setattr(G, "fetch_metrics", lambda *a, **k: _ON)
    G.assert_sampling_compatible("http://h:8000", params)


@pytest.mark.parametrize("params,expect", [
    ({"min_p": 0.05}, ["min_p"]),
    ({"logit_bias": {"13": -100}}, ["logit_bias"]),
    ({"min_p": 0.05, "logit_bias": {"13": -100}}, ["min_p", "logit_bias"]),
])
def test_refuses_when_the_engine_is_speculating(params, expect, monkeypatch):
    monkeypatch.setattr(G, "fetch_metrics", lambda *a, **k: _ON)
    with pytest.raises(G.SamplingGuardError) as ei:
        G.assert_sampling_compatible("http://h:8000", params)
    msg = str(ei.value)
    for name in expect:
        assert name in msg
    assert "--mtp 0" in msg, "the refusal must name the fix"


def test_passes_when_the_engine_is_not_speculating(monkeypatch):
    monkeypatch.setattr(G, "fetch_metrics", lambda *a, **k: _OFF)
    G.assert_sampling_compatible("http://h:8000", {"min_p": 0.05})


def test_unknown_refuses_by_default(monkeypatch):
    """Fail-closed. An unreachable /metrics is the state a guard is most likely
    to meet on a box mid-boot, and it is the state in which a pass is a lie."""
    monkeypatch.setattr(G, "fetch_metrics", lambda *a, **k: None)
    with pytest.raises(G.SamplingGuardError):
        G.assert_sampling_compatible("http://h:8000", {"min_p": 0.05})
    # ...and the opt-down exists, explicitly, for a caller that has other evidence
    G.assert_sampling_compatible("http://h:8000", {"min_p": 0.05},
                                 allow_unknown=True)


def test_a_clean_request_costs_no_round_trip(monkeypatch):
    """The guard sits on a hot path. Nothing at risk => no HTTP at all."""
    def _boom(*a, **k):
        raise AssertionError("fetch_metrics called for a request with no min_p")
    monkeypatch.setattr(G, "fetch_metrics", _boom)
    G.assert_sampling_compatible("http://h:8000", {"temperature": 0.6})


# --------------------------------------------------------------------------- #
# The tracked-source gate
# --------------------------------------------------------------------------- #

# Word-bounded so `min_pmi`, `_min_power_limit` and friends do not match. The
# spellings cover a Python kwarg/dict key, a shell/env var and a JSON field.
_PARAM_RE = re.compile(
    r"""(?<![A-Za-z0-9_])(min_p|logit_bias)(?![A-Za-z0-9_])["']?\s*[=:]""")

_GATE_EXTS = (".py", ".sh", ".json", ".yaml", ".yml")

#: Files allowed to name the parameter without calling the guard. Keep this
#: list short and reasoned — a stale exemption is how a gate becomes a hole.
_EXEMPT = {
    "tools/vast/serve_sampling_guard.py": "the guard itself",
    "tools/vast/test_serve_sampling_guard.py": "this test",
}


def _tracked_sources() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    rels = [r for r in out.stdout.split("\0") if r.endswith(_GATE_EXTS)]
    # Corpora and research papers are data, not code that can send a request.
    return [Path(r) for r in rels
            if not r.startswith(("data/", "docs/research/"))]


def test_source_gate_can_actually_fail():
    """The gate has no live hits, so prove it is not vacuous."""
    assert _PARAM_RE.search('payload = {"min_p": 0.05, "top_p": 0.95}')
    assert _PARAM_RE.search("resp = client.completions(min_p=0.05)")
    assert _PARAM_RE.search('{"logit_bias": {"13": -100}}')
    # ...and that it does not fire on the near-misses that made the naive
    # substring version of this unusable.
    assert not _PARAM_RE.search("kept = filter_merges_by_pmi(l, h, min_pmi=2.0)")
    assert not _PARAM_RE.search("def _min_power_limit(gpus):")


def test_no_tracked_source_sets_min_p_without_the_guard():
    """`min_p`/`logit_bias` requests are REFUSED (per-request 400) under
    speculative decoding, and MTP is ON by default on the launch_serve path
    since 2026-08-27. A caller that sets either must consult
    serve_sampling_guard (or pin `--mtp 0` and say so in the same file), so the
    refusal lands at preflight instead of mid-run."""
    offenders = []
    for rel in _tracked_sources():
        if str(rel) in _EXEMPT:
            continue
        try:
            text = (REPO / rel).read_text(errors="replace")
        except OSError:
            continue
        if not _PARAM_RE.search(text):
            continue
        if "serve_sampling_guard" in text or "--mtp 0" in text or "SERVE_MTP=0" in text:
            continue
        offenders.append(str(rel))
    assert not offenders, (
        "these files set min_p/logit_bias with no spec-decode guard: %s\n"
        "Either call serve_sampling_guard.assert_sampling_compatible() before "
        "the request, or serve with --mtp 0 and say so in the file."
        % ", ".join(sorted(offenders)))


def test_the_exemption_list_has_no_stale_entries():
    for rel in _EXEMPT:
        assert (REPO / rel).exists(), "stale exemption: %s" % rel
