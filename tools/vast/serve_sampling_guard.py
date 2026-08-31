#!/usr/bin/env python3
"""Refuse to send `min_p` / `logit_bias` to a server that is speculating.

vLLM warns once at startup ("min_p and logit_bias parameters won't work with
speculative decoding") and then REJECTS each request that carries either:
`SamplingParams._validate_spec_decode` raises per request, an HTTP 400 on the
API server (verified on the fork at the pin, sampling_params.py:887). So the
failure is loud — but it lands MID-RUN, after the box is rented and the sweep
is under way, and a retry loop can misread a 400 as a transient. This guard
moves the same refusal to PREFLIGHT, before any spend, with the fix named. It
became reachable on 2026-08-27 when MTP flipped ON by default on the
`launch_serve.sh` path (owner directive; run of record
`<upstream-bench>/archive/runs/2026-08-27-v14-lora-mtp/`; mechanism + the
unlanded fork fix: docs/plans/witness/MINP_UNDER_SPEC_DECODE_2026-08-27.md).

The launcher cannot do this check: it never sees a request. So the guard is
client-side, and it asserts the ENGINE rather than the flag — `/metrics` carries
`vllm:spec_decode_*` series when speculative decoding is on and exactly zero of
them when it is off (measured, same run). A caller that reasons from "I did not
pass --mtp" is reasoning from the wrong end of a default that just moved.

Fail-closed: an unreadable `/metrics` is UNKNOWN, and unknown refuses. A guard
that waves requests through whenever it cannot see is not a guard.

Usage:

    from serve_sampling_guard import assert_sampling_compatible
    assert_sampling_compatible(base_url, params, api_key=key)   # raises or returns

    # or from a shell, before a sweep:
    python3 tools/vast/serve_sampling_guard.py http://127.0.0.1:8000 --param min_p=0.05
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

#: Sampling parameters vLLM rejects per-request under speculative decoding.
#: Both are named in the engine's own startup warning; do not add anything here
#: that the engine does not actually refuse.
SPEC_INCOMPATIBLE_PARAMS = ("min_p", "logit_bias")

#: Values that mean "not requested". `min_p=0.0` is vLLM's own default and is
#: indistinguishable from omitting it, so it must not trip the guard — a guard
#: that fires on every request is one people disable.
_INERT = (None, 0, 0.0, "", {}, [])

#: The discriminator. Any series with this prefix means the engine built a
#: spec-decode metrics group, which it does only when speculating.
_SPEC_METRIC_PREFIX = "vllm:spec_decode_"


class SamplingGuardError(RuntimeError):
    """A request would have lost a sampling parameter without saying so."""


def _metrics_url(base_url: str) -> str:
    """`/metrics` sits at the server root, not under the OpenAI `/v1` prefix."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/metrics"


def fetch_metrics(base_url: str, *, api_key: str | None = None,
                  timeout: float = 10.0) -> str | None:
    """Prometheus text from the served engine, or None if it cannot be read."""
    req = urllib.request.Request(_metrics_url(base_url))
    if api_key:
        req.add_header("Authorization", "Bearer %s" % api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def spec_decode_series(metrics_text: str) -> list[str]:
    """Distinct `vllm:spec_decode_*` series names present in the scrape."""
    names = set()
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.startswith(_SPEC_METRIC_PREFIX):
            names.add(name)
    return sorted(names)


def spec_decode_engaged(base_url: str, *, api_key: str | None = None,
                        timeout: float = 10.0) -> bool | None:
    """True / False / None-for-unknown. None is a refusal input, not a pass."""
    text = fetch_metrics(base_url, api_key=api_key, timeout=timeout)
    if text is None:
        return None
    return bool(spec_decode_series(text))


def incompatible_params(params: dict) -> list[str]:
    """Which of the caller's sampling params the engine would refuse."""
    return [k for k in SPEC_INCOMPATIBLE_PARAMS
            if k in params and params[k] not in _INERT]


def assert_sampling_compatible(base_url: str, params: dict, *,
                               api_key: str | None = None,
                               timeout: float = 10.0,
                               allow_unknown: bool = False) -> None:
    """Raise SamplingGuardError if the engine would refuse `params` mid-run.

    Cheap-path first: a request that sets neither parameter cannot lose one, so
    it never costs an HTTP round trip.
    """
    at_risk = incompatible_params(params)
    if not at_risk:
        return
    engaged = spec_decode_engaged(base_url, api_key=api_key, timeout=timeout)
    if engaged is False:
        return
    if engaged is None and allow_unknown:
        return
    why = ("is SPECULATING" if engaged
           else "could not be read (/metrics unreachable) — UNKNOWN, which refuses")
    raise SamplingGuardError(
        "refusing to send %s: the server at %s %s, and vLLM rejects these "
        "parameters under speculative decoding — every such request 400s "
        "mid-run. Failing here, before any spend, instead.\n"
        "  Fix ONE of:\n"
        "    * relaunch the serve with `--mtp 0` (tools/vast/launch_serve.sh) "
        "and re-run; or\n"
        "    * drop %s from the sampling params and say so in the readout — "
        "the diversity lever is not in force.\n"
        "  MTP defaults ON since 2026-08-27; not passing `--mtp 1` no longer "
        "means it is off."
        % (", ".join(at_risk), base_url, why, ", ".join(at_risk)))


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base_url", help="e.g. http://127.0.0.1:8000 (or .../v1)")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="sampling param to check, repeatable (e.g. min_p=0.05)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--allow-unknown", action="store_true",
                    help="treat an unreadable /metrics as a pass (default: refuse)")
    a = ap.parse_args(argv)

    key = a.api_key
    if key is None and a.api_key_file:
        with open(a.api_key_file) as fh:
            key = fh.read().strip()

    params = {}
    for kv in a.param:
        k, _, v = kv.partition("=")
        try:
            params[k] = json.loads(v)
        except ValueError:
            params[k] = v

    engaged = spec_decode_engaged(a.base_url, api_key=key, timeout=a.timeout)
    print("spec_decode_engaged: %s" % {True: "YES", False: "no",
                                       None: "UNKNOWN"}[engaged])
    try:
        assert_sampling_compatible(a.base_url, params, api_key=key,
                                   timeout=a.timeout,
                                   allow_unknown=a.allow_unknown)
    except SamplingGuardError as exc:
        print("!! %s" % exc, file=sys.stderr)
        return 1
    print("OK: %s" % (", ".join(sorted(params)) or "no params to check"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
