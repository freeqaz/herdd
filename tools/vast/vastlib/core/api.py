"""vastlib.core.api — the vast.ai HTTP kernel: one funnel, one retry policy.

Why this module exists
----------------------
Every byte this tooling exchanges with vast.ai goes through `request_soft`.
It is the single seam the whole tree binds to: 61 `monkeypatch.setattr(<mod>,
"request_soft", ...)` sites across the suite, four live sibling importers
(`fleetd`, `workflowctl`, `hosts`, `boxstate`), and — most importantly — the
autouse conftest fixture `_block_mutating_api_calls`, which wraps this one
function to stop the unit suite from issuing a real PUT against the live fleet
with the repo `.env` key. Pulling it out of `herdd.py` first is what lets
every ring above it be imported, typed and tested without dragging 20k lines of
CLI along.

Five names, and the shape of each is a contract:

* `request_soft` — non-exiting HTTP with transient retry. Shape A
  (`core.result.Soft`: `(ok, data, err)`). `data` is parsed JSON on success
  (`{}` for an empty body, the raw `str` when the body is not JSON) and `None`
  on failure; `err` is the semi-structured, string-matched channel described in
  `core/result.py` ("HTTP <code> …" / "network …" / "error …" / "config: …").
* `request` — the raising twin, four lines over `request_soft`, which
  `sys.exit`s on `not ok`. It is what makes one-shot CLI commands survive a
  blip yet still die loudly on a fatal. Ported verbatim, exit and all.
* `_classify_http` — "transient" (retry) vs "fatal" (stop), over an int, an
  exception, or the err STRING `request_soft` returns. The string arm is the
  reason `err` is not a typed object: this function parses `HTTP (\\d{3})` back
  out of prose, and routes a `"config:"` prefix to fatal so a missing API key
  is never retried forever as a network blip.
* `_api_key_soft` — shape D (`core.result.ValueErr`: `(value, err)`, data
  FIRST, mirrored against shape B). Reads `VASTAI_API_KEY`, then the
  undocumented-but-live `VAST_API_KEY` fallback. Both names are kept: the
  fallback appears in no error string and in no doc, and is read by real
  environments.
* `api_key` — its exiting twin, for the CLI paths that predate the soft form.

The retry policy is deliberately INLINE in `request_soft` and not a helper:
full-jitter exponential backoff, `cap = min(30.0, 0.5 * (2 ** attempt))`,
`_sleep(random.uniform(0, cap))`, evaluated only when the failure classifies
transient AND attempts remain. Extracting it would put the one thing four tests
assert on (attempt count, sleep count, "sleeps before, not after, success")
behind an indirection with no second caller.

What is deliberately NOT here
-----------------------------
* **No `VastClient` class yet.** Plan §5 sketches one with module-level
  default-client functions kept "precisely so the patch idiom survives". This
  port lands the module-level functions only, because the conftest live-fleet
  guard patches a MODULE ATTRIBUTE: the moment a caller can hold its own client
  instance, the guard covers a path the caller no longer takes, and the failure
  mode is a real PUT against a real box rather than a red test. A class is
  additive later — but only together with a guard that binds to the class (or
  a proof that the module-level function is the sole path for every mutating
  helper). Introducing it in the same step as the move would have made a
  behavior-preserving port into a safety change.
* **No general rate limiter, and no token bucket.** Bid pacing stays
  `bidpolicy.BID_RATE_LIMIT_S` and the dash-cache keeps its own submission gate
  (`DASH_MARKET_MAX_RPS`), both owned by their callers. What DOES live here now
  is a single-endpoint gate on `v0/bundles/` — see `_bundles_pace`: that one
  endpoint has a self-reported 5 req/s ceiling and enough independent callers
  (fleetd's per-box floor + on-demand reads, the launch search, the dash
  survey) that no caller can see the aggregate rate. It is a min-interval
  gate on the FUNNEL, which is the only place the aggregate exists.
* **No API base-URL override.** `API` is a hardcoded module constant with no
  env rung anywhere in the tree. Tests that need a different host patch
  `urlopen`, not the URL.
* **No `config` import.** `api_key`/`_api_key_soft` read `os.environ`
  directly, exactly as they did in `herdd.py`. The `.env` is folded into
  `os.environ` once by `config.load_env()` at CLI startup, so the dependency is
  on the PROCESS environment, not on this package's config module — routing the
  read through an accessor would add a rung to a precedence that currently has
  none. Both rows stay listed in `config.ENV_SITES_TODO`; plan §9 owns them.
* **No boot/deadline backoff.** `_boot_deadline_backoff` widens a boot-SLA
  deadline; it is not HTTP retry and travels with the box-health cluster.
* **No second urllib importer.** `boxes.remote`'s exec result-poller is the
  only other one in the package, deliberately (it fetches a PRE-SIGNED URL with
  no auth header, so it cannot go through this funnel).

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 2,
2026-08-16), the contiguous `config / http` block. Behavior-preserving: bodies
copied, annotations and `core.result` types added, nothing else — the returned
NamedTuples compare and unpack identically to the bare tuples they replace.
Every symbol carries its `# moved-from:` marker (grammar: `vastlib/README.md`
§2). The flat `herdd.py` copies stay live until step 6, so both are callable
during the port and every existing `herdd.request_soft` patch keeps steering
`herdd`'s own callers.
"""

from __future__ import annotations

import json
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from vastlib.core import acctfault, result

# moved-from: herdd.API
API = "https://console.vast.ai/api"


# moved-from: herdd.api_key
def api_key() -> str:
    k = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY")
    if not k:
        sys.exit("error: VASTAI_API_KEY not set (env or .env)")
    return k


# moved-from: herdd._api_key_soft
def _api_key_soft() -> result.ValueErr:
    """Like api_key() but NON-exiting: returns (key, err). A missing key is a
    fatal-config condition the supervisor loop must see as fatal (exit), not a
    transient (retry) — but it must NOT sys.exit mid-loop."""
    # The `VAST_API_KEY` fallback is undocumented (it appears in neither error
    # string) and live. Order matters: VASTAI_API_KEY wins.
    k = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY")
    if not k:
        return result.ValueErr(None, "config: VASTAI_API_KEY not set (env or .env)")
    return result.ValueErr(k, None)


# moved-from: herdd._classify_http
def _classify_http(err: object) -> str:
    """'transient' (retry) vs 'fatal' (stop) for a request_soft failure. Accepts
    an int status, an exception, or the err STRING request_soft returns.
      transient: URLError / socket.timeout / HTTP 408 / 429 / 5xx  -> a blip
      fatal:     missing key / HTTP 400 / 401 / 403 / 404          -> a bug/auth
    Unknown -> 'transient' (safest: retry-then-degrade, never a false fatal exit).
    INVARIANT: an API outage classifies transient, so it can never drive an
    eviction/relaunch or a terminal fold."""
    code = None
    if isinstance(err, bool):
        return "transient"
    if isinstance(err, int):
        code = err
    elif isinstance(err, urllib.error.HTTPError):
        code = err.code
    elif isinstance(err, (urllib.error.URLError, socket.timeout, TimeoutError)):
        return "transient"
    elif isinstance(err, str):
        if err.startswith("config:"):
            return "fatal"
        if err.startswith(("network ", "error ")):   # URLError/timeout/socket
            return "transient"
        m = re.search(r"HTTP (\d{3})", err)
        if m:
            code = int(m.group(1))
    if code is None:
        return "transient"
    if code in (408, 429) or code >= 500:
        return "transient"
    return "fatal"                                    # 400/401/403/404 (and other 4xx)


# --------------------------------------------------------------------------- #
# the one paced endpoint
# --------------------------------------------------------------------------- #
# vast rate-limits `v0/bundles/` at 5 req/s and says so in the 429 body
# (`'limit': 5.0`). 4/s leaves headroom for a second process (a CLI search
# running beside the daemon) without needing to coordinate with it.
BUNDLES_MAX_RPS = 4.0
_BUNDLES_PACE_LOCK = threading.Lock()
# Module-level MUTABLE state carried across calls: the whole point is that the
# NEXT caller sees the previous send. A list so the lock guards one object.
_BUNDLES_LAST_SEND = [0.0]
# Diagnostics only — `fleet status` surfaces these so a pacer that is actually
# costing the tick wall-clock is visible without a per-pace journal line.
_BUNDLES_PACE_STATS = {"sends": 0, "waits": 0, "waited_s": 0.0}


def _is_bundles(path: str) -> bool:
    return path.lstrip("/").startswith("v0/bundles")


def _bundles_pace(_sleep: Callable[[float], object] = time.sleep) -> None:
    """Block until this caller may send its bundles request, holding EVERY
    caller in this process to `BUNDLES_MAX_RPS` in aggregate.

    Paces, never drops: a skipped market read blinds the eviction/decay arms,
    and `request_soft`'s backoff only hides a 429 after it has already cost the
    read. Guards the initial send only — a retry is already jitter-backed off.
    One lock, one monotonic stamp: the fleetd tick is a single thread issuing a
    serial burst, so there is no burst to smooth and nothing a token bucket
    would buy."""
    gap = 1.0 / BUNDLES_MAX_RPS if BUNDLES_MAX_RPS > 0 else 0.0
    with _BUNDLES_PACE_LOCK:
        now = time.monotonic()
        wait = _BUNDLES_LAST_SEND[0] + gap - now
        if wait > 0:
            # Bounded by construction: `wait <= gap` unless the clock jumped,
            # and a monotonic clock does not.
            wait = min(wait, gap)
            _sleep(wait)
            now += wait
            _BUNDLES_PACE_STATS["waits"] += 1
            _BUNDLES_PACE_STATS["waited_s"] = round(
                _BUNDLES_PACE_STATS["waited_s"] + wait, 3)
        _BUNDLES_LAST_SEND[0] = now
        _BUNDLES_PACE_STATS["sends"] += 1


def bundles_pace_stats() -> dict[str, float]:
    """Snapshot of the bundles gate: sends, how many of them waited, and the
    total wall-clock it cost. Cumulative for the life of the process."""
    with _BUNDLES_PACE_LOCK:
        return dict(_BUNDLES_PACE_STATS)


# moved-from: herdd.request_soft
def request_soft(method: str, path: str,
                 body: Any = None,                    # noqa: ANN401 — arbitrary JSON
                 timeout: int = 60,
                 retries: int = 5,
                 _sleep: Callable[[float], object] = time.sleep) -> result.Soft:
    """Non-exiting HTTP with transient-retry. Returns (ok, data, err).

    Retries transient failures (network/timeout/408/429/5xx) up to `retries`
    times with full-jitter exponential backoff (cap 30s); returns immediately on
    a fatal (missing key / 400 / 401 / 403 / 404). `request()` delegates here, so
    one-shot CLI survives a blip yet still sys.exits on a fatal. retries=0 gives
    single-shot behavior for callers that manage their own loop. On success err
    is None and data is parsed JSON ({} for empty body, raw str if not JSON)."""
    key, kerr = _api_key_soft()
    if kerr:
        return result.Soft(False, None, kerr)         # fatal-config; loop-safe
    data = json.dumps(body).encode() if body is not None else None
    url = f"{API}/{path.lstrip('/')}"
    err: str | None = None
    if _is_bundles(path):
        # NOT `_sleep`: that name belongs to the retry backoff, and four tests
        # count its calls.
        _bundles_pace()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
            if not raw:
                return result.Soft(True, {}, None)
            try:
                return result.Soft(True, json.loads(raw), None)
            except json.JSONDecodeError:
                return result.Soft(True, raw, None)
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                msg: Any = json.loads(raw)
            except Exception:
                msg = raw[:300]
            err = f"HTTP {e.code} on {method} {path}: {msg}"
            kind = _classify_http(e.code)
        except urllib.error.URLError as e:
            err = f"network {e} on {method} {path}"
            kind = "transient"
        except Exception as e:                          # socket.timeout, etc.
            err = f"error {e} on {method} {path}"
            kind = "transient"
        if kind == "fatal" or attempt >= retries:
            # LATCH an account-level refusal here and nowhere else. This is the
            # only funnel every lane shares, and the seams that must not blame a
            # host for it (a strike, an alarm's remedy) are nowhere near the
            # call that saw it — see `core.acctfault`.
            acctfault.note(err)
            return result.Soft(False, None, err)
        cap = min(30.0, 0.5 * (2 ** attempt))           # full-jitter exp backoff
        _sleep(random.uniform(0, cap))
    return result.Soft(False, None, err)


# moved-from: herdd.request
def request(method: str, path: str,
            body: Any = None,                         # noqa: ANN401 — arbitrary JSON
            timeout: int = 60) -> Any:                # noqa: ANN401 — mirrors Soft.data
    ok, data, err = request_soft(method, path, body, timeout)
    if not ok:
        sys.exit(f"error: {err}")
    return data
