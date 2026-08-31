"""`vastlib.core.api` — the HTTP kernel, characterized, plus the live-fleet guard.

Why this file exists
--------------------
Two jobs, and the second one is the important one.

1. **Characterization.** `_classify_http`'s err-prefix grammar, `_api_key_soft`'s
   two-name env precedence, `request_soft`'s retry/backoff/fatal-stop policy and
   `request`'s `sys.exit` are ported verbatim out of `herdd.py`. Everything
   here asserts what the ported copy ANSWERS, in its own right. It used to also
   assert equality with the still-live flat original on every input, which is
   how the add-only port was proved (both sides agreeing, rather than a patch
   site moving); plan §8 step 6d deleted the original, and those three parity
   sweeps went with it — see the deletion notes in §1, §2 and §5 and the
   surviving binding assertion in §8.

2. **The guard meta-test.** `conftest.py`'s autouse `_block_mutating_api_calls`
   is what stops this suite from issuing a real PUT against the live fleet with
   the repo `.env` key. It reaches its target through `sys.modules`, so a target
   that is renamed, moved or simply not listed does not fail — it silently stops
   being guarded, and the next test that drives a mutating path bills a real
   box. `vastlib.core.api.request_soft` was a SECOND copy of that funnel and was
   invisible to the fixture until it was added to
   `conftest._GUARDED_REQUEST_SOFT_MODULES`. After step 6d there is one body but
   still TWO guarded bindings — the fixture wraps each module attribute
   separately, and `herdd.request_soft` is the spelling `launch_serve.sh` and
   the five flat-module consumers reach — so both entries in that tuple stay. The tests at the bottom of this
   file assert (a) every listed target still resolves to a callable, and (b) a
   mutating call through the vastlib copy is actually refused. Plan §4 lists
   "conftest live-fleet guards keep biting" as a frozen contract; this is that
   contract, executable.

Network safety: nothing here can reach the network. Every test either patches
`urlopen` (the transport seam — `api.urllib.request` IS the stdlib module
object, so the patch is global and steers both copies) or unsets both API-key
env vars, and the one test that deliberately drives a PUT does BOTH, so its
assertion distinguishes "the guard refused" from "there was no key" instead of
depending on the guard to avoid a request.

Provenance: new in the vastlib package, plan §8 step 2 (`core/`). The expected
values are inherited from `test_supervise.py` §2a/§2b (unchanged there);
nothing was repointed into this file.
"""

from __future__ import annotations

import importlib
import inspect
import io
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conftest  # noqa: E402
import herdd  # noqa: E402
from vastlib.core import api, result  # noqa: E402


# --------------------------------------------------------------------------
# 1. _classify_http — the err-prefix grammar
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 599])
def test_classify_http_transient_int_codes(code):
    assert api._classify_http(code) == "transient"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_classify_http_fatal_int_codes(code):
    assert api._classify_http(code) == "fatal"


def test_classify_http_httperror_instance():
    transient = urllib.error.HTTPError("u", 503, "unavail", None, io.BytesIO(b""))
    fatal = urllib.error.HTTPError("u", 404, "missing", None, io.BytesIO(b""))
    assert api._classify_http(transient) == "transient"
    assert api._classify_http(fatal) == "fatal"


def test_classify_http_network_and_timeout_are_transient():
    assert api._classify_http(urllib.error.URLError("boom")) == "transient"
    assert api._classify_http(socket.timeout()) == "transient"
    assert api._classify_http(TimeoutError()) == "transient"


def test_classify_http_config_prefix_is_fatal():
    """The `config:` prefix is a TYPED CHANNEL, not prose (core/result.py, shape
    D). It is what stops a missing API key from being retried forever as a
    transient blip — the prefix match, not the message text."""
    assert api._classify_http("config: VASTAI_API_KEY not set") == "fatal"
    assert api._classify_http("config: anything at all") == "fatal"
    # ...and only as a PREFIX: the same word mid-string is not the channel.
    assert api._classify_http("bad config: whatever") == "transient"


def test_classify_http_string_forms():
    assert api._classify_http("HTTP 429 on GET x: rate limited") == "transient"
    assert api._classify_http("HTTP 401 on GET x: nope") == "fatal"
    assert api._classify_http("network ConnectionResetError on GET x") == "transient"
    assert api._classify_http("error timed out on GET x") == "transient"


def test_classify_http_status_is_regexed_out_of_the_prose():
    """`err` is a semi-structured channel: the status is parsed back out of the
    message request_soft itself formatted. Exactly three digits after 'HTTP '."""
    assert api._classify_http("HTTP 500 on PUT /v0/instances/1/: boom") == "transient"
    assert api._classify_http("HTTP 4041 on GET x") == "fatal"     # matches "404"
    assert api._classify_http("HTTP 40 on GET x") == "transient"   # no 3-digit match
    assert api._classify_http("prefixed HTTP 503 somewhere") == "transient"


def test_classify_http_bool_never_treated_as_an_http_code():
    # bool is an int subclass in Python; the guard must catch it explicitly
    assert api._classify_http(True) == "transient"
    assert api._classify_http(False) == "transient"


def test_classify_http_unknown_defaults_transient():
    # "safest: retry-then-degrade, never a false fatal exit"
    assert api._classify_http("something weird") == "transient"
    assert api._classify_http(None) == "transient"
    assert api._classify_http(object()) == "transient"


# `test_classify_http_parity_with_herdd` and its 37-probe `_CLASSIFY_CORPUS`
# lived here, sweeping both copies of the classifier for a drift on any single
# input. Plan §8 step 6d thinned `herdd.py`: `herdd._classify_http` is an
# identity re-export of `api._classify_http`, so the sweep ran one function
# twice and compared it with itself. Deleted — the ten characterization tests
# above own the err-prefix grammar, which is what a drift would have shown up
# as. The one statement that survives the thinning is the binding, asserted in
# `test_the_launcher_re_exports_rather_than_redefines` at the bottom of this
# file.


# --------------------------------------------------------------------------
# 2. _api_key_soft / api_key — env precedence, both names
# --------------------------------------------------------------------------

def test_api_key_soft_reads_vastai_api_key(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "primary")
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    assert api._api_key_soft() == ("primary", None)


def test_api_key_soft_falls_back_to_vast_api_key(monkeypatch):
    """The `VAST_API_KEY` fallback is UNDOCUMENTED — it appears in no error
    string and no doc — and live. Dropping it in the port would have broken
    every environment that only sets that name."""
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.setenv("VAST_API_KEY", "fallback")
    assert api._api_key_soft() == ("fallback", None)


def test_api_key_soft_precedence_vastai_wins(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "primary")
    monkeypatch.setenv("VAST_API_KEY", "fallback")
    assert api._api_key_soft().value == "primary"


def test_api_key_soft_empty_primary_falls_through(monkeypatch):
    """`or`, not `os.environ.get(a, os.environ.get(b))`: an EMPTY primary is
    falsy and falls through to the fallback. Preserved verbatim."""
    monkeypatch.setenv("VASTAI_API_KEY", "")
    monkeypatch.setenv("VAST_API_KEY", "fallback")
    assert api._api_key_soft() == ("fallback", None)


def test_api_key_soft_missing_is_a_typed_config_error(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    k, err = api._api_key_soft()
    assert k is None
    assert err == "config: VASTAI_API_KEY not set (env or .env)"
    # The coupling that makes a missing key a stop rather than a retry storm:
    assert api._classify_http(err) == "fatal"


def test_api_key_soft_shape_is_value_first_and_tuple_compatible(monkeypatch):
    """Shape D (`ValueErr`) — DATA first, the mirror of `(ok, err)`. Adopting
    the NamedTuple must not change unpacking, indexing or `==`."""
    monkeypatch.setenv("VASTAI_API_KEY", "primary")
    r = api._api_key_soft()
    assert isinstance(r, result.ValueErr)
    assert isinstance(r, tuple) and len(r) == 2
    assert r == ("primary", None)
    assert r[0] == "primary" and r[1] is None
    value, err = r
    assert (value, err) == ("primary", None)


# `test_api_key_soft_parity_with_herdd` swept the same six env shapes through
# both copies. Same story as the classifier: post-6d `herdd._api_key_soft` is
# this module's function, so the sweep compared it with itself. Deleted; the
# six characterization tests above cover exactly those shapes (primary,
# fallback, precedence, empty-primary fall-through, missing, tuple shape).


def test_api_key_returns_the_key(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "primary")
    assert api.api_key() == "primary"


def test_api_key_exits_when_unset(monkeypatch):
    """The exiting twin of `_api_key_soft`, ported verbatim including the exit."""
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        api.api_key()
    assert "VASTAI_API_KEY not set" in str(e.value)


# --------------------------------------------------------------------------
# 3. request_soft — the transport seam, retry, backoff, fatal-stop
# --------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(url, code, body=b""):
    return urllib.error.HTTPError(url, code, "boom", None, io.BytesIO(body))


def _raising_urlopen(exc_factory):
    """A fake `urlopen` that RAISES. Spelled out rather than inlined as a lambda
    because `lambda ...: _http_error(...)` *returns* the HTTPError — and an
    HTTPError is a context manager with a `.read()`, so the bug reads as a
    successful empty response instead of a failure."""
    def fake_urlopen(req, timeout=None):
        raise exc_factory(req)
    return fake_urlopen


def _unguarded(fn):
    """The real function behind conftest's guard wrapper.

    `_block_mutating_api_calls` is autouse, so during every test in this suite
    `api.request_soft` / `herdd.request_soft` are the fixture's `guarded`
    closure, not the ported function. Introspection tests need the original."""
    if getattr(fn, "__name__", "") != "guarded":
        return fn
    reals = [c.cell_contents for c in (fn.__closure__ or ())
             if callable(c.cell_contents)]
    assert len(reals) == 1, "guard closure shape changed — update _unguarded()"
    return reals[0]


def test_transport_seam_is_the_stdlib_module_object():
    """`api.urllib.request` IS `urllib.request`, the same object `herdd`
    reaches. That is why the suite's 13 existing
    `monkeypatch.setattr(herdd.urllib.request, "urlopen", ...)` sites need no
    repoint to steer the vastlib copy: they patch one global module attribute
    and both funnels see it."""
    assert api.urllib.request is urllib.request
    assert api.urllib.request is herdd.urllib.request


def test_request_soft_success_parses_json(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b'{"instances": []}'))
    r = api.request_soft("GET", "v1/instances/")
    assert isinstance(r, result.Soft)
    assert r == (True, {"instances": []}, None)


def test_request_soft_empty_body_is_an_empty_dict(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b""))
    assert api.request_soft("GET", "v1/instances/") == (True, {}, None)


def test_request_soft_non_json_body_comes_back_raw(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b"not json at all"))
    assert api.request_soft("GET", "v1/instances/") == (True, "not json at all", None)


def test_request_soft_builds_the_url_and_auth_header(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.headers)
        seen["method"] = req.get_method()
        seen["timeout"] = timeout
        return _FakeResp(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    api.request_soft("GET", "/v1/instances/", timeout=17)
    # leading slash stripped from path, single slash against the API constant
    assert seen["url"] == f"{api.API}/v1/instances/"
    assert api.API == "https://console.vast.ai/api"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["method"] == "GET"
    assert seen["timeout"] == 17


def test_request_soft_transient_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = {"n": 0}
    sleeps = []

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(req.full_url, 503)
        return _FakeResp(b'{"instances": []}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, data, err = api.request_soft("GET", "v1/instances/", retries=5,
                                     _sleep=lambda s: sleeps.append(s))
    assert ok is True
    assert data == {"instances": []}
    assert err is None
    assert calls["n"] == 3           # 2 failures + 1 success
    assert len(sleeps) == 2          # backed off before each retry, not after success
    assert all(s >= 0 for s in sleeps)


def test_request_soft_fatal_stops_immediately_no_retry(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(req.full_url, 401, b'{"msg":"bad key"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def must_not_sleep(s):
        pytest.fail("request_soft must not back off on a fatal (401)")

    ok, data, err = api.request_soft("GET", "v1/instances/", retries=5,
                                     _sleep=must_not_sleep)
    assert ok is False
    assert data is None
    assert "HTTP 401" in err
    assert calls["n"] == 1            # no retries for a fatal


def test_request_soft_404_is_also_fatal(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        _raising_urlopen(lambda req: _http_error(req.full_url, 404)))
    ok, data, err = api.request_soft("GET", "v1/instances/", retries=5,
                                     _sleep=lambda s: pytest.fail("no retry on 404"))
    assert ok is False and "HTTP 404" in err


def test_request_soft_exhausts_retries_and_returns_the_last_error(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = {"n": 0}
    sleeps = []

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, data, err = api.request_soft("GET", "v1/instances/", retries=3,
                                     _sleep=lambda s: sleeps.append(s))
    assert (ok, data) == (False, None)
    assert err.startswith("network ")
    assert calls["n"] == 4            # retries + 1 attempts
    assert len(sleeps) == 3           # one fewer sleep than attempts: none after the last
    assert api._classify_http(err) == "transient"


def test_request_soft_retries_zero_is_single_shot(monkeypatch):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, _, err = api.request_soft("GET", "v1/instances/", retries=0,
                                  _sleep=lambda s: pytest.fail("retries=0 must not sleep"))
    assert ok is False and calls["n"] == 1


def test_request_soft_backoff_is_full_jitter_capped_at_30s(monkeypatch):
    """`cap = min(30.0, 0.5 * (2 ** attempt))`, slept as `uniform(0, cap)`.

    `random.uniform` is pinned to its upper bound so the jitter window's CEILING
    is asserted exactly — the doubling, and the 30s clamp that stops attempt 6+
    from sleeping 32s, 64s, 128s."""
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        _raising_urlopen(lambda req: urllib.error.URLError("unreachable")))
    bounds = []
    monkeypatch.setattr(api.random, "uniform", lambda a, b: bounds.append((a, b)) or b)
    sleeps = []
    api.request_soft("GET", "v1/instances/", retries=8, _sleep=lambda s: sleeps.append(s))
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert all(lo == 0 for lo, _hi in bounds)          # full jitter: window starts at 0
    assert max(sleeps) == 30.0


def test_request_soft_missing_key_is_fatal_and_never_touches_network(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    called = {"n": 0}

    def fake_urlopen(req, timeout=None):
        called["n"] += 1
        raise AssertionError("must never hit the network without an API key")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok, data, err = api.request_soft("GET", "v1/instances/")
    assert ok is False
    assert data is None
    assert called["n"] == 0
    assert err.startswith("config:")
    assert api._classify_http(err) == "fatal"


def test_request_soft_signature_is_the_patchable_surface():
    """The patchable surface is the contract: 61 seam sites in the suite call
    this with `retries=` and `_sleep=` by keyword, and the guard's pass-through
    forwards `(method, path, body, *a, **k)` positionally.

    Was `…_matches_herdd`, comparing this signature to
    `inspect.signature(_unguarded(herdd.request_soft))`. Post-6d the launcher
    re-exports this function, so both `_unguarded` calls unwrap to one object
    and the comparison was a signature against itself. The literal parameter
    list below is what it was standing in for.
    """
    ours = inspect.signature(_unguarded(api.request_soft))
    assert list(ours.parameters) == ["method", "path", "body", "timeout",
                                     "retries", "_sleep"]
    assert ours.parameters["_sleep"].default is time.sleep


# --------------------------------------------------------------------------
# 4. request — the raising twin
# --------------------------------------------------------------------------

def test_request_returns_the_payload(monkeypatch):
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(True, {"id": 7}, None))
    assert api.request("GET", "v1/instances/") == {"id": 7}


def test_request_exits_on_failure(monkeypatch):
    """`request()` is the raising twin and it raises by EXITING — ported
    verbatim, not softened. Every one-shot CLI command depends on this."""
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: result.Soft(False, None, "HTTP 401 on GET x: nope"))
    with pytest.raises(SystemExit) as e:
        api.request("GET", "v1/instances/")
    assert str(e.value) == "error: HTTP 401 on GET x: nope"


def test_request_resolves_request_soft_as_a_module_attribute(monkeypatch):
    """Late binding is the whole reason the conftest guard works: `request()`
    looks `request_soft` up in module globals at CALL time, so wrapping the
    module attribute also covers everything routed through the raising twin.
    A `from .api import request_soft` inside this module would have made the
    guard vacuous for `request()` and nothing would have gone red."""
    seen = []
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: seen.append(a) or result.Soft(True, "ok", None))
    assert api.request("PUT", "v0/instances/1/", {"state": "stopped"}, 5) == "ok"
    assert seen == [("PUT", "v0/instances/1/", {"state": "stopped"}, 5)]


def test_request_exits_on_a_missing_key(monkeypatch):
    """End to end through the real `request_soft`, with no key and no network."""
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("no network without a key"))
    with pytest.raises(SystemExit) as e:
        api.request("GET", "v1/instances/")
    assert "config:" in str(e.value)


# --------------------------------------------------------------------------
# 5. THE LIVE-FLEET GUARD — conftest._block_mutating_api_calls
# --------------------------------------------------------------------------

def test_every_guard_target_exists():
    """A guard that cannot find its target does not fail — it stops guarding.

    `_block_mutating_api_calls` walks `sys.modules` and skips anything absent or
    lacking `request_soft`, which is correct (an unimported module needs no
    guard) and is exactly why a RENAME would be silent: `hasattr` goes False,
    the fixture returns, and the next mutating test bills a real box. So the
    roster is asserted here, by importing each target and demanding a callable.
    Adding a third copy of the funnel means adding it to
    `conftest._GUARDED_REQUEST_SOFT_MODULES` and this test is what says so."""
    assert "herdd" in conftest._GUARDED_REQUEST_SOFT_MODULES
    assert "vastlib.core.api" in conftest._GUARDED_REQUEST_SOFT_MODULES
    for modname in conftest._GUARDED_REQUEST_SOFT_MODULES:
        mod = importlib.import_module(modname)
        fn = getattr(mod, "request_soft", None)
        assert callable(fn), f"{modname}.request_soft is not a callable guard target"
    assert conftest._READ_METHODS == frozenset({"GET", "HEAD", "OPTIONS"})


def test_mutating_call_through_vastlib_is_blocked(monkeypatch):
    """A PUT through `vastlib.core.api.request_soft` must be refused by the
    fixture, in the `(False, None, err)` shape every caller already survives.

    Fail-safe by construction: both API-key vars are unset and `urlopen` is
    rigged to fail the test, so if the guard were MISSING this would return the
    `config:` error (or fail loudly) rather than issuing a live PUT. The
    assertion therefore distinguishes "the guard refused" from "there was no
    key" — which is what makes it a test of the guard and not of the env."""
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("guard let a PUT reach the network"))
    ok, data, err = api.request_soft("PUT", "/v0/instances/1/", {"state": "stopped"})
    assert ok is False
    assert data is None
    assert err.startswith("test isolation: PUT /v0/instances/1/ blocked")
    assert "_block_mutating_api_calls" in err


@pytest.mark.parametrize("method", ["PUT", "POST", "DELETE", "PATCH", "put", "delete"])
def test_guard_blocks_every_mutating_method_case_insensitively(monkeypatch, method):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("guard let a mutation through"))
    ok, _, err = api.request_soft(method, "/v0/instances/1/", {"state": "stopped"})
    assert ok is False and "test isolation" in err


def test_guard_passes_reads_through_to_the_real_function(monkeypatch):
    """Reads must reach the REAL implementation: several tests legitimately
    probe the unreachable-API path, and the retry tests above only work because
    a GET is not intercepted. If this went red the guard would be shadowing the
    unit under test — the exact failure that reverted the `_put_*` layer."""
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b'{"ok": 1}'))
    assert api.request_soft("GET", "v1/instances/") == (True, {"ok": 1}, None)


def test_guard_still_covers_the_herdd_copy(monkeypatch):
    """NOT a parity test, and not tautological post-6d — keep it.

    There is one body now, but conftest's `_GUARDED_REQUEST_SOFT_MODULES`
    wraps each MODULE ATTRIBUTE separately, and `herdd.request_soft` is a
    binding external callers still reach (`launch_serve.sh`'s heredoc does
    `import herdd; herdd.request(...)`). Dropping `"herdd"` from that
    tuple would leave that binding unwrapped and a stray PUT would go to the
    live API from a test run — which is exactly what this exercises.
    """
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("guard let a PUT reach the network"))
    ok, _, err = herdd.request_soft("PUT", "/v0/instances/1/", {"state": "stopped"})
    assert ok is False and "test isolation" in err


# --------------------------------------------------------------------------
# 8. the launcher's bindings — what is left of the deleted parity halves
# --------------------------------------------------------------------------
def test_the_launcher_re_exports_rather_than_redefines():
    """One body per name, reachable under both spellings.

    Three parity sweeps in this file (`_classify_http` over 37 probes,
    `_api_key_soft` over six env shapes, `request_soft`'s signature) compared
    `herdd.<name>` to `api.<name>` while both were real implementations.
    Plan §8 step 6d left one implementation, so those comparisons became
    self-comparisons and are deleted. This is the residue with teeth: a second
    body under any of these names in `herdd.py` would un-do the port for
    every consumer that still addresses the flat module (`boxstate.py`,
    `hosts.py`, `hostfacts.py`, `bid_echo_probe.py`, `launch_serve.sh`), and
    conftest's request_soft guard would be wrapping the wrong function.
    """
    for name in ("_classify_http", "_api_key_soft", "API"):
        assert getattr(herdd, name) is getattr(api, name), (
            f"herdd.{name} is a second body again — the launcher must "
            f"re-export vastlib.core.api's object, never redefine it")
    # `request` / `request_soft` are wrapped per-module by conftest's autouse
    # guard, so compare what is underneath the wrappers, not the wrappers.
    for name in ("request", "request_soft"):
        assert _unguarded(getattr(herdd, name)) is _unguarded(getattr(api, name)), name


# --------------------------------------------------------------------------
# 9. the v0/bundles/ pacer — the ONE endpoint with a known ceiling
# --------------------------------------------------------------------------
class _FakeClock:
    """A monotonic clock the test drives, so a pacing assertion is exact
    instead of a wall-clock tolerance. `sleep` ADVANCES it, which is the whole
    property under test: the gate must charge itself for what it slept."""

    def __init__(self, t=1_000.0):
        self.t = t
        self.slept = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


@pytest.fixture
def paced(monkeypatch):
    """A fresh gate on a fake clock. `api.time` is looked up at CALL time in
    `_bundles_pace`, so replacing the module attribute steers it and nothing
    else (`request_soft`'s `_sleep` default was bound at def)."""
    clock = _FakeClock()
    monkeypatch.setattr(api, "time", clock)
    monkeypatch.setattr(api, "_BUNDLES_LAST_SEND", [0.0])
    monkeypatch.setattr(api, "_BUNDLES_PACE_STATS",
                        {"sends": 0, "waits": 0, "waited_s": 0.0})
    return clock


def test_bundles_pace_holds_the_configured_rate(paced):
    gap = 1.0 / api.BUNDLES_MAX_RPS
    for _ in range(5):
        api._bundles_pace(paced.sleep)
    # The first send is free (nothing preceded it); each of the other four
    # waits exactly one gap. Five sends across four gaps == the budget.
    assert paced.slept == [gap] * 4
    assert api.BUNDLES_MAX_RPS < 5.0, "no headroom under vast's stated ceiling"


def test_bundles_pace_never_drops_a_read(paced):
    """A skipped market read blinds the defend/decay arms, so the gate delays
    and never refuses: every caller that entered got a send."""
    for _ in range(12):
        api._bundles_pace(paced.sleep)
    assert api.bundles_pace_stats()["sends"] == 12


def test_no_single_pace_sleeps_longer_than_one_gap(paced):
    """The pathological shape for a min-interval gate is a debt that
    accumulates: 2N boxes into a tick and the last caller sleeps for the whole
    burst. It cannot happen — each wait is bounded by the gap itself."""
    gap = 1.0 / api.BUNDLES_MAX_RPS
    for _ in range(20):
        api._bundles_pace(paced.sleep)
    assert max(paced.slept) <= gap
    assert sum(paced.slept) == pytest.approx(19 * gap)


def test_a_caller_that_took_its_time_is_not_paced_at_all(paced):
    """Real work between two reads (an HTTP round trip) already spends the
    gap. Charging it again would be a pure tax on the tick."""
    api._bundles_pace(paced.sleep)
    paced.t += 5.0                                   # the request itself
    api._bundles_pace(paced.sleep)
    assert paced.slept == []


def test_bundles_pace_is_serialized_across_threads(monkeypatch):
    """The tick is one thread, but `fleet status` answers on another and the
    dash survey runs a pool — the gate is where the aggregate rate exists, so
    it has to hold with all of them inside it at once."""
    gap = 1.0 / api.BUNDLES_MAX_RPS
    monkeypatch.setattr(api, "_BUNDLES_LAST_SEND", [0.0])
    monkeypatch.setattr(api, "_BUNDLES_PACE_STATS",
                        {"sends": 0, "waits": 0, "waited_s": 0.0})
    ready = threading.Barrier(8)

    def worker():
        ready.wait()                                 # all eight, genuinely at once
        api._bundles_pace(lambda s: None)            # no real sleeping in a test

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = api.bundles_pace_stats()
    assert st["sends"] == 8
    # The first is free; the other seven each charged themselves one gap, so
    # the eight sends span the budget rather than landing together.
    assert st["waits"] == 7
    assert st["waited_s"] == pytest.approx(7 * gap)


@pytest.mark.parametrize("path,paced_expected", [
    ("v0/bundles/", True),
    ("/v0/bundles/", True),          # the leading slash is stripped downstream
    ("v0/instances/", False),
    ("v1/instances/", False),
    ("v0/bundles_not_really/", True),  # prefix match: over- not under-inclusive
])
def test_only_the_bundles_path_is_paced(monkeypatch, path, paced_expected):
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(api, "_bundles_pace", lambda *a: calls.append(path))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b"{}"))
    # UNGUARDED: conftest refuses a POST, and the refusal returns before the
    # gate — which would make this test pass for the wrong reason.
    _unguarded(api.request_soft)("POST", path, {"limit": 1})
    assert bool(calls) is paced_expected


def test_the_pacer_does_not_spend_the_retry_backoff_budget(monkeypatch):
    """`_sleep` belongs to the retry policy and four tests count its calls. A
    pace charged to it would read as a retry that never happened."""
    monkeypatch.setenv("VASTAI_API_KEY", "test-key")
    monkeypatch.setattr(api, "_BUNDLES_LAST_SEND", [0.0])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b"{}"))
    backoff = []
    _unguarded(api.request_soft)("POST", "v0/bundles/", {"limit": 1},
                                 _sleep=backoff.append)
    _unguarded(api.request_soft)("POST", "v0/bundles/", {"limit": 1},
                                 _sleep=backoff.append)
    assert backoff == []                              # both succeeded first try
