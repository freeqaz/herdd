"""`vastlib.core.result` — the tuple-compatibility guarantee, made executable.

Why this file exists
--------------------
`core/result.py` is types + helpers only: it ports no function, so there is no
parity to assert against `herdd.py` — which is why this file outlived the
pure-parity ones. (`test_vastlib_core_labels.py` was the contrast case: it
asserted a duplication rather than a contract, and was deleted at plan §8 step
6d when the thinning removed the second copy it compared against.)
What it has instead is one load-bearing promise, and the entire value of the
module rests on it:

    every type here is a `typing.NamedTuple`, so it IS a `tuple` — unpackable,
    integer-indexable, and `==` to the bare tuple literal the 39 `*_soft`
    functions return today.

That promise is what makes adopting a type at a future port site a pure
annotation change. If it broke, the failure would not be loud: `_put_bid_soft(
iid, target)[0]` (two live sites, `herdd.py` when this was written,
`vastlib.boxes.lifecycle` since the port) would raise, and every existing
test asserting `_put_state_soft(...) == (True, None)` (test_lifecycle.py) would
fail *in the module being ported*, not here. So it is asserted here, cheaply,
before anything depends on it.

The second thing this file pins is the taxonomy's two semantic traps, as tests
rather than prose: `OkErr` and `OkData` are the SAME positional shape with
opposite meanings in slot 2, and `ValueErr` is `OkErr` mirrored. A test that
shows `OkData(True, "loading") == OkErr(True, "loading")` is the clearest
statement of why they have separate names — the type system cannot tell them
apart, and the reader must.

No expectation here is inherited from an older test; nothing was repointed into
this file. Provenance: new in the vastlib package, plan §8 step 2 (`core/`).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vastlib.core import result  # noqa: E402

ALL_TYPES = (result.Soft, result.OkErr, result.OkData, result.ValueErr, result.ProcResult)


# --------------------------------------------------------------------------
# The compatibility guarantee
# --------------------------------------------------------------------------

def test_soft_is_a_tuple_instance() -> None:
    """The whole point: `Soft` IS a tuple, not a tuple-like object.

    A dataclass in this slot would have looked identical at the call sites that
    use attribute access, and broken every one that unpacks, indexes, or
    compares against a bare tuple literal.
    """
    s = result.Soft(True, {"id": 1}, None)
    assert isinstance(s, tuple)
    assert type(s).__mro__[1] is tuple      # direct subclass, no proxy in between
    assert len(s) == 3


def test_every_result_type_is_a_tuple_instance() -> None:
    """Not just `Soft` — the guarantee is family-wide."""
    samples = (
        result.Soft(True, "d", None),
        result.OkErr(True, None),
        result.OkData(True, "running"),
        result.ValueErr("key", None),
        result.ProcResult(0, "out", ""),
    )
    for s in samples:
        assert isinstance(s, tuple), f"{type(s).__name__} is not a tuple"
    assert {t.__name__ for t in ALL_TYPES} == {type(s).__name__ for s in samples}


def test_no_result_tuple_is_ever_falsy() -> None:
    """`if not soft_fn():` is always wrong, and stays always wrong.

    A non-empty tuple is truthy regardless of its contents, so a caller that
    tests the RESULT instead of the flag reads success on every failure. That
    is a property of tuples, and pinning it here is what stops someone
    "fixing" a type into something with a custom `__bool__`.
    """
    for value in (
        result.Soft(False, None, "HTTP 404 no such instance"),
        result.OkErr(False, "boom"),
        result.OkData(False, "loading"),
        result.ValueErr(None, "config: VASTAI_API_KEY not set (env or .env)"),
        result.ProcResult(127, "", "rclone not found on PATH"),
    ):
        assert bool(value) is True


# --------------------------------------------------------------------------
# Shape A — Soft(ok, data, err)
# --------------------------------------------------------------------------

def test_soft_unpacks_and_indexes() -> None:
    ok, data, err = result.Soft(True, {"instances": []}, None)
    assert ok is True
    assert data == {"instances": []}
    assert err is None

    s = result.Soft(False, None, "network timed out")
    assert s[0] is False and s[1] is None and s[2] == "network timed out"
    assert s.ok is False and s.data is None and s.err == "network timed out"


def test_soft_equals_the_bare_tuple() -> None:
    assert result.Soft(True, {"a": 1}, None) == (True, {"a": 1}, None)
    assert (True, {"a": 1}, None) == result.Soft(True, {"a": 1}, None)
    assert hash(result.Soft(True, "x", None)) == hash((True, "x", None))
    assert result.Soft(False, "", "err") != (False, None, "err")   # ""/None stay distinct


def test_soft_field_truthiness() -> None:
    """The failure-slot asymmetry the module docstring warns about.

    `request_soft` fails with `data=None`; the exec/copy/ssh trio fail with
    `data=""`. Both are falsy, so `if not data` agrees — anything stronger
    does not.
    """
    request_style = result.Soft(False, None, "HTTP 500 upstream")
    exec_style = result.Soft(False, "", "execute request failed")
    assert not request_style.data and not exec_style.data
    assert request_style.data is None and exec_style.data == ""
    assert bool(request_style.ok) is False and bool(exec_style.err) is True

    success = result.Soft(True, {}, None)
    assert bool(success.ok) is True
    assert not success.data          # an EMPTY payload is falsy on success too
    assert not success.err


def test_ssh_style_success_carries_stderr_in_data() -> None:
    """Trap 1, as a test: ok=True is transport-only for the ssh/copy pair."""
    s = result.Soft(True, "stdout text" + "permission denied\n", None)
    assert s.ok is True
    assert "permission denied" in s.data      # the remote command FAILED
    assert s.err is None                      # and the result says nothing about it


# --------------------------------------------------------------------------
# Shape B — OkErr(ok, err)
# --------------------------------------------------------------------------

def test_okerr_unpacks_indexes_and_equals_the_bare_tuple() -> None:
    ok, err = result.OkErr(True, None)
    assert ok is True and err is None
    assert result.OkErr(True, None) == (True, None)          # test_lifecycle.py's shape
    assert result.OkErr(False, "HTTP 404 not found")[0] is False   # _put_bid_soft(...)[0]
    assert len(result.OkErr(True, None)) == 2


def test_okerr_permits_a_non_none_err_on_success() -> None:
    """No invariant assertion, deliberately: fleetd's dry-run path returns this.

    `fleetd.Hooks.park/resume/destroy` return `(True, "dry-run")`. Any
    `assert err is None if ok` in `core/result.py` would fire on the dry-run
    lane of the daemon, which is the one lane that must never crash.
    """
    dry_run = result.OkErr(True, "dry-run")
    assert dry_run == (True, "dry-run")
    assert dry_run.ok is True and dry_run.err == "dry-run"


def test_okerr_field_truthiness() -> None:
    api_prose = result.OkErr(False, "insufficient balance")   # HTTP 200, success:false
    assert not api_prose.ok and bool(api_prose.err) is True
    assert not result.OkErr(True, None).err


# --------------------------------------------------------------------------
# Shape C — OkData(ok, data): slot 2 is a PAYLOAD
# --------------------------------------------------------------------------

def test_okdata_unpacks_and_equals_the_bare_tuple() -> None:
    ok, st = result.OkData(True, "running")
    assert ok is True and st == "running"
    assert result.OkData(False, "loading") == (False, "loading")
    assert result.OkData(False, "loading").data == "loading"


def test_okdata_and_okerr_are_indistinguishable_as_tuples() -> None:
    """Trap 2: the type system cannot separate these — only the name does.

    `_wait_states_soft` returns `(False, last_status)`, where slot 2 is a live
    vast status string. Typing it `OkErr` would relabel `"loading"` as an
    error and any `if err:` caller would report a failure that did not happen.
    """
    assert result.OkData(False, "loading") == result.OkErr(False, "loading")
    assert result.OkData._fields == ("ok", "data")
    assert result.OkErr._fields == ("ok", "err")


# --------------------------------------------------------------------------
# Shape D — ValueErr(value, err): mirrored slot order
# --------------------------------------------------------------------------

def test_valueerr_unpacks_mirrored_and_equals_the_bare_tuple() -> None:
    k, err = result.ValueErr("sk-live-xxx", None)
    assert k == "sk-live-xxx" and err is None
    missing = result.ValueErr(None, "config: VASTAI_API_KEY not set (env or .env)")
    assert missing == (None, "config: VASTAI_API_KEY not set (env or .env)")
    assert missing.value is None
    assert missing[0] is None                     # slot 0 is the VALUE, not an ok flag


def test_valueerr_config_prefix_is_preserved_verbatim() -> None:
    """The err string is a typed channel: `_classify_http` matches `config:`."""
    missing = result.ValueErr(None, "config: VASTAI_API_KEY not set (env or .env)")
    assert missing.err is not None and missing.err.startswith("config:")
    assert not missing.value and bool(missing.err) is True


def test_valueerr_is_not_okerr_despite_the_same_arity() -> None:
    """`(None, "config: …")` and `(False, "config: …")` are different things."""
    assert result.ValueErr(None, "config: x") != result.OkErr(False, "config: x")
    assert result.ValueErr._fields == ("value", "err")


# --------------------------------------------------------------------------
# Shape E — ProcResult(rc, stdout, stderr)
# --------------------------------------------------------------------------

def test_procresult_unpacks_and_equals_the_bare_tuple() -> None:
    rc, out, errtxt = result.ProcResult(0, "transferred: 3\n", "2026/08/16 stats\n")
    assert rc == 0 and out.startswith("transferred")
    assert result.ProcResult(127, "", "rclone not found on PATH") == (
        127,
        "",
        "rclone not found on PATH",
    )


def test_procresult_success_test_is_rc_zero_not_truthiness() -> None:
    """`rc == 0` succeeds; `bool(rc)` is inverted, and stderr lies on success."""
    good = result.ProcResult(0, "out", "rclone progress noise\n")
    bad = result.ProcResult(127, "", "rclone not found on PATH")
    assert good.rc == 0 and bad.rc != 0
    assert not good.rc and bool(bad.rc) is True        # truthiness is BACKWARDS here
    assert good.stderr                                  # populated on a SUCCESSFUL run


# --------------------------------------------------------------------------
# Constructors and adapters
# --------------------------------------------------------------------------

def test_ok_helper_builds_the_canonical_success() -> None:
    assert result.ok({"id": 7}) == (True, {"id": 7}, None)
    assert isinstance(result.ok("x"), result.Soft)
    assert result.ok().data is None            # no payload is a legitimate success
    assert result.ok("x").err is None


def test_err_helper_defaults_data_to_none_and_keeps_the_message() -> None:
    e = result.err("HTTP 404 no such instance")
    assert e == (False, None, "HTTP 404 no such instance")
    assert e.ok is False and e.data is None


def test_err_helper_takes_an_explicit_empty_data_slot() -> None:
    """The exec/copy/ssh trio fail with `data=""`, not None — opt in explicitly."""
    e = result.err("execute request failed", data="")
    assert e == (False, "", "execute request failed")
    assert e.data == "" and e.data is not None


def test_err_never_puts_an_empty_string_in_the_err_slot() -> None:
    """`err` is None or a message; `""` belongs to the data slot only."""
    assert result.ok("payload").err is None
    assert result.err("boom").err == "boom"


def test_as_pair_drops_the_payload() -> None:
    assert result.ok({"a": 1}).as_pair() == (True, None)
    assert result.err("boom").as_pair() == (False, "boom")
    assert isinstance(result.err("boom").as_pair(), result.OkErr)


def test_value_or_collapses_to_the_bare_optional_shape() -> None:
    """The lossy direction — shape F, where failure and emptiness are one value."""
    assert result.ok({"a": 1}).value_or() == {"a": 1}
    assert result.err("boom").value_or() is None
    assert result.err("boom").value_or([]) == []
    assert result.ok(None).value_or("fallback") is None      # success wins, even with None
