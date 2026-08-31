"""`restore_merged`: no reuse without a re-verify of THAT directory.

The property the whole B2 fast path rests on, and it is a property of CONTROL
FLOW — which is why the decision is a Python function and not a chain of shell
`if`s, and why the enumeration below is possible at all. Ported from
`driftr3-v10-27b-gen/test_v10_gen_bundle.py`, whose version of this test found a
real (expensive, not wrong) bug: a local dir that failed re-verify fell through
to a successful 52 GiB restore and `decide` reported `merge` anyway.

Portable: every effect is injected, so nothing here touches B2, a model, or a
disk beyond tmp_path.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modelkit import restore_merged as rm  # noqa: E402


def test_decide_never_returns_reuse_without_verification():
    """EXHAUSTIVE over every combination of step outcomes — 4x3x3x3x2 = 216
    states, including the ones no execution order can produce. A property that
    holds only on reachable inputs is one refactor away from not holding."""
    tri = (None, True, False)
    seen = set()
    for lp, lo, rp, po, ro in itertools.product((True, False), tri, (True, False),
                                                tri, tri):
        d = rm.decide(local_present=lp, local_ok=lo, remote_present=rp,
                      pull_ok=po, restored_ok=ro)
        seen.add(d.action)
        assert d.action in (rm.REUSE_LOCAL, rm.REUSE_RESTORED, rm.MERGE)
        if d.action in rm.REUSE_ACTIONS:
            assert d.verified is True, (lp, lo, rp, po, ro, d)
            source_ok = lo if d.action == rm.REUSE_LOCAL else ro
            assert source_ok is True, (lp, lo, rp, po, ro, d)
        else:
            assert d.verified is False, (lp, lo, rp, po, ro, d)
    # The enumeration must actually REACH all three verdicts; a decide() that
    # always merged would satisfy every assertion above.
    assert seen == {rm.REUSE_LOCAL, rm.REUSE_RESTORED, rm.MERGE}


def test_a_failed_local_dir_falls_through_to_the_restore():
    """The regression the ancestor's enumeration caught: purging a bad local dir
    must not throw away a completed restore and re-pay the base pull + merge."""
    d = rm.decide(local_present=True, local_ok=False, remote_present=True,
                  pull_ok=True, restored_ok=True)
    assert d.action == rm.REUSE_RESTORED and d.verified is True


@pytest.mark.parametrize("kw,action", [
    (dict(local_present=True, local_ok=True, remote_present=True,
          pull_ok=None, restored_ok=None), rm.REUSE_LOCAL),
    (dict(local_present=False, local_ok=None, remote_present=True,
          pull_ok=False, restored_ok=None), rm.MERGE),
    (dict(local_present=False, local_ok=None, remote_present=True,
          pull_ok=True, restored_ok=False), rm.MERGE),
    (dict(local_present=False, local_ok=None, remote_present=False,
          pull_ok=None, restored_ok=None), rm.MERGE),
    (dict(local_present=True, local_ok=False, remote_present=False,
          pull_ok=None, restored_ok=None), rm.MERGE),
])
def test_decide_named_outcomes(kw, action):
    assert rm.decide(**kw).action == action


def _spy(*, exists, local_ok, remote_has, pull_ok, restored_ok):
    """Record the ORDER of effects, so a test can assert on the sequence and not
    only on the verdict."""
    calls: list[tuple] = []

    def verify(d):
        n = len([c for c in calls if c[0] == "verify"])
        r = local_ok if (exists and n == 0) else restored_ok
        calls.append(("verify", d))
        return bool(r)

    return calls, dict(
        verify=verify,
        remote_has=lambda r: (calls.append(("has", r)), bool(remote_has))[1],
        pull=lambda r, dest: (calls.append(("pull", r, dest)), bool(pull_ok))[1],
        purge=lambda p: calls.append(("purge", p)),
        exists=lambda p: bool(exists),
        log=lambda m: None)


def test_resolve_verifies_a_present_local_dir_before_reusing_it():
    calls, eff = _spy(exists=True, local_ok=True, remote_has=True,
                      pull_ok=True, restored_ok=True)
    d = rm.resolve(merged_dir="/m", remote="checkpoints/x", **eff)
    assert d.action == rm.REUSE_LOCAL
    # Verified, and nothing else happened: no listing, no 52 GiB download.
    assert calls == [("verify", "/m")]


def test_resolve_purges_then_restores_then_reverifies():
    calls, eff = _spy(exists=True, local_ok=False, remote_has=True,
                      pull_ok=True, restored_ok=True)
    d = rm.resolve(merged_dir="/m", remote="checkpoints/x", **eff)
    assert d.action == rm.REUSE_RESTORED
    assert [c[0] for c in calls] == ["verify", "purge", "has", "pull", "verify"]


def test_resolve_purges_a_restore_that_fails_reverify():
    """A partially-correct model dir is worse than no model dir: the fallback is
    merely slow, the alternative is a wrong number nobody can see."""
    calls, eff = _spy(exists=False, local_ok=None, remote_has=True,
                      pull_ok=True, restored_ok=False)
    d = rm.resolve(merged_dir="/m", remote="checkpoints/x", **eff)
    assert d.action == rm.MERGE and d.verified is False
    assert [c[0] for c in calls] == ["has", "pull", "verify", "purge"]


def test_resolve_purges_the_partial_when_the_pull_does_not_complete():
    calls, eff = _spy(exists=False, local_ok=None, remote_has=True,
                      pull_ok=False, restored_ok=None)
    d = rm.resolve(merged_dir="/m", remote="checkpoints/x", **eff)
    assert d.action == rm.MERGE
    assert [c[0] for c in calls] == ["has", "pull", "purge"]
    # A pull that did not complete is never verified — verifying a partial and
    # trusting the answer is the failure this ordering exists to exclude.
    assert not any(c[0] == "verify" for c in calls)


def test_resolve_asks_b2_nothing_when_there_is_no_remote_publish():
    calls, eff = _spy(exists=False, local_ok=None, remote_has=False,
                      pull_ok=None, restored_ok=None)
    assert rm.resolve(merged_dir="/m", remote="checkpoints/x", **eff).action \
        == rm.MERGE
    assert [c[0] for c in calls] == ["has"]


def test_guard_verifier_injects_a_family_without_restore_importing_one(tmp_path):
    """The coupling this hoist removes: the two bundle copies of this file
    differed in exactly one token, the name of the guard module they imported.
    `resolve` never knew which guard it was calling and still does not."""
    from modelkit import merge_guard

    reports: list[dict] = []
    verify = merge_guard and rm.guard_verifier(
        merge_guard.load_spec("qwen36-27b"), reports=reports, log=lambda m: None)
    empty = tmp_path / "merged"
    empty.mkdir()
    assert verify(str(empty)) is False
    assert reports and reports[0]["family"] == "qwen36-27b"


def test_resolve_accepts_any_one_argument_verifier():
    """`verify` is a plain callable, so a caller can inject a stricter gate — a
    grade-B content check on a restored dir, say — without this module growing a
    branch per model family."""
    seen: list[str] = []
    calls, eff = _spy(exists=True, local_ok=True, remote_has=False,
                      pull_ok=None, restored_ok=None)
    eff["verify"] = lambda d: (seen.append(d), True)[1]
    assert rm.resolve(merged_dir="/m", remote="checkpoints/x",
                      **eff).action == rm.REUSE_LOCAL
    assert seen == ["/m"]
