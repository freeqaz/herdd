"""`fleet deploy` must prefer the LOCAL repo (FLEET_REVIEW_2026-08-14 item 5).

Landed work on this workstation lives on local `main` and is never pushed
unasked — ~25 commits ahead at review time — so a deploy that only ever fetched
`origin` silently shipped stale code. It did, twice, on 2026-08-14 (`c524a5a9`
instead of the merge it was meant to ship), and both times the operator had to
`git -C <release-checkout> fetch <local-repo> main` by hand before
`fleet deploy --ref <sha>` would resolve.

These tests never touch ~/.local/share/vast-fleetd, never write a unit, never
restart anything and never reach the network: the pure resolver is exercised
directly, and the fetch path runs against throwaway git repos in `tmp_path`.
"""
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from vastlib.fleet import daemon, deploy  # noqa: E402


# --------------------------------------------------------------------------- #
# the pure decision
# --------------------------------------------------------------------------- #
def test_bare_deploy_prefers_local_main():
    """The headline: no --ref, local fetch worked -> local/main, not origin."""
    assert deploy.resolve_deploy_ref(None, local_ok=True, origin_ok=True) == \
        (deploy.DEPLOY_LOCAL_REF, "local")


def test_origin_is_the_fallback_not_the_default():
    assert deploy.resolve_deploy_ref(None, local_ok=False, origin_ok=True) == \
        (deploy.DEPLOY_REF_DEFAULT, "origin")


def test_neither_fetch_is_reported_as_unfetched():
    """A stale origin/main resolved as-is is exactly the 2026-08-14 failure, so
    it gets its own source label rather than passing as an origin deploy."""
    assert deploy.resolve_deploy_ref(None, local_ok=False, origin_ok=False) == \
        (deploy.DEPLOY_REF_DEFAULT, "unfetched")


def test_explicit_ref_always_wins():
    for lo in (True, False):
        for oo in (True, False):
            assert deploy.resolve_deploy_ref("deadbee", local_ok=lo,
                                             origin_ok=oo) == ("deadbee",
                                                               "explicit")


# --------------------------------------------------------------------------- #
# the fetch path, against real throwaway repos
# --------------------------------------------------------------------------- #
def _git(cwd, *args, check=True):
    p = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                       text=True, timeout=120)
    if check and p.returncode != 0:
        raise AssertionError(f"git {args} failed: {p.stderr}")
    return p.stdout.strip()


def _commit(repo, name, text):
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", text)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path):
    """(local_repo, upstream, checkout) — `checkout` is cloned from `local`
    with `origin` pointed at `upstream`, the shape `_clone_deploy_checkout`
    builds. `local` then advances WITHOUT pushing: the real situation."""
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    upstream = tmp_path / "upstream.git"
    _git(tmp_path, "init", "--quiet", "--bare", "-b", "main", str(upstream))
    local = tmp_path / "local"
    local.mkdir()
    _git(local, "init", "--quiet", "-b", "main")
    base = _commit(local, "a.txt", "base")
    _git(local, "remote", "add", "origin", str(upstream))
    _git(local, "push", "--quiet", "origin", "main")
    checkout = tmp_path / "release"
    _git(tmp_path, "clone", "--quiet", str(local), str(checkout))
    _git(checkout, "remote", "set-url", "origin", str(upstream))
    _git(checkout, "fetch", "--quiet", "origin")
    return local, upstream, checkout, base


def test_local_fetch_resolves_an_unpushed_commit(repos):
    """The whole point: a commit that exists ONLY on local main is deployable."""
    local, _upstream, checkout, base = repos
    landed = _commit(local, "b.txt", "landed but never pushed")
    lines = []
    ref, source = deploy.prepare_deploy_ref(str(checkout), str(local),
                                            log=lines.append)
    assert (ref, source) == (deploy.DEPLOY_LOCAL_REF, "local")
    assert _git(checkout, "rev-parse", ref) == landed
    # ...and origin/main is demonstrably the stale answer the old code took.
    assert _git(checkout, "rev-parse", deploy.DEPLOY_REF_DEFAULT) == base


def test_the_source_is_printed(repos):
    """A deploy that cannot say which tree it shipped is how c524a5a9 went out
    twice. Both the source and the local repo path must appear in the log."""
    local, _upstream, checkout, _base = repos
    _commit(local, "b.txt", "landed")
    lines = []
    deploy.prepare_deploy_ref(str(checkout), str(local), log=lines.append)
    out = "\n".join(lines)
    assert "deploy source: LOCAL repo" in out
    assert str(local) in out


def test_explicit_local_sha_resolves_without_a_manual_fetch(repos):
    """The manual step this removes: `--ref <sha-only-on-local>` now resolves
    because the local fetch already ran."""
    local, _upstream, checkout, _base = repos
    sha = _commit(local, "b.txt", "landed")
    ref, source = deploy.prepare_deploy_ref(str(checkout), str(local), sha,
                                            log=lambda *_: None)
    assert (ref, source) == (sha, "explicit")
    assert _git(checkout, "rev-parse", ref) == sha


def test_origin_fallback_when_the_local_repo_is_gone(repos, tmp_path):
    """A missing/moved source repo degrades to origin — it does not fail the
    deploy, and it says so."""
    _local, _upstream, checkout, base = repos
    lines = []
    ref, source = deploy.prepare_deploy_ref(str(checkout),
                                            str(tmp_path / "no-such-repo"),
                                            log=lines.append)
    assert (ref, source) == (deploy.DEPLOY_REF_DEFAULT, "origin")
    assert _git(checkout, "rev-parse", ref) == base
    assert "local fetch skipped/failed" in "\n".join(lines)


def test_both_fetches_dead_refuses(repos, tmp_path):
    """Origin present but unreachable AND no local repo: nothing was refreshed,
    so the deploy refuses rather than shipping whatever the checkout held —
    the pre-2026-08-14 strictness, preserved."""
    _local, _upstream, checkout, _base = repos
    _git(checkout, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    ref, source = deploy.prepare_deploy_ref(str(checkout),
                                            str(tmp_path / "no-such-repo"),
                                            log=lambda *_: None)
    assert (ref, source) == (None, None)


def test_source_repo_equal_to_the_checkout_is_not_fetched(repos):
    """Deploying the release checkout FROM itself is a no-op fetch, not an
    error loop; origin still answers."""
    _local, _upstream, checkout, base = repos
    ref, source = deploy.prepare_deploy_ref(str(checkout), str(checkout),
                                            log=lambda *_: None)
    assert (ref, source) == (deploy.DEPLOY_REF_DEFAULT, "origin")
    assert _git(checkout, "rev-parse", ref) == base


# --------------------------------------------------------------------------- #
# the local repo is derived at runtime, never baked
# --------------------------------------------------------------------------- #
def test_local_source_repo_is_this_repo():
    """Derived from `__file__`'s git toplevel — so a moved/renamed checkout,
    a worktree or a symlinked path all still deploy from the right tree."""
    root = deploy.local_source_repo()
    assert os.path.isdir(os.path.join(root, ".git")) or \
        os.path.exists(os.path.join(root, ".git"))
    assert os.path.isfile(os.path.join(root, "tools", "vast", "fleetd.py"))


def test_local_source_repo_falls_back_outside_a_repo(tmp_path):
    assert deploy.local_source_repo(str(tmp_path)) == daemon.repo_root()


def test_no_absolute_home_path_is_baked_into_the_module():
    """Repo convention: no /home/<user> in code.

    Class C (refactor step 6): the invariant used to be read off the flat
    `fleetd.py` alone. The deploy logic now lives in `vastlib/fleet/`, so the
    scan covers BOTH — the flat file until step 6d retires it, and every
    vastlib.fleet module that holds the moved bodies. Both must hold while
    both exist; a file that vanishes must not make this pass by default,
    hence the non-triviality needle below.
    """
    scanned = [os.path.join(_HERE, "fleetd.py")]
    _fleet_pkg = os.path.dirname(os.path.abspath(deploy.__file__))
    scanned += sorted(os.path.join(_fleet_pkg, n)
                      for n in os.listdir(_fleet_pkg) if n.endswith(".py"))
    for path in scanned:
        src = open(path).read()
        # non-triviality: an empty/thinned file can never satisfy this vacuously
        assert "DEPLOY_CHECKOUT_DEFAULT" in src or "deploy_checkout_path" in src \
            or "def _reconcile_loop" in src or "def fleet_request" in src \
            or "def load_state" in src or "class Hooks" in src \
            or "def normalize_ceiling" in src or "__all__" in src, \
            f"{path} is too thin to guard anything — re-point this test"
        assert "/home/" not in src, path


# --------------------------------------------------------------------------- #
# THE SELF-CONFIRMING DEPLOY (2026-08-16)
#
# `herdd fleet deploy --ref main` printed a successful fetch, printed
# `checkout @ 430df0f0 (main, source=explicit)`, restarted fleetd, printed
# `VERIFIED: fleetd is live at rev=430df0f0 (expected 430df0f0)` and exited 0 —
# while 589f84dd had landed and both `local/main` and `origin/main` in the
# release checkout pointed at it. The deployed `tools/vast/herdd.py` contained
# zero occurrences of the function the deploy existed to ship.
#
# Two compounding defects:
#   1. `--ref main` was handed to `rev-parse` verbatim, and in the release
#      checkout `main` IS `DEPLOY_BRANCH` — the branch `checkout -B` leaves
#      behind. `rev-parse` prefers refs/heads/main over any remote-tracking ref
#      of the same name, so the "requested ref" named the checkout's CURRENT
#      position and the checkout was a no-op.
#   2. The VERIFIED line compared the live rev against HEAD read AFTER the
#      checkout. Whatever shipped was therefore "expected" by construction, so
#      the verification could not fail on a no-op.
# --------------------------------------------------------------------------- #
def test_ref_main_never_resolves_to_the_deploy_branch():
    """THE REGRESSION, as a pure decision. `main` must be tried against the
    fetched refs and never against the branch the last deploy left behind."""
    cands = deploy.deploy_ref_candidates("main")
    assert cands == ["local/main", "origin/main"]
    assert deploy.DEPLOY_BRANCH not in cands


def test_a_qualified_ref_or_a_sha_is_passed_through_literally():
    """The operator was specific; be literal. Rewriting `origin/main` to
    `local/origin/main` would break the one spelling that always worked."""
    assert deploy.deploy_ref_candidates("origin/main") == ["origin/main"]
    assert deploy.deploy_ref_candidates("local/main") == ["local/main"]
    assert deploy.deploy_ref_candidates("refs/tags/v1") == ["refs/tags/v1"]
    assert deploy.deploy_ref_candidates("589f84dd") == ["589f84dd"]
    assert deploy.deploy_ref_candidates("") == []


def test_a_non_main_branch_name_still_falls_back_to_itself():
    """Only the deploy branch is excluded. A topic branch that exists locally in
    the checkout and nowhere else is still deployable."""
    assert deploy.deploy_ref_candidates("topic") == \
        ["local/topic", "origin/topic", "topic"]


def test_ref_main_resolves_to_the_fetched_sha_not_the_stale_head(repos):
    """End to end against real repos, in the incident's exact shape: the
    checkout sits on an old `main`, the fetch brings in a newer one, and
    `--ref main` must name the NEW sha."""
    local, _upstream, checkout, base = repos
    landed = _commit(local, "b.txt", "landed but never pushed")
    deploy.prepare_deploy_ref(str(checkout), str(local), "main",
                              log=lambda *_: None)
    assert _git(checkout, "rev-parse", "HEAD") == base       # still stale
    sha, resolved = deploy._resolve_deploy_target(str(checkout), "main",
                                                  log=lambda *_: None)
    assert sha == landed and sha != base
    assert resolved == deploy.DEPLOY_LOCAL_REF
    # ...and the pre-fix behaviour is demonstrably the stale answer:
    assert _git(checkout, "rev-parse", "main") == base


def _deploy_args(checkout, local, ref):
    import argparse
    return argparse.Namespace(checkout=str(checkout), source=str(local),
                              ref=ref, python=None, no_restart=True,
                              force=False, dry_run=True, verify_timeout=5)


def test_a_deploy_that_would_not_move_the_tree_exits_nonzero(repos,
                                                             monkeypatch,
                                                             capsys):
    """THE VALUABLE ONE. Wedge the checkout so it cannot move, and assert the
    command FAILS instead of printing VERIFIED. Before the fix this shape
    reported success: HEAD was read after the checkout and compared to itself.

    `--no-restart` returns before any systemd call, so if the guard ever stops
    firing this test fails LOUD (rc 0) rather than shelling out."""
    local, _upstream, checkout, base = repos
    landed = _commit(local, "b.txt", "landed but never pushed")
    real_git = deploy._git

    def wedged(cwd, *args, **kw):
        if args[:2] == ("checkout", "-B"):
            return 0, ""                    # claims success, moves nothing
        return real_git(cwd, *args, **kw)

    monkeypatch.setattr(deploy, "_git", wedged)
    rc = deploy.cmd_deploy(_deploy_args(checkout, local, "main"))
    out = capsys.readouterr().out
    assert rc != 0, "a deploy that did not move the tree reported success"
    assert "did NOT move" in out
    assert "VERIFIED" not in out
    assert landed != base


def test_a_deploy_that_does_move_the_tree_gets_past_the_movement_gate(repos,
                                                                      capsys):
    """The positive control for the test above — without it, "rc != 0" proves
    nothing, since this fixture cannot reach rc 0 at all.

    It stops at the `tools/vast/fleetd.py does not exist` gate, which is
    correct: the throwaway repo holds one text file, and the next steps would
    write a systemd unit into the real `~/.config/systemd/user`. What is asserted
    is that the tree MOVED and the movement gate stayed quiet — i.e. the failure
    is the fixture's shape, not the guard."""
    local, _upstream, checkout, base = repos
    landed = _commit(local, "b.txt", "landed but never pushed")
    deploy.cmd_deploy(_deploy_args(checkout, local, "main"))
    out = capsys.readouterr().out
    assert _git(checkout, "rev-parse", "HEAD") == landed != base
    assert "did NOT move" not in out
    assert "resolved as local/main" in out
    assert "fleetd.py does not exist" in out       # the gate it DID stop at


def test_a_diverged_release_checkout_refuses_rather_than_discarding_commits(
        repos, capsys):
    """`checkout -B` force-moves the branch; the dirty check only covers
    UNCOMMITTED work. A release checkout with local commits means something is
    not what the operator thinks it is — say so instead of eating them."""
    local, _upstream, checkout, _base = repos
    _commit(local, "b.txt", "landed")
    stray = _commit(checkout, "stray.txt", "committed in the release checkout")
    rc = deploy.cmd_deploy(_deploy_args(checkout, local, "main"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "would DISCARD" in out and "--force" in out
    assert _git(checkout, "rev-parse", "HEAD") == stray       # untouched


def test_force_overrides_the_divergence_refusal(repos, capsys):
    local, _upstream, checkout, _base = repos
    landed = _commit(local, "b.txt", "landed")
    _commit(checkout, "stray.txt", "committed in the release checkout")
    a = _deploy_args(checkout, local, "main")
    a.force = True
    deploy.cmd_deploy(a)
    assert "would DISCARD" not in capsys.readouterr().out
    assert _git(checkout, "rev-parse", "HEAD") == landed
