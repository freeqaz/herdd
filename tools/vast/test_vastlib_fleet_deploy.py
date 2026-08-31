"""`vastlib.fleet.deploy` — the release-checkout deploy, ported out of `fleetd.py`.

What this file is for
---------------------
Three classes of claim, and nothing else:

1. **The path anchors.** `TOOLS_VAST_DIR` / `_REPO_ROOT` are the one part of the
   port that is not textually verbatim, because the module sits three levels
   deeper than `fleetd.py`. Getting the depth wrong raises nothing — it silently
   relocates the generated unit's `WorkingDirectory=` and the `.env` the daemon
   hot-reloads (that exact regression, one `dirname` too few, shipped on
   2026-08-09 and nothing alarmed). Only a comparison against the flat file's
   own resolution can catch it, so that comparison is here.

2. **The fail-closed order** (manifest H7) and the two self-confirming-deploy
   traps (H6): a bare `--ref main` must never resolve the deploy branch, and the
   "did the tree move" assert compares against the RESOLVED TARGET, never
   against HEAD read back after the checkout.

3. **The one NEW step**, `ensure_deploy_deps` — pip-install + import probe into
   the release venv, fail-closed BEFORE the unit is rewritten. The live release
   venv has no pydantic (measured 2026-08-16), so a restart onto the merged
   layout without it is a crash-loop with the fleet unsupervised. The ordering
   is the assertion: at probe time the unit file must not have been touched, and
   on probe failure nothing may reach `systemctl`.

Everything here is hermetic: **no git, no pip, no systemd, no socket, no
network**. `deploy._git` and `deploy.subprocess` are patched as MODULE
ATTRIBUTES (plan §8b — the call form is what makes the patch steer), and the
one daemon-facing call, `client.fleet_request`, is patched the same way.

This file does not re-test `test_fleetd.py` / `test_fleet_deploy_local.py`,
which exercised the flat copies unedited until step 6d and reach this module
through `fleetd`'s re-exports now; where a claim overlaps, the assertion text is
deliberately the same string.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

from vastlib.fleet import client, deploy              # noqa: E402


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class FakeProc:
    """Just enough of `subprocess.CompletedProcess` for the two call sites."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeSubprocess:
    """Stands in for the `subprocess` module attribute of `deploy`.

    Records every argv in order — the ordering assertions in this file read it —
    and lets a test script per-argv results by a substring key.
    """

    DEVNULL = -3

    def __init__(self, results: dict[str, FakeProc] | None = None,
                 on_run=None):
        self.results = results or {}
        self.calls: list[list[str]] = []
        self.on_run = on_run

    def _result(self, argv: list[str]) -> FakeProc:
        joined = " ".join(argv)
        for key, proc in self.results.items():
            if key in joined:
                return proc
        return FakeProc(0)

    def run(self, argv, **kw):                          # noqa: ANN001, ANN003
        self.calls.append(list(argv))
        if self.on_run is not None:
            self.on_run(list(argv))
        return self._result(list(argv))

    def call(self, argv, **kw):                         # noqa: ANN001, ANN003
        self.calls.append(list(argv))
        return self._result(list(argv)).returncode

    def argvs(self, needle: str) -> list[list[str]]:
        return [c for c in self.calls if needle in " ".join(c)]


class FakeGit:
    """A `deploy._git` replacement dispatching on the git argv.

    `_git`'s contract is total — (rc, text), never raises — so the double is
    total too. Unmatched commands return (0, "") rather than blowing up, which
    keeps a test's script to the commands it actually cares about.
    """

    def __init__(self, **answers: tuple[int, str]):
        self.answers = answers
        self.calls: list[tuple[str, ...]] = []
        self.script: dict[tuple[str, ...], tuple[int, str]] = {}

    def __call__(self, cwd: str, *args: str, timeout: float = 180):
        self.calls.append(args)
        for prefix, ans in self.script.items():
            if args[:len(prefix)] == prefix:
                return ans
        return 0, ""

    def when(self, *prefix: str):
        def setter(rc: int, out: str = "") -> None:
            self.script[tuple(prefix)] = (rc, out)
        return setter

    def ran(self, *prefix: str) -> bool:
        return any(c[:len(prefix)] == tuple(prefix) for c in self.calls)


def _release_tree(tmp_path: Path, *, with_vastlib: bool = True,
                  with_script: bool = True) -> Path:
    """The on-disk shape `cmd_deploy` probes with `os.path.isdir` / `isfile`.

    Only the shape — no git objects, because `_git` is a double here.
    """
    rel = tmp_path / "release"
    (rel / ".git").mkdir(parents=True)
    (rel / "tools" / "vast").mkdir(parents=True)
    if with_script:
        (rel / "tools" / "vast" / "fleetd.py").write_text("#\n")
    if with_vastlib:
        (rel / "tools" / "vast" / "vastlib").mkdir()
        (rel / "tools" / "vast" / "vastlib" / "requirements.txt").write_text(
            "pydantic>=2\n")
    return rel


def _source_tree(tmp_path: Path, *, env: bool = True) -> Path:
    src = tmp_path / "src"
    (src / ".git").mkdir(parents=True)
    if env:
        (src / ".env").write_text("K=V\n")
    return src


def _args(checkout: Path, source: Path, **over):         # noqa: ANN003
    import argparse
    ns = dict(checkout=str(checkout), source=str(source), ref="main",
              python=sys.executable, no_restart=True, force=False,
              dry_run=False, verify_timeout=1.0)
    ns.update(over)
    return argparse.Namespace(**ns)


HEAD_BEFORE = "1111111111111111111111111111111111111111"
TARGET = "2222222222222222222222222222222222222222"


def _happy_git(target: str = TARGET, before: str = HEAD_BEFORE) -> FakeGit:
    """A checkout that fetches, resolves the target, moves, and audits clean."""
    g = FakeGit()
    g.when("remote")(0, "origin")
    g.when("fetch")(0, "")
    g.when("status", "--porcelain")(0, "")
    g.when("rev-parse", "--verify")(0, target)
    g.when("rev-parse", "HEAD")(0, target)
    g.when("rev-parse", "--short")(0, target[:9])
    g.when("rev-parse", "--abbrev-ref")(0, "main")
    g.when("checkout", "-B")(0, "")
    g.when("merge-base")(0, "")
    return g


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway HOME — the unit is written under ~/.config/systemd/user."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


def _unit_path(home: Path) -> Path:
    return home / ".config" / "systemd" / "user" / client.FLEET_UNIT_NAME


class FakeClock:
    """`deploy.time`, so a poll deadline costs no wall clock.

    `_verify_live_rev` bounds itself with `time.time()` and paces with
    `time.sleep()`; a real 5 s spin per negative case is the only thing that
    makes these tests slow, and a fake clock also proves the loop is bounded by
    the DEADLINE rather than by a retry count.
    """

    def __init__(self, start: float = 1_000.0):
        self.now = start
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


# --------------------------------------------------------------------------- #
# 1. the path anchors — the one non-verbatim line, pinned against the flat file
# --------------------------------------------------------------------------- #
def test_tools_vast_dir_is_the_directory_fleetd_calls_HERE():
    """`fleetd._HERE` is `tools/vast`; three dirnames from this deeper module."""
    assert deploy.TOOLS_VAST_DIR == str(VAST_DIR)
    assert os.path.isfile(os.path.join(deploy.TOOLS_VAST_DIR, "fleetd.py"))


def test_repo_root_matches_fleetd_computation():
    """The ported anchor == what `fleetd.py` computes for itself at runtime.

    `fleetd.repo_root()` is `dirname(dirname(_HERE))` with
    `_HERE = dirname(abspath(fleetd.__file__))` — three dirnames from its own
    path. From `vastlib/fleet/deploy.py` the same root is five.
    """
    fleetd_py = os.path.abspath(str(VAST_DIR / "fleetd.py"))
    expected = os.path.dirname(os.path.dirname(os.path.dirname(fleetd_py)))
    assert deploy._REPO_ROOT == expected
    assert os.path.isfile(os.path.join(deploy._REPO_ROOT, "tools", "vast",
                                       "fleetd.py"))


def test_naive_file_arithmetic_here_would_be_wrong():
    """Copying the two-dirname expression verbatim lands in `tools/vast`.

    Not an error — a silently different directory, which is why the 2026-08-09
    regression cost a dead hot-reload and a wrong `WorkingDirectory=` with
    nothing raising. Only the comparison catches it.
    """
    naive = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(deploy.__file__))))
    assert naive == deploy.TOOLS_VAST_DIR
    assert naive != deploy._REPO_ROOT


def test_repo_root_agrees_with_the_daemons_own_anchor():
    """Two modules, one root. `daemon.repo_root()` is the ported symbol; this
    module cannot import it (daemon imports deploy — the reverse edge is a
    cycle), so the values are pinned equal instead."""
    daemon = pytest.importorskip("vastlib.fleet.daemon",
                                 reason="sibling module still in flight")
    assert deploy._REPO_ROOT == daemon.repo_root()


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False),
                                            (None, False)])
def test_dry_run_reader_agrees_with_the_daemons(monkeypatch, value, expected):
    """Same env key, same predicate — a deploy must not write an
    `Environment=FLEETD_DRY_RUN=1` line the daemon would disagree with."""
    if value is None:
        monkeypatch.delenv("FLEETD_DRY_RUN", raising=False)
    else:
        monkeypatch.setenv("FLEETD_DRY_RUN", value)
    assert deploy._dry_run_enabled() is expected
    daemon = pytest.importorskip("vastlib.fleet.daemon",
                                 reason="sibling module still in flight")
    assert deploy._dry_run_enabled() == daemon.dry_run_enabled()


def test_the_unit_name_is_read_from_the_protocol_module_not_re_declared():
    """One literal. `client.FLEET_UNIT_NAME` is the systemd contract the CLI
    reads too; a second copy here is how one contract drifts into two."""
    assert not hasattr(deploy, "UNIT_NAME")
    assert client.FLEET_UNIT_NAME == "vast-fleetd.service"


# --------------------------------------------------------------------------- #
# 2. `_git`'s total contract
# --------------------------------------------------------------------------- #
def test_git_never_raises_so_the_audit_can_report(monkeypatch):
    """Every deploy step depends on this: a git that blows up must come back as
    a reason, not a traceback out of an audit."""
    class Boom:
        DEVNULL = -3

        def run(self, *a, **k):                          # noqa: ANN002, ANN003
            raise OSError("git: command not found")

    monkeypatch.setattr(deploy, "subprocess", Boom())
    rc, out = deploy._git("/nowhere", "rev-parse", "HEAD")
    assert rc == 1
    assert "command not found" in out


# --------------------------------------------------------------------------- #
# 3. `checkout_audit` — the fail-closed gate, TEXT included (§4 unit-text)
# --------------------------------------------------------------------------- #
def test_audit_accepts_a_clean_main_checkout(tmp_path, monkeypatch):
    rel = _release_tree(tmp_path)
    g = FakeGit()
    g.when("rev-parse", "--abbrev-ref")(0, "main")
    g.when("status", "--porcelain")(0, "")
    monkeypatch.setattr(deploy, "_git", g)
    assert deploy.checkout_audit(str(rel)) == []


def test_audit_rejects_a_missing_path_and_a_non_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "_git", FakeGit())
    assert deploy.checkout_audit(str(tmp_path / "nope")) == \
        [f"{tmp_path / 'nope'} does not exist"]
    plain = tmp_path / "plain"
    plain.mkdir()
    assert any("is not a git checkout" in b
               for b in deploy.checkout_audit(str(plain)))


def test_audit_rejects_a_linked_worktree(tmp_path, monkeypatch):
    """`.git` as a FILE — removable by `git worktree remove|prune`, which is
    exactly how out/land-main died under a running daemon."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    monkeypatch.setattr(deploy, "_git", FakeGit())
    bad = deploy.checkout_audit(str(wt))
    assert any("LINKED WORKTREE" in b and "out/land-main" in b for b in bad)


def test_audit_rejects_a_scratch_path(tmp_path, monkeypatch):
    scratch = tmp_path / "out" / "land-main"
    (scratch / ".git").mkdir(parents=True)
    monkeypatch.setattr(deploy, "_git", FakeGit())
    assert any("scratch" in b for b in deploy.checkout_audit(str(scratch)))


def test_audit_rejects_a_non_main_branch_and_a_dirty_tree(tmp_path, monkeypatch):
    rel = _release_tree(tmp_path)
    g = FakeGit()
    g.when("rev-parse", "--abbrev-ref")(0, "someones-topic")
    g.when("status", "--porcelain")(0, " M a.py\n M b.py")
    monkeypatch.setattr(deploy, "_git", g)
    bad = deploy.checkout_audit(str(rel))
    assert any("'someones-topic'" in b and "whatever branch someone last left"
               in b for b in bad)
    assert any("2 tracked file(s) modified" in b for b in bad)


# --------------------------------------------------------------------------- #
# 4. ref resolution — the two self-confirming-deploy traps
# --------------------------------------------------------------------------- #
def test_a_bare_main_never_resolves_the_deploy_branch():
    """H6: `refs/heads/main` is the branch `checkout -B` creates, so resolving it
    means deploying where the last deploy left the checkout."""
    assert deploy.deploy_ref_candidates("main") == ["local/main", "origin/main"]


def test_a_bare_topic_still_falls_through_to_the_local_name():
    assert deploy.deploy_ref_candidates("topic") == \
        ["local/topic", "origin/topic", "topic"]


@pytest.mark.parametrize("ref", ["origin/main", "local/main", "refs/tags/v1",
                                 "589f84dd"])
def test_a_qualified_ref_or_a_sha_passes_through_untouched(ref):
    assert deploy.deploy_ref_candidates(ref) == [ref]


def test_no_ref_resolves_to_no_candidates():
    assert deploy.deploy_ref_candidates(None) == []
    assert deploy.deploy_ref_candidates("  ") == []


def test_resolve_deploy_ref_prefers_local_then_origin_then_reports_unfetched():
    assert deploy.resolve_deploy_ref(None, local_ok=True, origin_ok=True) == \
        (deploy.DEPLOY_LOCAL_REF, "local")
    assert deploy.resolve_deploy_ref(None, local_ok=False, origin_ok=True) == \
        (deploy.DEPLOY_REF_DEFAULT, "origin")
    assert deploy.resolve_deploy_ref(None, local_ok=False, origin_ok=False) == \
        (deploy.DEPLOY_REF_DEFAULT, "unfetched")
    assert deploy.resolve_deploy_ref("deadbee", local_ok=True) == \
        ("deadbee", "explicit")


def test_resolve_deploy_target_names_the_candidate_that_won(tmp_path,
                                                            monkeypatch):
    g = FakeGit()
    g.script[("rev-parse", "--verify", "--quiet", "local/main^{commit}")] = \
        (0, TARGET)
    monkeypatch.setattr(deploy, "_git", g)
    lines: list[str] = []
    sha, resolved = deploy._resolve_deploy_target(str(tmp_path), "main",
                                                  log=lines.append)
    assert (sha, resolved) == (TARGET, "local/main")
    assert any("resolved as local/main" in ln for ln in lines)


def test_resolve_deploy_target_reports_everything_it_tried(tmp_path,
                                                           monkeypatch):
    g = FakeGit()
    g.when("rev-parse", "--verify")(1, "")
    monkeypatch.setattr(deploy, "_git", g)
    lines: list[str] = []
    assert deploy._resolve_deploy_target(str(tmp_path), "main",
                                         log=lines.append) == (None, None)
    assert any("tried: local/main, origin/main" in ln for ln in lines)


# --------------------------------------------------------------------------- #
# 5. `prepare_deploy_ref` — local-first, and refuse when nothing refreshed
# --------------------------------------------------------------------------- #
def test_prepare_fetches_local_first_and_says_which_tree_it_shipped(
        tmp_path, monkeypatch):
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    monkeypatch.setattr(deploy, "_git", g)
    lines: list[str] = []
    ref, source = deploy.prepare_deploy_ref(str(rel), str(src), None,
                                            log=lines.append)
    assert (ref, source) == (deploy.DEPLOY_LOCAL_REF, "local")
    fetches = [c for c in g.calls if c[:1] == ("fetch",)]
    assert fetches[0][2] == str(src), "the LOCAL fetch must come first"
    assert fetches[1][2:] == ("origin",), "origin is fetched additionally"
    assert any("deploy source: LOCAL repo" in ln for ln in lines)


def test_prepare_refuses_when_neither_fetch_refreshed_anything(tmp_path,
                                                               monkeypatch):
    """Pre-2026-08-14 strictness: a failed origin fetch was already fatal. It
    stays fatal when the local fetch cannot cover for it."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("fetch")(1, "could not read from remote")
    monkeypatch.setattr(deploy, "_git", g)
    lines: list[str] = []
    assert deploy.prepare_deploy_ref(str(rel), str(src), None,
                                     log=lines.append) == (None, None)
    assert any("refusing to deploy a revision nobody just fetched" in ln
               for ln in lines)


def test_prepare_falls_back_to_origin_when_the_source_repo_is_gone(tmp_path,
                                                                   monkeypatch):
    rel = _release_tree(tmp_path)
    g = _happy_git()
    monkeypatch.setattr(deploy, "_git", g)
    lines: list[str] = []
    ref, source = deploy.prepare_deploy_ref(str(rel), str(tmp_path / "gone"),
                                            None, log=lines.append)
    assert (ref, source) == (deploy.DEPLOY_REF_DEFAULT, "origin")
    assert any("is not a git checkout" in ln for ln in lines)


def test_the_local_fetch_never_writes_a_remote_into_the_checkout(tmp_path,
                                                                 monkeypatch):
    """A one-shot fetch of a PATH, deliberately not `git remote add`: nothing
    durable in the checkout config, so a moved source degrades to one failed
    fetch instead of poisoning every later one."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    monkeypatch.setattr(deploy, "_git", g)
    deploy._fetch_local_main(str(rel), str(src), log=lambda _: None)
    assert not g.ran("remote", "add")
    assert g.ran("fetch", "--quiet", str(src))


# --------------------------------------------------------------------------- #
# 6. `_ensure_env_link` — direction matters (it points AT the main checkout)
# --------------------------------------------------------------------------- #
def test_env_link_is_written_into_the_checkout_and_points_at_the_source(
        tmp_path):
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    note = deploy._ensure_env_link(str(rel), str(src), log=lambda _: None)
    link = rel / ".env"
    assert link.is_symlink()
    assert os.readlink(link) == str(src / ".env")
    assert note.startswith(".env: linked ->")


def test_env_link_reports_a_missing_source_instead_of_failing(tmp_path):
    rel = _release_tree(tmp_path)
    src = _source_tree(tmp_path, env=False)
    note = deploy._ensure_env_link(str(rel), str(src), log=lambda _: None)
    assert "MISSING" in note and "no credentials" in note
    assert not (rel / ".env").exists()


def test_a_dangling_env_link_counts_as_present(tmp_path):
    """`lexists`, not `exists` — a dangling link is never silently replaced."""
    rel = _release_tree(tmp_path)
    os.symlink(str(tmp_path / "nowhere" / ".env"), str(rel / ".env"))
    note = deploy._ensure_env_link(str(rel), str(tmp_path / "src"),
                                   log=lambda _: None)
    assert note.startswith(".env: present (symlink -> ")


# --------------------------------------------------------------------------- #
# 7. `cmd_deploy` — the fail-closed order (H7) and the movement assert (H6)
# --------------------------------------------------------------------------- #
def test_deploy_refuses_a_dirty_release_checkout(tmp_path, monkeypatch, capsys,
                                                 home):
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("status", "--porcelain")(0, " M tools/vast/fleetd.py")
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "_git", g)
    monkeypatch.setattr(deploy, "subprocess", sp)
    assert deploy.cmd_deploy(_args(rel, src)) == 1
    assert "uncommitted tracked changes" in capsys.readouterr().out
    assert not g.ran("checkout", "-B")
    assert not _unit_path(home).exists()


def test_a_diverged_release_checkout_refuses_rather_than_discarding_commits(
        tmp_path, monkeypatch, capsys, home):
    """`checkout -B` force-moves the branch; the dirty check only covers
    UNCOMMITTED work."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("rev-parse", "HEAD")(0, HEAD_BEFORE)
    g.when("merge-base")(1, "")                    # not an ancestor
    g.when("log")(0, "cafe123 committed in the release checkout")
    monkeypatch.setattr(deploy, "_git", g)
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    assert deploy.cmd_deploy(_args(rel, src)) == 1
    out = capsys.readouterr().out
    assert "would DISCARD" in out and "--force" in out
    assert not g.ran("checkout", "-B")
    assert not _unit_path(home).exists()


def test_force_overrides_the_divergence_refusal(tmp_path, monkeypatch, capsys,
                                                home):
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("merge-base")(1, "")
    monkeypatch.setattr(deploy, "_git", g)
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    assert deploy.cmd_deploy(_args(rel, src, force=True)) == 0
    assert "would DISCARD" not in capsys.readouterr().out
    assert g.ran("checkout", "-B")


def test_a_deploy_that_would_not_move_the_tree_exits_nonzero(tmp_path,
                                                             monkeypatch,
                                                             capsys, home):
    """THE VALUABLE ONE (H6). `checkout -B` claims success and moves nothing.
    The assert compares HEAD to the RESOLVED TARGET, so it fires; comparing HEAD
    to itself is what made the 2026-08-16 no-op deploy self-confirming."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("rev-parse", "HEAD")(0, HEAD_BEFORE)     # never becomes TARGET
    g.when("merge-base")(0, "")                     # ancestor, so no divergence
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "_git", g)
    monkeypatch.setattr(deploy, "subprocess", sp)
    assert deploy.cmd_deploy(_args(rel, src)) == 1
    out = capsys.readouterr().out
    assert "did NOT move" in out and "VERIFIED" not in out
    assert not _unit_path(home).exists()
    assert sp.argvs("systemctl") == []


def test_the_short_rev_is_gits_own_abbreviation_not_a_slice(tmp_path,
                                                            monkeypatch,
                                                            capsys, home):
    """The string compared against the daemon's `rev=` comes from
    `rev-parse --short`; a hardcoded 7 breaks verification once git grows it."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("rev-parse", "--short")(0, "22222222")   # 8, not 7
    monkeypatch.setattr(deploy, "_git", g)
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    assert deploy.cmd_deploy(_args(rel, src, ref=None)) == 0
    assert "@ 22222222 (local/main, source=local)" in capsys.readouterr().out


def test_deploy_refuses_an_unfit_checkout_with_rc_2(tmp_path, monkeypatch,
                                                    capsys, home):
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    g = _happy_git()
    g.when("rev-parse", "--abbrev-ref")(0, "someones-topic")
    monkeypatch.setattr(deploy, "_git", g)
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    assert deploy.cmd_deploy(_args(rel, src)) == 2
    assert "the deploy checkout is unfit" in capsys.readouterr().out
    assert not _unit_path(home).exists()


def test_deploy_refuses_when_execstart_would_not_exist(tmp_path, monkeypatch,
                                                       capsys, home):
    """The 2026-08-09 shape: a unit whose ExecStart is gone crash-loops."""
    rel = _release_tree(tmp_path, with_script=False)
    src = _source_tree(tmp_path)
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    assert deploy.cmd_deploy(_args(rel, src)) == 1
    assert "cannot exec" in capsys.readouterr().out
    assert not _unit_path(home).exists()


def test_deploy_refuses_a_missing_interpreter(tmp_path, monkeypatch, capsys,
                                              home):
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    a = _args(rel, src, python=str(tmp_path / "no-such-python"))
    assert deploy.cmd_deploy(a) == 1
    assert "does not exist" in capsys.readouterr().out
    assert not _unit_path(home).exists()


def test_deploy_writes_the_unit_and_links_env_without_restarting(tmp_path,
                                                                 monkeypatch,
                                                                 capsys, home):
    """The whole write path, minus systemd: the unit points at the RELEASE tree
    and the gitignored `.env` is linked so the daemon does not come up blind."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", sp)
    assert deploy.cmd_deploy(_args(rel, src)) == 0
    txt = _unit_path(home).read_text()
    assert f"ExecStart={sys.executable} {rel}/tools/vast/fleetd.py serve" in txt
    assert f"WorkingDirectory={rel}" in txt
    assert os.path.realpath(rel / ".env") == os.path.realpath(src / ".env")
    assert "restart" in capsys.readouterr().out.lower()
    assert sp.argvs("systemctl") == [], "--no-restart returns before systemd"


def test_the_activation_is_a_restart_and_enable_is_never_now(tmp_path,
                                                             monkeypatch,
                                                             home):
    """H8. `enable --now` no-ops on an active unit and reports success while the
    old config keeps running, so the deploy path always uses `restart`."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", sp)
    monkeypatch.setattr(deploy, "_verify_live_rev", lambda rev, deadline_s=60.0: 0)
    assert deploy.cmd_deploy(_args(rel, src, no_restart=False)) == 0
    systemctl = [" ".join(c) for c in sp.argvs("systemctl")]
    assert any(c.endswith("daemon-reload") for c in systemctl)
    assert any(c.endswith(f"enable {client.FLEET_UNIT_NAME}") for c in systemctl)
    assert any(c.endswith(f"restart {client.FLEET_UNIT_NAME}") for c in systemctl)
    assert not any("--now" in c for c in systemctl)


# --------------------------------------------------------------------------- #
# 8. NEW vs flat fleetd.py — the dependency step, fail-closed BEFORE restart
# --------------------------------------------------------------------------- #
def test_deps_installs_requirements_then_probes_the_import(tmp_path,
                                                           monkeypatch):
    rel = _release_tree(tmp_path)
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "subprocess", sp)
    ok, note = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                         log=lambda _: None)
    assert ok and "OK in /venv/bin/python3" in note
    pip, probe = sp.calls
    assert pip[:4] == ["/venv/bin/python3", "-m", "pip", "install"]
    assert pip[-1] == str(rel / "tools" / "vast" / "vastlib" /
                          "requirements.txt")
    assert probe == ["/venv/bin/python3", "-c", deploy.DEPLOY_IMPORT_PROBE]
    # The probe must cover EVERY import `fleetd.py serve` performs, not just the
    # engine: `fleetd.run()` imports `vastlib.cli._compose` to close the
    # cross-ring seams before it dispatches. A venv that has `fleet.daemon` and
    # not `cli._compose` is the crash-loop-on-RestartSec=5 shape this step
    # exists to refuse, so both names are asserted here by content.
    assert "vastlib.fleet.daemon" in deploy.DEPLOY_IMPORT_PROBE
    assert "vastlib.cli._compose" in deploy.DEPLOY_IMPORT_PROBE


def test_the_probe_runs_with_the_checkouts_tools_vast_on_pythonpath(tmp_path,
                                                                    monkeypatch):
    """The probe must import what `serve` will — the entry script inserts its
    own directory into `sys.path`, so the probe puts it on PYTHONPATH."""
    rel = _release_tree(tmp_path)
    seen: dict[str, str] = {}

    class Recorder(FakeSubprocess):
        def run(self, argv, **kw):                       # noqa: ANN001, ANN003
            if "-c" in argv:
                seen.update(kw.get("env") or {})
                seen["cwd"] = kw.get("cwd") or ""
            return super().run(argv, **kw)

    monkeypatch.setattr(deploy, "subprocess", Recorder())
    ok, _ = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                      log=lambda _: None)
    assert ok
    assert seen["PYTHONPATH"].split(os.pathsep)[0] == \
        str(rel / "tools" / "vast")
    assert seen["cwd"] == str(rel)


def test_a_failed_pip_install_is_fail_closed(tmp_path, monkeypatch):
    """An installer that RAN and refused stays fail-closed: a half-installed venv
    is not something the import probe can be trusted to judge."""
    rel = _release_tree(tmp_path)
    sp = FakeSubprocess({"pip": FakeProc(1, "", "No matching distribution")})
    monkeypatch.setattr(deploy, "subprocess", sp)
    ok, note = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                         log=lambda _: None)
    assert ok is False
    assert "install" in note and "No matching distribution" in note
    assert not [c for c in sp.calls if deploy.DEPLOY_IMPORT_PROBE in " ".join(c)], \
        "the probe must not run after a failed install"


def test_a_pipless_venv_falls_through_to_the_probe(tmp_path, monkeypatch):
    """Measured 2026-08-17: the release venv is uv-managed and has no `pip`
    module, so `python -m pip` exits 1 with "No module named pip" without ever
    judging the requirements. That is NOT a refusal — treating it as one blocked
    the deploy of a venv that already imported the daemon. The probe decides."""
    rel = _release_tree(tmp_path)
    sp = FakeSubprocess({"-m pip install": FakeProc(
        1, "", "/venv/bin/python3: No module named pip")})
    monkeypatch.setattr(deploy, "subprocess", sp)
    monkeypatch.setattr(deploy.shutil, "which", lambda _n: None)   # no uv either
    ok, note = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                         log=lambda _: None)
    assert ok is True, note
    assert deploy.DEPLOY_IMPORT_PROBE in note
    assert [c for c in sp.calls if deploy.DEPLOY_IMPORT_PROBE in " ".join(c)], \
        "the probe MUST run when no installer could execute"


def test_uv_is_tried_when_the_venv_has_no_pip(tmp_path, monkeypatch):
    """The fallback that makes a uv-managed release venv installable at all."""
    rel = _release_tree(tmp_path)
    sp = FakeSubprocess({"-m pip install": FakeProc(
        1, "", "No module named pip")})
    monkeypatch.setattr(deploy, "subprocess", sp)
    monkeypatch.setattr(deploy.shutil, "which", lambda _n: "/usr/bin/uv")
    ok, _ = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                      log=lambda _: None)
    uv_calls = [c for c in sp.calls if c and c[0] == "/usr/bin/uv"]
    assert ok is True
    assert uv_calls and uv_calls[0][:4] == ["/usr/bin/uv", "pip", "install",
                                            "--python"], uv_calls


def test_a_failed_import_probe_is_fail_closed_and_says_why(tmp_path,
                                                           monkeypatch):
    """H1, the measured one: the live release venv has no pydantic, so the probe
    is what stands between a merge and a crash-loop on RestartSec=5."""
    rel = _release_tree(tmp_path)
    sp = FakeSubprocess({deploy.DEPLOY_IMPORT_PROBE: FakeProc(
        1, "", "ModuleNotFoundError: No module named 'pydantic'")})
    monkeypatch.setattr(deploy, "subprocess", sp)
    ok, note = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                         log=lambda _: None)
    assert ok is False
    assert "No module named 'pydantic'" in note
    assert "crash-loop" in note and "unsupervised" in note


def test_a_pre_refactor_revision_installs_nothing_and_probes_nothing(tmp_path,
                                                                     monkeypatch):
    """Rollback stays possible (plan §8 step 7): a revision with no `vastlib`
    is a stdlib-only daemon, so the step is skipped, not failed."""
    rel = _release_tree(tmp_path, with_vastlib=False)
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "subprocess", sp)
    ok, note = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                         log=lambda _: None)
    assert ok and "pre-refactor tree" in note
    assert sp.calls == []


def test_a_missing_requirements_file_still_probes(tmp_path, monkeypatch):
    """The PROBE is the gate, not the install."""
    rel = _release_tree(tmp_path)
    (rel / "tools" / "vast" / "vastlib" / "requirements.txt").unlink()
    sp = FakeSubprocess()
    monkeypatch.setattr(deploy, "subprocess", sp)
    ok, _ = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                      log=lambda _: None)
    assert ok
    assert [c[1] for c in sp.calls] == ["-c"]


def test_a_subprocess_that_cannot_run_at_all_is_fail_closed(tmp_path,
                                                            monkeypatch):
    class Boom(FakeSubprocess):
        def run(self, argv, **kw):                       # noqa: ANN001, ANN003
            raise OSError("Text file busy")

    rel = _release_tree(tmp_path)
    monkeypatch.setattr(deploy, "subprocess", Boom())
    ok, note = deploy.ensure_deploy_deps(str(rel), "/venv/bin/python3",
                                         log=lambda _: None)
    assert ok is False and "Text file busy" in note


def test_the_probe_failure_leaves_the_old_unit_untouched(tmp_path, monkeypatch,
                                                         capsys, home):
    """THE ORDERING CLAIM. A dependency abort must happen BEFORE the unit is
    rewritten and before anything is restarted: the old unit keeps running the
    old revision, which is the only safe failure available here."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    unit = _unit_path(home)
    unit.parent.mkdir(parents=True)
    unit.write_text("OLD UNIT\n")
    sp = FakeSubprocess({deploy.DEPLOY_IMPORT_PROBE:
                         FakeProc(1, "", "ModuleNotFoundError: pydantic")})
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", sp)
    rc = deploy.cmd_deploy(_args(rel, src, no_restart=False))
    assert rc == deploy.DEPLOY_DEPS_RC == 4
    assert unit.read_text() == "OLD UNIT\n", "the unit was rewritten anyway"
    assert sp.argvs("systemctl") == [] and sp.argvs("loginctl") == []
    out = capsys.readouterr().out
    assert "the unit was NOT rewritten and nothing was restarted" in out


def test_force_does_not_bypass_the_dependency_probe(tmp_path, monkeypatch,
                                                    home):
    """`--force` is for an audit finding an operator has judged acceptable. "The
    interpreter cannot import the daemon" is not one of those."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    sp = FakeSubprocess({deploy.DEPLOY_IMPORT_PROBE:
                         FakeProc(1, "", "ImportError")})
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", sp)
    assert deploy.cmd_deploy(_args(rel, src, force=True)) == \
        deploy.DEPLOY_DEPS_RC
    assert not _unit_path(home).exists()


def test_the_deps_step_runs_before_the_unit_is_written_on_the_happy_path(
        tmp_path, monkeypatch, home):
    """The positive control for the ordering test above: same sequence, probe
    green — at probe time the unit must still be absent, and present after."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    unit = _unit_path(home)
    seen: list[bool] = []
    sp = FakeSubprocess(on_run=lambda argv: seen.append(unit.exists())
                        if "-c" in argv else None)
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", sp)
    assert deploy.cmd_deploy(_args(rel, src)) == 0
    assert seen == [False], "the probe ran after the unit was written"
    assert unit.exists()


# --------------------------------------------------------------------------- #
# 9. `render_unit` — the ExecStart literal (plan §4 unit-text contract)
# --------------------------------------------------------------------------- #
def test_render_unit_execstart_names_tools_vast_fleetd_py_serve():
    txt = deploy.render_unit("/usr/bin/python3", "/x/tools/vast/fleetd.py", "/x")
    assert "ExecStart=/usr/bin/python3 /x/tools/vast/fleetd.py serve" in txt
    assert "WorkingDirectory=/x" in txt
    assert "Documentation=file:///x/tools/vast/FLEETD_DESIGN.md" in txt
    assert "Restart=always" in txt and "RestartSec=5" in txt
    assert "Environment=" not in txt


def test_render_unit_carries_the_dry_run_env_only_when_asked():
    txt = deploy.render_unit("/p", "/s", "/r", dry_run=True)
    assert "Environment=FLEETD_DRY_RUN=1" in txt


# --------------------------------------------------------------------------- #
# 10. `_verify_live_rev` — the proof, across the socket
# --------------------------------------------------------------------------- #
def test_verify_live_rev_passes_on_a_match(monkeypatch, capsys):
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"rev": "newrev2", "pid": 7},
                                         None))
    assert deploy._verify_live_rev("newrev2", deadline_s=5.0) == 0
    assert "VERIFIED" in capsys.readouterr().out


def test_verify_live_rev_fails_when_the_daemon_reports_another_rev(monkeypatch,
                                                                   capsys):
    """`systemctl restart` returning 0 is not evidence. On 2026-08-07 it was 0
    and the daemon was unchanged."""
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"rev": "oldrev1", "pid": 1},
                                         None))
    clock = FakeClock()
    monkeypatch.setattr(deploy, "time", clock)
    assert deploy._verify_live_rev("newrev2", deadline_s=5.0) == 3
    out = capsys.readouterr().out
    assert "NOT VERIFIED" in out and "oldrev1" in out
    assert clock.slept == [2.0, 2.0, 2.0], "the poll cadence is 2s to a deadline"


def test_verify_live_rev_swallows_a_transport_error_and_keeps_polling(
        monkeypatch, capsys):
    """H12: the per-poll `try` is load-bearing — a daemon still starting refuses
    the socket, and that must be a retry, not a traceback out of the deploy."""
    calls: list[int] = []

    def boom(*a, **k):                                   # noqa: ANN002, ANN003
        calls.append(1)
        raise ConnectionRefusedError("no socket yet")

    monkeypatch.setattr(client, "fleet_request", boom)
    monkeypatch.setattr(deploy, "time", FakeClock())
    assert deploy._verify_live_rev("newrev2", deadline_s=1.0) == 3
    assert calls, "the polling loop never called fleet_request"
    assert "rev=None" in capsys.readouterr().out


def test_verify_live_rev_is_the_exit_code_of_a_restarting_deploy(tmp_path,
                                                                 monkeypatch,
                                                                 home):
    """The gate is wired to the command's return value, not just printed."""
    rel, src = _release_tree(tmp_path), _source_tree(tmp_path)
    monkeypatch.setattr(deploy, "_git", _happy_git())
    monkeypatch.setattr(deploy, "subprocess", FakeSubprocess())
    monkeypatch.setattr(client, "fleet_request",
                        lambda *a, **k: (True, {"rev": "stale", "pid": 1},
                                         None))
    monkeypatch.setattr(deploy, "time", FakeClock())
    assert deploy.cmd_deploy(_args(rel, src, no_restart=False)) == 3


def test_render_unit_bakes_the_deployed_tick_interval():
    """The tick quantizes every rung of eviction recovery, and a live daemon
    keeps whatever interval its unit was written with. So the operating point
    is set HERE — `daemon.TICK_S` stays the in-code default."""
    txt = deploy.render_unit("/p", "/s", "/r")
    assert "ExecStart=/p /s serve --interval 15" in txt
    assert deploy.UNIT_INTERVAL_S == 15.0
    from vastlib.fleet import daemon
    assert daemon.TICK_S == 45.0, "the CODE default is not the operating point"


def test_the_deployed_interval_depends_on_the_bundles_pacer():
    """A 15s tick raises the v0/bundles/ burst frequency 3x against a 5 req/s
    ceiling. The gate that makes that safe must exist, and under the limit."""
    from vastlib.core import api
    assert callable(api._bundles_pace)
    assert api.BUNDLES_MAX_RPS <= 4.0
