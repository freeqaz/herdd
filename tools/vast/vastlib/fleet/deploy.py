"""vastlib.fleet.deploy — move the RELEASE checkout, re-point the unit, PROVE it.

Why this exists
---------------
`cmd_install_unit` bakes `__file__` into `ExecStart`, so the daemon executes
whichever working tree installed it, forever after. Every incident this module
exists to stop came from that one fact (the full pair is recorded verbatim at
the `DEPLOY_*` constants below): a peer moved the shared checkout to their
branch and `fleet restart` reported success with none of the merged fix live;
then the worktree that mitigated it was deleted under a running daemon, which
survived only because Linux keeps a deleted inode mapped.

So the deploy target is a checkout that is (a) outside the repo, (b) its own
clone rather than a linked worktree, and (c) never used for interactive work —
and `deploy` is the only supported way to move it. Everything here is
fail-closed, in this order (plan-manifest H7): `prepare_deploy_ref` refusing
when neither fetch refreshed -> rc 1; uncommitted tracked changes -> rc 1;
divergence (HEAD not an ancestor of the target, listing what `checkout -B`
would DISCARD) -> rc 1; the tree did not actually move -> rc 1; `checkout_audit`
-> rc 2; the **dependency probe** -> rc 4 (NEW, see below); missing
`tools/vast/fleetd.py` or missing interpreter -> rc 1; and finally
`_verify_live_rev` -> rc 3 unless the *running* daemon reports the revision
just checked out.

The one functional delta from the flat original
-----------------------------------------------
`ensure_deploy_deps` is **NEW** — it is not a port of anything in `fleetd.py`,
and it is the reason `_deploy_python`'s "the module set is stdlib-only"
docstring had to be rewritten here rather than copied. Plan §6/§10 and cutover
step 6 anticipated it: `vastlib` depends on pydantic v2, `fleet.daemon` is a
composition root, and the live release venv (python 3.13.13 at
`~/.local/share/vast-fleetd/venv/bin/python3`) was MEASURED on 2026-08-16 to
have no pydantic at all. Restarting that unit onto the merged layout without
installing first is not a failed deploy, it is a crash-loop on `RestartSec=5`
with the fleet unsupervised. So the deploy pip-installs
`tools/vast/vastlib/requirements.txt` into the release venv and then probes
`import vastlib.fleet.daemon` in that interpreter, and a failure of either
ABORTS before the unit file is rewritten — the old unit keeps running the old
revision, which is the only safe failure here. Every line of it is marked
`NEW vs flat fleetd.py`.

What is deliberately NOT here
-----------------------------
* **No policy, no fleet state, no socket server.** `_verify_live_rev` is the
  only thing that talks to the daemon, and it does so as a CLIENT, through
  `fleet.client.fleet_request` — the same socket `fleet ping` uses.
* **No `git remote add`.** `_fetch_local_main` is a one-shot fetch of a path so
  that no absolute machine path is ever persisted in the release checkout's
  config; a moved source degrades to "the local fetch failed" instead of
  poisoning every later fetch.
* **No unification of `enable` and `enable --now`.** `cmd_deploy` calls `enable`
  WITHOUT `--now` and always activates via `restart`; `cmd_install_unit` (which
  stays with `daemon.py`) uses `enable --now`. `enable --now` no-ops on an
  already-active unit and reports success while the old config keeps running.
  The difference is deliberate (H8).
* **No `.env` key renames, ever.** `_ensure_env_link` symlinks the release
  checkout's `.env` at the MAIN checkout's file (verified live 2026-08-16:
  `~/.local/share/vast-fleetd/checkout/.env -> <repo>/.env`), and the daemon
  hot-reloads it in-process, so a renamed `FLEETD_*` key takes effect on the
  LIVE daemon the moment someone edits that file — no deploy needed.
* No `__file__` arithmetic in any function. The two path anchors are module
  constants (`TOOLS_VAST_DIR`, `_REPO_ROOT`) with the depth recomputed for this
  file's location and PINNED BY A TEST against `fleetd.py`'s own resolution —
  the 2026-08-09 regression here was one `dirname` too few, and it was silent
  in both places it broke.

Provenance: moved from `tools/vast/fleetd.py` (the 441-line deploy block, plan
§8 step 5, the `fleet/` decomposition), 2026-08-16. Behavior-preserving except
for `ensure_deploy_deps` and its call site, both marked. Mechanical exceptions,
documented at their sites: the path anchors above, and `client.fleet_request`
in place of `herdd.fleet_request` (module-attribute form, plan §8b, so the
existing patch idiom survives). `UNIT_TEMPLATE` / `render_unit` live HERE
rather than in `daemon.py` because `daemon.py` imports this module (`main`
dispatches `cmd_deploy`; `cmd_install_unit` needs `checkout_audit` +
`render_unit`) and the reverse edge would be a genuine import cycle — for the
same reason `UNIT_NAME`, `repo_root` and `dry_run_enabled` are NOT re-ported
here but read from their one home (see the comment block below). Every symbol carries its
`# moved-from:` marker (grammar: `vastlib/README.md` §2). The flat `fleetd.py`
copies stay live until step 6.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

from vastlib.fleet import client

# `tools/vast` — three dirnames up from `tools/vast/vastlib/fleet/deploy.py`,
# and the repo root two more above that.
#
# THIS IS THE ONE PIECE OF THE PORT THAT IS NOT TEXTUALLY VERBATIM, and it is
# what keeps `repo_root()` and `local_source_repo()` behaviorally verbatim.
# `fleetd.py` computes `_HERE = dirname(abspath(__file__))` (= `tools/vast`)
# and `repo_root()` as `dirname(dirname(_HERE))`. Copied unchanged into this
# file those expressions land two levels too deep: `_HERE` would be
# `tools/vast/vastlib/fleet` and the "repo root" would be `tools/vast`.
#
# Nothing would raise. `repo_root()` feeds the generated unit's
# `WorkingDirectory=` and `Documentation=`, and (in `daemon.py`) `_env_stat`'s
# `.env` path — which is exactly the 2026-08-09 regression, where one dirname
# too FEW put the root at `<repo>/tools`: the unit got
# `WorkingDirectory=<repo>/tools`, `_env_stat` stat'ed a path that never exists,
# and `_maybe_reload_env` compared None to None forever, so the hot-reload never
# fired once and nothing alarmed. Silent by design in the original; a comparison
# is the only thing that catches it, hence
# `test_vastlib_fleet_deploy.py::test_repo_root_matches_fleetd_computation`
# and `::test_naive_file_arithmetic_here_would_be_wrong`.
#
# Hoisted to module constants for the same reason `core.config._HERE` and
# `boxes.ssh._REPO_ROOT` are: the depth is a property of the module's location,
# not of the function, so a package that moves again fixes exactly two lines.
# moved-from: fleetd._HERE
TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(os.path.dirname(TOOLS_VAST_DIR))


# THREE SYMBOLS THE FLAT DEPLOY BLOCK USED ARE DELIBERATELY NOT RE-PORTED HERE,
# because each already has exactly one ported home and a second copy is how one
# contract becomes two literals that drift:
#
#   `UNIT_NAME`         -> `client.FLEET_UNIT_NAME` (the CLI reads it too, so it
#                          belongs to the protocol module; used inline below).
#   `repo_root()`       -> `daemon.repo_root()`; the arithmetic is `_REPO_ROOT`
#                          above, which is the same five dirnames from the same
#                          directory, and a test pins the two EQUAL.
#   `dry_run_enabled()` -> `daemon.dry_run_enabled()`; `_dry_run_enabled` below
#                          reads the same key for the same reason.
#
# In all three cases the missing edge is the same one: `daemon.py` imports THIS
# module (`main` dispatches `cmd_deploy`; `cmd_install_unit` needs
# `checkout_audit` + `render_unit`), so importing it back would be a cycle.
# `client` is imported freely — it has no edge to either of us.


# RE-ANCHORED, not verbatim (cli-surface.json H3, "depth change hazard"). The
# flat body was `os.path.join(os.path.dirname(os.path.abspath(__file__)),
# "fleetd.py")` evaluated inside `herdd.py`, i.e. `tools/vast/fleetd.py`. From
# `vastlib/fleet/` the same expression names `vastlib/fleet/fleetd.py`, which
# does not exist — and the failure is a `subprocess.call` on a missing path, not
# an import error, so `fleet install` / `fleet deploy` would fail at the moment
# an operator is trying to fix the daemon. `TOOLS_VAST_DIR` above is the one
# place that arithmetic lives; `test_vastlib_cli_fleet.py` pins the result
# against the flat `herdd._fleetd_script()`.
# moved-from: herdd._fleetd_script
def _fleetd_script() -> str:
    return os.path.join(TOOLS_VAST_DIR, "fleetd.py")


# NOT a port — `daemon.dry_run_enabled` is the ported symbol (see above). Same
# env key, same predicate; `test_vastlib_fleet_deploy.py` pins them equal in
# both states so this cannot drift into a deploy that writes an
# `Environment=FLEETD_DRY_RUN=1` line the daemon disagrees with.
def _dry_run_enabled() -> bool:
    return os.environ.get("FLEETD_DRY_RUN") == "1"


# --- the deploy checkout: a RELEASE path, not whichever tree you ran from -----
#
# `cmd_install_unit` bakes `__file__` into `ExecStart`, so the daemon executes
# that working tree forever after. Every incident this block exists to stop came
# from that one fact:
#
#   2026-08-07  the unit pointed at the SHARED checkout. A peer moved it to
#               their branch; `fleet restart` reported success and the merged
#               fix had zero of its four markers live.
#   2026-08-09  the mitigation for the above was a worktree pinned to main at
#               `<repo>/out/land-main`. Someone deleted it. The daemon survived
#               only because Linux keeps a deleted inode mapped (`/proc/<pid>/cwd`
#               read `... (deleted)`), and the next `fleet restart` would not
#               have deployed the wrong revision — it would have failed to exec
#               and crash-looped on RestartSec=5 with the fleet unsupervised.
#
# So the deploy target is a checkout that is (a) OUTSIDE the repo, so a cleanup
# sweep over `out/` cannot take it, (b) its own clone rather than a linked
# worktree, so `git worktree prune|remove` cannot take it either and it may hold
# `main` while a landing worktree also holds `main`, and (c) never used for
# interactive work. `deploy` is the only supported way to move it.
# moved-from: fleetd.DEPLOY_CHECKOUT_ENV
DEPLOY_CHECKOUT_ENV = "FLEETD_CHECKOUT"
# moved-from: fleetd.DEPLOY_CHECKOUT_DEFAULT
DEPLOY_CHECKOUT_DEFAULT = "~/.local/share/vast-fleetd/checkout"
# moved-from: fleetd.DEPLOY_PYTHON_DEFAULT
DEPLOY_PYTHON_DEFAULT = "~/.local/share/vast-fleetd/venv/bin/python3"
# moved-from: fleetd.DEPLOY_REF_DEFAULT
DEPLOY_REF_DEFAULT = "origin/main"
# The LOCAL repo's `main`, fetched into the release checkout as a plain
# remote-tracking ref. This is the DEFAULT deploy source — see
# `prepare_deploy_ref`: landed work on this workstation lives on local `main`
# and is never pushed unasked, so `origin/main` is routinely tens of commits
# behind and a bare `fleet deploy` shipped STALE code (2026-08-14: c524a5a9
# instead of the merge it was meant to ship, twice in one morning, each time
# needing a manual `git fetch <local-repo> main` into the checkout first).
# moved-from: fleetd.DEPLOY_LOCAL_REF
DEPLOY_LOCAL_REF = "local/main"
# moved-from: fleetd.DEPLOY_BRANCH
DEPLOY_BRANCH = "main"
# Path segments that mark a tree as scratch — the dirs this repo's sessions
# create and delete freely. `out/` is where land-main died; `.claude/worktrees`
# is the agent worktree pool, which is pruned wholesale.
# moved-from: fleetd.DEPLOY_SCRATCH_SEGMENTS
DEPLOY_SCRATCH_SEGMENTS = ("out", ".claude", "worktrees")

# --- NEW vs flat fleetd.py: the dependency step (plan §6/§10, cutover step 6) --
# The release venv must be able to IMPORT the package the unit is about to
# execute. See `ensure_deploy_deps`. Paths are relative to the deploy checkout
# so that a rollback to a pre-vastlib revision (§8 step 7's rollback path) finds
# no package, installs nothing, probes nothing, and still deploys.
DEPLOY_REQUIREMENTS_REL = os.path.join("tools", "vast", "vastlib",
                                       "requirements.txt")
DEPLOY_PACKAGE_REL = os.path.join("tools", "vast", "vastlib")
# BOTH halves of what the launcher runs, not just the engine. `fleetd.py::run()`
# imports `vastlib.cli._compose` to close the cross-ring seams before it
# dispatches (see the composition block at the bottom of `tools/vast/fleetd.py`),
# so a release venv that can import `fleet.daemon` but not `cli._compose` is
# exactly the crash-loop-on-RestartSec=5 shape this probe exists to refuse. Both
# names, one `-c`, so the probe still costs one process.
DEPLOY_IMPORT_PROBE = "import vastlib.fleet.daemon, vastlib.cli._compose"
# Its own exit code, so an operator (and the cutover runbook) can tell a
# dependency abort apart from an audit refusal (2), a fail-closed refusal (1)
# and an unverified restart (3).
DEPLOY_DEPS_RC = 4
DEPLOY_PIP_TIMEOUT_S = 900
DEPLOY_PROBE_TIMEOUT_S = 120


# moved-from: fleetd.deploy_checkout_path
def deploy_checkout_path() -> str:
    return os.path.expanduser(os.environ.get(DEPLOY_CHECKOUT_ENV)
                              or DEPLOY_CHECKOUT_DEFAULT)


# moved-from: fleetd._git
def _git(cwd: str, *args: str, timeout: float = 180) -> tuple[int, str]:
    """(rc, stdout) — never raises, so an audit can report instead of crashing."""
    try:
        p = subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip() or (p.stderr or "").strip()
    except Exception as e:                                   # noqa: BLE001
        return 1, str(e)[:300]


# moved-from: fleetd.checkout_audit
def checkout_audit(path: str) -> list[str]:
    """Every reason `path` is unfit to point a systemd unit at, worst first.

    Empty list == fit to deploy. This is the gate that would have caught both
    incidents above at install time rather than at the next restart."""
    bad: list[str] = []
    if not os.path.isdir(path):
        return [f"{path} does not exist"]
    dotgit = os.path.join(path, ".git")
    if not os.path.exists(dotgit):
        return [f"{path} is not a git checkout"]
    if not os.path.isdir(dotgit):
        bad.append("it is a LINKED WORKTREE (.git is a file) — a worktree is "
                   "removable by `git worktree remove|prune` and by any sweep "
                   "over its parent; that is exactly how out/land-main died")
    parts = set(os.path.abspath(path).split(os.sep))
    hit = sorted(parts & set(DEPLOY_SCRATCH_SEGMENTS))
    if hit:
        bad.append(f"its path contains scratch segment(s) {hit} — scratch dirs "
                   f"get swept; a release path must not live in one")
    rc, branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0:
        bad.append(f"cannot read HEAD: {branch}")
    elif branch != DEPLOY_BRANCH:
        bad.append(f"it is on {branch!r}, not {DEPLOY_BRANCH!r} — the deployed "
                   f"revision would be whatever branch someone last left here")
    rc, dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
    if rc == 0 and dirty:
        n = len(dirty.splitlines())
        bad.append(f"{n} tracked file(s) modified — a release path carries no "
                   f"uncommitted work, so `rev=` would not describe what runs")
    return bad


# moved-from: fleetd._deploy_python
def _deploy_python(explicit: str | None = None) -> str:
    """The interpreter to bake into `ExecStart`.

    CORRECTED AT THE PORT (the flat docstring is now false, plan-manifest H1).
    It read: "the fleetd/herdd module set is stdlib-only (yaml is optional,
    with a fallback parser), so this needs no dependencies — only a pinned
    VERSION, which is why we do not just take `/usr/bin/python3` and inherit
    whatever the distro rolls to next."

    The pinned-VERSION half still holds and is still why this does not take
    `/usr/bin/python3`. The stdlib-only half does not: the daemon this deploys
    is `vastlib.fleet.daemon`, and `vastlib` requires pydantic v2 (plan §6).
    The interpreter returned here is therefore also the one
    `ensure_deploy_deps` installs into and probes, and a deploy that cannot
    make `import vastlib.fleet.daemon` work in it does not restart anything.
    """
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    cand = os.path.expanduser(DEPLOY_PYTHON_DEFAULT)
    return cand if os.path.exists(cand) else sys.executable


# moved-from: fleetd._ensure_env_link
def _ensure_env_link(checkout: str, source_repo: str,
                     log: Callable[[str], object] = print) -> str:
    """`.env` is gitignored, so a fresh clone has none and the daemon comes up
    with no credentials at all. Link it to the canonical one — one source of
    truth, and `_env_stat` follows the symlink so N5 hot-reload still sees an
    edit. Returns a note for the caller to print.

    DIRECTION (verified live 2026-08-16, and the reason plan §4's grep-gate on
    `.env` key names binds against the MAIN checkout): the link is written INTO
    the release checkout and POINTS AT the source repo —
    `~/.local/share/vast-fleetd/checkout/.env -> <main-checkout>/.env`. The
    daemon re-reads that file in-process, so an edit there reaches the running
    daemon with no deploy at all; a renamed key reaches it the same way.
    `os.path.lexists` (not `exists`) so a dangling link counts as present and
    this never silently replaces one."""
    dst = os.path.join(checkout, ".env")
    if os.path.lexists(dst):
        return f".env: present ({'symlink -> ' + os.readlink(dst) if os.path.islink(dst) else 'regular file'})"  # noqa: E501 — operator line kept byte-identical to fleetd's
    src = os.path.join(source_repo, ".env")
    if not os.path.isfile(src):
        return (f".env: MISSING and no source at {src} — the daemon will have "
                f"no credentials; create {dst} before relying on this deploy")
    os.symlink(src, dst)
    log(f">> linked {dst} -> {src}")
    return f".env: linked -> {src}"


# moved-from: fleetd.local_source_repo
def local_source_repo(start: str | None = None) -> str:
    """The LOCAL repo to deploy FROM, derived at RUNTIME from where this file
    actually sits — the git toplevel containing `__file__`, never a baked path.

    The dirname answer (`_REPO_ROOT`, what `daemon.repo_root()` returns) is
    right for a normal checkout; this asks git instead so a linked worktree, a
    symlinked path or a relocated clone all resolve to the toplevel git
    considers authoritative. Falls back to `_REPO_ROOT` when git cannot answer
    (no git, not a repo), which keeps the old behaviour rather than failing the
    deploy.

    Ported form: the default start is `TOOLS_VAST_DIR` — the directory
    `fleetd.py`'s `_HERE` names, which after the port belongs to the thin
    launcher. Starting from this module's own directory would ask git about
    `tools/vast/vastlib/fleet`, which resolves to the same toplevel today but
    only by luck of both living in one repo."""
    here = start or TOOLS_VAST_DIR
    rc, out = _git(here, "rev-parse", "--show-toplevel")
    if rc == 0 and out and os.path.isdir(out):
        return os.path.abspath(out)
    return _REPO_ROOT


# moved-from: fleetd.resolve_deploy_ref
def resolve_deploy_ref(explicit_ref: str | None = None, *,
                       local_ok: bool = False, origin_ok: bool = False,
                       local_ref: str = DEPLOY_LOCAL_REF,
                       origin_ref: str = DEPLOY_REF_DEFAULT) -> tuple[str, str]:
    """PURE. Which revision a `fleet deploy` should resolve, and where it came
    from. Returns `(ref, source)` with `source` in:

      * `"explicit"` — the operator named `--ref`; it is resolved as given
        against a checkout both fetches have just refreshed, which is precisely
        the manual `git fetch <local-repo> main` step this removes.
      * `"local"`    — the DEFAULT when the local fetch succeeded. Local `main`
        is where landed work lives on this box.
      * `"origin"`   — the local fetch failed (or was skipped) and origin's did.
      * `"unfetched"`— neither refreshed anything; `origin_ref` is resolved as
        whatever the checkout already had. The caller decides whether that is a
        legitimate as-is deploy or a hard stop; this function only reports it,
        because a silent fall-through to a stale `origin/main` is the exact
        failure being fixed.

    Split out from `cmd_deploy` so the decision is testable without a git
    checkout, a systemd unit or a live daemon."""
    if explicit_ref:
        return explicit_ref, "explicit"
    if local_ok:
        return local_ref, "local"
    if origin_ok:
        return origin_ref, "origin"
    return origin_ref, "unfetched"


# moved-from: fleetd.deploy_ref_candidates
def deploy_ref_candidates(explicit_ref: str | None, *,
                          deploy_branch: str = DEPLOY_BRANCH,
                          local_ref: str = DEPLOY_LOCAL_REF,
                          origin_ref: str = DEPLOY_REF_DEFAULT) -> list[str]:
    """PURE. The refs `git rev-parse` should be tried against, in order, for an
    operator's `--ref`. Returns a list; the first that resolves wins.

    THE TRAP THIS EXISTS FOR (2026-08-16). `--ref main` — the most natural thing
    to type — used to be handed to `rev-parse` verbatim. In the release checkout
    `main` is `DEPLOY_BRANCH`: the branch `cmd_deploy` itself creates with
    `checkout -B`, which advances only when a deploy moves it. `rev-parse`
    resolves `refs/heads/main` ahead of any remote-tracking ref of the same
    name, so `--ref main` named the checkout's CURRENT position. The fetch
    worked, `local/main` and `origin/main` both moved to the new sha, and the
    deploy then "checked out" the sha it was already on, restarted fleetd onto
    unchanged code, and printed VERIFIED. Observed shipping 430df0f0 when
    589f84dd had landed; the deployed herdd.py contained zero occurrences of
    the function the deploy was for.

    So a BARE name resolves against the fetched refs FIRST and the deploy branch
    never at all — deploying "the branch we last deployed" is not a thing anyone
    means. A name that is already qualified (`origin/x`, `refs/...`) or is a sha
    is passed through untouched: the operator was specific, so be literal."""
    ref = str(explicit_ref or "").strip()
    if not ref:
        return []
    if "/" in ref or not re.fullmatch(r"[0-9A-Za-z._-]+", ref) \
            or re.fullmatch(r"[0-9a-fA-F]{7,40}", ref):
        return [ref]
    lp = local_ref.rsplit("/", 1)[0] if "/" in local_ref else "local"
    op = origin_ref.rsplit("/", 1)[0] if "/" in origin_ref else "origin"
    out = [f"{lp}/{ref}", f"{op}/{ref}"]
    if ref != deploy_branch:
        out.append(ref)
    return out


# moved-from: fleetd._resolve_deploy_target
def _resolve_deploy_target(
        checkout: str, ref: str,
        log: Callable[[str], object] = print) -> tuple[str | None, str | None]:
    """`(sha, resolved_ref)` for the ref an operator asked for, or `(None, None)`.

    Walks `deploy_ref_candidates` and takes the first that resolves in the
    checkout, saying which one won — a deploy that cannot name the ref it
    resolved is one nobody can audit."""
    tried: list[str] = []
    for cand in deploy_ref_candidates(ref):
        rc, sha = _git(checkout, "rev-parse", "--verify", "--quiet", f"{cand}^{{commit}}")
        if rc == 0 and sha:
            if cand != ref:
                log(f">> --ref {ref} resolved as {cand} (the deploy branch "
                    f"{DEPLOY_BRANCH!r} is never a deploy SOURCE — it is where "
                    f"the last deploy left the checkout)")
            return sha, cand
        tried.append(cand)
    log(f"!! cannot resolve {ref!r} in {checkout} (tried: {', '.join(tried)})")
    return None, None


# moved-from: fleetd._fetch_local_main
def _fetch_local_main(
        checkout: str, source_repo: str,
        log: Callable[[str], object] = print) -> tuple[bool, str | None]:
    """Fetch the LOCAL repo's `main` into the checkout as `local/main`.

    Deliberately a one-shot fetch of a path, NOT `git remote add`: nothing
    persists in the checkout's config, so no absolute machine path is ever
    written anywhere durable, and a moved/deleted source repo degrades to "the
    local fetch failed" instead of poisoning every later fetch."""
    if not source_repo or not os.path.isdir(os.path.join(source_repo, ".git")):
        return False, f"{source_repo} is not a git checkout"
    try:
        if os.path.samefile(source_repo, checkout):
            return False, "source repo IS the release checkout"
    except OSError:
        pass
    rc, out = _git(checkout, "fetch", "--quiet", source_repo,
                   f"+refs/heads/{DEPLOY_BRANCH}:refs/remotes/{DEPLOY_LOCAL_REF}",
                   timeout=1800)
    if rc != 0:
        return False, out
    log(f">> fetched local {source_repo}#{DEPLOY_BRANCH} -> {DEPLOY_LOCAL_REF}")
    return True, None


# moved-from: fleetd.prepare_deploy_ref
def prepare_deploy_ref(
        checkout: str, source_repo: str, explicit_ref: str | None = None,
        log: Callable[[str], object] = print
) -> tuple[str | None, str | None]:
    """Refresh the release checkout LOCAL-FIRST, then pick the ref to deploy.

    Order and rationale (FLEET_REVIEW_2026-08-14 item 5): the local repo is
    fetched first because that is where landed work is, origin is fetched
    additionally (not instead) so `origin/main` and any origin sha stay
    resolvable, and the SOURCE is printed because a deploy that cannot say
    which tree it shipped is how `c524a5a9` went out twice.

    Returns `(ref, source)`, or `(None, None)` when every fetch that was
    attempted failed — refusing there preserves the pre-2026-08-14 strictness
    (a failed origin fetch was already fatal) while letting either side cover
    for the other."""
    local_ok, local_err = _fetch_local_main(checkout, source_repo, log=log)
    if not local_ok:
        log(f">> local fetch skipped/failed ({local_err}) — falling back to origin")
    rc, remotes = _git(checkout, "remote")
    has_origin = rc == 0 and "origin" in remotes.split()
    origin_ok = False
    if has_origin:
        rc, out = _git(checkout, "fetch", "--quiet", "origin", timeout=1800)
        origin_ok = rc == 0
        if not origin_ok:
            log(f"!! origin fetch failed in {checkout}: {out}")
    else:
        log(f">> {checkout} has no `origin` remote")
    if not local_ok and not origin_ok and has_origin:
        log(f"!! no ref source could be refreshed in {checkout} — refusing to "
            f"deploy a revision nobody just fetched")
        return None, None
    ref, source = resolve_deploy_ref(explicit_ref, local_ok=local_ok,
                                     origin_ok=origin_ok)
    where = {"local": f"LOCAL repo {source_repo}",
             "origin": "ORIGIN (the local fetch did not run)",
             "explicit": ("operator --ref, resolved against "
                          + ("local+origin" if local_ok and origin_ok else
                             "local" if local_ok else
                             "origin" if origin_ok else "the checkout as-is")),
             "unfetched": "NOTHING FETCHED — the checkout as it already was"}[source]
    log(f">> deploy source: {where} -> {ref}")
    return ref, source


# moved-from: fleetd._clone_deploy_checkout
def _clone_deploy_checkout(
        path: str, source_repo: str,
        log: Callable[[str], object] = print) -> tuple[int, str]:
    """Bootstrap the release checkout. Cloned from the LOCAL repo first (same
    filesystem => hardlinked objects, so a 1.3 GB object store costs ~nothing
    and survives a later `gc` in the source by inode refcount), then repointed
    at the real upstream so it fetches from origin like any other clone."""
    rc, url = _git(source_repo, "remote", "get-url", "origin")
    if rc != 0:
        return 1, f"cannot read origin url from {source_repo}: {url}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f">> cloning {source_repo} -> {path} (hardlinked objects)")
    p = subprocess.run(["git", "clone", "--quiet", source_repo, path],
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return 1, f"clone failed: {(p.stderr or '')[:400]}"
    rc, out = _git(path, "remote", "set-url", "origin", url)
    if rc != 0:
        return 1, f"remote set-url failed: {out}"
    log(f">> origin -> {url}")
    return 0, url


# ---------------------------------------------------------------------------
# NEW vs flat fleetd.py — the dependency step. Not a port of anything; there is
# no counterpart in tools/vast/fleetd.py. Plan §6 ("fleetd's release venv must
# get it too — `fleet deploy` grows a `pip install -r
# tools/vast/vastlib/requirements.txt` step into its audited venv, and the
# deploy audit fails if the import probe fails (fail-closed, pre-restart)"),
# §10 ("pydantic missing from fleetd venv at deploy") and cutover step 6.
# ---------------------------------------------------------------------------
def ensure_deploy_deps(checkout: str, python: str,
                       log: Callable[[str], object] = print) -> tuple[bool, str]:
    """Make the release interpreter able to IMPORT what the unit will execute.

    NEW AT THE PORT, and the only functional delta in this module. Measured
    2026-08-16: the live release venv
    (`~/.local/share/vast-fleetd/venv/bin/python3`, 3.13.13) has no pydantic —
    `import pydantic` raises ModuleNotFoundError. `vastlib` requires pydantic v2
    (plan §6) and `vastlib.fleet.daemon` is what `fleetd.py serve` now imports,
    so restarting that unit onto the merged layout without installing first is
    not a failed deploy: it is a crash-loop on `RestartSec=5` with the fleet
    unsupervised and nothing watching the boxes. Exactly the 2026-08-09 shape,
    reached from the other direction.

    Two steps, in order, both against the interpreter `_deploy_python` chose:

      1. `pip install -r <checkout>/tools/vast/vastlib/requirements.txt`
      2. `<python> -c "import vastlib.fleet.daemon"`, with `PYTHONPATH` set to
         the checkout's `tools/vast` — the same directory the entry script
         inserts into `sys.path`, so the probe imports what `serve` will.

    The PROBE, not the install, is the gate: a venv that already satisfies the
    requirements passes with an install that changed nothing, and an install
    that "succeeded" while leaving the import broken still fails. Returns
    `(ok, note)`; the caller aborts on `ok is False` BEFORE writing the unit
    file, so the old unit keeps running the old revision.

    ROLLBACK IS NOT BROKEN BY THIS (plan §8 step 7 deploys the prior rev to roll
    back): a revision with no `tools/vast/vastlib` directory is a pre-refactor
    tree whose daemon really is stdlib-only, so both steps are skipped with a
    note rather than failing a deploy that would have worked. `--force` does not
    bypass any of it — force is for an operator who has judged an audit finding
    acceptable, and "the interpreter cannot import the daemon" is not a finding
    anyone can accept."""
    pkg = os.path.join(checkout, DEPLOY_PACKAGE_REL)
    if not os.path.isdir(pkg):
        return True, (f"deps: no {DEPLOY_PACKAGE_REL} in this revision "
                      f"(pre-refactor tree) — nothing to install, nothing to "
                      f"probe")
    req = os.path.join(checkout, DEPLOY_REQUIREMENTS_REL)
    install_note = ""
    if os.path.isfile(req):
        status, install_note = _install_requirements(python, req, log)
        if status == "failed":
            # An installer RAN and refused: still fail closed. A half-installed
            # venv is exactly what the probe cannot be trusted to judge.
            return False, f"deps: install of {req} failed: {install_note}"
        if status == "unavailable":
            # No installer could run at all (uv-managed venv with no `pip`
            # module, no uv on PATH). Measured 2026-08-17: that aborted the
            # deploy of a venv which already imported the daemon. Advisory —
            # the PROBE is the gate, per this function's contract.
            log(f"!! deps: could not install — {install_note}")
            log("   NOT fatal: continuing to the import probe, which is the gate")
    else:
        log(f">> {req} does not exist — installing nothing; the import probe "
            f"is the gate")
    env = dict(os.environ)
    tools_vast = os.path.join(checkout, "tools", "vast")
    env["PYTHONPATH"] = os.pathsep.join(
        [tools_vast] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        p = subprocess.run([python, "-c", DEPLOY_IMPORT_PROBE], cwd=checkout,
                           env=env, capture_output=True, text=True,
                           timeout=DEPLOY_PROBE_TIMEOUT_S)
    except Exception as e:                                   # noqa: BLE001
        return False, f"deps: import probe could not run: {str(e)[:300]}"
    if p.returncode != 0:
        tail = ((p.stderr or "") + (p.stdout or "")).strip()[-600:]
        # Name the install failure here too: when both fail, the install is
        # usually the cause and the probe only the symptom.
        pre = f"deps: install also failed ({install_note})\n" if install_note else ""
        return False, (f"{pre}deps: `{DEPLOY_IMPORT_PROBE}` FAILED in {python} "
                       f"(exit {p.returncode}) — restarting the unit onto this "
                       f"interpreter would crash-loop on RestartSec=5 with the "
                       f"fleet unsupervised:\n{tail}")
    return True, f"deps: `{DEPLOY_IMPORT_PROBE}` OK in {python}"


def _install_requirements(python: str, req: str,
                          log: Callable[[str], object],
                          ) -> tuple[str, str]:
    """Install `req` into `python`'s environment: pip, then `uv pip`.

    Returns `(status, note)` with status one of:
      * `ok`          — an installer ran and succeeded;
      * `failed`      — an installer RAN and refused (caller fails closed: a
                        half-installed venv is not something the probe can judge);
      * `unavailable` — nothing could even run, i.e. the venv has no `pip` module
                        and no uv is on PATH. Distinguished from `failed` because
                        this box's release venv is uv-managed and pip-less, and
                        treating that as a refusal blocked the deploy of a venv
                        that already imported the daemon (measured 2026-08-17).
    """
    missing_pip = ("no module named pip", "no module named 'pip'")
    attempts: list[list[str]] = [
        [python, "-m", "pip", "install", "--quiet",
         "--disable-pip-version-check", "-r", req]]
    uv = shutil.which("uv")
    if uv:
        attempts.append([uv, "pip", "install", "--python", python, "-r", req])
    notes: list[str] = []
    unavailable = True          # until some installer actually runs and answers
    for argv in attempts:
        log(f">> {' '.join(argv[:3])} … -r {req}")
        try:
            p = subprocess.run(argv, capture_output=True, text=True,
                               timeout=DEPLOY_PIP_TIMEOUT_S)
        except Exception as e:                               # noqa: BLE001
            notes.append(f"{os.path.basename(argv[0])}: could not run: "
                         f"{str(e)[:200]}")
            continue
        if p.returncode == 0:
            return "ok", ""
        out = ((p.stderr or "") + (p.stdout or "")).strip()
        notes.append(f"{os.path.basename(argv[0])} exited {p.returncode}: "
                     f"{out[-300:]}")
        if not any(m in out.lower() for m in missing_pip):
            unavailable = False        # it ran and refused on the merits
    if not uv:
        notes.append("uv not on PATH — no fallback installer")
    return ("unavailable" if unavailable else "failed"), "; ".join(notes)


# moved-from: fleetd.cmd_deploy
def cmd_deploy(args: argparse.Namespace) -> int:
    """Move the RELEASE checkout to a known revision, re-point the unit at it,
    restart, and PROVE the running daemon is that revision.

    The proof at the end is the point. `systemctl restart` returning 0 means
    systemd accepted the request, not that the code changed — the 2026-08-07
    incident had a green restart and an unchanged daemon. So this command exits
    non-zero unless the live `rev=` equals the revision it just checked out.

    Where the revision COMES FROM is `prepare_deploy_ref`: the local repo
    (derived at runtime from this file's git toplevel) is fetched first and its
    `main` is the default, with origin fetched as well and used as the fallback.
    Before 2026-08-14 this fetched origin only, so a bare `fleet deploy` on a
    box that never pushes shipped whatever origin last saw.

    ONE STEP IS NEW AT THE PORT (`ensure_deploy_deps`, marked at its call site
    below): the release venv is made able to import `vastlib.fleet.daemon`
    before anything is written or restarted, and a failure aborts with
    `DEPLOY_DEPS_RC`. Everything else is verbatim."""
    checkout = os.path.abspath(os.path.expanduser(args.checkout
                                                  or deploy_checkout_path()))
    source_repo = os.path.abspath(os.path.expanduser(args.source
                                                     or local_source_repo()))

    if not os.path.isdir(os.path.join(checkout, ".git")):
        rc, msg = _clone_deploy_checkout(checkout, source_repo)
        if rc != 0:
            print(f"!! {msg}")
            return 1

    ref, source = prepare_deploy_ref(checkout, source_repo, args.ref)
    if ref is None:
        return 1
    rc, dirty = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if rc == 0 and dirty:
        print(f"!! {checkout} has uncommitted tracked changes — refusing to "
              f"deploy a tree whose `rev=` would not describe what runs:\n{dirty}")
        return 1
    target, resolved = _resolve_deploy_target(checkout, ref)
    if target is None:
        return 1
    # DIVERGENCE. `checkout -B` force-moves the deploy branch, which silently
    # discards any commit made in the release checkout. The dirty check above
    # only covers UNCOMMITTED work. Refuse rather than eat it — a release
    # checkout should never have local commits, so this firing means something
    # is not what the operator thinks it is.
    rc, before_out = _git(checkout, "rev-parse", "HEAD")
    before = before_out if rc == 0 else None
    if before and before != target and not args.force:
        anc, _ = _git(checkout, "merge-base", "--is-ancestor", before, target)
        if anc != 0:
            rc, extra = _git(checkout, "log", "--oneline", f"{target}..{before}")
            print(f"!! {checkout} HEAD {before[:12]} is NOT an ancestor of "
                  f"{resolved} {target[:12]} — deploying would DISCARD:\n{extra}\n"
                  f"   (--force to overwrite the release checkout anyway)")
            return 1
    # -B so a fresh clone (or a checkout someone detached) lands on `main`
    # pointing at the ref, rather than leaving a detached HEAD nobody can audit.
    rc, out = _git(checkout, "checkout", "-B", DEPLOY_BRANCH, target)
    if rc != 0:
        print(f"!! checkout failed: {out}")
        return 1
    # THE TREE ACTUALLY MOVED. Asserted against the RESOLVED TARGET, not against
    # whatever HEAD happens to say afterwards. Reading HEAD and calling it
    # "expected" is what made the 2026-08-16 no-op deploy self-confirming: the
    # deployed revision was expected by definition, so the check could not fail.
    rc, head_full = _git(checkout, "rev-parse", "HEAD")
    if rc != 0 or head_full != target:
        print(f"!! {checkout} did NOT move: asked for {resolved} "
              f"{target[:12]}, HEAD is {(head_full or '?')[:12]}. Refusing to "
              f"restart fleetd onto a tree that is not the requested revision.")
        return 1
    # Abbreviated with git's OWN rule, not sliced: `git_rev()` — what the daemon
    # will report over the socket — uses `rev-parse --short`, whose length grows
    # to stay unambiguous. A hardcoded 7 would fail verification on any repo
    # where it grew to 8.
    rc, head = _git(checkout, "rev-parse", "--short", "HEAD")
    if rc != 0:
        print(f"!! cannot read HEAD: {head}")
        return 1
    print(f">> {checkout} @ {head} ({resolved}, source={source})")
    if before == target:
        print(f".. already at {head} — this deploy changes no CODE; the unit "
              f"rewrite and restart below still run")

    bad = checkout_audit(checkout)
    if bad and not args.force:
        print(f"!! the deploy checkout is unfit ({len(bad)} reason(s)):")
        for b in bad:
            print(f"   - {b}")
        print("   (--force to install anyway)")
        return 2
    print(_ensure_env_link(checkout, source_repo))

    script = os.path.join(checkout, "tools", "vast", "fleetd.py")
    if not os.path.isfile(script):
        print(f"!! {script} does not exist — refusing to write a unit whose "
              f"ExecStart cannot exec (that is the crash-loop shape)")
        return 1
    python = _deploy_python(args.python)
    if not os.path.exists(python):
        print(f"!! interpreter {python} does not exist")
        return 1
    # --- NEW vs flat fleetd.py (see `ensure_deploy_deps`) --------------------
    # Deliberately placed HERE: after the interpreter is known and BEFORE the
    # unit file is written, so a dependency failure leaves the installed unit
    # byte-identical and the old daemon running. `--force` does not bypass it.
    deps_ok, deps_note = ensure_deploy_deps(checkout, python)
    print(deps_note if deps_ok else f"!! {deps_note}")
    if not deps_ok:
        print(f"   the unit was NOT rewritten and nothing was restarted — the "
              f"running daemon keeps its current revision. Fix the release "
              f"venv ({python}) and re-run the deploy.")
        return DEPLOY_DEPS_RC
    # --- end NEW ------------------------------------------------------------
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    unit_path = os.path.join(unit_dir, client.FLEET_UNIT_NAME)
    text = render_unit(python, script, checkout,
                       dry_run=args.dry_run or _dry_run_enabled())
    with open(unit_path, "w") as f:
        f.write(text)
    print(f">> wrote {unit_path}")
    for line in text.splitlines():
        if line.startswith(("ExecStart=", "WorkingDirectory=", "Environment=")):
            print(f"   {line}")

    if args.no_restart:
        print("next: systemctl --user daemon-reload && "
              f"systemctl --user restart {client.FLEET_UNIT_NAME}")
        return 0
    subprocess.call(["systemctl", "--user", "daemon-reload"])
    # `enable` (no --now) is idempotent and cannot mask a config change; the
    # activation is ALWAYS a restart, never `enable --now`, which no-ops on an
    # already-active unit and reports success while running the old config.
    subprocess.call(["systemctl", "--user", "enable", client.FLEET_UNIT_NAME],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = subprocess.call(["systemctl", "--user", "restart",
                          client.FLEET_UNIT_NAME])
    print(f"systemctl --user restart {client.FLEET_UNIT_NAME} -> rc={rc}")
    if rc != 0:
        return rc
    subprocess.call(["loginctl", "enable-linger", os.environ.get("USER") or ""],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return _verify_live_rev(head, deadline_s=args.verify_timeout)


# moved-from: fleetd._verify_live_rev
def _verify_live_rev(expect_rev: str, deadline_s: float = 60.0) -> int:
    """Poll the daemon's own socket until it reports a rev, then compare.

    Deliberately not a log line and not an exit code: the daemon is asked what
    revision it is RUNNING, over the same socket `fleet ping` uses.

    THREE THINGS MUST AGREE END TO END or this gate lies in one direction or the
    other: git's own abbreviation length (`rev-parse --short`, never a slice to
    7 — `cmd_deploy` says so at the call site), `daemon.git_rev()`'s
    `repo_root()`, and `Server.handle`'s ping payload key `rev`.

    Ported form: `client.fleet_request` in place of `herdd.fleet_request` —
    called as a MODULE ATTRIBUTE so the existing `monkeypatch.setattr(...,
    "fleet_request", ...)` idiom keeps steering it (plan §8b)."""
    end = time.time() + max(5.0, float(deadline_s or 60.0))
    last: Any = None
    while time.time() < end:
        try:
            ok, data, _err = client.fleet_request("ping", _timeout=5, _retries=0)
        except Exception:                                    # noqa: BLE001
            ok, data = False, None
        if ok and isinstance(data, dict) and data.get("rev"):
            last = data
            if client.rev_matches(data.get("rev"), expect_rev):
                print(f"VERIFIED: fleetd is live at rev={data['rev']} "
                      f"pid={data.get('pid')} (expected {expect_rev})")
                return 0
        time.sleep(2.0)
    got = (last or {}).get("rev")
    print(f"!! NOT VERIFIED: expected rev={expect_rev}, daemon reports "
          f"rev={got!r} after {deadline_s:g}s. The unit was written but the "
          f"running daemon is not that revision — check "
          f"`systemctl --user status {client.FLEET_UNIT_NAME}`.")
    return 3


# The DEPLOYED tick interval. `daemon.TICK_S` stays 45s — this is the operating
# point, not the default, and it is set here because the unit is the only thing
# that decides what the live fleet runs at.
#
# 15s because the tick quantizes every rung of eviction recovery: detection,
# classify, relaunch, re-watch. Measured ~25 min of nightly downtime at 45s,
# projected 8-10 at 15s.
#
# THIS DEPENDS ON `core.api._bundles_pace`. The tick issues 2 POST v0/bundles/
# per jobs box back to back and that endpoint's ceiling is 5 req/s; a 15s tick
# raises the burst FREQUENCY 3x. Without the pacer this number reintroduces
# 429s that `request_soft`'s backoff would swallow silently — a market read
# lost, not an error raised. Do not raise the rate here without checking it.
UNIT_INTERVAL_S = 15.0

# moved-from: fleetd.UNIT_TEMPLATE
UNIT_TEMPLATE = """\
[Unit]
Description=vast fleet-supervision daemon (fleetd)
Documentation=file://{repo}/tools/vast/FLEETD_DESIGN.md
After=network-online.target

[Service]
Type=simple
ExecStart={python} {script} serve --interval {interval:g}
WorkingDirectory={repo}
Restart=always
RestartSec=5
{env_lines}
[Install]
WantedBy=default.target
"""


# moved-from: fleetd.render_unit
def render_unit(python: str, script: str, repo: str,
                dry_run: bool = False,
                interval: float = UNIT_INTERVAL_S) -> str:
    """The systemd USER unit text. Generated AT INSTALL TIME on the operator's
    machine — never committed, so no absolute machine paths enter git.
    Restart=always (S5): the fleet must not go unwatched because the daemon
    exited cleanly for any reason.

    `{script} serve` is why `tools/vast/fleetd.py` must keep its exact path and
    its `serve` subcommand (plan §3 Zone E, §4). The installed unit is NOT
    rewritten by a merge — the cutover has to re-run `deploy` for a new layout
    to take effect, which is also why the daemon keeps running the old flat
    `fleetd.py` from its own checkout while this branch is developed. The same
    fact is why `--interval` is baked here rather than left to the in-code
    default: a live daemon keeps its old interval until somebody re-deploys."""
    env = "Environment=FLEETD_DRY_RUN=1\n" if dry_run else ""
    return UNIT_TEMPLATE.format(python=python, script=script, repo=repo,
                                interval=interval, env_lines=env)
