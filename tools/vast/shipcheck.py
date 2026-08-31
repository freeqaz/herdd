#!/usr/bin/env python3
"""Derived guards over tools/vast/ship_manifest.txt — what reaches a rented box.

Two checks, both stdlib-only and $0 (no B2, no vast API, no network):

  imports  IMPORT CLOSURE. Every module the manifest ships out of a FLAT
           bare-import directory (tools/witness, tools/pipeline) is parsed for
           its TOP-LEVEL imports. An import that resolves to a sibling .py which
           exists locally but is NOT shipped is a gap: the box gets the importer
           and not the imported, so the job dies at its own import gate.

           The bake already catches this late (eval-env/smoke_check.py check (f)
           imports the closure against the pruned tree), but `herdd sync` had
           NO equivalent — which is how the 2026-07-30 frontier wave shipped
           witness_frontier.py -> inplace_build and score_frontier_resumable.py
           -> c2rs_prefilter with neither module in the manifest (fixed
           36602adb). The manifest's own "matches no tracked file" guard cannot
           see this shape: every listed pathspec was valid; the problem was a
           pathspec that was never listed.

           TWO manifests, one detector (2026-08-14). `imports` now runs the
           same closure over the JOBD BUNDLE as well — `_job_attach_files()` in
           vastlib/jobs/bundle.py (it lived in herdd.py until the vast-tooling
           refactor), the hardcoded list flattened into /workspace/jobd/ on
           every jobs box. It was guarded only by a name-pinning test, so
           jobmeta.py's module-scope `from bidpolicy import DEFEND_*` shipped
           without bidpolicy.py and every `python3 jobd.py` call on every box
           launched after 84d09ab1 died silently (jobd.sh swallows them), which
           left boxes looking alive and idle while billing. `shipcheck jobd`
           runs that half alone (the cheap pre-launch gate).

  env      ENV STALENESS. Ship-manifest files whose CONTENT differs from what
           the baked env actually shipped are present on the WORKSTATION and
           absent from the box's baked tree, so a job requesting them fails
           on-box (2026-07-30 round 1: SCORE_BACKEND=pool against
           env-20260729-1617, which predates the pool scorer).

           The comparison is per-file sha256 against the MANIFEST's
           `shipped_files` map (bake.sh takes it from the finished tar). That
           map is what makes the gate honest on a DIRTY tree: the bake ships
           working-tree bytes, so an uncommitted file is usually a file that
           ALREADY SHIPPED. The older rev-diff heuristic could not tell, and
           reported every peer session's uncommitted file as STALE ENV — which
           is how both live 2026-07-30 preflights ended up run with
           --allow-stale-env. Identical bytes are never staleness; differing,
           never-shipped and vanished files still all fail.

           Envs baked before 2026-07-30 carry no `shipped_files`; those fall
           back to the legacy rev diff, labelled as approximate in the output.

           The verdict is advisory about ONE thing only: a `herdd sync` after
           the bake overlays the box additively and this check cannot see that
           it happened — it reports the DELTA and says "sync required unless
           already synced".

CLI:
    python3 tools/vast/shipcheck.py imports [--repo DIR]   (manifest + jobd bundle)
    python3 tools/vast/shipcheck.py jobd    [--repo DIR]   (jobd bundle only)
    python3 tools/vast/shipcheck.py env     [--repo DIR]
                                            [--env-manifest FILE | --env-version VER]
    python3 tools/vast/shipcheck.py all     [...]

Exit: 0 clean · 1 a check has teeth and failed · 2 usage / unreadable manifest.
An UNRESOLVABLE env manifest is a NOTE with exit 0 (nothing to compare against
is not evidence of freshness — but it is also not a finding).
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

MANIFEST_REL = "tools/vast/ship_manifest.txt"

# Directories whose modules are imported BY BARE NAME on a box (sys.path is
# pointed at the dir; there is no package). `dir -> the sibling dirs a bare
# `import X` inside it can resolve to`. tools/pipeline reaches mining/ through
# `_pipeline_path.setup()`, so mine_attempts.py is a legal bare-name target.
FLAT_NAMESPACES = {
    # `tools/vast` was UNSCANNED until 2026-08-14, which is why the ship
    # manifest carried the same bidpolicy gap as the jobd bundle below and no
    # check saw it: the detector was correct and simply not pointed here.
    "tools/vast": ("tools/vast", "tools/vast/onstart"),
    "tools/vast/onstart": ("tools/vast/onstart", "tools/vast"),
    "tools/witness": ("tools/witness",),
    "tools/pipeline": ("tools/pipeline", "tools/pipeline/mining"),
    "tools/pipeline/mining": ("tools/pipeline", "tools/pipeline/mining"),
}

# The SECOND "what ships" manifest in this repo. `ship_manifest.txt` above is
# declarative and guarded by the closure check; `_job_attach_files()`
# (`vastlib/jobs/bundle.py`, re-exported as `herdd._job_attach_files` for the
# flat callers) is a hardcoded Python list that defines the jobd bundle, and
# until 2026-08-14 it
# was guarded only by a name-pinning test — which is how jobmeta.py's
# module-scope `from bidpolicy import DEFEND_*` (84d09ab1) shipped to boxes with
# no bidpolicy.py beside it and killed every `python3 jobd.py` call on them.
#
# The bundle is delivered FLAT into /workspace/jobd/, so a file from
# tools/vast/onstart/ and one from tools/vast/ become siblings on the box: a
# bare import from either can legally resolve to either dir. (Basename
# uniqueness across the two is asserted separately by
# test_broker_env.test_jobd_bundle_flat_names_unique.)
JOBD_BUNDLE_DIRS = ("tools/vast", "tools/vast/onstart")
JOBD_FLAT_NAMESPACES = {d: JOBD_BUNDLE_DIRS for d in JOBD_BUNDLE_DIRS}
# Where the jobd bundle's file list lives, for the FIX hint (it is code, not a
# manifest file, so the fix is an edit to a function).
JOBD_MANIFEST_HINT = "vastlib/jobs/bundle.py:_job_attach_files()"

# The bundle whose rev the eval-env MANIFEST records for this repo.
ENV_MANIFEST_REPO = "upstream-monorepo"
# Where bake.sh leaves its versioned manifests (gitignored build output).
DIST_REL = "out/eval-env/dist"


class ShipcheckError(Exception):
    """Unreadable manifest / not a git checkout — a usage-class failure."""


# --------------------------------------------------------------------------- #
# manifest -> the shipped file set
# --------------------------------------------------------------------------- #
def parse_manifest(repo_root: str):
    """(includes, excludes) git pathspecs from ship_manifest.txt. Mirrors
    herdd._load_ship_manifest and bake.sh's parse_ship_manifest — one format,
    three readers, so keep this in step with both."""
    path = os.path.join(repo_root, MANIFEST_REL)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as e:
        raise ShipcheckError(f"can't read ship manifest {path}: {e}") from e
    includes, excludes = [], []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        (excludes if line.startswith("!") else includes).append(line.lstrip("!").strip())
    if not includes:
        raise ShipcheckError(f"{path} has no include pathspecs")
    return includes, excludes


def _git(repo_root: str, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", repo_root, *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def shipped_files(repo_root: str) -> set[str]:
    """The tracked files the manifest actually ships, as repo-relative paths.
    `git ls-files` is the enumerator on every consumer, so it is the enumerator
    here: untracked files can never ship, and neither can they close a gap."""
    includes, excludes = parse_manifest(repo_root)
    specs = list(includes) + [f":(exclude){e}" for e in excludes]
    rc, out = _git(repo_root, "ls-files", "-z", "--", *specs)
    if rc != 0:
        raise ShipcheckError(f"git ls-files failed in {repo_root}")
    return {p for p in out.split("\0") if p}


# --------------------------------------------------------------------------- #
# check 1: import closure
# --------------------------------------------------------------------------- #
def top_level_imports(path: str) -> set[str]:
    """Bare module names imported at MODULE TOP LEVEL by the file at `path`.

    Only `tree.body` is walked, deliberately: an import nested in `try:` /
    `if:` / a function is a GUARDED or lazy import whose absence the module
    itself tolerates (`gbt`'s sklearn, `anthropic_client`'s anthropic), and
    flagging those would make the gate cry wolf. `from . import x` and
    `from pkg.mod import x` are skipped — only the flat first component of an
    absolute import can name a sibling .py.
    """
    try:
        with open(path, "rb") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def import_closure_gaps(repo_root: str, shipped: set[str] | None = None,
                        namespaces: dict | None = None) -> list[dict]:
    """Each gap: a SHIPPED module in a flat namespace whose top-level import
    resolves to a sibling .py that exists locally but does not ship.

    Sorted, one row per (importer, missing) pair, so the fix is a copy-paste of
    `missing` into ship_manifest.txt.

    `namespaces` overrides FLAT_NAMESPACES so the SAME resolution logic can be
    aimed at a different manifest — the jobd bundle uses JOBD_FLAT_NAMESPACES.
    Do not fork this function per manifest: one detector, N manifests, or the
    next manifest is unguarded exactly the way the jobd bundle was.
    """
    ship = shipped_files(repo_root) if shipped is None else shipped
    ns = FLAT_NAMESPACES if namespaces is None else namespaces
    gaps = []
    for rel in sorted(ship):
        d = os.path.dirname(rel)
        search = ns.get(d)
        if not search or not rel.endswith(".py"):
            continue
        for name in sorted(top_level_imports(os.path.join(repo_root, rel))):
            for cand_dir in search:
                cand = f"{cand_dir}/{name}.py"
                if cand in ship:
                    break                      # shipped: closed
                if os.path.isfile(os.path.join(repo_root, cand)):
                    gaps.append({"importer": rel, "module": name, "missing": cand})
                    break                      # local but unshipped: the gap
    return gaps


def _jobd_bundle_source(repo_root: str) -> tuple[str, str]:
    """(module-name, file) of the module that OWNS `_job_attach_files` in
    `repo_root` — `vastlib/jobs/bundle.py` since the vast-tooling refactor,
    `herdd.py` in a pre-package checkout.

    The fallback is not politeness: `--repo DIR` is routinely pointed at ANOTHER
    checkout (a box's tree, a peer worktree, `main` while the refactor lands),
    and those still carry the list in the fat `herdd.py`. Order matters —
    prefer the package home, because a post-refactor `herdd.py` re-exports the
    name rather than defining it (see `jobd_bundle_files` for why that is not
    good enough).
    """
    tools_vast = os.path.join(repo_root, "tools", "vast")
    for name, rel in (("_shipcheck_jobd_bundle",
                       os.path.join("vastlib", "jobs", "bundle.py")),
                      ("_shipcheck_herdd", "herdd.py")):
        src = os.path.join(tools_vast, rel)
        if os.path.isfile(src):
            return name, src
    raise ShipcheckError(
        f"no vastlib/jobs/bundle.py and no herdd.py under {tools_vast} — "
        "cannot read the jobd bundle")


def jobd_bundle_files(repo_root: str) -> set[str]:
    """The jobd bundle as repo-relative paths, DERIVED from the one source of
    truth (`vastlib.jobs.bundle._job_attach_files()`) — never a second hand-kept
    list, which is the defect this whole module exists to catch.

    The owner is imported lazily and BY PATH so shipcheck stays stdlib-only at
    module scope and so `--repo DIR` checks THAT checkout's bundle, not this
    process's. Raises ShipcheckError rather than returning an empty set: a
    bundle we could not read is not a bundle with no gaps.

    WHY THIS LOADS `vastlib/jobs/bundle.py` AND NOT `herdd.py`
    ------------------------------------------------------------
    Until the refactor it loaded `herdd.py`, which DEFINED the list. The thin
    launcher only re-exports it (`herdd.py`: `from vastlib.jobs.bundle import
    _job_attach_files`), and an already-imported package wins over any
    `sys.path` insert — so in a process that has already imported `vastlib`
    (every `herdd shipcheck …`, because `vastlib.cli.shipcheck` imports this
    module from inside a live vastlib), path-loading the TARGET checkout's
    launcher silently yields THIS checkout's function. Measured 2026-08-16
    against a copy of the tree whose `bundle.py` returned a sentinel: cold
    process -> sentinel (right checkout); warm process -> this checkout's 16
    files, relpath'd against the foreign root into `../../…` keys, i.e. the
    "empty/garbage bundle reads as no gaps" shape the module exists to prevent.
    Path-loading the owning module executes THAT file, so its `__file__`-derived
    paths are the target's in both cases (re-measured: sentinel, warm and cold).
    Its `vastlib.*` helper imports still resolve to the warm copy; only the
    manifest itself is checkout-scoped, which is the guarantee the callers need.
    """
    import importlib.util
    modname, src = _jobd_bundle_source(repo_root)
    spec = importlib.util.spec_from_file_location(modname, src)
    mod = importlib.util.module_from_spec(spec)
    tools_vast = os.path.join(repo_root, "tools", "vast")
    sys.path.insert(0, tools_vast)          # its bare-name Zone S siblings
    try:
        spec.loader.exec_module(mod)
        files = mod._job_attach_files()
    except Exception as e:                            # noqa: BLE001 — degrade loudly
        raise ShipcheckError(f"could not read _job_attach_files() from {src}: {e}") from e
    finally:
        if sys.path and sys.path[0] == tools_vast:
            sys.path.pop(0)
    out = set()
    for f in files:
        rel = os.path.relpath(os.path.abspath(f), repo_root).replace(os.sep, "/")
        out.add(rel)
    if not out:
        raise ShipcheckError("_job_attach_files() returned nothing")
    return out


def jobd_import_closure_gaps(repo_root: str,
                             shipped: set[str] | None = None) -> list[dict]:
    """Import closure over the JOBD BUNDLE (the second manifest). Same detector,
    different manifest and different flat namespace — see JOBD_FLAT_NAMESPACES."""
    ship = jobd_bundle_files(repo_root) if shipped is None else shipped
    return import_closure_gaps(repo_root, ship, namespaces=JOBD_FLAT_NAMESPACES)


def format_import_gaps(gaps: list[dict], manifest_hint: str = MANIFEST_REL,
                       what: str = "manifest") -> list[str]:
    if not gaps:
        return [f">> import closure OK ({what}): every top-level import of a shipped "
                "flat-namespace module is itself shipped"]
    out = [f"!! IMPORT CLOSURE BROKEN ({what}): {len(gaps)} shipped module(s) "
           f"top-import a module that does NOT ship —"]
    for g in gaps:
        out.append(f"     {g['importer']}  ->  import {g['module']}   "
                   f"(exists locally at {g['missing']}, not in {manifest_hint})")
    seen, adds = set(), []
    for g in gaps:
        if g["missing"] not in seen:
            seen.add(g["missing"])
            adds.append(g["missing"])
    out.append(f"   FIX: add to {manifest_hint} —")
    out += [f"     {p}" for p in adds]
    out.append("   (a box gets the importer without the imported: the job dies at "
               "its own import gate, mid-wave, on a GPU you are paying for)")
    return out


# --------------------------------------------------------------------------- #
# check 2: env staleness
# --------------------------------------------------------------------------- #
_VER_RE = re.compile(r"^\d{8}-\d{4}-[0-9a-f]{8}$")


def resolve_env_manifest(repo_root: str, path: str | None = None,
                         version: str | None = None) -> str | None:
    """Locate a baked eval-env MANIFEST json. Explicit --env-manifest wins, then
    --env-version against the local dist dir, then the NEWEST local dist
    manifest (this workstation's last bake — in our workflow that is the env the
    boxes are running). None = nothing to compare against."""
    if path:
        return path if os.path.isfile(path) else None
    dist = os.path.join(repo_root, DIST_REL)
    if version:
        p = os.path.join(dist, f"env-{version}.MANIFEST.json")
        return p if os.path.isfile(p) else None
    hits = sorted(glob.glob(os.path.join(dist, "env-*.MANIFEST.json")))
    return hits[-1] if hits else None           # names sort by YYYYMMDD-HHMM


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shipped_files_digest(files: dict) -> str:
    """Roll-up over a path -> sha256 map. Byte-identical to bake.sh's, and it
    must stay so: it is the cheap equality fast-path on both sides."""
    roll = hashlib.sha256()
    for p in sorted(files):
        roll.update(f"{p}\0{files[p]}\n".encode())
    return roll.hexdigest()


def _uncommitted_shipped(repo_root: str, specs: list[str]) -> list[str]:
    rc, out = _git(repo_root, "status", "--porcelain", "--", *specs)
    if rc != 0:
        return []
    return sorted(l[3:].strip() for l in out.splitlines() if l.strip())


def check_env_staleness(repo_root: str, manifest_path: str | None) -> dict:
    """Does this checkout's ship-manifest set still match what the env shipped?

    Two comparison modes, recorded in `mode`:

      content     (MANIFEST carries `shipped_files`) — per-file sha256 against
                  the bytes the bake actually tarred. Exact on a dirty tree,
                  which is the whole reason it exists: an UNCOMMITTED file whose
                  bytes match the bake already shipped and is NOT staleness. The
                  rev-diff heuristic could not tell, so it failed the gate every
                  time any peer session had a ship-manifest file open — and a
                  gate that cries wolf is one operators route around.
      git-legacy  (older env, no `shipped_files`) — the original rev diff plus
                  `git status`, approximate on a dirty tree. Unchanged, and
                  labelled as approximate in the formatted output.

    Verdicts:
      unresolved  no MANIFEST to compare against (NOT evidence of freshness)
      unknown-rev the recorded rev is not in this checkout (legacy mode only)
      fresh       every shipped file matches (content) / nothing moved (legacy)
      stale       -> `herdd sync` or re-bake. Content mode fails on any of:
                  `differs` (box holds different bytes), `added` (never shipped
                  at all), `removed` (shipped then deleted/renamed here, so the
                  box carries a file this checkout no longer has).
    """
    res = {"verdict": "unresolved", "manifest": manifest_path, "version": None,
           "created_utc": None, "rev": None, "dirty_bake": False, "mode": None,
           "changed": [], "uncommitted": [],
           "differs": [], "added": [], "removed": [],
           "uncommitted_identical": [], "head_pinned": [], "n_compared": 0}
    if not manifest_path:
        return res
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            man = json.load(fh)
    except (OSError, ValueError) as e:
        res["error"] = f"unreadable env manifest: {e}"
        return res
    repo_rec = (man.get("repos") or {}).get(ENV_MANIFEST_REPO) or {}
    rev = repo_rec.get("rev")
    res.update(version=man.get("version"), created_utc=man.get("created_utc"),
               rev=rev, dirty_bake=bool(repo_rec.get("dirty")))

    baked = man.get("shipped_files")
    if isinstance(baked, dict) and baked:
        res["mode"] = "content"
        return _check_env_content(repo_root, res, man, baked)

    res["mode"] = "git-legacy"
    if not rev:
        res["error"] = f"env manifest has no repos.{ENV_MANIFEST_REPO}.rev"
        return res
    if _git(repo_root, "cat-file", "-e", f"{rev}^{{commit}}")[0] != 0:
        res["verdict"] = "unknown-rev"
        return res
    includes, excludes = parse_manifest(repo_root)
    specs = list(includes) + [f":(exclude){e}" for e in excludes]
    rc, out = _git(repo_root, "diff", "--name-only", rev, "HEAD", "--", *specs)
    if rc != 0:
        res["error"] = "git diff against the baked rev failed"
        return res
    res["changed"] = sorted(p for p in out.splitlines() if p.strip())
    res["uncommitted"] = _uncommitted_shipped(repo_root, specs)
    res["verdict"] = "stale" if res["changed"] or res["uncommitted"] else "fresh"
    return res


def _check_env_content(repo_root: str, res: dict, man: dict, baked: dict) -> dict:
    """Content-mode body of check_env_staleness (see its docstring)."""
    includes, excludes = parse_manifest(repo_root)
    specs = list(includes) + [f":(exclude){e}" for e in excludes]
    try:
        current = shipped_files(repo_root)
    except ShipcheckError as e:
        res["error"] = str(e)
        return res

    # A upstream-monorepo HEAD pin deliberately ships the HEAD blob instead of the
    # working-tree copy, so a difference there is the pin working as designed,
    # not drift. Report it, never fail on it.
    pinned = {hp.get("path") for hp in (man.get("head_pins") or [])
              if hp.get("bundle") == ENV_MANIFEST_REPO and hp.get("path")}

    cur: dict[str, str] = {}
    unreadable = []
    for rel in sorted(current):
        try:
            cur[rel] = sha256_file(os.path.join(repo_root, rel))
        except OSError:
            unreadable.append(rel)
    res["n_compared"] = len(cur)
    res["unreadable"] = unreadable

    # cheap equality fast-path: one digest instead of a 3-way set diff
    digest = man.get("shipped_files_digest")
    if digest and not unreadable and shipped_files_digest(cur) == digest:
        res["verdict"] = "fresh"
        res["uncommitted_identical"] = _uncommitted_shipped(repo_root, specs)
        return res

    for rel, sha in cur.items():
        if rel not in baked:
            res["added"].append(rel)
        elif baked[rel] != sha:
            (res["head_pinned"] if rel in pinned else res["differs"]).append(
                {"path": rel, "baked": baked[rel], "current": sha})
    res["added"] = sorted(res["added"])
    res["differs"].sort(key=lambda d: d["path"])
    res["head_pinned"].sort(key=lambda d: d["path"])
    res["removed"] = sorted(set(baked) - set(cur) - set(unreadable))

    # THE FIX: an uncommitted file whose bytes match the bake is reported as
    # explicitly not-stale, so the operator can see why the gate stayed green.
    differing = {d["path"] for d in res["differs"]} | set(res["added"])
    res["uncommitted_identical"] = [p for p in _uncommitted_shipped(repo_root, specs)
                                    if p not in differing]
    res["verdict"] = ("stale" if (res["differs"] or res["added"]
                                  or res["removed"] or unreadable) else "fresh")
    return res


def format_env_verdict(res: dict, box_hint: str = "<box>") -> list[str]:
    v = res["verdict"]
    ver = res.get("version") or "?"
    if v == "unresolved":
        return ["note: env staleness NOT CHECKED — no baked eval-env MANIFEST found "
                f"(looked in {DIST_REL}/env-*.MANIFEST.json; pass --env-manifest). "
                "This is not a freshness verdict."]
    if res.get("error"):
        return [f"note: env staleness NOT CHECKED — {res['error']}"]
    if v == "unknown-rev":
        return [f"note: env staleness NOT CHECKED — env-{ver} records upstream-monorepo "
                f"rev {(res['rev'] or '')[:12]}, which is not in this checkout "
                f"(git fetch, or re-bake)."]
    head = (f"env-{ver} (baked {res.get('created_utc')}) from upstream-monorepo "
            f"{(res['rev'] or '')[:12]}")
    if res.get("mode") == "content":
        return _format_env_content(res, head, box_hint)
    if v == "fresh":
        return [f">> env staleness OK: {head} — no ship-manifest file has changed "
                f"since (legacy git-rev compare: this env records no shipped_files, "
                f"so the comparison is approximate — re-bake for a content compare)"]
    out = [f"!! STALE ENV: {head}",
           "   [legacy git-rev compare — this env predates shipped_files, so an "
           "uncommitted file cannot be distinguished from one that already shipped]",
           f"   {len(res['changed'])} ship-manifest file(s) changed in git since that "
           f"bake — present HERE, absent from the box's baked tree:"]
    out += [f"     {p}" for p in res["changed"][:40]]
    if len(res["changed"]) > 40:
        out.append(f"     … +{len(res['changed']) - 40} more")
    if res["uncommitted"]:
        out.append(f"   plus {len(res['uncommitted'])} UNCOMMITTED ship-manifest "
                   f"file(s) (these ship on the next bake, never via git):")
        out += [f"     {p}" for p in res["uncommitted"][:20]]
    if res.get("dirty_bake"):
        out.append("   note: the bake ran against a DIRTY tree, so the recorded rev "
                   "is an approximation of what it shipped.")
    out.append(f"   SYNC REQUIRED (unless you already synced this checkout): "
               f"herdd sync {box_hint}   — or re-bake for a clean slate.")
    return out


def _fmt_ident(res: dict) -> list[str]:
    """The not-stale disclosure: uncommitted files whose bytes already shipped."""
    ident = res.get("uncommitted_identical") or []
    if not ident:
        return []
    out = [f"   ({len(ident)} uncommitted ship-manifest file(s) are byte-identical "
           f"to what the bake shipped — NOT staleness:"]
    out += [f"      {p}" for p in ident[:20]]
    if len(ident) > 20:
        out.append(f"      … +{len(ident) - 20} more")
    out[-1] += ")"
    return out


def _format_env_content(res: dict, head: str, box_hint: str) -> list[str]:
    def rows(items, n=40):
        out = [f"     {d['path']}   baked {d['baked'][:12]} -> here "
               f"{d['current'][:12]}" for d in items[:n]]
        if len(items) > n:
            out.append(f"     … +{len(items) - n} more")
        return out

    if res["verdict"] == "fresh":
        out = [f">> env staleness OK: {head} — all {res.get('n_compared', 0)} "
               f"ship-manifest file(s) are byte-identical to what the bake shipped "
               f"(sha256 content compare)"]
        return out + _fmt_ident(res)

    out = [f"!! STALE ENV: {head}",
           "   [sha256 content compare against the bake's shipped_files — "
           "differences below are REAL drift, not a dirty-tree artefact]"]
    if res["differs"]:
        out.append(f"   {len(res['differs'])} ship-manifest file(s) DIFFER from the "
                   f"bytes the bake shipped (the box holds the old content):")
        out += rows(res["differs"])
    if res["added"]:
        out.append(f"   {len(res['added'])} ship-manifest file(s) are NEW here and "
                   f"were never shipped (the box does not have them at all):")
        out += [f"     {p}" for p in res["added"][:40]]
        if len(res["added"]) > 40:
            out.append(f"     … +{len(res['added']) - 40} more")
    if res["removed"]:
        out.append(f"   {len(res['removed'])} file(s) shipped in that bake but are "
                   f"gone from this checkout (deleted/renamed/de-listed — the box "
                   f"still carries them):")
        out += [f"     {p}" for p in res["removed"][:40]]
        if len(res["removed"]) > 40:
            out.append(f"     … +{len(res['removed']) - 40} more")
    if res.get("unreadable"):
        out.append(f"   {len(res['unreadable'])} ship-manifest file(s) are tracked "
                   f"but unreadable here, so nothing could be compared:")
        out += [f"     {p}" for p in res["unreadable"][:20]]
    out += _fmt_ident(res)
    if res["head_pinned"]:
        out.append(f"   ({len(res['head_pinned'])} HEAD-pinned file(s) differ by "
                   f"design — the bake shipped the HEAD blob, not the working copy:)")
        out += rows(res["head_pinned"], 20)
    out.append(f"   SYNC REQUIRED (unless you already synced this checkout): "
               f"herdd sync {box_hint}   — or re-bake for a clean slate.")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _repo_root(arg: str | None) -> str:
    if arg:
        return os.path.abspath(arg)
    rc, out = _git(os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                   "--show-toplevel")
    if rc != 0 or not out.strip():
        raise ShipcheckError("not inside a git checkout (pass --repo)")
    return out.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("check", choices=("imports", "jobd", "env", "all"))
    ap.add_argument("--repo", help="upstream-monorepo checkout (default: this one)")
    ap.add_argument("--env-manifest", help="baked eval-env MANIFEST json")
    ap.add_argument("--env-version", help="env version, resolved in out/eval-env/dist")
    ap.add_argument("--box", default="<box>", help="box id, for the fix hint")
    ap.add_argument("--warn-only", action="store_true",
                    help="report findings but always exit 0")
    a = ap.parse_args(argv)
    try:
        root = _repo_root(a.repo)
        rc = 0
        if a.check in ("imports", "all"):
            gaps = import_closure_gaps(root)
            for ln in format_import_gaps(gaps, MANIFEST_REL, "ship manifest"):
                print(ln, file=sys.stderr if gaps else sys.stdout)
            rc |= 1 if gaps else 0
        if a.check in ("imports", "jobd", "all"):
            # the second manifest: one detector, both lists (2026-08-14)
            try:
                jgaps = jobd_import_closure_gaps(root)
            except ShipcheckError as e:
                # A checkout with no jobd-bundle owner (neither
                # vastlib/jobs/bundle.py nor herdd.py) has no jobd bundle to
                # check — a NOTE, not a finding and not a verdict (same doctrine
                # as an unresolvable env manifest). Asking for `jobd`
                # explicitly, or a bundle that exists and could not be READ,
                # still errors. The owner probe is `_jobd_bundle_source` itself
                # so this branch can never disagree with the loader about what
                # "the checkout has a bundle" means.
                try:
                    _jobd_bundle_source(root)
                    has_owner = True
                except ShipcheckError:
                    has_owner = False
                if a.check == "jobd" or has_owner:
                    raise
                print(f"note: jobd bundle NOT CHECKED — {e}")
                jgaps = []
            else:
                for ln in format_import_gaps(jgaps, JOBD_MANIFEST_HINT,
                                             "jobd bundle"):
                    print(ln, file=sys.stderr if jgaps else sys.stdout)
            rc |= 1 if jgaps else 0
        if a.check in ("env", "all"):
            res = check_env_staleness(
                root, resolve_env_manifest(root, a.env_manifest, a.env_version))
            stale = res["verdict"] == "stale"
            for ln in format_env_verdict(res, a.box):
                print(ln, file=sys.stderr if stale else sys.stdout)
            rc |= 1 if stale else 0
    except ShipcheckError as e:
        print(f"!! shipcheck: {e}", file=sys.stderr)
        return 2
    return 0 if a.warn_only else rc


if __name__ == "__main__":
    sys.exit(main())
