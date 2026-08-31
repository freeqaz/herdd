"""Portable tests for shipcheck.py — the derived guards over ship_manifest.txt.

Runs in the toolchain-free lane (`pytest -m "not integration"`): no B2, no vast
API, no network. Two kinds of test:

  * REAL-REPO guards (the ones that bite): THIS checkout's ship manifest must be
    import-closed. That is the assertion that would have failed the 2026-07-30
    frontier wave round 2 on the workstation instead of on a rented box —
    witness_frontier.py top-imported inplace_build, score_frontier_resumable.py
    top-imported c2rs_prefilter, and neither was in the manifest (fix 36602adb).
    The manifest's existing "pathspec matches no tracked file" guard is blind to
    this: every listed pathspec was valid: the problem was one never listed.

  * SYNTHETIC repos (tmp_path + `git init`) pinning the semantics: guarded /
    lazy / function-local imports do NOT count, stdlib does not count, sibling
    search dirs resolve (tools/pipeline -> mining/), and the env-staleness
    verdict lattice (unresolved / unknown-rev / fresh / stale).
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shipcheck as sc  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", ".."))

pytestmark = pytest.mark.skipif(not shutil.which("git"), reason="needs git")


# =============================================================================
# real repo: the manifest must be import-closed
# =============================================================================
def test_real_manifest_is_import_closed():
    gaps = sc.import_closure_gaps(REPO_ROOT)
    assert gaps == [], "\n".join(sc.format_import_gaps(gaps))


def test_e2_vendored_module_list_is_import_closed():
    """A hand-maintained VENDOR list is the same defect surface as the ship
    manifest — the importer travels, the imported does not — so the same closure
    applies to it. e2-paired-score/sync_scorer_files.sh had exactly this gap on
    2026-07-30 (its rehearsal died `No module named 'inplace_build'`), found by
    running the guard against its MODULES array."""
    sh = os.path.join(REPO_ROOT, "tools", "witness", "jobs", "e2-paired-score",
                      "sync_scorer_files.sh")
    if not os.path.isfile(sh):
        pytest.skip("e2-paired-score bundle is gone")
    mods = _bash_array(sh, "MODULES")
    assert mods, "could not parse MODULES=( … ) — keep this parser in step"
    shipped = {f"tools/witness/{m}" for m in mods}
    gaps = sc.import_closure_gaps(REPO_ROOT, shipped)
    assert gaps == [], "\n".join(sc.format_import_gaps(gaps))


@pytest.mark.parametrize("bundle", ["e2-paired-score", "m2-p2a"])
def test_vendored_modules_are_gitignored(bundle):
    """Every module a bundle vendors must be gitignored IN that bundle dir.

    Each sync script's header says "never commit a vendored copy here", but the
    thing that enforces it is a .gitignore — a SECOND hand-maintained list over
    the same set, so it drifts exactly the way the ship manifest does. On
    2026-08-05 the canonical-drift-taxonomy work added drift_class.py and
    dtk_split_generation.py to both MODULES arrays and to neither .gitignore;
    e2-paired-score had five older entries missing the same way. Seven untracked
    vendored copies sat in the tree, one `git add -A` from being committed
    against the rule in the file's own header.

    `git check-ignore` is the oracle, not a text match on .gitignore — the
    question is whether git ignores the path, however that is arranged.
    """
    d = os.path.join(REPO_ROOT, "tools", "witness", "jobs", bundle)
    sh = os.path.join(d, "sync_scorer_files.sh")
    if not os.path.isfile(sh):
        pytest.skip(f"{bundle} bundle is gone")
    mods = _bash_array(sh, "MODULES") + _bash_array(sh, "NEW_TOOLS")
    assert mods, "could not parse MODULES=( … ) — keep this parser in step"
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=d, input="\n".join(mods),
                       capture_output=True, text=True)
    ignored = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    missing = sorted(set(mods) - ignored)
    assert not missing, (
        f"{bundle}/sync_scorer_files.sh vendors these, but git does not ignore them "
        f"in {bundle}/ — add them to its .gitignore: {missing}")


@pytest.mark.parametrize("name", ["env_identity.py", "env_obj_digests.json"])
def test_q6_vendored_env_identity_pair_matches_the_repo_originals(name):
    """q6-round1-evals ships a COMMITTED copy of the eval-env identity pair, and
    it must stay byte-identical to `tools/vast/eval-env/`'s.

    Why committed rather than gitignored-and-synced like e2-paired-score's
    MODULES: the pair is what makes the round-1 instrument nameable, and the
    sync-script convention leans on an operator remembering a pre-submit step.
    Measured 2026-08-08: the stage-B artifact's OWN shipped
    `env_obj_digests.json` predates the commit that recorded stage B, so
    `run.sh` S0.b2 against that env returns UNKNOWN rc 3 (`die 6`) with the
    env's copy and PRISTINE rc 0 with the repo's — i.e. the bundle-local pair
    is load-bearing, not belt-and-braces. `run.sh` probes `$_here` FIRST and
    `env_identity.py` binds DIGEST_DB to its own directory, so the committed
    pair wins outright over the env's.

    Drift here fails CLOSED (an unrecorded bake reads UNKNOWN, never
    "probably fine"), so this guard exists to make the drift loud rather than
    to prevent a wrong answer.
    """
    bundle = os.path.join(REPO_ROOT, "tools", "witness", "jobs",
                          "q6-round1-evals", name)
    origin = os.path.join(REPO_ROOT, "tools", "vast", "eval-env", name)
    if not os.path.isfile(bundle):
        pytest.skip("q6-round1-evals bundle is gone")
    with open(origin, "rb") as f:
        want = f.read()
    with open(bundle, "rb") as f:
        got = f.read()
    assert got == want, (
        f"tools/witness/jobs/q6-round1-evals/{name} has drifted from "
        f"tools/vast/eval-env/{name} — re-copy it. A stale digest DB in the "
        f"bundle makes S0.b2 refuse (UNKNOWN rc 3) on any bake recorded after "
        f"the copy was taken.")


def _bash_array(path, name):
    """The words of a `NAME=( … )` bash array literal, comments stripped (bash
    honours `#` inside a multi-line array assignment — verified)."""
    import re
    body = re.search(rf"^{name}=\((.*?)^\)", open(path, encoding="utf-8").read(),
                     re.S | re.M)
    if not body:
        return []
    out = []
    for line in body.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        out += line.split()
    return out


def test_real_repo_regression_dropping_two_modules_is_caught():
    """The exact 2026-07-30 round-2 shape, reconstructed: drop the two modules
    36602adb added and the check must name both."""
    ship = sc.shipped_files(REPO_ROOT)
    dropped = {"tools/witness/inplace_build.py",
               "tools/witness/c2rs_prefilter.py"}
    if not dropped <= ship:
        pytest.skip("those modules no longer ship — the regression is moot")
    gaps = sc.import_closure_gaps(REPO_ROOT, ship - dropped)
    assert {g["missing"] for g in gaps} == dropped
    assert "witness_frontier.py" in " ".join(g["importer"] for g in gaps)


# =============================================================================
# real repo: the SECOND manifest — the jobd bundle (_job_attach_files())
#
# ship_manifest.txt is declarative and was guarded here from the start; the jobd
# bundle is a hardcoded Python list in herdd.py and was guarded only by a
# name-pinning test. On 2026-08-14 jobmeta.py grew a module-scope
# `from bidpolicy import DEFEND_*` (84d09ab1) and bidpolicy.py was never added
# to the bundle: every `python3 jobd.py ...` on every box launched after that
# commit died `ModuleNotFoundError`, silently (jobd.sh runs them `|| true`), so
# the boxes looked alive and idle while billing. The repo layout hides it —
# jobd.py adds its PARENT dir, which is tools/vast/ — so only a check that
# models the FLAT delivery can see it. Same detector, second manifest.
# =============================================================================
def test_jobd_bundle_is_import_closed():
    gaps = sc.jobd_import_closure_gaps(REPO_ROOT)
    assert gaps == [], "\n".join(
        sc.format_import_gaps(gaps, sc.JOBD_MANIFEST_HINT, "jobd bundle"))


def test_jobd_bundle_is_derived_from_the_owning_module_not_a_second_list():
    """The bundle set must come from `_job_attach_files()` itself — a hand-kept
    copy in shipcheck would rot into the same blind spot it guards.

    The owner is `vastlib/jobs/bundle.py` since the vast-tooling refactor (it
    was `herdd.py`, which now only re-exports the name). Renamed from
    `…_derived_from_herdd_…` with that move.
    """
    files = sc.jobd_bundle_files(REPO_ROOT)
    assert "tools/vast/onstart/jobd.py" in files      # sanity: still the daemon
    assert "tools/vast/jobmeta.py" in files
    assert all(f.startswith("tools/vast/") for f in files), sorted(files)
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "vast"))
    from vastlib.jobs import bundle                   # noqa: PLC0415
    assert files == {os.path.relpath(f, REPO_ROOT).replace(os.sep, "/")
                     for f in bundle._job_attach_files()}


def test_jobd_bundle_regression_dropping_bidpolicy_is_caught():
    """The 2026-08-14 shape, reconstructed: drop bidpolicy.py from the bundle
    and the check must name it, attributed to jobmeta.py. This is the assertion
    that would have cost $0 instead of a fleet of idle boxes."""
    ship = sc.jobd_bundle_files(REPO_ROOT)
    dropped = "tools/vast/bidpolicy.py"
    if dropped not in ship:
        pytest.fail("bidpolicy.py is not in the jobd bundle — the fix regressed; "
                    "jobmeta.py top-imports it and the flat bundle has no parent "
                    "dir to fall back on")
    gaps = sc.jobd_import_closure_gaps(REPO_ROOT, ship - {dropped})
    assert {g["missing"] for g in gaps} == {dropped}
    assert {g["importer"] for g in gaps} == {"tools/vast/jobmeta.py"}


def test_jobd_bundle_closure_crosses_the_onstart_boundary(tmp_path):
    """The bundle is FLATTENED, so an onstart/ file and a tools/vast/ file are
    siblings on the box: the namespace must span both dirs in BOTH directions,
    or a gap in one half reads as closed.

    Synthetic on purpose. An earlier version asserted this against the real
    jobd.py top-importing jobmeta — which coupled it to another file's import
    style and went red the moment that import was (correctly) wrapped in a
    try: for the capability gate. The namespace is the invariant; who exercises
    it this week is not.
    """
    assert set(sc.JOBD_FLAT_NAMESPACES) == {"tools/vast", "tools/vast/onstart"}
    for search in sc.JOBD_FLAT_NAMESPACES.values():
        assert set(search) == {"tools/vast", "tools/vast/onstart"}

    root = tmp_path / "repo"
    for rel, body in {
        "tools/vast/onstart/daemon.py": "import sibling_up\n",   # onstart -> vast
        "tools/vast/helper.py": "import sibling_down\n",         # vast -> onstart
        "tools/vast/sibling_up.py": "x = 1\n",
        "tools/vast/onstart/sibling_down.py": "y = 1\n",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    ship = {"tools/vast/onstart/daemon.py", "tools/vast/helper.py"}
    gaps = sc.jobd_import_closure_gaps(str(root), ship)
    assert {(g["importer"], g["missing"]) for g in gaps} == {
        ("tools/vast/onstart/daemon.py", "tools/vast/sibling_up.py"),
        ("tools/vast/helper.py", "tools/vast/onstart/sibling_down.py")}
    # …and shipping them closes it from either side
    assert sc.jobd_import_closure_gaps(
        str(root), ship | {"tools/vast/sibling_up.py",
                           "tools/vast/onstart/sibling_down.py"}) == []


def test_jobd_bundle_unreadable_is_an_error_not_an_empty_pass():
    """A bundle we could not read is not a bundle with no gaps."""
    with pytest.raises(sc.ShipcheckError):
        sc.jobd_bundle_files(str(REPO_ROOT) + "/does/not/exist")


def test_repo_flag_reads_THAT_checkouts_bundle_not_this_processs(tmp_path):
    """`--repo DIR` must read DIR's list, in a process that has already imported
    `vastlib` — which is every `herdd shipcheck …`, because
    `vastlib.cli.shipcheck` imports this module from inside a live vastlib.

    This is not hypothetical. While `jobd_bundle_files` path-loaded `herdd.py`
    it read the right NAME and the wrong CHECKOUT: the thin launcher only
    re-exports `_job_attach_files` from `vastlib.jobs.bundle`, and an
    already-imported package beats the `sys.path` insert, so a warm process got
    THIS checkout's list relpath'd against the foreign root into `../../…` keys
    — a garbage bundle that reads as "no gaps". Loading the OWNING module makes
    the body the target's in both process states. Measured 2026-08-16.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "vast"))
    import vastlib.jobs.bundle  # noqa: F401,PLC0415 — warm the package on purpose

    owner = tmp_path / "tools" / "vast" / "vastlib" / "jobs" / "bundle.py"
    owner.parent.mkdir(parents=True)
    owner.write_text('def _job_attach_files():\n'
                     '    return ["/sentinel/tools/vast/only_in_that_checkout.py"]\n')
    (tmp_path / "tools" / "vast" / "herdd.py").write_text(
        "raise AssertionError('the launcher must not be the source of truth')\n")

    assert sc._jobd_bundle_source(str(tmp_path))[1] == str(owner)
    files = sc.jobd_bundle_files(str(tmp_path))
    # one entry, and it is THAT checkout's (the relpath prefix is just how far
    # tmp_path sits from /sentinel — the identity of the file is the assertion)
    assert len(files) == 1
    assert next(iter(files)).endswith("sentinel/tools/vast/only_in_that_checkout.py")


def test_a_pre_package_checkout_still_resolves_through_herdd(tmp_path):
    """`--repo` is routinely pointed at another tree (a box's, a peer worktree,
    `main` while the refactor lands) where the list still lives in the fat
    `herdd.py`. The fallback keeps those readable; drop it and shipcheck
    reports `jobd bundle NOT CHECKED` on every pre-package checkout."""
    vc = tmp_path / "tools" / "vast" / "herdd.py"
    vc.parent.mkdir(parents=True)
    vc.write_text('def _job_attach_files():\n'
                  '    return ["%s/tools/vast/legacy.py"]\n' % tmp_path)
    assert sc._jobd_bundle_source(str(tmp_path)) == ("_shipcheck_herdd", str(vc))
    assert sc.jobd_bundle_files(str(tmp_path)) == {"tools/vast/legacy.py"}


def test_a_checkout_with_no_bundle_owner_is_a_note_not_a_verdict(tmp_path, capsys):
    """No `vastlib/jobs/bundle.py` AND no `herdd.py` -> the jobd half is a
    NOTE under `imports`/`all` (a checkout without the tool has no bundle to
    check) but still an error when `jobd` is asked for explicitly."""
    (tmp_path / "tools" / "vast").mkdir(parents=True)
    with pytest.raises(sc.ShipcheckError):
        sc.jobd_bundle_files(str(tmp_path))
    assert sc.main(["jobd", "--repo", str(tmp_path)]) == 2
    assert "!! shipcheck" in capsys.readouterr().err


def test_cli_jobd_check_on_the_real_repo_exits_zero(capsys):
    assert sc.main(["jobd", "--repo", REPO_ROOT]) == 0
    assert "jobd bundle" in capsys.readouterr().out


# =============================================================================
# synthetic repos: import-closure semantics
# =============================================================================
def _mkrepo(tmp_path, files: dict, manifest: str):
    """A git checkout with `files` (repo-relative -> text) plus a ship manifest,
    everything committed (the enumerator is `git ls-files`, so untracked files
    can neither ship nor close a gap)."""
    root = tmp_path / "repo"
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    m = root / sc.MANIFEST_REL
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(manifest)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qm", "init"], cwd=root, check=True)
    return str(root)


def test_top_level_import_of_unshipped_sibling_is_a_gap(tmp_path):
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "import helper\n",
        "tools/witness/helper.py": "x = 1\n",
    }, "tools/witness/a.py\n")
    gaps = sc.import_closure_gaps(root)
    assert [(g["importer"], g["module"], g["missing"]) for g in gaps] == [
        ("tools/witness/a.py", "helper", "tools/witness/helper.py")]


def test_shipped_sibling_closes_the_gap(tmp_path):
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "from helper import thing\n",
        "tools/witness/helper.py": "thing = 1\n",
    }, "tools/witness/a.py\ntools/witness/helper.py\n")
    assert sc.import_closure_gaps(root) == []


def test_guarded_and_lazy_imports_are_not_gaps(tmp_path):
    """try/except, if-guarded and function-local imports are the shapes the
    manifest deliberately leaves out (gbt's sklearn, anthropic_client's
    anthropic). Flagging them would make the gate cry wolf."""
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py":
            "try:\n    import optional_dep\nexcept ImportError:\n"
            "    optional_dep = None\n"
            "if False:\n    import cond_dep\n"
            "def f():\n    import lazy_dep\n    return lazy_dep\n",
        "tools/witness/optional_dep.py": "",
        "tools/witness/cond_dep.py": "",
        "tools/witness/lazy_dep.py": "",
    }, "tools/witness/a.py\n")
    assert sc.import_closure_gaps(root) == []


def test_stdlib_and_absent_modules_are_not_gaps(tmp_path):
    """A gap needs a LOCAL file to point at: stdlib (json), a third-party wheel
    (numpy) and a plain typo all resolve to nothing on disk and stay silent —
    the bake's import smoke owns those."""
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "import json\nimport numpy\nimport nope_typo\n",
    }, "tools/witness/a.py\n")
    assert sc.import_closure_gaps(root) == []


def test_relative_and_dotted_imports_do_not_false_positive(tmp_path):
    """`from . import x` is a package import (not a flat bare name) and
    `import pkg.mod` keys on its FIRST component — a shipped `pkg/` closes it."""
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "from . import sibling\nimport pkg.mod\n",
        "tools/witness/sibling.py": "",
        "tools/witness/pkg/__init__.py": "",
        "tools/witness/pkg/mod.py": "",
    }, "tools/witness/a.py\ntools/witness/pkg\n")
    assert sc.import_closure_gaps(root) == []


def test_sibling_search_dir_resolves_mining(tmp_path):
    """tools/pipeline reaches mining/ via _pipeline_path.setup(), so a bare
    `import mine_attempts` legitimately resolves one dir over — shipped there,
    it is closed; unshipped, it is a gap naming the mining/ path."""
    files = {
        "tools/pipeline/a.py": "import mine_attempts\n",
        "tools/pipeline/mining/mine_attempts.py": "",
    }
    closed = _mkrepo(tmp_path / "c", files,
                     "tools/pipeline/a.py\ntools/pipeline/mining/mine_attempts.py\n")
    assert sc.import_closure_gaps(closed) == []
    open_ = _mkrepo(tmp_path / "o", files, "tools/pipeline/a.py\n")
    assert [g["missing"] for g in sc.import_closure_gaps(open_)] == \
        ["tools/pipeline/mining/mine_attempts.py"]


def test_excluded_subtree_does_not_ship_and_cannot_close(tmp_path):
    """'!' excludes are how tests/ stay off boxes; an import satisfied only by
    an EXCLUDED file is still a gap."""
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "import helper\n",
        "tools/witness/sub/helper.py": "",
        "tools/witness/helper.py": "",
    }, "tools/witness\n!tools/witness/helper.py\n")
    assert [g["missing"] for g in sc.import_closure_gaps(root)] == \
        ["tools/witness/helper.py"]


def test_untracked_sibling_does_not_close_a_gap(tmp_path):
    root = _mkrepo(tmp_path, {"tools/witness/a.py": "import helper\n"},
                   "tools/witness/a.py\ntools/witness/helper.py\n")
    (open(os.path.join(root, "tools/witness/helper.py"), "w")).close()
    assert [g["missing"] for g in sc.import_closure_gaps(root)] == \
        ["tools/witness/helper.py"]


def test_manifest_missing_or_empty_is_a_usage_error(tmp_path):
    root = _mkrepo(tmp_path, {"tools/witness/a.py": ""}, "# only comments\n")
    with pytest.raises(sc.ShipcheckError):
        sc.shipped_files(root)
    os.remove(os.path.join(root, sc.MANIFEST_REL))
    with pytest.raises(sc.ShipcheckError):
        sc.parse_manifest(root)


def test_format_import_gaps_lists_each_missing_path_once(tmp_path):
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "import helper\n",
        "tools/witness/b.py": "import helper\n",
        "tools/witness/helper.py": "",
    }, "tools/witness/a.py\ntools/witness/b.py\n")
    text = "\n".join(sc.format_import_gaps(sc.import_closure_gaps(root)))
    assert text.count("FIX") == 1
    fix = text.split("FIX:")[1]
    assert fix.count("tools/witness/helper.py") == 1


# =============================================================================
# env staleness
# =============================================================================
def _env_manifest(tmp_path, rev, *, version="20260101-0000-deadbeef", dirty=False,
                  shipped=None, head_pins=None, digest=True):
    """A baked-env MANIFEST. `shipped` (path -> sha256) is what a POST-2026-07-30
    bake records; omit it for the legacy rev-only shape."""
    man = {"version": version, "created_utc": "2026-01-01T00:00:00Z",
           "repos": {sc.ENV_MANIFEST_REPO: {"rev": rev, "dirty": dirty}}}
    if head_pins is not None:
        man["head_pins"] = head_pins
    if shipped is not None:
        man["shipped_files"] = shipped
        if digest:
            man["shipped_files_digest"] = sc.shipped_files_digest(shipped)
    p = tmp_path / f"env-{version}.MANIFEST.json"
    p.write_text(json.dumps(man))
    return str(p)


def _commit(root, rel, body, msg="edit"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(body)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qm", msg], cwd=root, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()


def test_env_fresh_when_nothing_shipped_changed_since_the_bake(tmp_path):
    root = _mkrepo(tmp_path, {"tools/witness/a.py": "x=1\n"},
                   "tools/witness/a.py\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    _commit(root, "docs/notes.md", "not shipped\n")     # not in the manifest
    res = sc.check_env_staleness(root, _env_manifest(tmp_path, rev))
    assert res["verdict"] == "fresh", res
    assert ">> env staleness OK" in "\n".join(sc.format_env_verdict(res))


def test_env_stale_lists_the_changed_shipped_files(tmp_path):
    """The 2026-07-30 round-1 shape: a module the box's baked env predates."""
    root = _mkrepo(tmp_path, {"tools/witness/a.py": "x=1\n"},
                   "tools/witness\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    _commit(root, "tools/witness/plscore.py", "# the pool scorer\n")
    res = sc.check_env_staleness(root, _env_manifest(tmp_path, rev))
    assert res["verdict"] == "stale"
    assert res["changed"] == ["tools/witness/plscore.py"]
    text = "\n".join(sc.format_env_verdict(res, box_hint="4242"))
    assert "STALE ENV" in text and "SYNC REQUIRED" in text and "4242" in text


def test_env_legacy_manifest_reports_uncommitted_shipped_files(tmp_path):
    """LEGACY path, unchanged by the content compare: an env with no
    `shipped_files` cannot tell an already-shipped uncommitted file from a
    changed one, so it still reports both — but it now SAYS it is approximate."""
    root = _mkrepo(tmp_path, {"tools/witness/a.py": "x=1\n"},
                   "tools/witness\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    with open(os.path.join(root, "tools/witness/a.py"), "a") as fh:
        fh.write("y=2\n")
    res = sc.check_env_staleness(root, _env_manifest(tmp_path, rev))
    assert res["mode"] == "git-legacy"
    assert res["verdict"] == "stale"
    assert res["changed"] == [] and res["uncommitted"] == ["tools/witness/a.py"]
    assert "legacy git-rev compare" in "\n".join(sc.format_env_verdict(res))


def test_env_unresolved_is_a_note_not_a_verdict(tmp_path):
    root = _mkrepo(tmp_path, {"tools/witness/a.py": ""}, "tools/witness\n")
    assert sc.resolve_env_manifest(root) is None
    res = sc.check_env_staleness(root, None)
    assert res["verdict"] == "unresolved"
    text = "\n".join(sc.format_env_verdict(res))
    assert "NOT CHECKED" in text and "not a freshness verdict" in text


def test_env_unknown_rev_is_a_note(tmp_path):
    root = _mkrepo(tmp_path, {"tools/witness/a.py": ""}, "tools/witness\n")
    res = sc.check_env_staleness(root, _env_manifest(tmp_path, "0" * 40))
    assert res["verdict"] == "unknown-rev"
    assert "NOT CHECKED" in "\n".join(sc.format_env_verdict(res))


def test_env_dirty_bake_is_flagged_as_approximate(tmp_path):
    root = _mkrepo(tmp_path, {"tools/witness/a.py": "x=1\n"},
                   "tools/witness\n")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    _commit(root, "tools/witness/b.py", "y=2\n")
    res = sc.check_env_staleness(root, _env_manifest(tmp_path, rev, dirty=True))
    assert res["dirty_bake"] is True
    assert "approximation" in "\n".join(sc.format_env_verdict(res))


def test_resolve_env_manifest_prefers_explicit_then_version_then_newest(tmp_path):
    root = tmp_path / "repo"
    dist = root / sc.DIST_REL
    dist.mkdir(parents=True)
    old = dist / "env-20260101-0000-aaaaaaaa.MANIFEST.json"
    new = dist / "env-20260202-1200-bbbbbbbb.MANIFEST.json"
    for p in (old, new):
        p.write_text("{}")
    assert sc.resolve_env_manifest(str(root)) == str(new)
    assert sc.resolve_env_manifest(str(root), version="20260101-0000-aaaaaaaa") == str(old)
    assert sc.resolve_env_manifest(str(root), version="nope") is None
    assert sc.resolve_env_manifest(str(root), path=str(old)) == str(old)


# =============================================================================
# env staleness, CONTENT mode (MANIFEST carries shipped_files)
#
# The defect these pin (measured 2026-07-30): the rev-based gate reported STALE
# ENV for 5 uncommitted ship-manifest files that were byte-identical to what the
# bake had tarred — they HAD shipped. A gate that fires on any peer session's
# open file is one operators learn to --allow-stale-env past, and then it misses
# the real staleness it exists for.
# =============================================================================
def _sha(root, rel):
    return sc.sha256_file(os.path.join(root, rel))


def _shipped_now(root):
    """What a bake run against THIS working tree would record."""
    return {rel: _sha(root, rel) for rel in sc.shipped_files(root)}


def _rev(root):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()


def _content_repo(tmp_path):
    return _mkrepo(tmp_path, {"tools/witness/a.py": "x=1\n",
                              "tools/witness/b.py": "y=1\n"},
                   "tools/witness\n")


def test_env_content_uncommitted_but_identical_file_is_NOT_stale(tmp_path):
    """THE FIX. The bake tarred the dirty working-tree bytes; the file is still
    uncommitted here and still byte-identical, so nothing is stale."""
    root = _content_repo(tmp_path)
    with open(os.path.join(root, "tools/witness/a.py"), "a") as fh:
        fh.write("# an uncommitted edit that the bake shipped verbatim\n")
    baked = _shipped_now(root)                       # bake runs on the dirty tree
    res = sc.check_env_staleness(
        root, _env_manifest(tmp_path, _rev(root), dirty=True, shipped=baked))
    assert res["mode"] == "content"
    assert res["verdict"] == "fresh", res
    assert res["differs"] == [] and res["added"] == [] and res["removed"] == []
    text = "\n".join(sc.format_env_verdict(res))
    assert "STALE ENV" not in text and "env staleness OK" in text
    # and it discloses WHY it stayed green
    assert res["uncommitted_identical"] == ["tools/witness/a.py"]
    assert "byte-identical" in text


def test_env_content_modified_file_is_stale_with_sha_pair(tmp_path):
    """Real drift still FAILS — committed or not, content is the judge."""
    root = _content_repo(tmp_path)
    baked = _shipped_now(root)
    with open(os.path.join(root, "tools/witness/a.py"), "w") as fh:
        fh.write("x=2  # genuinely different from what shipped\n")
    res = sc.check_env_staleness(
        root, _env_manifest(tmp_path, _rev(root), shipped=baked))
    assert res["verdict"] == "stale"
    assert [d["path"] for d in res["differs"]] == ["tools/witness/a.py"]
    assert res["differs"][0]["baked"] == baked["tools/witness/a.py"]
    assert res["differs"][0]["current"] == _sha(root, "tools/witness/a.py")
    text = "\n".join(sc.format_env_verdict(res, box_hint="4242"))
    assert "STALE ENV" in text and "DIFFER" in text and "SYNC REQUIRED" in text
    assert baked["tools/witness/a.py"][:12] in text     # short sha pair shown
    assert _sha(root, "tools/witness/a.py")[:12] in text
    assert "4242" in text


def test_env_content_committed_change_is_stale_too(tmp_path):
    """The 2026-07-30 round-1 shape (a module the baked env predates), now seen
    as content rather than as a rev delta."""
    root = _content_repo(tmp_path)
    baked = _shipped_now(root)
    _commit(root, "tools/witness/plscore.py", "# the pool scorer\n")
    res = sc.check_env_staleness(
        root, _env_manifest(tmp_path, _rev(root), shipped=baked))
    assert res["verdict"] == "stale"
    assert res["added"] == ["tools/witness/plscore.py"]
    assert "never shipped" in "\n".join(sc.format_env_verdict(res))


def test_env_content_file_missing_from_the_bake_is_stale(tmp_path):
    """Present here, absent from the bake — the box does not have it at all."""
    root = _content_repo(tmp_path)
    baked = _shipped_now(root)
    baked.pop("tools/witness/b.py")
    res = sc.check_env_staleness(
        root, _env_manifest(tmp_path, _rev(root), shipped=baked))
    assert res["verdict"] == "stale"
    assert res["added"] == ["tools/witness/b.py"] and res["differs"] == []
    assert "tools/witness/b.py" in "\n".join(sc.format_env_verdict(res))


def test_env_content_file_removed_here_is_stale(tmp_path):
    """Shipped then deleted/renamed here: the box carries a module this checkout
    no longer has. The rev heuristic caught this as a git deletion; content mode
    must not be weaker."""
    root = _content_repo(tmp_path)
    baked = _shipped_now(root)
    subprocess.run(["git", "rm", "-q", "tools/witness/b.py"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qm", "drop b"], cwd=root, check=True)
    res = sc.check_env_staleness(
        root, _env_manifest(tmp_path, _rev(root), shipped=baked))
    assert res["verdict"] == "stale"
    assert res["removed"] == ["tools/witness/b.py"]
    assert "gone from this checkout" in "\n".join(sc.format_env_verdict(res))


def test_env_content_head_pinned_difference_is_disclosed_not_stale(tmp_path):
    """BAKE_HEAD_PATHS ships the HEAD blob on purpose, so a working-tree
    difference there is the pin working — report it, never fail on it."""
    root = _content_repo(tmp_path)
    baked = _shipped_now(root)                        # == HEAD blobs, tree is clean
    with open(os.path.join(root, "tools/witness/a.py"), "w") as fh:
        fh.write("x=99  # mid-edit; the bake pinned the HEAD blob instead\n")
    res = sc.check_env_staleness(root, _env_manifest(
        tmp_path, _rev(root), shipped=baked,
        head_pins=[{"bundle": sc.ENV_MANIFEST_REPO, "path": "tools/witness/a.py"}]))
    assert res["verdict"] == "fresh"
    assert res["differs"] == []
    assert [d["path"] for d in res["head_pinned"]] == ["tools/witness/a.py"]


def test_env_content_digest_fast_path_agrees_with_the_per_file_compare(tmp_path):
    """The roll-up is only a fast path: with and without it, same verdict."""
    root = _content_repo(tmp_path)
    baked = _shipped_now(root)
    for digest in (True, False):
        res = sc.check_env_staleness(
            root, _env_manifest(tmp_path, _rev(root), shipped=baked, digest=digest,
                                version=f"20260101-000{int(digest)}-deadbeef"))
        assert res["verdict"] == "fresh" and res["n_compared"] == 2
    with open(os.path.join(root, "tools/witness/a.py"), "a") as fh:
        fh.write("drift\n")
    for digest in (True, False):
        res = sc.check_env_staleness(
            root, _env_manifest(tmp_path, _rev(root), shipped=baked, digest=digest,
                                version=f"20260102-000{int(digest)}-deadbeef"))
        assert res["verdict"] == "stale"
        assert [d["path"] for d in res["differs"]] == ["tools/witness/a.py"]


def test_env_content_mode_does_not_need_the_baked_rev_in_this_checkout(tmp_path):
    """`unknown-rev` was a git-only failure mode: content mode compares bytes,
    so an unfetched rev is no longer a reason the gate cannot answer."""
    root = _content_repo(tmp_path)
    res = sc.check_env_staleness(
        root, _env_manifest(tmp_path, "0" * 40, shipped=_shipped_now(root)))
    assert res["verdict"] == "fresh" and res["mode"] == "content"


def test_env_empty_shipped_files_map_falls_back_to_legacy(tmp_path):
    """`shipped_files: {}` is not 'nothing shipped' — it is an unrecorded bake.
    Treating it as content truth would call every env fresh."""
    root = _content_repo(tmp_path)
    p = _env_manifest(tmp_path, _rev(root), shipped={})
    assert sc.check_env_staleness(root, p)["mode"] == "git-legacy"


# =============================================================================
# CLI
# =============================================================================
def test_cli_imports_on_the_real_repo_exits_zero():
    assert sc.main(["imports", "--repo", REPO_ROOT]) == 0


def test_cli_warn_only_never_fails(tmp_path, capsys):
    root = _mkrepo(tmp_path, {
        "tools/witness/a.py": "import helper\n",
        "tools/witness/helper.py": "",
    }, "tools/witness/a.py\n")
    assert sc.main(["imports", "--repo", root]) == 1
    assert sc.main(["imports", "--repo", root, "--warn-only"]) == 0
    assert "IMPORT CLOSURE BROKEN" in capsys.readouterr().err


def test_cli_bad_repo_is_exit_2(tmp_path):
    assert sc.main(["imports", "--repo", str(tmp_path / "nope")]) == 2


# =============================================================================
# herdd wiring: `sync` refuses a non-import-closed manifest BEFORE any rsync
# =============================================================================
def _broken_repo(tmp_path):
    return _mkrepo(tmp_path, {
        "tools/witness/a.py": "import helper\n",
        "tools/witness/helper.py": "",
    }, "tools/witness/a.py\n")


def test_sync_gate_refuses_a_broken_closure(tmp_path, capsys):
    from vastlib.jobs import bundle
    with pytest.raises(SystemExit) as e:
        bundle._sync_import_gate(_broken_repo(tmp_path))
    assert "refusing to sync" in str(e.value)
    assert "IMPORT CLOSURE BROKEN" in capsys.readouterr().err


def test_sync_gate_warn_only_returns(tmp_path, capsys):
    from vastlib.jobs import bundle
    bundle._sync_import_gate(_broken_repo(tmp_path), warn_only=True)
    assert "--no-import-check" in capsys.readouterr().err


def test_sync_gate_passes_the_real_repo(tmp_path, capsys):
    from vastlib.jobs import bundle
    bundle._sync_import_gate(REPO_ROOT)
    assert "import closure OK" in capsys.readouterr().out


def test_sync_gate_never_blocks_on_its_own_failure(tmp_path, capsys):
    """A guard that can't run must degrade to a NOTE — never become the reason a
    sync is impossible."""
    from vastlib.jobs import bundle
    bundle._sync_import_gate(str(tmp_path / "not-a-repo"))
    assert "import-closure check skipped" in capsys.readouterr().err


def test_sync_with_explicit_paths_skips_the_manifest_gate(monkeypatch, tmp_path):
    """`sync <id> tools/pipeline` overrides the manifest wholesale, so a
    manifest-derived guard has no bearing on that call."""
    from vastlib.boxes import lifecycle
    from vastlib.cli import sync as cli_sync
    from vastlib.jobs import bundle
    calls = []
    monkeypatch.setattr(bundle, "_sync_import_gate",
                        lambda *a, **k: calls.append(a))
    monkeypatch.setattr(lifecycle, "_get_instance",
                        lambda i: (_ for _ in ()).throw(SystemExit("stop")))
    ns = argparse_ns(id=1, paths=["tools/pipeline"], dest="/d", dry_run=True,
                     no_import_check=False)
    with pytest.raises(SystemExit):
        cli_sync.run(ns)
    assert calls == []


def argparse_ns(**kw):
    import argparse
    return argparse.Namespace(**kw)
