"""The vastlib static lane — ruff, mypy and import-linter, wrapped in pytest.

Why this file exists
--------------------
`tools/vast/` has no CI. The `vastlib` package (plan §5) is the one part of the
tree that is meant to be strictly typed and to hold a real dependency DAG, and
neither property survives without something that checks it on every run. Pytest
is the only gate this directory actually has, so the static lane rides in it —
plan §6 names this file explicitly as the enforcement point.

Three tools, three checks, plus one meta-check:

  ruff check      pycodestyle + pyflakes + isort + flake8-annotations, config
                  at vastlib/ruff.toml (auto-discovered; ruff resolves config
                  per file by walking up from the file).
  mypy            strict on `vastlib.*`, config at vastlib/mypy.ini, invoked
                  as `-p vastlib` from cwd=tools/vast.
  lint-imports    the layered contract at vastlib/importlinter.ini —
                  core -> {market,boxes,launch,storage} ->
                  {supervise,jobs,fleet,workflows} -> cli.
  + a non-vacuousness probe for mypy (see below).

The mypy probe is not decoration
--------------------------------
Measured 2026-08-16 while writing this lane: `mypy tools/vast/vastlib` from the
repo root names the modules `tools.vast.vastlib.*`, the `[mypy-vastlib,vastlib.*]`
strict section matches nothing, and an untyped `def probe(x): return x` passes
clean. A green checker that is not looking is worse than no checker, so
`test_mypy_strictness_is_not_vacuous` re-proves the strict block bites by
type-checking a deliberately untyped module in a tmp copy of the package.

Degrading honestly
------------------
None of the three tools is a declared dependency of this repo (they are dev
tooling, not runtime). When one is absent the test SKIPS with the exact
`uv pip install` command in the message — a silent pass would mean the package
loses its only guarantee the moment someone runs the suite in a lean venv.

What is deliberately NOT here
-----------------------------
* No repo-wide linting. Every config lives inside `vastlib/` and is passed
  explicitly (or discovered hierarchically, for ruff); running ruff or mypy on
  any other path in this repo behaves exactly as it did before this lane
  existed, which is to say with no configuration at all.
* No import of `vastlib` into the pytest process. The tools are shelled out to
  so that a broken package under construction fails as a test failure with the
  tool's own diagnostics, not as a collection error that takes the rest of
  `tools/vast/` down with it.
* No writes outside `tmp_path`: caches are redirected and
  `PYTHONDONTWRITEBYTECODE` keeps `__pycache__` out of the source tree.

Provenance: created 2026-08-16, step 1 of docs/plans/vast-tooling-refactor-v2.md §8.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Hermetic to cwd: everything is resolved from THIS file's location, never from
# the directory pytest happens to have been invoked in.
VAST_DIR = Path(__file__).resolve().parent
VASTLIB = VAST_DIR / "vastlib"

def _install_hint() -> str:
    """The exact command to run, naming the interpreter actually in use.

    Built at runtime rather than hardcoded to `.venv/bin/python`: a skip whose
    remedy points at the wrong venv is a skip nobody acts on, and the suite is
    run from several (repo `.venv`, fleetd's release venv, a bare system
    python). The repo's venv is uv-managed and has no `pip` binary, hence
    `uv pip install --python <interpreter>`.
    """
    return f"uv pip install --python {sys.executable} ruff mypy import-linter"


def _tool_dirs() -> list[Path]:
    """Bin directories to search, most-specific first.

    `Path(sys.executable).parent` and NOT `.resolve().parent`: a venv's
    `bin/python` is a symlink to the base interpreter, so resolving it walks
    straight OUT of the venv. Measured here 2026-08-16 — resolving
    `.venv/bin/python` lands in `~/.local/share/uv/python/cpython-3.13.../bin`,
    which holds no ruff, and the whole lane skipped silently while all three
    tools sat installed two directories away. The resolved dir is kept as a
    fallback for the case where pytest is run through a non-venv wrapper.
    """
    exe = Path(sys.executable)
    dirs = [exe.parent, exe.resolve().parent]
    return list(dict.fromkeys(dirs))


def _tool(name: str) -> str:
    """Locate a dev tool, preferring the venv that is running pytest.

    Resolution order is deliberate: the bin directory of `sys.executable`
    first, so the lane checks the package with the same environment the suite
    was launched in, then `PATH`. Skips (loudly) rather than failing when the
    tool is absent — see the module docstring.
    """
    searched = _tool_dirs()
    for d in searched:
        found = shutil.which(name, path=str(d))
        if found:
            return found
    on_path = shutil.which(name)
    if on_path:
        return on_path
    pytest.skip(
        f"{name!r} is not installed in {[str(d) for d in searched]} or on PATH, "
        f"so the vastlib static lane did NOT run. This is a coverage hole, not "
        f"a pass. Install it with:  {_install_hint()}"
    )


def _run(argv: list[str], *, cwd: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Shell out with a hermetic environment: no bytecode, no stray caches."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Belt and braces — every invocation below also passes an explicit
    # cache flag, but a tool that grows a new cache should still land in tmp.
    env["HOME"] = str(tmp_path)
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["RUFF_CACHE_DIR"] = str(tmp_path / "ruff-cache")
    env["MYPY_CACHE_DIR"] = str(tmp_path / "mypy-cache")
    return subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _fail(what: str, proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"{what} failed (exit {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_vastlib_package_layout() -> None:
    """The skeleton exists and every subpackage is a real package.

    Cheap, but it is the precondition for the other three: import-linter's
    layers contract names each subpackage, and a missing `__init__.py` turns a
    contract violation into a confusing "module not found" instead.
    """
    assert (VASTLIB / "__init__.py").is_file(), f"no vastlib package at {VASTLIB}"
    expected = [
        "core",
        "market",
        "boxes",
        "launch",
        "storage",
        "supervise",
        "jobs",
        "fleet",
        "workflows",
        "cli",
    ]
    missing = [p for p in expected if not (VASTLIB / p / "__init__.py").is_file()]
    assert not missing, f"subpackages missing an __init__.py: {missing}"
    for cfg in ("ruff.toml", "mypy.ini", "importlinter.ini", "requirements.txt"):
        assert (VASTLIB / cfg).is_file(), f"vastlib/{cfg} is missing"


def test_ruff_clean(tmp_path: Path) -> None:
    """`ruff check` on vastlib, using vastlib/ruff.toml.

    No `--config`: ruff's hierarchical resolution finds `vastlib/ruff.toml` by
    walking up from each checked file, which is exactly the behavior an editor
    or LSP gets for free. Testing it the same way the tooling sees it means a
    config that fails to be discovered fails HERE rather than passing in CI and
    silently doing nothing in a developer's editor.
    """
    ruff = _tool("ruff")
    proc = _run(
        [ruff, "check", "--no-cache", "--output-format", "full", str(VASTLIB)],
        cwd=VAST_DIR,
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, _fail("ruff check tools/vast/vastlib", proc)


def test_mypy_strict_clean(tmp_path: Path) -> None:
    """`mypy -p vastlib` from cwd=tools/vast, strict per vastlib/mypy.ini.

    The cwd and the `-p` are both load-bearing: they are what makes mypy name
    the modules `vastlib.*` so the strict per-module section applies. See
    test_mypy_strictness_is_not_vacuous, which proves that it does.
    """
    mypy = _tool("mypy")
    proc = _run(
        [
            mypy,
            "--config-file",
            str(VASTLIB / "mypy.ini"),
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "-p",
            "vastlib",
        ],
        cwd=VAST_DIR,
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, _fail("mypy -p vastlib", proc)


def test_mypy_strictness_is_not_vacuous(tmp_path: Path) -> None:
    """Prove the strict block BITES, by giving mypy something it must reject.

    The package is copied into tmp (nothing is written to the source tree) and
    a deliberately untyped def is added. If mypy reports success, the strict
    per-module section is not matching — the exact failure measured on
    2026-08-16, when `mypy tools/vast/vastlib` from the repo root named the
    modules `tools.vast.vastlib.*` and checked 12 files without complaint.

    This is the plan's §7.3 vacuousness discipline applied to the checker
    itself: a guard that cannot fire is not a guard.
    """
    mypy = _tool("mypy")
    stage = tmp_path / "stage"
    stage.mkdir()
    shutil.copytree(VASTLIB, stage / "vastlib")
    probe = stage / "vastlib" / "core" / "_vacuousness_probe.py"
    probe.write_text(
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "def probe(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    proc = _run(
        [
            mypy,
            "--config-file",
            str(stage / "vastlib" / "mypy.ini"),
            "--cache-dir",
            str(tmp_path / "probe-cache"),
            "-p",
            "vastlib",
        ],
        cwd=stage,
        tmp_path=tmp_path,
    )
    assert proc.returncode != 0 and "no-untyped-def" in proc.stdout, (
        "mypy accepted an untyped def inside vastlib — the strict per-module "
        "section is NOT applying, so every green mypy run in this lane is "
        "meaningless. Check the module naming (it must be `vastlib.*`, not "
        "`tools.vast.vastlib.*`).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_import_linter_contract(tmp_path: Path) -> None:
    """The layered dependency contract, per vastlib/importlinter.ini.

    lint-imports builds a real import graph, so `vastlib` has to be importable:
    cwd is `tools/vast` and PYTHONPATH carries it too, which is the same
    sys.path shape the Zone E entry scripts set up. No package under
    construction is imported into the pytest process itself.
    """
    lint_imports = _tool("lint-imports")
    env_path = str(VAST_DIR)
    prev = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = env_path if not prev else f"{env_path}{os.pathsep}{prev}"
    try:
        proc = _run(
            [
                lint_imports,
                "--config",
                str(VASTLIB / "importlinter.ini"),
                "--no-cache",
            ],
            cwd=VAST_DIR,
            tmp_path=tmp_path,
        )
    finally:
        if prev is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prev
    assert proc.returncode == 0, _fail("lint-imports (vastlib layers)", proc)
    assert "Contracts: 1 kept, 0 broken." in proc.stdout, _fail(
        "lint-imports did not report the expected contract count", proc
    )


#: The Zone E entry scripts. `vastlib` may never import either: they sit ABOVE
#: the package (plan §3) and both now consist of nothing but re-exports OF it,
#: so an import here would be a cycle dressed as a convenience.
ENTRY_SCRIPT_MODULES = ("herdd", "fleetd")


def test_vastlib_never_imports_an_entry_script() -> None:
    """No module under `vastlib/` imports `herdd` or `fleetd`.

    Not a style rule. The fat `fleetd.py` was reachable from
    `vastlib/fleet/client.py` by a LAZY `import fleetd` inside a function — the
    shape that survives a module-level grep, passes every test while the flat
    file still has a body, and at the step-6 thinning either imports a launcher
    that imports the package back, or resolves a name that is now a re-export of
    the very module doing the importing. Both copies of that lazy import died
    with the fat bodies; this is what keeps them dead.

    AST, not grep: a comment mentioning `import fleetd` (there is one, in
    `fleet/client.py`, explaining the history) must not fail the test, and an
    import nested inside a function must not pass it.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(VASTLIB.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]] if node.level == 0 else []
            else:
                continue
            for name in names:
                if name in ENTRY_SCRIPT_MODULES:
                    rel = path.relative_to(VASTLIB.parent)
                    offenders.append(f"{rel}:{node.lineno}: imports {name}")
    assert not offenders, (
        "vastlib imports a Zone E entry script (plan §3 — the package is BELOW "
        "the launchers, never above them):\n  " + "\n  ".join(offenders))
