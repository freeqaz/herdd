"""The jobd bundle must IMPORT in the shape it is delivered in — flat.

Why this file exists (2026-08-14). 84d09ab1 added an unguarded module-scope
`from bidpolicy import DEFEND_CHEAP, DEFEND_DEAR, DEFEND_MODES` to jobmeta.py
and did not add bidpolicy.py to `bundle._job_attach_files()`. Nothing caught
it, because every EXISTING check runs the bundle in the REPO layout:

  * jobd.py puts both its own dir AND its parent on sys.path, and in the repo
    the parent is tools/vast/ — where bidpolicy.py lives. So `pytest`,
    `rehearse.sh` and `herdd job run-local` all import it happily.
  * the bundle is delivered FLAT into /workspace/jobd/ (boot tar via
    _stage_jobd_bootstrap, or `job attach` scp), where there is no such parent.

So the break existed ONLY in the delivered layout, and on a box it was silent:
jobd.sh calls `python3 jobd.py ...` as `>/dev/null 2>&1 || true`, so B2 event
emission and ticket parsing no-oped while the bash/rclone half kept writing
JOBD_STATUS. Boxes looked alive and idle, forever, billing.

The test therefore reproduces the DELIVERY, not the repo: copy exactly the
files `_job_attach_files()` returns into a flat tmpdir and import them in a
subprocess with `-P` (no script-dir/cwd on sys.path) and an environment whose
PYTHONPATH is that tmpdir and nothing else. The file list is read from
`_job_attach_files()` itself — a hand-kept copy here would rot into the same
blind spot it is meant to guard.

Two layers, deliberately:

  * this file — RUNTIME. Catches what a static scan cannot: a `from x import
    NAME` where NAME is gone, an import that raises, a module-scope side effect
    that needs a file the bundle lacks.
  * shipcheck.py `jobd_import_closure_gaps` (tests in test_shipcheck.py) —
    STATIC, and the same detector already aimed at ship_manifest.txt. Catches
    the gap without executing anything, names the fix, and runs in the
    `herdd shipcheck` / pre-launch lane.

$0, offline, no B2, no vast API, no GPU.
"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402  (kept ONLY as the tools/vast/ path anchor — the
                 # launcher stays at tools/vast/herdd.py, plan §7.2 class C7)
from vastlib.jobs import bundle  # noqa: E402

# Bundle members that must import CLEANLY on a bare box python: no third-party
# packages, no GPU, no network. (The rest of the bundle is shell, or probes that
# legitimately top-import torch/transformers and fail open on the box.)
_MUST_IMPORT = ("jobmeta", "runmeta", "bidpolicy")


@pytest.fixture(scope="module")
def flat_bundle(tmp_path_factory):
    """The bundle as the box gets it: basenames only, one flat directory.

    The PARENT of this directory matters and must not be a shared scratch root.
    jobd.py puts its own dir AND its parent on sys.path (that asymmetry is the
    whole subject of this file), so a bare `TemporaryDirectory()` lands the
    bundle directly under $TMPDIR and hands the import machinery the shared tmp
    root as a search path. Two consequences, one slow and one silent: a stray
    `jobmeta.py` there would satisfy these imports from the wrong place, and
    `_fill_cache` listdir's the whole thing on every miss — measured 19 s and a
    120 s test timeout against a 180k-entry `~/tmp` on 2026-08-27.
    `tmp_path_factory` gives a private, nearly-empty parent.
    """
    files = bundle._job_attach_files()
    td = str(tmp_path_factory.mktemp("jobd-flat"))
    for f in files:
        assert os.path.isfile(f), f"bundle file missing on disk: {f}"
        shutil.copy(f, os.path.join(td, os.path.basename(f)))
    return td


def _run(td, *args):
    """python3 in the flat dir, with the REPO deliberately unreachable.

    `-P` (PYTHONSAFEPATH) keeps the script dir and cwd off sys.path so nothing
    is on the path by accident; PYTHONPATH is set to the flat dir and only the
    flat dir, which is exactly what jobd's own sys.path munging achieves on a
    box. A stale PYTHONPATH inherited from the pytest process would silently
    re-add tools/vast/ and make this test pass through the very hole it exists
    to find, so the env is built up, not copied.
    """
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": os.environ.get("HOME", td),
           "PYTHONPATH": td,
           "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-P", *args], cwd=td, env=env,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("mod", _MUST_IMPORT)
def test_bundle_module_imports_from_the_flat_layout(flat_bundle, mod):
    r = _run(flat_bundle, "-c", f"import {mod}")
    assert r.returncode == 0, (
        f"`import {mod}` FAILS in the delivered flat layout (it passes in the "
        f"repo, where the parent dir tools/vast/ closes the gap). Every "
        f"python3 call jobd.sh makes on a box would die here, silently:\n"
        f"{r.stderr.strip()}\n"
        f"FIX: add the missing module to bundle._job_attach_files() (and to "
        f"_PINNED_JOBD_BUNDLE in test_broker_env.py).")


def test_jobd_py_runs_from_the_flat_layout(flat_bundle):
    """`jobd.py --help` forces every top-level import in the daemon's entry
    point — the cheapest invocation that proves the box's `python3 jobd.py ...`
    calls can start at all."""
    r = _run(flat_bundle, "jobd.py", "--help")
    assert r.returncode == 0, (
        f"`python3 jobd.py --help` FAILS in the delivered flat layout — jobd's "
        f"event emission and ticket parsing are dead on every box launched from "
        f"this bundle (jobd.sh swallows the error with `|| true`, so the box "
        f"just looks idle while it bills):\n{r.stderr.strip()}")
    assert "usage:" in r.stdout


def test_the_flat_layout_is_actually_isolated(flat_bundle):
    """Guard the guard: if the subprocess could see tools/vast/ after all, every
    assertion above would be vacuous. A module that is a tools/vast/ sibling but
    deliberately NOT in the bundle must be unimportable.

    THE CANARY MUST BE A SIBLING THAT STILL HAS A BODY. `vastconf.py` and
    `jobmatrix.py` become one-line deprecation shims at plan step 7
    (`from vastlib.… import *`-shaped re-exports). Those still fail to import in
    the flat subprocess — vastlib lives UNDER tools/vast, so a subprocess that
    could see the shim could see vastlib too, and the contrapositive holds — but
    they fail on `No module named 'vastlib'`, which is one indirection away from
    the thing being asserted: a broken shim target would then look exactly like
    a correctly isolated subprocess. `hosts.py` / `boxstate.py` are the
    deferred-not-ported siblings and keep real bodies, so they are tried first,
    and the assertion now requires the failure to NAME the canary.
    """
    here = os.path.dirname(os.path.abspath(herdd.__file__))
    bundled = {os.path.basename(f) for f in bundle._job_attach_files()}
    canary = next((n for n in ("hosts.py", "boxstate.py", "vastconf.py")
                   if os.path.isfile(os.path.join(here, n)) and n not in bundled), None)
    assert canary, "no unbundled tools/vast/ sibling left to use as a canary"
    r = _run(flat_bundle, "-c", f"import {canary[:-3]}")
    assert r.returncode != 0, (
        f"{canary} imported from the 'flat' bundle — the subprocess can still "
        f"see the repo, so these tests prove nothing.")
    assert f"No module named '{canary[:-3]}'" in r.stderr, (
        f"{canary} failed to import for the WRONG reason — the isolation this "
        f"test guards is 'the repo is invisible', not 'the module is broken':\n"
        f"{r.stderr.strip()}")


# --- the pre-spend gate: both delivery paths refuse a broken bundle ---------- #
def test_import_gate_passes_the_real_bundle(capsys):
    bundle._jobd_import_gate(bundle._job_attach_files())   # must not exit
    assert "IMPORT CLOSURE BROKEN" not in capsys.readouterr().err


def test_import_gate_refuses_a_bundle_missing_bidpolicy(capsys):
    """`_stage_jobd_bootstrap` (launch --jobs) and `cmd_job_attach` both call
    this BEFORE any B2 write or ssh, so the 2026-08-14 bundle would have been
    refused at the workstation instead of shipped to a box."""
    files = [f for f in bundle._job_attach_files()
             if os.path.basename(f) != "bidpolicy.py"]
    with pytest.raises(SystemExit) as e:
        bundle._jobd_import_gate(files)
    assert "not import-closed" in str(e.value)
    err = capsys.readouterr().err
    assert "bidpolicy" in err and "jobmeta.py" in err


def test_import_gate_escape_hatch_is_explicit(monkeypatch, capsys):
    files = [f for f in bundle._job_attach_files()
             if os.path.basename(f) != "bidpolicy.py"]
    monkeypatch.setenv("JOBD_NO_IMPORT_CHECK", "1")
    bundle._jobd_import_gate(files)                          # must not exit
    assert "shipping the broken bundle anyway" in capsys.readouterr().err


def test_every_bundled_python_file_at_least_parses(flat_bundle):
    """Cheap syntax gate over the whole bundle (probes included): a file that
    does not compile can never run on a box, and py_compile needs no imports."""
    py = [f for f in os.listdir(flat_bundle) if f.endswith(".py")]
    assert py, "no python in the bundle?"
    r = _run(flat_bundle, "-m", "py_compile", *py)
    assert r.returncode == 0, r.stderr.strip()
