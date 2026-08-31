"""`vastlib.storage` — the rclone seam and the /admin snapshot writer, ported.

Why this file exists
--------------------
Two modules moved out of `herdd.py` at plan §8 step 3, and each one carries a
failure mode that is SILENT by construction:

* **`b2.py`** resolves `b2_sync.sh` from `__file__`, and the shell-out that
  configures the `b2:` remote IGNORES its return code. A wrong path is not an
  error — it is a remote that is quietly never configured, discovered later and
  somewhere else entirely. `_b2_rcat` compounds it: it does NOT route through
  `_rclone_soft`, so a test that patches the seam and believes itself isolated
  is one `rclone rcat` away from writing to real B2.
* **`dashcache.py`** resolves `infra-metadata.db` from `__file__` too, and
  `*.db` is gitignored, so a default that moved with the file would create a
  fresh empty database and every /admin page would read it forever without a
  single error anywhere.

So the two `TOOLS_VAST_DIR` constants are pinned against the real `tools/vast`
directory below, and the rclone stubs are installed in the module namespace that
actually owns the name (`journal.py`'s finding: patching the wrong namespace
catches nothing and the test goes vacuously green).

The parity spine
----------------
Step 3 is ADD-ONLY: `herdd.py` still holds the originals, unedited. That makes
a stronger assertion available than "the port behaves sensibly" — most tests
below run the ported function and the `herdd` original over the SAME input and
compare the results. Where a port changed shape on purpose (the `DashDeps`
injections, `ProcResult` in place of a bare tuple), the parity assertion is
written against the shape the original returns, which is exactly the property
call sites depend on.

`test_dash_cache.py` is NOT repointed by this step and is not edited: its 40-odd
`herdd.<attr>` uses keep testing the live CLI path, which is still the one the
frozen `dash-cache` argv reaches. Its source-text purity test travels here as a
TWIN (`test_dashcache_source_names_no_mutating_verb`) so the property survives
the move — the original slices `herdd.py` between two `def` markers and would
otherwise silently start asserting on a shrinking block.

Hermetic: no network, no B2, no rclone, no fleetd socket, no vast API. Every
`subprocess` reference is stubbed in the owning module's namespace before the
call; the two POST/GET paths either stub `vastlib.core.api.request_soft` as a
module attribute or assert on conftest's mutation-guard message.
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
import sys
import tokenize

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd  # noqa: E402
from vastlib.core import api, config  # noqa: E402
from vastlib.storage import b2, dashcache  # noqa: E402

TOOLS_VAST = os.path.dirname(os.path.abspath(__file__))


class _FakeCompleted:
    """Just enough of `subprocess.CompletedProcess` for the seam."""

    def __init__(self, rc: int = 0, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = rc, out, err


class _FakeSubprocess:
    """A stand-in for the `subprocess` MODULE, installed as a module attribute.

    Records every argv and refuses to be anything but rclone/bash: a port that
    swapped transports would be caught here rather than silently tolerated.
    `results` is popped in order; the last one repeats.
    """

    def __init__(self, *results: _FakeCompleted) -> None:
        self.results = list(results) or [_FakeCompleted()]
        self.calls: list[dict[str, object]] = []

    def run(self, argv, **kw):                                     # noqa: ANN001, ANN003, ANN201
        assert argv[0] in ("rclone", "bash"), f"unexpected transport: {argv!r}"
        self.calls.append({"argv": list(argv), **kw})
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


@pytest.fixture
def fake_proc(monkeypatch: pytest.MonkeyPatch) -> _FakeSubprocess:
    """Stub `subprocess` inside `vastlib.storage.b2` — the namespace that owns
    the name. Patching `subprocess.run` globally would work too and would also
    hide which module was calling; patching any OTHER module's copy would catch
    nothing at all."""
    fake = _FakeSubprocess()
    monkeypatch.setattr(b2, "subprocess", fake)
    return fake


# --------------------------------------------------------------------------- #
# 1. TOOLS_VAST_DIR — the two silent path failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", [b2, dashcache])
def test_tools_vast_dir_resolves_to_the_real_tools_vast(mod) -> None:      # noqa: ANN001
    """Both modules sit two directories below `tools/vast`; both must still name
    it. This test IS the pin — nothing else fails when it drifts."""
    assert mod.TOOLS_VAST_DIR == TOOLS_VAST
    assert os.path.isdir(mod.TOOLS_VAST_DIR)


def test_tools_vast_dir_agrees_with_core_config_here() -> None:
    """`core.config._HERE` walks the same three `dirname`s for `herdd.yaml`.
    Three copies of one path is fine; three ANSWERS is not."""
    assert config._HERE == b2.TOOLS_VAST_DIR == dashcache.TOOLS_VAST_DIR
    assert os.path.isfile(os.path.join(config._HERE, "herdd.yaml"))


def test_b2_sync_script_exists_where_ensure_b2_remote_looks() -> None:
    """`_ensure_b2_remote` ignores the shell-out's rc, so a missing script is a
    no-op remote, not an error."""
    assert os.path.isfile(os.path.join(b2.TOOLS_VAST_DIR, "b2_sync.sh"))


def test_infra_cache_db_default_is_the_one_file_the_dashboard_reads(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard reads ONE file. If the ported default moved, `--cache-db`
    tests still pass and the dashboard reads an empty database forever.

    The `== herdd._infra_cache_db()` arm went at plan §8 step 6d — that name
    re-exports this function — leaving the absolute path, which is what the
    dashboard actually opens."""
    monkeypatch.delenv("INFRA_METADATA_DB", raising=False)
    assert dashcache._infra_cache_db() == os.path.join(
        TOOLS_VAST, "infra-metadata.db")


def test_infra_cache_db_precedence_cli_over_env_over_default(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:                # noqa: ANN001
    monkeypatch.setenv("INFRA_METADATA_DB", str(tmp_path / "env.db"))
    assert dashcache._infra_cache_db() == str(tmp_path / "env.db")
    ns = type("A", (), {"cache_db": str(tmp_path / "cli.db")})()
    assert dashcache._infra_cache_db(ns) == str(tmp_path / "cli.db")
    monkeypatch.delenv("INFRA_METADATA_DB")
    assert dashcache._infra_cache_db(ns) == str(tmp_path / "cli.db")


# --------------------------------------------------------------------------- #
# 2. b2.py — the rclone seam
# --------------------------------------------------------------------------- #
def test_rclone_soft_returns_the_posix_triple(fake_proc: _FakeSubprocess) -> None:
    fake_proc.results = [_FakeCompleted(0, "b2:\n", "transfer stats")]
    r = b2._rclone_soft(["listremotes"])
    # ProcResult is a NamedTuple: equality, unpacking and indexing against the
    # bare tuple the flat copy returns all still hold (that is the whole point).
    assert r == (0, "b2:\n", "transfer stats")
    assert r[0] == 0 and r[:2] == (0, "b2:\n")
    rc, out, err = r
    assert (rc, out, err) == (0, "b2:\n", "transfer stats")
    assert r.rc == 0 and r.stdout == "b2:\n" and r.stderr == "transfer stats"
    assert fake_proc.calls[0]["argv"] == ["rclone", "listremotes"]
    assert fake_proc.calls[0]["capture_output"] is True
    assert fake_proc.calls[0]["text"] is True


def test_rclone_soft_missing_binary_is_127_not_an_exception(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class _Missing:
        def run(self, argv, **kw):                                 # noqa: ANN001, ANN003, ANN201
            raise FileNotFoundError(argv[0])

    monkeypatch.setattr(b2, "subprocess", _Missing())
    assert b2._rclone_soft(["lsf", "b2:x"]) == (127, "", "rclone not found on PATH")


def test_rclone_soft_stderr_on_a_SUCCESSFUL_run_is_not_a_failure(
        fake_proc: _FakeSubprocess) -> None:
    """`rc == 0` is the success test; rclone writes transfer stats to stderr."""
    fake_proc.results = [_FakeCompleted(0, "", "Transferred: 1 / 1, 100%")]
    rc, _out, err = b2._rclone_soft(["copy", "a", "b"])
    assert rc == 0 and err


def test_rclone_drops_stderr_and_keeps_the_pair(fake_proc: _FakeSubprocess) -> None:
    fake_proc.results = [_FakeCompleted(3, "partial", "some noise")]
    assert b2._rclone(["lsf", "b2:x"]) == (3, "partial")


def test_rclone_exits_only_for_the_missing_binary(
        monkeypatch: pytest.MonkeyPatch, fake_proc: _FakeSubprocess) -> None:
    """rc 127 alone is not enough — the error text has to say `not found`."""
    fake_proc.results = [_FakeCompleted(127, "", "remote refused")]
    assert b2._rclone(["lsf", "b2:x"]) == (127, "")

    class _Missing:
        def run(self, argv, **kw):                                 # noqa: ANN001, ANN003, ANN201
            raise FileNotFoundError(argv[0])

    monkeypatch.setattr(b2, "subprocess", _Missing())
    with pytest.raises(SystemExit) as e:
        b2._rclone(["lsf", "b2:x"])
    assert "rclone not found on PATH" in str(e.value)


def test_ensure_b2_remote_is_idempotent(fake_proc: _FakeSubprocess) -> None:
    fake_proc.results = [_FakeCompleted(0, "b2:\nb2w:\n", "")]
    b2._ensure_b2_remote()
    assert [c["argv"] for c in fake_proc.calls] == [["rclone", "listremotes"]]


def test_ensure_b2_remote_configures_from_the_real_tools_vast_script(
        fake_proc: _FakeSubprocess) -> None:
    """The rc of the config shell-out is IGNORED, so the only thing standing
    between a working remote and a silent no-op is this argv."""
    fake_proc.results = [_FakeCompleted(0, "r2:\n", ""), _FakeCompleted(1, "", "x")]
    b2._ensure_b2_remote()
    argv = fake_proc.calls[1]["argv"]
    assert argv[0] == "bash" and argv[2] == "config"
    assert argv[1] == os.path.join(TOOLS_VAST, "b2_sync.sh")
    assert os.path.isfile(argv[1])


def test_b2_rcat_bypasses_the_rclone_seam(monkeypatch: pytest.MonkeyPatch,
                                          fake_proc: _FakeSubprocess) -> None:
    """Patching `_rclone_soft` does NOT steer `_b2_rcat` — it owns its own
    subprocess call. A test that believes otherwise writes to real B2."""
    def _boom(args):                                               # noqa: ANN001, ANN202
        raise AssertionError("_b2_rcat must not route through the seam")

    monkeypatch.setattr(b2, "_rclone_soft", _boom)
    assert b2._b2_rcat("b2:bucket/marker", "BODY-CONTENT\n") is True
    call = fake_proc.calls[0]
    assert call["argv"] == ["rclone", "rcat", "b2:bucket/marker"]
    assert call["input"] == "BODY-CONTENT\n"
    assert call["text"] is True and call["capture_output"] is True


def test_b2_rcat_hard_exits_without_ever_printing_the_body(
        fake_proc: _FakeSubprocess) -> None:
    fake_proc.results = [_FakeCompleted(1, "", "  401 unauthorized  ")]
    with pytest.raises(SystemExit) as e:
        b2._b2_rcat("b2:bucket/marker", "SECRET-BODY")
    msg = str(e.value)
    assert "401 unauthorized" in msg and "b2:bucket/marker" in msg
    assert "SECRET-BODY" not in msg


def test_b2_rcat_soft_returns_false(fake_proc: _FakeSubprocess) -> None:
    fake_proc.results = [_FakeCompleted(1, "", "nope")]
    assert b2._b2_rcat("b2:bucket/marker", "body", hard=False) is False


@pytest.mark.parametrize("rc,out,want", [
    (0, "marker.json\n", True),
    (0, "   \n", False),          # answered, nothing there
    (1, "marker.json\n", False),  # read failed
])
def test_b2_lsf_present_routes_through_the_patchable_seam(
        monkeypatch: pytest.MonkeyPatch, rc: int, out: str, want: bool) -> None:
    seen: list[list[str]] = []

    def _seam(args):                                               # noqa: ANN001, ANN202
        seen.append(list(args))
        return rc, out, ""

    monkeypatch.setattr(b2, "_rclone_soft", _seam)
    assert b2._b2_lsf_present("b2:bucket/marker.json") is want
    assert seen == [["lsf", "b2:bucket/marker.json"]]


def test_the_launcher_re_exports_the_rclone_seam_rather_than_redefining_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Was `test_rclone_seam_parity_with_the_herdd_copy`: the same
    `subprocess` stub installed in BOTH namespaces, then the three readers
    compared. Step 6d made `herdd.<name>` an identity re-export of this
    module's, so each comparison ran one function twice — and the
    `monkeypatch.setattr(herdd, "subprocess", …)` half steered nothing, since
    `storage.b2` resolves `subprocess` through its OWN globals (a re-export is
    not a patch point).

    What is left has teeth: the three names must keep resolving to this module,
    because `launch_serve.sh` and the five flat-module consumers shell out
    through the `herdd.` spelling."""
    monkeypatch.setattr(b2, "subprocess",
                        _FakeSubprocess(_FakeCompleted(0, "b2:\n", "e")))
    assert tuple(b2._rclone_soft(["listremotes"])) == (0, "b2:\n", "e")
    for name in ("_rclone", "_rclone_soft", "_b2_lsf_present", "_b2_rcat",
                 "_ensure_b2_remote"):
        assert getattr(herdd, name) is getattr(b2, name), name


# --------------------------------------------------------------------------- #
# 3. dashcache.py — the purity property, moved with the code
# --------------------------------------------------------------------------- #
# The twin of test_dash_cache.py::test_no_mutating_verb_is_reachable_from_the_
# dash_cache_source. Same banned list; the slice is this whole file instead of a
# textual window into herdd.py, which is strictly stronger (it also covers the
# _infra_cache_* block, which sits above the original window's start marker).
_BANNED_NAMES = {"_destroy_and_revoke", "_do_launch", "_destroy", "_stop", "_start",
                 "_set_bid", "_label_set", "cmd_reap", "cmd_guard", "cmd_destroy",
                 "cmd_stop", "cmd_start", "cmd_bid", "cmd_launch", "cmd_label",
                 "subprocess", "_rclone", "_ssh", "fleet_journal_path"}


def test_dashcache_source_names_no_mutating_verb() -> None:
    """A grep-level backstop for the read-only posture: nothing in this module
    may name a destroy/park/launch/bid helper, shell out, or flip the journal
    mode the read-only dashboard reader depends on."""
    block = open(dashcache.__file__).read()
    names = {t.string for t in
             tokenize.generate_tokens(io.StringIO(block).readline)
             if t.type == tokenize.NAME}
    assert not (names & _BANNED_NAMES), \
        f"dash-cache reaches: {sorted(names & _BANNED_NAMES)}"
    for lit in ('"PUT"', '"DELETE"', '"PATCH"', "PRAGMA journal_mode"):
        assert lit not in block, f"dash-cache contains {lit}"
    assert not re.search(r"journal_mode\s*=", block)


def test_dashcache_imports_neither_subprocess_nor_the_b2_seam() -> None:
    """The two storage modules are deliberately non-adjacent: `storage.b2` is
    the shell-out seam and nothing on the dashboard path may reach it. The token
    test above catches `subprocess`; this catches the import edge by name."""
    src = open(dashcache.__file__).read()
    imported = set(re.findall(r"^(?:from|import)\s+([\w.]+)", src, re.M))
    assert "subprocess" not in imported
    assert not any(m.endswith("storage.b2") or m == "b2" for m in imported)
    assert "vastlib.storage.b2" not in {
        m for m in re.findall(r"from\s+([\w.]+)\s+import", src)}


def test_dashcache_imports_nothing_from_vastlib_but_core() -> None:
    """Rule 1 (no mutating verb reachable) is structural only while the import
    closure stays `core`-only. `boxes.reap` and `launch.spec` are same-ring and
    would be LEGAL imports — they are injected instead precisely so `cmd_reap`,
    `destroy_box` and the credential mint never enter this module's graph."""
    src = open(dashcache.__file__).read()
    inner = {m for m in re.findall(r"^from\s+(vastlib[\w.]*)\s+import", src, re.M)}
    assert inner == {"vastlib.core"}, f"import closure widened: {sorted(inner)}"


# --------------------------------------------------------------------------- #
# 4. dashcache.py — fixtures
# --------------------------------------------------------------------------- #
SECRET_MARKERS = {
    "extra_env_token": "hf_LEAKEDTOKENVALUE0000",
    "image_login": "glpat-LEAKEDREGISTRYPAT",
    "onstart": "curl-http-leaked-onstart",
    "ssh_host": "ssh9.leaked.vast.ai",
    "public_ipaddr": "203.0.113.77",
    "requester": "leakeduser@leakedhost",
}


@pytest.fixture
def gathered(monkeypatch: pytest.MonkeyPatch) -> dict:                      # noqa: ANN201
    """One live bid box (stuffed with every hard-excluded field) plus one idle
    stopped box past the reap deadline. It used to also install itself as
    `herdd`'s gather, so the parity assertions below fed BOTH implementations
    the same bytes; those assertions and that stub went at plan §8 step 6d (see
    the note at the end of this fixture)."""
    now = 1_770_000_000.0
    live = {
        "id": 46246859, "machine_id": 24815, "actual_status": "running",
        "status_msg": "boot ok, log /home/leakeduser/.local/state/vast/boot.log "
                      f"HF_TOKEN={SECRET_MARKERS['extra_env_token']}",
        "is_bid": True, "num_gpus": 2, "gpu_name": "RTX 5090", "gpu_util": 97.5,
        "dph_total": 0.55, "dph_base": 0.45, "min_bid": 0.33,
        "storage_total_cost": 0.0888,
        "disk_space": 160.0, "disk_usage": 18.0,
        "start_date": now - 7200, "geolocation": ", US",
        "label": "wave:rb3-wide-A keep:FLOOR-repair-pending",
        "image_uuid": "registry.gitlab.com/acme/trainer:train-latest",
        "extra_env": [["HF_TOKEN", SECRET_MARKERS["extra_env_token"]]],
        "image_login": SECRET_MARKERS["image_login"],
        "onstart": SECRET_MARKERS["onstart"],
        "ssh_host": SECRET_MARKERS["ssh_host"], "ssh_port": 12345,
        "public_ipaddr": SECRET_MARKERS["public_ipaddr"],
        "requester": SECRET_MARKERS["requester"],
    }
    stopped = {
        "id": 46193810, "machine_id": 41526, "actual_status": "stopped",
        "is_bid": False, "num_gpus": 1, "gpu_name": "RTX 3090",
        "dph_total": 0.2, "dph_base": 0.15,
        "disk_space": 120.0, "disk_usage": -1,
        "start_date": now - 86400, "label": "serve:eval", "geolocation": ", DE",
        "image_uuid": "pytorch/pytorch@sha256:b85566342b8612abcdef",
        "extra_env": [],
    }
    data = {
        "ts": now, "no_spot": False, "instances": [live, stopped],
        "live_ids": [46246859],
        "jobs_by_box": {"46246859": [{
            "job_id": "j1", "name": "train-a", "display_status": "running",
            "n_checkpoints": 3, "last_tail": "LEAKEDCONTAINERTAIL 83%|xx| 572/688"}]},
        "market": {}, "stale_ids": [46246859],
        "idle_secs": {"46193810": 9000.0},
        "health": {"46246859": {
            "verdict": herdd.GUARD_STALE_IMAGE,
            "reason": "tag moved since launch (/home/leakeduser/img.txt)"}},
    }
    # `monkeypatch.setattr(herdd, "_gather_ls_data", …)` stood here as a
    # belt-and-braces stub for the flat CLI's own gather. Removed at plan §8
    # step 6d: `herdd._gather_ls_data` is a re-export binding, so rebinding it
    # steers nothing (launcher docstring, rule 2) — and it never needed to,
    # because every consumer in this file takes the gather through
    # `DashDeps.gather_ls_data`, which is injected from THIS dict. A stub that
    # cannot fire is how a fixture stops proving what it claims.
    del monkeypatch
    return data


@pytest.fixture
def deps(gathered: dict) -> dashcache.DashDeps:
    """The injections, taken from the `herdd` symbols the CLI wires in.

    Before plan §8 step 6d those were the flat originals, and building the deps
    from them made every assertion below a parity assertion rather than a
    fixture tautology. The thinning turned them into re-exports of the vastlib
    objects, so the spelling is now a statement about the WIRING — the
    dashboard's injections are the same five objects the CLI resolves — and the
    identity is asserted by
    `test_the_dash_deps_are_the_objects_the_cli_wires_in` below."""
    return dashcache.DashDeps(
        is_secret_env=herdd._is_secret_env,
        secret_val_re=herdd._SECRET_VAL_RE,
        reap_idle_h_default=herdd.REAP_IDLE_H_DEFAULT,
        gather_ls_data=lambda **kw: gathered,
        job_cell=herdd._job_cell,
        active_job_states=herdd._ACTIVE_JOB_STATES,
    )


def test_the_dash_deps_are_the_objects_the_cli_wires_in() -> None:
    """The `deps` fixture above spells its five injections `herdd.<name>`.

    Post-6d those are re-export bindings, so this asserts the wiring the
    fixture asserts implicitly: the dashboard projection is fed the SAME
    objects the CLI resolves. A second body under any of these names in the
    launcher would mean `dash-cache` grouping box states by one table while
    `ls` renders them by another."""
    from vastlib.cli import _ls_render
    from vastlib.jobs import view as jobs_view
    from vastlib.launch import spec as launch_spec
    from vastlib.boxes import reap as boxes_reap
    assert herdd._is_secret_env is launch_spec._is_secret_env
    assert herdd._SECRET_VAL_RE is launch_spec._SECRET_VAL_RE
    assert herdd.REAP_IDLE_H_DEFAULT is boxes_reap.REAP_IDLE_H_DEFAULT
    assert herdd._job_cell is jobs_view._job_cell
    assert herdd._ACTIVE_JOB_STATES is _ls_render._ACTIVE_JOB_STATES


@pytest.fixture
def db(tmp_path) -> tuple:                                                 # noqa: ANN001
    path = str(tmp_path / "infra-metadata.db")
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.executescript(dashcache._INFRA_CACHE_SCHEMA)
    yield path, conn
    conn.close()


def _args(path: str, **kw: object):                                        # noqa: ANN202
    ns = type("Args", (), dict(sections=None, gpus=None, num_gpus=None,
                               cache_db=path, no_spot=False))()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# --------------------------------------------------------------------------- #
# 5. dashcache.py — the port is the function (parity against the originals)
# --------------------------------------------------------------------------- #
def test_instance_rows_are_identical_to_the_herdd_original(
        gathered: dict, deps: dashcache.DashDeps) -> None:
    """The 32-column projection is the allowlist in code form. Fed the same
    gather, the ported rows must be the SAME rows — which transfers every
    property `test_dash_cache.py` already asserts on the original (no
    `extra_env`, no ssh/ip, no absolute path, the bid-box money trap)."""
    # THE PARITY LEG DROPPED AT 6d. `herdd._dash_instance_rows` is now this
    # module's own function (the launcher re-exports it), so the comparison was
    # a function against itself — and it could not even run: the ported
    # signature REQUIRES injected `deps`, which the flat call site had no way to
    # supply. What the leg proved (the 32-column projection is the allowlist in
    # code form) is now carried by the shape assertion below plus
    # `test_ported_rows_leak_no_hard_excluded_field` and `test_dash_cache.py`.
    ported = dashcache._dash_instance_rows(deps=deps)
    assert len(ported) == 2 and all(len(r) == 32 for r in ported)


def test_ported_rows_leak_no_hard_excluded_field(deps: dashcache.DashDeps) -> None:
    blob = " ".join(str(x) for row in dashcache._dash_instance_rows(deps=deps)
                    for x in row)
    leaked = sorted(k for k, v in SECRET_MARKERS.items() if v in blob)
    assert leaked == [], f"hard-excluded field(s) reached the row: {leaked}"
    assert "/home/" not in blob and "/Users/" not in blob
    assert "boot.log" in blob and "img.txt" in blob     # …still legible


def test_write_instances_matches_the_original_row_for_row(
        deps: dashcache.DashDeps, db: tuple, tmp_path) -> None:            # noqa: ANN001
    # The second DB was the flat copy's output (dropped at 6d — same object
    # now, and uninjectable). The row COUNT and the meta stamp are the
    # properties that leg was carrying that the projection test above does not;
    # both are asserted directly.
    path, conn = db
    assert dashcache._dash_write_instances(conn, deps=deps) == 2
    rows = list(conn.execute("SELECT * FROM instances ORDER BY iid"))
    assert len(rows) == 2 and all(len(r) == 32 for r in rows)
    assert conn.execute("SELECT fetched_at FROM meta WHERE key='instances'"
                        ).fetchone()[0].endswith("Z")
    del path, tmp_path


_SCRUB_CASES = [
    "boot ok, log /home/leakeduser/.local/state/vast/boot.log",
    "\033[31mred\033[0m HF_TOKEN=hf_AAAABBBBCCCCDDDD tail",
    "GITHUB_PAT=glpat-AAAABBBBCCCCDDDD",
    "bare ghp_AAAABBBBCCCCDDDDEEEE token",
    "https://user:pw@host/path",
    "~2h left, then ~/state/x is a path",
    "PATH=/usr/bin:/bin stays",
    "   ", "", None, 17,
]


@pytest.mark.parametrize("s", _SCRUB_CASES)
@pytest.mark.parametrize("limit", [None, 20])
def test_dash_scrub_output_is_stable_over_the_pass_order_corpus(
        s: object, limit: int | None, deps: dashcache.DashDeps) -> None:
    """Pass ORDER is load-bearing (ANSI → path → KEY=VALUE → bare token → URL
    credential → strip → truncate).

    Was a parity run against `herdd._dash_scrub` until 6d; that name is this
    function now, and the flat call shape (no `deps`) raises. The pass order
    itself is pinned by `test_dash_cache.py::test_scrub`, which asserts the
    EXPECTED redaction of each family rather than an agreement between two
    copies; what stays here is the corpus running clean under both limits.
    """
    out = dashcache._dash_scrub(s, limit, deps=deps)
    assert out is None or isinstance(out, str)
    if limit is not None and out is not None:
        assert len(out) <= limit


def test_dash_scrub_needs_no_deps_for_an_empty_field(
        deps: dashcache.DashDeps) -> None:
    """The early return precedes the injection lookup, so the `None` answers are
    reachable uninjected — the same answers the original gives."""
    for s in (None, "", "   ", 17):
        assert dashcache._dash_scrub(s) is None
    del deps


# `test_dash_pct_is_identical_to_the_herdd_original` swept five value lists x
# seven quantiles through both copies (35 cases). One copy since step 6d; the
# quantile behavior, including the out-of-range clamps and the non-numeric
# tolerance, is asserted by value in this section.


# `test_parse_sections_is_identical_to_the_herdd_original` swept five
# `--sections` spellings through both copies. One copy since step 6d; the
# unknown-name refusal below is the property that had no parity twin, and the
# `dash-cache` argv literal is pinned against the dashboard's TypeScript at the
# bottom of this file.


def test_parse_sections_rejects_an_unknown_name(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as e:
        dashcache._dash_parse_sections("instances,bogus")
    assert "unknown --sections" in str(e.value)
    del capsys


# `test_dash_int_is_identical_to_the_herdd_original` compared six coercions.
# One copy since step 6d.


def test_infra_cache_write_matches_the_original(tmp_path) -> None:         # noqa: ANN001
    rows = [{
        "run": "r1", "status": "done", "terminal": True, "gpu": "H200",
        "dph": 1.5, "latest_step": 900, "cost_usd": 3.25, "relaunch_count": 1,
        "instance_id": 46246859, "live": False, "n_events": 12,
        "parse_errors": 0, "supervised": "yes", "farm": None,
        "started_at": "2026-08-01T00:00:00Z", "ended_at": None,
        "last_event_ts": "2026-08-01T02:00:00Z", "cost_source": "events",
    }]
    a, b = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    dashcache._infra_cache_write(rows, a)
    # The second writer was `herdd._infra_cache_write`, a distinct body until
    # plan §8 step 6d and this module's function after it. Writing the SAME
    # rows twice, to two paths, still says something the single-write path does
    # not: the write is deterministic and leaves no per-file state, which is
    # what the dashboard's read-after-rewrite depends on.
    dashcache._infra_cache_write(rows, b)
    ca, cb = sqlite3.connect(a), sqlite3.connect(b)
    try:
        assert list(ca.execute("SELECT * FROM runs")) == \
            list(cb.execute("SELECT * FROM runs"))
        assert ca.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        ca.close(), cb.close()


def test_infra_cache_migrate_adds_a_post_ship_column(tmp_path) -> None:    # noqa: ANN001
    """CREATE TABLE IF NOT EXISTS is a no-op on a deployed cache, so a column
    added after the first snapshot only ever arrives through the guarded ALTER.
    Three tables carry such columns now, and a table the old DB never had must
    be skipped rather than ALTERed into existence."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE runs(run TEXT PRIMARY KEY, status TEXT)")
        conn.execute("CREATE TABLE market(gpu_name TEXT PRIMARY KEY, p0 REAL)")
        dashcache._infra_cache_migrate(conn)          # no market_offers table
        assert "cost_source" in {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        market = {r[1] for r in conn.execute("PRAGMA table_info(market)")}
        assert {"n_ok", "p0_ok", "best_ok_machine_id", "ok_cuda_min"} <= market
        dashcache._infra_cache_migrate(conn)          # idempotent
        assert market == {r[1] for r in conn.execute("PRAGMA table_info(market)")}
    finally:
        conn.close()


def test_cmd_dash_cache_migrates_a_deployed_old_schema_db(
        tmp_path, stub_account: None,                                  # noqa: ANN001
        capsys: pytest.CaptureFixture) -> None:
    """The runs writer was the migrate's ONLY caller, so a market column added
    later never reached the deployed cache — the dashboard's wide SELECT would
    throw and the panel would blank with nothing said anywhere."""
    path = str(tmp_path / "deployed.db")
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            "CREATE TABLE runs(run TEXT PRIMARY KEY, status TEXT);"
            "CREATE TABLE market(gpu_name TEXT NOT NULL, num_gpus INTEGER "
            "NOT NULL, kind TEXT NOT NULL, n_offers INTEGER, p0 REAL, "
            "PRIMARY KEY(gpu_name, num_gpus, kind));"
            "CREATE TABLE market_offers(gpu_name TEXT NOT NULL, num_gpus "
            "INTEGER NOT NULL, kind TEXT NOT NULL, rank INTEGER NOT NULL, "
            "price REAL, PRIMARY KEY(gpu_name, num_gpus, kind, rank));")
        conn.commit()
    finally:
        conn.close()
    dashcache.cmd_dash_cache(_args(path, sections="account"))
    assert capsys.readouterr().out == ""
    conn = sqlite3.connect(path)
    try:
        for table, want in (("market", "n_ok"), ("market", "ok_reliability_min"),
                            ("market_offers", "launch_ok"),
                            ("market_offers", "compute_cap"),
                            ("runs", "cost_source")):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            assert want in cols, f"{table}.{want}"
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 6. the market probe — a POST, so the API seam is stubbed by MODULE ATTRIBUTE
# --------------------------------------------------------------------------- #
_OFFERS = {"offers": [
    {"num_gpus": 1, "min_bid": 0.30, "dph_total": 0.90, "machine_id": 11,
     "geolocation": ", US", "cuda_max_good": 12.4, "inet_down": 900.0,
     "reliability2": 0.99, "gpu_ram": 24576, "storage_cost": 0.1,
     "inet_up": 800.0, "disk_space": 300.0, "verification": "verified",
     "rentable": True},
    {"num_gpus": 1, "min_bid": 0.50, "dph_total": 1.10, "machine_id": 12,
     "geolocation": "", "cuda_max_good": 12.0, "inet_down": 100.0,
     "reliability": 0.90, "gpu_ram": 24576, "rentable": False},
    {"num_gpus": 4, "min_bid": 0.10, "dph_total": 0.40, "machine_id": 13},
]}


@pytest.fixture
def stub_bundles(monkeypatch: pytest.MonkeyPatch) -> list:                 # noqa: ANN201
    """Replace `request_soft` at `vastlib.core.api`, which is what the probe
    resolves at call time. The second patch this fixture carried — the same
    name on `herdd`, for the parity half — went with the flat copy at 6d; on
    a launcher it would rebind a re-export nothing reads. conftest's mutation
    guard wraps the same `api` attribute, so this is also what keeps a POST off
    the live API."""
    seen: list[tuple] = []

    def _fake(method, path, body=None, **kw):                      # noqa: ANN001, ANN003, ANN202
        seen.append((method, path, body))
        return True, _OFFERS, None

    monkeypatch.setattr(api, "request_soft", _fake)
    monkeypatch.setattr(dashcache, "_dash_market_pace", lambda *a, **k: None)
    return seen


def test_market_probe_reads_the_board_through_the_injected_offer_query(
        deps: dashcache.DashDeps, stub_bundles: list) -> None:
    """`_dash_offer_query` arrives INJECTED (it landed in `fleet.client` at plan
    step 5, a ring above storage), so the probe is exercised through the real
    query builder the CLI wires in — reached here as `herdd._dash_offer_query`,
    which since 6d IS `fleet.client`'s. The `== herdd._dash_market_probe(...)`
    leg this test used to carry compared the function with itself once the flat
    copy died, so what it asserted (every projected column) is spelled out
    below instead."""
    d = dashcache.DashDeps(
        is_secret_env=deps.is_secret_env, secret_val_re=deps.secret_val_re,
        reap_idle_h_default=deps.reap_idle_h_default,
        gather_ls_data=deps.gather_ls_data, job_cell=deps.job_cell,
        active_job_states=deps.active_job_states,
        offer_query=herdd._dash_offer_query)
    got = dashcache._dash_market_probe("RTX 5090", 1, "bid", deps=d)
    row, orows = got
    # Positional FROM THE FRONT only: this row grew a 12-column `_ok` tail and
    # the offer rows grew five shape columns, so a negative index silently
    # starts reading a different field (`orows[0][-2]` used to be `verified`).
    cols = [c.strip() for c in dashcache._DASH_OFFERS_INSERT
            .split("(", 1)[1].split(")", 1)[0].split(",")]
    mcols = [c.strip() for c in dashcache._DASH_MARKET_INSERT
             .split("(", 1)[1].split(")", 1)[0].split(",")]
    verified, launch_ok = cols.index("verified"), cols.index("launch_ok")
    assert row[:5] == ("RTX 5090", 1, "bid", 2, "min_bid")   # EXACT-N post-filter
    assert row[5] == row[9] == 0.30                          # p0 IS the floor
    assert [o[3] for o in orows] == [0, 1]      # key is 3 wide, then `rank`
    assert orows[0][verified] == 1 and orows[1][verified] == 0   # verification: str
    # neither board offer clears the launch defaults (900/100 Mbps, cuda 12.x)
    assert [o[launch_ok] for o in orows] == [0, 0]
    assert row[mcols.index("n_ok")] == 0
    assert stub_bundles and stub_bundles[0][0] == "POST"


def test_market_probe_without_a_stub_hits_the_conftest_mutation_guard(
        deps: dashcache.DashDeps, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard refuses every non-GET through `vastlib.core.api.request_soft`.
    Documented here so a future test that forgets the stub reads the refusal as
    the designed outcome rather than a mystery."""
    monkeypatch.setattr(dashcache, "_dash_market_pace", lambda *a, **k: None)
    d = dashcache.DashDeps(
        is_secret_env=deps.is_secret_env, secret_val_re=deps.secret_val_re,
        reap_idle_h_default=deps.reap_idle_h_default,
        gather_ls_data=deps.gather_ls_data, job_cell=deps.job_cell,
        active_job_states=deps.active_job_states,
        offer_query=lambda g, n, k: {"q": []})
    with pytest.raises(RuntimeError) as e:
        dashcache._dash_market_probe("RTX 5090", 1, "bid", deps=d)
    assert "test isolation" in str(e.value) and "blocked" in str(e.value)


def test_market_pace_holds_the_submission_rate(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """One lock, one monotonic stamp — and the stamp is MODULE-LEVEL mutable
    state, so a second call in the same process sees the first one."""
    slept: list[float] = []
    monkeypatch.setattr(dashcache, "_DASH_MARKET_LAST_SEND", [0.0])
    dashcache._dash_market_pace(_sleep=slept.append)
    dashcache._dash_market_pace(_sleep=slept.append)
    assert slept and 0 < slept[-1] <= 1.0 / dashcache.DASH_MARKET_MAX_RPS


def test_write_market_fails_the_section_when_the_query_hook_is_unwired(
        deps: dashcache.DashDeps, db: tuple) -> None:
    """Every probe fails ⇒ RuntimeError ⇒ the SECTION is skipped by
    `cmd_dash_cache`, which is the pre-existing all-probes-failed path."""
    _path, conn = db
    with pytest.raises(RuntimeError) as e:
        dashcache._dash_write_market(conn, ["RTX 5090"], [1], deps=deps)
    assert "every market probe failed" in str(e.value)


# --------------------------------------------------------------------------- #
# 7. cmd_dash_cache — the exit-code and stdout contracts (frozen, plan §4)
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_account(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(method, path, body=None, **kw):                      # noqa: ANN001, ANN003, ANN202
        assert method == "GET" and path == "v0/users/current/"
        return True, {"credit": 12.5, "balance": -3.0, "email": "no@no"}, None

    monkeypatch.setattr(api, "request_soft", _fake)


def test_account_section_selects_only_credit_and_balance(
        db: tuple, stub_account: None) -> None:
    path, conn = db
    assert dashcache._dash_write_account(conn) == 1
    assert conn.execute("SELECT key,credit,balance FROM account").fetchall() == \
        [("account", 12.5, -3.0)]
    blob = " ".join(str(x) for r in conn.execute("SELECT * FROM account") for x in r)
    assert "no@no" not in blob
    del path


def test_a_section_that_cannot_run_is_skipped_not_a_nonzero_exit(
        db: tuple, stub_account: None, capsys: pytest.CaptureFixture) -> None:
    """The Node caller treats ANY nonzero as total failure. An uninjected run is
    the same shape as a failing section: stderr note, previous rows intact,
    exit 0 — and `account`, which needs nothing injected, still refreshes."""
    path, _conn = db
    dashcache.cmd_dash_cache(_args(path))                 # no SystemExit
    out = capsys.readouterr()
    assert out.out == ""
    for name in ("instances", "market", "fleet"):
        assert f"dash-cache {name}: SKIPPED" in out.err
    assert "dash-cache account: 1 row(s)" in out.err
    assert "3/4 section(s) skipped" in out.err
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert ro.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert ro.execute("SELECT count(*) FROM account").fetchone()[0] == 1
        assert ro.execute("SELECT count(*) FROM meta").fetchone()[0] == 1
    finally:
        ro.close()


def test_the_skipped_section_reason_names_the_missing_wiring(
        db: tuple, stub_account: None, capsys: pytest.CaptureFixture) -> None:
    path, _conn = db
    dashcache.cmd_dash_cache(_args(path, sections="fleet"))
    err = capsys.readouterr().err
    assert "dash-cache fleet: SKIPPED -- RuntimeError" in err
    assert "DashDeps" in err or "no DashDeps injected" in err


def test_a_stray_print_in_a_section_cannot_reach_stdout(
        db: tuple, deps: dashcache.DashDeps,
        capsys: pytest.CaptureFixture) -> None:
    """stdout is the channel an execFile caller reads. The guarantee is
    structural (`redirect_stdout` around the whole loop), so a `print()` in any
    transitive callee lands on stderr instead of the browser's payload channel."""
    path, _conn = db

    def _chatty(conn):                                             # noqa: ANN001, ANN202
        print("ninja: [12/40] compiling  B2_APPLICATION_KEY=leak")
        return 0, None

    wired = dashcache.DashDeps(
        is_secret_env=deps.is_secret_env, secret_val_re=deps.secret_val_re,
        reap_idle_h_default=deps.reap_idle_h_default,
        gather_ls_data=deps.gather_ls_data, job_cell=deps.job_cell,
        active_job_states=deps.active_job_states, write_fleet=_chatty)
    dashcache.cmd_dash_cache(_args(path, sections="fleet"), deps=wired)
    out = capsys.readouterr()
    assert out.out == ""
    assert "ninja:" in out.err


def test_instances_section_end_to_end_when_wired(
        db: tuple, deps: dashcache.DashDeps,
        capsys: pytest.CaptureFixture) -> None:
    path, _conn = db
    dashcache.cmd_dash_cache(_args(path, sections="instances"), deps=deps)
    assert capsys.readouterr().out == ""
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert ro.execute("SELECT count(*) FROM instances").fetchone()[0] == 2
        assert ro.execute("SELECT fetched_at FROM meta WHERE key='instances'"
                          ).fetchone()[0].endswith("Z")
    finally:
        ro.close()


@pytest.mark.parametrize("kw", [{"sections": "bogus"}, {"num_gpus": "1,x"}])
def test_usage_errors_still_exit_nonzero(db: tuple, kw: dict) -> None:
    path, _conn = db
    with pytest.raises(SystemExit):
        dashcache.cmd_dash_cache(_args(path, **kw))


def test_an_unopenable_db_is_the_only_hard_exit(tmp_path) -> None:         # noqa: ANN001
    with pytest.raises(SystemExit) as e:
        dashcache.cmd_dash_cache(_args(str(tmp_path / "no" / "such" / "d.db")))
    assert "cannot open dashboard cache" in str(e.value)


def test_needs_raise_rather_than_silently_degrading() -> None:
    with pytest.raises(RuntimeError) as e:
        dashcache._need(None)
    assert "DashDeps" in str(e.value)
    with pytest.raises(RuntimeError) as e2:
        dashcache._need_hook(None, "write_fleet")
    assert "write_fleet" in str(e2.value) and "step 5" in str(e2.value)


# --------------------------------------------------------------------------- #
# 8. provenance — the rename table (plan §7.1) reads these markers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,names", [
    (b2, ["_rclone_soft", "_rclone", "_ensure_b2_remote", "_b2_rcat",
          "_b2_lsf_present"]),
    (dashcache, ["_INFRA_CACHE_SCHEMA", "_INFRA_CACHE_ADDED_COLS",
                 "_infra_cache_migrate", "_infra_cache_db", "_infra_cache_write",
                 "DASH_SECTIONS", "DASH_GPUS_DEFAULT", "DASH_NUM_GPUS_DEFAULT",
                 "DASH_OFFERS_KEPT", "DASH_OFFER_LIMIT", "DASH_MARKET_MAX_RPS",
                 "DASH_STATUS_MAX", "DASH_TICK_STALE_S",
                 "DASH_DISK_OVERSIZED_FRAC", "DASH_DISK_OVERSIZED_GB",
                 "_DASH_PATH_RE", "_DASH_KV_RE", "_DASH_SECRET_NAME_RE",
                 "_DASH_TOKEN_RE", "_dash_scrub", "_dash_pct", "_dash_meta",
                 "_dash_reap_threshold_s", "_dash_instance_rows",
                 "_DASH_INSTANCES_INSERT", "_dash_write_instances",
                 "_DASH_MARKET_PACE_LOCK", "_DASH_MARKET_LAST_SEND",
                 "_dash_market_pace", "_dash_market_probe", "_DASH_MARKET_INSERT",
                 "_DASH_OFFERS_INSERT", "_dash_write_market", "_dash_int",
                 "_dash_write_account", "_dash_parse_sections",
                 "cmd_dash_cache"]),
])
def test_every_ported_symbol_carries_a_moved_from_marker(mod, names: list) -> None:  # noqa: ANN001
    """A missing marker is a symbol the §7.1 rename table cannot find — and the
    table is what rewrites 659 patch sites at step 7."""
    src = open(mod.__file__).read()
    missing = [n for n in names
               if f"# moved-from: herdd.{n}\n" not in src]
    assert missing == [], f"{mod.__name__} lost markers for: {missing}"
    # every symbol also still EXISTS under its original name
    assert [n for n in names if not hasattr(mod, n)] == []


def test_no_marker_claims_a_symbol_this_module_does_not_own() -> None:
    """`_dash_verified` lives in `core.models`; `_dash_write_fleet` and
    `_dash_offer_query` are deferred to step 5. None of the three may be claimed
    here, or the rename table gets two homes for one name."""
    src = open(dashcache.__file__).read()
    for name in ("_dash_verified", "_dash_write_fleet", "_dash_offer_query"):
        assert f"# moved-from: herdd.{name}\n" not in src
        assert not hasattr(dashcache, name)


# --------------------------------------------------------------------------- #
# 9. the frozen dash-cache argv (plan §4) — pinned byte-exact, from both ends
# --------------------------------------------------------------------------- #
_DASHBOARD = os.path.join(TOOLS_VAST, "dashboard")
_ARGV_LITERAL = "python3 tools/vast/herdd.py dash-cache --sections"


@pytest.mark.skipif(not os.path.isdir(_DASHBOARD), reason="dashboard tree absent")
def test_the_dashboard_spawns_exactly_the_frozen_argv() -> None:
    """Four spawn/copy sites outside this package bind the literal, and none of
    them is in a language this suite can type-check. The path is `herdd.py`
    itself, so the CLI wiring for this command may never move (plan §3, Zone E)
    and nothing may ever be appended to the argv."""
    src = open(os.path.join(_DASHBOARD, "lib", "vast-admin.ts")).read()
    frozen = re.search(r"DASH_CACHE_ARGV[^=]*=\s*Object\.freeze\(\[(.*?)\]\)",
                       src, re.S)
    assert frozen, "DASH_CACHE_ARGV is no longer a frozen array literal"
    parts = re.findall(r"'([^']+)'", frozen.group(1))
    assert parts == ["tools/vast/herdd.py", "dash-cache", "--sections"]
    assert "[...DASH_CACHE_ARGV, want.join(',')]" in src, \
        "the section list is no longer the only variable element"


@pytest.mark.skipif(not os.path.isdir(_DASHBOARD), reason="dashboard tree absent")
def test_the_dashboard_section_vocabulary_matches_DASH_SECTIONS() -> None:
    """The spawned `--sections` list is built from SECTION_TTL_MS' own keys. A
    name in one and not the other is a usage error (exit 1) at spawn time, or a
    panel that is never refreshed."""
    src = open(os.path.join(_DASHBOARD, "lib", "vast-admin.ts")).read()
    block = re.search(r"SECTION_TTL_MS\s*=\s*\{(.*?)\}\s*as const", src, re.S)
    assert block
    keys = re.findall(r"^\s*(\w+):", block.group(1), re.M)
    assert set(keys) == set(dashcache.DASH_SECTIONS)
    assert dashcache._dash_parse_sections(",".join(keys)) == dashcache.DASH_SECTIONS


@pytest.mark.skipif(not os.path.isdir(_DASHBOARD), reason="dashboard tree absent")
@pytest.mark.parametrize("rel", [
    os.path.join("components", "admin", "FleetPanel.tsx"),
    os.path.join("components", "admin", "shared.tsx"),
])
def test_the_ui_copy_quotes_the_same_command(rel: str) -> None:
    """Two more literal copies live in the UI's "run this yourself" copy. They
    are documentation, but they are documentation of a frozen path."""
    assert _ARGV_LITERAL in open(os.path.join(_DASHBOARD, rel)).read()
