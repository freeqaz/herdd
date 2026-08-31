"""`vastlib.workflows.matrix` — the ported jobmatrix, held to its four traps.

Why this file exists
--------------------
`test_jobmatrix.py` and the six submit-preflight tests in
`test_job_submit_preflight.py` drive the FLAT `jobmatrix.py`, which stays live
and unedited through the add-only phase (plan §8). They cannot see the ported
copy, and three of its properties are silent when broken:

1. **The repo root** (manifest H3). The flat `_repo_root()` counts three
   `dirname`s, correct only at `tools/vast/` depth; copied verbatim into
   `vastlib/workflows/` it yields `tools/vast` and NOTHING RAISES — the wrong
   root reaches `jobmeta.asset_preflight(repo_root=)` inside a bare
   `except Exception` that downgrades to "note: … skipped", so the staleness
   gate silently becomes a no-op and a stale-asset submit passes. Pinned
   against the flat module's own resolution, with the naive arithmetic proved
   wrong rather than asserted wrong.

2. **The cwd-debris footgun** (H2), and this is the one place the port is
   deliberately NOT behavior-identical: the flat copy anchors the bundle
   staging dir and the local manifest copy to `os.getcwd()`. These tests pin
   the new `_REPO_ROOT` anchoring AND assert the flat file still has its bug,
   so the delta stays a decision instead of becoming an accident.

3. **The three pre-spend gates** (H4). Each preflight's FINDINGS call sits in a
   bare `except Exception` that logs "skipped" and continues, while the refusal
   sits outside. So a port that breaks an import or a signature converts a
   refusal into a note nobody reads. Every gate here gets a POSITIVE CONTROL:
   its dependency is broken and the refusal is asserted to still fire. The VRAM
   gate is the exception — its handler `continue`s, i.e. a broken findings call
   really does skip that arm's refusal — and that is asserted as observed
   behavior rather than papered over.

4. **Class identity under the bare-name authoring contract** (H1) — the same
   hazard as `spec.py`'s, sharpened by the fact that the flat module carries
   the `sys.modules.setdefault` mitigation and this copy deliberately does not.

What is deliberately NOT here
-----------------------------
* No edit or repoint of any flat test. `test_b2_write_scope.py:248` reads
  `tools/vast/jobmatrix.py` BY PATH and asserts the preflight symbol names
  appear in its text; it stays green because step 5 is add-only, and it is the
  shim commit's job to repoint it (step 6/7).
* No network, no subprocess, no B2. Every `jobmeta` seam is patched through the
  MODULE ATTRIBUTE (`matrix.jobmeta.<name>`) — the same idiom
  `test_job_submit_preflight.py` uses, and the reason the port keeps
  `import jobmeta` rather than `from jobmeta import …`.
* No `Arm`/`Variant` redesign. `_job_name`/`_entrypoint` are set after
  construction and both dataclasses are mutable on purpose (H5); the tests
  pin that so a frozen/slots "cleanup" fails loudly instead of at runtime.

Provenance: created 2026-08-16 alongside `vastlib/workflows/matrix.py`, plan §8
step 5, manifest `workflows-core.json`.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import jobmatrix as flat_jobmatrix                     # noqa: E402  twin, still live

from vastlib.workflows import matrix                   # noqa: E402

_SHA = "ab" * 32


@pytest.fixture()
def bundle(tmp_path):
    """A minimal real bundle dir — `jobmeta.validate_job_config` checks that
    the entrypoint exists, so this is not stubbable away."""
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "run.sh").write_text("#!/bin/sh\n")
    return d


def _exp(**over):
    kw = dict(name="exp-t", entrypoint="run.sh",
              axes={"rank": {"r16": {"LORA_R": "16"}, "r64": {"LORA_R": "64"}}},
              env={"A": "1"}, results=["out/**"], needs={"gpu": True})
    kw.update(over)
    return matrix.Experiment(**kw)


def _stub_gates(monkeypatch, **over):
    """Neutral (passing) stubs for every pre-spend gate, patched through the
    module attribute so the patch actually steers the port."""
    stubs = {
        "b2_write_preflight": lambda cfg, d: [],
        "b2_write_scope_report": lambda wf, allow_unscoped=False: ([], False),
        "asset_preflight": lambda probe, **k: [],
        "asset_preflight_report": lambda f, strict=False, allow_stale=False: ([], False),
        "vram_gate_findings": lambda cfg: [],
        "vram_gate_report": lambda v, allow_drift=False: ([], False),
    }
    stubs.update(over)
    for name, fn in stubs.items():
        assert hasattr(matrix.jobmeta, name), name   # no vacuous patch (plan §7.3)
        monkeypatch.setattr(matrix.jobmeta, name, fn)


def _stub_bundle_write(monkeypatch, size=1024):
    def _write_bundle(src, tmp):
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(b"x" * size)
        return {"sha256": _SHA, "zst_size": size}
    monkeypatch.setattr(matrix.jobmeta, "write_bundle", _write_bundle)


def _code_text(path):
    """The module's CODE, with comments and string literals removed.

    The text assertions below ("no `sys.modules`", "no `os.getcwd()`") are
    about what the module DOES. Both files quote the very constructs they must
    not execute — the step-7 shim recipe and the cwd bug are spelled out in the
    docstrings and inline comments on purpose — so a naive substring test over
    the raw file is guaranteed to fail on its own documentation. Tokenizing and
    dropping COMMENT/STRING is the narrowest way to ask the real question."""
    import tokenize
    with open(path, "rb") as fh:
        toks = [t for t in tokenize.tokenize(fh.readline)
                if t.type not in (tokenize.COMMENT, tokenize.STRING)]
    return " ".join(t.string for t in toks)


# --------------------------------------------------------------------------- #
# H3 — the repo root, the one line of the port that is not verbatim
# --------------------------------------------------------------------------- #
def test_repo_root_is_the_checkout_root_not_the_package_dir():
    """Was `test_repo_root_matches_flat_jobmatrix`; the flat comparison went
    tautological at step 7 (`jobmatrix._repo_root IS matrix._repo_root` now), so
    it is pinned against the filesystem instead of against itself."""
    assert matrix._repo_root() == matrix._REPO_ROOT
    assert os.path.isdir(os.path.join(matrix._REPO_ROOT, "tools", "vast"))
    assert os.path.isdir(os.path.join(matrix._REPO_ROOT, "tools", "vast",
                                      "vastlib", "workflows"))
    # the package dir is three levels below the root — never equal to it
    assert matrix._REPO_ROOT != os.path.dirname(os.path.abspath(matrix.__file__))


def test_naive_file_arithmetic_here_would_be_wrong():
    """The trap, proved rather than described: the flat expression applied to
    THIS file's path yields tools/vast, and nothing about that raises."""
    naive = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(matrix.__file__))))
    assert naive == str(VAST_DIR)
    assert naive != matrix._REPO_ROOT


# --------------------------------------------------------------------------- #
# H2 — cwd debris: the deliberate behavioral delta
# --------------------------------------------------------------------------- #
class _StopAtBundle(Exception):
    def __init__(self, tmp):
        super().__init__(tmp)
        self.tmp = tmp


def test_staging_and_manifest_anchor_to_the_repo_not_the_cwd(monkeypatch, bundle,
                                                             tmp_path):
    """Run `submit` from an unrelated cwd and neither default path may point at
    it. The flat copy fails this by construction — see the next test."""
    _stub_gates(monkeypatch)
    made = []
    monkeypatch.setattr(matrix.os, "makedirs",
                        lambda p, exist_ok=False: made.append(p))
    monkeypatch.setattr(matrix.jobmeta, "write_bundle",
                        lambda src, tmp: (_ for _ in ()).throw(_StopAtBundle(tmp)))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(_StopAtBundle) as ei:
        matrix.submit(_exp(), str(bundle), "9", log=lambda s: None)

    want_dir = os.path.join(matrix._REPO_ROOT, "out", "jobs", "_bundles")
    assert ei.value.tmp == os.path.join(want_dir, "pending.tar.zst")
    assert made == [want_dir], "the staging dir must be created, not assumed"
    assert str(tmp_path) not in ei.value.tmp


def test_the_cwd_bug_is_gone_from_BOTH_copies():
    """Replaces `test_flat_copy_still_carries_the_cwd_bug` (step 7).

    That test pinned the ADD-ONLY phase: the flat `jobmatrix.py` kept anchoring
    the staging dir and the local manifest to `os.getcwd()` until the shim
    commit folded its body away. The shim landed — one of the two outcomes its
    own docstring named — so the assertion is now the opposite one: NEITHER
    copy may name `os.getcwd()` in executable code, because there is only one
    implementation left and it anchors to `_REPO_ROOT`. `matrix.py`'s
    cwd-delta note was updated in the same commit."""
    for path in (VAST_DIR / "jobmatrix.py",
                 VAST_DIR / "vastlib" / "workflows" / "matrix.py"):
        assert "os . getcwd" not in _code_text(path), path


def test_injected_staging_dir_wins_and_is_created(monkeypatch, bundle, tmp_path):
    """`staging_dir=` / `local_out=` stay the injection seam; a nonexistent one
    is created rather than failing deep inside jobmeta.write_bundle."""
    _stub_gates(monkeypatch)
    _stub_bundle_write(monkeypatch)
    staging = tmp_path / "nested" / "deep"
    man = matrix.submit(_exp(), str(bundle), "9", dry_run=True,
                        staging_dir=str(staging), log=lambda s: None)
    assert staging.is_dir()
    assert (staging / f"{_SHA}.tar.zst").is_file()
    assert man["bundle_sha256"] == _SHA


# --------------------------------------------------------------------------- #
# H4 — the three pre-spend gates, each with a positive control
# --------------------------------------------------------------------------- #
def _boom(*a, **k):
    raise RuntimeError("dependency broken by the test")


def test_b2_scope_refusal_survives_a_broken_findings_call(monkeypatch, bundle):
    """POSITIVE CONTROL: break `b2_write_preflight`, and the refusal computed
    from the (empty) findings must STILL fire. The swallow is only allowed to
    downgrade the finding, never the gate."""
    _stub_gates(monkeypatch, b2_write_preflight=_boom,
                b2_write_scope_report=lambda wf, allow_unscoped=False: (
                    ["unscoped write: b2:foo/"], not allow_unscoped))
    log = []
    with pytest.raises(matrix.MatrixError) as ei:
        matrix.submit(_exp(), str(bundle), "9", log=log.append)
    assert "writes a B2 prefix the box has no" in str(ei.value)
    assert any("note: B2 write-scope preflight skipped" in ln for ln in log)
    assert "unscoped write: b2:foo/" in log


def test_b2_scope_override_flag_still_reaches_the_report(monkeypatch, bundle):
    _stub_gates(monkeypatch, b2_write_preflight=_boom,
                b2_write_scope_report=lambda wf, allow_unscoped=False: (
                    [], not allow_unscoped))
    _stub_bundle_write(monkeypatch)
    # explicit staging_dir: the default anchors under the REPO's out/ tree and a
    # unit test has no business writing there (H2 covers the default path).
    matrix.submit(_exp(), str(bundle), "9", dry_run=True,
                  staging_dir=str(bundle / "_stage"),
                  allow_unscoped_writes=True, log=lambda s: None)


def test_asset_staleness_refusal_survives_a_broken_findings_call(monkeypatch,
                                                                 bundle):
    """POSITIVE CONTROL for the gate whose silent-disarm is H3's consequence:
    with `asset_preflight` broken, the report still decides and still refuses."""
    _stub_gates(monkeypatch, asset_preflight=_boom,
                asset_preflight_report=lambda f, strict=False, allow_stale=False: (
                    ["stale: b2:x/y"], not allow_stale))
    log = []
    exp = _exp(tracks={"corpora/x.jsonl": "tools/vast/herdd.py"})
    with pytest.raises(matrix.MatrixError) as ei:
        matrix.submit(exp, str(bundle), "9", log=log.append)
    assert "stale asset(s) on B2" in str(ei.value)
    assert any("note: asset staleness preflight skipped" in ln for ln in log)


def test_asset_preflight_receives_the_repo_root_not_tools_vast(monkeypatch,
                                                               bundle):
    """The H3 failure would be invisible without this: a wrong root does not
    raise, it just makes the staleness check compare against files that are not
    there. Assert the argument, not the outcome."""
    seen = {}

    def _spy(probe, repo_root=None, runner=None, bucket=None):
        seen["repo_root"] = repo_root
        seen["tracks"] = dict(probe.get("tracks") or {})
        return []

    _stub_gates(monkeypatch, asset_preflight=_spy)
    _stub_bundle_write(monkeypatch)
    exp = _exp(tracks={"corpora/x.jsonl": "tools/vast/herdd.py"})
    matrix.submit(exp, str(bundle), "9", dry_run=True,
                  staging_dir=str(bundle / "_stage"), log=lambda s: None)
    assert seen["repo_root"] == matrix._REPO_ROOT
    assert os.path.isfile(os.path.join(seen["repo_root"], seen["tracks"]["corpora/x.jsonl"]))


def test_asset_check_can_be_disabled_entirely(monkeypatch, bundle):
    _stub_gates(monkeypatch, asset_preflight=_boom, asset_preflight_report=_boom)
    _stub_bundle_write(monkeypatch)
    exp = _exp(tracks={"corpora/x.jsonl": "tools/vast/herdd.py"})
    matrix.submit(exp, str(bundle), "9", dry_run=True, check_assets=False,
                  staging_dir=str(bundle / "_stage"), log=lambda s: None)


def test_vram_refusal_fires_per_arm(monkeypatch, bundle):
    """The refusing half: a finding that refuses names the ARM, because a
    matrix's arms differ in exactly the dimensions that move the footprint."""
    _stub_gates(monkeypatch,
                vram_gate_findings=lambda cfg: [{"arm": cfg["name"]}],
                vram_gate_report=lambda v, allow_drift=False: (
                    ["measured peak 40G"], not allow_drift))
    log = []
    with pytest.raises(matrix.MatrixError) as ei:
        matrix.submit(_exp(), str(bundle), "9", log=log.append)
    assert "needs.gpu_ram_gb is below a" in str(ei.value)
    assert any(ln.startswith("[") and "measured peak 40G" in ln for ln in log)


def test_vram_gate_skips_the_arm_when_its_findings_call_breaks(monkeypatch,
                                                               bundle):
    """POSITIVE CONTROL, and the honest answer it gives: unlike the two gates
    above, this handler `continue`s — so a broken findings call skips that arm's
    refusal even though the report would have refused. Verbatim from the flat
    module; pinned here so the vacuousness is a KNOWN property and not a
    surprise the next reader has to rediscover."""
    _stub_gates(monkeypatch, vram_gate_findings=_boom,
                vram_gate_report=lambda v, allow_drift=False: (["x"], True))
    _stub_bundle_write(monkeypatch)
    log = []
    matrix.submit(_exp(), str(bundle), "9", dry_run=True,
                  staging_dir=str(bundle / "_stage"), log=log.append)
    notes = [ln for ln in log if ln.startswith("note: VRAM sizing check skipped")]
    assert len(notes) == 2                       # one per arm, and no refusal


def test_every_gate_is_reached_before_any_bundle_write(monkeypatch, bundle):
    """Ordering is the whole value of a pre-spend gate: all three run before
    `write_bundle`, so a refusal costs nothing and happens before a box exists."""
    order = []
    _stub_gates(monkeypatch,
                b2_write_preflight=lambda cfg, d: order.append("b2") or [],
                asset_preflight=lambda probe, **k: order.append("assets") or [],
                vram_gate_findings=lambda cfg: order.append("vram") or [])
    monkeypatch.setattr(matrix.jobmeta, "write_bundle",
                        lambda src, tmp: order.append("bundle") or
                        (_ for _ in ()).throw(_StopAtBundle(tmp)))
    exp = _exp(tracks={"corpora/x.jsonl": "tools/vast/herdd.py"})
    with pytest.raises(_StopAtBundle):
        matrix.submit(exp, str(bundle), "9", staging_dir=str(bundle / "_stage"),
                      log=lambda s: None)
    assert order == ["b2", "assets", "vram", "vram", "bundle"]


# --------------------------------------------------------------------------- #
# H7 — frozen constants and the laptop/box wire contract
# --------------------------------------------------------------------------- #
def test_frozen_constants_are_their_literals():
    """The `== flat_jobmatrix.X` half of this test went tautological at step 7
    (the flat name re-exports these very objects), so only the literals — the
    half that can still fail — are asserted."""
    assert matrix.MANIFEST_VERSION == 1
    assert matrix.RESERVED_ENV == ("EXP_ID", "ARM_ID")
    assert matrix.MATRIX_FILENAME == "matrix.py"


def test_manifest_and_ticket_carry_the_frozen_wire_keys(monkeypatch, bundle):
    """Both directions of the jobd seam: the block this writes
    ({exp_id, arm, axes}) and the keys `onstart/jobd.py` reads out of it. jobd
    is Zone S — it never imports this module, so the only thing binding the two
    is the key spelling."""
    _stub_gates(monkeypatch)
    _stub_bundle_write(monkeypatch)
    tickets = []
    monkeypatch.setattr(matrix.jobmeta, "bundle_exists", lambda sha, **k: True)
    monkeypatch.setattr(matrix.jobmeta, "make_ticket",
                        lambda jid, sha, actor, cfg, box: tickets.append(cfg) or
                        {"job_id": jid})
    monkeypatch.setattr(matrix.jobmeta, "write_ticket",
                        lambda t, **k: (True, "jobs/queue/9/x.json", None))
    monkeypatch.setattr(matrix.jobmeta, "emit_event", lambda *a, **k: None)

    rcats = []

    def _runner(args, input=None):
        rcats.append((args[0], input))
        return 0, "", ""

    man = matrix.submit(_exp(), str(bundle), "9", runner=_runner,
                        bucket="test-bucket", staging_dir=str(bundle / "_stage"),
                        local_out=str(bundle / "_out"), log=lambda s: None)

    assert man["v"] == matrix.MANIFEST_VERSION
    assert set(man) == {"v", "exp_id", "name", "bundle_sha256", "box",
                        "submitted_ts", "actor", "arms"}
    assert [a["arm"] for a in man["arms"]] == ["r16", "r64"]
    assert set(man["arms"][0]) == {"arm", "job_id", "axes", "env"}

    cfg = tickets[0]
    assert set(cfg["experiment"]) == {"exp_id", "arm", "axes"}
    assert cfg["experiment"]["exp_id"] == man["exp_id"]
    assert cfg["experiment"]["arm"] == "r16"
    assert cfg["experiment"]["axes"] == {"rank": "r16"}
    assert cfg["env"]["EXP_ID"] == man["exp_id"] and cfg["env"]["ARM_ID"] == "r16"

    assert [c[0] for c in rcats] == ["rcat"]     # the manifest, ONE object
    assert (bundle / "_out" / f"{man['exp_id']}.json").is_file()

    jobd = (VAST_DIR / "onstart" / "jobd.py").read_text()
    assert 'cfg.get("experiment")' in jobd
    assert "experiment.get('exp_id')" in jobd


def test_dry_run_touches_no_b2_mutation(monkeypatch, bundle):
    _stub_gates(monkeypatch)
    _stub_bundle_write(monkeypatch)
    for name in ("upload_bundle", "write_ticket", "emit_event", "bundle_exists"):
        monkeypatch.setattr(matrix.jobmeta, name, _boom)

    def _runner(args, input=None):
        raise AssertionError(f"dry-run reached the runner: {args}")

    log = []
    man = matrix.submit(_exp(), str(bundle), "9", dry_run=True, runner=_runner,
                        staging_dir=str(bundle / "_stage"), log=log.append)
    assert len(man["arms"]) == 2
    assert sum("[dry-run]" in ln for ln in log) == 2


# --------------------------------------------------------------------------- #
# H1 — class identity under the bare-name authoring contract
# --------------------------------------------------------------------------- #
_MATRIX_SRC = """\
from jobmatrix import Experiment, Variant

EXPERIMENT = Experiment(
    name="exp-t", entrypoint="run.sh",
    axes={"rank": {"r16": {"LORA_R": "16"},
                   "r64": Variant(env={"LORA_R": "64"}, timeout_s=99)}},
)
"""


def test_an_authored_matrix_loads_through_BOTH_names_as_ONE_class(bundle):
    """THE hazard, executable — now in its closed state (INVERTED at step 7).

    Before the shim this asserted the split was open: an authored matrix
    resolves `Experiment` off `sys.path` — i.e. from the flat module — so the
    ported loader's isinstance check failed with an error naming the wrong
    cause ("must define a module-level EXPERIMENT"). `tools/vast/jobmatrix.py`
    is now a pure re-export of `vastlib.workflows.matrix`, so the bare name and
    the package yield one `Experiment` class and the ported loader works on
    real authored matrices. The flat file must still EXIST — it is the frozen
    authoring contract and deleting it is not an option."""
    (bundle / "matrix.py").write_text(_MATRIX_SRC)
    assert (VAST_DIR / "jobmatrix.py").is_file()

    assert flat_jobmatrix.Experiment is matrix.Experiment
    for loader in (matrix.load_experiment, flat_jobmatrix.load_experiment):
        exp = loader(str(bundle))
        assert isinstance(exp, matrix.Experiment)
        assert isinstance(exp, flat_jobmatrix.Experiment)


def test_sys_modules_alias_closes_the_identity_split(bundle, monkeypatch):
    """The step-7 recipe, proven: with the bare name aliased to the ported
    module (what the flat re-export shim will do), the authored file and the
    loader share ONE Experiment class."""
    (bundle / "matrix.py").write_text(_MATRIX_SRC)
    monkeypatch.setitem(sys.modules, "jobmatrix", matrix)

    exp = matrix.load_experiment(str(bundle))
    assert isinstance(exp, matrix.Experiment)
    arms = matrix.expand(exp)
    assert [a.name for a in arms] == ["r16", "r64"]
    assert arms[1].timeout_s == 99                       # the Variant override


def test_importing_the_port_does_not_hijack_the_bare_name():
    """Zone P owns no global state. A `sys.modules.setdefault` here would hand
    authored matrices the package class while everything still calling the flat
    module checks its own — and `test_jobmatrix.py`'s `import jobmatrix` would
    quietly receive the wrong module."""
    assert sys.modules["jobmatrix"] is flat_jobmatrix
    assert sys.modules["jobmatrix"] is not matrix
    code = _code_text(VAST_DIR / "vastlib" / "workflows" / "matrix.py")
    assert "sys . modules" not in code
    assert "sys . path" not in code
    assert "from jobmeta import" not in code             # module-attribute form only
    assert "from disksize import" not in code


def test_zone_s_and_sibling_seams_are_module_objects():
    """The patch idiom the whole suite rests on (plan §8b): `jobmeta` and
    `disksize` are bound as MODULES, so `monkeypatch.setattr(matrix.jobmeta,
    …)` steers the port. A `from jobmeta import write_bundle` would make every
    such patch vacuous — green tests steering nothing."""
    import types
    assert isinstance(matrix.jobmeta, types.ModuleType)
    assert isinstance(matrix.disksize, types.ModuleType)
    assert matrix.jobmeta is flat_jobmatrix.jobmeta       # one Zone S module object


# --------------------------------------------------------------------------- #
# H5 — the hidden Arm attributes, and the deliberate mutability
# --------------------------------------------------------------------------- #
def test_arm_carries_undeclared_attributes_set_after_construction():
    """`_job_name`/`_entrypoint` are assigned by `_resolve`, not declared as
    fields. A frozen/slots "cleanup" of this dataclass is an AttributeError at
    runtime — it is a separate commit with its own test, not part of the move."""
    arms = matrix.expand(_exp())
    assert [a.job_name for a in arms] == ["exp-t-r16", "exp-t-r64"]
    assert "_job_name" not in matrix.Arm.__dataclass_fields__
    assert "_entrypoint" not in matrix.Arm.__dataclass_fields__
    assert arms[0].job_config()["entrypoint"] == "run.sh"

    bare = matrix.Arm(name="x", axes={}, env={}, timeout_s=1, needs={},
                      results=[], checkpoint_s=None, checkpoints=None)
    with pytest.raises(AttributeError):
        bare.job_name


def test_variant_and_arm_stay_mutable():
    """Matrix authors share and mutate `Variant`s; that is the DSL, not a
    defect."""
    v = matrix.Variant(env={"A": "1"})
    v.env["B"] = "2"
    v.timeout_s = 5
    arms = matrix.expand(_exp())
    arms[0].name = "renamed"
    assert arms[0].name == "renamed"


# --------------------------------------------------------------------------- #
# expansion semantics — determinism, the reserved keys, the hard errors
# --------------------------------------------------------------------------- #
def test_expansion_is_the_declaration_ordered_cross_product():
    exp = _exp(axes={"base": {"q3": {"B": "q3"}, "lfm": {"B": "lfm"}},
                     "rank": {"r16": {"R": "16"}, "r64": {"R": "64"}}})
    arms = matrix.expand(exp)
    assert [a.name for a in arms] == ["q3-r16", "q3-r64", "lfm-r16", "lfm-r64"]
    assert arms[0].axes == {"base": "q3", "rank": "r16"}
    assert arms[0].env["ARM_ID"] == "q3-r16"
    assert [a.name for a in matrix.expand(exp)] == [a.name for a in arms]


def test_later_axes_win_on_conflict_and_env_is_str_coerced():
    exp = _exp(env={"LORA_R": "8", "N": 1},
               axes={"a": {"x": {"LORA_R": "16"}},
                     "b": {"y": matrix.Variant(env={"LORA_R": "64"},
                                               needs={"gpu_ram_gb": 24},
                                               timeout_s=11)}})
    (arm,) = matrix.expand(exp)
    assert arm.env["LORA_R"] == "64"
    assert arm.env["N"] == "1"                    # str()-coerced, not int
    assert arm.needs == {"gpu": True, "gpu_ram_gb": 24}
    assert arm.timeout_s == 11


def test_reserved_env_is_refused_at_both_levels():
    with pytest.raises(matrix.MatrixError) as ei:
        matrix.expand(_exp(env={"EXP_ID": "mine"}))
    assert "reserved" in str(ei.value)
    with pytest.raises(matrix.MatrixError) as ei:
        matrix.expand(_exp(axes={"a": {"x": {"ARM_ID": "mine"}}}))
    assert "reserved" in str(ei.value)


def test_hard_errors_never_degrade_into_a_silent_skip():
    with pytest.raises(matrix.MatrixError):                       # not an Experiment
        matrix.expand("nope")
    with pytest.raises(matrix.MatrixError):                       # no axes
        matrix.expand(_exp(axes={}))
    with pytest.raises(matrix.MatrixError):                       # empty axis
        matrix.expand(_exp(axes={"a": {}}))
    with pytest.raises(matrix.MatrixError) as ei:                 # non-slug key
        matrix.expand(_exp(axes={"a": {"Not A Slug": {}}}))
    assert "must already be a slug" in str(ei.value)
    with pytest.raises(matrix.MatrixError) as ei:                 # >40 char job name
        matrix.expand(_exp(name="x" * 40, axes={"a": {"k": {}}}))
    assert "40-char slug" in str(ei.value)
    with pytest.raises(matrix.MatrixError) as ei:                 # everything excluded
        matrix.expand(_exp(exclude=lambda arm: True))
    assert "exclude() dropped every arm" in str(ei.value)
    with pytest.raises(matrix.MatrixError) as ei:                 # bad variant type
        matrix.expand(_exp(axes={"a": {"k": 7}}))
    assert "expected a dict of env vars or a" in str(ei.value)


def test_exclude_drops_only_the_named_arm():
    arms = matrix.expand(_exp(exclude=lambda arm: arm.axes == {"rank": "r64"}))
    assert [a.name for a in arms] == ["r16"]


def test_load_experiment_reports_a_missing_matrix_file(tmp_path):
    with pytest.raises(matrix.MatrixError) as ei:
        matrix.load_experiment(str(tmp_path))
    assert "no matrix.py in" in str(ei.value)


# --------------------------------------------------------------------------- #
# the $0 promise of `expand`
# --------------------------------------------------------------------------- #
def test_cmd_expand_prints_arms_and_asks_disksize_not_b2(monkeypatch, bundle):
    """`expand` promises $0 and no B2: no rclone, no byte sizes. The disk shape
    comes from `disksize.matrix_disk_shape` — called through the module
    attribute, so the sibling can be repointed at `vastlib` when it lands."""
    (bundle / "matrix.py").write_text(_MATRIX_SRC)
    monkeypatch.setitem(sys.modules, "jobmatrix", matrix)     # step-7 alias, see H1
    seen = {}

    def _shape(cfgs):
        seen["n"] = len(cfgs)
        return {"asset_dedup_saving": 1, "scratch_peak_gb": 12,
                "distinct_assets": ["a"], "arms": len(cfgs),
                "asset_entries_naive": 2, "scratch_concurrent": len(cfgs)}

    monkeypatch.setattr(matrix.disksize, "matrix_disk_shape", _shape)
    monkeypatch.setattr(matrix.jobmeta, "_default_runner", _boom)

    ns = type("NS", (), {"dir": str(bundle), "matrix": "matrix.py"})()
    buf = io.StringIO()
    with redirect_stdout(buf):
        matrix._cmd_expand(ns)
    out = buf.getvalue()
    assert seen["n"] == 2
    assert "experiment 'exp-t': 2 arms" in out
    assert "r16" in out and "r64" in out
    assert "scratch peaks at 12G" in out
