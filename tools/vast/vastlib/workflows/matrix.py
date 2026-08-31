"""`vastlib.workflows.matrix` — experiment matrices as Python (ported `jobmatrix.py`).

Why this exists
---------------
The authoring layer above `jobmeta.py` for multi-arm experiments (bakeoffs):
same data / different models, same model / different data, same model+data /
different method (LoRA rank, target layers, quant, …). A matrix is a plain
Python file in the job folder (`matrix.py`) that builds an `Experiment`; this
module expands it into N ordinary v1 jobs.

Why Python, not YAML: a matrix wants cross-products, exclusions, computed env,
and per-arm resource overrides — in YAML that becomes string templating; in
Python it's comprehensions and functions. YAML (`job-config.yaml`) remains the
authoring surface for SINGLE jobs; the frozen v1 wire schema is untouched —
every expanded arm ships as a canonical-JSON ticket exactly like a hand-written
job (design of record: `tools/vast/MATRIX_DESIGN.md`).

Key property: all arms share ONE content-addressed bundle (the code is the
same; only ticket config/env differs), so an N-arm matrix uploads one bundle
and writes N tickets. Each ticket carries a first-class `experiment` block
({exp_id, arm, axes}); jobd echoes exp_id/arm onto every lifecycle event it
emits, so any arm's event log audits back to its experiment even without the
manifest. Plain jobs (no block) are unaffected.

A minimal `matrix.py`:

    from jobmatrix import Experiment, Variant

    EXPERIMENT = Experiment(
        name="exp3-reader",
        entrypoint="run.sh",
        timeout_s=4 * 3600,
        env={"DATA_FILE": "reader_sft_train.jsonl", "LORA_R": "32"},
        results=["out/**"],
        checkpoint_s=300,
        needs={"gpu": True, "gpu_ram_gb": 48, "venv": "serve"},
        axes={
            "base": {
                "qwen3-8b": {"BASE_SLUG": "qwen3-8b"},
                "lfm":      Variant(env={"BASE_SLUG": "lfm25-1.2b-thinking"},
                                    needs={"gpu_ram_gb": 24}),
            },
            "rank": {
                "r16": {"LORA_R": "16"},
                "r64": {"LORA_R": "64"},
            },
        },
        exclude=lambda arm: arm.axes == {"base": "lfm", "rank": "r64"},
    )

THIS FILE OWNS THE CLASSES — and the bare name still resolves to them
---------------------------------------------------------------------
Since plan step 7, `tools/vast/jobmatrix.py` is a re-export shim over this
module (plus the frozen CLI entry point), so `jobmatrix.Experiment is
matrix.Experiment` and `load_experiment` here works on real authored matrices.
The hazard that shaped both files (manifest `workflows-core.json` H1) is worth
keeping in view, because it is sharper here than in `spec.py`: the flat module
carried the mitigation from the start.

* Every authored matrix says `from jobmatrix import Experiment, Variant` — a
  BARE-NAME import off `sys.path`. Five live in-repo examples (measured
  2026-08-16): `tools/witness/jobs/{phase1-cot-train,q6-round1-arms,
  repair-lifter-train,v7-longctx-train}/matrix.py` plus
  `repair-lifter-train/matrix_smoke.py`; more exist outside the repo.
* `load_experiment` executes that file with `runpy.run_path` and then checks
  `isinstance(exp, Experiment)` against **its own** class object.
* The shim does `sys.modules.setdefault("jobmatrix", sys.modules[__name__])`
  for exactly this reason, and it is LOAD-BEARING there rather than decorative:
  on the frozen CLI path the file runs as `__main__`, so the bare name is
  unregistered and the authored matrix's import would otherwise load the file a
  second time under a second `Experiment` class.

**That `setdefault` stays out of this copy, permanently.** Executed from inside
`vastlib` it would claim the bare name `jobmatrix` for the whole process at
import time, and `test_jobmatrix.py`'s `import jobmatrix` would silently receive
the package module instead of the file it means to test — the same two-class
break pointed the other way, plus a `sys.modules` side effect Zone P has no
business having. `test_vastlib_workflows_matrix.py` pins both halves: that an
authored matrix now loads through EITHER name as one class, and that importing
this module does not hijack the bare name.

The shim recipe, as landed at step 7 (do not improvise a replacement)
---------------------------------------------------------------------
`tools/vast/jobmatrix.py` is a re-export shim, not a deletion. The landed file
extends the sketch below with the six private names tests read through the flat
name (`_as_variant`, `_cmd_expand/status/submit`, `_repo_root`, `_resolve`), the
two module seams (`jobmeta`, `disksize`), and the `sys.path` bootstrap Zone E is
allowed to carry:

    # tools/vast/jobmatrix.py  (compatibility shim + frozen CLI path)
    import sys
    import vastlib.workflows.matrix as _m
    from vastlib.workflows.matrix import (        # noqa: F401  re-export
        MANIFEST_VERSION, MATRIX_FILENAME, RESERVED_ENV, Arm, Experiment,
        MatrixError, Variant, exp_status, expand, load_experiment,
        read_manifest, submit, validate_experiment, main,
    )
    # An authored matrix.py resolving the bare name must get the SAME module
    # object the loader used, or isinstance() sees two Experiment classes.
    sys.modules.setdefault("jobmatrix", sys.modules[__name__])
    sys.modules.setdefault("vastlib.workflows.matrix", _m)
    if __name__ == "__main__":                    # frozen CLI path, see below
        main()

Two things that recipe must keep, permanently:

1. **`python3 tools/vast/jobmatrix.py expand|submit|status` is a frozen surface**
   (`MATRIX_DESIGN.md`, the `herdd` skill at SKILL.md:42/:252, the help string
   `herdd.py:10781` prints, and three shell wrappers' comments). Hence the
   `__main__` guard lives in the shim; this package module deliberately has
   none — it cannot bootstrap `sys.path` for its own bare-name Zone S imports,
   and Zone E is the only place a bootstrap may live (plan §3).
2. **Plain deletion is UNSAFE — permanently.** The bare name is a frozen
   AUTHORING contract: every matrix file ever written contains
   `from jobmatrix import`, including ones on laptops no grep can reach, and
   including the generated fixture strings in `test_jobmatrix.py:129`,
   `test_launch_jobs_box.py:213` and `test_job_submit_preflight.py:687`. There
   is no deprecation window that finds them. The shim is the terminal state.
   (Separately: `test_jobd_bundle_imports_flat.py:115` picks the first existing
   of `vastconf.py`/`jobmatrix.py`/`hosts.py` as its "not in the bundle" canary,
   and §4 freezes that test unchanged — so at least one of the three must keep
   existing as a real file.)

The cwd-debris footgun this copy fixed (delta CLOSED at step 7)
--------------------------------------------------------------
The flat `jobmatrix.py` anchored the bundle staging dir and the local manifest
copy to `os.getcwd()` (`:405`, `:462`), so `submit` run from any subdirectory or
worktree scattered a fresh `out/jobs/_bundles` + `out/experiments` tree wherever
the operator happened to stand — while the log line at `:466` printed
`out/experiments/<exp>.json` as if it were repo-relative, i.e. the message was
wrong exactly when the debris was worst. Worse, `:405` never `makedirs`'d the
staging dir, so a nonexistent one failed deep inside `jobmeta.write_bundle` with
a path nobody chose. **This copy anchors both to `_REPO_ROOT` and creates the
staging dir** (manifest H2). That was a DELTA between two live copies until step
7; the flat file is now a re-export shim over this module, so the bug is gone
from the only implementation there is and
`test_vastlib_workflows_matrix.py::test_the_cwd_bug_is_gone_from_BOTH_copies`
holds both files to it. The `staging_dir=` / `local_out=` kwargs remain the
injection seam tests use.

What is deliberately NOT here
-----------------------------
* No `sys.path.insert` (flat `:65`). Zone P forbids it (plan §3); `jobmeta` is
  Zone S and legitimately imported bare-name, and `disksize` is an absorbed
  sibling still living flat — imported bare-name transitionally, to be repointed
  at `vastlib` when its own port lands.
* No `sys.modules` aliasing — see above; it belongs in the flat shim.
* No sandbox around `runpy.run_path`. Executing an authored matrix is
  ARBITRARY CODE EXECUTION by design, laptop-side only, and it is *why* the
  identity problem exists at all. A future hardening pass must not quietly swap
  in a data-only parser: that would break the DSL (`exclude=lambda`, computed
  env, comprehensions), which is the entire reason the format is Python.
* No frozen/slots rewrite of `Variant`/`Arm`. Both are mutable on purpose
  (matrix authors share and mutate `Variant`s), and `_resolve` sets
  `_job_name`/`_entrypoint` on an `Arm` after construction — a slots or frozen
  conversion is a runtime `AttributeError`, and it is a separate commit with its
  own test (manifest H5), not a drive-by of the move.

Provenance: verbatim port of `tools/vast/jobmatrix.py` @ `ea8360dc`, plan §8
step 5, manifest `workflows-core.json`. Changes are exactly: the three items
above, the `_REPO_ROOT` anchoring, and the annotations Zone P strict typing
requires (plan §6). No preflight, message, default or wire key changed.
"""
from __future__ import annotations

import fnmatch
import json
import os
import runpy
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import disksize

import jobmeta

if TYPE_CHECKING:                      # `main` imports argparse lazily, verbatim
    import argparse

# The repo root, five levels up from `tools/vast/vastlib/workflows/matrix.py`.
#
# THE FLAT COPY COUNTS THREE (`jobmatrix._repo_root`, "same 3x-dirname
# convention as herdd._repo_root"), which is correct only at `tools/vast/`
# depth. Copied verbatim into this file the expression yields `tools/vast`, and
# NOTHING RAISES: the wrong root is fed to `jobmeta.asset_preflight(repo_root=)`,
# whose caller wraps it in a bare `except Exception` that downgrades to
# "note: asset staleness preflight skipped" — i.e. a silently disarmed pre-spend
# gate and a stale-asset submit that passes (manifest H3). Hoisted to a module
# constant for the same reason `boxes.ssh._REPO_ROOT` and `core.config._HERE`
# are: the depth is a property of the module's location, not of the function.
#
# `test_vastlib_workflows_matrix.py::test_repo_root_matches_flat_jobmatrix`
# pins it against the flat module's own resolution.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# moved-from: jobmatrix.MATRIX_FILENAME
MATRIX_FILENAME = "matrix.py"
# FROZEN WIRE CONSTANT: written as the `v` key of the B2 experiment manifest,
# which has no schema negotiation.
# moved-from: jobmatrix.MANIFEST_VERSION
MANIFEST_VERSION = 1
# env keys the expander owns; a matrix that sets them is a hard error.
# moved-from: jobmatrix.RESERVED_ENV
RESERVED_ENV = ("EXP_ID", "ARM_ID")


# moved-from: jobmatrix.MatrixError
class MatrixError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# DSL surface
# --------------------------------------------------------------------------- #
# moved-from: jobmatrix.Variant
@dataclass
class Variant:
    """One value of one axis. `env` is the common case (a bare dict in an axis
    is shorthand for Variant(env=...)); the other fields override the
    experiment-level job config for every arm containing this variant —
    e.g. a bigger base bumps needs.gpu_ram_gb, a slower method bumps
    timeout_s. Overrides from later-declared axes win on conflict."""
    env: dict[str, Any] = field(default_factory=dict)
    timeout_s: int | None = None
    needs: dict[str, Any] | None = None
    results: list[Any] | None = None
    checkpoint_s: int | None = None
    checkpoints: list[Any] | None = None


# moved-from: jobmatrix.Arm
@dataclass
class Arm:
    """A resolved point of the matrix (output of expand(); do not construct by
    hand). `name` is the deterministic slug (`<key>-<key>-…` in axis order);
    the JOB name is `<experiment.name>-<name>`."""
    name: str
    axes: dict[str, Any]            # axis name -> variant key
    env: dict[str, Any]             # fully merged env (base -> variants in axis order)
    timeout_s: int
    needs: dict[str, Any]
    results: list[Any]
    checkpoint_s: int | None
    checkpoints: list[Any] | None
    tracks: dict[str, Any] | None = None

    @property
    def job_name(self) -> str:  # set by expand()
        # UNDECLARED ON PURPOSE (manifest H5): `_resolve` assigns `_job_name`
        # and `_entrypoint` after construction. Declaring them as fields — or
        # converting this dataclass to frozen/slots — is a separate commit with
        # its own test, not part of the move.
        return self._job_name  # type: ignore[attr-defined,no-any-return]

    def job_config(self) -> dict[str, Any]:
        """The raw v1 job-config dict for this arm (validate with
        jobmeta.validate_job_config before shipping)."""
        cfg: dict[str, Any] = {
            "version": 1,
            "name": self.job_name,
            "entrypoint": self._entrypoint,  # type: ignore[attr-defined]  # see job_name
            "timeout_s": self.timeout_s,
            "env": dict(self.env),
            "results": list(self.results),
            "needs": dict(self.needs),
        }
        if self.checkpoint_s is not None:
            cfg["checkpoint_s"] = self.checkpoint_s
        if self.checkpoints is not None:
            cfg["checkpoints"] = list(self.checkpoints)
        if self.tracks:
            cfg["tracks"] = dict(self.tracks)
        return cfg


# moved-from: jobmatrix.Experiment
@dataclass
class Experiment:
    """The matrix: a base job config + named axes of named variants. Expansion
    is the cross product of the axes, in declaration order (dicts are
    ordered), minus `exclude`d arms. Everything is plain Python — build axes
    with comprehensions, share variant dicts, compute env values."""
    name: str
    entrypoint: str
    axes: dict[str, Any]                     # axis name -> {variant key -> dict | Variant}
    env: dict[str, Any] = field(default_factory=dict)
    # Default read at CLASS-DEFINITION time — an import-order coupling to the
    # Zone S `jobmeta` module, preserved verbatim.
    timeout_s: int = jobmeta.DEFAULT_TIMEOUT_S
    results: list[Any] = field(default_factory=list)
    checkpoint_s: int | None = None
    checkpoints: list[Any] | None = None
    needs: dict[str, Any] = field(default_factory=dict)
    exclude: Any = None            # callable(Arm) -> bool (True = drop the arm)
    # PROVENANCE for the B2 objects this experiment reads: {full B2 key ->
    # repo-relative source file}. The DSL has no `assets:` field, so a matrix
    # entrypoint pulls what it needs from B2 itself — which is precisely the
    # path no staleness check could see before. Declaring it here makes
    # `jobmatrix submit` refuse when the staged object no longer matches the
    # repo file it mirrors. Absent => the preflight is a no-op (jobmeta:
    # `_normalize_tracks`).
    tracks: dict[str, Any] | None = None


# moved-from: jobmatrix._as_variant
def _as_variant(v: Any, axis: object, key: object) -> Variant:  # noqa: ANN401 — DSL input
    if isinstance(v, Variant):
        return v
    if isinstance(v, dict):
        return Variant(env=v)
    raise MatrixError(
        f"axis {axis!r} variant {key!r}: expected a dict of env vars or a "
        f"Variant (got {type(v).__name__})")


# moved-from: jobmatrix.expand
def expand(exp: Experiment) -> list[Arm]:
    """Deterministically expand the matrix into arms. Raises MatrixError on an
    unusable name (slug rules / >40 chars / collision) or reserved env —
    never silently truncates or skips."""
    if not isinstance(exp, Experiment):
        raise MatrixError(f"expected an Experiment (got {type(exp).__name__})")
    exp_slug = jobmeta.slugify(exp.name)
    if not exp.axes:
        raise MatrixError("experiment has no axes — use a plain job-config.yaml job")
    for bad in RESERVED_ENV:
        if bad in exp.env:
            raise MatrixError(f"env key {bad!r} is reserved (set by the expander)")

    axis_names = list(exp.axes)
    variants: list[list[tuple[str, Variant]]] = []
    for ax in axis_names:
        vs = exp.axes[ax]
        if not isinstance(vs, dict) or not vs:
            raise MatrixError(f"axis {ax!r} must be a non-empty dict of variants")
        row = []
        for key, v in vs.items():
            var = _as_variant(v, ax, key)
            if jobmeta.slugify(str(key)) != str(key):
                raise MatrixError(
                    f"axis {ax!r} variant key {key!r} must already be a slug "
                    f"(lowercase alnum + dashes) — it becomes part of the JOB_ID")
            for bad in RESERVED_ENV:
                if bad in var.env:
                    raise MatrixError(
                        f"axis {ax!r} variant {key!r}: env key {bad!r} is reserved")
            row.append((str(key), var))
        variants.append(row)

    arms: list[Arm] = []
    seen: set[str] = set()

    def _walk(i: int, keys: list[str], vars_: list[Variant]) -> None:
        if i == len(variants):
            arms.append(_resolve(exp, exp_slug, axis_names, keys, vars_, seen))
            return
        for key, var in variants[i]:
            _walk(i + 1, keys + [key], vars_ + [var])

    _walk(0, [], [])

    if exp.exclude is not None:
        arms = [a for a in arms if not exp.exclude(a)]
        if not arms:
            raise MatrixError("exclude() dropped every arm")
    return arms


# moved-from: jobmatrix._resolve
def _resolve(exp: Experiment, exp_slug: str, axis_names: list[str], keys: list[str],
             vars_: list[Variant], seen: set[str]) -> Arm:
    name = "-".join(keys)
    job_name = f"{exp_slug}-{name}"
    if len(job_name) > 40:
        raise MatrixError(
            f"arm {name!r}: job name {job_name!r} exceeds the 40-char slug "
            f"limit — shorten the experiment name or variant keys")
    jobmeta.slugify(job_name)          # raises if unusable
    if name in seen:
        raise MatrixError(f"duplicate arm name {name!r}")
    seen.add(name)

    env = {str(k): str(v) for k, v in exp.env.items()}
    timeout_s, needs = exp.timeout_s, dict(exp.needs)
    results = list(exp.results)
    checkpoint_s, checkpoints = exp.checkpoint_s, exp.checkpoints
    for var in vars_:                  # axis order; later axes win on conflict
        env.update({str(k): str(v) for k, v in var.env.items()})
        if var.timeout_s is not None:
            timeout_s = var.timeout_s
        if var.needs is not None:
            needs.update(var.needs)
        if var.results is not None:
            results = list(var.results)
        if var.checkpoint_s is not None:
            checkpoint_s = var.checkpoint_s
        if var.checkpoints is not None:
            checkpoints = list(var.checkpoints)
    env["ARM_ID"] = name

    arm = Arm(name=name, axes=dict(zip(axis_names, keys)), env=env,
              timeout_s=timeout_s, needs=needs, results=results,
              checkpoint_s=checkpoint_s,
              checkpoints=list(checkpoints) if checkpoints is not None else None,
              tracks=dict(exp.tracks) if exp.tracks else None)
    arm._job_name = job_name           # type: ignore[attr-defined]  # see Arm.job_name
    arm._entrypoint = exp.entrypoint   # type: ignore[attr-defined]  # see Arm.job_name
    return arm


# --------------------------------------------------------------------------- #
# loading + validation
# --------------------------------------------------------------------------- #
# moved-from: jobmatrix.load_experiment
def load_experiment(bundle_dir: str, matrix_file: str = MATRIX_FILENAME) -> Experiment:
    """Execute `<bundle_dir>/matrix.py` and return its module-level EXPERIMENT.
    The matrix file ships inside the bundle (it's just content); it is only
    ever executed on the laptop.

    ARBITRARY CODE EXECUTION by design (see the module header), and the
    `isinstance` check below is the one the bare-name authoring contract binds:
    an authored file's `from jobmatrix import Experiment` resolves the flat
    `tools/vast/jobmatrix.py`, which since step 7 re-exports THIS `Experiment`.
    Give that file a class of its own again and every authored matrix fails
    here with the misleading "must define a module-level EXPERIMENT"."""
    path = os.path.join(bundle_dir, matrix_file)
    if not os.path.isfile(path):
        raise MatrixError(f"no {matrix_file} in {bundle_dir}")
    ns = runpy.run_path(path)
    exp = ns.get("EXPERIMENT")
    if not isinstance(exp, Experiment):
        raise MatrixError(
            f"{path} must define a module-level EXPERIMENT = Experiment(...)")
    return exp


# moved-from: jobmatrix.validate_experiment
def validate_experiment(exp: Experiment, bundle_dir: str) -> list[tuple[Arm, Any, Any]]:
    """Expand + run every arm's config through the v1 validator (entrypoint
    existence, timeout bounds, glob safety, needs shape). Fails fast BEFORE
    any upload. Returns [(arm, canonical_cfg, warnings)]."""
    out = []
    for arm in expand(exp):
        cfg, warns = jobmeta.validate_job_config(arm.job_config(), bundle_dir)
        out.append((arm, cfg, warns))
    return out


# --------------------------------------------------------------------------- #
# submit: ONE bundle, N tickets, an experiment manifest
# --------------------------------------------------------------------------- #
# moved-from: jobmatrix._repo_root
def _repo_root() -> str:
    """The repo root. Kept as a function for the flat module's call shape; the
    arithmetic lives in `_REPO_ROOT` because the depth is a property of this
    file's location (the flat copy's 3x-dirname is wrong from here — see the
    constant's comment)."""
    return _REPO_ROOT


# moved-from: jobmatrix.submit
def submit(exp: Experiment, bundle_dir: str, box: object, *,
           runner: Any = jobmeta._default_runner,   # noqa: ANN401 — Zone S seam, tests inject
           bucket: str | None = None, actor: str | None = None,
           only: str | None = None, dry_run: bool = False,
           staging_dir: str | None = None, local_out: str | None = None,
           check_assets: bool = True, strict_assets: bool = False,
           allow_stale_assets: bool = False,
           allow_unscoped_writes: bool = False, allow_vram_drift: bool = False,
           repo_root: str | None = None,
           log: Callable[[str], Any] = print) -> dict[str, Any]:
    """Expand, validate, bundle once, then submit every (matching) arm as an
    ordinary v1 job. Returns the experiment manifest. `only` is an fnmatch
    glob on arm names. --dry-run does everything except B2 mutations."""
    box = str(box)
    validated = validate_experiment(exp, bundle_dir)
    if only:
        validated = [t for t in validated if fnmatch.fnmatch(t[0].name, only)]
        if not validated:
            raise MatrixError(f"--only {only!r} matched no arms")
    for _, _, warns in validated:
        for w in warns:
            log(f"warn: {w}")

    # B2 write-scope preflight — SAME seam as `herdd job submit`
    # (jobmeta.b2_write_preflight). Pure, $0, no network, so it runs first: the
    # arms share one bundle, so the scan is one pass over the bundle's text, and
    # the declared `scope.write` is unioned across arms. A matrix submit is how
    # a paired training run reaches a box, and a paired run is exactly where an
    # unentitled publish costs two fully-trained arms (v7, 2026-08-05).
    _scope_write = []
    for _arm, cfg, _w in validated:
        for p in ((cfg.get("scope") or {}).get("write") or []):
            if p not in _scope_write:
                _scope_write.append(p)
    try:
        _wf = jobmeta.b2_write_preflight(
            {"scope": {"write": _scope_write}} if _scope_write else {}, bundle_dir)
    except Exception as e:                    # never crash a submit on the check
        log(f"note: B2 write-scope preflight skipped ({e})")
        _wf = []
    # OUTSIDE the try, deliberately: the refusal must still fire when the
    # findings computation above was swallowed (manifest H4 — a broken import or
    # signature drift must not turn a refusal into a note nobody reads).
    _lines, _refuse = jobmeta.b2_write_scope_report(  # type: ignore[no-untyped-call]  # Zone S jobmeta is untyped
        _wf, allow_unscoped=allow_unscoped_writes)
    for ln in _lines:
        log(ln)
    if _refuse:
        raise MatrixError(
            "refusing to submit — this bundle writes a B2 prefix the box has no "
            "key for (see above): route it through the granted remote, or pass "
            "--allow-unscoped-writes for a single-key box")

    # Staged-asset staleness preflight — SAME seam as `herdd job submit`
    # (jobmeta.asset_preflight), run BEFORE bundling so a refusal costs nothing
    # and, critically, before a box is rented. Arms share one bundle and one
    # `tracks:` map, so the union over arms is checked once. A `tracks:`
    # mismatch refuses even on --dry-run: a dry run whose whole job is to prove
    # the submit is sound must not report sound when it is not.
    if check_assets:
        tracks: dict[str, Any] = {}
        assets: dict[str, Any] = {}
        for _arm, cfg, _w in validated:
            tracks.update(cfg.get("tracks") or {})
            for a in (cfg.get("assets") or []):     # future-proof: the DSL has
                assets[a["name"]] = a               # no assets field TODAY
        probe = {"tracks": tracks, "assets": list(assets.values())}
        if tracks or assets:
            try:
                findings = jobmeta.asset_preflight(  # type: ignore[no-untyped-call]  # Zone S jobmeta is untyped
                    probe, repo_root=repo_root or _repo_root(),
                    runner=runner, bucket=bucket)
            except Exception as e:            # never crash a submit on the check
                log(f"note: asset staleness preflight skipped ({e})")
                findings = []
            lines, refuse = jobmeta.asset_preflight_report(  # type: ignore[no-untyped-call]  # Zone S jobmeta is untyped
                findings, strict=strict_assets, allow_stale=allow_stale_assets)
            for ln in lines:
                log(ln)
            if refuse:
                raise MatrixError(
                    "refusing to submit — stale asset(s) on B2 (re-stage command "
                    "printed above), or override with --allow-stale-assets / "
                    "--no-asset-check")

    # needs.gpu_ram_gb vs measured VRAM, PER ARM — a matrix's arms differ in
    # exactly the dimensions that move the footprint (base, window, quant), so
    # checking the bundle once would check the wrong thing. Same rule as
    # `herdd job submit`: refuse only on a floor below an already-measured
    # peak (tools/vast/VRAM_SIZING.md).
    for _arm, _cfg, _w in validated:
        try:
            _v = jobmeta.vram_gate_findings(_cfg)
        except Exception as e:
            # NOTE the `continue`: unlike the two gates above, a swallowed
            # findings call here skips this arm's refusal entirely. Verbatim
            # from the flat module; the test file pins it as observed behavior
            # rather than pretending the gate survives its own failure.
            log(f"note: VRAM sizing check skipped for {_arm} ({e})")
            continue
        lines, refuse = jobmeta.vram_gate_report(  # type: ignore[no-untyped-call]
            _v, allow_drift=allow_vram_drift)
        for ln in lines:
            log(f"[{_arm}] {ln}")
        if refuse:
            raise MatrixError(
                f"refusing to submit — arm {_arm}'s needs.gpu_ram_gb is below a "
                f"peak this shape has already measured (see above), or override "
                f"with --allow-vram-drift")

    # _REPO_ROOT, NOT os.getcwd(): the flat copy stages bundles and the manifest
    # copy under the PROCESS cwd, so a submit from a subdirectory or a worktree
    # scatters a stray `out/` tree there while the log line below still claims
    # `out/experiments/…` (manifest H2). `makedirs` because the flat copy has
    # none and a missing dir surfaces as a confusing failure inside
    # `jobmeta.write_bundle`.
    staging = staging_dir or os.path.join(_REPO_ROOT, "out", "jobs", "_bundles")
    os.makedirs(staging, exist_ok=True)
    tmp = os.path.join(staging, "pending.tar.zst")
    info = jobmeta.write_bundle(bundle_dir, tmp)
    sha = info["sha256"]
    blob = os.path.join(staging, f"{sha}.tar.zst")
    os.replace(tmp, blob)
    if info["zst_size"] > jobmeta.BUNDLE_WARN_BYTES:
        log(f"warn: bundle is {info['zst_size'] / (1 << 20):.0f} MiB — stage "
            f"large inputs separately (stage_run.sh pattern)")

    exp_id = jobmeta.mint_job_id(exp.name)
    actor = actor or jobmeta._default_actor()  # type: ignore[no-untyped-call]  # Zone S jobmeta is untyped
    log(f">> experiment {exp_id}: {len(validated)} arms, bundle {sha[:12]}… "
        f"({info['zst_size']} B) -> box {box}")

    exists = False if dry_run else jobmeta.bundle_exists(sha, runner=runner, bucket=bucket)
    if not dry_run and not exists:
        ok, err = jobmeta.upload_bundle(  # type: ignore[no-untyped-call]
            blob, sha, runner=runner, bucket=bucket)
        if not ok:
            raise MatrixError(f"bundle upload failed: {err}")
    log(f">> bundle: {'dedupe HIT' if exists else ('would upload' if dry_run else 'uploaded')} "
        f"(ONE object for all arms)")

    manifest = {"v": MANIFEST_VERSION, "exp_id": exp_id, "name": exp.name,
                "bundle_sha256": sha, "box": box,
                "submitted_ts": jobmeta.now_ts(), "actor": actor, "arms": []}
    for arm, cfg, _ in validated:
        cfg = dict(cfg)
        cfg["env"] = dict(cfg["env"], EXP_ID=exp_id)
        # first-class association (audit seam): jobd `prepare` reads this block
        # and echoes exp_id/arm onto every lifecycle event it emits, so any
        # arm's event log traces to its experiment without the manifest.
        # WIRE CONTRACT across the laptop/box boundary (onstart/jobd.py:79,:138)
        # — these three key names are frozen.
        cfg["experiment"] = {"exp_id": exp_id, "arm": arm.name, "axes": arm.axes}
        job_id = jobmeta.mint_job_id(arm.job_name)
        manifest["arms"].append(
            {"arm": arm.name, "job_id": job_id, "axes": arm.axes, "env": cfg["env"]})
        if dry_run:
            log(f">> [dry-run] {arm.name}: would write ticket "
                f"jobs/queue/{box}/{job_id}.json")
            continue
        ticket = jobmeta.make_ticket(job_id, sha, actor, cfg, box)
        ok, key, err = jobmeta.write_ticket(  # type: ignore[no-untyped-call]
            ticket, runner=runner, bucket=bucket)
        if not ok:
            raise MatrixError(f"arm {arm.name}: ticket write failed: {err}")
        jobmeta.emit_event(job_id, "submitted", actor=actor, runner=runner,
                           bucket=bucket, bundle_sha256=sha, name=cfg["name"],
                           entrypoint=cfg["entrypoint"],
                           timeout_s=cfg["timeout_s"], box=box, exp_id=exp_id,
                           arm=arm.name)
        log(f">> {arm.name}: JOB_ID={job_id}")

    if not dry_run:
        body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        key = f"experiments/{exp_id}/manifest.json"
        rc, _, err = runner(
            ["rcat", jobmeta._q(bucket, key)], input=body)  # type: ignore[no-untyped-call]
        if rc != 0:
            raise MatrixError(f"manifest write failed: {err}")
        local = local_out or os.path.join(_REPO_ROOT, "out", "experiments")
        os.makedirs(local, exist_ok=True)
        with open(os.path.join(local, f"{exp_id}.json"), "w") as f:
            f.write(body)
        log(f">> manifest: {key} (+ out/experiments/{exp_id}.json)")
        log(f">> follow  : python3 tools/vast/jobmatrix.py status {exp_id}")
    return manifest


# moved-from: jobmatrix.read_manifest
def read_manifest(exp_id: str, *,
                  runner: Any = jobmeta._default_runner,  # noqa: ANN401 — Zone S seam
                  bucket: str | None = None) -> dict[str, Any]:
    jobmeta.validate_job_id(exp_id)
    key = f"experiments/{exp_id}/manifest.json"          # hoisted to fit the ignore
    rc, out, err = runner(["cat", jobmeta._q(bucket, key)])  # type: ignore[no-untyped-call]
    if rc != 0:
        raise MatrixError(f"no manifest for {exp_id}: {err}")
    return json.loads(out)  # type: ignore[no-any-return]  # json.loads -> Any


# moved-from: jobmatrix.exp_status
def exp_status(exp_id: str, *,
               runner: Any = jobmeta._default_runner,  # noqa: ANN401 — Zone S seam
               bucket: str | None = None,
               live_iids: Iterable[str] = ()) -> dict[str, Any]:
    """Manifest + per-arm folded job status. Without `live_iids` the fold's
    raw `status` is reported; `interrupted` derivation needs the vast API
    (`herdd job status <JOB_ID>` per arm)."""
    man = read_manifest(exp_id, runner=runner, bucket=bucket)
    rows = []
    for a in man["arms"]:
        view = jobmeta.read_job(a["job_id"], runner=runner, bucket=bucket,
                                live_iids=live_iids)
        rows.append({"arm": a["arm"], "job_id": a["job_id"], "axes": a["axes"],
                     "status": view["status"],
                     "display_status": view["display_status"] if live_iids else view["status"],
                     "rc": view["rc"], "fail_reason": view["fail_reason"],
                     "last_event_ts": view["last_event_ts"]})
    return {"exp_id": exp_id, "name": man["name"], "box": man["box"],
            "bundle_sha256": man["bundle_sha256"], "arms": rows}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
# moved-from: jobmatrix._cmd_expand
def _cmd_expand(a: argparse.Namespace) -> None:
    exp = load_experiment(os.path.abspath(a.dir), a.matrix)
    rows = validate_experiment(exp, os.path.abspath(a.dir))
    axis_names = list(exp.axes)
    print(f"experiment {exp.name!r}: {len(rows)} arms "
          f"(axes: {', '.join(f'{ax}[{len(exp.axes[ax])}]' for ax in axis_names)})")
    for arm, cfg, warns in rows:
        deltas = {k: v for k, v in cfg["env"].items()
                  if k != "ARM_ID" and exp.env.get(k) != v}
        print(f"  {arm.name:<28} axes={arm.axes} timeout={cfg['timeout_s']}s "
              f"needs={cfg['needs']} env~{deltas}")
        for w in warns:
            print(f"    warn: {w}")

    # Structural disk shape (velvet P4b). Stays inside `expand`'s $0/no-B2
    # promise: no byte sizes, no rclone — just the two rules people get
    # backwards, in opposite directions. Assets DEDUPE across arms (one shared
    # name-keyed cache), scratch does NOT (arms run concurrently, each building
    # its own tree). Deduping scratch under-allocates, which kills a job on a
    # rented box; summing assets over-allocates, which is what this plan exists
    # to stop.
    shape = disksize.matrix_disk_shape(  # type: ignore[no-untyped-call]  # unported sibling
        [cfg for _arm, cfg, _w in rows])
    if shape["asset_dedup_saving"] or shape["scratch_peak_gb"]:
        print(f"disk shape: {shape['distinct_assets'] and len(shape['distinct_assets'])} "
              f"distinct asset(s) across {shape['arms']} arms "
              f"({shape['asset_entries_naive']} entries — staged ONCE, shared cache)")
        if shape["scratch_peak_gb"]:
            print(f"            scratch peaks at {shape['scratch_peak_gb']:g}G "
                  f"({shape['scratch_concurrent']} concurrent arms x declared "
                  f"needs.scratch_gb — NOT deduped, arms run at the same time)")
        print("            byte sizes + a recommended --disk: "
              "`herdd job submit` prints them per arm")


# moved-from: jobmatrix._cmd_submit
def _cmd_submit(a: argparse.Namespace) -> None:
    exp = load_experiment(os.path.abspath(a.dir), a.matrix)
    submit(exp, os.path.abspath(a.dir), a.box, only=a.only, dry_run=a.dry_run,
           check_assets=not a.no_asset_check, strict_assets=a.strict_assets,
           allow_stale_assets=a.allow_stale_assets,
           allow_unscoped_writes=getattr(a, "allow_unscoped_writes", False),
           allow_vram_drift=getattr(a, "allow_vram_drift", False))


# moved-from: jobmatrix._cmd_status
def _cmd_status(a: argparse.Namespace) -> None:
    st = exp_status(a.exp_id)
    print(f"experiment {st['exp_id']} ({st['name']}) box={st['box']}")
    for r in st["arms"]:
        extra = f" rc={r['rc']}" if r["rc"] is not None else ""
        extra += f" reason={r['fail_reason']}" if r["fail_reason"] else ""
        print(f"  {r['arm']:<28} {r['status']:<10}{extra}  {r['job_id']}")


# moved-from: jobmatrix.main
def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="jobmatrix — expand a Python experiment matrix into v1 jobs",
        epilog="docs: tools/vast/MATRIX_DESIGN.md, tools/vast/JOBS_DESIGN.md")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("expand", help="expand + validate, print arms ($0, no B2)")
    p.add_argument("dir")
    p.add_argument("--matrix", default=MATRIX_FILENAME)
    p.set_defaults(func=_cmd_expand)

    p = sub.add_parser("submit", help="one bundle + one ticket per arm")
    p.add_argument("dir")
    p.add_argument("--box", required=True)
    p.add_argument("--matrix", default=MATRIX_FILENAME)
    p.add_argument("--only", help="fnmatch glob on arm names")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict-assets", dest="strict_assets", action="store_true",
                   help="turn the HEURISTIC (runset sentinel) B2-staleness warning "
                        "into a refusal too. A `tracks:`-declared mismatch always "
                        "refuses, with or without this flag")
    p.add_argument("--allow-stale-assets", dest="allow_stale_assets",
                   action="store_true",
                   help="submit anyway when a `tracks:`-declared staged object "
                        "differs from the repo file it mirrors (run the B2 bytes "
                        "on purpose)")
    p.add_argument("--allow-vram-drift", dest="allow_vram_drift",
                   action="store_true",
                   help="submit anyway when an arm's needs.gpu_ram_gb is below a "
                        "peak that shape has already MEASURED "
                        "(tools/vast/VRAM_SIZING.md)")
    p.add_argument("--no-asset-check", dest="no_asset_check", action="store_true",
                   help="skip the B2-staleness preflight entirely")
    p.add_argument("--allow-unscoped-writes", dest="allow_unscoped_writes",
                   action="store_true",
                   help="submit anyway when the bundle writes a B2 prefix no "
                        "shipped box key is scoped to (single-key box only)")
    p.set_defaults(func=_cmd_submit)

    p = sub.add_parser("status", help="manifest + per-arm folded status")
    p.add_argument("exp_id")
    p.set_defaults(func=_cmd_status)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except (MatrixError, jobmeta.JobmetaError) as e:
        sys.exit(f"error: {e}")
