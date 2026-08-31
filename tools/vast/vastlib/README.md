**Hub:** [vast tooling](../README.md) › [fleet](../INDEX_FLEET.md) — the three-zone layout every file under tools/vast obeys

# `vastlib` — the vast tooling package (Zone P)

Plan of record: [`docs/plans/vast-tooling-refactor-v2.md`](../../../docs/plans/vast-tooling-refactor-v2.md) — read its §3 (three-zone layout), §5 (architecture) and §6 (typing and tooling) before
adding anything here. This file is the operational condensation of those
sections plus the one convention the later steps depend on: the `moved-from:`
marker.

---

## 1. The three zones

Everything under `tools/vast/` is in exactly one of three zones. Which zone a
file is in decides what it may import, whether it may use pydantic, and whether
it is allowed to move.

### Zone S — shipped flat leaves

```
tools/vast/{jobmeta,runmeta,bidpolicy,metrics_probe,gemm_probe,
            parse_vllm_mem,triton_cache}.py
tools/vast/onstart/*
```

These are copied **by basename** into every job bundle and executed on a rented
box, in a flat directory where the repo does not exist.

- Bare-name imports only. No package-relative imports, no `vastlib`.
- **stdlib only.** No pydantic — box-side pydantic is explicitly a later step
  (plan §0.3). `vastlib/requirements.txt` must never grow a Zone S consumer.
- `vastlib` **may** import them (bare name; `tools/vast` is on `sys.path` for
  every entry script). They may **never** import `vastlib`.
- No dual-form `try: from . import x / except ImportError: import x` modules.
  That ambiguity is precisely what the flat-bundle test exists to distrust.
- Enforced by `test_jobd_bundle_imports_flat.py` (dynamic: every leaf imports
  bare-name under `python3 -P` in a flat, repo-invisible dir) and by
  `shipcheck.py`'s `FLAT_NAMESPACES` (its static twin).
- On eventual pyproject promotion (plan §0.1) these **stay behind**.

`bidpolicy.py` and `runmeta.py` are the boundary case — shipped *and* imported
by the package. They stay Zone S flat files; `vastlib` imports them bare-name.

### Zone P — this package (`tools/vast/vastlib/`)

Workstation-only. Strict-typed, pydantic allowed at the API boundary.

- **No `sys.path` manipulation anywhere inside the package.** It is written
  installable-clean from day one so promotion to a real pyproject package is a
  move, not a rewrite.
- `vastlib.`-absolute or relative imports only, plus stdlib, plus pydantic,
  plus Zone S bare names.
- `from __future__ import annotations` in every file (enforced: ruff `I002`).
- Full annotations on public **and** private defs (enforced: ruff `ANN` + mypy
  strict on `vastlib.*`).
- Dependency DAG, strictly downward (enforced: import-linter):

  ```
  core  ->  {market, boxes, launch, storage}
        ->  {supervise, jobs, fleet, workflows}
        ->  cli
  ```

  `core` imports stdlib + Zone S only. Nothing imports `cli`. `fleet.daemon`
  and `cli.main` are the only composition roots.

### Zone E — entry scripts

```
tools/vast/herdd.py   -> sys.path bootstrap + vastlib.cli.main:main
tools/vast/fleetd.py    -> sys.path bootstrap + vastlib.fleet.daemon:main
```

Both keep their **exact current paths**, forever. The `herdd-reaper.timer`
systemd unit executes `tools/vast/herdd.py reap -y` every 15 minutes and
destroys boxes; `fleetd cmd_deploy` hard-requires `tools/vast/fleetd.py` to
exist as a script at that path; `wave_driver.py`, four dashboard TypeScript
spawn sites (frozen `dash-cache` argv), eight shell scripts, the `herdd`
skill and ~550 markdown references all bind the literal path. The command a
user types never changes. These two files are the *only* place a `sys.path`
bootstrap is allowed to exist.

**Do not edit Zone S or Zone E files during a port step** unless that step's
entry in plan §8 says so.

---

## 2. The `moved-from:` marker — grammar

Every symbol ported into `vastlib` carries a marker comment recording where it
came from. Plan §7.1 generates the `old attr -> new module.attr` rename table
from these markers, and that table is what mechanically rewrites 1,734
references and 659 `monkeypatch.setattr` sites across 44 test files. **A missing
marker is a symbol the test migration cannot find.** The table is committed and
doubles as the doc of where everything went.

### Rules

1. The marker is a **comment line directly above** the ported `def` / `class` /
   module-level assignment — no blank line between marker and definition. If the
   definition has decorators, the marker goes above the decorators.
2. Exactly one marker per line, one line per marker.
3. Grammar:

   ```
   # moved-from: <source-module>.<original_name>
   ```

   `<source-module>` is the flat module basename without `.py`;
   `<original_name>` is the identifier as it existed there.
4. `<source-module>` is one of the files being absorbed:

   ```
   # moved-from: herdd.request_soft
   # moved-from: herdd._job_handoff_emit
   # moved-from: fleetd.Hooks
   # moved-from: vastconf._boot_knob
   # moved-from: ladder_core.observe_bid
   # moved-from: workflowctl.cmd_workflow_run
   # moved-from: salvage.salvage_disk
   ```
5. **Renames** — when the ported name differs from the original, the marker
   still names the *original*, and the rename is stated after a `->`:

   ```
   # moved-from: herdd._num_dph -> Instance.dph
   ```

   Use this for every accessor that becomes a property or a method. The table
   generator needs the arrow to emit a non-identity mapping.
6. **Merges** — when one ported symbol absorbs several originals (the
   raising/soft pairs collapsing to one implementation, the guard verdict sets
   collapsing into `GuardVerdict`), emit one marker line per original, stacked:

   ```
   # moved-from: herdd._GUARD_ZOMBIE
   # moved-from: herdd._GUARD_ADVISORY
   # moved-from: fleetd._guard_short
   class GuardVerdict(enum.Enum):
   ```
7. **New code** carries no marker. If a symbol has no marker, the table
   generator treats it as new — which is a claim that it did not exist before,
   and reviewers should check it.
8. Markers are permanent. They are provenance, not scaffolding; do not strip
   them once the table is generated.

### What the marker does NOT record

Line numbers (they drift with every rebase and killed v1's citations) and
behavior notes. A port is behavior-preserving by rule (plan §7.4); if the port
*had* to change behavior, that is a found drift — stop and diagnose, do not
annotate it here.

---

## 3. The static lane

`tools/vast` has no CI, so the checks are wrapped in pytest:
`tools/vast/test_vastlib_static.py` runs all three tools and skips loudly (with
the exact `uv pip install` command) when one is missing.

| Tool | Config | Scope |
|---|---|---|
| `ruff check` | `vastlib/ruff.toml` | auto-discovered — ruff resolves config per file by walking up |
| `mypy` | `vastlib/mypy.ini` | explicit `--config-file`; run as `cd tools/vast && mypy -p vastlib` |
| `lint-imports` | `vastlib/importlinter.ini` | explicit `--config`; needs `PYTHONPATH=tools/vast` |

All three configs live **inside the package**, deliberately: nothing outside
`tools/vast/vastlib` changes behavior when someone runs ruff or mypy elsewhere
in the repo (there is no repo-wide config today, and this branch does not add
one). Each config file's header explains its own scoping choice.

Run them by hand:

```sh
.venv/bin/ruff check tools/vast/vastlib
( cd tools/vast && ../../.venv/bin/mypy --config-file vastlib/mypy.ini -p vastlib )
PYTHONPATH=tools/vast .venv/bin/lint-imports --config tools/vast/vastlib/importlinter.ini
pytest tools/vast/test_vastlib_static.py -q
```

Install the tools + the package dependency:

```sh
uv pip install --python .venv/bin/python -e ".[vast]"
uv pip install --python .venv/bin/python ruff mypy import-linter
```

### One trap, already sprung

Invoked the obvious way — `mypy tools/vast/vastlib` from the repo root — mypy
names the modules `tools.vast.vastlib.core`, so the `[mypy-vastlib,vastlib.*]`
strict block matches nothing and the entire strict configuration is **vacuous**.
Measured 2026-08-16: an untyped `def probe(x): return x` passed clean. Hence the
`-p vastlib` invocation, and hence
`test_vastlib_static.py::test_mypy_strictness_is_not_vacuous`, which
type-checks a deliberately untyped probe in a tmp copy of the package on every
run and fails if mypy does *not* complain. Green from a checker that is not
looking is the failure mode this whole lane exists to prevent.

---

## 4. Module docstring convention

Every module here carries the house header — the strongest existing quality
signal in `tools/vast` (see `vastconf.py` and `ladder_core.py`, which the
package generalizes):

1. **Why this exists** — the defect it kills or the boundary it draws.
2. **What is deliberately NOT here** — with the reason. This section is what
   stops the next person from re-adding the thing you removed.
3. **Provenance** — where the code came from and which plan step moved it.

Subpackage `__init__.py` files additionally list their planned modules, so the
skeleton documents its own shape before the code arrives.
