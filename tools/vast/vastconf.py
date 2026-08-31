#!/usr/bin/env python3
"""vastconf — DEPRECATION SHIM. The code now lives in `vastlib.core.config`.

Why this file still exists
--------------------------
Plan §3 of docs/plans/vast-tooling-refactor-v2.md: an absorbed flat sibling
becomes a re-export shim for one release before it is deleted. Four callers
still spell the bare name, and one of them is not Python:

  * `tools/vast/herdd.py` and `tools/vast/fleetd.py` — the thin launchers —
    `import vastconf` for the MODULE OBJECT, and herdd.py additionally
    re-exports an 18-name facade so `herdd.load_env` keeps resolving for
    hosts.py / boxstate.py / hostfacts.py / bid_echo_probe.py;
  * `tools/vast/local_smoke.py` reads `require_local_gpu`;
  * `tools/vast/parked_lifecycle.py` imports it function-locally, twice;
  * `tools/vast/launch_serve.sh` (lines 625-640) runs a python heredoc with
    `sys.path.insert(0, $SERVE_TOOLS_DIR)` — that is tools/vast, so `vastlib`
    resolves from here — and prints `vastconf.DISK_DEFAULT_SERVE_GB` to size the
    serve box's disk.

That last one is why this file exports all 32 names and not the launcher's 18.
`DISK_DEFAULT_SERVE_GB` is not in the facade, and the shell around the heredoc
has `2>/dev/null || echo 0` habits, so dropping it would not crash a launch — it
would quietly rent the WRONG DISK. Same shape for `require_local_gpu`,
`local_gpu_allowed`, `fleetd_adopt_default_budget_usd` (10 reads) and
`ADOPT_DEFAULT_BUDGET_USD`. The list below is exactly the 32 names this file
used to define; sizing it from herdd.py's facade is the mistake to avoid.

What is deliberately NOT here
-----------------------------
  * NO function or class body, and no re-derived constant. A shim that computes
    anything has forked the resolver, which is precisely the outcome the port
    exists to prevent — one `_boot_knob` precedence chain in the process, not
    two.
  * NO `sys.modules['vastconf'] = vastlib.core.config` alias. It is tempting
    (it would make the two names one object) and it is wrong here: nothing
    path-loads an authored file that imports this module, so there is no
    bare-name `isinstance` contract to close, and the alias would silently
    convert test_vastlib_core_config.py's whole DIFFERENTIAL section into
    self-comparisons AND change what `monkeypatch.setattr(vastconf, ...)`
    steers. If a patch should steer the port, that is a test edit, not a
    sys.modules trick.
  * NO `sys.path` bootstrap. vastconf.py never had a `__main__` path, so every
    importer already has tools/vast on the path (launch_serve.sh's heredoc puts
    it there itself).
  * `_HERE` is re-exported, not recomputed. `vastlib.core.config._HERE` climbs
    two extra directories to land on the SAME string this file's own
    `dirname(abspath(__file__))` produced — tools/vast — and that string is what
    `load_herdd_config()` resolves `herdd.yaml` against. Re-exporting keeps
    the one value pinned by test_vastlib_core_config.py; it is no longer this
    file's own dirname, which is harmless because nothing recomputes it.

Known consequence, recorded rather than fixed: `monkeypatch.setattr(vastconf,
'load_herdd_config', ...)` does not steer vastlib code, which reads
`config.load_herdd_config`. That was already true before this shim (plan step
6d moved the readers into the package); the shim does not change it. The pattern
to copy when a test needs both spellings steered is
test_vastlib_fleet_rows.py's autouse fixture, which patches BOTH and says why.

Provenance: bodies moved to `vastlib/core/config.py` at plan step 2 (the
add-only phase kept both live as twins); this file became a shim at plan step 7.
The design record — the `.env` walk, the herdd.yaml merge order, the runset
reserved-key lattice, and the CLI > env > yaml > constant convention — is
unchanged and lives in the ported module's docstring.
"""
from __future__ import annotations

from vastlib.core.config import (  # noqa: F401  (re-export surface, see __all__)
    _BOOT_KNOB_DEFAULTS,
    _HERE,
    _REPO_CONFIG,
    _RUNSET_ENV_KEY_RE,
    _RUNSET_ENV_RESERVED,
    _RUNSET_ENV_RESERVED_PREFIXES,
    _USER_CONFIG,
    ADOPT_DEFAULT_BUDGET_USD,
    DISK_DEFAULT_FLEETD_GB,
    DISK_DEFAULT_LAUNCH_GB,
    DISK_DEFAULT_SERVE_GB,
    DISK_DEFAULT_SUPERVISE_GB,
    DISK_DEFAULT_TRAIN_GB,
    DISK_DEFAULT_WORKFLOW_GB,
    FLEETD_ADOPT_BUDGET_ENV,
    FLEETD_ADOPT_BUDGET_KEY,
    JOBS_HANDOFF_UNSAFE_ENV,
    JOBS_HANDOFF_UNSAFE_KEY,
    LOCAL_GPU_ENV,
    LOCAL_GPU_KEY,
    _adopt_cfg,
    _boot_knob,
    _load_yaml_file,
    _parse_simple_yaml,
    _runset_env_defaults,
    default_disk_gb,
    fleetd_adopt_default_budget_usd,
    jobs_handoff_enabled,
    load_env,
    load_herdd_config,
    local_gpu_allowed,
    require_local_gpu,
)

#: The frozen shim surface: exactly the 32 top-level names tools/vast/vastconf.py
#: defined before the port — a SUPERSET of herdd.py's 18-name facade by the 14
#: the launcher never carried. test_vastlib_core_config.py reads this list, which
#: is what stops the two drifting apart silently.
__all__ = [
    "ADOPT_DEFAULT_BUDGET_USD",
    "DISK_DEFAULT_FLEETD_GB",
    "DISK_DEFAULT_LAUNCH_GB",
    "DISK_DEFAULT_SERVE_GB",
    "DISK_DEFAULT_SUPERVISE_GB",
    "DISK_DEFAULT_TRAIN_GB",
    "DISK_DEFAULT_WORKFLOW_GB",
    "FLEETD_ADOPT_BUDGET_ENV",
    "FLEETD_ADOPT_BUDGET_KEY",
    "JOBS_HANDOFF_UNSAFE_ENV",
    "JOBS_HANDOFF_UNSAFE_KEY",
    "LOCAL_GPU_ENV",
    "LOCAL_GPU_KEY",
    "_BOOT_KNOB_DEFAULTS",
    "_HERE",
    "_REPO_CONFIG",
    "_RUNSET_ENV_KEY_RE",
    "_RUNSET_ENV_RESERVED",
    "_RUNSET_ENV_RESERVED_PREFIXES",
    "_USER_CONFIG",
    "_adopt_cfg",
    "_boot_knob",
    "_load_yaml_file",
    "_parse_simple_yaml",
    "_runset_env_defaults",
    "default_disk_gb",
    "fleetd_adopt_default_budget_usd",
    "jobs_handoff_enabled",
    "load_env",
    "load_herdd_config",
    "local_gpu_allowed",
    "require_local_gpu",
]
