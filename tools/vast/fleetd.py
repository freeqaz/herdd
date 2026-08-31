#!/usr/bin/env python3
"""fleetd — Zone E entry script for the fleet-supervision daemon.

THIN LAUNCHER. Every line of behavior lives in `vastlib.fleet`; this file is a
sys.path bootstrap, a set of identity re-exports, and `main()`. The engine is
`vastlib.fleet.daemon` (tick, `Server`, argparse tree), persistence is
`vastlib.fleet.state`, the derived tables are `vastlib.fleet.rows`, the release
deploy is `vastlib.fleet.deploy`, the mutating seam is `vastlib.fleet.hooks`,
and the client half of the socket protocol is `vastlib.fleet.client`. Design of
record: `tools/vast/FLEETD_DESIGN.md` (process model + control surface) and
`SUPERVISE_DESIGN.md` (policy spec); the refactor that emptied this file is
`docs/plans/vast-tooling-refactor-v2.md` §3 (Zone E) / §8 step 6.

WHY THIS FILE STILL EXISTS, AT THIS EXACT PATH
----------------------------------------------
The INSTALLED systemd unit bakes an absolute `ExecStart={python} {script}
serve` at install time and is NOT rewritten by a merge: a running daemon keeps
executing the path it was installed with. Move or rename this file and the next
restart crash-loops on RestartSec=5 with the whole fleet unsupervised. The same
path is pinned by `vastlib.fleet.deploy` (`TOOLS_VAST_DIR`, `render_unit`,
`_fleetd_script`), by the reaper unit, and by ~550 doc references. It is a
frozen contract (plan §4), not an accident of history.

The four subcommands are frozen with it — `main` accepts exactly:
  serve         run the daemon (socket thread + reconcile thread).
                FLEETD_DRY_RUN=1 makes every mutating action a logged no-op.
  install-unit  generate the user unit AT RUNTIME (absolute paths never enter
                git), enable it, and enable-linger. Bootstrap/soak only.
  deploy        THE deploy path: move the RELEASE checkout to a known revision,
                re-point the unit, restart, and PROVE the live `rev=` is that
                revision.
  status        one-shot dump of the persisted state (no daemon needed).

THE sys.path BOOTSTRAP BELOW IS LOAD-BEARING — DO NOT DROP IT
-------------------------------------------------------------
systemd runs this script by ABSOLUTE path with `WorkingDirectory=<repo root>`,
so the cwd does NOT supply `tools/vast`. A bare script run gets `tools/vast` as
`sys.path[0]` for free, but every other invocation shape (`-m`, a console-script
shim, a wrapper, the release venv's interpreter) does not — and the flat Zone S
siblings (`vastconf`, `notify`, `bidpolicy`, `runmeta`, …) that `vastlib`
imports BARE-NAME would stop resolving. `vastlib.fleet.deploy`'s dependency
probe (`DEPLOY_IMPORT_PROBE`, run against the release interpreter before the
restart) is written against exactly this contract: it sets PYTHONPATH to the
checkout's `tools/vast` because that is "the same directory the entry script
inserts into sys.path".

RE-EXPORTS ARE PLAIN `from … import` BINDINGS, ON PURPOSE
----------------------------------------------------------
`inspect.getsource(fleetd.Fleet._tick_watch)` and friends must keep returning
the daemon's source, and a monkeypatch of a re-exported name must land on the
same object the tests compare by identity. So: no module-level `__getattr__`,
no lazy proxies, no wrapper functions, no subclasses. Each name below is the
identical object from its `vastlib.fleet` home. Four names had two candidate
homes in the port and are pinned here by ruling, not by import order:
`VERSION` → `client.FLEET_PROTO_VERSION` (the wire protocol version, not
`state.VERSION`'s schema version — they are different numbers that happen to be
equal), `STATE_NAME` / `JOURNAL_NAME` → `state` (the on-disk filenames), and
`dry_run_enabled` → `daemon`.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import vastconf                                          # noqa: E402,F401  (shared defaults)
import notify                                            # noqa: E402,F401  (pure leaf)

from vastlib.fleet.client import (                       # noqa: E402
    FLEET_PROTO_VERSION as VERSION,
    FLEET_UNIT_NAME as UNIT_NAME,
)
from vastlib.fleet.daemon import (                       # noqa: E402
    DESTROY_CONFIRM_OBS,
    DESTROY_CONFIRM_S,
    DESTROY_TTL_S,
    EXPENSIVE_DPH_USD,
    Fleet,
    GONE_CONFIRM_TICKS,
    HEALTH_EVERY_S,
    JOBS_POLICY_DEFAULTS,
    MAX_OBS_DT_S,
    MAX_PAUSE_S,
    PARKED_STATES,
    POLICY_PROFILES,
    PROFILES,
    PYHALF_CONFIRM_S,
    Server,
    TICK_JITTER_FRAC,
    TICK_JITTER_S,
    TICK_S,
    UNWATCHED_GRACE_EXPENSIVE_S,
    UNWATCHED_GRACE_S,
    cmd_install_unit,
    cmd_serve,
    cmd_status,
    dry_run_enabled,
    main,
    make_policy,
    notify_policy_enabled,
    pyhalf_broken,
    repo_root,
)
from vastlib.fleet.deploy import (                       # noqa: E402
    DEPLOY_BRANCH,
    DEPLOY_CHECKOUT_DEFAULT,
    DEPLOY_CHECKOUT_ENV,
    DEPLOY_LOCAL_REF,
    DEPLOY_PYTHON_DEFAULT,
    DEPLOY_REF_DEFAULT,
    _git,
    _resolve_deploy_target,
    _verify_live_rev,
    checkout_audit,
    cmd_deploy,
    deploy_checkout_path,
    deploy_ref_candidates,
    local_source_repo,
    prepare_deploy_ref,
    render_unit,
    resolve_deploy_ref,
)
from vastlib.fleet.hooks import Hooks                     # noqa: E402
from vastlib.fleet.rows import (                          # noqa: E402
    BOOT_EVIDENCE_S,
    EXEMPT_LABEL_TOKENS,
    JOBD_FRESH_S,
    RETENTION_NOTES,
    UNWATCHED_STALE_S,
    _num,
    _RETENTION_FATE,
    _retention_fate,
    _retention_status_map,
    ceiling_rows,
    handoff_predecessor,
    label_exempt,
    normalize_ceiling,
    reconcile_rows,
    recoveries_in_flight,
    retention_alarms,
    retention_rows,
    stray_rows,
    watch_box_iid,
    workload_evidence,
)
from vastlib.fleet.state import (                         # noqa: E402
    JOURNAL_MAX_BYTES,
    JOURNAL_NAME,
    LOCK_NAME,
    REPLACEMENT_STATE_KEYS,
    RUN_STATE_KEYS,
    STATE_NAME,
    _replacement_state_persist,
    _replacement_state_restore,
    _run_lane_state_persist,
    _run_lane_state_restore,
    acquire_single_instance_lock,
    iso,
)

# The re-export surface, spelled out so pyflakes reads these as exports rather
# than dead imports, and so `import fleetd; fleetd.<name>` keeps working for the
# ~30 test modules and the doc/runbook references that still name this module.
__all__ = [
    # vastlib.fleet.client — the socket protocol's frozen constants
    "UNIT_NAME", "VERSION",
    # vastlib.fleet.daemon — the engine, its tunables, and the entry points
    "DESTROY_CONFIRM_OBS", "DESTROY_CONFIRM_S", "DESTROY_TTL_S",
    "EXPENSIVE_DPH_USD", "Fleet", "GONE_CONFIRM_TICKS", "HEALTH_EVERY_S",
    "JOBS_POLICY_DEFAULTS",
    "MAX_OBS_DT_S", "MAX_PAUSE_S", "PARKED_STATES", "POLICY_PROFILES",
    "PROFILES", "PYHALF_CONFIRM_S", "Server", "TICK_JITTER_FRAC",
    "TICK_JITTER_S", "TICK_S",
    "UNWATCHED_GRACE_EXPENSIVE_S", "UNWATCHED_GRACE_S", "cmd_install_unit",
    "cmd_serve", "cmd_status", "dry_run_enabled", "main", "make_policy",
    "notify_policy_enabled", "pyhalf_broken", "repo_root",
    # vastlib.fleet.deploy — the release checkout / unit / verify path
    "DEPLOY_BRANCH", "DEPLOY_CHECKOUT_DEFAULT", "DEPLOY_CHECKOUT_ENV",
    "DEPLOY_LOCAL_REF", "DEPLOY_PYTHON_DEFAULT", "DEPLOY_REF_DEFAULT", "_git",
    "_resolve_deploy_target", "_verify_live_rev", "checkout_audit",
    "cmd_deploy", "deploy_checkout_path", "deploy_ref_candidates",
    "local_source_repo", "prepare_deploy_ref", "render_unit",
    "resolve_deploy_ref",
    # vastlib.fleet.hooks — the mutating seam
    "Hooks",
    # vastlib.fleet.rows — the derived tables (pure folds over state)
    "BOOT_EVIDENCE_S", "EXEMPT_LABEL_TOKENS", "JOBD_FRESH_S",
    "RETENTION_NOTES", "UNWATCHED_STALE_S", "_RETENTION_FATE", "_num",
    "_retention_fate", "_retention_status_map", "ceiling_rows",
    "handoff_predecessor", "label_exempt", "normalize_ceiling",
    "reconcile_rows", "recoveries_in_flight", "retention_alarms",
    "retention_rows", "stray_rows", "watch_box_iid", "workload_evidence",
    # vastlib.fleet.state — persistence, the journal, the single-instance lock
    "JOURNAL_MAX_BYTES", "JOURNAL_NAME", "LOCK_NAME", "REPLACEMENT_STATE_KEYS",
    "RUN_STATE_KEYS", "STATE_NAME", "_replacement_state_persist",
    "_replacement_state_restore", "_run_lane_state_persist",
    "_run_lane_state_restore", "acquire_single_instance_lock", "iso",
]


# --------------------------------------------------------------------------- #
# THE COMPOSITION STEP — this launcher is a composition root too
# --------------------------------------------------------------------------- #
# `vastlib/cli/_compose.py::bind()` points the cross-ring seams (`SEAM_BINDINGS`)
# at the modules that DEFINE them. Until it runs they raise on purpose, so a miss
# is loud instead of silently launching a broken box.
#
# It was called from four places, ALL in the `cli` ring — and the daemon does not
# go through `cli`. So inside the fleetd process every seam stayed raising, and
# the FIRST replacement launch after a pull condemnation died on it:
#
#   !! pull watchdog: replacement launch failed (unlaunchable: replacement launch
#      error: compose_jobs_launch_env: not ported yet — rebind this module
#      attribute ...)
#
#   fleet.daemon tick -> supervise.{run_lane,job_lane} -> replacement._relaunch
#     -> launch._do_launch -> compose_jobs_launch_env   (RAISE)
#
# — every tick, forever, with the condemned box never replaced. Same path reaches
# `fleet_watch_best_effort` (the successor would launch UNWATCHED) and, through
# `lifecycle._destroy_and_revoke`, `fleet_operator_intent`.
#
# WHY THIS IS LEGAL HERE AND NOWHERE BELOW. `fleet.daemon` cannot call `bind()`
# itself: `cli` is the TOP layer of `importlinter.ini`'s `vastlib-layers`
# contract and nothing inside `vastlib` may import it (the contract's own note:
# "NOTHING imports `cli`. It is the top layer and a composition root; the other
# composition root is fleet.daemon"). This file is Zone E — outside
# `root_package = vastlib` entirely — so it is the one place that can hold both
# halves, exactly as `herdd.py`'s `cli.main.main()` does for the CLI process.
#
# WHY IT IS NOT AN IMPORT-TIME BIND. `_compose.py`'s docstring is explicit:
# importing a module must not change what another module's attributes do, or
# `test_vastlib_launch.py`'s "these seams still raise" census passes or fails on
# pytest's collection ORDER (measured: green alone, four failures behind
# `test_vastlib_cli_main.py`). ~20 test modules do `import fleetd`. So the bind
# lives in `run()` — the RUN-time entry path — and `import fleetd` is still inert.
#
# `main` itself is untouched, and deliberately: the header rule above ("RE-EXPORTS
# ARE PLAIN `from … import` BINDINGS") means `fleetd.main` must stay the identical
# object as `vastlib.fleet.daemon.main`. Wrapping it would break that. `run()` is a
# NEW name that binds and then dispatches.
def _bind_cross_ring_seams() -> None:
    """Close `_compose.SEAM_BINDINGS` for this process.

    Imported inside the function, not at module scope, so `import fleetd` costs
    nothing extra and stays free of the `cli` ring — the bind must be a RUN-time
    event, and an import that only ever happens on the run path says so in one
    place. `bind()` is idempotent (it is four attribute assignments)."""
    from vastlib.cli import _compose            # Zone E may import cli; vastlib may not
    _compose.bind()


def run(argv: list[str] | None = None) -> int:
    """The launcher's entry path: compose, then dispatch.

    `ExecStart={python} {script} serve` (`fleet.deploy.render_unit`) runs this
    file as `__main__`, and `herdd fleet install|deploy` shells out to
    `fleetd.py install-unit|deploy` — every real invocation is a fresh process
    that arrives here, so this is the one place the daemon's composition can
    live."""
    _bind_cross_ring_seams()
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
