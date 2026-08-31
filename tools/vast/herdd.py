#!/usr/bin/env python3
"""herdd — Zone E entry script for the vast.ai control CLI.

THIN LAUNCHER. Every line of behavior lives in `vastlib`; this file is a
`sys.path` bootstrap, a re-export surface, and `main()`. The parser tree and
the command dispatch are `vastlib.cli` (`cli/main.py` is the composition root,
and the only place the three cross-ring seams get bound); the rings beneath it
are `core` (HTTP, config, payload models, labels, formatting), `market`,
`boxes`, `launch`, `storage`, `jobs`, `supervise`, `fleet` and `workflows`.
Design of record: `docs/plans/vast-tooling-refactor-v2.md` §3 (the three zones)
and §8 step 6; the package's own map is `tools/vast/vastlib/README.md`.
Runbooks unchanged: `tools/vast/README.md`, `TRAINING.md`, `EVALS_RUNBOOK.md`,
`JOBS_DESIGN.md`, `SPOT_DESIGN.md`.

WHY THIS FILE STILL EXISTS, AT THIS EXACT PATH
----------------------------------------------
`tools/vast/herdd.py` is a frozen contract (plan §4). The systemd reaper unit
bakes an absolute `ExecStart=… tools/vast/herdd.py reap -y` at install time
and is not rewritten by a merge; `vastlib.fleet.deploy` renders the same tree
into the fleetd unit; `wave_driver`, the dashboard's argv, the `herdd` skill
and ~550 doc references all name it. Move or rename this file and the reaper
starts failing silently every 15 minutes. A console-script alias is NOT a
substitute — the argv in those units is a literal path.

THE sys.path BOOTSTRAP BELOW IS LOAD-BEARING — DO NOT DROP IT
-------------------------------------------------------------
`vastlib` imports the Zone S flat leaves (`vastconf`, `jobmeta`, `runmeta`,
`bidpolicy`, `imageref`, `notify`, `salvage`, …) BY BARE NAME, because those
files also ship inside the jobd bundle, where there is no package to be part of
(`test_jobd_bundle_imports_flat.py` enforces it). A bare script run gets
`tools/vast` as `sys.path[0]` for free; systemd's `WorkingDirectory=<repo root>`
plus an absolute script path does not, and neither does any wrapper, venv shim
or `-m` shape. So the insert is unconditional and first, exactly as the fat file
did it.

THE RE-EXPORT SURFACE, AND WHY IT IS NOT DECORATION
---------------------------------------------------
`herdd.<name>` is a live external contract for five in-repo consumers that
still address the flat module — `boxstate.py`, `hosts.py`, `hostfacts.py`,
`bid_echo_probe.py` and `parked_lifecycle.py` (`workflowctl.py` was absorbed at
step 7: the shim consults no herdd name, so its 27 attributes left this
surface) — plus `launch_serve.sh`, whose python heredoc does `import herdd`
for `request` / `_serve_boot_sla_condemn` / `_serve_self_park_soft`.
`shipcheck.py` now loads `vastlib/jobs/bundle.py` directly and falls back to
loading THIS FILE only on a pre-package checkout; `_job_attach_files` stays
re-exported here because the doc corpus (JOBS_CONFIG.md et al.) addresses it as
`herdd._job_attach_files`. The suite reaches ~230 more.

Four rules govern what may appear below:

1. **Identity bindings only.** Plain `from … import name`, no module-level
   `__getattr__`, no lazy proxies, no wrapper functions. `inspect.getsource`,
   `is`-comparisons and `monkeypatch.setattr(<owner>, …)` all have to land on
   the same object, and PEP-562 cannot serve the early-bound test aliases
   (`X = herdd.y`, evaluated at test-module import) that a call-site grep
   never sees.
2. **A RE-EXPORT IS NOT A PATCH POINT.** `monkeypatch.setattr(herdd, "…", …)`
   rebinds only this module's namespace. Once a body lives in
   `vastlib.core.api` and its callers resolve `api.request_soft` from THEIR
   globals, that patch is vacuous and the test goes green against live code —
   silently, which is why the conftest fleet/market guards exist. Patch the
   OWNING vastlib module. The one exception is rule 3.
3. **Module objects steer; names do not.** `jobmeta`, `runmeta`, `vastconf`,
   `bidpolicy`, `imageref`, `salvage` and the stdlib modules below are
   re-exported as MODULE OBJECTS, so `monkeypatch.setattr(herdd.jobmeta,
   "read_job", …)` mutates the one shared module object every `vastlib` caller
   resolves through — provided those callers keep the module-attribute call
   form (`jobmeta.read_job(...)`, never `from jobmeta import read_job`).
4. **Nothing new is invented here.** Every name below exists in a `vastlib`
   module or a Zone S leaf. In particular the `vastconf` facade is exactly the
   18 names the fat file re-exported; the other 14 `vastconf` top-level names
   (`default_disk_gb`, `require_local_gpu`, `_adopt_cfg`, …) were never
   `herdd` attributes and must not become them.

Names DELIBERATELY ABSENT, whose absence is load-bearing:

* `add_search_filters` — the flat search-parser builder.
  `test_vastlib_cli_surface.py` detects the post-thinning state by the absence
  of the flat parser plumbing; re-export it and that test captures the vastlib
  parser and compares it to itself. Its body lives in `vastlib.cli.search`.
* `_put_label_soft` — a dead twin that died with the fat body. The live label
  writer is `vastlib.core.labels`.
* `_b2_write_soft` — never defined, anywhere, in any revision. One test patches
  it with `raising=False`; creating it here would convert a detectable latent
  defect into a silent one.
* the `cmd_*` handlers and the `add_*_parser` builders other than the two the
  suite still diffs — they are `vastlib.cli` internals and `main()` is the
  supported way to reach them.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Zone S leaves and the stdlib modules the suite steers THROUGH this namespace
# (rule 3). Imported for their module objects, not for a name inside them.
import bidpolicy                                          # noqa: E402,F401
import imageref                                           # noqa: E402,F401
import jobmeta                                            # noqa: E402,F401
import random                                             # noqa: E402,F401
import runmeta                                            # noqa: E402,F401
import salvage                                            # noqa: E402,F401
import shutil                                             # noqa: E402,F401
import socket                                             # noqa: E402,F401
import subprocess                                         # noqa: E402,F401
import time                                               # noqa: E402,F401
import urllib.request                                     # noqa: E402,F401
import vastconf                                           # noqa: E402,F401

# The vastconf facade — EXACTLY the 18 names the fat file re-exported (rule 4).
# `load_env` in particular is reached as `herdd.load_env` (never
# `vastconf.load_env`) by hosts.py, boxstate.py, hostfacts.py and
# bid_echo_probe.py; that attribute has to stay alive.
from vastconf import (                                    # noqa: E402,F401
    DISK_DEFAULT_LAUNCH_GB,
    DISK_DEFAULT_SUPERVISE_GB,
    DISK_DEFAULT_TRAIN_GB,
    JOBS_HANDOFF_UNSAFE_ENV,
    JOBS_HANDOFF_UNSAFE_KEY,
    _BOOT_KNOB_DEFAULTS,
    _REPO_CONFIG,
    _RUNSET_ENV_KEY_RE,
    _RUNSET_ENV_RESERVED,
    _RUNSET_ENV_RESERVED_PREFIXES,
    _USER_CONFIG,
    _boot_knob,
    _load_yaml_file,
    _parse_simple_yaml,
    _runset_env_defaults,
    jobs_handoff_enabled,
    load_env,
    load_herdd_config,
)
# The PURE bid-defense and handoff decision cores. Kept in THIS namespace on
# purpose: bidpolicy is Zone S (it ships in the bundle) and was never ported,
# and the supervise suite EARLY-BINDS fifteen of these at test-module import
# (`_bid_action = herdd._bid_action`), which no later setattr can serve.
from bidpolicy import (                                   # noqa: E402,F401
    BID_CEILING_ONDEMAND_FRAC,
    BID_DECAY_POLLS,
    BID_FALLBACK_DPH_MULT,
    BID_MAX_MULT,
    BID_MIN_CUSHION_MULT,
    BID_MIN_STEP,
    BID_ONDEMAND_EPS,
    BID_RATE_LIMIT_S,
    BID_TARGET_MULT,
    BID_TARGET_MULT_UNPRICED,
    BID_TARGET_ONDEMAND_FRAC,
    DEFEND_AT,
    EVICTION_HOST_FAILURE,
    EVICTION_ONDEMAND,
    EVICTION_OUTBID,
    EVICTION_UNKNOWN,
    HANDOFF_CKPT_FRESH_MULT,
    HANDOFF_COOLDOWN_S,
    HANDOFF_DEADLINE_S,
    HANDOFF_DWELL_POLLS,
    HANDOFF_FENCE_HOLD_ETA_S,
    HANDOFF_FENCE_TIMEOUT_S,
    HANDOFF_FENCE_UNWIND_S,
    HANDOFF_MAX,
    HANDOFF_PARK_BID,
    HANDOFF_TTL_MARGIN_S,
    HANDOFF_TTL_S,
    HANDOFF_WARN_PCT,
    HANDOFF_WINDOW_H,
    LIVE_STATES,
    MAX_REPLACEMENTS,
    NOT_LIVE_DEBOUNCE,
    REPLACE_CEILING_MULT,
    REPLACEMENT_RETENTION_H,
    SPOT_FASTDEATH_S,
    Action,
    HandoffAction,
    Replacement,
    Retention,
    _HANDOFF_PRE_CUTOVER,
    _actor_is_cli,
    _bid_action,
    _bid_target,
    _decay_candidate,
    _default_max_bid,
    _evict_reason,
    _guardrail_exceeded,
    _handoff_arm_refusal,
    _handoff_candidate_ok,
    _handoff_candidate_target,
    _handoff_fence_hold,
    _handoff_headroom_ok,
    _handoff_trigger,
    _next_decay_streak,
    _preferred_ceiling,
    _preferred_ceiling_alarm,
    _refresh_default_ceiling,
    _spend_time_exceeded,
    _underbid_parked,
    bid_can_win,
    classify_eviction,
    handoff_poll,
    mk_handoff_state,
    mk_poll_state,
    poll,
    replacement_decision,
    retention_plan,
)
# Image-ref parsing + registry digest resolution (the STALE-IMAGE signal's
# source of truth). Both cache dicts are re-exported BY IDENTITY, so
# `herdd._digest_cache.clear()` still clears the dict imageref reads. The
# resolvers reach their in-block callees through imageref's own globals — steer
# those with `monkeypatch.setattr(imageref, …)`, never through here.
from imageref import (                                    # noqa: E402,F401
    IMAGE_DIGEST_ENV,
    _digest_cache,
    _ref_digest_cache,
    _skopeo_digest,
    _split_image,
    image_ref_digest,
    image_tag_digest,
)

# --------------------------------------------------------------------------- #
# The vastlib re-export surface, one block per home module (rule 1: identity
# bindings). Grouped and ordered by module so a reader can see, at a glance,
# where the body actually lives — that module, not this file, is the patch
# target (rule 2).
# --------------------------------------------------------------------------- #
from vastlib.boxes.health import (                        # noqa: E402,F401
    BootThroughputSampler, BoxHealth, GUARD_BOOTING, GUARD_LOADING_SLOW,
    GUARD_STALE_IMAGE, GUARD_ZOMBIE_LOADING_STALL, GUARD_ZOMBIE_NO_JOBD,
    GUARD_ZOMBIE_PYHALF, GuardVerdict, _get_instance_soft, _iso_ftz_to_epoch,
    _to_pull_bytes, boot_health_watch, build_throughput_observer,
    classify_box_health, gather_fleet_health, jobd_status_pyhalf,
    parse_pull_progress,
)
from vastlib.boxes.lifecycle import (                     # noqa: E402,F401
    _destroy_soft, _instances_soft, _launch_preflight, _put_state_soft,
    _revoke_box_keys, destroy_box, find_matching_instance, launch_instance,
    set_bid, stop_box,
)
from vastlib.boxes.reap import (                          # noqa: E402,F401
    REAP_IDLE_H_DEFAULT, _guard_evidence_bits, _guard_fix_plan,
)
from vastlib.boxes.ssh import (                           # noqa: E402,F401
    SSH_STRICTMODES_HINT, pub_key_text, ssh_access_warning,
    ssh_authorized_keys_snippet,
)
from vastlib.core.api import (                            # noqa: E402,F401
    API, _api_key_soft, _classify_http, request, request_soft,
)
from vastlib.core.fmt import (                            # noqa: E402,F401
    _ANSI_RE, _Pal, _Progress, _color_on, _fmt_age, _ls_cols, _ts_age_s,
    _ts_to_epoch, dollars, fmt_offer,
)
from vastlib.core.labels import (                         # noqa: E402,F401
    HANDOFF_LABEL_SUFFIX, _KEEP_UNTIL_FMT, _KEEP_UNTIL_RE, _job_handoff_label,
    _keep_retention_info, _keep_until_ts, _reap_kept, retention_keep_label,
)
from vastlib.core.models import (                         # noqa: E402,F401
    MarketRead, SSH_INJECT_MARKER, _JOB_PRIMARY_SHAPE_KEYS, _dash_verified,
    _disk_frac, _disk_gb, _gpu_ram_gb, _instance_env, _instance_image,
    _instance_run_label, _instance_serve_label, _instance_standing_bid,
    _job_primary_inst, _job_primary_shape, _label_value, _num_dph, _rates,
    _storage_day, effective_cores, instance_has_ssh_inject, instance_ssh_install,
)
from vastlib.fleet.client import (                        # noqa: E402,F401
    FLEET_PROTO_VERSION, FLEET_SOCK_TIMEOUT_S, FLEET_UNIT_NAME,
    _dash_offer_query, _dash_write_fleet, _fleet_delegation_disabled,
    _fleet_policy, _fleet_requester, _supervise_argv, fleet_follow,
    fleet_journal_path, fleet_recoveries_in_flight, fleet_request,
    fleet_sock_path, fleet_state_dir, fleet_state_path,
    fleet_watch_best_effort, fleet_watch_supervision,
)
from vastlib.fleet.deploy import _fleetd_script           # noqa: E402,F401
from vastlib.jobs.bundle import (                         # noqa: E402,F401
    _job_attach_files, _jobd_boot_snippet, _jobd_import_gate,
    _stage_jobd_bootstrap, _sync_file_list, compose_jobs_launch_env,
)
from vastlib.jobs.control import (                        # noqa: E402,F401
    _job_cancel_kill_script, _requeue_refusal, _vram_advisory,
)
from vastlib.jobs.risk import (                           # noqa: E402,F401
    CKPT_STALL_MULT, _STEP_DELTA_FLOOR_S, _TQDM_RE, _attempt_start_epoch,
    _ckpt_watchdog_alarm, _job_eta_s, _job_pct, _jobs_ckpt_stale,
    _jobs_defend_hint, _jobs_prior_runtime_h, _jobs_work_horizon_h,
    _step_delta_s, _tqdm_points,
)
from vastlib.jobs.runlocal import (                       # noqa: E402,F401
    _JOB_LOCAL, _JOB_LOCAL_SUBCOMMANDS, _run_local_asset_warnings,
)
from vastlib.jobs.submit import (                         # noqa: E402,F401
    _apply_env_overrides, _repo_root,
)
from vastlib.jobs.view import (                           # noqa: E402,F401
    JOB_DEF_HOMES, _JOB_VIEW_CACHE_KEY, _JOB_VIEW_CACHE_V, _JOB_VIEW_STICKY,
    _fold_fleet_jobs, _hb_age_s, _job_cell, _job_log_provenance, _job_progress,
    _parse_farm_status,
)
from vastlib.launch.launch import _do_launch              # noqa: E402,F401
from vastlib.launch.spec import (                         # noqa: E402,F401
    _EXPECTED_DEFAULT_IMAGE, _MINTED_PAIRS, _SECRET_VAL_RE,
    _TRAIN_FALLBACK_IMAGE, _b2_eu_pairs, _build_launch_spec, _ephemeral_hours,
    _is_secret_env, _last_stopping_actor, _mask_image_login, _minted_expiry,
    _r2_tc_pairs, _raw_events_soft, _read_run_soft, _require_image,
    _ship_b2_env, _split_env_secrets, _status_marker_soft, hf_login_snippet,
    hf_token_text, image_login_arg, parse_base_gate_stdout,
)
from vastlib.market.offers import (                       # noqa: E402,F401
    GPU_ALIASES, GPU_DEFAULT_POLICY_TIERS, OFFER_SCAN_LIMIT,
    VRAM_SEARCH_TOLERANCE, _gpu_policy_tiers, _inet_floor, build_search_query,
    gpu_family_names, gpu_ram_floor_mib, normalize_gpu, pick_cheapest_offer,
    pick_offers, search_offers,
)
from vastlib.market.pricing import (                      # noqa: E402,F401
    BID_HISTORY_MAX, HANDOFF_ODPROBE_MAX, RELAUNCH_ODPROBE_MAX,
    _auto_bid_price, _bid_history_for, _hist_field, _market_chunk_floor,
    _market_chunk_floors, _market_min_bid_read, _market_min_bid_soft,
    _market_ondemand_soft, _note_standing_bid, _self_floor_reset,
)
from vastlib.storage.b2 import (                          # noqa: E402,F401
    _b2_lsf_present, _b2_rcat, _ensure_b2_remote, _rclone, _rclone_soft,
)
from vastlib.storage.dashcache import (                   # noqa: E402,F401
    _INFRA_CACHE_SCHEMA, _dash_instance_rows, _dash_int, _dash_market_pace,
    _dash_market_probe, _dash_parse_sections, _dash_pct, _dash_scrub,
    _dash_write_instances, _infra_cache_db, _infra_cache_write,
)
from vastlib.supervise.handoff import (                   # noqa: E402,F401
    _handoff_complete, _handoff_run_signals, _handoff_synced_epoch_soft,
)
from vastlib.supervise.job_lane import (                  # noqa: E402,F401
    _JobLaneFloorHooks, job_supervise_init, job_supervise_tick,
)
from vastlib.supervise.journal import (                   # noqa: E402,F401
    _iso_z, _job_handoff_emit, _job_ladder_journal, _sup_emit,
)
from vastlib.supervise.replacement import (               # noqa: E402,F401
    SERVE_SELF_PARK_FRESH_S, _accrue_cost, _job_observed_lifetime_h,
    _job_rebid_ladder, _reset_run_markers, _serve_boot_sla_condemn,
    _serve_self_park_soft,
)
from vastlib.supervise.run_lane import (                  # noqa: E402,F401
    _self_floor_guard, supervise_finalize, supervise_init, supervise_tick,
)

# `cli` names the suite still diffs against their vastlib twins, plus the two
# renamed at the port (`cmd_workflow_<verb>` -> `<module>.run`,
# `add_<x>_parser` -> `<package>.add_parser`). NOT the whole cli surface: see
# the "deliberately absent" list in the module docstring.
from vastlib.cli._args import _add_cmd                    # noqa: E402,F401
from vastlib.cli._ls_render import (                      # noqa: E402,F401
    _ACTIVE_JOB_STATES, _gather_ls_data, _render_ls, _render_minimal,
)
from vastlib.cli._runsets import (                        # noqa: E402,F401
    _load_runset_config, _load_runset_spot_config,
)
from vastlib.cli.fleet import add_parser as add_fleet_parser      # noqa: E402,F401
from vastlib.cli.metrics import _metrics_probe_path       # noqa: E402,F401
from vastlib.cli.notify import add_parser as add_notify_parser    # noqa: E402,F401
from vastlib.cli.notify._get import NOTIFY_GONE_RC        # noqa: E402,F401
from vastlib.cli.sync import _load_ship_manifest          # noqa: E402,F401
from vastlib.cli.workflow.plan import run as cmd_workflow_plan    # noqa: E402,F401
from vastlib.cli.workflow.resume import run as cmd_workflow_resume  # noqa: E402,F401
from vastlib.cli.workflow.run import run as cmd_workflow_run      # noqa: E402,F401
from vastlib.cli.workflow.status import run as cmd_workflow_status  # noqa: E402,F401

from vastlib.cli.main import main                         # noqa: E402,F401

# The three guard-verdict tables `boxes.health` collapsed into `GuardVerdict`
# (plan §5 unification: `.is_zombie` / `.is_advisory` / `.short`). Rebuilt from
# the enum rather than restated, so there is still exactly one source of truth
# for membership — a flat re-export is impossible because the objects no longer
# exist upstream, and a hand-copied literal is how the two copies drift.
_GUARD_ZOMBIE_VERDICTS = frozenset(v.value for v in GuardVerdict if v.is_zombie)
_GUARD_ADVISORY_VERDICTS = frozenset(v.value for v in GuardVerdict if v.is_advisory)
_GUARD_VERDICT_SHORT = {v.value: v.short for v in GuardVerdict if v.short != v.value}


if __name__ == "__main__":
    main()
