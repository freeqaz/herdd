"""`herdd dash-cache` — refresh the dashboard's /admin snapshot, and BIND the deps.

The body landed in `storage.dashcache.cmd_dash_cache` at plan step 3. What did
NOT land with it, and cannot, is the set of reads that live ABOVE `storage` in
the plan §5 DAG. `DashDeps` is that seam, and this module is the only place that
fills it: the composition root is where an upward edge legitimately becomes a
downward call.

The six bindings, and why each is an injection and not an import
----------------------------------------------------------------
PERMANENT (`cli` and `fleet` both sit above `storage` — an import would be a DAG
violation, not a style preference):
  `gather_ls_data`      `cli/_ls_render._gather_ls_data`
  `job_cell`            `jobs.view._job_cell`
  `active_job_states`   `cli/_ls_render._ACTIVE_JOB_STATES`
  `write_fleet`         `fleet.client._dash_write_fleet`
  `offer_query`         `fleet.client._dash_offer_query`
SIDEWAYS, injected on purpose: importing `launch.spec` and `boxes.reap` from
`storage` would be LEGAL and would widen that module's import closure from
`core`-only to one containing `cmd_reap`, `destroy_box` and the credential mint
— a read-only cache module would then transitively import the destroy path.
  `is_secret_env` / `secret_val_re`   `launch.spec`
  `reap_idle_h_default`               `boxes.reap.REAP_IDLE_H_DEFAULT`

Frozen argv — four dashboard spawn sites call this
--------------------------------------------------
`--sections` / `--gpus` / `--num-gpus` / `--cache-db` / `--no-spot` are an
external contract (`dashboard/DESIGN_V5_ADMIN.md`), and the three defaults are
INTERPOLATED into help from `storage.dashcache`'s own constants — the vastlib
copies, per cli-surface.json hazard H4.

The command is READ-ONLY by construction and exits 0 or 1 only; a failed section
is skipped with its previous rows intact (a stale panel beats an empty one), and
stdout is kept empty because an `execFile` caller reads it.

Provenance: parser block moved from `tools/vast/herdd.py` `main()` and the
`DashDeps` binding written for it, plan §8 step 6, 2026-08-16,
behavior-preserving. Body: `storage/dashcache.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import reap as reap_mod
from vastlib.cli import _args, _docs, _ls_render
from vastlib.fleet import client as fleet_client
from vastlib.jobs import view as jobs_view
from vastlib.launch import spec
from vastlib.storage import dashcache


def _deps() -> dashcache.DashDeps:
    """Everything `storage.dashcache` cannot import, bound at the composition root."""
    return dashcache.DashDeps(
        is_secret_env=spec._is_secret_env,
        secret_val_re=spec._SECRET_VAL_RE,
        reap_idle_h_default=reap_mod.REAP_IDLE_H_DEFAULT,
        gather_ls_data=_ls_render._gather_ls_data,
        job_cell=jobs_view._job_cell,
        active_job_states=_ls_render._ACTIVE_JOB_STATES,
        write_fleet=fleet_client._dash_write_fleet,
        offer_query=fleet_client._dash_offer_query,
        census_query=fleet_client._dash_census_query,
    )


def run(a: argparse.Namespace) -> None:
    dashcache.cmd_dash_cache(a, deps=_deps())


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pdc = add_cmd(sub, "dash-cache",
                  "refresh the dashboard /admin snapshot in infra-metadata.db "
                  "(READ-ONLY; instances/market/fleet/account)",
                  _docs.DOC_README, _docs.DOC_DASH_V5,
                  "NOTE: read-only by construction — no launch/park/bid/"
                  "destroy/reap verb is reachable from this path",
                  "NOTE: exits 0 or 1 only; a failed section is skipped with "
                  "its previous rows kept. stdout stays empty.")
    pdc.add_argument("--sections", metavar="LIST",
                     help="comma list of sections to refresh "
                          f"(default all: {','.join(dashcache.DASH_SECTIONS)})")
    # DELIBERATELY NOT interpolated, unlike its two neighbours: the probe set
    # is the one default here that CHANGES (discovery widens it as vast lists
    # new silicon), and enumerating it put a 22-name string into the frozen
    # CLI-surface fixture — so every edit to the probe list reddened
    # `test_vastlib_cli_surface.py` and needed a fixture amendment to land.
    # That tax bought nothing: a reader who needs the list reads the constant,
    # which carries the rationale the help line never could. Naming the symbol
    # keeps the help honest AND lets the probe set move freely. See
    # `--sections` / `--num-gpus` below, which stay interpolated because their
    # defaults are short and genuinely fixed.
    pdc.add_argument("--gpus", metavar="LIST",
                     help="comma list of GPU names for the market survey "
                          "(default: the built-in probe set — see "
                          "`vastlib.storage.dashcache.DASH_GPUS_DEFAULT`)")
    pdc.add_argument("--num-gpus", metavar="LIST",
                     help="comma list of GPU counts probed per GPU "
                          f"(default: {','.join(map(str, dashcache.DASH_NUM_GPUS_DEFAULT))})")
    pdc.add_argument("--cache-db", metavar="PATH",
                     help="override the infra-metadata.db path "
                          "(default: tools/vast/infra-metadata.db, env "
                          "INFRA_METADATA_DB)")
    pdc.add_argument("--no-spot", action="store_true",
                     help="skip the per-machine spot-floor probe in the "
                          "instances gather (faster, leaves spot/avail unknown)")
    pdc.set_defaults(func=run)
    return pdc
