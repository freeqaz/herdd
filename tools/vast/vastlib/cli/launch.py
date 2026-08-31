"""`herdd launch` — rent a box. The biggest parser block in the surface (33 flags).

The body is a thin wrapper over `launch.launch._do_launch(a)`, whose return
value (`cid, offer_id, dph`) `cmd_train` consumes; this command reads only the
`cid`, for the `--jobs` watch-ladder hint. Everything else here is the flag
surface, and it is the flag surface that has to survive the port byte-for-byte.

The four defaults that are NOT literals
---------------------------------------
Each is interpolated into printed help, so each must come from the VASTLIB copy
of its constant (cli-surface.json hazard H4 — during the add-only wave the flat
file still holds its own, and importing the wrong one renders identically today
and drifts silently later):

  `--image`  default from `default_image()` below — read from `herdd.yaml` at
             runtime; the help quotes `launch.spec._EXPECTED_DEFAULT_IMAGE`.
  `--disk`   `core.config.default_disk_gb("launch")`, in BOTH the default and
             the help string, so the two cannot disagree.
  `--price`  `bidpolicy.BID_TARGET_MULT` — the auto-bid multiple over the live
             spot floor.
  `--boot-health`  two entries from `core.config._BOOT_KNOB_DEFAULTS`.

`pl.set_defaults(type="bid")` — launch, and only launch, defaults to
interruptible+auto-priced (AUTOBID_DESIGN / owner ruling). It is a
`set_defaults` rather than a different `add_search_filters` default precisely so
`search` and `supervise` keep the shared `ondemand` default; a port that moved
the value into the shared block would silently flip two other commands.

Where `default_image` lives, and why
------------------------------------
The flat `main()` read `load_herdd_config()["default_image"]` once in its
prologue and closed over it for the two parsers that bake it into a flag default
(`launch`, `supervise`). The registry loop has one two-argument call shape for
all 29 commands, so the read moved HERE, next to the help text that quotes it —
`cli/supervise.py` calls this function too. Same source, same value, same
printed bytes; hazard H5 still binds on the diff test (this default is
environment-dependent, so `herdd.yaml` must be pinned on both arms).

What is deliberately NOT here
-----------------------------
* The launch itself. Offer pick, bid pricing, secret split, key mint, the
  create POST and the post-launch watch are all `launch/launch.py` and
  `launch/spec.py`.
* The fail-closed image gate. `_require_image` refuses at the paths that
  actually create a container; a parser-level `required=True` would break `ls`,
  `stop` and `destroy` on a broken config, which is exactly the failure mode
  the lazy resolution exists to avoid.

Provenance: moved from `tools/vast/herdd.py` (`cmd_launch`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _compose, _docs
from vastlib.core import config
from vastlib.fleet import client as fleet_client
from vastlib.launch import launch as launch_mod
from vastlib.launch import spec

import bidpolicy


# DELIBERATELY MARKER-LESS (ruled 2026-08-16, wave 6a). This is a PARTIAL port:
# the body is `main()`'s two-line prologue
# (`cfg = load_herdd_config(); default_image = cfg.get("default_image")`),
# hoisted here because `launch` and `supervise` are the only parsers that bake
# the value into a flag default. It carried a `# moved-from: herdd.main (…)`
# marker, which was MALFORMED — the grammar is `<module>.<name>[ -> <new>]` with
# no room for prose — and spelling it `herdd.main -> default_image` would give
# `herdd.main` a second rename target. `vastlib.cli.main.main` owns that
# mapping: it is the name an external caller or a `monkeypatch.setattr` site
# actually reaches. A fragment of a function body has no rename claim on the
# function. Recorded in `gen_rename_table.py::KNOWN_MARKERLESS`.
def default_image() -> str | None:
    """The `--image` default: `herdd.yaml`'s `default_image`, or None.

    Resolution stays LAZY and never raises. `main()` builds the parser for EVERY
    subcommand, so dying on an unreadable config would take `ls`, `show`, `stop`
    and `destroy` down with it — precisely the commands you need when the config
    is broken and a box is billing. An absent default becomes `None`, and only
    the paths that actually create a container refuse (`_require_image`).
    """
    cfg = config.load_herdd_config()
    image = cfg.get("default_image")
    return image if image is None else str(image)


# moved-from: herdd.cmd_launch
def run(a: argparse.Namespace) -> None:
    """Thin wrapper over _do_launch: cmd_train consumes the returned
    (cid, offer_id, dph); here only the cid is read, to address the
    watch-ladder hint at the box that was just rented.

    The `_compose.bind()` line is not an extra step, it is what makes `--jobs`
    and `--fleet-watch` work at all: `launch/launch.py` reaches
    `compose_jobs_launch_env` and `fleet_watch_best_effort` through raising
    seams, because their definitions sit in rings ABOVE `launch` and no import
    from there is legal at any timing. `cli` is the only ring that may see both,
    so the wiring happens here, in the command, immediately before the call that
    needs it. `cli.main.main()` calls the same `bind()` for every other
    subcommand; the duplication is three attribute assignments and it is what
    lets a caller drive `run(ns)` directly — as `test_vastlib_cli_launch.py`
    does — without reproducing the composition root.
    """
    _compose.bind()
    cid, _offer, _dph = launch_mod._do_launch(a)
    # The `--fleet-watch` registration `_do_launch` just made is BARE — no
    # ladder. Say so where the operator still has the id, but only for a jobs
    # box: `--profile jobs` is wrong advice for a serve or manual box, and it
    # cannot live one ring down (`launch/` may not import `fleet/`).
    if cid and getattr(a, "jobs", False) and getattr(a, "fleet_watch", False):
        fleet_client.print_bare_watch_hint(cid, "jobs")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    from vastlib.cli import search  # local: `search` owns the shared filter block

    pl = add_cmd(sub, "launch", "launch an instance (auto-picks cheapest offer unless --offer)",
                 _docs.DOC_README, _docs.DOC_TRAINING, _docs.DOC_SKILL_IMAGE,
                 "NOTE: onstart is capped at 16384 bytes incl. auto-prepends (hf_login, image_login)")  # noqa: E501 — verbatim parser block (plan §7.4)
    search.add_search_filters(pl)
    # launch defaults to interruptible+auto-priced (AUTOBID_DESIGN / owner ruling):
    # trade downtime risk for cost. Stable serve/eval boxes pass --type ondemand.
    # Scoped to launch — search/supervise keep the shared default.
    pl.set_defaults(type="bid")
    pl.add_argument("--offer", type=int, help="explicit offer id (skip auto-pick)")
    pl.add_argument("--offer-machine", dest="offer_machine", type=int, default=None,
                    metavar="ID",
                    help="the machine_id behind --offer. vast's offer `id` filter "
                         "returns no rows for live offers, so a pin alone cannot be "
                         "auto-priced; this supplies the machine the working market "
                         "reads need. Prefer `--machine ID` (search, no pin) when "
                         "you don't need that exact chunk")
    pl.add_argument("--image", default=default_image(),
                    help=f"(default from herdd.yaml's default_image, else "
                         f"{spec._EXPECTED_DEFAULT_IMAGE!r}). There is NO stock "
                         f"fallback — launch REFUSES when neither is available "
                         f"(see tools/vast/herdd.yaml)")
    pl.add_argument("--disk", dest="disk", type=int,
                    default=config.default_disk_gb("launch"),
                    help=f"container disk GB (default "
                         f"{config.default_disk_gb('launch')}; set "
                         f"default_disk in herdd.yaml to change it)")
    pl.add_argument("--cc-allow", dest="cc_allow", default=None,
                    metavar="SM[,SM...]",
                    help="architecture allowlist as sm levels, e.g. "
                         "'80,86,89,90'. Narrows the offer search AND is "
                         "stamped into the box env (LAUNCH_CC_ALLOW), so every "
                         "automatic replacement of this box is held to the same "
                         "silicon — a workload whose kernels have no sm_120 "
                         "image (flash_attn 2.8.3) must not be rehosted onto an "
                         "RTX PRO 6000 by an eviction. Offers that do not "
                         "advertise compute_cap are excluded while this is set")
    pl.add_argument("--onstart", help="onstart script: file path or inline string")
    pl.add_argument("--env", action="append", help="KEY=VALUE (repeatable)")
    pl.add_argument("--eval-env-ver", dest="eval_env_ver", default=None, metavar="V",
                    help="pin the baked eval-env tarball version (-> EVAL_ENV_VER "
                         "in the box env). This is the ONLY pin "
                         "onstart/fetch_eval_env.sh can see: jobd provisions the "
                         "env BEFORE the job's own .job.env exists, so a bundle "
                         "or `job submit --env` pin documents the choice but "
                         "cannot steer the fetch. Sugar over --env")
    pl.add_argument("--port", action="append", help="expose port N (repeatable)")
    pl.add_argument("--jupyter", action="store_true", help="expose 8080")
    pl.add_argument("--runtype", default="ssh_direct", choices=["ssh", "ssh_direct", "ssh_proxy", "jupyter"])  # noqa: E501 — verbatim parser block (plan §7.4)
    pl.add_argument("--label", default=None)
    pl.add_argument("--price", type=float, default=None,
                    help=f"bid $/hr (interruptible). Default: auto = {bidpolicy.BID_TARGET_MULT:g}x the "  # noqa: E501 — verbatim parser block (plan §7.4)
                         f"live spot floor, clamped below on-demand — you rarely set this")
    pl.add_argument("--template-id", type=int, default=None)
    pl.add_argument("--ssh", dest="ssh", action="store_true", default=True,
                    help="attach local ssh pubkey + install/repair it in "
                         "authorized_keys at every boot (DEFAULT since "
                         "2026-07-31; kept for back-compat)")
    pl.add_argument("--no-ssh", dest="ssh", action="store_false",
                    help="do NOT install the ssh pubkey (box is un-debuggable)")
    pl.add_argument("--ssh-key-file", default=None, help="pubkey path (default ~/.ssh/id_ed25519.pub|id_rsa.pub)")  # noqa: E501 — verbatim parser block (plan §7.4)
    pl.add_argument("--hf-token", default=None,
                    help="explicit HuggingFace token (default: auto from .env / "
                         "~/.config/herdd/hf_token / ~/.cache/huggingface/token)")
    pl.add_argument("--no-hf-token", action="store_true",
                    help="do NOT auto-upload a HuggingFace token")
    pl.add_argument("--login", default=None,
                    help="explicit Vast image_login string for a private image, e.g. "
                         "'-u USER -p TOKEN registry.example.com' (default: a minted "
                         "R2 pull token when --image is on registry.example.com)")
    pl.add_argument("--no-registry-login", action="store_true",
                    help="do NOT attach private-registry pull creds")
    pl.add_argument("--jobs", action="store_true",
                    help="start jobd at boot (provision-time): the box polls its "
                         "B2 queue (jobs/queue/<IID>/) immediately and self-parks "
                         "when the queue drains. Needs B2_* in env.")
    pl.add_argument("--no-idle-park", dest="no_idle_park", action="store_true",
                    help="[--jobs] do NOT self-park when the queue drains "
                         "(box runs until destroyed — opt out of the default)")
    pl.add_argument("--idle-park-grace", dest="idle_park_grace", type=int, default=None,
                    metavar="SECS", help="[--jobs] idle grace after the queue "
                    "drains before self-park (default 600)")
    pl.add_argument("--no-job-deadline", dest="no_job_deadline", type=int, default=None,
                    metavar="SECS", help="[--jobs] park deadline when NO job ever "
                    "arrives (default 3600)")
    pl.add_argument("--wait", type=int, default=0, metavar="SECS", help="wait up to N s for 'running' then print ssh")  # noqa: E501 — verbatim parser block (plan §7.4)
    pl.add_argument("--boot-health", dest="boot_health", action="store_true",
                    help="[--wait] watch the docker image-pull throughput while "
                         "the box is loading; on a sustained-slow host (< "
                         f"{int(config._BOOT_KNOB_DEFAULTS['BOOT_MIN_MBPS'])} MB/s over "
                         f"{config._BOOT_KNOB_DEFAULTS['BOOT_MBPS_WINDOW_S']}s) print a "
                         "condemnation + destroy hint and exit nonzero (single-shot: "
                         "no auto-relaunch). Knobs: BOOT_MIN_MBPS / BOOT_MBPS_WINDOW_S "
                         "/ BOOT_HEALTH_POLL_S (env or herdd.yaml).")
    pl.add_argument("--dry-run", action="store_true")
    pl.add_argument("--force", action="store_true",
                    help="skip the live-run preflight: launch even if a "
                         "run:<ID>-labelled box is already live (double-write risk)")
    pl.add_argument("--fleet-watch", dest="fleet_watch", action="store_true", default=True,
                        help="no-op: ON by default since 2026-08-20 (FLEET_REVIEW_2026-08-20 "
                             "item 3; kept for back-compat, --no-fleet-watch opts out). "
                             "It registers a BARE watch and that is NOT supervision — "
                             "observation + alarms, no bid defense, no outbid rescue, no "
                             "eviction replacement. The spend-capable ladder is a separate "
                             "`fleet watch <IID> --profile jobs|run|serve --budget N`, armed "
                             "AFTER the box has non-terminal tickets (launch_jobs_box.sh "
                             "does the whole order).")
    pl.add_argument("--no-fleet-watch", dest="fleet_watch", action="store_false",
                        help="do NOT register the box with fleetd after launch")
    pl.set_defaults(func=run)
    return pl
