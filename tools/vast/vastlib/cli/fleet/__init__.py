"""vastlib.cli.fleet — `herdd fleet <sub>`, the operator's half of the daemon.

Why this subpackage exists
--------------------------
`fleetd` owns ALL box babysitting (`tools/vast/FLEETD_DESIGN.md`); agents talk
to it instead of spawning `supervise` processes. `fleet` is that conversation:
sixteen subcommands, every one of them a socket round trip plus a table — or,
for `log` / `report`, a direct read of the append-only journal precisely because
post-mortem is when the daemon is down.

The group is one of the four nested dispatchers in `main()`
(`job` / `fleet` / `workflow` / `notify`). Its builder already had the shape
plan §5 wants — `add_fleet_parser(sub, _add_cmd_fn)`, taking the parser factory
by INJECTION — so the composition root keeps owning `_add_cmd` and hands the
same one to every group. That injection is preserved here verbatim: `add_parser`
is the flat `add_fleet_parser`, and the sixteen per-command modules each own
their own subparser block and their `run(a)`.

The dispatch seam is unchanged. `pf.set_defaults(func=run)` on the group and
`p.set_defaults(fleetfunc=<module>.run)` on each subcommand reproduce the flat
`a.func(a)` -> `a.fleetfunc(a)` chain byte for byte; the CLI-surface diff
(plan §4/§8) compares this tree against the flat one while both are alive.

What is deliberately NOT here
-----------------------------
* **Any daemon protocol.** `fleet_request`, `_fleet_call_or_die`,
  `_fleet_requester`, the socket/journal/state paths and `FLEET_UNIT_NAME` all
  live in `vastlib.fleet.client`, which the daemon imports too. Every command
  module reaches them BY MODULE ATTRIBUTE (`client.fleet_request(...)`), never
  by `from ... import`, so a `monkeypatch.setattr(vastlib.fleet.client, ...)`
  is seen by the CLI — the patch idiom the test migration (plan §7.2) depends
  on.
* **Any policy.** What a watch does with a budget, when a replacement is
  bought, whether a destroy is deferred — all of that is the daemon's. These
  modules serialize flags into a request dict and render the reply.
* **The daemon's own CLI.** `fleetd.py serve|install-unit|deploy|status` is a
  separate, frozen 4-subcommand surface (`.port_manifests/
  fleet-daemon-deploy.json`). `fleet install` / `fleet deploy` SHELL OUT to it
  through `deploy._fleetd_script()`; they do not import its parser.
* `_fmt_age` (it is `core.fmt`, shared with nothing else in this group's ring)
  and `_fleetd_script` (it is `fleet.deploy`, where the `tools/vast` anchor
  already lives). Both are cli-surface.json H3 helpers with a home BELOW cli.

Provenance: moved from `tools/vast/herdd.py` (`cmd_fleet` + `cmd_fleet_*` +
`add_fleet_parser`), plan §8 step 6, 2026-08-16. Step 6 is ADD-ONLY at this
commit: `herdd.py` keeps its own copies.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from vastlib.cli.fleet import (
    ack,
    deploy,
    destroy,
    hosts,
    install,
    log,
    park,
    pause,
    ping,
    report,
    restart,
    resume,
    spend,
    status,
    uninstall,
    unwatch,
    watch,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vastlib.cli._args import AddCmd


# moved-from: herdd.cmd_fleet -> run
def run(a: argparse.Namespace) -> None:
    """Dispatch `herdd fleet <action>`."""
    a.fleetfunc(a)


# moved-from: herdd.add_fleet_parser -> add_parser
def add_parser(sub: object, _add_cmd_fn: AddCmd) -> argparse.ArgumentParser:
    """`herdd fleet <sub>` (FLEETD_DESIGN §5)."""
    pf = _add_cmd_fn(sub, "fleet",
                     "talk to fleetd, the always-running fleet-supervision daemon "
                     "(watch/pause/park/resume/destroy) — never spawn a supervise "
                     "process again",
                     "tools/vast/FLEETD_DESIGN.md", "tools/vast/SUPERVISE_DESIGN.md")
    fsub = pf.add_subparsers(dest="fleetcmd", required=True)
    pf.set_defaults(func=run)

    # Order is the help page. Each module owns its own flag block and its
    # `p.set_defaults(fleetfunc=...)`; this list is the only place the sequence
    # lives, and the CLI-surface diff reads it as the subcommand order.
    ping.add_parser(fsub)
    status.add_parser(fsub)
    ack.add_parser(fsub)
    watch.add_parser(fsub)
    unwatch.add_parser(fsub)
    pause.add_parser(fsub)
    park.add_parser(fsub)
    resume.add_parser(fsub)
    destroy.add_parser(fsub)
    spend.add_parser(fsub)
    hosts.add_parser(fsub)
    log.add_parser(fsub)
    report.add_parser(fsub)
    install.add_parser(fsub)
    deploy.add_parser(fsub)
    uninstall.add_parser(fsub)
    restart.add_parser(fsub)
    return pf
