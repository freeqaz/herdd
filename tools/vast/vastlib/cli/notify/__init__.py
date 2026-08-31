"""vastlib.cli.notify — `herdd notify <sub>`, the read-only probe over vast's
notification channel.

Why this subpackage exists
--------------------------
NOTIFY_DESIGN §4 slice S1. Read-only, stateless, policy-free: three GETs and
three tables. The cursor, the journaling and the degradation rules live in
fleetd's tick (S2a); this command is the probe instrument an operator reaches
for when a box vanished and they want to know whether vast says it was outbid.

**This is the FOURTH nested dispatcher, and plan §5 names only three**
(`job` / `fleet` / `workflow`). `notify` landed after the plan was written —
`cli-surface.json` hazard H2. It is also the reference shape for the registry
loop: `add_notify_parser(sub, _add_cmd_fn)` already takes the parser factory by
INJECTION, which is exactly how the composition root hands every group the same
`_add_cmd`.

What is deliberately NOT here
-----------------------------
* **Any write.** Nothing under `notify` writes; see `_get.py` for the D3
  reasoning about `seen_through_at`.
* The renderers, the path constants and `is_gone`. They are
  `tools/vast/notify.py` — a flat leaf shared with fleetd's poll tick, so the
  CLI and the daemon read the same feed through the same parser. It stays flat
  deliberately (plan §3 lists no home for it, and duplicating it is how one
  parser becomes two).
* `_notify_get` / `NOTIFY_GONE_RC`, which `cli-surface.json` proposes for this
  file: they are in `_get.py` instead, because the three command modules import
  them and this module imports the three command modules. See `_get.py`'s
  docstring for the cycle.

Provenance: moved from `tools/vast/herdd.py` (`cmd_notify`, the three
`cmd_notify_*` handlers and `add_notify_parser`), plan §8 step 6, 2026-08-16.
Step 6 is ADD-ONLY at this commit: `herdd.py` keeps its own copies.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from vastlib.cli._docs import DOC_FLEETD, DOC_NOTIFY
from vastlib.cli.notify import inbox, types, webhooks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vastlib.cli._args import AddCmd


# moved-from: herdd.cmd_notify -> run
def run(a: argparse.Namespace) -> None:
    """Dispatch `herdd notify <action>`."""
    a.notifyfunc(a)


# moved-from: herdd.add_notify_parser -> add_parser
def add_parser(sub: object, _add_cmd_fn: AddCmd) -> argparse.ArgumentParser:
    """`herdd notify <sub>` — NOTIFY_DESIGN §4 slice S1.

    Read-only, stateless, policy-free: three GETs and three tables. The cursor,
    the journaling and the degradation rules live in fleetd's tick (S2a); this
    command is the probe instrument an operator reaches for when a box vanished
    and they want to know whether vast says it was outbid."""
    pn = _add_cmd_fn(sub, "notify",
                     "read vast's notification channel: the inbox feed (outbid "
                     "rows carry the DISPLACING price), the event catalog, and "
                     "the webhook subscriptions — read-only, no state",
                     DOC_NOTIFY, DOC_FLEETD)
    nsub = pn.add_subparsers(dest="notifycmd", required=True)
    pn.set_defaults(func=run)

    # Order is the help page; each module owns its own flags, epilog and
    # `p.set_defaults(notifyfunc=...)`.
    inbox.add_parser(nsub)
    types.add_parser(nsub)
    webhooks.add_parser(nsub)
    return pn
