"""`herdd notify types` — the event catalog and each key's default channels.

Why this module exists
----------------------
`GET /api/v0/notification-types` is the published half of the channel: which
events exist, what topic and category they carry, and which channels each key
defaults to. `webhooks` appearing in a key's channel list is what a
`POST /webhooks/` subscription flips on — so this table is how you find out
what a subscription would even be able to deliver.

Provenance: moved from `tools/vast/herdd.py::cmd_notify_types`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import sys

import notify

from vastlib.cli import _args
from vastlib.cli._docs import DOC_NOTIFY
from vastlib.cli.notify import _get


# moved-from: herdd.cmd_notify_types -> run
def run(a: argparse.Namespace) -> None:
    """`GET /api/v0/notification-types` — the event catalog and which channels
    each key defaults to. `webhooks` in a key's channel list is what a
    `POST /webhooks/` subscription flips on."""
    data, err = _get._notify_get(notify.TYPES_PATH)
    if data is None:
        sys.exit(f"error: {err}")
    if a.json:
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
        return
    sys.stdout.write(notify.render_types(data))  # type: ignore[no-untyped-call]


def add_parser(nsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = nsub.add_parser(
        "types", help="the event catalog: key, topic, category, default channels",
        epilog=_args._docs_epilog(DOC_NOTIFY),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    p.set_defaults(notifyfunc=run)
