"""`herdd notify webhooks` — what is subscribed, which today is nothing.

Why this module exists
----------------------
EMPTY is the current and CORRECT state (NOTIFY_DESIGN D5: the webhook slice is
deferred behind an owner ingress decision), so an empty list prints as headroom
rather than as an error. A command that treated "no subscriptions" as a failure
would manufacture an alarm for a decision that was made deliberately.

Provenance: moved from `tools/vast/herdd.py::cmd_notify_webhooks`, plan §8
step 6.
"""

from __future__ import annotations

import argparse
import json
import sys

import notify

from vastlib.cli import _args
from vastlib.cli._docs import DOC_NOTIFY
from vastlib.cli.notify import _get


# moved-from: herdd.cmd_notify_webhooks -> run
def run(a: argparse.Namespace) -> None:
    """`GET /api/v0/webhooks/` — what is subscribed. EMPTY is the current and
    correct state (NOTIFY_DESIGN D5: the webhook slice is deferred behind an
    owner ingress decision), so it prints as headroom, never as an error."""
    data, err = _get._notify_get(notify.WEBHOOKS_PATH)
    if data is None:
        sys.exit(f"error: {err}")
    if a.json:
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
        return
    sys.stdout.write(notify.render_webhooks(data))  # type: ignore[no-untyped-call]


def add_parser(nsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = nsub.add_parser(
        "webhooks", help="webhook subscriptions (there are none today, by "
                         "design — NOTIFY_DESIGN D5)",
        epilog=_args._docs_epilog(DOC_NOTIFY),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true")
    p.set_defaults(notifyfunc=run)
