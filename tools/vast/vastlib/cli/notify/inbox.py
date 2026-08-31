"""`herdd notify inbox` — the richest read in vast's notification channel.

Why this module exists
----------------------
`GET /api/v0/notifications/inbox/` is a HIDDEN endpoint (NOTIFY_DESIGN §1.3):
undocumented, revocable, and today the only place `outbid` rows carry a
structured `{instance_id, machine_id, your_bid, new_min_bid}` — i.e. the
DISPLACING price, which nothing else in the API reports. A 404 is therefore a
first-class outcome with its own exit code (`NOTIFY_GONE_RC`), not a traceback:
the endpoint's disappearance is an expected end state a caller should be able
to test for.

What is deliberately NOT here
-----------------------------
* Any reshaping under `--json`. That flag emits the API payload VERBATIM, rows
  unmodified and `--limit` NOT applied, because it is the fixture-capture path
  — a `--json` that reshaped what the server said would silently launder the
  evidence this command exists to collect.
* The table. `notify.render_inbox` is the flat leaf's, shared with fleetd's
  poll tick.

Provenance: moved from `tools/vast/herdd.py::cmd_notify_inbox`, plan §8
step 6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import notify

from vastlib.cli import _args
from vastlib.cli._docs import DOC_AUTOBID, DOC_NOTIFY
from vastlib.cli.notify import _get


# moved-from: herdd.cmd_notify_inbox -> run
def run(a: argparse.Namespace) -> None:
    """`GET /api/v0/notifications/inbox/` — the richest read in the channel.

    HIDDEN endpoint (NOTIFY_DESIGN §1.3): undocumented, revocable, and today
    the only place `outbid` rows carry a structured
    `{instance_id, machine_id, your_bid, new_min_bid}`. A 404 is therefore a
    first-class outcome with its own exit code, not a traceback."""
    data, err = _get._notify_get(notify.INBOX_PATH)
    if data is None:
        if notify.is_gone(err):  # type: ignore[no-untyped-call]
            print("notify: hidden endpoint gone (expected someday) — see "
                  "tools/vast/NOTIFY_DESIGN.md", file=sys.stderr)
            sys.exit(_get.NOTIFY_GONE_RC)
        sys.exit(f"error: {err}")
    if a.json:
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
        return
    sys.stdout.write(notify.render_inbox(data, time.time(), limit=a.limit))  # type: ignore[no-untyped-call]


def add_parser(nsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = nsub.add_parser(
        "inbox", help="the notification feed: age, type, instance, machine, "
                      "and for an outbid our bid -> the displacing min bid",
        epilog=_args._docs_epilog(DOC_NOTIFY, DOC_AUTOBID),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true",
                   help="emit the API payload VERBATIM (rows unmodified, "
                        "--limit not applied — this is the fixture-capture "
                        "path, so it must not reshape what the server said)")
    p.add_argument("--limit", type=int, default=0, metavar="N",
                   help="show only the N newest rows in the table (0 = all; "
                        "the server's own window is ~50 rows / ~3 days)")
    p.set_defaults(notifyfunc=run)
