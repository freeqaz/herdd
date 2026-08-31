"""`herdd logs <id>` — ask vast to upload an instance's container logs.

One PUT and a sentence. The endpoint is asynchronous: it does not RETURN logs,
it schedules an upload to vast's S3, which is why the printed line points at the
portal rather than at stdout. Nothing here waits, polls or fetches.

What is deliberately NOT here
-----------------------------
* Any log RETRIEVAL. Reading a box's logs live is `herdd ssh <id> --exec …`
  or the jobs lane's `job logs`; this command exists only for the case where
  the box is unreachable and the portal is the only surface left.
* Error handling. `core.api.request` already exits non-zero with the API's own
  message on failure, and a second layer of it here would only obscure that.

Provenance: moved from `tools/vast/herdd.py` (`cmd_logs`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.core import api


# moved-from: herdd.cmd_logs
def run(a: argparse.Namespace) -> None:
    api.request("PUT", f"v0/instances/{a.id}/logs/", {})
    print(f"requested logs for {a.id}; they upload to S3 — fetch from the portal or retry show.")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    plg = add_cmd(sub, "logs", "request instance logs", _docs.DOC_README, _docs.DOC_DEBUG)
    plg.add_argument("id", type=int); plg.set_defaults(func=run)  # noqa: E702 — verbatim parser block (plan §7.4)
    return plg
