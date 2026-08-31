"""`herdd whoami` — who the API key belongs to, and whether it can still pay.

Three lines of account state, and the first command in the registry because it
is the first command an operator runs: a wrong `VAST_API_KEY` in `.env`, an
expired key, or a $0 balance all present downstream as a launch that refuses or
a box that never boots, and all three are visible here in one GET.

`can_pay` / `has_billing` are printed beside the balance deliberately — a
non-zero credit with `can_pay: False` is the shape that makes a launch fail with
a message about the offer rather than about the account.

What is deliberately NOT here
-----------------------------
* Any judgement. This command prints what the API says and exits 0; the
  spend refusals live in the launch/bid paths where an operator can still act.
* Key repair. Where the key comes from is `core.config.load_env`.

Provenance: moved from `tools/vast/herdd.py` (`cmd_whoami`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse

from vastlib.cli import _args, _docs
from vastlib.core import api, fmt


# moved-from: herdd.cmd_whoami
def run(a: argparse.Namespace) -> None:
    d = api.request("GET", "v0/users/current/")
    print(f"user   : {d.get('id')}  {d.get('email')}")
    print(f"credit : {fmt.dollars(d.get('credit'))}   balance: {fmt.dollars(d.get('balance'))}")
    print(f"can_pay: {d.get('can_pay')}   has_billing: {d.get('has_billing')}")


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pwho = add_cmd(sub, "whoami", "show account credit/balance",
                   _docs.DOC_README)
    pwho.set_defaults(func=run)
    return pwho
