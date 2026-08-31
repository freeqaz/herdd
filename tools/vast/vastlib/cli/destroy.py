"""`herdd destroy` — the only teardown that stops storage billing.

An argparse-only shim: the body landed in `boxes.lifecycle.cmd_destroy` at plan
step 3 (with the credential revoke it fans out to), so this module is the parser
block and nothing else.

The `NOTE:` in the help is the whole point of the command's existence as a
separate verb from `stop`: a stopped or outbid box keeps billing its ALLOCATED
disk — measured 2026-07-30 at $2.13–$4.62/day/box — so "I parked it" is not
"I stopped paying for it". `destroy` is what ends the meter.

Provenance: parser block moved from `tools/vast/herdd.py` `main()`, plan §8
step 6, 2026-08-16, behavior-preserving. Body: `boxes/lifecycle.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pd = add_cmd(sub, "destroy", "destroy instance(s) permanently",
                 _docs.DOC_README,
                 "NOTE: this is the only teardown that stops storage billing (stop/outbid keep billing disk)")  # noqa: E501 — verbatim help text (plan §7.4)
    pd.add_argument("id", type=int, nargs="*")
    pd.add_argument("--all", action="store_true", help="destroy ALL my instances")
    pd.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    pd.set_defaults(func=lifecycle.cmd_destroy)
    return pd
