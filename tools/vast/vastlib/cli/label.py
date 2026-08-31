"""`herdd label <id> <label>` — set an instance's label.

An argparse-only shim: the body landed in `boxes.lifecycle.cmd_label` at plan
step 3, so this module is the parser block and nothing else.

The label is not decoration — it is the fleet's only durable per-box metadata,
and three separate lanes parse it: `run:<ID>` / `serve:<ID>` ownership, the
`keep:<why>[:until]` reaper opt-out (`core.labels`), and the `:handoff`
understudy suffix. The grammar lives in `core.labels`; this command writes
whatever string it is given, deliberately, because a validating `label` would
make an unrecognized-but-intentional label unwritable.

Provenance: parser block moved from `tools/vast/herdd.py` `main()`, plan §8
step 6, 2026-08-16, behavior-preserving. Body: `boxes/lifecycle.py`.
"""

from __future__ import annotations

import argparse

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    plb = add_cmd(sub, "label", "set instance label", _docs.DOC_README)
    plb.add_argument("id", type=int); plb.add_argument("label")  # noqa: E702 — verbatim parser block (plan §7.4)
    plb.set_defaults(func=lifecycle.cmd_label)
    return plb
