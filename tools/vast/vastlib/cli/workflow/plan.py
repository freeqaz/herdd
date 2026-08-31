"""`herdd workflow plan` — validate a WORKFLOW module, write its spec, spend nothing.

Why this module exists
----------------------
`plan` is the only workflow verb guaranteed not to cost money: it loads the
`WORKFLOW` module, writes `spec.json`, and validates every expanded child
bundle and every `InputRef` wiring, so a bad stage fails closed BEFORE anything
is submitted. Keeping it its own module keeps that guarantee readable — the
handler has no box seam to reach for.

What is deliberately NOT here
-----------------------------
* The validation itself, and `--online`'s strict checks (B2 asset staleness,
  image digests, credential lifetime, spend estimate) — `workflows.ctl.
  plan_workflow` owns all of it. This module chooses `online=True` or not, and
  prints.
* Any default for `--online` beyond `False`: the offline path is the one that
  runs with no network and no credentials, and it stays the default.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_plan` plus its
`main()`-inline parser block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import imageref

from vastlib.boxes import lifecycle
from vastlib.storage import b2
from vastlib.workflows import ctl as workflowctl


# moved-from: herdd.cmd_workflow_plan -> run
def run(a: argparse.Namespace) -> None:
    """Validate a WORKFLOW module + write its `spec.json` (no run loop, no
    box). `--online` (roadmap M4-T1: B2-manifest/asset/image/credential
    strict checks + spend estimate) additionally resolves per-profile image
    digests (via `image_ref_digest`, agreeing byte-for-byte with the box
    resolver) and worst-case spend; the offline default path is unchanged."""
    b2._ensure_b2_remote()
    if a.online:
        rc, result = workflowctl.plan_workflow(
            a.path, online=True, actor=lifecycle._cli_actor(),
            image_resolver=imageref.image_ref_digest, now_epoch=time.time())
    else:
        rc, result = workflowctl.plan_workflow(a.path, actor=lifecycle._cli_actor())
    if a.json:
        print(json.dumps(result, indent=2))
    elif rc == workflowctl.EXIT_OK:
        print(f">> planned workflow {result.get('wf_id')}")
    else:
        print(f"error: {result.get('error')}", file=sys.stderr)
    sys.exit(rc)


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("plan", help="validate a WORKFLOW module + write spec.json "
                        "(no run loop, no box)")
    p.add_argument("path", help="path to a WORKFLOW module (.py defining `WORKFLOW`)")
    p.add_argument("--online", action="store_true",
                   help="strict B2/asset/image/credential checks + spend estimate "
                        "(M4-T1; not yet implemented — offline plan always runs)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(wffunc=run)
