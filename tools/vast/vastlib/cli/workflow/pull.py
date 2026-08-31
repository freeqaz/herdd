"""`herdd workflow pull` — download the terminal stage's results to out/workflows/<WF_ID>/.

Why this module exists
----------------------
`pull` is the "give me the artifacts" verb, and its whole subtlety is WHICH
stage's artifacts. The answer needs the spec, not just the folded view: the
terminal stage is `wf.stages[-1]` by declaration order, which only `spec.json`
knows. So this handler reads the spec first and fails closed with the spec's
own error if it cannot.

The destination convention is deliberately the same as `job pull`'s —
`out/workflows/<WF_ID>/` under the repo root, overridable by a positional —
so the two lanes drop artifacts in sibling trees instead of two invented
places.

What is deliberately NOT here
-----------------------------
* The transfer. `workflows.ctl.pull_workflow` resolves the stage, finds its
  `job_id` and calls `jobmeta.pull_results`; the Zone S exceptions that
  escape it (`JobmetaError` / `RunmetaError`) are mapped to `EXIT_ARTIFACT`
  here because exit codes are a CLI concern.
* A `--stage` flag. `pull_workflow` takes one, and the flat CLI deliberately
  does not expose it (the terminal stage is what an operator means); adding it
  would be a new flag, not a port.

Provenance: moved from `tools/vast/herdd.py::cmd_workflow_pull` +
`_workflow_pull_dest` plus its `main()`-inline parser block, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import os
import sys

from vastlib.jobs import submit
from vastlib.storage import b2
from vastlib.workflows import ctl as workflowctl

import jobmeta
import runmeta


# `_repo_root` is reached through `jobs.submit` (its ported home) rather than
# recomputed here: it is a patched seam in the job lane, and a second copy
# would answer a different question the moment a test steered one of them.
# moved-from: herdd._workflow_pull_dest
def _workflow_pull_dest(a: argparse.Namespace) -> str:
    # The local annotation is the one addition to a verbatim body: `a.dest` is
    # `Any` off the Namespace, and a bare `return` of it trips mypy's
    # `warn_return_any` in the strict lane.
    dest: str = a.dest or os.path.join(submit._repo_root(), "out", "workflows", a.wf_id)
    return dest


# moved-from: herdd.cmd_workflow_pull -> run
def run(a: argparse.Namespace) -> None:
    """Download the terminal stage's (or `wf.stages[-1]`'s) results ->
    out/workflows/<WF_ID>/, mirroring `cmd_job_pull`'s dest convention."""
    b2._ensure_b2_remote()
    try:
        wf = workflowctl.read_spec(a.wf_id)
    except workflowctl.WorkflowCtlError as e:
        sys.exit(f"error: {e}")
    dest = _workflow_pull_dest(a)
    try:
        rc, result = workflowctl.pull_workflow(wf, a.wf_id, dest)
    except (jobmeta.JobmetaError, runmeta.RunmetaError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(workflowctl.EXIT_ARTIFACT)
    if rc != workflowctl.EXIT_OK:
        print(f"error: {result.get('error')}", file=sys.stderr)
        sys.exit(rc)
    files = result.get("files") or []
    print(f">> pulled {len(files)} result file(s) for stage "
          f"{result.get('stage')} -> {dest}")
    for f in files:
        print(f"   {f}")


def add_parser(wsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = wsub.add_parser("pull", help="download the terminal stage's results -> "
                        "out/workflows/<WF_ID>/")
    p.add_argument("wf_id")
    p.add_argument("dest", nargs="?", default=None)
    p.set_defaults(wffunc=run)
