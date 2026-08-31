"""vastlib.workflows — multi-step orchestration, and the module that breaks the one real cycle.

Why this layer exists
---------------------
`herdd` and `workflowctl` import each other at module top level, both eager.
It is latent only because neither dereferences the other at module-exec time —
a real import cycle that has been one refactor away from biting for months.
Placing the workflow controller HERE, one ring below `cli` and strictly above
`core`/`boxes`/`jobs`, breaks it by construction: workflows consume `vastlib`,
never `herdd`, and nothing in `vastlib` imports workflows except `cli`.

Planned contents (plan §5)
--------------------------
  `workflowctl.py`, `workflowmeta.py`, `workflow.py` and `jobmatrix.py`
  absorbed. Their flat files at `tools/vast/` become deprecation shims for one
  release and are then deleted.

What is deliberately NOT here
-----------------------------
* No job submission mechanics — a workflow composes `jobs.submit` and
  `jobs.control`; it does not re-implement the bundle or the queue.
* No supervision loop of its own.
* Nothing `cli` needs to reach around. If a workflow command needs a box
  operation, it calls down through `boxes`/`jobs` like every other consumer;
  the point of this module is that the arrow only ever points one way.

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 5.
"""

from __future__ import annotations
