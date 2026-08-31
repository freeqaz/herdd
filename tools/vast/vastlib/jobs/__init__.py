"""vastlib.jobs — the jobs-v2 lane: build a bundle, submit it, watch it, price its risk.

Why this layer exists
---------------------
`_job_attach_files` decides what gets copied onto a box, and two independent
checkers read that decision: `test_jobd_bundle_imports_flat.py` (dynamic —
proves every shipped leaf imports bare-name under `python3 -P` in a flat,
repo-invisible directory) and `shipcheck.py`'s `FLAT_NAMESPACES` (static). It
must therefore have exactly one home, and both checkers must read it from
there. The risk metrics next door are the opposite kind of code — 16 of 17
functions transitively pure — and are the package's designated first module to
reach 100% typed and tested.

Planned modules (plan §5)
-------------------------
  bundle.py    `_job_attach_files` as the single source of truth, the jobd
               bootstrap staging, the import gate.
  submit.py    submission.
  scan.py      bulk fold: N job event logs in O(1) rclone calls, for the
               commands that read the WHOLE queue (`job orphans`, `job ls`).
               Not in plan §5 — added 2026-08-17 against a measured defect
               (275 tickets = 275 `rclone copy` subprocesses = 139 s). Its
               docstring carries the measurement and the freshness contract.
  view.py      status/list/inspect rendering.
  control.py   retarget, requeue, cancel, orphan reconciliation.
  risk.py      cluster C23's pure metrics: ETA, work-at-risk, defend hints,
               checkpoint staleness.
  runlocal.py  the run-local lane. `require_local_gpu` stays in
               `core.config` — the `allow_local_gpu` switch is one knob in one
               place (owner ruling 2026-08-11) and does not get a second home
               here.

What is deliberately NOT here
-----------------------------
* **Zone S.** `jobmeta.py`, `runmeta.py` and `bidpolicy.py` are shipped flat
  leaves; `vastlib` imports them bare-name and they never import back. Nothing
  in this package may be a `try/except ImportError` dual-form module — that
  ambiguity is exactly what the flat-bundle test exists to distrust.
* `autotune.py` and `jobcommon/` are bundle-side (they run ON the box) and are
  NOT absorbed (plan §3).
* No supervision of a running job — `supervise.job_lane` owns the tick loop,
  the eviction handling, and the handoff accrual.

Provenance: skeleton created 2026-08-16, plan §8 step 1. `risk.py` arrives in
step 3 (separable rings); the rest in step 5.
"""

from __future__ import annotations
