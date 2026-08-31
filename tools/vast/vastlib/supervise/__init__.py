"""vastlib.supervise — the knot: keeping runs and jobs alive across evictions.

Why this layer exists
---------------------
This is the hard part of the whole file. Four clusters — supervise-run-lane,
job-handoff-state, eviction-replace, job-supervise-loop — total ~5,400 body
lines and are mutually recursive (23/15/13/13 cross-edges). They are ported as
ONE unit (plan §8 step 4) because no smaller cut leaves the tree green, and
`test_supervise.py`'s 426 references migrate with them in the same step.

Two leaves were misfiled inside the biggest cluster and called from four
others; they come out FIRST, in step 2, so that everything else can be ported
without dragging the knot along: `_sup_emit`, `_job_handoff_emit`,
`_job_ladder_journal`, `_iso_z` -> `journal.py`.

Planned modules (plan §5)
-------------------------
  journal.py      the extracted-first leaves. Every event schema stays
                  BYTE-IDENTICAL — these bodies are a B2 wire contract.
  state.py        TypedDict structural types (total=False) over the SAME
                  `st` / `jc` / handoff dicts, plus the key inventory as
                  documentation. (The original dataclass design was measured
                  impossible: open key sets, `.pop()` as first-class state,
                  Zone-S constructors, and no whole-dict serialization seam —
                  see state.py's docstring and plan §5 as corrected
                  2026-08-16.) Several keys are a persisted wire format
                  (`REPLACEMENT_STATE_KEYS`, `RUN_STATE_KEYS`): renaming one
                  silently drops durable state across a daemon restart.
  handoff.py      the already-pure handoff builders/accruers for both lanes,
                  park-bid, prefence — typed as they move.
  replacement.py  eviction replace, understudy, the rebid ladder, boot SLA,
                  pull condemn/watchdog (clusters C24 + C27).
  run_lane.py     `supervise_init` / `tick` / `finalize` + `RunLaneFloorHooks`.
  job_lane.py     `job_supervise_init` / `tick` + `JobLaneFloorHooks`.
  retention.py    the retain-or-destroy sweep.

What is deliberately NOT here
-----------------------------
* **Lane unification.** The run lane and the jobs lane stay MIRRORED. The six
  pinned divergences are deliberate (v1 §7 / `FLEET_REVIEW_2026-08-14.md`
  item 1); a money-path unification is its own owner-called change, not a
  side effect of moving files. Port the mirror as a mirror.
* No bid state-machine of its own — `ladder_core.py` is the one copy of the
  per-tick bookkeeping and it stays that way (it exists precisely because this
  code had two hand-written copies).
* No daemon. The fleetd tick loop and its unix socket are `fleet/`; this ring
  supplies the policy the daemon drives, and it must stay callable from a
  plain CLI invocation with no daemon present.

Provenance: skeleton created 2026-08-16, plan §8 step 1. `journal.py` arrives
in step 2; the rest in step 4, as one unit.
"""

from __future__ import annotations
