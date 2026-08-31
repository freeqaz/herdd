#!/usr/bin/env python3
"""salvage — DEPRECATION SHIM. The code now lives in `vastlib.boxes.salvage`.

Why this file still exists
--------------------------
Plan §3 of docs/plans/vast-tooling-refactor-v2.md: an absorbed flat sibling does
not disappear the day its body moves into the package — "their flat files become
deprecation shims for one release, then deleted". Two callers still spell the
bare name and both are load-bearing:

  * `tools/vast/herdd.py` does `import salvage` for the MODULE OBJECT (not for
    a name inside it), so `herdd.salvage.<anything>` keeps resolving;
  * `tools/vast/test_salvage.py` — the 62 KB owner suite — does
    `import salvage as S` and hands its NamedTuples into ported functions.

That second one is why this is a RE-EXPORT and never a copy: `S.LsEntry(...)`
constructed by the flat suite has to be the same class the ported code checks,
and `salvage.advance` has to be the same function object the ported ladder runs.
Every name below is bound to the object `vastlib.boxes.salvage` owns; there is
exactly one implementation in the process.

What is deliberately NOT here
-----------------------------
  * NO function or class body. Not one. A shim that redefines anything forks the
    implementation, which is the failure mode the shim exists to prevent.
  * NO `sys.modules['salvage'] = vastlib.boxes.salvage` alias. Nothing
    path-loads an authored file that imports this module and nothing does an
    `isinstance` against a bare-name class, so the re-export is the whole
    contract; aliasing would silently turn `salvage` and the port into one
    object and convert test_vastlib_boxes_salvage.py's differential asserts into
    self-comparisons without anyone editing them.
  * NO `sys.path` bootstrap. salvage.py has no `__main__` path (the CLI is
    `herdd salvage`, not `python3 salvage.py`), so every importer already has
    tools/vast on the path.
  * NOT the ten `_salvage_*` / `_mk_salvage_*` helpers that `vastlib.boxes.salvage`
    also owns. Those came from herdd.py, never from this file. Re-exporting
    them here would invent a surface the flat module never offered and let a
    caller reach a herdd-provenance name through the salvage name. The list
    below is exactly the 51 names this file used to define.

Provenance: bodies moved to `vastlib/boxes/salvage.py` at plan step 3 (the
add-only phase kept both live as twins); this file became a shim at plan step 7.
Design/observation record is unchanged and still in the ported module's
docstring — including the 2026-08-05 live-box findings and the losslessness
argument for treating `copy_direct` success as non-evidence.
"""
from __future__ import annotations

from vastlib.boxes.salvage import (  # noqa: F401  (re-export surface, see __all__)
    _CKPT,
    _FATAL_SURVEY,
    _LS_ABSENT,
    _LS_LINE,
    DEST_FREE_MARGIN,
    DEST_READY_STATES,
    JOBS_ROOT,
    LOUD_OUTCOMES,
    OUTCOME_COPY_REFUSED,
    OUTCOME_DEAD_GONE,
    OUTCOME_DEST_NOT_READY,
    OUTCOME_DISABLED,
    OUTCOME_NOTHING_FOUND,
    OUTCOME_NOTHING_NEWER,
    OUTCOME_PARTIAL,
    OUTCOME_SALVAGED,
    OUTCOME_UNVERIFIABLE,
    SALVAGE_DEADLINE_S,
    SALVAGE_DEST_WAIT_S,
    SALVAGE_KEEP_N,
    SALVAGE_MAX_GB,
    SALVAGE_ROOT,
    TERMINAL_OUTCOMES,
    CkptDir,
    LsEntry,
    SalvagePlan,
    Step,
    Verification,
    _advance_copying,
    _advance_pending,
    _copy_status_soft,
    _finish,
    _strip_ls_date,
    advance,
    b2_salvage_prefix,
    checkpoint_step,
    ckpt_dirs_from_survey,
    dest_path,
    landed_path,
    new_record,
    parse_ls_l,
    parse_ls_l_strict,
    parse_ls_lr,
    parse_ls_lr_strict,
    pick_dest,
    plan_salvage,
    split_ckpt_rel,
    survey_dead_box,
    survey_dest_files,
    survey_is_fatal,
    verify_salvage,
)

#: The frozen shim surface: exactly the 51 top-level names tools/vast/salvage.py
#: defined before the port. Not a style flourish — it is what stops the export
#: list drifting toward `vastlib.boxes.salvage`'s superset, and what
#: test_vastlib_boxes_salvage.py's surface check reads.
__all__ = [
    "CkptDir",
    "DEST_FREE_MARGIN",
    "DEST_READY_STATES",
    "JOBS_ROOT",
    "LOUD_OUTCOMES",
    "LsEntry",
    "OUTCOME_COPY_REFUSED",
    "OUTCOME_DEAD_GONE",
    "OUTCOME_DEST_NOT_READY",
    "OUTCOME_DISABLED",
    "OUTCOME_NOTHING_FOUND",
    "OUTCOME_NOTHING_NEWER",
    "OUTCOME_PARTIAL",
    "OUTCOME_SALVAGED",
    "OUTCOME_UNVERIFIABLE",
    "SALVAGE_DEADLINE_S",
    "SALVAGE_DEST_WAIT_S",
    "SALVAGE_KEEP_N",
    "SALVAGE_MAX_GB",
    "SALVAGE_ROOT",
    "SalvagePlan",
    "Step",
    "TERMINAL_OUTCOMES",
    "Verification",
    "_CKPT",
    "_FATAL_SURVEY",
    "_LS_ABSENT",
    "_LS_LINE",
    "_advance_copying",
    "_advance_pending",
    "_copy_status_soft",
    "_finish",
    "_strip_ls_date",
    "advance",
    "b2_salvage_prefix",
    "checkpoint_step",
    "ckpt_dirs_from_survey",
    "dest_path",
    "landed_path",
    "new_record",
    "parse_ls_l",
    "parse_ls_l_strict",
    "parse_ls_lr",
    "parse_ls_lr_strict",
    "pick_dest",
    "plan_salvage",
    "split_ckpt_rel",
    "survey_dead_box",
    "survey_dest_files",
    "survey_is_fatal",
    "verify_salvage",
]
