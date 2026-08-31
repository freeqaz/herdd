"""vastlib.storage.b2 — the rclone/B2 seam: five functions, one subprocess.

Why this module exists
----------------------
Every durable byte this tooling owns leaves the workstation through `rclone`.
Park a box, tear one down, write a run marker, check whether a handoff object
landed — all of it ends at one of the five functions below, and the mapping
pass counted 51 inbound calls plus 47 `monkeypatch.setattr` sites riding on
their exact return shapes. Pulling them out of `herdd.py` gives that seam a
name and a single import path, so a caller ported in a later step patches
`vastlib.storage.b2.<fn>` instead of reaching into a 20k-line script.

The shapes, verbatim (plan §5 / `core.result`)
----------------------------------------------
* `_rclone_soft` is the **primitive** and `_rclone` is its raising-ish wrapper —
  the inverse of the usual raising/soft pair, and worth knowing before
  "simplifying" either. `_rclone_soft` returns the POSIX triple
  `(rc, stdout, stderr)` (`core.result.ProcResult`, shape E): `rc == 0` is the
  success test, and `stderr` is routinely non-empty on a *successful* run.
* `_rclone` drops `stderr`, returning `(rc, stdout)`, and `sys.exit`s only for
  the "rclone is not installed" case (`rc == 127` *and* `"not found"` in the
  error text).
* `_b2_rcat` returns a bare bool and **does not route through `_rclone_soft`**
  — it owns its own `subprocess.run` because it feeds the object body on stdin.
  Patching the rclone seam therefore leaves every rcat write LIVE; that is the
  one write path in here that can reach real B2 from an unguarded test, and it
  is why `test_vastlib_storage.py` stubs this module's `subprocess` attribute
  rather than the seam.
* `_b2_lsf_present` is a bool over `_rclone_soft` (so it *is* steerable by a
  seam patch).

Two name twins that a bare-name rename table will get wrong
-----------------------------------------------------------
`_rclone` exists three more times in `tools/vast`, and only one of them is this
function:

* **`ckpt_retention._rclone(args, runner=None)`** returns the 3-tuple
  `(rc, out, err)` and takes an injectable runner; **`herdd._rclone(args)`**
  (the one ported here) returns the 2-tuple `(rc, out)` and `sys.exit`s on a
  missing binary. Different arity, different failure mode, same name. The three
  patch sites at `test_ckpt_retention.py:549,559,568` target the *ckpt_retention*
  one through the alias `R` and must not be repointed here. **Key the plan §7.1
  rename table on `herdd.<attr>`, never on the bare attribute name.**
* `hostfacts.py:213` holds a fourth copy (a staticmethod inside its B2 client)
  and `boxstate.py:128` a fifth (the degraded fallback in its
  `if not _HAVE_HERDD` branch). Neither is a port of this seam; neither is
  touched by this module.

External reach-ins, recorded and deliberately untouched
--------------------------------------------------------
Four live callers reach `herdd._rclone_soft` as a module attribute today:
`fleetd.py:1152` (`Hooks.jobd_status`), `boxstate.py:81`, and
`parked_lifecycle.py:708` and `:849` (both `lambda args:
herdd._rclone_soft(args)[:2]`, truncating the triple to a pair). This step is
ADD-ONLY, so all four keep working against the flat copy. At step 6 the thin
`herdd.py` launcher MUST keep `_rclone_soft` bound as a module attribute or
`boxstate` silently drops to its degraded copy — better, repoint all four at
this module in the same branch.

What is deliberately NOT here
-----------------------------
* **No new transport abstraction, and no merge with `supervise.journal`.** The
  run/box event writers own their own rcat inside `runmeta.emit_event` /
  `jobmeta.emit_box_event` (Zone S), with their own `runner` seam and the
  `b2:` vs `b2w:` split-key choice. Those and this module happen to both end at
  `rclone`; the shapes differ (a bucket-aware envelope writer versus a bare
  argv), and unifying them would put a second implementation of a wire contract
  in the tree.
* **No `_b2_write_soft`.** `test_handoff_drain_abort.py:370` patches that name
  with `raising=False` and no such definition exists anywhere — a vacuous patch
  and a standing latent defect (plan §1). Creating one here to make the patch
  land would convert a detectable defect into a silent one; §7.3's meta-test
  exists to catch exactly this.
* **No `_farm_status_by_run` / `_ckpt_steps_by_run` / `_train_summary_step`.**
  They sit textually next to the seam in `herdd.py` and take
  `runner=_rclone_soft` as a **default argument** — bound at def time, so a
  module-attribute patch does not steer them and tests pass `runner=`
  explicitly. They belong to the `cmd_runs` status fold, not to storage; keep
  that calling convention when they move.
* **No bucket, key or path policy.** This module moves bytes at whatever path
  it is handed. `B2_BUCKET` and the key split live in `runmeta`/`jobmeta`;
  credential *minting* is `launch.spec` and credential *discovery* is
  `core.config`.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 3,
2026-08-16), rev 2b188979. Behavior-preserving: bodies copied, annotations
added, plus the two documented mechanical changes — `ProcResult` in place of
the bare 3-tuple literal (a `NamedTuple`, so `==`, unpacking, indexing and
slicing against the old tuple all still hold), and `TOOLS_VAST_DIR` in place of
`os.path.dirname(os.path.abspath(__file__))`, which moved two directories
deeper with the file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

from vastlib.core import result

__all__ = [
    "TOOLS_VAST_DIR",
    "_b2_lsf_present",
    "_b2_rcat",
    "_ensure_b2_remote",
    "_rclone",
    "_rclone_soft",
]

# `tools/vast/` — the directory the flat `herdd.py` lived in, and the one
# `b2_sync.sh` still lives in. In the flat module this was
# `os.path.dirname(os.path.abspath(__file__))`; this file sits two directories
# deeper, so the same path has to be walked back up. Computed exactly the way
# `core.config._HERE` is, and pinned by
# `test_vastlib_storage.py::test_tools_vast_dir_resolves_to_the_real_tools_vast`
# — the failure mode is SILENT (the `b2_sync.sh` shell-out below ignores its rc,
# so a wrong path is a no-op remote that is never configured, and every later
# `b2:` operation fails somewhere else entirely).
TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# moved-from: herdd._rclone_soft
def _rclone_soft(args: Sequence[str]) -> result.ProcResult:
    """rclone that never sys.exits: (rc, stdout, stderr). rc=127 when rclone is
    absent (mirrors runmeta._default_runner). For the supervisor loop."""
    try:
        r = subprocess.run(["rclone", *args], capture_output=True, text=True)
        return result.ProcResult(r.returncode, r.stdout, r.stderr)
    except FileNotFoundError:
        return result.ProcResult(127, "", "rclone not found on PATH")


# moved-from: herdd._rclone
def _rclone(args: Sequence[str]) -> tuple[int, str]:
    """Run rclone, return (rc, stdout). stderr is swallowed (status view)."""
    rc, out, err = _rclone_soft(args)
    if rc == 127 and "not found" in (err or ""):
        sys.exit("error: rclone not found on PATH (needed to read B2 run state)")
    return rc, out


# moved-from: herdd._ensure_b2_remote
def _ensure_b2_remote() -> None:
    """Configure the b2: rclone remote from env (idempotent) via b2_sync.sh."""
    rc, out = _rclone(["listremotes"])
    if "b2:" in (out or ""):
        return
    script = os.path.join(TOOLS_VAST_DIR, "b2_sync.sh")
    subprocess.run(["bash", script, "config"], capture_output=True, text=True)


# moved-from: herdd._b2_rcat
def _b2_rcat(path: str, body: str, hard: bool = True) -> bool:
    """rclone rcat: write `body` to a B2 object via stdin. hard=True exits on
    failure (proves the B2 creds work before any money is spent); hard=False is
    best-effort. Never prints the body."""
    r = subprocess.run(["rclone", "rcat", path], input=body, text=True,
                       capture_output=True)
    if r.returncode != 0 and hard:
        sys.exit(f"error: failed to write B2 marker {path}: {(r.stderr or '').strip()}")
    return r.returncode == 0


# moved-from: herdd._b2_lsf_present
def _b2_lsf_present(path: str) -> bool:
    """True when `rclone lsf PATH` lists a non-empty result (object exists)."""
    rc, out, _ = _rclone_soft(["lsf", path])
    return rc == 0 and bool((out or "").strip())
