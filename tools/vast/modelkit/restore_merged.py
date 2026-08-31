#!/usr/bin/env python3
"""Decide, for THIS box, whether a merged dir is reused, restored, or rebuilt.

WHY A B2 ROUND TRIP EXISTS. Merging a 27B-class model costs a ~52 GiB base pull
plus a CPU-side merge that materialises the model in bf16 in host RAM, and it
must happen before a single token is generated. On spot that setup is paid AGAIN
on every eviction: one arm took four evictions into a livelock where setup cost
approximated box lifetime and a full cycle banked ZERO net rows. Pushing the
merged dir to B2 once turns the second and every later box's merge into a
download.

WHY THE RE-VERIFY IS NOT OPTIONAL, AND IS THE POINT OF THIS FILE. A restored dir
has no merge process behind it, so it arrives with no guard evidence of its own —
and the failure this whole lane is built around is SILENT: substituting 1 of N
tensors yields a "merged" model that loads, serves, answers normally, and SCORES
AS THE BASE. A truncated or half-pushed restore has exactly that shape. So every
reuse — local or restored — goes through the SAME verifier, and a dir that fails
it is deleted rather than repaired: a partially-correct model dir is worse than
no model dir, because the fallback (pull + merge) is merely slow while the
alternative is a wrong number nobody can see.

`decide()` is a PURE function of the outcomes and `resolve()` is the executor
that calls it with real effects injected. That split is deliberate: the property
that must hold — *no `reuse` verdict is ever returned without a verification of
that same directory having just succeeded* — is a property of the control flow,
and control flow embedded in shell `if` statements cannot be tested. Here it is
enumerable over every combination of outcomes.

HOISTED, WITH THE VERIFIER MADE A PARAMETER. The bundle copies differ from each
other in exactly one token — `import merge_guard_27b` vs `import merge_guard_g4`
— which is why there were two of them. `main` now takes `--family` and drives
`merge_guard`'s spec-driven verifier; `resolve` never knew which guard it was
calling and still does not.

    restore_merged.py --merged DIR --remote checkpoints/... --transport SH
                      --family NAME [--base-pins F] [--fingerprint F]
                      [--report F] [--action-out F]
prints the action on stdout: `reuse-local` | `reuse-restored` | `merge`.
exit 0 always (an unusable dir is a decision, not an error); 2 on usage.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Callable, NamedTuple

REUSE_LOCAL = "reuse-local"
REUSE_RESTORED = "reuse-restored"
MERGE = "merge"
REUSE_ACTIONS = (REUSE_LOCAL, REUSE_RESTORED)


class Decision(NamedTuple):
    action: str
    reason: str
    verified: bool          # did a guard re-verify of THIS dir just pass?


def decide(*, local_present: bool, local_ok: bool | None,
           remote_present: bool, pull_ok: bool | None,
           restored_ok: bool | None) -> Decision:
    """Pure. `*_ok` are None when that step was never reached.

    The ONLY two ways out with a `reuse` action are through an `ok is True`, and
    both set `verified=True`. There is deliberately no `assume`, no `--force`,
    and no "it was verified on the box that pushed it" branch: the push happened
    on a different box, possibly a different CPU, and a verification that ran
    somewhere else is not a verification of these bytes.

    A LOCAL DIR THAT FAILS RE-VERIFY FALLS THROUGH TO THE RESTORE, it does not
    short-circuit to `merge`. That distinction was a bug in the ancestor until
    an enumerating test found it: `resolve` correctly purged the bad local dir
    and went on to pull a good one from B2, and this function then reported
    `merge` anyway — throwing away a completed 52 GiB restore and re-paying the
    whole base-pull-plus-merge. The failure was expensive rather than wrong,
    which is exactly the kind that survives review.
    """
    if local_present and local_ok:
        return Decision(REUSE_LOCAL,
                        "merged dir already on this box and re-verified", True)
    if remote_present:
        if not pull_ok:
            return Decision(MERGE,
                            "B2 held a published merged dir but the pull did "
                            "not complete — purged the partial; full merge",
                            False)
        if restored_ok:
            return Decision(REUSE_RESTORED,
                            "merged dir restored from B2 and re-verified here",
                            True)
        return Decision(MERGE,
                        "restored merged dir FAILED guard re-verify — purged; "
                        "a wrong-but-loadable model dir is the one failure "
                        "mode that scores as the base in silence",
                        False)
    if local_present:
        return Decision(MERGE,
                        "merged dir on disk FAILED guard re-verify — purged, "
                        "and B2 holds no published replacement; falling back "
                        "to a full base pull + merge",
                        False)
    return Decision(MERGE, "no merged dir on this box and none published to B2",
                    False)


def resolve(*, merged_dir: str, remote: str,
            verify: Callable[[str], bool],
            remote_has: Callable[[str], bool],
            pull: Callable[[str, str], bool],
            purge: Callable[[str], None],
            exists: Callable[[str], bool] = os.path.isdir,
            log: Callable[[str], None] = print) -> Decision:
    """Executor. Every effect is injected so the ordering is testable."""
    local_present = exists(merged_dir)
    local_ok = None
    if local_present:
        log(f">> merged dir present at {merged_dir} — re-verifying "
            f"(never reused on presence alone)")
        local_ok = bool(verify(merged_dir))
        if not local_ok:
            log(">> local merged dir FAILED re-verify — purging")
            purge(merged_dir)

    remote_present = pull_ok = restored_ok = None
    if not (local_present and local_ok):
        remote_present = bool(remote_has(remote))
        if remote_present:
            log(f">> B2 holds a published merged dir at {remote} — restoring")
            pull_ok = bool(pull(remote, merged_dir))
            if pull_ok:
                log(">> restored — re-verifying (a restore is NOT evidence)")
                restored_ok = bool(verify(merged_dir))
                if not restored_ok:
                    log(">> restored merged dir FAILED re-verify — purging")
                    purge(merged_dir)
            else:
                log(">> restore pull did not complete — purging the partial")
                purge(merged_dir)

    d = decide(local_present=local_present, local_ok=local_ok,
               remote_present=bool(remote_present), pull_ok=pull_ok,
               restored_ok=restored_ok)
    log(f">> decision: {d.action} — {d.reason}")
    return d


# --- real effects ------------------------------------------------------------
def _transport(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", script, *args], text=True)


def guard_verifier(spec, *, base_pins=None, frozen_fingerprint=None,
                   reports: list | None = None,
                   log: Callable[[str], None] | None = None
                   ) -> Callable[[str], bool]:
    """A `verify` callable over `merge_guard.verify_merged_dir` for one family.

    Kept a factory rather than a hardcoded import so a caller can inject any
    verifier with the same one-argument shape — which is what made the two
    bundle copies of this file differ in a single token, and is the coupling
    this hoist removes.
    """
    try:                                         # pragma: no cover - trivial
        from . import merge_guard
    except ImportError:                          # pragma: no cover - trivial
        import merge_guard                       # type: ignore[no-redef]

    emit = log or (lambda m: print(m, file=sys.stderr))

    def verify(d: str) -> bool:
        problems, rep = merge_guard.verify_merged_dir(
            d, spec, base_pins=base_pins, frozen_fingerprint=frozen_fingerprint)
        if reports is not None:
            reports.append(rep)
        for p in problems:
            emit(f"   !! {p}")
        return not problems

    return verify


def main(argv=None) -> int:
    try:                                         # pragma: no cover - trivial
        from . import merge_guard
    except ImportError:                          # pragma: no cover - trivial
        import merge_guard                       # type: ignore[no-redef]

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merged", required=True)
    ap.add_argument("--remote", required=True,
                    help="B2 key prefix, e.g. checkpoints/<name>-merged/<sha12>/model")
    ap.add_argument("--transport", required=True,
                    help="b2_transport.sh — `has` / `pull` / `push`")
    ap.add_argument("--family", required=True,
                    help="a modelkit family_specs/ name, or a path to a spec JSON")
    ap.add_argument("--base-pins")
    ap.add_argument("--fingerprint",
                    help="a merged-dir fingerprint published alongside the "
                         "checkpoint. Optional, and it can only ever ADD "
                         "checks — compare_fingerprint appends problems and "
                         "removes none — so a stale or absent one weakens "
                         "nothing.")
    ap.add_argument("--report")
    ap.add_argument("--action-out",
                    help="write the bare action here. The caller reads THIS, "
                         "not the tail of a merged stream: a `2>&1 | tail -1` "
                         "is correct only because CPython block-buffers stdout "
                         "to a pipe and flushes it after the unbuffered stderr "
                         "log. That is a property of the interpreter's "
                         "buffering, not of this program, and the thing it "
                         "decides is whether a 52 GiB model dir is trusted.")
    a = ap.parse_args(argv)

    try:
        spec = merge_guard.load_spec(a.family)
    except merge_guard.SpecError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2

    pins = json.loads(open(a.base_pins).read()) if a.base_pins else None
    fp = (json.loads(open(a.fingerprint).read())
          if a.fingerprint and os.path.isfile(a.fingerprint) else None)
    reports: list[dict] = []

    d = resolve(
        merged_dir=a.merged, remote=a.remote,
        verify=guard_verifier(spec, base_pins=pins, frozen_fingerprint=fp,
                              reports=reports),
        remote_has=lambda r: _transport(a.transport, "has", r).returncode == 0,
        pull=lambda r, dest: _transport(a.transport, "pull", r, dest).returncode == 0,
        purge=lambda p: shutil.rmtree(p, ignore_errors=True),
        log=lambda m: print(m, file=sys.stderr))

    if a.report:
        with open(a.report, "w") as fh:
            json.dump({"action": d.action, "reason": d.reason,
                       "verified": d.verified, "merged_dir": a.merged,
                       "remote": a.remote, "family": spec["family"],
                       "verify_reports": reports}, fh, indent=1, sort_keys=True)
    if a.action_out:
        with open(a.action_out, "w") as fh:
            fh.write(d.action + "\n")
    print(d.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
