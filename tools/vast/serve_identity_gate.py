#!/usr/bin/env python3
"""ON-BOX identity gate: the directory about to be served IS the artifact asked for.

Runs on the rented box, after the pull, before any `vllm serve` argv exists.
`onstart/serve_vllm.sh` stages it here beside `merged_fingerprint.py` — neither
can ride the 16 KiB onstart wire, so both come down the same per-serve B2 prefix
as `parse_vllm_mem.py`.

THE FAILURE IT REFUSES IS INVISIBLE EVERYWHERE ELSE. On 2026-08-21 a stale
`MODEL_B2` inherited through `/etc/environment` made a box serve the wrong
weights while `/v1/models` named the model that was asked for and
`serve_ready.sh` passed. Every name-level check agreed with every other
name-level check, because they were all reading the same label. Only a claim
about BYTES, made on the box that holds them, can separate those.

TWO INPUTS, AND THEY COME FROM DIFFERENT PLACES ON PURPOSE:

    the DIRECTORY   pulled from B2 by this box, moments ago
    the EXPECTATION composed on the WORKSTATION from the committed registry

If the expectation were read from the guards published beside the weights, B2
would be corroborating B2 and a re-published or renamed prefix would agree with
itself. `serve_artifact.py expect` is the composer, and it never touches B2.

NO SKIP PATH LIVES HERE. Calling this script means an expectation was shipped;
deciding that none was is the caller's job, and it is one loud line there. A
gate with its own skip branch is a gate that can be talked out of firing.

GRADE A vs GRADE B (MERGED_MODEL_ARTIFACTS.md §3). Grade A is the sorted
NAME/SIZE list — the only claim that is true across hosts for a freshly merged
dir. Grade B is the per-file sha256 rollup, which is what a RESTORE should be
gated on, because a restored dir is supposed to be a byte-for-byte copy. A
seed whose `content_sha256` is null (= UNMEASURED, never "clean") degrades
this to grade A with a distinct line naming what that leaves open — a
same-size content swap.

    serve_identity_gate.py --dir DIR --expect EXPECT.json
                           [--fingerprint-tool PATH] [--dirhash-tool PATH]
                           [--out REPORT.json]

Last stdout line is the verdict token, so a shell can read it without JSON:
`IDENTITY_VERIFIED <grade> <sha12> <sha256>` · `IDENTITY_MISMATCH` ·
`IDENTITY_CANNOT_CHECK`. Both shas are on the line because the two consumers
want different ones — the SERVE_STATUS marker is a single short line and takes
the 12, `serve_summary.json` records the full digest — and one `awk` beats a
second interpreter start on a box mid-boot.
Both refusals are refusals; they are separate tokens because the remedies are
opposite — a mismatch means the WEIGHTS are wrong, a cannot-check means the
GATE is broken, and a box that reports the second as the first sends an
operator hunting a model defect that does not exist.
exit 0 verified · 1 mismatch · 2 cannot check (FAIL CLOSED).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

VERDICT_OK = "IDENTITY_VERIFIED"
VERDICT_BAD = "IDENTITY_MISMATCH"
VERDICT_CANNOT = "IDENTITY_CANNOT_CHECK"

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CANNOT_CHECK = 2

SHA12 = 12

#: Where a staged tool can be, most-specific first. The B2 boot-pull lane ships
#: no repo checkout, so `/workspace` is where `serve_vllm.sh` puts what it
#: pulled; a dev box running this from the repo finds `modelkit/` instead.
def _tool_candidates(name: str) -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, name),
        os.path.join(here, "modelkit", name),
        f"/workspace/{name}",
        f"/workspace/eval/upstream-monorepo/tools/vast/modelkit/{name}",
    ]


def load_tool(name: str, explicit: str | None = None):
    """Import a staged single-file module BY PATH, or return None.

    By path and not by package: `merged_fingerprint.py` is self-contained by
    contract precisely so it can be staged alone, and an `import modelkit.…`
    here would reintroduce the package dependency the contract removes — a gate
    that cannot import is discovered at the moment it was supposed to refuse.
    """
    for cand in ([explicit] if explicit else []) + _tool_candidates(name):
        if not cand or not os.path.isfile(cand):
            continue
        spec = importlib.util.spec_from_file_location(name[:-3], cand)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:                             # noqa: BLE001
            continue
        mod.__gate_path__ = cand                      # type: ignore[attr-defined]
        return mod
    return None


def validate_expectation(exp: object) -> list[str]:
    """Shape problems with a frozen expectation. Pure; empty means usable.

    Unusable is CANNOT CHECK, not "assume fine": an expectation that fails to
    parse is the shape a truncated B2 pull of the expectation itself takes, and
    that must not read as a pass.
    """
    bad: list[str] = []
    if not isinstance(exp, dict):
        return [f"not a JSON object ({type(exp).__name__})"]
    if exp.get("schema_version") != 1:
        bad.append(f"schema_version {exp.get('schema_version')!r} != 1")
    grade = exp.get("grade")
    if grade not in ("A", "B"):
        bad.append(f"grade {grade!r} not in ('A', 'B') — an expectation with no "
                   f"grade pins nothing")
    if not isinstance(exp.get("fingerprint_sha256"), str):
        bad.append("fingerprint_sha256 missing — grade A is the floor, not an "
                   "option")
    if not isinstance(exp.get("n_files"), int) or isinstance(
            exp.get("n_files"), bool):
        bad.append(f"n_files {exp.get('n_files')!r} is not an integer")
    if grade == "B" and not isinstance(exp.get("content_sha256"), str):
        bad.append(f"grade B declared but content_sha256 is "
                   f"{exp.get('content_sha256')!r}")
    return bad


def compare_grade_a(exp: dict, fp: dict) -> list[str]:
    """Problems between the expectation and a live grade-A fingerprint. Pure.

    The sha is compared, not the file list, because the registry pins the sha —
    and the sha IS the sorted name/size list, so agreeing on it is agreeing on
    every name and every size. `n_files` is checked separately only to make the
    common failure (a short restore) say so in one line instead of "the digests
    differ".
    """
    bad: list[str] = []
    if fp.get("n_files") != exp.get("n_files"):
        bad.append(f"n_files: this dir holds {fp.get('n_files')}, the registry "
                   f"pins {exp.get('n_files')} — a short or fat restore")
    if fp.get("sha256") != exp.get("fingerprint_sha256"):
        bad.append(
            f"fingerprint_sha256: {str(fp.get('sha256'))[:SHA12]}… != pinned "
            f"{str(exp.get('fingerprint_sha256'))[:SHA12]}… — the file set or "
            f"the sizes differ from the artifact this serve was launched for")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the model dir about to serve")
    ap.add_argument("--expect", required=True,
                    help="the frozen expectation JSON (serve_artifact.py expect)")
    ap.add_argument("--fingerprint-tool", help="path to merged_fingerprint.py")
    ap.add_argument("--dirhash-tool", help="path to dirhash.py (grade B only)")
    ap.add_argument("--out", help="write the full verdict report here")
    a = ap.parse_args(argv)

    report: dict = {"dir": a.dir, "expect": a.expect}

    def finish(rc: int, problems: list[str], **extra) -> int:
        report["problems"] = problems
        report["ok"] = rc == EXIT_OK
        report.update(extra)
        if a.out:
            try:
                with open(a.out, "w") as fh:
                    json.dump(report, fh, indent=1, sort_keys=True)
            except OSError as e:                      # noqa: BLE001
                print(f"!! identity gate: report write failed ({e!r})",
                      file=sys.stderr)
        for p in problems:
            print(f"!!   {p}", file=sys.stderr)
        if rc == EXIT_OK:
            return rc
        print(f"!! Refusing {a.dir}: a model dir that is not the artifact this "
              f"serve was launched for boots, answers, and scores like the "
              f"baseline. Nothing downstream can see it.", file=sys.stderr)
        print(VERDICT_BAD if rc == EXIT_MISMATCH else VERDICT_CANNOT)
        return rc

    try:
        exp = json.loads(open(a.expect).read())
    except Exception as e:                            # noqa: BLE001
        return finish(EXIT_CANNOT_CHECK,
                      [f"--expect {a.expect}: unreadable/unparseable ({e!r})"])
    problems = validate_expectation(exp)
    if problems:
        return finish(EXIT_CANNOT_CHECK,
                      [f"--expect {a.expect}: {p}" for p in problems])
    report["artifact"] = exp.get("artifact")
    report["grade_expected"] = exp["grade"]

    if not os.path.isdir(a.dir):
        return finish(EXIT_CANNOT_CHECK, [f"{a.dir}: not a directory"])

    mf = load_tool("merged_fingerprint.py", a.fingerprint_tool)
    if mf is None:
        return finish(EXIT_CANNOT_CHECK, [
            "merged_fingerprint.py not on this box — the gate could not run, "
            "which is a REFUSAL and not a skip (stage it beside this script or "
            "pass --fingerprint-tool)"])
    report["fingerprint_tool"] = mf.__gate_path__

    fp = mf.fingerprint_dir(a.dir, exclude=mf.DEFAULT_EXCLUDE)
    report["fingerprint"] = {"n_files": fp["n_files"], "sha256": fp["sha256"]}
    problems = compare_grade_a(exp, fp)

    # The publisher's own count, when the pull brought its receipt along. Pure
    # corroboration — it was written by the box that pushed, so it can only ever
    # CONTRADICT the registry, never confirm the bytes.
    rec, why = mf.read_pushed_receipt(a.dir)
    if rec is not None:
        rec_bad = mf.corroborate_receipt(rec, fp)
        report["receipt"] = rec
        problems += rec_bad
    else:
        report["receipt"] = None
        report["receipt_note"] = why

    grade_verified = "A"
    if exp["grade"] == "B":
        dh = load_tool("dirhash.py", a.dirhash_tool)
        if dh is None:
            return finish(EXIT_CANNOT_CHECK, problems + [
                "grade B is pinned for this artifact but dirhash.py is not on "
                "this box — the stronger claim was requested and cannot be "
                "made, so this is a REFUSAL"])
        report["dirhash_tool"] = dh.__gate_path__
        got = dh.rollup(dh.manifest(a.dir))
        report["content_sha256"] = got
        if got != exp["content_sha256"]:
            problems.append(
                f"content_sha256: {got[:SHA12]}… != pinned "
                f"{str(exp['content_sha256'])[:SHA12]}… — same file set, "
                f"different BYTES. This is the swap grade A is blind to")
        else:
            grade_verified = "B"
    else:
        # Distinct, loud, and never suppressed: the operator must be able to
        # tell "checked the bytes" from "checked the shape" in a serve log.
        print(f">> identity gate: GRADE A ONLY for artifact "
              f"{exp.get('artifact')!r} — content_sha256 is null in the "
              f"registry (UNMEASURED, not clean). A same-size content swap is "
              f"INVISIBLE to this gate.", file=sys.stderr)
        print(f">>   mint the grade-B pin ($0, on a box that already holds a "
              f"verified copy): python3 modelkit/gate_dir.py --dir {a.dir} "
              f"--emit", file=sys.stderr)

    if problems:
        return finish(EXIT_MISMATCH, problems, grade_verified=None)

    sha12 = fp["sha256"][:SHA12]
    report["ident_sha256"] = fp["sha256"]
    report["ident_grade"] = grade_verified
    finish(EXIT_OK, [], grade_verified=grade_verified)
    print(f">> {a.dir}: {fp['n_files']} files, grade-{grade_verified} identity "
          f"{sha12}… matches artifact {exp.get('artifact')!r}")
    print(f"{VERDICT_OK} {grade_verified} {sha12} {fp['sha256']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
