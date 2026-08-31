#!/usr/bin/env python3
"""MODEL-DIR IDENTITY GATE (grade B, content) — fail-closed, before a token.

The CLI over `dirhash.py`. It refuses two failures that look identical from the
outside, both of which serve something that is not the model the run claims:

  1. **the wrong checkpoint.** A B2 prefix is mutable and a path is a string. A
     pull that lands the previous merge, a sibling arm, or a half-synced dir
     produces a server that boots, answers, and scores like a baseline.
  2. **a partially-wired model.** A merged dir short a shard, or the
     `.textonly` intermediate, has the same signature — and the merge guards
     cannot see it, because they only assert that the adapted tensors moved and
     the frozen ones did not.

Neither is visible in a score: both read as "the baseline reproduced". The only
thing separating them from a good run is the bytes, so this gate compares bytes
and nothing else. Use it on a dir that was PULLED; on a dir this box just
MERGED use `merged_fingerprint.py` instead (grade A — see its docstring for
why a bitwise claim there would be a lie).

FAIL CLOSED. Being unable to check is a FAILURE, never a pass: no `--expect-sha`
and no `--frozen`, an empty directory, an unreadable file, a `--frozen` that
does not parse. "Could not check" reading as "checked and fine" is the whole
failure class this exists to remove.

    gate_dir.py --dir DIR --expect-sha HEX [--frozen M.json] [--ignore NAME ...]
                [--out gates/model_identity.json] [--min-files N]
    gate_dir.py --dir DIR --emit            # mint a pin, no verdict

exit 0 identity confirmed · 1 MISMATCH · 2 cannot check (fail closed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Dual-mode: importable as `modelkit.gate_dir`, runnable as a bare script from a
# box directory that holds both files side by side. The serve lane stages this
# pair together; `merged_fingerprint.py` is the one file that travels alone.
try:                                             # pragma: no cover - trivial
    from . import dirhash
except ImportError:                              # pragma: no cover - trivial
    import dirhash                               # type: ignore[no-redef]


def _fail(msg: str, code: int) -> int:
    print(f"!! {msg}", file=sys.stderr)
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the model directory to gate")
    ap.add_argument("--expect-sha", default=None,
                    help="rollup sha256 the invocation pinned (see --emit)")
    ap.add_argument("--frozen", type=Path, default=None,
                    help="per-file {size,sha256} manifest for a precise diff")
    ap.add_argument("--ignore", nargs="*", default=list(dirhash.DEFAULT_IGNORE))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--emit", action="store_true",
                    help="print the manifest + rollup and exit 0; NO verdict")
    ap.add_argument("--min-files", type=int, default=2,
                    help="a dir with fewer files than this is a failed transfer")
    a = ap.parse_args(argv)

    d = Path(a.dir)
    if not d.is_dir():
        return _fail(f"model dir does not exist: {d}", 2)
    try:
        man = dirhash.manifest(d, tuple(a.ignore))
    except OSError as e:
        return _fail(f"cannot read {d}: {e!r} — refusing to treat an unreadable "
                     f"model dir as verified", 2)

    got = dirhash.rollup(man)
    total = sum(e["size"] for e in man.values())
    rep = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dir": str(d), "n_files": len(man), "total_bytes": total,
        "rollup_sha256": got, "expect_sha256": a.expect_sha, "files": man,
    }

    def _write() -> None:
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rep, indent=1, sort_keys=True))

    if a.emit:
        print(json.dumps(rep, indent=1, sort_keys=True))
        print(f"\n>> rollup sha256 = {got}   ({len(man)} files, {total} B)",
              flush=True)
        _write()
        return 0

    problems: list[str] = []
    if not man:
        problems.append(f"{d} holds NO files — the pull moved nothing "
                        f"(rclone exits 0 on a copy that transferred nothing)")
    elif len(man) < a.min_files:
        problems.append(f"{d} holds only {len(man)} file(s) (< --min-files "
                        f"{a.min_files}) — a truncated or interrupted transfer")

    if not a.expect_sha and not a.frozen:
        problems.append(
            "neither --expect-sha nor --frozen was given. This gate FAILS "
            "CLOSED: an unpinned model is exactly the wrong-checkpoint case it "
            "exists to catch. Mint the pin with --emit.")

    if a.expect_sha:
        want = a.expect_sha.strip().lower()
        if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
            problems.append(f"--expect-sha {a.expect_sha!r} is not a 64-hex "
                            f"sha256 — a malformed pin is not a pin")
        elif want != got:
            problems.append(f"ROLLUP MISMATCH: {got} != pinned {want}")

    if a.frozen:
        try:
            fro = json.loads(a.frozen.read_text())
        except Exception as e:                   # noqa: BLE001
            problems.append(f"--frozen {a.frozen} unreadable/unparseable: {e!r}")
            fro = None
        if isinstance(fro, dict):
            fro = fro.get("files", fro)          # accept a full gate report too
        if isinstance(fro, dict):
            problems += dirhash.compare_manifest(fro, man)
        elif fro is not None:
            problems.append(f"--frozen {a.frozen} is not a manifest mapping")

    rep["ok"] = not problems
    rep["problems"] = problems
    _write()

    if problems:
        print(f"!! MODEL IDENTITY GATE FAILED for {d}", file=sys.stderr)
        for p in problems:
            print(f"!!   {p}", file=sys.stderr)
        print("!! This is NOT the model the invocation pinned. Refusing: a wrong "
              "or short model serves cleanly and reads as 'the baseline "
              "reproduced'.", file=sys.stderr)
        return 2 if not a.expect_sha and not a.frozen else 1

    print(f">> MODEL IDENTITY OK: {d} — {len(man)} files, {total} B, "
          f"rollup sha256 {got}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
