#!/usr/bin/env python3
"""GRADE-B model-dir identity: a per-file CONTENT manifest and a rollup sha.

The complement to `merged_fingerprint.py`, and the distinction is the whole
point of having both:

    Grade A  merged_fingerprint   sorted NAME/SIZE list       a dir this box MERGED
    Grade B  here                 sha256 of every file        a dir this box PULLED

A freshly merged dir is NOT bit-reproducible across hosts (CPU bf16
`merge_and_unload`), so grade B would fail honest boxes and must not gate it. A
RESTORED dir is a byte-for-byte copy of an already-published artifact, so the
bytes ARE comparable — and there grade A is too weak: it passes a dir whose
shards are the right shape and the wrong weights, which is exactly the
wrong-checkpoint failure (a pull that lands the previous merge, a sibling arm,
or a half-synced dir produces a server that boots, answers, and scores like a
baseline).

THE ROLLUP is one hex string, because a job-submit `--env` pin has to be one
value. sha256 over the canonical table

    "<relpath>\\0<size>\\0<sha256-of-file>\\n"   sorted by relpath, joined

so it is stable across machines, insensitive to mtime and mode, and covers the
file NAMES and the file SET as well as the content.

RECURSIVE AND SYMLINK-FOLLOWING: the HF blob layout points every file at
`blobs/<sha>`, so a manifest that stat'd the link would record the link size.

    dirhash.py <dir> [--ignore NAME ...]        # per-file manifest, JSON
    dirhash.py <dir> --rollup                   # the one-line pin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: Transport artefacts, not snapshot content: jobd's asset byte-total marker and
#: the publisher's completion receipt (written AFTER the content it attests to,
#: so it can never be inside a manifest of that content).
DEFAULT_IGNORE = (".complete", "PUSHED.json")

CHUNK = 1 << 22


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def manifest(d: Path | str, ignore=DEFAULT_IGNORE) -> dict[str, dict]:
    """Recursive {relpath: {size, sha256}}, symlinks resolved."""
    d = Path(d)
    skip = set(ignore)
    out: dict[str, dict] = {}
    for p in sorted(d.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(d).as_posix()
        if p.name in skip or rel in skip:
            continue
        rp = p.resolve()
        out[rel] = {"size": rp.stat().st_size, "sha256": sha256_file(rp)}
    return out


def rollup(man: dict[str, dict]) -> str:
    """The one-value pin over a manifest. See the module docstring."""
    h = hashlib.sha256()
    for rel in sorted(man):
        e = man[rel]
        h.update(f"{rel}\0{e['size']}\0{e['sha256']}\n".encode())
    return h.hexdigest()


def compare_manifest(frozen: dict[str, dict], got: dict[str, dict],
                     *, limit: int = 12) -> list[str]:
    """Problems between a frozen manifest and a live one. Pure.

    SIZE is reported instead of SHA256 when both differ: the size is the
    actionable half of a truncated-transfer report, and printing both for every
    shard of a 52 GiB model buries it.
    """
    bad: list[str] = []

    def _trunc(names: list[str]) -> str:
        return f"{names[:limit]}{' …' if len(names) > limit else ''}"

    miss = sorted(set(frozen) - set(got))
    extra = sorted(set(got) - set(frozen))
    if miss:
        bad.append(f"MISSING vs frozen: {_trunc(miss)}")
    if extra:
        bad.append(f"UNEXPECTED vs frozen: {_trunc(extra)}")
    for rel in sorted(set(frozen) & set(got)):
        w, g = frozen[rel], got[rel]
        if w.get("size") != g["size"]:
            bad.append(f"SIZE {rel}: {g['size']} != {w.get('size')}")
        elif w.get("sha256") != g["sha256"]:
            bad.append(f"SHA256 {rel}: {g['sha256']} != {w.get('sha256')}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE))
    ap.add_argument("--rollup", action="store_true",
                    help="print only the rollup sha256 pin")
    a = ap.parse_args(argv)
    if not Path(a.dir).is_dir():
        print(f"!! {a.dir}: not a directory", file=sys.stderr)
        return 2
    man = manifest(a.dir, tuple(a.ignore))
    if a.rollup:
        print(rollup(man))
    else:
        print(json.dumps(man, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
