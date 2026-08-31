#!/usr/bin/env python3
"""GRADE-A merged-dir fingerprint: the sorted NAME/SIZE list and a sha over it.

THE ONE IMPLEMENTATION. This rule shipped as `fingerprint_dir` in
`merge_guard_27b.py` and again in `merge_guard_g4.py`, and a third time as a
drifted REIMPLEMENTATION in a banked run's `gate_model_identity.py` — which
excluded `PUSHED.json` and read it back as a receipt, a behaviour the two
in-repo copies never had. Three copies of a hash that MUST agree is how a gate
stops gating: nothing errors, the digests simply stop meaning the same thing.

WHAT GRADE A CLAIMS, AND WHAT IT DOES NOT. It claims the FILE SET and each
file's SIZE. It does NOT claim content. That is deliberate and it is not a
weaker version of a content hash — it is the only claim that is TRUE across
hosts for this artifact: the merge runs `merge_and_unload()` in bf16 on the CPU
and a different host CPU may differ in the last bit, so a bitwise claim about a
freshly merged dir would be a lie that fails honest boxes. The weight-level
claim rides the marker's must-move / must-not-move guard deltas, which the
restorer re-checks independently.

    Grade A (here)          a dir the local box MERGED     name/size
    Grade B (dirhash.py)    a dir the local box PULLED     sha256 per file

Use grade B whenever the bytes are supposed to be a byte-for-byte copy of an
already-published artifact; a name/size fingerprint passes a dir whose shards
are the right shape and the wrong weights.

THE RECEIPT. `b2_transport.sh push` writes `PUSHED.json` LAST, after a
read-back, so its presence is what `has` reads to mean "this prefix holds a
COMPLETE publish". It carries `{"complete": true, "files": N, "ts_utc": …}`.
`read_pushed_receipt` + `corroborate_receipt` turn that into a second,
independent count of the payload — it is CORROBORATION, never authority: the
receipt was written by the pushing box and a restorer that trusted it would be
trusting a claim about bytes it has not looked at.

The receipt is written AFTER the fingerprint it attests to, so it can never be
inside it — hence `DEFAULT_EXCLUDE`. `fingerprint_dir`'s default is the empty
exclusion set, which is exactly the ancestor rule; the CLI opts in.

    merged_fingerprint.py --dir DIR [--frozen F] [--receipt] [--out F]
                          [--exclude NAME ...] [--emit]

prints `MERGED_VERIFIED` or `REFUSED`; exit 0 verified · 1 mismatch ·
2 cannot check (fail closed).

SELF-CONTAINED BY CONTRACT: no intra-package imports, no third-party imports.
The serve lane stages this file to a box on its own, as a standalone gate
script. A `from . import …` here would make that staging silently produce a
gate that cannot run — and a gate that cannot run is discovered at the moment
it was supposed to refuse something.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

#: Transport artefacts, not model content: the publisher's completion receipt
#: and jobd's asset byte-total marker. Opt-in — see the module docstring.
DEFAULT_EXCLUDE = ("PUSHED.json", ".complete")

RECEIPT_NAME = "PUSHED.json"

VERDICT_OK = "MERGED_VERIFIED"
VERDICT_REFUSED = "REFUSED"


def fingerprint_dir(d: str, *, exclude: tuple[str, ...] | frozenset[str] = ()) -> dict:
    """{n_files, files:[{f,size}], sha256} over the sorted NAME/SIZE list.

    `exclude` defaults to EMPTY so this is bit-for-bit the rule the bundle
    copies froze; a caller that wants the receipt kept out of its own
    fingerprint passes `exclude=DEFAULT_EXCLUDE`.
    """
    skip = set(exclude)
    files = sorted(
        ({"f": e.name, "size": e.stat().st_size}
         for e in os.scandir(d) if e.is_file() and e.name not in skip),
        key=lambda r: r["f"])
    blob = json.dumps(files, sort_keys=True).encode()
    return {"n_files": len(files), "files": files,
            "sha256": hashlib.sha256(blob).hexdigest()}


def compare_fingerprint(frozen: dict, got: dict, *,
                        size_exempt: tuple[str, ...] | frozenset[str] = ()) -> list[str]:
    """Problems between a recorded fingerprint and a live one. Pure.

    `size_exempt` names files whose SIZE is host-dependent by construction — the
    merge marker embeds an absolute adapter path and a timestamp — and it is an
    exemption from the SIZE comparison only: the file must still be present, and
    its CONTENT is the marker check's job.
    """
    bad: list[str] = []
    exempt = set(size_exempt)
    want = {r["f"]: r["size"] for r in frozen.get("files", [])}
    have = {r["f"]: r["size"] for r in got.get("files", [])}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    if missing:
        bad.append(f"MISSING files vs the recorded fingerprint: {missing}")
    if extra:
        bad.append(f"UNEXPECTED files vs the recorded fingerprint: {extra}")
    for f in sorted(set(want) & set(have)):
        if f in exempt:
            continue
        if want[f] != have[f]:
            bad.append(f"SIZE {f}: {have[f]} != recorded {want[f]}")
    return bad


# --- the publisher's receipt -------------------------------------------------
def read_pushed_receipt(path: str) -> tuple[dict | None, str | None]:
    """Read `PUSHED.json` from a dir (or a direct file path).

    Returns `(receipt, None)` or `(None, why-not)`. Absence is NOT an error
    here: a freshly MERGED dir has no receipt and never had one, and only the
    caller knows whether it was expecting a restored dir. Unreadable or
    non-object content IS an error, because that is a receipt that exists and
    cannot be believed.
    """
    p = os.path.join(path, RECEIPT_NAME) if os.path.isdir(path) else path
    if not os.path.isfile(p):
        return None, f"{RECEIPT_NAME}: absent under {path}"
    try:
        rec = json.loads(open(p).read())
    except Exception as e:                       # noqa: BLE001 — report, not raise
        return None, f"{RECEIPT_NAME}: unreadable ({e!r})"
    if not isinstance(rec, dict):
        return None, f"{RECEIPT_NAME}: not a JSON object ({type(rec).__name__})"
    return rec, None


def corroborate_receipt(receipt: dict, fp: dict) -> list[str]:
    """Problems between the publisher's receipt and a live fingerprint. Pure.

    Two facts, and they are the two the receipt is in a position to state:
    the push declared itself COMPLETE, and it declared how many payload files it
    put there. A count that disagrees with what is on disk is a short or a fat
    restore — the failure `has` cannot see, because `has` only stats the marker.

    NOT AUTHORITY. Passing this proves the restore agrees with a claim made by
    the box that pushed it. It does not make the bytes right; that is what the
    fingerprint and the marker guards are for. A caller that reaches a verdict
    on the receipt ALONE has replaced a measurement with a promise.
    """
    bad: list[str] = []
    if receipt.get("complete") is not True:
        bad.append(f"{RECEIPT_NAME}.complete: {receipt.get('complete')!r} != True "
                   f"— the publisher never declared this prefix finished")
    n = receipt.get("files")
    if not isinstance(n, int) or isinstance(n, bool):
        bad.append(f"{RECEIPT_NAME}.files: {n!r} is not an integer count")
    elif n != fp.get("n_files"):
        bad.append(f"{RECEIPT_NAME}.files: publisher pushed {n} payload files, "
                   f"this dir holds {fp.get('n_files')} — a short or fat restore, "
                   f"which `has` cannot see (it only stats the marker)")
    return bad


# --- CLI ---------------------------------------------------------------------
def _load(p: str | None) -> tuple[dict | None, str | None]:
    if not p:
        return None, None
    try:
        return json.loads(open(p).read()), None
    except Exception as e:                       # noqa: BLE001
        return None, f"--frozen {p}: unreadable/unparseable ({e!r})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the merged dir to fingerprint")
    ap.add_argument("--frozen", help="a previously recorded fingerprint to "
                                     "compare this dir's file set against")
    ap.add_argument("--out", help="write the fingerprint (or the full verdict "
                                  "report, with --frozen) here")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                    help="names kept OUT of the fingerprint (transport "
                         "artefacts). Pass --exclude with no names to fingerprint "
                         "every file, which is the bundle-era rule.")
    ap.add_argument("--size-exempt", nargs="*", default=[],
                    help="files exempt from the SIZE comparison only — e.g. a "
                         "merge marker that embeds a host path and a timestamp")
    ap.add_argument("--receipt", action="store_true",
                    help="also corroborate against PUSHED.json; REQUIRES one to "
                         "be present (absent = cannot check = refuse)")
    ap.add_argument("--emit", action="store_true",
                    help="print the fingerprint and exit 0; NO verdict")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.dir):
        print(f"!! {a.dir}: not a directory", file=sys.stderr)
        print(VERDICT_REFUSED)
        return 2

    fp = fingerprint_dir(a.dir, exclude=tuple(a.exclude))
    report = {"dir": a.dir, "fingerprint": fp, "excluded": sorted(a.exclude)}

    if a.emit:
        print(json.dumps(report, indent=1, sort_keys=True))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(fp, fh, indent=1, sort_keys=True)
        return 0

    problems: list[str] = []
    cannot_check = False

    frozen, why = _load(a.frozen)
    if why:
        problems.append(why)
        cannot_check = True
    elif frozen is not None:
        problems += compare_fingerprint(frozen, fp,
                                        size_exempt=tuple(a.size_exempt))

    if a.receipt:
        rec, why = read_pushed_receipt(a.dir)
        if rec is None:
            # FAIL CLOSED. `--receipt` is the caller saying it expects a
            # restored dir; "the receipt is missing" is the exact shape of the
            # race the write-last protocol exists to exclude.
            problems.append(f"{why} — --receipt was requested, so this is a "
                            f"REFUSAL and not a skip")
            cannot_check = True
        else:
            report["receipt"] = rec
            problems += corroborate_receipt(rec, fp)

    if frozen is None and not a.receipt:
        problems.append("neither --frozen nor --receipt was given: there is "
                        "nothing to compare this dir against. Mint a "
                        "fingerprint with --emit and pass it as --frozen.")
        cannot_check = True

    report["problems"] = problems
    report["ok"] = not problems
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)

    if problems:
        for p in problems:
            print(f"!!   {p}", file=sys.stderr)
        print(f"!! Refusing {a.dir}: a merged dir that does not match what was "
              f"published serves cleanly and reads as 'the baseline "
              f"reproduced'.", file=sys.stderr)
        print(VERDICT_REFUSED)
        return 2 if cannot_check else 1

    print(f">> {a.dir}: {fp['n_files']} files, fp {fp['sha256'][:12]}…")
    print(VERDICT_OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
