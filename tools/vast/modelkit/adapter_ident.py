#!/usr/bin/env python3
"""Compute the ADAPTER IDENTITY sha that `job-config.yaml` pins.

`frontier-wave/run.sh`'s S0 gate hashes each model directory and hard-stops when
the result differs from the `ADAPTER_*_SHA256` / `BASE_SHA256` pin, so a stale
asset cache or a swapped checkpoint fails closed instead of silently producing a
mixed-adapter wave. Setting those pins therefore means computing the same hash
on the workstation.

Before this module existed there was no tool for that, and the value was
obtained by re-implementing the rule in a throwaway script (2026-07-31, wiring
the v7 pair). Two copies of a hash that MUST agree is precisely the kind of
thing that drifts silently: a divergence would not error, it would just make
every pin wrong and every wave fail its own identity gate.

The hash is NOT a file sha. It is sha256 over the JSON of the sorted
`{rel, size, sha256}` records of the required files — so it covers the file
set, not just one blob, and a missing required file is an error rather than a
different-but-plausible digest.

    python3 tools/witness/adapter_ident.py out/jobs/<job>/out
    python3 tools/witness/adapter_ident.py --base /path/to/base-model

FOLLOW-UP (deliberately not done here): `run.sh`'s S0 block still carries its
own inline copy. Unifying them means the wave runner importing this module,
which is a change to a live grading path — not something to land while a wave
is queued (LAUNCH_DEFECT_POSTMORTEM_2026-07-30). `test_adapter_ident.py` pins
the two together in the meantime by executing run.sh's own implementation.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

# The required-file patterns run.sh's S0 uses, verbatim. An adapter is its
# config + its weights; a base model is its config + every shard.
ADAPTER_REQ = ["adapter_config.json", "adapter_model.safetensors"]
BASE_REQ = ["config.json", "*.safetensors"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ident(d: str, req: list[str]) -> str:
    """sha256 over the sorted {rel,size,sha256} records of `req` under `d`.

    Raises FileNotFoundError when a required pattern matches nothing — the pin
    is meant to be unforgeable, so an absent weights file must not quietly
    produce a hash of whatever else happened to be there.
    """
    recs = []
    for pat in req:
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if not hits:
            raise FileNotFoundError(f"{d}: missing required {pat}")
        for h in hits:
            recs.append({"rel": os.path.basename(h),
                         "size": os.path.getsize(h),
                         "sha256": sha256_file(h)})
    recs.sort(key=lambda r: r["rel"])
    return hashlib.sha256(json.dumps(recs, sort_keys=True).encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="+", help="model directory (repeatable)")
    ap.add_argument("--base", action="store_true",
                    help="hash as a BASE model (config.json + *.safetensors) "
                         "instead of a LoRA adapter")
    ap.add_argument("--json", action="store_true",
                    help="emit {dir: sha} instead of one sha per line")
    args = ap.parse_args(argv)

    req = BASE_REQ if args.base else ADAPTER_REQ
    out = {}
    for d in args.dir:
        try:
            out[d] = ident(d, req)
        except FileNotFoundError as e:
            print(f"!! {e}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        for d, s in out.items():
            print(f"{s}  {d}" if len(out) > 1 else s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
