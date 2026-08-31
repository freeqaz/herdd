"""modelkit — the merged-model-artifact machinery, hoisted out of job bundles.

Every piece here shipped first as a per-bundle copy (`driftr3-v10-27b-gen/`,
`driftr3-v9-gemma4-gen/`, `v9-serve-parity/`, …). Four to five identical copies
of a gate is not four gates, it is one gate and four places for it to rot — and
the failure these gates exist to catch is SILENT (a merged dir that loads,
serves, answers normally and scores as the BASE), so rot is invisible until a
number is already wrong.

WHAT IS AND IS NOT HOISTED. The bundles are of-record and bundle-hashed, so
nothing here rewrites them. This package is for NEW consumers; the existing
copies stay where they are and `test_modelkit_bundle_parity.py` makes any drift
between them visible instead of silent.

Stdlib only, like the rest of `tools/vast` — these files get staged onto boxes
whose interpreter is whatever the image ships.

Layout:
  merged_fingerprint.py  grade-A NAME/SIZE fingerprint + PUSHED.json receipt.
                         Single self-contained file on purpose: the serve lane
                         stages it to a box ALONE.
  dirhash.py             grade-B CONTENT manifest + rollup sha.
  gate_dir.py            fail-closed CLI gate over the grade-B manifest.
  merge_guard.py         family-spec-driven structural verifier (family_specs/).
  restore_merged.py      reuse-local / reuse-restored / merge, verifier injected.
  registry/              one JSON per published artifact; `registry.py` reads it.
  b2_transport.sh        has / pull / push, PUSHED.json written LAST.

Conventions doc: docs/architecture/MERGED_MODEL_ARTIFACTS.md.
"""
