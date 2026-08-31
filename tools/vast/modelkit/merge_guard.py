#!/usr/bin/env python3
"""The merged-dir STRUCTURAL guard, driven by a per-family spec JSON.

WHY THE GUARD IS DATA. `merge_guard_27b.py` and `merge_guard_g4.py` are the same
program twice: identical skeleton (`check_marker` / `fingerprint_dir` /
`compare_fingerprint` / `check_index` / `verify_merged_dir`), different
CONSTANTS — tensor geometry, must-move probe keys, must-not-move frozen keys,
marker name. A third family means a third copy, and by then the skeleton has
drifted in ways no test compares. So the skeleton lives here once and the
constants live in `family_specs/<family>.json`, where adding a family is a data
change and the reviewable diff is the numbers themselves.

WHAT THE STRUCTURAL GUARD PROVES, stated plainly: that the WEIGHTS MOVED — the
expected number of text tensors were substituted, an adapted weight differs from
base, and the named frozen ones do not. It does not prove the model's BEHAVIOUR
moved; a behavioural divergence preflight is a separate, per-box control and
nothing here substitutes for it.

WHY IT IS RE-RUN ON A RESTORE. A restored dir has no merge process behind it, so
it arrives with no guard evidence of its own — and the failure this whole lane
is built around is SILENT: substituting 1 of N tensors yields a "merged" model
that loads, serves, answers normally and SCORES AS THE BASE. A truncated or
half-pushed restore has exactly that shape, and no score can see it.

A GUARD THAT IS "CHECKED IF FOUND" IS NOT A GUARD. Every key a spec names is
required PRESENT (or, for `absent_keys`, required ABSENT). A condition that can
pass by not being evaluated is not a condition — that is the rule the ancestors
adopted after a base whose visual tower was absent would have skipped the vision
check in silence and still reported a clean merge.

DERIVED, NEVER TYPED. A spec's `derived` block restates an arithmetic relation
between geometry constants (`wrapper == total - text`), and `load_spec` checks
it. A transposed digit in either constant cannot survive the load. Marker
expectations reference geometry by `@name` rather than repeating the literal,
for the same reason: two copies of a number that must agree is two numbers that
happen to agree today.

    merge_guard.py verify <merged-dir> --family qwen36-27b
                          [--base-pins F] [--fingerprint F]
                          [--report F] [--fingerprint-out F]
    merge_guard.py specs                       # list known families

exit 0 ok, 1 problems, 2 usage/spec error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

try:                                             # pragma: no cover - trivial
    from . import merged_fingerprint as mfp
except ImportError:                              # pragma: no cover - trivial
    import merged_fingerprint as mfp             # type: ignore[no-redef]

SPEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "family_specs")

SPEC_SCHEMA_VERSION = 1
_REQUIRED_SPEC_KEYS = ("schema_version", "family", "marker_name", "geometry",
                       "marker_fields", "probe_keys", "frozen_keys", "index")
_KNOWN_SPEC_KEYS = _REQUIRED_SPEC_KEYS + (
    "description", "derived", "absent_keys", "marker_absent_field",
    "base_pin_check", "merge")

#: The OPTIONAL `merge:` block's shapes. `relayout` rebuilds the merged text
#: stack inside the BASE repo layout (a wrapper config, a vision tower and an
#: MTP head are copied verbatim and only the text tensors are substituted);
#: `flat` saves what `merge_and_unload` produced, because the base already IS
#: the text-only extraction and the key sets are equal.
MERGE_KINDS = ("relayout", "flat")

#: Every quantity a merger is allowed to write into a marker field, named once.
#: A spec maps `marker_fields` names onto these; anything else is a REFUSAL at
#: load, so a marker field the merger cannot fill is caught here rather than at
#: the end of a multi-hour merge — and a field nobody fills would otherwise read
#: as `None != required 851` on a merge that was perfectly fine.
MEASURED = (
    "n_base_actual",       # tensors in the base's own index, counted
    "n_base_expected",     # the geometry constant the merge was handed
    "n_text_actual",       # tensors in the text-only save, counted
    "n_text_expected",
    "n_substituted",       # tensors written from the merge
    "n_kept",              # tensors copied from base verbatim
    "n_merged_actual",     # tensors in the finished merged dir
    "n_lora_attached",     # LoRA tensors PEFT bound — an ADAPTER property
    "keys_equal_base",     # bool: merged key set == base key set
    "arch",                # architectures[0] of the merged config
    "model_type",
)


class SpecError(ValueError):
    """A family spec that cannot be trusted. Never degraded into a warning."""


# --- spec loading ------------------------------------------------------------
def _resolve(value, geometry: dict):
    """`"@name"` -> geometry[name]; anything else is a literal."""
    if isinstance(value, str) and value.startswith("@"):
        name = value[1:]
        if name not in geometry:
            raise SpecError(f"@{name} does not name a geometry constant "
                            f"(have {sorted(geometry)})")
        return geometry[name]
    return value


def _check_derived(spec: dict) -> None:
    """`{"wrapper_tensors": ["base_total", "-", "text"]}` must reproduce."""
    geo = spec["geometry"]
    for name, expr in (spec.get("derived") or {}).items():
        if (not isinstance(expr, list) or len(expr) != 3
                or expr[1] not in ("-", "+")):
            raise SpecError(f"derived[{name}]: {expr!r} is not [a, '-'|'+', b]")
        a, op, b = expr
        for operand in (a, b):
            if operand not in geo:
                raise SpecError(f"derived[{name}]: {operand!r} is not a "
                                f"geometry constant")
        want = geo[a] - geo[b] if op == "-" else geo[a] + geo[b]
        if name not in geo:
            raise SpecError(f"derived[{name}]: no such geometry constant")
        if geo[name] != want:
            raise SpecError(
                f"derived[{name}]: the spec states {geo[name]} but "
                f"{a} {op} {b} = {want}. One of the three is a typo, and a "
                f"guard whose halves disagree is two constants, not a guard.")


def _check_merge_block(spec: dict) -> None:
    """Validate the OPTIONAL `merge:` block — a spec a MERGER can execute.

    The guard itself never reads it. It is validated here anyway because
    `load_spec` is the only place a spec is checked at all, and a family whose
    merge block is wrong would otherwise be discovered by a box that has already
    paid for a 52 GiB base pull.

    The binding rule is TOTALITY: every `marker_fields` name must be mapped, so
    a guard condition the merger cannot fill is a load-time refusal instead of a
    `None != required 851` on a merge that did nothing wrong.
    """
    merge = spec.get("merge")
    if merge is None:
        return
    if not isinstance(merge, dict):
        raise SpecError(f"merge: {merge!r} is not an object")
    unknown = sorted(set(merge) - {"kind", "marker_map"})
    if unknown:
        raise SpecError(f"merge: unknown key(s) {unknown}")
    if merge.get("kind") not in MERGE_KINDS:
        raise SpecError(f"merge.kind: {merge.get('kind')!r} not in "
                        f"{list(MERGE_KINDS)}")
    mm = merge.get("marker_map")
    if not isinstance(mm, dict):
        raise SpecError(f"merge.marker_map: {mm!r} is not an object")
    bad = sorted(v for v in mm.values() if v not in MEASURED)
    if bad:
        raise SpecError(f"merge.marker_map: {bad} are not quantities a merger "
                        f"measures (have {list(MEASURED)})")
    unmapped = sorted(set(spec["marker_fields"]) - set(mm))
    if unmapped:
        raise SpecError(
            f"merge.marker_map: marker field(s) {unmapped} are required by the "
            f"guard and have no measured quantity to fill them. A merger cannot "
            f"write them, so the guard would refuse every merge of this family.")
    if spec.get("marker_absent_field") and merge["kind"] != "flat":
        # Only the flat merge reads the base's key set directly; the relayout
        # walks the base index and would need a second pass to answer it.
        raise SpecError(f"merge: marker_absent_field is only fillable by a "
                        f"'flat' merge, not {merge['kind']!r}")


def load_spec(family: str) -> dict:
    """Load and VALIDATE `family_specs/<family>.json` (or a path to one).

    Every failure here raises. A spec that half-parses would produce a guard
    that half-checks, and a guard that half-checks reports PASS.
    """
    path = family if os.path.sep in family or family.endswith(".json") else \
        os.path.join(SPEC_DIR, f"{family}.json")
    if not os.path.isfile(path):
        raise SpecError(f"no family spec at {path} (known: {known_families()})")
    try:
        spec = json.loads(open(path).read())
    except Exception as e:                       # noqa: BLE001
        raise SpecError(f"{path}: unparseable ({e!r})") from e
    if not isinstance(spec, dict):
        raise SpecError(f"{path}: not a JSON object")

    missing = [k for k in _REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise SpecError(f"{path}: missing required key(s) {missing}")
    unknown = sorted(set(spec) - set(_KNOWN_SPEC_KEYS))
    if unknown:
        # Refused, not ignored: a typo'd key that is silently dropped is a
        # condition the operator believes is armed and is not.
        raise SpecError(f"{path}: unknown key(s) {unknown} — refusing a spec "
                        f"with fields this guard does not read")
    if spec["schema_version"] != SPEC_SCHEMA_VERSION:
        raise SpecError(f"{path}: schema_version {spec['schema_version']} != "
                        f"{SPEC_SCHEMA_VERSION}")
    if not isinstance(spec["geometry"], dict) or not spec["geometry"]:
        raise SpecError(f"{path}: geometry must be a non-empty object")
    if not spec["probe_keys"]:
        raise SpecError(f"{path}: probe_keys is empty — with no must-move key "
                        f"a no-op merge passes every remaining check")
    _check_derived(spec)
    # Resolve @refs once, so a bad reference fails at LOAD and not at the
    # moment the guard was supposed to refuse something.
    spec["marker_fields"] = {k: _resolve(v, spec["geometry"])
                             for k, v in spec["marker_fields"].items()}
    spec["index"] = dict(spec["index"])
    spec["index"]["expect_tensors"] = _resolve(spec["index"]["expect_tensors"],
                                               spec["geometry"])
    if spec.get("absent_keys") and not spec.get("marker_absent_field"):
        raise SpecError(f"{path}: absent_keys without marker_absent_field — "
                        f"the guard would have nothing to compare them against")
    try:
        _check_merge_block(spec)
    except SpecError as e:
        raise SpecError(f"{path}: {e}") from e
    return spec


def known_families() -> list[str]:
    if not os.path.isdir(SPEC_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SPEC_DIR) if f.endswith(".json"))


# --- marker checks (pure) ----------------------------------------------------
def check_marker(marker: dict, spec: dict) -> list[str]:
    """Every structural condition of the spec, against the merge marker.

    Pure: no I/O, so the whole condition set is unit-testable on a dict.
    Returns human-readable problems; empty means PASS.
    """
    bad: list[str] = []

    for field, expect in spec["marker_fields"].items():
        got = marker.get(field)
        if got != expect:
            bad.append(f"{field}: {got!r} != required {expect!r}")

    guards = marker.get("guards")
    if not isinstance(guards, dict):
        bad.append("guards: missing or not a mapping — the merge recorded no "
                   "must-move/must-not-move evidence")
        return bad

    for key in spec["probe_keys"]:
        g = guards.get(key)
        if not isinstance(g, dict):
            bad.append(f"guards[{key}]: MISSING — no evidence this adapted "
                       f"weight moved; a no-op merge scores as the BASE")
            continue
        d = g.get("max_abs_delta")
        if not isinstance(d, (int, float)) or isinstance(d, bool) or not d > 0:
            bad.append(f"guards[{key}].max_abs_delta: {d!r} — must be > 0")
        if g.get("must_move") is not True:
            bad.append(f"guards[{key}].must_move: {g.get('must_move')!r} != True")

    for key in spec["frozen_keys"]:
        g = guards.get(key)
        if not isinstance(g, dict):
            bad.append(f"guards[{key}]: MISSING — this frozen tensor was never "
                       f"checked, and an unevaluated condition is not a "
                       f"condition")
            continue
        d = g.get("max_abs_delta")
        if d != 0 and d != 0.0:
            bad.append(f"guards[{key}].max_abs_delta: {d!r} != 0 — a frozen "
                       f"tensor moved")
        if g.get("must_move") is not False:
            bad.append(f"guards[{key}].must_move: {g.get('must_move')!r} != False")

    field = spec.get("marker_absent_field")
    if field:
        want = list(spec.get("absent_keys") or [])
        got = marker.get(field)
        if not isinstance(got, list) or sorted(got) != sorted(want):
            bad.append(f"{field}: {got!r} != {want!r} — the base is not the "
                       f"extraction this adapter was trained against")

    return bad


# --- the merged dir's own index ---------------------------------------------
def _safetensors_keys(path: str) -> set[str] | None:
    """Tensor names from a safetensors HEADER — no torch, no full read."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
    except Exception:                            # noqa: BLE001
        return None
    return {k for k in head if k != "__metadata__"}


def dir_tensor_keys(d: str) -> set[str] | None:
    """Every tensor name in a model dir, sharded or single-file. Header-only."""
    idx_p = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx_p):
        try:
            return set(json.loads(open(idx_p).read())["weight_map"])
        except Exception:                        # noqa: BLE001
            return None
    single = os.path.join(d, "model.safetensors")
    return _safetensors_keys(single) if os.path.isfile(single) else None


def check_index(merged_dir: str, spec: dict) -> list[str]:
    """The merged dir must describe the expected tensor count, and every shard
    it names must be on disk and non-empty.

    This is what catches a TRUNCATED restore, which is the realistic B2 failure
    and the one a file COUNT alone misses. `allow_single_file` handles the
    unsharded shape on purpose: `save_pretrained` shards at 5 GB by default, but
    a `max_shard_size` change writes one `model.safetensors` and no index, and a
    gate that only understood the sharded shape would then fail a good merge.
    """
    want = spec["index"]["expect_tensors"]
    allow_single = bool(spec["index"].get("allow_single_file"))
    bad: list[str] = []
    idx_p = os.path.join(merged_dir, "model.safetensors.index.json")
    single_p = os.path.join(merged_dir, "model.safetensors")

    if os.path.isfile(idx_p):
        try:
            weight_map = json.loads(open(idx_p).read())["weight_map"]
        except Exception as e:                   # noqa: BLE001 — report, not raise
            return [f"unreadable model.safetensors.index.json ({e!r})"]
        if len(weight_map) != want:
            bad.append(f"index holds {len(weight_map)} tensors != required "
                       f"{want}")
        for shard in sorted(set(weight_map.values())):
            p = os.path.join(merged_dir, shard)
            if not os.path.isfile(p):
                bad.append(f"shard {shard}: MISSING")
            elif os.path.getsize(p) == 0:
                bad.append(f"shard {shard}: EMPTY")
        return bad

    if allow_single and os.path.isfile(single_p):
        keys = _safetensors_keys(single_p)
        if keys is None:
            bad.append("model.safetensors: unreadable safetensors header")
        elif len(keys) != want:
            bad.append(f"model.safetensors holds {len(keys)} tensors != "
                       f"required {want}")
        return bad

    if allow_single:
        return [f"no model.safetensors.index.json and no model.safetensors "
                f"under {merged_dir}"]
    return [f"no model.safetensors.index.json under {merged_dir}"]


# --- base-pin content checks -------------------------------------------------
def _sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def check_wrapper_files(merged_dir: str, base_pins: dict,
                        *, skip_suffixes=(".safetensors",),
                        skip_prefixes=("_",)) -> list[str]:
    """Every NON-weight base file the relayout copies VERBATIM must still be the
    base's bytes.

    This is the one content-level check available to a restorer, and it is the
    one that matters: a merged dir built from a DIFFERENT base snapshot has the
    right tensor counts, passes every must-move/must-not-move guard, and serves
    a different model. `model.safetensors.index.json` is included by this rule
    (its name does not end in `.safetensors`), which is what pins the shard
    layout the substitution loop walked.
    """
    bad: list[str] = []
    checked = 0
    for name, rec in sorted(base_pins.items()):
        if name.startswith(tuple(skip_prefixes)) or name.endswith(tuple(skip_suffixes)):
            continue
        if "sha256" not in rec:
            continue
        p = os.path.join(merged_dir, name)
        if not os.path.isfile(p):
            bad.append(f"WRAPPER {name}: MISSING from the merged dir — the "
                       f"relayout copies it verbatim from base")
            continue
        size = os.path.getsize(p)
        if size != rec["size"]:
            bad.append(f"WRAPPER {name}: size {size} != base {rec['size']}")
            continue
        got = _sha256_file(p)
        if got != rec["sha256"]:
            bad.append(f"WRAPPER {name}: sha256 {got} != base {rec['sha256']} — "
                       f"this merged dir was NOT built from the pinned base")
        else:
            checked += 1
    if checked == 0 and not bad:
        bad.append("no wrapper file carried a sha256 pin — refusing to report a "
                   "content check that checked nothing")
    return bad


def _base_bytes(base_pins: dict, name: str) -> tuple[bytes | None, str | None]:
    """The pinned base file's exact bytes, or (None, why-not). FAIL CLOSED.

    `content` in the pins file is NOT a second authority: it is used only if it
    reproduces the sha256 AND the size already pinned for that file. The only
    two outcomes are "these are provably the pinned base's bytes" and a hard
    problem — a pin edited without its content, or the reverse, fails loudly
    instead of degrading into an unchecked comparison.
    """
    rec = base_pins.get(name)
    if not isinstance(rec, dict) or "sha256" not in rec or "size" not in rec:
        return None, f"CONFIG {name}: no sha256/size pin in the base pins file"
    text = rec.get("content")
    if not isinstance(text, str):
        return None, (f"CONFIG {name}: the base pins file carries no `content` "
                      f"for it, so the base's own bytes are unavailable and the "
                      f"comparison CANNOT BE PERFORMED — refusing to pass a "
                      f"check that never ran")
    raw = text.encode()
    got = hashlib.sha256(raw).hexdigest()
    if got != rec["sha256"] or len(raw) != rec["size"]:
        return None, (f"CONFIG {name}: the pins file's embedded `content` "
                      f"hashes {got} at {len(raw)} B, not the pinned "
                      f"{rec['sha256']} at {rec['size']} B — the pins file "
                      f"disagrees with itself and neither half can be trusted")
    return raw, None


def _semantic_config_problems(merged_dir: str, base_pins: dict, name: str,
                              exempt_keys: tuple[str, ...]) -> list[str]:
    """Key-by-key equality against the base's own bytes, `exempt_keys` aside.

    STRICTER THAN THE SHA IT REPLACES, not a relaxation: a sha can only say
    "differs", while this compares every key both ways and NAMES the field that
    moved regardless of formatting, key order or whitespace. The exemption
    exists because `generation_config.json` embeds the WRITING library's version
    stamp — provenance about the writer, not a property of the model — and a
    sha256 on it called a good merge a base swap.
    """
    p = os.path.join(merged_dir, name)
    if not os.path.isfile(p):
        return [f"CONFIG {name}: MISSING from the merged dir"]
    base_raw, why = _base_bytes(base_pins, name)
    if base_raw is None:
        return [why]
    try:
        want = json.loads(base_raw)
        got = json.loads(open(p, "rb").read())
    except Exception as e:                       # noqa: BLE001 — report, not raise
        return [f"CONFIG {name}: unparseable JSON ({e!r}) — refusing to pass a "
                f"comparison that could not be performed"]
    if not isinstance(want, dict) or not isinstance(got, dict):
        return [f"CONFIG {name}: not a JSON object on one side "
                f"(base {type(want).__name__}, merged {type(got).__name__})"]

    want = {k: v for k, v in want.items() if k not in exempt_keys}
    got = {k: v for k, v in got.items() if k not in exempt_keys}
    bad: list[str] = []
    for k in sorted(set(want) - set(got)):
        bad.append(f"CONFIG {name}: key {k!r} MISSING from the merged copy "
                   f"(base has {want[k]!r})")
    for k in sorted(set(got) - set(want)):
        bad.append(f"CONFIG {name}: key {k!r} ADDED by the merged copy "
                   f"({got[k]!r}) — the base does not carry it")
    for k in sorted(set(want) & set(got)):
        if got[k] != want[k]:
            bad.append(f"CONFIG {name}: {k} = {got[k]!r} != base {want[k]!r} — "
                       f"this merged dir was NOT built from the pinned base")
    return bad


def check_config_files(merged_dir: str, base_pins: dict, *,
                       bytewise=("config.json",), semantic=None) -> list[str]:
    """The named config files must still be the BASE's — two grades.

    `bytewise` files come out of `save_pretrained` byte-identical, so sha256 +
    size against the pin, no exemptions. `semantic` files do not (see
    `_semantic_config_problems`), so they are compared key by key with a named
    exemption list.

    THE TOKENIZER IS DELIBERATELY NOT IN THIS SET, and the exemption is named
    rather than silent: `save_pretrained` re-serialises it, so the merged copy
    is semantically the base's and byte-wise its own. What pins it instead is
    the grade-B gate over the BASE dir, plus the fact that the merge reads that
    same gated dir.
    """
    semantic = semantic or {}
    bad: list[str] = []
    checked = 0

    for name in bytewise:
        rec = base_pins.get(name)
        if not isinstance(rec, dict) or "sha256" not in rec:
            bad.append(f"CONFIG {name}: no sha256 pin in the base pins file")
            continue
        p = os.path.join(merged_dir, name)
        if not os.path.isfile(p):
            bad.append(f"CONFIG {name}: MISSING from the merged dir")
            continue
        size = os.path.getsize(p)
        got = _sha256_file(p)
        if size != rec["size"]:
            bad.append(f"CONFIG {name}: size {size} != base {rec['size']}")
        elif got != rec["sha256"]:
            bad.append(f"CONFIG {name}: sha256 {got} != base {rec['sha256']} — "
                       f"this merged dir was NOT built from the pinned base")
        else:
            checked += 1

    for name, opts in sorted(semantic.items()):
        probs = _semantic_config_problems(
            merged_dir, base_pins, name,
            tuple((opts or {}).get("exempt_keys", ())))
        bad += probs
        if not probs:
            checked += 1

    if checked == 0 and not bad:
        bad.append("no config file was actually compared — refusing to report a "
                   "content check that checked nothing")
    return bad


# --- the whole verification --------------------------------------------------
def _base_pin_problems(merged_dir: str, base_pins: dict, spec: dict) -> list[str]:
    cfg = spec.get("base_pin_check") or {}
    kind = cfg.get("kind", "wrapper_files")
    if kind == "wrapper_files":
        return check_wrapper_files(
            merged_dir, base_pins,
            skip_suffixes=tuple(cfg.get("skip_suffixes", (".safetensors",))),
            skip_prefixes=tuple(cfg.get("skip_prefixes", ("_",))))
    if kind == "config_files":
        return check_config_files(
            merged_dir, base_pins,
            bytewise=tuple(cfg.get("bytewise", ("config.json",))),
            semantic=cfg.get("semantic") or {})
    raise SpecError(f"base_pin_check.kind {kind!r} is not one of "
                    f"'wrapper_files' / 'config_files'")


def verify_merged_dir(merged_dir: str, spec: dict, *,
                      base_pins: dict | None = None,
                      frozen_fingerprint: dict | None = None
                      ) -> tuple[list[str], dict]:
    """Full artifact-level re-verification. Returns (problems, report)."""
    marker_name = spec["marker_name"]
    problems: list[str] = []
    report: dict = {"dir": merged_dir, "family": spec["family"],
                    "geometry": dict(spec["geometry"])}

    if not os.path.isdir(merged_dir):
        return ([f"{merged_dir}: not a directory"], report)

    marker_p = os.path.join(merged_dir, marker_name)
    if not os.path.isfile(marker_p):
        problems.append(f"{marker_name}: MISSING — this dir carries no record "
                        f"of the merge that produced it")
    else:
        try:
            marker = json.loads(open(marker_p).read())
        except Exception as e:                   # noqa: BLE001
            problems.append(f"{marker_name}: unreadable ({e!r})")
        else:
            report["marker"] = marker
            problems += check_marker(marker, spec)

    problems += check_index(merged_dir, spec)

    fp = mfp.fingerprint_dir(merged_dir)
    report["fingerprint"] = fp
    if frozen_fingerprint is not None:
        # The marker embeds an absolute adapter path and a timestamp, so its
        # SIZE is host-dependent by construction. Exempt from the SIZE compare
        # only — it must still be PRESENT, and its CONTENT is check_marker's job.
        problems += mfp.compare_fingerprint(frozen_fingerprint, fp,
                                            size_exempt=(marker_name,))

    if base_pins is not None:
        problems += _base_pin_problems(merged_dir, base_pins, spec)

    report["problems"] = problems
    report["ok"] = not problems
    return (problems, report)


def _load(p: str | None) -> dict | None:
    return None if not p else json.loads(open(p).read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("specs", help="list known model families")
    v = sub.add_parser("verify", help="re-verify a merged dir")
    v.add_argument("dir")
    v.add_argument("--family", required=True,
                   help="a family_specs/ name, or a path to a spec JSON")
    v.add_argument("--base-pins", help="the frozen base pins file — enables the "
                                       "verbatim-wrapper / config content check")
    v.add_argument("--fingerprint", help="a previously recorded fingerprint to "
                                         "compare the file set against")
    v.add_argument("--report", help="write the full JSON report here")
    v.add_argument("--fingerprint-out", help="write this dir's fingerprint here")
    a = ap.parse_args(argv)

    if a.cmd == "specs":
        for f in known_families():
            print(f)
        return 0

    try:
        spec = load_spec(a.family)
    except SpecError as e:
        print(f"!! {e}", file=sys.stderr)
        return 2

    problems, report = verify_merged_dir(
        a.dir, spec, base_pins=_load(a.base_pins),
        frozen_fingerprint=_load(a.fingerprint))

    if a.report:
        with open(a.report, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
    if a.fingerprint_out:
        with open(a.fingerprint_out, "w") as fh:
            json.dump(report["fingerprint"], fh, indent=1, sort_keys=True)

    if problems:
        print(f"!! MERGE GUARD FAILED for {a.dir} ({spec['family']}):",
              file=sys.stderr)
        for p in problems:
            print(f"   {p}", file=sys.stderr)
        print("!! Refusing a merged dir that does not satisfy its family's "
              "structural guard. A merge that is silently a no-op serves the "
              "BASE model under the adapter's label.", file=sys.stderr)
        return 1
    geo = ", ".join(f"{k}={v}" for k, v in sorted(spec["geometry"].items()))
    print(f">> MERGE GUARD OK for {a.dir} ({spec['family']}): {geo}, "
          f"{len(spec['probe_keys'])} probe(s) moved, "
          f"{len(spec['frozen_keys'])} frozen tensors did not, "
          f"{report['fingerprint']['n_files']} files "
          f"(fp {report['fingerprint']['sha256'][:12]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
