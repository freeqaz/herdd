#!/usr/bin/env python3
"""The MODEL-ARTIFACT REGISTRY: one committed JSON per published artifact.

WHY IT IS IN GIT. Before this, "where does MERGEDDEMOA live and what is it" was
spread across a `run.sh`'s shell variables, a `job-config.yaml`'s `--env` pins
and an operator's memory. Each consumer re-derived the B2 key from an adapter
sha, each one had its own copy of the served name and the VRAM floor, and
nothing compared them. A wrong path is not a crash — it is a server that boots,
answers, and scores like a baseline.

THE LAYOUT IS DERIVED, NOT TYPED. A merged artifact lives at

    <b2_root>/<adapter_ident[:12]>/model      the weights
    <b2_root>/<adapter_ident[:12]>/guards     merge guard json, fingerprint, marker

and `validate` RECOMPUTES both prefixes from `b2_root` + `adapter_ident` rather
than reading them. Keying on the adapter sha is what makes a merged dir
impossible to serve for a different adapter: a dir published under a name that
does not carry its adapter's identity is one rename away from serving the wrong
weights under the right label.

FAIL CLOSED ON SHAPE. An unknown key is a REFUSAL, not a shrug — a typo'd field
is a fact the operator believes is recorded and is not. That also makes the file
format safe for a mechanical publisher: appending an entry is writing one JSON
file whose name is its `id`, and `check` is the acceptance test.

TWO HASH FIELDS, AND THEY ARE NOT INTERCHANGEABLE (see MERGED_MODEL_ARTIFACTS.md):
  `fingerprint_sha256`  grade A, over the sorted NAME/SIZE list. What the
                        publishing box could honestly claim about a dir it had
                        just merged (CPU bf16 merge is not bit-reproducible).
  `content_sha256`      grade B, the per-file sha256 rollup. `null` means
                        UNMEASURED, never "clean" — a consumer that treats a
                        null as a pass has replaced a measurement with a
                        default.

A BOX CANNOT COMMIT, so the publisher does not write here. `merge-publish-v1`
emits a CANDIDATE into its job results (`jobs/` — the only prefix a jobs box's
key grants) and `fold` is the workstation half: it re-validates the candidate
against THIS checkout's registry, refuses an overwrite, and writes the entry
for an operator to commit. The split is the write scoping made honest — the box
publishes weights under `checkpoints/`, the registry is git-only by design.

`pin-base` is the same handoff for the OTHER entry a merge touches. A publish
measures the BASE snapshot it merged against (`base_rollup_sha256`,
`base_n_files`, `guards/base_pins.json`) because that box is the only place
holding those bytes for free — and until this verb existed that measurement
died in the candidate's provenance while the base entry kept its `null`. It is
deliberately NOT part of `fold`: `fold` mints one new merged entry and refuses
an existing id, `pin-base` edits an existing base entry in place, and the input
need not be a candidate at all.

NOT to be confused with `tools/vast/registry/`, which is the private OCI
*image* registry. Different artifact, different transport, no relation.

    registry.py ls | show <id> | check | path <id> | fold <candidate.json>
                | pin-base <candidate.json|base_pins.json|guards_summary.json>
exit 0 ok, 1 invalid, 2 usage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

REGISTRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "registry")

SCHEMA_VERSION = 1
#: The wrapper a publishing box emits. Versioned SEPARATELY from the entry
#: schema: the candidate carries provenance that never enters git, so the two
#: move for different reasons and a shared number would couple them.
CANDIDATE_SCHEMA_VERSION = 1
KINDS = ("base", "merged")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

#: `<b2_root>/<sha12>/<leaf>` — the leaves a merged artifact publishes.
MERGED_LEAVES = ("model", "guards")
SHA12 = 12

_COMMON_REQUIRED = ("schema_version", "id", "kind", "b2_root")
_COMMON_OPTIONAL = ("description", "family", "content_sha256", "n_files")

_BASE_REQUIRED: tuple[str, ...] = ()
_BASE_OPTIONAL = ("pins",)

_MERGED_REQUIRED = ("adapter_ident", "base", "family", "b2_model", "b2_guards",
                    "fingerprint_sha256", "n_files", "serve")
_MERGED_OPTIONAL: tuple[str, ...] = ()

_SERVE_REQUIRED = ("served_name", "dtype", "max_len", "min_vram_gb", "tp",
                   "lora_forbidden")


class RegistryError(ValueError):
    """A registry file that cannot be trusted. Never degraded to a warning."""


def _allowed(kind: str) -> set[str]:
    if kind == "base":
        return set(_COMMON_REQUIRED + _COMMON_OPTIONAL + _BASE_REQUIRED
                   + _BASE_OPTIONAL)
    return set(_COMMON_REQUIRED + _COMMON_OPTIONAL + _MERGED_REQUIRED
               + _MERGED_OPTIONAL)


def merged_prefixes(b2_root: str, adapter_ident: str) -> dict[str, str]:
    """The published prefixes an adapter sha implies. Derivation, not lookup."""
    sha12 = adapter_ident[:SHA12]
    root = b2_root.rstrip("/")
    return {f"b2_{leaf}": f"{root}/{sha12}/{leaf}" for leaf in MERGED_LEAVES}


def validate(entry: dict, *, stem: str | None = None) -> list[str]:
    """Every shape problem with one entry. Pure; empty means the entry is sound.

    `stem` is the filename without `.json`; when given, `id` must equal it. A
    registry where the filename and the id can disagree has two names for one
    artifact, and a lookup by either finds a different thing.
    """
    bad: list[str] = []
    if not isinstance(entry, dict):
        return [f"not a JSON object ({type(entry).__name__})"]

    kind = entry.get("kind")
    if kind not in KINDS:
        # Everything downstream is kind-dependent, so this is the one problem
        # that stops the check rather than adding to it.
        return [f"kind: {kind!r} not in {list(KINDS)}"]

    missing = [k for k in _COMMON_REQUIRED if k not in entry]
    missing += [k for k in (_MERGED_REQUIRED if kind == "merged"
                            else _BASE_REQUIRED) if k not in entry]
    if missing:
        bad.append(f"missing required key(s) {sorted(missing)}")
    unknown = sorted(set(entry) - _allowed(kind))
    if unknown:
        bad.append(f"unknown key(s) {unknown} — refusing an entry with fields "
                   f"no consumer reads; a typo'd field is a fact the operator "
                   f"believes is recorded and is not")

    if entry.get("schema_version") != SCHEMA_VERSION:
        bad.append(f"schema_version: {entry.get('schema_version')!r} != "
                   f"{SCHEMA_VERSION}")

    ident = entry.get("id")
    if not isinstance(ident, str) or not ID_RE.match(ident):
        bad.append(f"id: {ident!r} is not [a-z0-9][a-z0-9._-]*")
    elif stem is not None and ident != stem:
        bad.append(f"id: {ident!r} != its filename stem {stem!r}")

    root = entry.get("b2_root")
    if not isinstance(root, str) or not root or root.startswith("/"):
        bad.append(f"b2_root: {root!r} is not a relative B2 key prefix")

    for key in ("content_sha256", "fingerprint_sha256"):
        v = entry.get(key)
        if v is None or key not in entry:
            continue                # null = UNMEASURED; see the module docstring
        if not isinstance(v, str) or not HEX64_RE.match(v):
            bad.append(f"{key}: {v!r} is not a 64-hex sha256")

    n = entry.get("n_files")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        bad.append(f"n_files: {n!r} is not a positive integer")

    if kind == "merged":
        bad += _validate_merged(entry, root)
    return bad


def _validate_merged(entry: dict, root) -> list[str]:
    bad: list[str] = []
    ident = entry.get("adapter_ident")
    if not isinstance(ident, str) or not HEX64_RE.match(ident):
        bad.append(f"adapter_ident: {ident!r} is not a 64-hex sha256")
    elif isinstance(root, str):
        want = merged_prefixes(root, ident)
        for key, prefix in want.items():
            got = entry.get(key)
            if got != prefix:
                bad.append(
                    f"{key}: {got!r} != {prefix!r} derived from b2_root + "
                    f"adapter_ident[:{SHA12}]. The prefix is DERIVED — an "
                    f"artifact published under a name that does not carry its "
                    f"adapter's identity is one rename away from serving the "
                    f"wrong weights under the right label.")

    if entry.get("fingerprint_sha256") is None:
        bad.append("fingerprint_sha256: a merged artifact with no grade-A "
                   "fingerprint cannot be re-verified after a restore")
    if entry.get("n_files") is None:
        bad.append("n_files: required for a merged artifact — it is what the "
                   "PUSHED.json receipt is corroborated against")

    serve = entry.get("serve")
    if not isinstance(serve, dict):
        bad.append(f"serve: {serve!r} is not an object")
        return bad
    for k in _SERVE_REQUIRED:
        if k not in serve:
            bad.append(f"serve.{k}: missing")
    unknown = sorted(set(serve) - set(_SERVE_REQUIRED))
    if unknown:
        bad.append(f"serve: unknown key(s) {unknown}")
    for k in ("max_len", "min_vram_gb", "tp"):
        v = serve.get(k)
        if k in serve and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            bad.append(f"serve.{k}: {v!r} is not a positive integer")
    for k in ("served_name", "dtype"):
        v = serve.get(k)
        if k in serve and (not isinstance(v, str) or not v):
            bad.append(f"serve.{k}: {v!r} is not a non-empty string")
    if "lora_forbidden" in serve and not isinstance(serve["lora_forbidden"], bool):
        bad.append(f"serve.lora_forbidden: {serve['lora_forbidden']!r} is not "
                   f"a boolean")
    return bad


# --- load / lookup -----------------------------------------------------------
def load(directory: str | None = None, *, strict: bool = True) -> dict[str, dict]:
    """Every entry in the registry, keyed by id.

    `strict` raises on the FIRST unsound entry. That is the default because a
    partially-loaded registry is the shape that serves the wrong model: a
    consumer asking for `mergeddemoa` and getting a `KeyError` at least fails, while
    one silently reading a half-validated entry does not.
    """
    d = directory or REGISTRY_DIR
    out: dict[str, dict] = {}
    if not os.path.isdir(d):
        raise RegistryError(f"no registry directory at {d}")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        p = os.path.join(d, fn)
        try:
            entry = json.loads(open(p).read())
        except Exception as e:                   # noqa: BLE001
            if strict:
                raise RegistryError(f"{p}: unparseable ({e!r})") from e
            continue
        problems = validate(entry, stem=stem)
        if problems and strict:
            raise RegistryError(f"{p}: " + "; ".join(problems))
        if not problems:
            out[entry["id"]] = entry
    return out


def get(artifact_id: str, directory: str | None = None) -> dict:
    reg = load(directory)
    if artifact_id not in reg:
        raise RegistryError(f"no artifact {artifact_id!r} in the registry "
                            f"(have {sorted(reg)})")
    return reg[artifact_id]


def resolve_base(entry: dict, directory: str | None = None) -> dict:
    """The BASE entry a merged artifact was built from.

    A dangling `base` is refused rather than returned as None: the base is what
    the divergence preflight compares against, and a merged model with no
    resolvable base has no control.
    """
    if entry.get("kind") != "merged":
        raise RegistryError(f"{entry.get('id')!r} is not a merged artifact")
    return get(entry["base"], directory)


def model_prefix(entry: dict) -> str:
    """The B2 key prefix holding the servable/payload directory.

    Merged artifacts publish under `<b2_root>/<sha12>/model` and the registry
    DERIVES that; a base artifact's root IS its payload dir. Canonical here
    rather than in a consumer: `serve_artifact` and the jobs `--artifact` lane
    must not be able to disagree about which prefix holds the weights.
    """
    return entry["b2_model"] if entry["kind"] == "merged" else entry["b2_root"]


#: Env-name suffixes `env_exports` emits, for docs and for a caller that wants
#: to say what a prefix will occupy. `_B2` is the payload prefix — the one a
#: `${<PREFIX>_B2}` asset template is expected to name.
ENV_SUFFIXES = ("B2", "B2_ROOT", "B2_GUARDS", "ID", "KIND", "FAMILY", "BASE_ID",
                "ADAPTER_IDENT", "FINGERPRINT", "N_FILES", "CONTENT_SHA",
                "SERVED_NAME", "MAX_LEN", "DTYPE", "TP", "MIN_VRAM_GB",
                "LORA_FORBIDDEN")


def env_exports(entry: dict, prefix: str) -> dict[str, str]:
    """`{<PREFIX>_<SUFFIX>: str}` — this artifact as submit-time env values.

    The composition half of asset parameterization: a job declares
    `assets[].b2: "${ADAPTER_B2}"` and the operator says `--artifact
    ADAPTER=mergeddemoa`, so the B2 prefix comes from the COMMITTED registry instead
    of an operator's shell history. Everything is stringified (the env channel
    is one-typed, and jobd exports these verbatim); a field a `base` entry does
    not carry maps to "", never to a guess.
    """
    if not re.match(r"^[A-Z][A-Z0-9_]*$", prefix or ""):
        raise RegistryError(
            f"artifact env prefix {prefix!r} must match [A-Z][A-Z0-9_]* — it is "
            f"spliced onto a shell identifier the box exports")
    s = entry.get("serve") or {}
    vals = {
        "B2": model_prefix(entry),
        "B2_ROOT": entry.get("b2_root"),
        "B2_GUARDS": entry.get("b2_guards"),
        "ID": entry.get("id"),
        "KIND": entry.get("kind"),
        "FAMILY": entry.get("family"),
        "BASE_ID": entry.get("base"),
        "ADAPTER_IDENT": entry.get("adapter_ident"),
        "FINGERPRINT": entry.get("fingerprint_sha256"),
        "N_FILES": entry.get("n_files"),
        "CONTENT_SHA": entry.get("content_sha256"),
        "SERVED_NAME": s.get("served_name"),
        "MAX_LEN": s.get("max_len"),
        "DTYPE": s.get("dtype"),
        "TP": s.get("tp"),
        "MIN_VRAM_GB": s.get("min_vram_gb"),
        # A bool has to become a shell-testable token, and "" is the one the
        # `[ -z "$X" ]` ladders everywhere else already understand.
        "LORA_FORBIDDEN": "1" if s.get("lora_forbidden") else "",
    }
    return {f"{prefix}_{k}": ("" if vals[k] is None else str(vals[k]))
            for k in ENV_SUFFIXES}


def check(directory: str | None = None) -> list[str]:
    """Every problem across the whole registry, including cross-entry ones.

    Cross-entry is why this is not just `validate` in a loop: a `base` or
    `family` that does not resolve is well-formed in isolation and broken in
    context, and context is where it is read.
    """
    d = directory or REGISTRY_DIR
    problems: list[str] = []
    entries: dict[str, dict] = {}
    if not os.path.isdir(d):
        return [f"no registry directory at {d}"]

    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            problems.append(f"{fn}: not a .json file — the registry holds one "
                            f"JSON per artifact and nothing else")
            continue
        p = os.path.join(d, fn)
        try:
            entry = json.loads(open(p).read())
        except Exception as e:                   # noqa: BLE001
            problems.append(f"{fn}: unparseable ({e!r})")
            continue
        for prob in validate(entry, stem=fn[:-5]):
            problems.append(f"{fn}: {prob}")
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            entries[entry["id"]] = entry

    try:                                         # pragma: no cover - trivial
        from . import merge_guard
    except ImportError:                          # pragma: no cover - trivial
        import merge_guard                       # type: ignore[no-redef]
    families = set(merge_guard.known_families())

    for ident, entry in sorted(entries.items()):
        base = entry.get("base")
        if entry.get("kind") == "merged" and base not in entries:
            problems.append(f"{ident}.json: base {base!r} is not an entry in "
                            f"this registry")
        elif entry.get("kind") == "merged" and entries[base].get("kind") != "base":
            problems.append(f"{ident}.json: base {base!r} has kind "
                            f"{entries[base].get('kind')!r}, not 'base'")
        fam = entry.get("family")
        if fam is not None and fam not in families:
            problems.append(f"{ident}.json: family {fam!r} has no "
                            f"family_specs/{fam}.json (have {sorted(families)})")
    return problems


# --- folding a publisher's candidate into the committed registry -------------
#: Provenance keys that RESTATE a fact the entry also carries. Each is compared
#: rather than trusted: the box composed both halves, so a disagreement means
#: one of its two writers is wrong and neither can be believed.
_ECHOED = ("adapter_ident", "family", "fingerprint_sha256", "content_sha256",
           "n_files")
#: Same rule where the two halves spell the same fact differently. `pin-base`
#: resolves the base entry through `entry.base`, so a provenance that names a
#: different one must be caught here rather than silently losing to the entry.
_ECHOED_ALIAS = {"base_artifact_id": "base"}


def validate_candidate(cand: dict) -> list[str]:
    """Every problem with a publisher's candidate wrapper. Pure.

    Checks the wrapper, then the entry through `validate`, then the ECHOED
    provenance against the entry. It does NOT reach the filesystem — the
    cross-entry half (does `base` resolve, does `family` have a spec, is the id
    already taken) is `fold`'s, because those are properties of the registry
    being folded into and not of the candidate.
    """
    bad: list[str] = []
    if not isinstance(cand, dict):
        return [f"not a JSON object ({type(cand).__name__})"]
    v = cand.get("candidate_schema_version")
    if v != CANDIDATE_SCHEMA_VERSION:
        bad.append(f"candidate_schema_version: {v!r} != "
                   f"{CANDIDATE_SCHEMA_VERSION}")
    entry = cand.get("entry")
    if not isinstance(entry, dict):
        # Nothing below can run without it, so this is the one problem that
        # stops the check rather than adding to it.
        bad.append(f"entry: {entry!r} is not a JSON object — a candidate with "
                   f"no entry has nothing to fold")
        return bad
    bad += [f"entry: {p}" for p in validate(entry, stem=entry.get("id")
                                            if isinstance(entry.get("id"), str)
                                            else None)]
    if entry.get("kind") != "merged":
        bad.append(f"entry.kind: {entry.get('kind')!r} — only a merged artifact "
                   f"is published mechanically; a base entry is hand-written")

    prov = cand.get("provenance")
    if not isinstance(prov, dict):
        bad.append(f"provenance: {prov!r} is not a JSON object — a mechanically "
                   f"published artifact with no record of what produced it is "
                   f"one nobody can trace back to an adapter or a job")
        return bad
    for key in _ECHOED + tuple(_ECHOED_ALIAS):
        if key not in prov:
            continue                 # optional; what is stated must AGREE
        ekey = _ECHOED_ALIAS.get(key, key)
        if prov[key] != entry.get(ekey):
            bad.append(f"provenance.{key} {prov[key]!r} != entry.{ekey} "
                       f"{entry.get(ekey)!r} — the publishing box wrote both, so "
                       f"a disagreement means one of its writers is wrong")
    if not prov.get("merge_job_id"):
        bad.append("provenance.merge_job_id: missing — the job that minted this "
                   "artifact is the only handle on its logs and its gates")
    return bad


def fold(candidate: dict, directory: str | None = None, *,
         force: bool = False, dry_run: bool = False) -> tuple[str, list[str]]:
    """Write a validated candidate's entry into the registry. (path, problems).

    Problems non-empty means NOTHING was written. Beyond `validate_candidate`
    this adds the three questions only the target registry can answer — does
    `base` resolve to a `kind: base` entry here, does `family` have a spec here,
    and is this id already taken — and then RE-CHECKS the whole registry after
    writing, reverting if the result is invalid. A fold that leaves the registry
    broken would be discovered by the next consumer rather than by the operator
    who ran it.
    """
    d = directory or REGISTRY_DIR
    problems = validate_candidate(candidate)
    if problems:
        return ("", problems)
    entry = dict(candidate["entry"])
    path = os.path.join(d, f"{entry['id']}.json")

    if os.path.exists(path) and not force:
        problems.append(
            f"{path} already exists. A published artifact id is a name other "
            f"files and runbooks point at, so re-pointing it silently is how a "
            f"consumer starts serving different weights under an unchanged "
            f"label. Pass --force to replace it deliberately.")
        return ("", problems)

    try:
        existing = load(d, strict=False)
    except RegistryError as e:
        # A missing directory is a REFUSAL and not an invitation to create one:
        # `--dir` pointing somewhere unexpected is the shape where a fold
        # "succeeds" into a tree no consumer reads.
        return ("", [f"{e}"])
    base = entry.get("base")
    if base not in existing:
        problems.append(f"base {base!r} is not an entry in {d} — publish or "
                        f"commit the base entry first; a merged artifact with "
                        f"no resolvable base has no control to compare against")
    elif existing[base].get("kind") != "base":
        problems.append(f"base {base!r} has kind {existing[base].get('kind')!r}, "
                        f"not 'base'")
    try:                                         # pragma: no cover - trivial
        from . import merge_guard
    except ImportError:                          # pragma: no cover - trivial
        import merge_guard                       # type: ignore[no-redef]
    families = set(merge_guard.known_families())
    if entry.get("family") not in families:
        problems.append(f"family {entry.get('family')!r} has no "
                        f"family_specs/<family>.json (have {sorted(families)})")
    if problems:
        return ("", problems)
    if dry_run:
        return (path, [])

    prior = open(path, "rb").read() if os.path.exists(path) else None
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(entry, fh, indent=1, sort_keys=True)
        fh.write("\n")
    after = check(d)
    if after:
        # REVERT. The pre-write checks are per-entry and cross-entry, but
        # `check` is the acceptance test the whole registry is read through, and
        # leaving a registry that fails it turns every later `load(strict=True)`
        # into a refusal nobody can attribute to this fold.
        if prior is None:
            os.remove(path)
        else:
            with open(path, "wb") as fh:
                fh.write(prior)
        return ("", [f"REVERTED — the registry would not have validated: {p}"
                     for p in after])
    return (path, [])


# --- promoting a publisher's BASE measurement into the base entry ------------
#: What `pin-base` accepts, named in one place because its refusals quote it.
PIN_SOURCES = ("a merge-publish registry candidate, its guards_summary.json, "
               "or a guards/base_pins.json manifest")


def _dirhash():                                  # pragma: no cover - trivial
    """The one rollup definition, under each of its three import spellings:
    package, this file run as a script, and the `jobcommon/` on-box prefix."""
    try:
        from . import dirhash
    except ImportError:
        try:
            import dirhash                        # type: ignore[no-redef]
        except ImportError:
            import modelkit_dirhash as dirhash    # type: ignore[no-redef]
    return dirhash


def _is_manifest(obj: dict) -> bool:
    return bool(obj) and all(isinstance(v, dict) and "size" in v
                             and "sha256" in v for v in obj.values())


def base_pin(source: dict) -> tuple[dict, list[str]]:
    """The base identity a publisher measured, from any of `PIN_SOURCES`. Pure.

    `{"content_sha256", "n_files", "base", "source"}` — `base` is None for the
    two shapes that carry no artifact name, and `n_files` is None for a
    candidate published before the count was threaded into provenance. A
    missing count is left UNMEASURED rather than derived from something else:
    the rollup and the count must describe the same listing or neither does.
    """
    if not isinstance(source, dict):
        return ({}, [f"not a JSON object ({type(source).__name__})"])

    if "candidate_schema_version" in source or "entry" in source:
        problems = validate_candidate(source)
        if problems:
            return ({}, problems)
        prov = source["provenance"]
        if not prov.get("base_rollup_sha256"):
            return ({}, ["provenance.base_rollup_sha256: missing — this run "
                         "published without a --base-dir, so it never measured "
                         "the base and there is nothing to promote"])
        return ({"content_sha256": prov["base_rollup_sha256"],
                 "n_files": prov.get("base_n_files"),
                 "base": source["entry"].get("base"),
                 "source": "candidate"}, [])

    if "base_rollup_sha256" in source:            # guards_summary.json
        return ({"content_sha256": source["base_rollup_sha256"],
                 "n_files": source.get("base_n_files"),
                 "base": None, "source": "guards_summary"}, [])

    if _is_manifest(source):                      # guards/base_pins.json
        # RECOMPUTED, not read: this shape states no rollup, so promoting it
        # measures the pins file rather than believing a number beside it.
        return ({"content_sha256": _dirhash().rollup(source),
                 "n_files": len(source), "base": None,
                 "source": "base_pins"}, [])

    if "content_sha256" in source or "fingerprint_sha256" in source:
        # A guards summary / fingerprint for the MERGED dir. Its hashes are the
        # artifact's, not the base's, and `fold` is where those belong.
        return ({}, ["this names the MERGED dir's hashes, not the base's — the "
                     "run published without a --base-dir, so it never measured "
                     "the base and there is nothing to promote"])
    return ({}, [f"unrecognised shape — expected {PIN_SOURCES}"])


def _json_indent(raw: str) -> int:
    """The file's own indent, so a two-field pin reads as a two-field diff."""
    for line in raw.splitlines():
        stripped = line.lstrip(" ")
        if stripped and stripped != line and stripped[0] in "\"}]":
            return len(line) - len(stripped)
    return 1                                      # `fold`'s


def pin_base(source: dict, base_id: str | None = None,
             directory: str | None = None, *, force: bool = False,
             dry_run: bool = False) -> tuple[str, list[str], str]:
    """Write a measured base identity into its base entry. (path, problems, note).

    Problems non-empty means NOTHING was written. Like `fold` it re-runs `check`
    over the whole registry afterwards and REVERTS if the result would not
    validate. Unlike `fold` it edits an entry that must already exist: a base
    entry is hand-written and committed, and minting one from a measurement
    would create an artifact record out of a hash.

    An EMPTY `note` with no problems is the no-op: the pin already says this.
    The publisher measures the base on every merge, so an operator folding the
    second artifact off one snapshot must not have to remember whether the
    first already pinned it.
    """
    d = directory or REGISTRY_DIR
    measured, problems = base_pin(source)
    if problems:
        return ("", problems, "")

    want = base_id or measured["base"]
    if not want:
        return ("", [f"no base artifact id: a {measured['source']} carries the "
                     f"base's BYTES but not its name, so --base-id says which "
                     f"entry these belong to"], "")
    if base_id and measured["base"] and base_id != measured["base"]:
        return ("", [f"--base-id {base_id!r} != the candidate's own base "
                     f"{measured['base']!r} — pinning one artifact's rollup "
                     f"onto another is how a base entry acquires an identity "
                     f"nothing was ever published against"], "")

    sha, n = measured["content_sha256"], measured["n_files"]
    if not isinstance(sha, str) or not HEX64_RE.match(sha):
        return ("", [f"measured content_sha256 {sha!r} is not a 64-hex sha256"],
                "")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        return ("", [f"measured n_files {n!r} is not a positive integer"], "")

    if not os.path.isdir(d):
        return ("", [f"no registry directory at {d}"], "")
    path = os.path.join(d, f"{want}.json")
    if not os.path.isfile(path):
        return ("", [f"no entry {want!r} in {d} — `pin-base` fills a hole in a "
                     f"committed base entry, it does not mint one"], "")
    raw = open(path).read()
    try:
        entry = json.loads(raw)
    except Exception as e:                       # noqa: BLE001
        return ("", [f"{path}: unparseable ({e!r})"], "")
    for prob in validate(entry, stem=want):
        problems.append(f"{path}: {prob}")
    if problems:
        return ("", problems + ["refusing to edit an entry that does not "
                                "already validate"], "")
    if entry.get("kind") != "base":
        return ("", [f"{want!r} has kind {entry.get('kind')!r}, not 'base' — a "
                     f"merged entry's own hashes come from `fold`"], "")

    clash = []
    if entry.get("content_sha256") not in (None, sha):
        clash.append(f"content_sha256: registry {entry['content_sha256']} != "
                     f"measured {sha}")
    if n is not None and entry.get("n_files") not in (None, n):
        clash.append(f"n_files: registry {entry['n_files']} != measured {n}")
    if clash and not force:
        return ("", [
            f"{want}: the measured base identity DISAGREES with the pin already "
            f"in this registry. A base snapshot is IMMUTABLE — one id, one set "
            f"of bytes — so this is NOT a stale pin to refresh. Either the B2 "
            f"prefix runbooks point at now holds different bytes than when it "
            f"was pinned, or the run that produced this measurement merged "
            f"against a different snapshot and published an artifact whose "
            f"`base` is a lie. Something already shipped is wrong and no "
            f"re-pin makes it right: find out WHICH before --force, because "
            f"--force overwrites the value every earlier claim was made "
            f"against.", *clash], "")

    changes = {"content_sha256": sha}
    if n is not None:
        changes["n_files"] = n
    if all(entry.get(k) == v for k, v in changes.items()):
        return (path, [], "")

    note = (f"grade B {sha[:12]}… over "
            + (f"{n} files" if n is not None else
               "an UNMEASURED file count (this source carries no base_n_files)"))
    if dry_run:
        return (path, [], note)

    entry.update(changes)                # in place: preserves the key order
    with open(path, "w") as fh:
        json.dump(entry, fh, indent=_json_indent(raw))
        fh.write("\n")
    after = check(d)
    if after:
        with open(path, "w") as fh:
            fh.write(raw)
        return ("", [f"REVERTED — the registry would not have validated: {p}"
                     for p in after], "")
    return (path, [], note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=None, help="registry directory override")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls", help="one line per artifact")
    sub.add_parser("check", help="validate every entry; exit 1 on any problem")
    s = sub.add_parser("show", help="print one entry as JSON")
    s.add_argument("id")
    pp = sub.add_parser("path", help="print a merged artifact's B2 model prefix")
    pp.add_argument("id")
    pp.add_argument("--leaf", default="model", choices=list(MERGED_LEAVES))
    fo = sub.add_parser("fold", help="validate a publisher's registry candidate "
                                     "and write its entry into the registry")
    fo.add_argument("candidate", help="the candidate JSON a merge-publish job "
                                      "left in its results")
    fo.add_argument("--force", action="store_true",
                    help="replace an existing entry with this id")
    fo.add_argument("--dry-run", action="store_true",
                    help="run every check and write nothing")
    pb = sub.add_parser("pin-base", help="promote a publisher's MEASURED base "
                                         "identity into that base entry")
    pb.add_argument("source", help=f"{PIN_SOURCES}")
    pb.add_argument("--base-id", default=None,
                    help="which base entry these bytes are; required unless the "
                         "source is a candidate (which names its own base)")
    pb.add_argument("--force", action="store_true",
                    help="overwrite a DISAGREEING pin. Read the refusal first: "
                         "a base snapshot is immutable, so a disagreement means "
                         "something already published is wrong")
    pb.add_argument("--dry-run", action="store_true",
                    help="run every check and write nothing")
    a = ap.parse_args(argv)

    if a.cmd == "pin-base":
        try:
            src = json.loads(open(a.source).read())
        except Exception as e:                   # noqa: BLE001
            print(f"!! {a.source}: unreadable/unparseable ({e!r})",
                  file=sys.stderr)
            return 1
        path, problems, note = pin_base(src, a.base_id, a.dir, force=a.force,
                                        dry_run=a.dry_run)
        for p in problems:
            print(f"!! {p}", file=sys.stderr)
        if problems:
            print(f"!! REFUSED {a.source}: {len(problems)} problem(s); nothing "
                  f"was written", file=sys.stderr)
            return 1
        if not note:
            print(f">> {path} already carries this pin — nothing to do")
            return 0
        if a.dry_run:
            print(f">> would pin {path}: {note} (--dry-run: nothing written)")
            return 0
        print(f">> pinned {path}: {note}")
        print(f">> COMMIT IT: until it is in git the base is still UNMEASURED "
              f"to every consumer — `null` never means clean.")
        return 0

    if a.cmd == "fold":
        try:
            cand = json.loads(open(a.candidate).read())
        except Exception as e:                   # noqa: BLE001
            print(f"!! {a.candidate}: unreadable/unparseable ({e!r})",
                  file=sys.stderr)
            return 1
        path, problems = fold(cand, a.dir, force=a.force, dry_run=a.dry_run)
        for p in problems:
            print(f"!! {p}", file=sys.stderr)
        if problems:
            print(f"!! REFUSED {a.candidate}: {len(problems)} problem(s); "
                  f"nothing was written", file=sys.stderr)
            return 1
        ident = cand["entry"]["id"]
        if a.dry_run:
            print(f">> would fold {ident} -> {path} (--dry-run: nothing written)")
            return 0
        print(f">> folded {ident} -> {path}")
        print(f">> COMMIT IT: a registry entry is only real once it is in git — "
              f"the box that published the weights could not write here.")
        # The same run measured the BASE for free. Say so here or the number
        # stays in provenance nobody reads and the base entry keeps its null.
        _, probs, note = pin_base(cand, None, a.dir, dry_run=True)
        if not probs and note:
            print(f">> AND the base is still unmeasured — this candidate can "
                  f"fill it: registry.py pin-base <candidate> ({note})")
        return 0

    if a.cmd == "check":
        problems = check(a.dir)
        for p in problems:
            print(f"!! {p}", file=sys.stderr)
        if problems:
            print(f"!! REGISTRY INVALID: {len(problems)} problem(s)",
                  file=sys.stderr)
            return 1
        print(f">> registry OK: {len(load(a.dir))} artifact(s)")
        return 0

    try:
        if a.cmd == "ls":
            for ident, e in sorted(load(a.dir).items()):
                extra = (e["serve"]["served_name"] if e["kind"] == "merged"
                         else e["b2_root"])
                print(f"{ident:<16} {e['kind']:<7} {extra}")
            return 0
        if a.cmd == "show":
            print(json.dumps(get(a.id, a.dir), indent=1, sort_keys=True))
            return 0
        e = get(a.id, a.dir)
        if e["kind"] != "merged":
            print(f"!! {a.id} is kind={e['kind']}; only a merged artifact has "
                  f"{a.leaf}/ prefixes (base root: {e['b2_root']})",
                  file=sys.stderr)
            return 1
        print(e[f"b2_{a.leaf}"])
        return 0
    except RegistryError as e:
        print(f"!! {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
