#!/usr/bin/env python3
"""The serve lane's door into the modelkit REGISTRY: resolve, gate, freeze.

`launch_serve.sh --model-artifact <slug>` is bash, so this is the one python
helper it shells out to — the same shape as `resolve_bakeoff_row`, and for the
same reason: a manifest lookup belongs somewhere it can be tested.

THREE VERBS, AND THEY ARE DELIBERATELY SEPARATE:

    resolve <slug>      shell K=V from the committed registry. NO network.
    expect  <slug>      the frozen identity EXPECTATION the box is gated on.
                        NO network — see below, this is the load-bearing part.
    gate    <slug>      the PRE-SPEND B2 check: present, complete, right count.
                        Network (rclone), read-only, fail-closed.

WHY `expect` MUST NOT TOUCH B2. The expectation is the operator's INTENT — "the
thing I asked to serve is artifact X, whose grade-A fingerprint is this". If it
were composed from the guards published beside the weights, the box would be
comparing B2 against B2 and every mismatch class this exists to catch (a
re-published prefix, a rename, a half-synced restore that happens to agree with
its own receipt) would corroborate itself. The committed registry JSON is in
git; B2 is not. Only one of those is a claim somebody signed.

WHAT THE GATE IS AND IS NOT. `gate` protects MONEY: it refuses to rent a box for
an artifact that is absent, partial, or the wrong size. It is not the identity
gate — that runs ON THE BOX, after the pull, over the bytes that were actually
downloaded (`serve_identity_gate.py`). A workstation-side check of a remote
listing cannot say anything about what the box received.

GRADE-B IS PRESENT ONLY WHERE MEASURED. `content_sha256: null` means exactly
that, never "clean". An artifact with a null grade-B pin is gated at GRADE A
with a loud, distinct line at every layer; minting the pin is the operator's
next $0-on-box step (`modelkit/gate_dir.py --emit` on a verified dir — a
`merge-publish-v1` fold arrives with it already measured).

exit 0 ok · 2 usage / unknown slug / no identity pin · 9 pre-spend refusal.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modelkit import registry  # noqa: E402  — after the sys.path pin

SCHEMA_VERSION = 1

#: Transport artefacts that live in the payload prefix but are not payload.
#: Same set as `merged_fingerprint.DEFAULT_EXCLUDE` / `dirhash.DEFAULT_IGNORE`,
#: restated here because this module counts REMOTE keys, not local files.
TRANSPORT_KEYS = ("PUSHED.json", ".complete")

EXIT_USAGE = 2
EXIT_REFUSED = 9


class Refusal(Exception):
    """A reason not to spend money. Never degraded to a warning."""


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #
def serve_fields(entry: dict) -> dict:
    """The `serve` block, or the honest empty defaults for a `base` artifact.

    A base entry carries no `serve` block by schema, so every serving knob is
    "the caller's or the tool's default" rather than a registry value. Returning
    empty strings (not None) keeps the shell contract one-typed: an empty K=V is
    exactly the `[ -z "$X" ]` the precedence ladder already tests.
    """
    s = entry.get("serve") or {}
    return {
        "served_name": s.get("served_name") or entry["id"],
        "dtype": s.get("dtype") or "",
        "max_len": s.get("max_len") or "",
        "min_vram_gb": s.get("min_vram_gb") or "",
        "tp": s.get("tp") or "",
        "lora_forbidden": "1" if s.get("lora_forbidden") else "",
    }


def model_prefix(entry: dict) -> str:
    """The B2 key prefix holding the servable directory.

    Delegates: the jobs `--artifact` lane composes the same prefix, and two
    copies of "which dir holds the weights" is a divergence nobody would notice.
    Kept as a module attribute because callers and tests bind this name.
    """
    return registry.model_prefix(entry)


def identity_pin(entry: dict) -> tuple[str, str | None, int | None, str | None]:
    """`(grade, fingerprint_sha256, n_files, content_sha256)` for this artifact.

    Grade A is the FLOOR and grade B is layered on it, never an alternative:
    `serve_identity_gate.validate_expectation` refuses an expectation with no
    `fingerprint_sha256`, and the `ident=` a verified box stamps into its marker
    IS that fingerprint truncated. So a content rollup measured WITHOUT one
    pins nothing — composing an expectation from it turns this pre-spend
    refusal into a failure on a box that is already rented.

    `grade` is "" when there is no usable pin, which is not a weaker gate but
    no gate, and every caller is required to refuse on it.
    """
    fp = entry.get("fingerprint_sha256")
    n = entry.get("n_files")
    content = entry.get("content_sha256")
    if not (fp and n):
        # `content` is reported even here, so a refusal can say the rollup IS
        # present and still unusable rather than claiming it is missing.
        return "", fp, n, content
    return ("B", fp, n, content) if content else ("A", fp, n, None)


def resolve_lines(entry: dict) -> list[str]:
    """`AR_*=<shell-quoted>` lines for `eval` in launch_serve.sh."""
    sv = serve_fields(entry)
    grade, fp, n_files, content = identity_pin(entry)
    out = {
        "AR_ID": entry["id"],
        "AR_KIND": entry["kind"],
        "AR_MODEL_B2": model_prefix(entry),
        "AR_SERVED_NAME": sv["served_name"],
        "AR_MAX_LEN": sv["max_len"],
        "AR_DTYPE": sv["dtype"],
        "AR_TP": sv["tp"],
        "AR_MIN_VRAM_GB": sv["min_vram_gb"],
        "AR_LORA_FORBIDDEN": sv["lora_forbidden"],
        "AR_GRADE": grade,
        "AR_FINGERPRINT": fp or "",
        "AR_N_FILES": n_files if n_files is not None else "",
        "AR_CONTENT_SHA": content or "",
    }
    return [f"{k}={shlex.quote(str(v))}" for k, v in out.items()]


# --------------------------------------------------------------------------- #
# expect — the frozen identity expectation
# --------------------------------------------------------------------------- #
def compose_expectation(entry: dict, *, now: str | None = None) -> dict:
    """The operator-side intent, from the COMMITTED registry and nothing else.

    Refuses an artifact with no identity pin rather than emitting an
    expectation that cannot fail. `--model-artifact` is the gated door; an
    artifact whose pins are null still has the ungated one (`--model b2:<root>`),
    and saying so is the difference between a refusal and a dead end.
    """
    grade, fp, n_files, content = identity_pin(entry)
    if not grade:
        # A rollup with no grade-A fingerprint is the confusing case: somebody
        # just ran `pin-base` and the entry looks measured. Say why it is not
        # enough rather than let the generic wording read as "nothing is there".
        why = ("its content rollup is measured but grade A is NOT, and grade A "
               "is the floor the on-box gate validates against (`ident=` in the "
               "SERVE_STATUS marker IS that fingerprint truncated)"
               if content else
               "neither hash is measured; `null` means UNMEASURED, never clean")
        raise Refusal(
            f"artifact {entry['id']!r} carries NO identity pin — {why} "
            f"(fingerprint_sha256={fp!r}, n_files={n_files!r}, "
            f"content_sha256={content!r}). There is nothing for the box to "
            f"check itself against, and a gate that cannot fail is not a "
            f"gate.\n"
            f"  mint grade A:  python3 tools/vast/modelkit/merged_fingerprint.py "
            f"--dir <verified dir> --emit\n"
            f"  mint grade B:  python3 tools/vast/modelkit/gate_dir.py "
            f"--dir <verified dir> --emit\n"
            f"  …then record it in "
            f"tools/vast/modelkit/registry/{entry['id']}.json.\n"
            f"  To serve it UNGATED (today's behaviour, no identity claim): "
            f"--model b2:{model_prefix(entry)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": entry["id"],
        "kind": entry["kind"],
        "b2_model": model_prefix(entry),
        "served_name": serve_fields(entry)["served_name"],
        "grade": grade,
        "fingerprint_sha256": fp,
        "n_files": n_files,
        "content_sha256": content,
        "composed_utc": now or datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": f"tools/vast/modelkit/registry/{entry['id']}.json",
    }


# --------------------------------------------------------------------------- #
# gate — the pre-spend B2 check
# --------------------------------------------------------------------------- #
def _rclone(args: list[str]) -> tuple[int, str, str]:
    """One rclone call. Seam for tests — every B2 read here goes through it."""
    try:
        p = subprocess.run(["rclone", *args], capture_output=True, text=True,
                           timeout=300)
    except FileNotFoundError:
        return 127, "", "rclone not on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", "rclone timed out"
    return p.returncode, p.stdout, p.stderr


def payload_keys(listing: str) -> list[str]:
    """Payload object names from `rclone lsf -R`, transport artefacts removed."""
    return [ln.strip() for ln in listing.splitlines()
            if ln.strip() and ln.strip() not in TRANSPORT_KEYS]


def check_remote(entry: dict, bucket: str, *,
                 rclone=_rclone) -> tuple[int, list[str]]:
    """`(bytes, notes)` for a present-and-complete artifact; raise Refusal else.

    Three independent facts, because each catches something the others miss:
      the completion MARKER   the publisher declared this prefix finished
      the object COUNT        against the registry's own `n_files`
      the byte TOTAL          which is also what sizes the box's disk

    `has`-style marker presence alone is what let a short restore through
    before: the marker is one stat and says nothing about the payload beside it.
    """
    prefix = f"b2:{bucket}/{model_prefix(entry)}"
    notes: list[str] = []

    rc, out, err = rclone(["lsf", "--files-only", "-R", prefix])
    if rc != 0:
        raise Refusal(f"cannot LIST {prefix} (rclone rc={rc}: "
                      f"{(err or '').strip()[:200]}) — cannot check is a "
                      f"REFUSAL, not a skip")
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not names:
        raise Refusal(f"{prefix}: ABSENT (no objects). Publish it first, or "
                      f"check the adapter_ident the prefix is derived from")

    marker = "PUSHED.json" if entry["kind"] == "merged" else ".complete"
    if marker not in names:
        raise Refusal(
            f"{prefix}: no {marker} — the publisher never declared this prefix "
            f"COMPLETE. It is written LAST, after a read-back, so its absence "
            f"is exactly the half-published prefix the ordering exists to make "
            f"visible")
    notes.append(f"{marker} present")

    n_want = entry.get("n_files")
    payload = payload_keys(out)
    if n_want is not None:
        if len(payload) != n_want:
            raise Refusal(
                f"{prefix}: {len(payload)} payload objects, registry says "
                f"{n_want} — a short or fat publish. The marker cannot see "
                f"this (it only stats itself)")
        notes.append(f"{len(payload)} objects == registry n_files")
    else:
        notes.append(f"{len(payload)} objects (registry n_files is null — "
                     f"UNCOUNTED, not verified)")

    if entry["kind"] == "merged":
        rc, out2, err = rclone(["cat", f"{prefix}/PUSHED.json"])
        if rc != 0:
            raise Refusal(f"{prefix}/PUSHED.json exists but is unreadable "
                          f"(rclone rc={rc}: {(err or '').strip()[:200]})")
        try:
            rec = json.loads(out2)
        except Exception as e:                        # noqa: BLE001
            raise Refusal(f"{prefix}/PUSHED.json is not JSON ({e!r}) — a "
                          f"receipt that exists and cannot be believed") from e
        if rec.get("complete") is not True:
            raise Refusal(f"{prefix}/PUSHED.json.complete="
                          f"{rec.get('complete')!r} != True")
        if n_want is not None and rec.get("files") != n_want:
            raise Refusal(f"{prefix}/PUSHED.json.files={rec.get('files')!r} "
                          f"but the registry pins n_files={n_want} — the "
                          f"publisher and the registry disagree about what was "
                          f"published")
        notes.append("PUSHED.json corroborates the registry count")

    rc, out3, err = rclone(["size", "--json", prefix])
    n_bytes = 0
    if rc == 0:
        try:
            n_bytes = int(json.loads(out3).get("bytes") or 0)
        except Exception:                             # noqa: BLE001
            n_bytes = 0
    if n_bytes <= 0:
        # Not fatal: the disk sizer already degrades to its static default with
        # a loud UNMEASURED note, and refusing here would trade a correct
        # launch for a missing measurement.
        notes.append("byte total UNMEASURED (rclone size failed) — the disk "
                     "auto-size falls back to its static default")
    else:
        notes.append(f"{n_bytes} bytes")
    return n_bytes, notes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _entry(slug: str, directory: str | None) -> dict:
    try:
        return registry.get(slug, directory)
    except registry.RegistryError as e:
        print(f"!! --model-artifact: {e}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE) from e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=None, help="registry directory override")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="AR_*=... shell assignments")
    r.add_argument("slug")

    e = sub.add_parser("expect", help="the frozen identity expectation JSON")
    e.add_argument("slug")
    e.add_argument("--out", help="write here instead of stdout")

    g = sub.add_parser("gate", help="pre-spend B2 verification (read-only)")
    g.add_argument("slug")
    g.add_argument("--bucket", required=True)
    g.add_argument("--quiet", action="store_true",
                   help="notes to stderr only; stdout stays the byte total")

    a = ap.parse_args(argv)
    entry = _entry(a.slug, a.dir)

    if a.cmd == "resolve":
        print("\n".join(resolve_lines(entry)))
        return 0

    if a.cmd == "expect":
        try:
            doc = compose_expectation(entry)
        except Refusal as ex:
            print(f"!! --model-artifact {a.slug}: {ex}", file=sys.stderr)
            return EXIT_USAGE
        blob = json.dumps(doc, indent=1, sort_keys=True)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(blob + "\n")
        else:
            print(blob)
        return 0

    try:
        n_bytes, notes = check_remote(entry, a.bucket)
    except Refusal as ex:
        print(f"!! artifact-gate: {a.slug}: {ex}", file=sys.stderr)
        print(f"!!   REFUSING to launch — nothing has been spent.",
              file=sys.stderr)
        return EXIT_REFUSED
    for n in notes:
        print(f">> artifact-gate: {a.slug}: {n}", file=sys.stderr)
    print(n_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
