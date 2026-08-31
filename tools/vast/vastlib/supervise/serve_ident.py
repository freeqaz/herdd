"""The SERVE_STATUS identity fields, parsed ONCE for every python reader.

`serve_ready.sh` owns the BASH copy of this grammar (`marker_models` /
`marker_ident`) because it runs where no python of ours is guaranteed. This is
the python copy, and it exists so the daemon side does not grow a THIRD reader:
`supervise.replacement` already reads that marker for the self-park signal and
the boot SLA, and a second parse of the same line — written from the same
paragraph of prose, months apart — is how two readers come to disagree about
what a line means. `test_fleetd_serve_identity.py` drives ONE table of marker
lines through both copies, so a change to either is a red test.

THE GRAMMAR (see serve_ready.sh for the writer's half):

    READY <ts-utc> <served-ids-csv|-> [ident=<sha12>]

POSITIONAL and APPEND-ONLY. Field 3 has been the id CSV since the marker
existed, `-` is the placeholder a box writes when it verified an identity but
parsed no ids, and `ident=` is therefore always field 4 when present. A marker
with no 4th field is a box that never gated its own weights — which is a
LEGACY/UNARMED launch, not a failure, and telling those two apart is most of
what `classify` is for.

WHY `unarmed` IS NOT `mismatch`. An absent claim and a wrong claim have
opposite remedies: the first says the box was launched without
`--model-artifact` (fix the launch), the second says the box is serving weights
that are not the artifact this watch was registered for (stop using it). A
reader that collapses them sends an operator to the wrong half of the system,
which is the same failure mode `serve_identity_gate.py` split
IDENTITY_MISMATCH from IDENTITY_CANNOT_CHECK to avoid.
"""
from __future__ import annotations

from typing import Any, Sequence

IDENT_PREFIX = "ident="
#: field 3 when the box verified an identity but parsed no served ids
MODELS_PLACEHOLDER = "-"

#: The `FAILED <ts> <reason>` markers `onstart/serve_vllm.sh` writes from the
#: on-box gate. Kept as an ordered tuple, not a set, so the remedy table below
#: and every rendering of it stay in one declared order.
FAILED_REASONS = ("identity_mismatch", "identity_cannot_check",
                  "identity_expect_missing", "identity_gate_missing")

#: What an operator does next, per gate refusal. The reasons are separate
#: because the remedies are opposite — a mismatch means the WEIGHTS are wrong,
#: a cannot-check means the GATE is broken — and an alarm that prints one
#: sentence for both sends someone hunting a model defect that does not exist.
FAILED_REMEDY = {
    "identity_mismatch":
        "the box's own gate refused the weights it pulled: they are NOT the "
        "artifact this serve was launched for. The box never served. Fix the "
        "artifact or the registry pin before relaunching",
    "identity_cannot_check":
        "the GATE broke, not the weights — merged_fingerprint.py/dirhash.py "
        "missing on the box, or an unreadable expectation. Check what "
        "launch_serve.sh staged into the per-serve B2 prefix",
    "identity_expect_missing":
        "the box found no identity_expect.json — the launch shipped no "
        "expectation. Relaunch with --model-artifact, or serve ungated on "
        "purpose with --model b2:<root>",
    "identity_gate_missing":
        "serve_identity_gate.py never reached the box — a staging failure in "
        "launch_serve.sh's ident_assets, not a model problem",
}

#: Every state `classify` can return. `off` means this watch carries no
#: expectation at all, and it is the ONLY state a legacy watch can reach.
STATES = ("off", "unreadable", "pending", "verified", "unarmed", "mismatch",
          "gate_failed")

#: The states that are worth an operator's attention. `verified` and `pending`
#: are the healthy shapes; `off`/`unreadable` are "this instrument has nothing
#: to say", which is never an alarm — an unreadable marker is a B2 blip and
#: alarming on it would make every network wobble look like a poisoned serve.
ALARM_STATES = ("mismatch", "unarmed", "gate_failed")


def _fields(rest: Sequence[str]) -> tuple[str, str]:
    """`(models-csv, ident)` from the marker tokens AFTER the timestamp.

    The one implementation. Both entry points below differ only in how much of
    the line they were handed, which is a property of their caller and not of
    the grammar.
    """
    models = rest[0] if rest else ""
    if models == MODELS_PLACEHOLDER:
        models = ""                      # a placeholder, not a model named "-"
    ident = ""
    for tok in rest[1:]:
        if tok.startswith(IDENT_PREFIX):
            ident = tok[len(IDENT_PREFIX):]
            break
    return models, ident


def marker_fields(line: str) -> tuple[str, str]:
    """`(models-csv, ident)` from a WHOLE marker line — `serve_ready.sh`'s shape."""
    return _fields(line.split()[2:])


def detail_fields(detail: str | None) -> tuple[str, str]:
    """`(models-csv, ident)` from `_serve_status_line_soft`'s `detail`, which is
    already the line minus its token and timestamp."""
    return _fields((detail or "").split())


def marker_models(line: str) -> str:
    return marker_fields(line)[0]


def marker_ident(line: str) -> str:
    return marker_fields(line)[1]


def failed_reason(detail: str | None) -> str | None:
    """The identity reason on a `FAILED` line, or None for any other failure.

    None is deliberately the answer for a non-identity FAILED: other machinery
    owns those, and claiming them here would put an identity remedy on an
    out-of-memory crash.
    """
    for tok in (detail or "").split():
        if tok in FAILED_REASONS:
            return tok
    return None


def classify(expect_ident: str | None, token: str | None,
             detail: str | None) -> dict[str, Any]:
    """The identity verdict for one marker reading. PURE — no I/O, no clock.

    `expect_ident` is the sha12 the watch was registered with. Falsy means this
    watch never claimed to know what the box should be serving, and the answer
    is `off` before the marker is even looked at: a legacy watch must not grow
    state, an alarm, or a code path it did not have before.
    """
    expect = (expect_ident or "").strip().lower()
    if not expect:
        return {"state": "off", "expected": None, "observed": None,
                "reason": None}
    if not token:
        return {"state": "unreadable", "expected": expect, "observed": None,
                "reason": None}
    if token == "READY":
        _models, ident = detail_fields(detail)
        if not ident:
            return {"state": "unarmed", "expected": expect, "observed": None,
                    "reason": None}
        if ident.lower() != expect:
            return {"state": "mismatch", "expected": expect,
                    "observed": ident, "reason": None}
        return {"state": "verified", "expected": expect, "observed": ident,
                "reason": None}
    if token == "FAILED":
        reason = failed_reason(detail)
        if reason:
            return {"state": "gate_failed", "expected": expect,
                    "observed": None, "reason": reason}
    # LAUNCHED / PULLING / SELF_PARKED / a FAILED nobody here owns: the box has
    # not answered the identity question yet, and silence is not evidence.
    return {"state": "pending", "expected": expect, "observed": None,
            "reason": None}
