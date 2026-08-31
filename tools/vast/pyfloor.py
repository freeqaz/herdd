"""Box python-floor syntax gate: refuse to SHIP code the box cannot parse.

The box runs whatever python its image ships (stock pytorch images: 3.11),
not what the shipping workstation runs — and a checkout that stages the jobd
bundle may itself be stale (the fleetd daemon runs its own checkout), so the
gate must bind at stage/attach time, on the file bytes, in every shipper.
Detector rationale + the authoritative container cross-check live in
test_box_python_floor.py, which imports these symbols.
"""

from __future__ import annotations

import io
import tokenize

# Raise this only with a measurement: the floor is the OLDEST python any image
# we launch on can ship, not the oldest we would like to support.
BOX_PYTHON_FLOOR = (3, 11)


def pep701_violations(src: str) -> list[tuple[int, str]]:
    """Constructs inside f-strings that 3.11's tokenizer rejects.

    Two shapes, both introduced by PEP 701 and both silent on 3.12+:
      (a) a replacement field whose expression spans a newline, in an f-string
          that is not triple-quoted;
      (b) a nested string inside a replacement field reusing the ENCLOSING
          f-string's own quote character.
    A token scan, not ast.parse(feature_version=...): PEP 701 lives in the
    tokenizer, which feature_version does not gate (measured 2026-08-19).
    Returns (lineno, what) for each. Empty list = clean at the floor.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        # Unparseable by THIS interpreter is a different failure; callers
        # report it rather than silently passing.
        return [(0, "file does not tokenize on the running interpreter")]

    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    if fstring_start is None:          # running ON the floor: nothing to detect
        return []

    bad: list[tuple[int, str]] = []
    stack: list[tuple[int, str, bool]] = []   # (row, quote, is_triple)
    for tok in toks:
        if tok.type == fstring_start:
            q = tok.string.lstrip("fFrRbB")
            stack.append((tok.start[0], q[:1], len(q) >= 3))
        elif tok.type == fstring_end:
            if stack:
                stack.pop()
        elif stack:
            row, quote, triple = stack[-1]
            if not triple and tok.start[0] > row and tok.type != tokenize.NL:
                bad.append((row, "multi-line expression in an f-string "
                                 "replacement field (PEP 701, 3.12+)"))
                stack[-1] = (tok.start[0], quote, triple)   # report once
            elif (tok.type == tokenize.STRING and not triple
                    and tok.string.lstrip("fFrRbB")[:1] == quote):
                bad.append((tok.start[0], f"nested {quote} inside an "
                                          f"f-string quoted with {quote} "
                                          f"(PEP 701, 3.12+)"))
    return bad


def floor_gaps(paths: list[str]) -> list[str]:
    """One formatted line per floor violation across *paths* (.py only).
    Empty list = every file parses at BOX_PYTHON_FLOOR."""
    lines: list[str] = []
    for p in paths:
        if not p.endswith(".py"):
            continue
        try:
            src = open(p, encoding="utf-8").read()
        except OSError as e:
            lines.append(f"{p}: unreadable ({e})")
            continue
        for ln, what in pep701_violations(src):
            lines.append(f"{p}:{ln}: {what}")
    return lines
