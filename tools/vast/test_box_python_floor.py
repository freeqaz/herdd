"""Every file ship_manifest.txt puts on a box must parse at the BOX python floor.

The box runs whatever python its IMAGE ships, which is not what this workstation
runs. Stock `pytorch/pytorch:2.4.0-*` ships 3.11.9; our baked train image is
newer. Box 48089639 (2026-08-19) parked with its whole python half dead because
`bidpolicy.py` used a multi-line expression inside an f-string replacement field
— PEP 701, legal only on 3.12+ — and `jobmeta` imports bidpolicy at module
scope, so jobd's boot selftest failed closed and 20 queued arms were stranded.

Why this is a TOKEN scan and not `ast.parse(feature_version=...)`: measured
2026-08-19, `ast.parse(src, feature_version=(3, 11))` accepts the offending
file. PEP 701 moved f-string handling into the tokenizer, and feature_version
does not gate tokenizer-level grammar — so an in-process parse gate reports
clean on exactly the bug that killed the box. The authoritative check is
`py_compile` under a real floor interpreter (see FLOOR_INTERPRETER_CMD below);
this scan is the portable stand-in that runs everywhere with no container.

The detector itself lives in pyfloor.py (prod code): the jobd stage/attach
gate calls it too, because the test suite never runs in the checkout that
matters most — fleetd's own, which re-stages the bundle on every replacement
launch and re-attach (that is how the fixed bidpolicy regressed onto boxes
48094838/48132001 hours after the fix landed here).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfloor import BOX_PYTHON_FLOOR, pep701_violations

# The authoritative cross-check, kept here so it is not folklore:
#   docker run --rm -v <repo>:/repo:ro pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
#     python3 -c "import py_compile; py_compile.compile('/repo/<file>', doraise=True)"
FLOOR_INTERPRETER_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_MANIFEST = _HERE / "ship_manifest.txt"


def shipped_python_files() -> list[Path]:
    """Repo-relative *.py entries of ship_manifest.txt that exist on disk."""
    out = []
    for raw in _MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.endswith(".py"):
            continue
        p = _REPO / line
        if p.is_file():
            out.append(p)
    return out


def test_manifest_lists_python_files():
    """Guard the guard: an empty file list would make every case below vacuous."""
    files = shipped_python_files()
    assert len(files) >= 8, f"ship manifest yielded only {len(files)} python files"


@pytest.mark.parametrize("path", shipped_python_files(), ids=lambda p: p.name)
def test_shipped_file_parses_at_box_floor(path: Path):
    bad = pep701_violations(path.read_text())
    assert not bad, (
        f"{path.relative_to(_REPO)} uses syntax newer than python "
        f"{'.'.join(map(str, BOX_PYTHON_FLOOR))}, which is the floor a box "
        f"image may ship:\n"
        + "\n".join(f"  line {ln}: {what}" for ln, what in bad))


def test_detector_catches_the_shape_that_killed_box_48089639():
    """The null-vector check: a detector that finds nothing must still fire on
    the real construct, or a green run means nothing."""
    offending = (
        'x = "observed"\n'
        's = (f"lifetime ("\n'
        '     f"{\'measured from its own dead replacements\'\n'
        '        if x == \'observed\' else\n'
        "        'ASSUMED'}\"\n"
        '     f")")\n')
    bad = pep701_violations(offending)
    assert bad, "detector missed the exact construct that parked the box"

    clean = ('note = ("measured" if x == "observed" else "ASSUMED")\n'
             's = f"lifetime ({note})"\n')
    assert not pep701_violations(clean), "detector fires on 3.11-legal code"
