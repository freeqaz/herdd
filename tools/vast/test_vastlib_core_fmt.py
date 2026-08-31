"""Behavior pins for `vastlib.core.fmt`, plus a parity tripwire against `herdd`.

Why this file exists
--------------------
The 14 atoms in `vastlib/core/fmt.py` are a verbatim-with-types move out of
`herdd.py` (plan §8 step 2, ADD-ONLY: `herdd.py` keeps its own copies until
step 6). The manifest's seam scan found ZERO monkeypatch sites and effectively
zero direct coverage for any of them — `dollars` alone has 52 inbound call
sites and not one test. So this file does two jobs:

1. **Characterization.** Each atom gets its behavior pinned, including the
   edges read off the body rather than off the docstring: `dollars` on junk and
   on `None`, `_money` on the 0-vs-None distinction, `_age_str` on negatives
   and on every band boundary, `_phys_lines` on a zero-visible-width line,
   `_color_on` on an EMPTY `NO_COLOR`.

2. **A drift tripwire.** For the whole porting window there are two copies of
   every atom. The parity tests import both and assert identical output on
   identical input, so a rebase that lands a peer's edit in `herdd.py` — or a
   "cleanup" of either copy — fails here instead of silently forking the fleet
   views. When step 6 deletes `herdd.py`'s copies, the parity half of this
   file goes with them; the characterization half stays.

Toolchain-free lane (`pytest -m "not integration"`): no network, no vast API,
no B2. `sys.stdout`, `shutil.get_terminal_size` and `os.environ` are the only
seams any of these atoms touch, and each is patched on the STDLIB module so
both copies see the same patch (patching `fmt.sys` would not steer
`herdd.sys`, and the parity assertion would be vacuous).
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import herdd as v  # noqa: E402
from vastlib.core import fmt  # noqa: E402


# --- helpers ---------------------------------------------------------------- #
class _FakeStdout:
    """Minimal stdout stand-in: only `isatty` is consulted by the atoms."""

    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, s: str) -> int:  # pragma: no cover - capture safety net
        return len(s)

    def flush(self) -> None:  # pragma: no cover - capture safety net
        return None


def _set_cols(monkeypatch: pytest.MonkeyPatch, cols: int) -> None:
    """Pin the terminal width, on `shutil` itself so both copies see it."""
    monkeypatch.setattr(shutil, "get_terminal_size", lambda *a: os.terminal_size((cols, 24)))


def _parity(name: str, *args: Any, **kw: Any) -> Any:
    """Call `fmt.<name>` and, IF a second copy still exists, `herdd.<name>`
    on the same input; assert equal.

    Returns the value so a caller can additionally assert what it should BE —
    parity alone would be satisfied by two identically-wrong copies, which is
    why every caller here also pins the expected value.

    THE SECOND COPY IS GONE (plan §8 step 6d): `herdd.py` is a thin launcher,
    so `herdd.<name>` is either absent or the SAME OBJECT as `fmt.<name>` —
    a comparison of a function with itself. This helper therefore compares only
    when it finds a genuinely distinct callable, which is exactly the condition
    the drift tripwire was written for. It is not a silent skip: the
    characterization half of every caller still runs, and `_flat_twin` going
    non-None again (a peer re-adding a body to the launcher) re-arms the
    comparison automatically. See this module's docstring — "when step 6
    deletes `herdd.py`'s copies, the parity half of this file goes with them;
    the characterization half stays".
    """
    fn = getattr(fmt, name)
    got = fn(*args, **kw)
    old_fn = _flat_twin(name)
    if old_fn is not None:
        old = old_fn(*args, **kw)
        assert got == old, f"{name}{args!r} drifted: vastlib={got!r} herdd={old!r}"
    return got


def _flat_twin(name: str) -> Any:
    """`herdd.<name>` when it is a SECOND implementation, else None."""
    old = getattr(v, name, None)
    return None if old is None or old is getattr(fmt, name) else old


# --- dollars ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("x", "want"),
    [
        (1.5, "$1.500"),
        (0, "$0.000"),
        (-2.25, "$-2.250"),
        (1.23456, "$1.235"),
        ("0.4", "$0.400"),
        (None, "$?"),
        ("abc", "$?"),
        ({}, "$?"),
    ],
)
def test_dollars(x: Any, want: str) -> None:
    """3dp, and TOTAL over junk: None/unparsable/wrong-type all fall to '$?'.

    That fallback is the contract, not an accident — the 52 call sites feed it
    raw vast.ai dict values that are routinely absent.
    """
    assert fmt.dollars(x) == want
    assert _parity("dollars", x) == want


# --- _image_short ----------------------------------------------------------- #
@pytest.mark.parametrize(
    ("image", "want"),
    [
        ("trainer:train-latest", "trainer:train-latest"),
        ("registry.gitlab.com/example/project/trainer:train-latest", "trainer:train-latest"),
        ("pytorch/pytorch@sha256:b85566342b86ffe0bc1c1e0e6c1d8b4a9f", "pytorch@b85566342b86"),
        (None, ""),
        ("", ""),
    ],
)
def test_image_short(image: str | None, want: str) -> None:
    """Host-stripped ref; a digest pin compresses to name@12hex. None -> ''."""
    assert fmt._image_short(image) == want
    assert _parity("_image_short", image) == want


# --- _Progress -------------------------------------------------------------- #
def test_progress_counts_and_defaults() -> None:
    """Starts (0, 0); `tick` defaults to 1; `add` and `tick` are independent."""
    p = fmt._Progress()
    assert p.read() == (0, 0)
    p.add(3)
    p.tick()
    assert p.read() == (1, 3)
    p.add(2)
    p.tick(4)
    assert p.read() == (5, 5)


def test_progress_does_not_clamp() -> None:
    """No clamping anywhere: done may exceed total, and a negative add sticks.

    Pinned because the ls spinner renders `done/total` raw — any clamping added
    later is a visible behavior change, not a tidy-up.
    """
    p = fmt._Progress()
    p.tick(7)
    p.add(-2)
    assert p.read() == (7, -2)


def test_progress_is_thread_safe() -> None:
    """The lock is the point — the ls gather ticks it from parallel workers."""
    p = fmt._Progress()

    def worker() -> None:
        for _ in range(250):
            p.tick()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert p.read() == (1000, 0)


def test_progress_counts_adds_and_ticks() -> None:
    """Was `…_parity_with_herdd`: the same op sequence driven through
    `fmt._Progress` and `v._Progress`. Step 6d made those one class (see
    `_flat_twin`), so the second instance added nothing."""
    p = fmt._Progress()
    p.add(9)
    p.tick()
    p.tick(3)
    assert p.read() == (4, 9)


# --- _money ----------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("val", "mark", "want"),
    [
        (None, False, "-"),
        (None, True, "-"),
        (0, False, "$0.000"),
        (1.5, False, "$1.500"),
        (1.5, True, "$1.500*"),
        ("junk", False, "$?"),
    ],
)
def test_money(val: Any, mark: bool, want: str) -> None:
    """Only None is '-'. Zero is a PRICE and renders as one; `mark` appends '*'.

    The 0-vs-None split is the edge worth pinning: a $0.000 spot cell and an
    unknown cell must stay distinguishable in the ls table.
    """
    assert fmt._money(val, mark) == want
    assert _parity("_money", val, mark) == want


# --- _Pal ------------------------------------------------------------------- #
_PAL_ACCESSORS: list[tuple[str, str]] = [
    ("bold", "1"),
    ("dim", "2"),
    ("red", "31"),
    ("green", "32"),
    ("yellow", "33"),
    ("blue", "34"),
    ("magenta", "35"),
    ("cyan", "36"),
    ("bgreen", "1;32"),
    ("bcyan", "1;36"),
    ("byellow", "1;33"),
]


@pytest.mark.parametrize(("name", "code"), _PAL_ACCESSORS)
def test_pal_on_wraps_and_off_is_identity(name: str, code: str) -> None:
    """Color on -> exact SGR wrap; color off -> identity, for every accessor.

    The identity half is what `test_guard.py` leans on when it renders the ls
    view with `_Pal(False)` and asserts on plain substrings.
    """
    assert getattr(fmt._Pal(True), name)("x") == f"\033[{code}mx\033[0m"
    assert getattr(fmt._Pal(False), name)("x") == "x"


def test_pal_coerces_truthiness() -> None:
    """`on` is bool()-ed, so any truthy/falsy value works at the call sites."""
    assert fmt._Pal(1).on is True
    assert fmt._Pal("").on is False
    assert fmt._Pal(None).on is False


@pytest.mark.parametrize(("name", "code"), _PAL_ACCESSORS)
def test_pal_accessor_wraps_only_when_on(name: str, code: str) -> None:
    """Was `…_parity_with_herdd`, comparing each accessor across the two
    `_Pal` classes; one class since step 6d. `_PAL_ACCESSORS` carries the SGR
    code for each name, so the wrapping can be asserted directly — which is
    what the deleted comparison was a proxy for."""
    assert getattr(fmt._Pal(False), name)("sample") == "sample"
    on = getattr(fmt._Pal(True), name)("sample")
    assert on != "sample" and "sample" in on and code in on


# --- _color_on -------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("tty", "env", "want"),
    [
        (True, {"TERM": "xterm-256color"}, True),
        (True, {}, True),
        (False, {"TERM": "xterm-256color"}, False),
        (True, {"TERM": "dumb"}, False),
        (True, {"NO_COLOR": "1", "TERM": "xterm-256color"}, False),
        # NO_COLOR="" is FALSY, so it does NOT disable color — read off the
        # body (`if os.environ.get("NO_COLOR")`), and the opposite of what the
        # no-color.org convention says. Pinned as found behavior.
        (True, {"NO_COLOR": "", "TERM": "xterm-256color"}, True),
    ],
)
def test_color_on(monkeypatch: pytest.MonkeyPatch, tty: bool, env: dict[str, str],
                  want: bool) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    for k, val in env.items():
        monkeypatch.setenv(k, val)
    monkeypatch.setattr(sys, "stdout", _FakeStdout(tty))
    assert fmt._color_on() is want


# --- _age_str --------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("sec", "want"),
    [
        (0, "0s"),
        (-5, "0s"),
        (89, "89s"),
        (90, "1m"),
        (90.9, "1m"),
        (5399, "89m"),
        (5400, "1h"),
        (172799, "47h"),
        (172800, "2d"),
        (864000, "10d"),
    ],
)
def test_age_str(sec: float, want: str) -> None:
    """Every band boundary, plus the negative clamp and float truncation."""
    assert fmt._age_str(sec) == want
    assert _parity("_age_str", sec) == want


# --- _ANSI_RE / _visw / _phys_lines ----------------------------------------- #
def test_ansi_re_strips_every_escape_shape() -> None:
    """Was `…_and_is_identical_to_herdd` (`.pattern` equality); one pattern
    since step 6d, and the substitutions below are what it has to do."""
    assert fmt._ANSI_RE.sub("", "\033[31mred\033[0m") == "red"
    assert fmt._ANSI_RE.sub("", "\033[1;36mx\033[0m plain") == "x plain"
    assert fmt._ANSI_RE.sub("", "no escapes here") == "no escapes here"


@pytest.mark.parametrize(
    ("s", "want"),
    [
        ("", 0),
        ("plain", 5),
        ("\033[31mred\033[0m", 3),
        ("\033[0m", 0),
    ],
)
def test_visw(s: str, want: int) -> None:
    assert fmt._visw(s) == want
    assert _parity("_visw", s) == want


def test_phys_lines_wraps_on_visible_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap math ignores escapes: 25 visible chars at 10 cols is 3 rows."""
    _set_cols(monkeypatch, 10)
    assert fmt._phys_lines(["abc"]) == 1
    assert fmt._phys_lines(["a" * 10]) == 1
    assert fmt._phys_lines(["a" * 11]) == 2
    assert fmt._phys_lines(["a" * 25]) == 3
    assert fmt._phys_lines([f"\033[31m{'a' * 25}\033[0m"]) == 3
    assert fmt._phys_lines(["abc", "a" * 25]) == 4
    assert fmt._phys_lines([]) == 0


def test_phys_lines_zero_width_line_still_costs_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank (or escape-only) line occupies one physical row, not zero.

    The `if vis else 1` branch — the cursor still has to move up past a blank
    separator, so getting this wrong smears the ls repaint.
    """
    _set_cols(monkeypatch, 10)
    assert fmt._phys_lines([""]) == 1
    assert fmt._phys_lines(["\033[0m"]) == 1
    assert fmt._phys_lines(["", "", "abc"]) == 3


def test_phys_lines_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_cols(monkeypatch, 13)
    for lines in ([], [""], ["abc"], ["a" * 40], ["\033[31m" + "b" * 27, "", "z"]):
        _parity("_phys_lines", lines)


# --- _ls_cols --------------------------------------------------------------- #
def test_ls_cols_none_when_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """None means 'no reflow' — piped output keeps every column."""
    monkeypatch.setattr(sys, "stdout", _FakeStdout(False))
    _set_cols(monkeypatch, 100)
    assert fmt._ls_cols() is None


def test_ls_cols_width_when_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", _FakeStdout(True))
    _set_cols(monkeypatch, 132)
    assert fmt._ls_cols() == 132


# --- _hms_secs -------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("t", "want"),
    [
        ("1:55:27", 6927),
        ("23:29", 1409),
        ("45", 45),
        ("0:00", 0),
        ("0:00:00", 0),
    ],
)
def test_hms_secs(t: str, want: int) -> None:
    """Base-60 fold over however many colon-separated fields there are."""
    assert fmt._hms_secs(t) == want
    assert _parity("_hms_secs", t) == want


def test_hms_secs_raises_on_junk() -> None:
    """Not soft — the caller (`_tqdm_points`) only ever feeds it a matched group."""
    with pytest.raises(ValueError):
        fmt._hms_secs("1:xx")


# --- _fmt_toks -------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("val", "want"),
    [
        (0, "0 tok/s"),
        (12.4, "12 tok/s"),
        (999, "999 tok/s"),
        (1000, "1.0k tok/s"),
        (1500, "1.5k tok/s"),
        # 999_999 is BELOW the 1e6 cut, so it renders as 1000.0k, not 1.0M.
        (999_999, "1000.0k tok/s"),
        (1_000_000, "1.0M tok/s"),
        (2_350_000, "2.4M tok/s"),
    ],
)
def test_fmt_toks(val: float, want: str) -> None:
    assert fmt._fmt_toks(val) == want
    assert _parity("_fmt_toks", val) == want


# --- _fmt_run_ts ------------------------------------------------------------ #
@pytest.mark.parametrize(
    ("ts", "want"),
    [
        ("20260816T123456789Z", "08-16 12:34"),
        # Exactly the len-13 minimum the guard admits.
        ("20260816T1234", "08-16 12:34"),
        ("20260816T123", "-"),
        ("", "-"),
        (None, "-"),
    ],
)
def test_fmt_run_ts(ts: str | None, want: str) -> None:
    """Positional slicing of the runmeta stamp; anything shorter than 13 is '-'."""
    assert fmt._fmt_run_ts(ts) == want
    assert _parity("_fmt_run_ts", ts) == want


# --- the marker contract ---------------------------------------------------- #
def test_every_ported_atom_carries_a_moved_from_marker() -> None:
    """Plan §7.1 generates the rename table from these markers.

    A missing marker is a symbol the test migration cannot find, which is how a
    patch site silently stops steering anything. Cheap to assert here, and it
    fails at port time rather than at step 6.
    """
    src = (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vastlib", "core", "fmt.py"))
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    names = [
        "dollars", "_image_short", "_Progress", "_money", "_Pal", "_color_on",
        "_age_str", "_ANSI_RE", "_hms_secs", "_fmt_toks", "_phys_lines",
        "_visw", "_ls_cols", "_fmt_run_ts",
    ]
    missing = [n for n in names if f"# moved-from: herdd.{n}\n" not in text]
    assert not missing, f"ported atoms without a moved-from marker: {missing}"
    for n in names:
        assert hasattr(fmt, n), f"{n} is not exported by vastlib.core.fmt"
        # Step 6d ended the ADD-ONLY window: `herdd.py` keeps only the
        # subset of these its external consumers reach, and every one it keeps
        # is this module's object. What must never happen is a SECOND body
        # reappearing there under the same name.
        assert _flat_twin(n) is None, (
            f"{n} has a second body in herdd again — the launcher must "
            f"re-export vastlib.core.fmt's object, never redefine it")


def test_fmt_offer_renders_and_keeps_its_KeyError_contract() -> None:
    """fmt_offer was adopted into core.fmt after wave 2 (it was in no
    manifest). Was parity over a full offer, a sparse offer exercising every
    .get() default, and the KeyError contract on the three required keys; step
    6d left one function, so the two comparisons went and the KeyError contract
    — the half that was never parity — stayed."""
    full = {"id": 12345678, "num_gpus": 4, "gpu_name": "RTX_5090",
            "gpu_ram": 32768, "dph_total": 1.2345, "min_bid": 0.9,
            "storage_cost": 150.0, "reliability": 0.99876,
            "inet_down": 8123.4, "machine_id": 22001, "host_id": 9001,
            "geolocation": "Sweden, SE"}
    sparse = {"id": 1, "num_gpus": 1, "gpu_name": "H100"}
    assert "12345678" in fmt.fmt_offer(full) and "RTX_5090" in fmt.fmt_offer(full)
    assert "H100" in fmt.fmt_offer(sparse)
    for missing in ("id", "num_gpus", "gpu_name"):
        broken = dict(sparse)
        del broken[missing]
        with pytest.raises(KeyError):
            fmt.fmt_offer(broken)


def test_ts_to_epoch_parses_the_live_shapes_and_rejects_the_rest() -> None:
    """_ts_to_epoch adopted into core.fmt by integrator ruling (risk.json /
    health.json DAG conflict). Was parity over the live format, the mmm-suffix,
    junk and the non-str rejections; one parser since step 6d, so the
    parses/rejects split is stated instead."""
    for ts in ("20260816T073519123Z", "20260816T073519Z", "20260816T073519"):
        assert fmt._ts_to_epoch(ts) == 1786865719.0, ts
    for ts in ("", "not-a-ts", "2026-08-16T07:35:19Z", None, 12345, 0.5,
               b"20260816T073519123Z"):
        assert fmt._ts_to_epoch(ts) is None, ts
    # NOT a rejection, and deliberately kept in the table: a truncated stamp
    # still parses (the parser is prefix-tolerant), landing on a DIFFERENT
    # instant. Pinned as found.
    assert fmt._ts_to_epoch("20260816T0735") == 1786863785.0
