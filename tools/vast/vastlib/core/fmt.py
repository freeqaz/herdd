"""vastlib.core.fmt — the pure render atoms the fleet views are built from.

Why this exists
---------------
`herdd.py`'s formatter cluster measured 46 inbound calls, and almost all of
that fan-in lands on a handful of one-liners: `dollars` alone is called 52
times, `_age_str` 29. They are referentially transparent over their arguments,
know nothing about the vast.ai data model, and were nonetheless trapped inside
a 20k-line module that could not be imported piecemeal. Pulling them out first
makes every ring above independently testable, and gives the ls/watch/dash
renderers one place to agree on what a dollar, an age, and a token rate look
like. `core.fmt` is a leaf: it imports stdlib only — not even `core.config`
(see below).

What is deliberately NOT here
-----------------------------
* **The ls/watch table renderers.** `_render_ls` (~480 lines, ~20 closures),
  `_render_minimal`, `fleet_daemon_banner` and `cmd_ls` stay in `herdd.py`
  until plan step 6 moves them to `cli/`. They know the instance dict, the
  health lattice, the job-view fold and the terminal repaint loop — the
  opposite of an atom. `_render_minimal`'s column contract is additionally
  frozen for the dashboard adapter.
* **The job-progress parsers** — `_tqdm_points`, `_step_rate`, `_job_progress`,
  `_job_cell`. Pure, but they parse a *job view's* log tail, which is
  jobs-domain semantics rather than rendering; they own `test_jobprogress_rate.py`
  and they consume two atoms from here (`_hms_secs`, `_fmt_toks`) rather than
  belonging beside them. They land in `jobs/` with their regexes.
* **Timestamp parsers.** `_iso_ftz_to_epoch`, `_ts_age_s`, `_hb_age_s`,
  `_ts_to_epoch` produce numbers from strings and belong to journal/jobs/
  supervise. `_fmt_run_ts` is here because it is epoch-free positional slicing
  that produces a *display* string. `_round_age` likewise stays behind: it
  feeds `classify_box_health`'s evidence dict, so it is evidence normalization,
  not rendering.
* **`fmt_offer`.** Pure and single-dep on `dollars`, but it renders the Offer
  shape (plan §5 puts offer code in `market/offers.py`) and five tests
  monkeypatch `herdd.fmt_offer` to stub launch output. Moving it here would
  silently un-stub them. It moves with `market/offers.py`.
* **A `core.config` dependency for `_color_on`.** `NO_COLOR` and `TERM` are
  terminal presentation, not vast configuration; reading them through bare
  `os.environ.get` (exactly as the original did) keeps this module a zero-dep
  leaf and the core DAG one node flatter. If the config router ever grows a
  terminal section, this is the site to revisit.

Provenance
----------
Verbatim-with-types move of 14 atoms from `tools/vast/herdd.py`, plan §8
step 2 of `docs/plans/vast-tooling-refactor-v2.md`. Each carries its
`# moved-from:` marker. Step 2 is ADD-ONLY: `herdd.py` keeps its own copies
until step 6, and `tools/vast/test_vastlib_core_fmt.py` pins the two
implementations against each other so a rebase that edits one and not the
other fails loudly.
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import sys
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from vastlib.core import models


# moved-from: herdd.dollars
def dollars(x: object) -> str:
    try:
        # `object` is the honest parameter type: the function is deliberately
        # total over arbitrary input — the except clause below IS the contract,
        # and every one of the 52 call sites relies on "$?" for junk.
        return f"${float(x):.3f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "$?"


# moved-from: herdd._image_short
def _image_short(image: str | None) -> str:
    """Registry-host-stripped image ref for the ls column ('trainer:train-latest');
    digest-pinned refs compress to name@12hex ('pytorch@b85566342b86')."""
    s = (image or "").rsplit("/", 1)[-1]
    name, sep, dg = s.partition("@sha256:")
    return f"{name}@{dg[:12]}" if sep else s


# moved-from: herdd._Progress
class _Progress:
    """Shared done/total counter for the ls gather fan-out, so the interactive
    spinner can show `refreshing 8/15`. Each parallel unit (job fold, machine
    market probe, image digest) registers itself and ticks on completion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.done = 0
        self.total = 0

    def add(self, n: int) -> None:
        with self._lock:
            self.total += n

    def tick(self, n: int = 1) -> None:
        with self._lock:
            self.done += n

    def read(self) -> tuple[int, int]:
        with self._lock:
            return self.done, self.total


# moved-from: herdd._money
def _money(v: object, mark: bool = False) -> str:
    """$/hr cell for the ls table; '-' for a missing value. `mark` appends '*'
    to flag the mode a box is actually billed at (on-demand vs spot/bid)."""
    if v is None:
        return "-"
    return f"{dollars(v)}{'*' if mark else ''}"


# moved-from: herdd._Pal
class _Pal:
    """ANSI palette for the ls view; every accessor degrades to identity when
    color is off (non-TTY, NO_COLOR, TERM=dumb)."""

    def __init__(self, on: object) -> None:
        self.on = bool(on)

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s: str) -> str:
        return self._w("1", s)

    def dim(self, s: str) -> str:
        return self._w("2", s)

    def red(self, s: str) -> str:
        return self._w("31", s)

    def green(self, s: str) -> str:
        return self._w("32", s)

    def yellow(self, s: str) -> str:
        return self._w("33", s)

    def blue(self, s: str) -> str:
        return self._w("34", s)

    def magenta(self, s: str) -> str:
        return self._w("35", s)

    def cyan(self, s: str) -> str:
        return self._w("36", s)

    def bgreen(self, s: str) -> str:
        return self._w("1;32", s)

    def bcyan(self, s: str) -> str:
        return self._w("1;36", s)

    def byellow(self, s: str) -> str:
        return self._w("1;33", s)


# moved-from: herdd._color_on
def _color_on() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


# moved-from: herdd._age_str
def _age_str(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    if sec < 172800:
        return f"{sec // 3600}h"
    return f"{sec // 86400}d"


#: The fleet client's own age formatter — a SECOND spelling of `_age_str`,
#: kept distinct because the two are not interchangeable: this one takes None
#: (an absent age renders "?", never "0s"), tops out at `h`+minutes rather than
#: rolling over to days, and takes the seconds as an int-able rather than
#: clamping negatives. `fleet status` prints a retention window and `fleet
#: spend --reconcile` prints a pre-watch head with it; swapping in `_age_str`
#: would silently turn "2h05" into "2h" and "?" into "0s".
# moved-from: herdd._fmt_age
def _fmt_age(secs: float | None) -> str:
    if secs is None:
        return "?"
    secs = int(secs)
    if secs < 90:
        return f"{secs}s"
    if secs < 5400:
        return f"{secs // 60}m"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}"


#: Consumers beyond `_phys_lines`/`_visw`: the dash-cache scrubber
#: (`_dash_scrub`) imports this rather than re-compiling its own.
# moved-from: herdd._ANSI_RE
_ANSI_RE = re.compile(r"\033\[[0-9;]*[A-Za-z]")


# moved-from: herdd._hms_secs
def _hms_secs(t: str) -> int:
    """'1:55:27' / '23:29' → seconds."""
    s = 0
    for p in t.split(":"):
        s = s * 60 + int(p)
    return s


# moved-from: herdd._fmt_toks
def _fmt_toks(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M tok/s"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k tok/s"
    return f"{v:.0f} tok/s"


# moved-from: herdd._phys_lines
def _phys_lines(lines: Iterable[str]) -> int:
    """Physical terminal rows a list of logical lines occupies, wrap-aware, so
    the repaint knows how far to move the cursor up."""
    cols = shutil.get_terminal_size().columns or 80
    n = 0
    for ln in lines:
        vis = len(_ANSI_RE.sub("", ln))
        n += max(1, -(-vis // cols)) if vis else 1
    return n


# moved-from: herdd._visw
def _visw(s: str) -> int:
    """Visible width of a possibly-ANSI-colored line."""
    return len(_ANSI_RE.sub("", s))


# moved-from: herdd._ls_cols
def _ls_cols() -> int | None:
    """Render width for the ls view; None (no reflow, full rows) when stdout
    is not a terminal, so piped/grep output keeps every column."""
    if not sys.stdout.isatty():
        return None
    return shutil.get_terminal_size().columns


# moved-from: herdd._fmt_run_ts
def _fmt_run_ts(ts: str | None) -> str:
    """runmeta fixed-width YYYYMMDDTHHMMSSmmmZ -> compact 'MM-DD HH:MM' (UTC), or
    '-' when absent. Pure slicing (the format is positional)."""
    if not ts or len(ts) < 13:
        return "-"
    return f"{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"


# moved-from: herdd.fmt_offer
def fmt_offer(o: Mapping[str, Any]) -> str:
    """One offer as a fixed-width search/launch row. Reads with .get() and
    defaults everywhere except id/num_gpus/gpu_name, exactly like the
    original — a payload missing those raises KeyError, and that is ported
    behavior, not a defect. Adopted here in wave 2 cleanup: the mapping left
    it out of every manifest (models.py's boundary notes assign render atoms
    to fmt), and its only dependency is dollars() above (plus, since the
    2026-08-16 re-port of flat 8f8b9bc7, models.effective_cores — the eff_cores
    column, so `search` shows the offer's SLICE of the host, not the host)."""
    eff = models.effective_cores(o)
    _disk = o.get("disk_space")
    _disk = f"{_disk:>5.0f}GB" if isinstance(_disk, (int, float)) else "    ?GB"
    return (f"id={o['id']:>10}  {o['num_gpus']}x {o['gpu_name']:<11} "
            f"{o.get('gpu_ram',0)/1024:>3.0f}GB  "
            f"eff_cores={('?' if eff is None else format(eff, '.0f')):>4}  "
            f"dph={dollars(o.get('dph_total')):>8}  "
            # BOTH disk facts. The column used to print only the price, which
            # reads as if capacity were not a constraint — and it is the binding
            # one: a machine advertising less `disk_space` than the launch's
            # `--disk` hands back a smaller container instead of refusing.
            f"bid={dollars(o.get('min_bid')):>8}  "
            f"disk={_disk} {dollars(o.get('storage_cost')):>7}/mo  "
            f"rel={o.get('reliability',0):.3f}  down={o.get('inet_down',0):>5.0f}Mbps  "
            f"m={o.get('machine_id','?')}  host={o.get('host_id','?')}  "
            f"{o.get('geolocation','?')}")


# moved-from: herdd._ts_to_epoch
def _ts_to_epoch(ts: object) -> float | None:
    """PURE. Convert a runmeta/jobmeta ts ('YYYYMMDDTHHMMSSmmmZ') to a UTC epoch
    float, or None if unparseable. Companion to _ts_age_s but `now`-injectable (the
    watchdog subtracts against a supplied `now` so it stays deterministic in tests).

    Landed in core (not jobs/risk or boxes/health, both of which call it):
    risk.json and health.json each need it and boxes sits below jobs in the
    import DAG, so any non-core home inverts an edge (integrator ruling,
    2026-08-16). Sits here with the other timestamp atoms (_hms_secs,
    _fmt_run_ts); ports of its callers use `fmt._ts_to_epoch(...)`."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts.rstrip("Z")[:15], "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        return None


# The `_ts_to_epoch` companion named in its docstring, landed here for the same
# reason: `cli/runs.py` and `boxstate.py` both read it, and `boxstate.py` reaches
# it as `herdd._ts_age_s` — an EXTERNAL caller of the flat module, so the thin
# launcher must re-export this name (recorded in the step-6d re-export list;
# `herdd-reexports.json` shape "plain import"). `fmt.json` argued timestamp
# PARSERS belong with journal/jobs/supervise rather than in `fmt`; the tie-break
# is that its two consumers sit in different rings (`cli` and an unabsorbed
# sibling) and `_ts_to_epoch` — the same parse, `now`-injectable — is already
# here. The duplicated body against `jobs.view._hb_age_s` is preserved, not
# merged: that dedup is a behavior-visible change (the two differ in what they
# accept) and belongs to a later step.
# DUPLICATE, RULED 2026-08-16 (wave 6a): `vastlib/cli/runs.py` landed its own
# copy in the same wave. THIS is the home, on the external caller:
# `boxstate.py` reaches the name as `herdd._ts_age_s`, so the thin launcher
# must re-export it — and a launcher re-exporting from `vastlib.cli.runs` would
# make Zone E depend on a command module at the TOP of the DAG. `cli/runs.py`
# deleted its copy (and its marker) and calls `fmt._ts_age_s`.
# moved-from: herdd._ts_age_s
def _ts_age_s(ts: object) -> float | None:
    """Seconds since a runmeta ts ('YYYYMMDDTHHMMSSmmmZ'). None if unparseable."""
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.datetime.strptime(ts.rstrip("Z")[:15], "%Y%m%dT%H%M%S")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds()
    except Exception:
        return None
