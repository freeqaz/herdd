"""`herdd ls` — the fleet view. Paint the cached snapshot first, then the truth.

The most-run command in the tool, and the only one with a redraw loop. Its shape
is deliberate and easy to break in a port:

1. **`--json` short-circuits everything** — raw `_instances()`, no snapshot read,
   no snapshot write, no banner. Machine callers get the API's own shape.
2. **`--minimal` is the agent lane**: a frozen TSV (`_ls_render._MINIMAL_COLS`),
   no color, no spinner, and it still refreshes the snapshot unless `--cached`.
   `--json-rows` is the same rows as JSON objects (one code path,
   `_minimal_rows`); positional IDs narrow every view, filtered at render so
   the snapshot cache stays full-fleet.
3. **Interactive** paints the previous snapshot IMMEDIATELY in full colour, runs
   the live gather on a worker thread, then erases the painted region and
   repaints. The spinner line under the stale table is the only "this is old"
   cue, which is why it carries the snapshot's age.
4. **Non-tty** skips all of that: gather, save, print once.

Two details that look incidental and are not
--------------------------------------------
* `_ls_cols()` is re-read at REPAINT time, not once at the top: the window may
  have been resized while the spinner ran, and a stale width prints a table that
  wraps.
* The worker catches `BaseException`, not `Exception`. `core.api.request`
  `sys.exit()`s on an API error — that is a `SystemExit`, which is NOT an
  `Exception`; catching only `Exception` would let it unwind through the thread,
  leave `result` empty, and print "unknown error" over a half-erased screen.
  `KeyboardInterrupt` during the join exits 130, the shell convention.

What is deliberately NOT here
-----------------------------
* The gather and both renderers. `_gather_ls_data`, `_market_map`,
  `_stale_image_ids`, `_render_ls` and `_render_minimal` are shared with
  `dash-cache` and live in `cli/_ls_render.py` (cli-surface.json hazard H3).
  This module owns only what one command reaches: the snapshot cache, the fleetd
  banner, and the paint loop.
* Any policy. `ls` never mutates a box, never bids, never parks. The single
  write it performs is the snapshot file under `XDG_CACHE_HOME`.

Provenance: moved from `tools/vast/herdd.py` (`cmd_ls`, `_LS_SNAPSHOT`,
`_ls_snapshot_save/_load`, `fleet_daemon_banner`, parser block in `main()`),
plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any, Sequence

from vastlib.boxes import lifecycle
from vastlib.cli import _args, _docs, _ls_render
from vastlib.core import fmt
from vastlib.fleet import client as fleet_client

# moved-from: herdd._LS_SNAPSHOT
_LS_SNAPSHOT = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "herdd", "ls-snapshot.json")


# moved-from: herdd._ls_snapshot_save
def _ls_snapshot_save(data: Any) -> None:  # noqa: ANN401 — the gather dict, passed through unread
    try:
        os.makedirs(os.path.dirname(_LS_SNAPSHOT), exist_ok=True)
        tmp = _LS_SNAPSHOT + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, _LS_SNAPSHOT)
    except Exception:
        pass


# moved-from: herdd._ls_snapshot_load
def _ls_snapshot_load() -> Any:  # noqa: ANN401 — whatever was serialized; callers duck-type it
    try:
        with open(_LS_SNAPSHOT) as fh:
            return json.load(fh)
    except Exception:
        return None


# moved-from: herdd.fleet_daemon_banner
def fleet_daemon_banner() -> str | None:
    """S5: a loud line when the fleet supervisor is DOWN or its tick is stale —
    printed by `ls` so 'who is babysitting?' is answerable at a glance. Silent on
    machines that never installed fleetd (no state dir, no socket)."""
    try:
        if not (os.path.exists(fleet_client.fleet_sock_path())
                or os.path.isdir(fleet_client.fleet_state_dir())):
            return None
        ok, data, err = fleet_client.fleet_request("ping", _timeout=3, _retries=0)
        pal = fmt._Pal(fmt._color_on())
        if not ok:
            return pal.red(f"!! fleetd DOWN ({err}) — NOTHING is babysitting this "
                           f"fleet (start: herdd fleet install)")
        age = data.get("tick_age_s")
        if age is not None and age > 3 * 45:
            return pal.red(f"!! fleetd tick STALE ({age}s) — supervision may be "
                           f"wedged (herdd fleet log -n 20)")
    except Exception:
        return None
    return None


def _narrow(data: Any, ids: Sequence[Any]) -> Any:  # noqa: ANN401 — gather dict, duck-typed
    """Apply the positional id filter; an id set matching NOTHING is an error
    (exit 2), never an empty table an agent might read as 'fleet is clean'."""
    if not ids:
        return data
    out = _ls_render._filter_boxes(data, ids)
    if not out["instances"]:
        print(f"no instance matches: {' '.join(str(x) for x in ids)}",
              file=sys.stderr)
        sys.exit(2)
    return out


# moved-from: herdd.cmd_ls
def run(a: argparse.Namespace) -> None:
    # tolerate a partial Namespace (older callers / tests pass only json=)
    ids = getattr(a, "ids", None) or []
    if a.json:
        ins = lifecycle._instances()
        if ids:
            want = {str(x) for x in ids}
            ins = [i for i in ins if str(i.get("id")) in want]
            if not ins:
                print(f"no instance matches: {' '.join(str(x) for x in ids)}",
                      file=sys.stderr)
                sys.exit(2)
        print(json.dumps(ins, indent=2))
        return
    json_rows = getattr(a, "json_rows", False)
    if not json_rows:                 # machine JSON must stay parseable
        _banner = fleet_daemon_banner()
        if _banner:
            print(_banner)
    cached = getattr(a, "cached", False)
    no_spot = getattr(a, "no_spot", False)
    if getattr(a, "minimal", False) or json_rows:
        data = (_ls_snapshot_load() if cached
                else _ls_render._gather_ls_data(no_spot=no_spot))
        if not data:
            print("no ls snapshot yet — run `herdd ls` once first",
                  file=sys.stderr)
            sys.exit(2)
        if not cached:
            _ls_snapshot_save(data)          # save FULL, filter at render
        data = _narrow(data, ids)
        if json_rows:
            print(json.dumps(_ls_render._minimal_rows(data), indent=2))
        else:
            print(_ls_render._render_minimal(data))
        return
    pal = fmt._Pal(fmt._color_on())
    snap = _ls_snapshot_load()

    if cached:
        if not snap:
            print("no ls snapshot yet — run `herdd ls` once first",
                  file=sys.stderr)
            sys.exit(2)
        age = fmt._age_str(time.time() - (snap.get("ts") or 0))
        banner = pal.yellow(f" ⟳ cached view from {age} ago (--cached, "
                            f"no live query)")
        print("\n".join(_ls_render._render_ls(_narrow(snap, ids), pal,
                                              banner=banner,
                                              cols=fmt._ls_cols())))
        return

    if not sys.stdout.isatty():
        data = _ls_render._gather_ls_data(no_spot=no_spot)
        _ls_snapshot_save(data)
        print("\n".join(_ls_render._render_ls(_narrow(data, ids), pal)))
        return

    # Interactive: paint the cached snapshot immediately in full color (the
    # spinner line below it is the only stale cue), gather live data in a
    # worker, then erase the region and repaint fresh.
    result: dict[str, Any] = {}
    errs: list[str] = []
    prog = fmt._Progress()

    def work() -> None:
        try:
            result["data"] = _ls_render._gather_ls_data(no_spot=no_spot, prog=prog)
        except BaseException as e:            # request() sys.exit()s on API error
            errs.append(f"{type(e).__name__}: {e}".strip())

    th = threading.Thread(target=work, daemon=True)
    th.start()

    painted = [0]

    def paint(lines: Sequence[str]) -> None:
        if painted[0]:
            sys.stdout.write(f"\033[{painted[0]}A\033[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        painted[0] = fmt._phys_lines(lines)

    if snap:
        age = fmt._age_str(time.time() - (snap.get("ts") or 0))
        spin_tail = f" · showing cached view from {age} ago"
        # stale paint filters WITHOUT the no-match exit: a just-launched box
        # is legitimately absent from the snapshot and present in the refresh.
        stale = _ls_render._filter_boxes(snap, ids) if ids else snap
        stale_lines = _ls_render._render_ls(stale, pal, cols=fmt._ls_cols())
    else:
        spin_tail = " · querying vast API + B2"
        stale_lines = []

    def spin_msg() -> str:
        # `refreshing 8/15` once the fan-out has registered its work units
        # (each job fold / market probe / image digest is one unit).
        d, t = prog.read()
        cnt = f" {d}/{t}" if t else ""
        return f"refreshing{cnt}{spin_tail}"

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    paint(stale_lines + [pal.dim(f" {frames[0]} {spin_msg()}")])
    k = 0
    try:
        while th.is_alive():
            th.join(0.09)
            k += 1
            tick = pal.dim(f" {frames[k % len(frames)]} {spin_msg()}")
            sys.stdout.write(f"\033[1A\r\033[K{tick}\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.exit(130)

    if "data" in result:
        data = result["data"]
        _ls_snapshot_save(data)          # save FULL, filter at render
        # re-read the width at repaint time — the window may have been
        # resized while the spinner ran
        paint(_ls_render._render_ls(_narrow(data, ids), pal,
                                    cols=fmt._ls_cols()))
    else:
        msg = errs[0] if errs else "unknown error"
        note = pal.red(f" ✗ refresh failed ({msg})")
        if snap:
            note += pal.yellow("  — showing the cached view above")
        paint(stale_lines + [note])
        sys.exit(1)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pls = add_cmd(sub, "ls", "list my instances",
                  _docs.DOC_OPERATIONS,
                  _docs.DOC_README,
                  "NOTE: a stopped/outbid instance still bills storage until destroyed — check after every session")  # noqa: E501 — verbatim parser block (plan §7.4)
    pls.epilog = (
        "legend:\n"
        "  ● live · ○ stopped; spot = interruptible, on-dem = reserved\n"
        "  N% after the GPU name = utilization; ACTIVE jobs show live training\n"
        "  progress (%done + step, s/it, avg tok/s, checkpoint count) parsed\n"
        "  from the job heartbeat; a cyan `cpu` chip = CPU-only job (the\n"
        "  bundle declared needs.gpu false — the GPU idling there is expected)\n"
        "  live rows: `up 7h3m ≈$4.71` = age since launch (start_date) + an\n"
        "  UPPER-BOUND accrued cost (dph x age; over-counts loading/parked\n"
        "  windows); `$3.21/$16` = fleetd watch spend vs budget cap, heating\n"
        "  dim -> yellow (60%) -> red (90%); `Σ$…` = watched with no cap;\n"
        "  blank = this box is unwatched\n"
        "  stopped rows: storage $/day + idle age + resume ✓ (gpus free) /\n"
        "  ✗ taken (held by another renter) + both resume rates; spot rate =\n"
        "  live market floor + disk (stale min_bid under --no-spot)\n\n"
        + (pls.epilog or ""))
    pls.add_argument("ids", nargs="*", metavar="ID",
                     help="narrow every view to these instance ids (exact "
                          "match); exit 2 when none match")
    pls.add_argument("--json", action="store_true")
    pls.add_argument("--json-rows", action="store_true",
                     help="the --minimal table as a JSON array of "
                          "{column: value} objects — same columns, same "
                          "strings, empty = N/A; combine with --cached for "
                          "zero network")
    pls.add_argument("--no-spot", action="store_true",
                     help="skip the live per-machine spot-floor query (faster; "
                          "spot column falls back to the box's stale min_bid)")
    pls.add_argument("--cached", action="store_true",
                     help="print the last snapshot instantly (no network); "
                          "banner shows its age")
    pls.add_argument("--minimal", action="store_true",
                     help="token-efficient TSV for agents: header + one row "
                          "per box (state,id,status,gpus,gpu,gpu_util,mode,"
                          "hourly,storage_day,disk_gb,disk_used_gb,idle,"
                          "avail,ondemand,spot,stale,label,jobs,phase,"
                          "cpu_util,uptime,spend_usd,budget_usd,cpu_jobs); "
                          "no color/spinner. Combine with --cached for zero "
                          "network.")
    pls.set_defaults(func=run)
    return pls
