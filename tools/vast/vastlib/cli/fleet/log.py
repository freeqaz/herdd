"""`herdd fleet log` — journal tail that works with the daemon DOWN.

Why this module exists
----------------------
It reads the append-only journal file directly rather than asking the socket,
because post-mortem is exactly when you need it and exactly when the daemon is
not answering. `-f` polls the same open file handle (seek back on a short read)
instead of shelling out to `tail`, and `KeyboardInterrupt` is a normal exit —
Ctrl-C on a log tail is not an error.

What is deliberately NOT here
-----------------------------
* Any journal schema. A line that does not parse as JSON is printed VERBATIM,
  which is what makes this usable against a partially-written or
  future-versioned journal.
* The path arithmetic. `client.fleet_journal_path()` owns it, and it is the
  same function the daemon writes through.

Provenance: moved from `tools/vast/herdd.py::cmd_fleet_log`, plan §8 step 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from vastlib.fleet import client


# moved-from: herdd.cmd_fleet_log -> run
def run(a: argparse.Namespace) -> None:
    """Journal tail — reads the append-only file directly, so it works even when
    the daemon is down (post-mortem is exactly when you need it)."""
    path = client.fleet_journal_path()
    if not os.path.exists(path):
        sys.exit(f"error: no journal at {path} (has fleetd ever run?)")

    def _emit(line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            rec = json.loads(line)
        except ValueError:
            print(line)
            return
        if a.iid and str(rec.get("iid") or "") != str(a.iid):
            return
        print(f"{rec.get('ts_iso', '')} {str(rec.get('iid') or '-'):<12}"
              f"{rec.get('event', '?'):<20}{json.dumps({k: v for k, v in rec.items() if k not in ('ts', 'ts_iso', 'iid', 'event')}, sort_keys=True)}")  # noqa: E501 — one journal row, kept as one expression: this is the printed line format and re-wrapping it is exactly the kind of edit a verbatim port must not make

    with open(path) as f:
        tail = f.readlines()[-int(a.n):]
        for ln in tail:
            _emit(ln)
        if not a.follow:
            return
        try:
            while True:
                where = f.tell()
                ln = f.readline()
                if not ln:
                    time.sleep(1.0)
                    f.seek(where)
                else:
                    _emit(ln)
        except KeyboardInterrupt:
            return


def add_parser(fsub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = fsub.add_parser("log", help="journal tail (works with the daemon down)")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", dest="n", type=int, default=50)
    p.add_argument("--iid", default=None)
    p.set_defaults(fleetfunc=run)
