"""`herdd metrics <id>` — is the job SATURATING the GPUs, or capped by something else?

The command exists to answer one question that `nvidia-smi` alone cannot: a box
at 100% "GPU util" may be issuing one tiny kernel per millisecond. Utilization is
blind; power draw, memory bandwidth, CPU busy, net rx/tx and disk I/O together
are not, and the probe renders all of them plus a one-line bottleneck verdict.

The mechanism worth understanding before touching this
------------------------------------------------------
`metrics_probe.py` is not INSTALLED on the box. Its source is piped over ssh
STDIN into `python3 -` and run there:

* it therefore works on a box that has never synced the tooling — which is the
  common case for a box that is misbehaving during boot;
* the BOX renders the table, the laptop only relays stdout. That is why
  `--json` is a relay flag rather than a formatter here, and why `--watch` is
  the probe's own loop (`--count 0 --interval N`) rather than a loop in this
  process;
* `python3 -` compiles stdin after EOF and only then runs, so closing
  `proc.stdin` is what STARTS the program. Dropping the `close()` hangs.

`--raw` is the escape hatch: `os.execvp` into a live `nvidia-smi dmon` on the
box, no probe involved, so a probe that itself breaks never blocks the operator.

What is deliberately NOT here
-----------------------------
* The metrics themselves. Sampling, the delta window, the throttle-reason
  decoding and the verdict all live in `metrics_probe.py` — a Zone S flat leaf,
  stdlib-only precisely so it runs on an unprovisioned box.
* Any history or storage. This is a live read; the durable per-run record is
  the `runs/` event log (`herdd runs`).

Provenance: moved from `tools/vast/herdd.py` (`cmd_metrics` and its
`_metrics_probe_path` helper, parser block in `main()`), plan §8 step 6,
2026-08-16, behavior-preserving. The one mechanical change: the probe is
resolved from `_TOOLS_VAST_DIR` below rather than from this module's own
`__file__`, which is three directories deeper.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from vastlib.boxes import lifecycle
from vastlib.boxes import ssh as ssh_mod
from vastlib.cli import _args, _docs

# `tools/vast/` — three dirnames up from `tools/vast/vastlib/cli/`. The flat
# `_metrics_probe_path` spelled this `os.path.dirname(os.path.abspath(__file__))`
# inside the function; the depth lives in one named constant here for the same
# reason `cli/_runsets.py::_HERE` does — a wrong depth is a silent
# "probe not found" against a box the operator is already trying to diagnose.
_TOOLS_VAST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# moved-from: herdd._metrics_probe_path
def _metrics_probe_path() -> str:
    return os.path.join(_TOOLS_VAST_DIR, "metrics_probe.py")


# moved-from: herdd.cmd_metrics
def run(a: argparse.Namespace) -> None:
    """Live host-metrics read of a running box: per-GPU util / mem / power / temp
    / throttle-reason + CPU busy & load / net rx-tx / disk I/O, plus a one-line
    bottleneck verdict — to see whether a job is SATURATING the GPUs or capped by
    network / CPU / disk / a thermal-or-power throttle.

    metrics_probe.py is piped over ssh STDIN and run on the box (`python3 - ...`),
    so it works even on a box that hasn't synced the tooling — the probe is
    stdlib-only and the BOX renders the table, the laptop just relays stdout.
    --watch streams (probe's own loop); --json relays raw JSON/NDJSON for agents;
    --raw drops to a live `nvidia-smi dmon`."""
    i = lifecycle._get_instance(a.id)
    host, port, _ = ssh_mod._pick_ssh_endpoint(i)
    if not (host and port):
        sys.exit(f"error: no ssh endpoint for {a.id} (status={i.get('actual_status')})")
    ssh_mod._warn_ssh_access(i)
    ssh = ["ssh", "-p", str(port), f"root@{host}",
           "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR"]

    if a.raw:                                   # native dmon stream, no probe
        os.execvp("ssh", ssh + ["nvidia-smi dmon"])

    probe = _metrics_probe_path()
    if not os.path.isfile(probe):
        sys.exit(f"error: probe not found: {probe}")
    with open(probe) as fh:
        src = fh.read()

    remote = ["python3", "-", "snapshot", "--window", str(a.window)]
    if a.json:
        remote.append("--json")
    if a.watch is not None:                     # stream forever until Ctrl-C
        secs = a.watch if a.watch and a.watch > 0 else 3
        remote += ["--count", "0", "--interval", str(secs)]
    remote_cmd = " ".join(shlex.quote(x) for x in remote)

    # stdin carries the probe SOURCE (python3 - compiles it after EOF, then runs
    # with no further stdin); stdout/stderr inherit the terminal so table blocks
    # / NDJSON stream live.
    proc = subprocess.Popen(ssh + [remote_cmd], stdin=subprocess.PIPE, text=True)
    try:
        proc.stdin.write(src)     # type: ignore[union-attr]  # stdin=PIPE always yields a pipe
        proc.stdin.close()        # type: ignore[union-attr]  # …and closing it is what STARTS python3 -
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    sys.exit(proc.returncode or 0)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pm = add_cmd(sub, "metrics",
                 "live GPU/CPU/net/disk utilization + bottleneck verdict for a box",
                 _docs.DOC_README, _docs.DOC_EVALS)
    pm.add_argument("id", type=int)
    pm.add_argument("--watch", nargs="?", type=int, const=3, default=None,
                    metavar="SECS",
                    help="stream every SECS seconds (default 3) instead of one shot")
    pm.add_argument("--window", type=float, default=1.0,
                    help="delta sample window seconds for cpu/net/disk rates (default 1.0)")
    pm.add_argument("--json", action="store_true",
                    help="relay raw JSON (one object per snapshot; NDJSON under --watch)")
    pm.add_argument("--raw", action="store_true",
                    help="skip the probe; stream a live `nvidia-smi dmon` instead")
    pm.set_defaults(func=run)
    return pm
