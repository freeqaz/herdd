"""`herdd tunnel <id>` — SSH local-forward a container port to 127.0.0.1.

The eval lane's private wire: a vLLM server on the box stays bound to the
container and is reached at `127.0.0.1:<local>` here, so nothing is exposed
publicly and `LLM_BASE_URL` points at the forward. Foreground by default —
Ctrl-C closes it — with `--background` detaching via `boxes.ssh._tunnel_background`
and writing a pidfile/log for automated evals.

The `--pidfile` / `--logfile` help quotes the default paths
(`out/vast_tunnel_<id>_<local>.pid` / `.log`) as literal text, exactly as the
flat parser did; the paths themselves are computed in `boxes.ssh`, and the two
spellings are one of the things the CLI-surface byte diff is watching
(cli-surface.json hazard H4).

What is deliberately NOT here
-----------------------------
* Tunnel lifecycle management (list / kill / reap stale pidfiles). There is no
  such verb today and inventing one during a behavior-preserving port would be
  the definition of scope creep.
* `-o ExitOnForwardFailure=yes` is not decoration: without it a forward that
  cannot bind reports success and the eval harness hangs on a dead socket.

Provenance: moved from `tools/vast/herdd.py` (`cmd_tunnel`, parser block in
`main()`), plan §8 step 6, 2026-08-16, behavior-preserving.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

from vastlib.boxes import lifecycle
from vastlib.boxes import ssh as ssh_mod
from vastlib.cli import _args, _docs


# moved-from: herdd.cmd_tunnel
def run(a: argparse.Namespace) -> None:
    """SSH local-forward: reach a remote container port (e.g. vLLM :8000) at
    127.0.0.1:<local> on this box. No public exposure needed — the model server
    stays private and the local eval harness points LLM_BASE_URL at the tunnel.
    Foreground by default (Ctrl-C closes); --background detaches + pidfiles it."""
    i = lifecycle._get_instance(a.id)
    host, port, _ = ssh_mod._pick_ssh_endpoint(i)
    if not (host and port):
        sys.exit(f"error: no ssh endpoint yet (status={i.get('actual_status')})")
    ssh_mod._warn_ssh_access(i)
    cmd = ["ssh", "-N", "-L", f"{a.local}:localhost:{a.remote}",
           "-p", str(port), f"root@{host}",
           "-o", "StrictHostKeyChecking=accept-new",
           "-o", "LogLevel=ERROR",
           "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes"]
    if a.print:
        print(" ".join(shlex.quote(c) for c in cmd)); return  # noqa: E702 — verbatim body (plan §7.4)
    if getattr(a, "background", False):
        ssh_mod._tunnel_background(a, cmd, host, port); return  # noqa: E702 — verbatim body (plan §7.4)
    print(f"tunneling 127.0.0.1:{a.local} -> {host}:{port} container:{a.remote}  (Ctrl-C to close)")
    os.execvp("ssh", cmd)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pt = add_cmd(sub, "tunnel", "SSH local-forward a remote container port (e.g. vLLM :8000)",
                 _docs.DOC_EVALS, _docs.DOC_README)
    pt.add_argument("id", type=int)
    pt.add_argument("--local", type=int, default=18087, help="local port (default 18087)")
    pt.add_argument("--remote", type=int, default=8000, help="remote container port (default 8000)")
    pt.add_argument("--print", action="store_true", help="print the ssh command, don't run it")
    pt.add_argument("--background", action="store_true",
                    help="spawn the forward detached (setsid) + write a pidfile/log and print "
                         "pid+port instead of blocking (for inline/automated evals)")
    pt.add_argument("--pidfile", help="pidfile path for --background "
                                      "(default out/vast_tunnel_<id>_<local>.pid)")
    pt.add_argument("--logfile", help="log path for --background "
                                      "(default out/vast_tunnel_<id>_<local>.log)")
    pt.set_defaults(func=run)
    return pt
