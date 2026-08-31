"""`herdd ssh <id>` — exec into a box, or print the command that would.

Resolves the instance's current ssh endpoint (it changes on every resume) and
`os.execvp`s into `ssh`. Two behaviours worth naming because they look like
decoration and are not:

* **`_ssh_auth_preflight` before the exec.** A bare `Permission denied
  (publickey)` is indistinguishable from a dozen different causes; the
  preflight says WHICH one — key never attached, key attached but not yet
  installed on the box, box not finished booting. It runs only on the exec
  path, never under `--print`, because `--print` promises to touch nothing.
* **`os.execvp`, not `subprocess`.** The process is REPLACED, so ssh owns the
  tty (interactive shells, ^C, scp-style pipes all behave) and the caller's
  exit code is ssh's own.

What is deliberately NOT here
-----------------------------
* Key management. Attaching and repairing the pubkey is `boxes.ssh` and the
  launch path (`--ssh` is on by default since 2026-07-31).
* Port forwarding — that is `herdd tunnel`, a separate verb because it has a
  background/pidfile lane this command has no business growing.

Provenance: moved from `tools/vast/herdd.py` (`cmd_ssh`, parser block in
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


# moved-from: herdd.cmd_ssh
def run(a: argparse.Namespace) -> None:
    i = lifecycle._get_instance(a.id)
    host, port, _ = ssh_mod._pick_ssh_endpoint(i)
    if not (host and port):
        sys.exit(f"error: no ssh endpoint yet (status={i.get('actual_status')})")
    cmd = ["ssh", "-p", str(port), f"root@{host}",
           "-o", "StrictHostKeyChecking=accept-new",
           "-o", "LogLevel=ERROR"]
    if a.exec:
        cmd.append(a.exec)
    if a.print:
        print(" ".join(shlex.quote(c) for c in cmd)); return  # noqa: E702 — verbatim body (plan §7.4)
    ssh_mod._ssh_auth_preflight(i, host, port)   # say WHY, don't leak a bare denial
    ssh_mod._debug_hold_reminder()
    os.execvp("ssh", cmd)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    pssh = add_cmd(sub, "ssh", "ssh into an instance (or --print the command)",
                   _docs.DOC_README, _docs.DOC_DEBUG)
    pssh.add_argument("id", type=int)
    pssh.add_argument("--exec", help="run a remote command instead of interactive shell")
    pssh.add_argument("--print", action="store_true", help="print the ssh command, don't run it")
    pssh.set_defaults(func=run)
    return pssh
