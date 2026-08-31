"""vastlib.boxes.ssh — how we reach a rented box, and why we usually can't.

Why this exists
---------------
Everything about getting a shell (or a forwarded port) onto a vast box lives
here: composing the onstart snippet that makes `ssh root@<box>` work at all,
registering the key with vast, enumerating and probing the two endpoints a box
can answer on, saying WHY a connection was refused before ssh leaks a bare
`Permission denied (publickey)`, and backgrounding a local-forward tunnel.

The module's centre of gravity is one root cause, root-caused live on
2026-07-31 against box 46449950 and written out in `ssh_authorized_keys_snippet`
below: vast's host daemon can write `/root/.ssh/authorized_keys` as its OWN
user (`vastai_kaalia:docker 0644`), and sshd's `StrictModes yes` refuses a file
owned by neither root nor the target user. A byte-perfect key in a
wrongly-owned file is indistinguishable, from the client, from no key at all —
which is why the diagnosis functions (`ssh_access_warning`,
`SSH_STRICTMODES_HINT`, `_ssh_auth_preflight`) are as load-bearing as the
snippet: without them the operator goes looking at `~/.ssh`, the vast account
keys and the host, all of which look FINE.

The classification half of that story — `SSH_INJECT_MARKER`,
`instance_ssh_install`, `instance_has_ssh_inject` — deliberately does NOT live
here. It landed in `core.models` with the other payload accessors, because it
reads an instance dict and nothing else, and because `ls` classifies boxes on
it without wanting anything ssh-shaped. This module imports the marker from
there rather than keeping a second copy of the string;
`test_vastlib_core_models.py` pins that the launcher re-exports this module's
object rather than redefining it (it pinned both against `herdd`'s original
until step 6d deleted that original).

What is deliberately NOT here
-----------------------------
* **`cmd_ssh` / `cmd_tunnel`.** They are argparse entry points (argv parsing,
  `os.execvp`, `sys.exit` on a missing endpoint) and move to `cli/` at plan
  step 6. `_print_ssh` and `_tunnel_background` — the parts that do work rather
  than dispatch — are here, and both keep their exact stdout/stderr split:
  `_print_ssh` writes the ssh command line to STDOUT and every diagnosis to
  STDERR, which is the same machine-readability contract `--print` has.
* **`_get_instance`.** It is a lifecycle read (`request` + `d.get("instances", d)`,
  and it *raises* — `sys.exit` via the raising `request`), so it lives in
  `boxes.lifecycle`. It is called here as a module attribute
  (`lifecycle._get_instance`) so the suite's patch of that seam keeps steering.
* **`_ssh_kill_job`** (`job cancel --hard`) and **`_revoke_box_keys`**. The
  first has the same shape as a salvage push and belongs to `jobs.control`; the
  second is a B2 key revoke — "key" in the minted-credential sense, not the ssh
  sense — and is ruled to `boxes.lifecycle`.
* **No key GENERATION.** `pub_key_text` reads an existing public key from an
  explicit path or the two usual defaults and returns None when there is none.
  Nothing here ever creates, uploads or rotates a private key.
* **No retry around the endpoint probe.** `_pick_ssh_endpoint` tries each
  candidate exactly once with a short connect timeout and falls back to the
  first candidate UNPROBED; a probe failure never blocks a connection attempt.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 3, the
`boxes/` ring), 2026-08-16. Behavior-preserving: bodies copied, annotations
added. Two mechanical exceptions, both documented at their site: `_REPO_ROOT`
(`_tunnel_background`'s three-`dirname` walk is wrong from this depth — see the
constant) and the `lifecycle.`/`api.`/`models.` module-attribute call form
required by plan §8b. Every symbol carries its `# moved-from:` marker (grammar:
`vastlib/README.md` §2). The flat `herdd.py` copies stay live until step 6.
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time

from vastlib.boxes import lifecycle
from vastlib.core import api, models

# The repo root, five levels up from `tools/vast/vastlib/boxes/ssh.py`.
#
# THIS IS THE ONE LINE OF `_tunnel_background` THAT IS NOT TEXTUALLY VERBATIM,
# and it is what keeps it behaviorally verbatim. `herdd.py` computes the same
# path inline as `dirname(dirname(dirname(abspath(__file__))))` — exactly three,
# correct only from `tools/vast/herdd.py`. Copied unchanged into this file the
# expression yields `tools/vast`, so `out/` and the default pidfile/logfile
# would silently relocate to `tools/vast/out/` — nothing raises, the tunnel
# still comes up, and only the printed teardown path (and the argparse help text
# that quotes these defaults verbatim) would be wrong. Hoisted to a module
# constant for the same reason `core.config._HERE` is one: the depth is a
# property of the module's location, not of the function, and a package that
# moves again should have to fix exactly one line.
#
# `test_vastlib_boxes_ssh.py::test_repo_root_matches_herdd_computation` pins
# it against the expression applied to `herdd.py`'s own path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# moved-from: herdd.pub_key_text
def pub_key_text(path_or_none: str | None) -> str | None:
    """Read a public key from an explicit path or the usual defaults."""
    cands = [path_or_none] if path_or_none else [
        os.path.expanduser("~/.ssh/id_ed25519.pub"),
        os.path.expanduser("~/.ssh/id_rsa.pub"),
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return open(c).read().strip()
    return None


# bounded re-assert window for the ownership repair (see the snippet docstring)
# moved-from: herdd.SSH_FIX_TRIES
SSH_FIX_TRIES = 30
# moved-from: herdd.SSH_FIX_SLEEP_S
SSH_FIX_SLEEP_S = 10


# moved-from: herdd.ssh_authorized_keys_snippet
def ssh_authorized_keys_snippet(pub: str) -> str:
    """Onstart prelude that makes `ssh root@<box>` work — on EVERY box we rent.

    This is deliberately NOT a bare `echo >> authorized_keys`. The failure it
    exists to prevent was root-caused live on 2026-07-31 (box 46449950): the
    container's /root/.ssh/authorized_keys ALREADY held the right key, byte for
    byte, yet every login failed with a bare `Permission denied (publickey)` on
    both the direct mapping and the sshN.vast.ai proxy, and the API cheerfully
    answered "SSH key already associated with instance." The file was owned by
    the vast host daemon's user:

        -rw-r--r-- 1 vastai_kaalia docker 665 ... /root/.ssh/authorized_keys

    sshd runs `StrictModes yes` (verified in `sshd -T` on a sibling box), which
    refuses an authorized_keys owned by neither the target user nor root — so a
    correct key in a wrongly-owned file is indistinguishable, from the client,
    from no key at all. A box that works has `root:root 0600`.

    So the snippet OWNS the file metadata rather than assuming it:

      * dedupe, not blind append — onstart re-runs on every resume, and the old
        `>>` grew the file by a duplicate line each time;
      * chown root:root + chmod 700/600 on the dir and the file — this is the
        actual repair, and it is what makes the fix host-agnostic;
      * a BOUNDED background re-assert (SSH_FIX_TRIES x SSH_FIX_SLEEP_S = 5 min),
        because vast's own key push races onstart and can re-create the file
        with host ownership *after* we already fixed it. Bounded and detached:
        it can neither delay the real onstart nor spin forever.

    POSIX-only (no `seq`) — some slim images have coreutils but not much else.
    """
    # The `tools/vast/herdd.py` path in the second emitted line is STALE
    # after this move and is left byte-identical on purpose: it is part of the
    # onstart text every box we rent carries, `herdd.py` still emits the same
    # bytes during the add-only phase, and `test_ssh_access.py` executes this
    # shell verbatim. Repointing it is a step-6 edit, not a port edit.
    q = shlex.quote(pub)
    return (
        f"{models.SSH_INJECT_MARKER}: sshd StrictModes needs root:root 0600 — see\n"
        "# ssh_authorized_keys_snippet() in tools/vast/herdd.py\n"
        "_herdd_fix_ssh() {\n"
        "mkdir -p /root/.ssh\n"
        f"grep -qxF {q} /root/.ssh/authorized_keys 2>/dev/null || "
        f"echo {q} >> /root/.ssh/authorized_keys\n"
        "chown root:root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null || true\n"
        "chmod 700 /root/.ssh 2>/dev/null || true\n"
        "chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true\n"
        "}\n"
        "_herdd_fix_ssh\n"
        f"( _n=0; while [ $_n -lt {SSH_FIX_TRIES} ]; do sleep {SSH_FIX_SLEEP_S}; "
        "_herdd_fix_ssh; _n=$((_n+1)); done ) >/dev/null 2>&1 &\n"
    )


# moved-from: herdd.with_ssh_inject
def with_ssh_inject(onstart: str | None, ssh_key_file: str | None = None,
                    pub: str | None = None) -> str | None:
    """Prepend `ssh_authorized_keys_snippet` to an onstart, idempotently.

    Returns the onstart unchanged when no local pubkey resolves (nothing to
    install) or when the marker is already present (a spec replay of an onstart
    we composed earlier must not stack a second copy)."""
    pub = pub or pub_key_text(ssh_key_file)
    if not pub:
        return onstart
    if onstart and models.SSH_INJECT_MARKER in onstart:
        return onstart
    return ssh_authorized_keys_snippet(pub) + (onstart or "")


# moved-from: herdd.attach_ssh_key_soft
def attach_ssh_key_soft(iid: int | str | None, pub: str | None = None,
                        ssh_key_file: str | None = None) -> bool:
    """Best-effort `POST /instances/<id>/ssh/`. Belt to the onstart snippet's
    braces: it registers the key with vast (so the ssh proxy and the host agent
    know about it) but is NOT sufficient on its own — 46449950 had the key
    associated at the API level and still denied every login. Never raises."""
    pub = pub or pub_key_text(ssh_key_file)
    if not pub or not iid:
        return False
    try:
        ok, _d, _err = api.request_soft("POST", f"v0/instances/{iid}/ssh/",
                                        {"ssh_key": pub})
        return bool(ok)
    except Exception:
        return False


# Kept in this module rather than `cli/debug.py`: `_print_ssh` (below) and
# `cmd_ssh` are its only callers, and it is a connect-time hint about the box
# you are connecting to. Flagged as a boundary symbol by the port manifest,
# which allows either home.
# moved-from: herdd._debug_hold_reminder
def _debug_hold_reminder() -> None:
    """Connect-time hint (STDERR only, so --print/_print_ssh stdout stays clean):
    a CRASHED training box holds SSH-able for FAIL_HOLD_MINUTES (default 15)."""
    print(
        "note: if this is a training box that CRASHED, it holds SSH-able for "
        "FAIL_HOLD_MINUTES (default 15) so you can debug/fix in place.\n"
        "      keep it open / tear it down with: "
        "tools/vast/debug_box.sh extend|stop <RUN_ID>  "
        "(or: herdd debug extend|stop <RUN_ID>)",
        file=sys.stderr,
    )


# moved-from: herdd.ssh_access_warning
def ssh_access_warning(i: models.Payload | None) -> str | None:
    """Pre-connect diagnosis for a box born without the authorized_keys repair.

    Returns a stderr-ready string, or None when the box looks ssh-able. Reads
    only the instance dict the caller already fetched — no probing, no cost.

    Why this exists: the raw symptom is `Permission denied (publickey)`, which
    reads like a local key problem and sends you looking at ~/.ssh, the vast
    account keys, or the host — all of which will look FINE, because vast
    happily reports the key associated and even writes it into the container.
    The real question is only ever "was this box created by a path that installs
    the key properly", and the instance's own stored onstart answers it."""
    if models.instance_ssh_install(i) != "none":
        return None
    iid = (i or {}).get("id")
    return (
        f"!! ssh: instance {iid} was created WITHOUT herdd's authorized_keys "
        f"install/repair (its stored onstart carries no {models.SSH_INJECT_MARKER!r}).\n"
        f"   Likely origin: a supervisor eviction relaunch, a handoff "
        f"understudy, a pre-2026-07-31 launch, or a box rented outside "
        f"herdd.\n"
        f"   Expect `Permission denied (publickey)` even though `herdd show "
        f"{iid}` lists your key and the API says it is associated: vast may "
        f"write /root/.ssh/authorized_keys as its own host user "
        f"(vastai_kaalia:docker 0644), and sshd's StrictModes refuses a file "
        f"owned by neither root nor the target user.\n"
        f"   The onstart is fixed at CREATE time, so a park/resume cannot "
        f"repair it. Reach the box over B2 instead (`herdd job submit` / "
        f"`runs` / `logs`), or destroy + relaunch — new launches install and "
        f"re-assert the key themselves.")


# moved-from: herdd.SSH_STRICTMODES_HINT
SSH_STRICTMODES_HINT = (
    "!! ssh: the box refused your key (publickey). Its onstart DOES install the "
    "key, so this is almost certainly the StrictModes footgun: vast's host "
    "daemon writes /root/.ssh/authorized_keys as its own user "
    "(vastai_kaalia:docker 0644) and sshd refuses a file owned by neither root "
    "nor the target user — the key is present and unreadable.\n"
    "   Boxes launched from 2026-07-31 on repair this themselves at boot; this "
    "one predates that, and the onstart is fixed at create time. Reach it over "
    "B2 (`boxstate.py` / `runs` / `job submit`), or destroy + relaunch.\n"
    "   Do NOT just 'relaunch on a different host' — that reflex was masking "
    "this bug. Detail: tools/vast/DEBUG_BOX.md.")


# moved-from: herdd._warn_ssh_access
def _warn_ssh_access(i: models.Payload | None) -> None:
    msg = ssh_access_warning(i)
    if msg:
        print(msg, file=sys.stderr)


# moved-from: herdd._ssh_auth_preflight
def _ssh_auth_preflight(i: models.Payload | None, host: str, port: int) -> None:
    """Say WHY before `ssh` leaks a bare `Permission denied (publickey)`.

    Free on a current box: the onstart marker alone is conclusive, so a "v2"
    install skips out with no network at all. Only an older/absent install pays
    for one BatchMode round trip (~1 s), and that population shrinks to zero as
    boxes rotate. Never blocks the connection — a probe failure is advisory."""
    kind = models.instance_ssh_install(i)
    if kind == "v2":
        return
    if kind == "none":
        _warn_ssh_access(i)
        return
    try:
        r = subprocess.run(
            ["ssh", "-p", str(port), f"root@{host}", "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8",
             "-o", "LogLevel=ERROR", "true"],
            capture_output=True, text=True, timeout=25)
    except Exception:
        return                                    # probe is best-effort, always
    if r.returncode != 0 and "publickey" in (r.stderr or ""):
        print(SSH_STRICTMODES_HINT, file=sys.stderr)


# moved-from: herdd._ssh_endpoints
def _ssh_endpoints(i: models.Payload) -> list[tuple[str, int, str]]:
    """Candidate (host, port, kind) ssh endpoints for an instance, best first.
    The DIRECT mapping (public_ipaddr + the container's 22/tcp HostPort) is
    ground truth from the machine; the api ssh_host/ssh_port can go STALE
    after a park/resume (observed live 2026-07-10: the ssh proxy stayed dead
    minutes after a restart while the re-mapped direct port served). Dedup
    preserves order."""
    out: list[tuple[str, int, str]] = []
    ip = i.get("public_ipaddr")
    mapped = (i.get("ports") or {}).get("22/tcp") or []
    hp = next((p.get("HostPort") for p in mapped if p.get("HostPort")), None)
    try:
        if ip and hp:
            out.append((ip, int(hp), "direct"))
    except (TypeError, ValueError):
        pass
    if i.get("ssh_host") and i.get("ssh_port"):
        e = (i["ssh_host"], int(i["ssh_port"]), "api")
        if not any(c[:2] == e[:2] for c in out):
            out.append(e)
    return out


# moved-from: herdd._pick_ssh_endpoint
def _pick_ssh_endpoint(i: models.Payload, probe_timeout: float = 4
                       ) -> tuple[str | None, int | None, str | None]:
    """First candidate endpoint that accepts a TCP connect (the stale-proxy
    guard); falls back to the first candidate unprobed if none answers.
    Returns (host, port, kind) or (None, None, None)."""
    cands = _ssh_endpoints(i)
    for host, port, kind in cands:
        try:
            with socket.create_connection((host, port), timeout=probe_timeout):
                return host, port, kind
        except OSError:
            continue
    return cands[0] if cands else (None, None, None)


# moved-from: herdd._print_ssh
def _print_ssh(iid: int | str) -> None:
    i = lifecycle._get_instance(iid)
    host, port, _ = _pick_ssh_endpoint(i)
    if host and port:
        print(f"ssh -p {port} root@{host}")
    else:
        print(f"(instance {iid} has no ssh endpoint yet; status={i.get('actual_status')})")
    _warn_ssh_access(i)
    _debug_hold_reminder()


# moved-from: herdd._tunnel_background
def _tunnel_background(a: argparse.Namespace, cmd: list[str], host: str,
                       port: int) -> None:
    """--background: spawn the ssh local-forward DETACHED (own session, survives
    this process) instead of os.execvp-blocking, so an automated eval can open
    the tunnel inline. Writes a pidfile + log and verifies the forward actually
    came up (ExitOnForwardFailure=yes makes ssh exit fast if the -L bind fails).
    Prints pid + port so it can be torn down: kill $(cat <pidfile>)."""
    out_dir = os.path.join(_REPO_ROOT, "out")   # see _REPO_ROOT: NOT __file__ here
    os.makedirs(out_dir, exist_ok=True)
    pidfile = a.pidfile or os.path.join(out_dir, f"vast_tunnel_{a.id}_{a.local}.pid")
    logfile = a.logfile or os.path.join(out_dir, f"vast_tunnel_{a.id}_{a.local}.log")
    logf = open(logfile, "ab")
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
            start_new_session=True,  # detach (setsid): outlives this herdd process
        )
    finally:
        logf.close()  # the child holds its own dup'd fd
    # Verify the forward bound: poll the local port, but bail early if ssh died
    # (ExitOnForwardFailure exits non-zero on a failed/duplicate bind or bad auth).
    up = False
    for _ in range(30):
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", a.local), timeout=1):
                up = True
                break
        except OSError:
            time.sleep(1)
    if proc.poll() is not None:
        try:
            with open(logfile) as fh:
                tail = fh.read()[-500:]
        except OSError:
            tail = ""
        sys.exit(f"error: tunnel ssh exited rc={proc.returncode} (forward failed / "
                 f"port {a.local} busy / auth?) — see {logfile}\n{tail}")
    with open(pidfile, "w") as fh:
        fh.write(f"{proc.pid}\n")
    state = "up" if up else "spawned (port not answering yet — check the log)"
    print(f"tunnel {state}: 127.0.0.1:{a.local} -> {host}:{port} container:{a.remote}  pid {proc.pid}")  # noqa: E501 — operator line kept byte-identical to herdd's
    print(f"  pidfile : {pidfile}")
    print(f"  log     : {logfile}")
    print(f"  teardown: kill $(cat {shlex.quote(pidfile)})")
