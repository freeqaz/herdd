"""vastlib.boxes.remote — running a command on a box we may not be able to log in to.

Why this exists
---------------
Two transports, in opposite states by construction, with one shared result
shape:

* `_vast_execute_soft` — `PUT /v0/instances/command/{id}/`, which runs
  `ls`/`rm`/`du` against an instance's filesystem **including a stopped one**,
  with no GPU contract entered (billing docs: charges begin at `running`). It
  is what makes disk salvage off an evicted box possible at all.
* `_ssh_exec_soft` — plain ssh, because vast **refuses `execute` on a RUNNING
  instance** (OBSERVED 2026-08-05, box 46866095: `400 invalid_args — "Execute
  command only avail on stopped instances. Use ssh to run commands on running
  instances."`).

Neither covers both states, so `boxes/salvage.py` picks between them per box —
and that only works because the two mirror each other's semantics EXACTLY. See
the transport-only-ok trap below; it is the single most important property in
this module.

The transport-only-`ok` contract (do not "improve" it)
------------------------------------------------------
`ok=True` means the TRANSPORT worked, not that the remote command succeeded.
vast's endpoint answers HTTP 200 with the command's stderr in the body and
never surfaces an exit code, so `_ssh_exec_soft` deliberately mirrors that: a
non-zero remote `ls` is still `ok=True` with its error text concatenated into
`data` (`(r.stdout or "") + (r.stderr or "")`). `ok=False` is reserved for
ssh's own failures (rc 255 / no endpoint / timeout / OSError). Likewise
`_vast_copy_direct_soft`'s `ok=True` means only "vast accepted the request" —
it is fire-and-forget, and the success slot holds a human MESSAGE string.

All three are `core.result.Soft` (shape A) with **`data=""` on failure, not
`None`** — `request_soft` is the one shape-A function that uses `None` there.
That asymmetry is load-bearing: `salvage.survey_dest_files` splits
absent/unreadable/listing off the `ls` ERROR TEXT precisely because `ok` cannot
be trusted, and normalising either half collapses a three-outcome verification
design into two, making `unverifiable` nearly unreachable.

What is deliberately NOT here
-----------------------------
* **No salvage policy.** What to survey, what to copy, what counts as verified
  and what the record does next is `boxes/salvage.py`. This module only knows
  how to make a box run a string and hand back its bytes.
* **No `_get_instance`.** `_ssh_exec_soft` needs one, and it lives in
  `boxes.lifecycle` (it uses the RAISING `request`, which is why the call here
  is wrapped in `except SystemExit`). Called as a module attribute so the
  suite's patch of that seam keeps steering.
* **No endpoint logic.** `boxes.ssh._pick_ssh_endpoint` owns the direct-vs-api
  candidate order and the stale-proxy probe; calling it as a module attribute
  is what keeps four existing patch sites live.
* **No auth on the result poll.** The `result_url` fetch is the only other
  `urllib` use in the package besides `core.api`, deliberately: it is a
  PRE-SIGNED URL and must be fetched WITHOUT our Authorization header, matching
  upstream `vast-python`. It therefore cannot go through the `request_soft`
  funnel, and a test that wants to steer it patches `urllib.request.urlopen`.
* **No rename of `_EXEC_NONCE_TAG`.** `"herdd_exec"` is a frozen string: it
  is written into a LIVE box's `result_url` log, which a half-migrated tree may
  still be reading with the old code, and the suite extracts the nonce with
  `__herdd_exec_([0-9a-f]+)_BEGIN__`.

Provenance: moved verbatim from `tools/vast/herdd.py` (plan §8 step 3),
2026-08-16, including the two load-bearing prose banners that precede the
symbols (the REST-vs-CLI transport rationale and the `result_url` per-instance
defect note) — they travel with the code they explain. Behavior-preserving:
bodies copied, annotations and `core.result.Soft` construction added (the
NamedTuple compares and unpacks identically to the bare tuples it replaces).
Every symbol carries its `# moved-from:` marker (grammar: `vastlib/README.md`
§2). The flat `herdd.py` copies stay live until step 6.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable

from vastlib.boxes import lifecycle, ssh
from vastlib.core import api, result

# --- salvage transports (the I/O half of salvage.py) ------------------------ #
#
# We drive the vast REST API directly rather than shelling out to the `vastai`
# CLI. NOTE (2026-08-05, owner-authorized): `vastai 1.5.2` IS now installed in
# the repo venv, so this is a real choice and not a workaround. An earlier claim
# that the CLI was "neutered to 17-byte `exit 0` stubs" was a RED HERRING — those
# two files live in throwaway ~/tmp/serve-* sandboxes and were never on PATH.
# The decision stands on its own three legs:
#   1. `herdd.py` already owns an authenticated, retry-classified HTTP layer
#      (`request_soft` + `_classify_http`). Salvage inherits transient-vs-fatal
#      retry — which is load-bearing here, since the whole "retry, do not declare
#      the disk lost" story keys off that classification — plus the `.env` key
#      resolution and the monkeypatch seams the suite already uses.
#   2. fleetd would otherwise gain a runtime dependency on a pip package it does
#      not install, inside a systemd user unit.
#   3. The CLI does NOT read `VASTAI_API_KEY` from the environment (it wants
#      `~/.config/vastai/vast_api_key` or an explicit `--api-key`), and a 403
#      "This action requires login" is what a missing flag looks like. We have
#      deliberately not written the key to a second location.
# The cost is that we pin two request shapes by hand; both are quoted from
# upstream `vast-python/vast.py` master and recorded in
# DISK_ACCESS_FINDINGS_2026-08-05.md. `vastai copy --explain` / `--curl` is the
# way to re-confirm either shape if the API moves.
# (That first leg now names `vastlib.core.api`, which is where `request_soft`
# and `_classify_http` landed; the argument is unchanged.)

# moved-from: herdd.SALVAGE_EXEC_POLL_S
SALVAGE_EXEC_POLL_S = 0.3         # first poll delay; backs off to the cap below.
# moved-from: herdd.SALVAGE_EXEC_POLL_CAP_S
SALVAGE_EXEC_POLL_CAP_S = 2.0     # per-poll ceiling once backed off.
# moved-from: herdd.SALVAGE_EXEC_BUDGET_S
SALVAGE_EXEC_BUDGET_S = 25.0      # WALL-CLOCK budget for one survey's result
                                  # poll. Upstream vast.py polls a flat 30 x
                                  # 0.3s = 9s; an `ls -lR` over a work tree with
                                  # several ~1 GB checkpoints, on a host that has
                                  # just had an instance evicted, is exactly the
                                  # slow case, and a 9s budget turns that into a
                                  # false "nothing to salvage". A timeout here is
                                  # no longer terminal (the record retries next
                                  # tick), so the budget is chosen to usually GET
                                  # the answer rather than to protect the tick.
# moved-from: herdd.SALVAGE_EXEC_MAX_POLLS
SALVAGE_EXEC_MAX_POLLS = 40       # request cap, so a fast-failing result_url
                                  # cannot spin the budget away in tight retries.


# --- correlating a result_url read with the call that asked for it ---------- #
#
# THE DEFECT (2026-08-05). `result_url` is a FIXED PER-INSTANCE log path, not a
# per-request one. Two callers surveying the same box therefore poll the same
# URL and can each read the OTHER's output — and a single caller can read the
# PREVIOUS call's output before the host daemon has overwritten it. Salvage's
# byte-for-byte verification treats that listing as its ORACLE, so a crossed or
# stale read is not a cosmetic glitch: it is a wrong answer presented as truth,
# in a code path whose whole purpose is to avoid publishing an authoritative
# false negative over a disk full of checkpoints.
#
# Observed crossed once under concurrency. A solo re-test was 0/6 — which is the
# point: this cannot be reproduced by running it alone, so "I could not
# reproduce it" is not evidence of absence and the guard is not optional.
#
# THE GUARD. Bracket the command with a per-call nonce and refuse any body that
# does not carry it. That fixes both shapes at once (another caller's output and
# our own stale output both lack this call's nonce) and it needs no lock, no
# coordination, and no assumption about how many processes are running.
#
# WHY IT DEGRADES INSTEAD OF INSISTING. vast's own docs describe `execute` as
# offering `ls` / `rm` / `du`; whether the host daemon accepts a compound
# command is not something we can settle without a stopped box to try it on.
# Making the guard mandatory would convert an unknown into a hard outage of a
# recovery path that is only ever used when something has already gone wrong.
# So a validation-shaped refusal of the wrapped form falls back to the bare
# command in DEGRADED mode, which substitutes a weaker but real check: read the
# body twice and require it to be byte-identical. A body that is being
# overwritten by a concurrent writer is unlikely to be stable across two reads;
# a settled one is. Degraded mode says so out loud rather than silently
# pretending to the same guarantee.
# moved-from: herdd._EXEC_NONCE_TAG
_EXEC_NONCE_TAG = "herdd_exec"


# moved-from: herdd._exec_nonce_markers
def _exec_nonce_markers() -> tuple[str, str]:
    """`(begin, end)` for one call. Hex so it is shell-safe unquoted, and long
    enough that it cannot collide with a filename in an `ls -lR` body."""
    n = secrets.token_hex(12)
    return f"__{_EXEC_NONCE_TAG}_{n}_BEGIN__", f"__{_EXEC_NONCE_TAG}_{n}_END__"


# moved-from: herdd._exec_wrap
def _exec_wrap(command: str, begin: str, end: str) -> str:
    """Bracket `command` so its output is self-identifying.

    `;` and not `&&`: the markers must be emitted even when the command fails,
    or a legitimate non-zero `ls` (a path that is simply absent — a real answer
    this module distinguishes from an unreadable disk) would look like a
    correlation failure and be retried until the deadline.
    """
    return f"echo {begin}; {command}; echo {end}"


# moved-from: herdd._exec_extract_nonce_block
def _exec_extract_nonce_block(text: str, begin: str, end: str) -> str | None:
    """Return the lines strictly between `begin` and `end`, or None.

    None means "this body is not this call's output" — another caller's, or our
    own previous one, or a partial write that has not reached the end marker
    yet. Every one of those is a keep-polling condition, never a result.

    The LAST begin marker wins: the log is append-shaped, so if our own marker
    somehow appears twice the newest block is ours.

    Matched by CONTAINMENT, not line equality, because vast prepends
    `writeable_path` to lines in some responses and an exact-match test would
    then silently never correlate — which would look identical to "the guard is
    working" while quietly disabling every survey. A 24-hex-char nonce cannot
    collide with a real `ls` line.
    """
    lines = text.splitlines()
    i = next((k for k in range(len(lines) - 1, -1, -1) if begin in lines[k]),
             None)
    if i is None:
        return None
    j = next((k for k in range(i + 1, len(lines)) if end in lines[k]), None)
    if j is None:
        return None
    return "\n".join(lines[i + 1:j])


# moved-from: herdd._exec_refusal_is_validation
def _exec_refusal_is_validation(err: object) -> bool:
    """Did vast refuse the WRAPPED command shape (=> retry bare), or is this a
    real failure (=> report it)?

    Deliberately narrow. A 404 means the instance is gone, which is the one
    answer salvage treats as authoritative — falling back on it would turn
    "disk reclaimed" into a second pointless request, and worse, could let a
    bare-command retry succeed against a different box's log.
    """
    low = str(err or "").lower()
    if any(t in low for t in ("404", "not found", "not_found",
                             "does not exist", "instance not found")):
        return False
    return any(t in low for t in ("invalid_args", "invalid arg", "400",
                                 "unsupported", "not allowed", "not permitted"))


# moved-from: herdd._vast_execute_soft
def _vast_execute_soft(iid: int | str, command: str, *,
                       tries: int = SALVAGE_EXEC_MAX_POLLS,
                       poll_s: float = SALVAGE_EXEC_POLL_S,
                       budget_s: float = SALVAGE_EXEC_BUDGET_S,
                       _sleep: Callable[[float], object] = time.sleep,
                       _now: Callable[[], float] = time.monotonic
                       ) -> result.Soft:
    """`PUT /api/v0/instances/command/{id}/` — run `ls`/`rm`/`du` on an instance's
    filesystem, INCLUDING a stopped/`exited` one, with no GPU contract entered.
    Returns `(ok, text, err)`; never raises, never sys.exits.

    The endpoint is ASYNCHRONOUS: the PUT answers `{"success": true,
    "result_url": ..., "writeable_path": ...}` and the output has to be fetched
    from `result_url` (a pre-signed URL — deliberately fetched WITHOUT our auth
    header, matching upstream) once the host daemon has written it. A 404 on the
    dead instance is the expected, meaningful failure: the host reclaimed it.

    `result_url` IS NOT PER-REQUEST — it is a fixed per-instance log path, so a
    naive read can return another concurrent caller's output, or this caller's
    own previous output. The command is therefore bracketed with a per-call
    nonce and a body without it is treated as "not ours yet", never as a result
    (see the block comment above `_exec_nonce_markers`). If vast refuses the
    wrapped shape, the call degrades to the bare command plus a
    read-twice-and-compare stability check, and says so.
    """
    begin, end = _exec_nonce_markers()
    ok, data, err = api.request_soft("PUT", f"/v0/instances/command/{iid}/",
                                     {"command": _exec_wrap(command, begin, end)},
                                     retries=1)
    correlated = True
    if not ok and _exec_refusal_is_validation(err):
        # The endpoint rejected the bracketed form, not the instance. Fall back
        # rather than lose the recovery path entirely — but do NOT pretend to
        # the nonce guarantee afterwards.
        print(f">> WARN: execute on {iid}: vast refused the nonce-bracketed "
              f"command ({err}); falling back to the bare command with a "
              f"read-twice-and-compare stability check instead. This is the "
              f"WEAKER guard — a crossed result_url read is possible.",
              file=sys.stderr)
        correlated = False
        ok, data, err = api.request_soft("PUT", f"/v0/instances/command/{iid}/",
                                         {"command": command}, retries=1)
    if not ok:
        return result.Soft(False, "", err or "execute request failed")
    if not isinstance(data, dict) or not data.get("success"):
        return result.Soft(False, "", str((data or {}).get("msg") or data))
    url = data.get("result_url")
    if not url:
        return result.Soft(False, "",
                           "execute answered success but returned no result_url")
    strip = data.get("writeable_path") or ""
    last = "result_url never returned 200"
    deadline = _now() + float(budget_s)
    delay = float(poll_s)
    prev_body: str | None = None      # degraded mode: the previous read
    for _ in range(max(1, int(tries))):
        _sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                if r.status == 200:
                    text = r.read().decode("utf-8", "replace")
                    if correlated:
                        block = _exec_extract_nonce_block(text, begin, end)
                        if block is not None:
                            return result.Soft(
                                True, _strip_writeable_path(block, strip), None)
                        # Someone else's output, our own stale output, or a
                        # partial write. Keep polling; this is exactly the read
                        # that used to be returned as truth.
                        last = ("result_url returned a body that does not "
                                "carry this call's nonce — a crossed or stale "
                                "listing, not this survey's answer")
                    elif prev_body is not None and text == prev_body:
                        return result.Soft(
                            True, _strip_writeable_path(text, strip), None)
                    else:
                        prev_body = text
                        last = ("result_url body not yet stable across two "
                                "reads (degraded, uncorrelated mode)")
                else:
                    last = f"result_url HTTP {r.status}"
        except Exception as e:                        # noqa: BLE001 — total by design
            last = f"{type(e).__name__}: {e}"
        if _now() >= deadline:
            break
        delay = min(delay * 1.6, SALVAGE_EXEC_POLL_CAP_S)
    return result.Soft(False, "", f"{last} (gave up after {budget_s:g}s)")


# moved-from: herdd._strip_writeable_path
def _strip_writeable_path(text: str, strip: str) -> str:
    """Remove vast's `writeable_path` prefix, ANCHORED TO LINE STARTS.

    Upstream does a bare `text.replace(writeable_path, "")` over the whole body.
    That prefix can be short or common, and an unanchored replace inside a
    FILENAME silently corrupts the survey the byte-for-byte verification is
    checked against — and mangles `ls -lR` section headers, which the parser
    splits on. Anchoring makes it a no-op on anything but the path prefix vast
    actually prepends.
    """
    if not strip:
        return text
    return "\n".join(ln[len(strip):] if ln.startswith(strip) else ln
                     for ln in text.splitlines())


# moved-from: herdd._vast_copy_direct_soft
def _vast_copy_direct_soft(src_iid: int | str, src_path: str,
                           dst_iid: int | str, dst_path: str) -> result.Soft:
    """`PUT /api/v0/commands/copy_direct/` — host-to-host copy between two
    instances, either of which may be stopped. Returns `(ok, msg, err)`.

    FIRE-AND-FORGET. A `success: true` here means the transfer was INITIATED
    ("check instance status bar for progress updates (~30 seconds delayed)"), not
    that a single byte landed. `salvage.verify_salvage` is what decides whether
    anything was actually salvaged; this function's `ok` is only "vast accepted
    the request".
    """
    body = {"client_id": "me", "src_id": _int_or(src_iid), "dst_id": _int_or(dst_iid),
            "src_path": src_path, "dst_path": dst_path}
    ok, data, err = api.request_soft("PUT", "/v0/commands/copy_direct/", body,
                                     retries=1)
    if not ok:
        return result.Soft(False, "", err or "copy_direct request failed")
    if not isinstance(data, dict) or not data.get("success"):
        return result.Soft(False, "", str((data or {}).get("msg") or data))
    return result.Soft(True, str(data.get("msg") or "copy initiated"), None)


# moved-from: herdd._int_or
def _int_or(v: Any) -> Any:  # noqa: ANN401 — passthrough of an id of unknown type
    """Instance ids go over the wire as ints (upstream `parse_vast_url` yields the
    bare id). A non-numeric id passes through unchanged rather than exploding —
    the API will reject it and we report that."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


# moved-from: herdd._ssh_exec_soft
def _ssh_exec_soft(iid: int | str, remote: str, *,
                   timeout: int = 180) -> result.Soft:
    """Run a command on a RUNNING box over ssh -> `(ok, text, err)`.

    Exists because **`execute` is refused on a running instance** (OBSERVED
    2026-08-05, box 46866095: `400 invalid_args — "Execute command only avail on
    stopped instances. Use ssh to run commands on running instances."`). The two
    ends of a salvage are in opposite states by construction, so they need
    opposite transports.

    Mirrors `execute`'s SEMANTICS on purpose: the vast endpoint answers HTTP 200
    with the command's stderr in the body and never surfaces its exit code, so a
    non-zero `ls` here is still `ok=True` with the error text in `text`. That
    keeps `salvage.survey_dest_files`' three-way "absent vs unreadable vs
    listing" classification identical across both transports. `ok=False` is
    reserved for ssh's own failures (rc 255 / no endpoint / timeout).
    """
    try:
        i = lifecycle._get_instance(iid)
    except SystemExit:
        return result.Soft(False, "", f"instance {iid} is not listed")
    host, port, _ = ssh._pick_ssh_endpoint(i)
    if not (host and port):
        return result.Soft(False, "", (f"no ssh endpoint for {iid} "
                                       f"(status={i.get('actual_status')})"))
    try:
        r = subprocess.run(["ssh", "-p", str(port), f"root@{host}",
                            "-o", "StrictHostKeyChecking=accept-new",
                            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
                            "-o", "BatchMode=yes", remote],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return result.Soft(False, "", f"ssh to {iid} timed out after {timeout}s")
    except OSError as e:
        return result.Soft(False, "", f"{type(e).__name__}: {e}")
    if r.returncode == 255:                       # ssh transport failure
        return result.Soft(False, "",
                           (r.stderr or "").strip()[:300] or "ssh exited 255")
    return result.Soft(True, (r.stdout or "") + (r.stderr or ""), None)


# --------------------------------------------------------------------------- #
# CRED-BROKER NONCE REGISTRATION
#
# Not a transport in the sense above — it is a one-shot POST to OUR broker, not
# to a box — but it lands here for the reason the module docstring gives for
# the `result_url` poll: this is the second place in the package that speaks
# `urllib` without going through `core.api`, deliberately (different host,
# different auth header, and it must never raise). Its callers are `cli/start`,
# `cli/job/attach` and, through the re-attach path, `cli/job/supervise` — three
# command modules in two different `cli/` subpackages, so a `cli`-side home
# would have been a shared helper reaching sideways across the top ring.
# --------------------------------------------------------------------------- #

# moved-from: herdd._broker_register
def _broker_register(iid: int | str, nonce: str) -> None:
    """Best-effort nonce registration with the cred broker (§2.2 /v1/register):
    an attach-time nonce is NOT in the box's launch extra_env, so the broker
    learns its sha256 out-of-band (rotation lane for boxes launched before
    nonce injection existed). Silent no-op unless BOTH CRED_BROKER_URL and
    CRED_BROKER_ADMIN_TOKEN are set; swallows EVERY error — attach must never
    fail on broker absence. The raw nonce never goes over this call."""
    url = os.environ.get("CRED_BROKER_URL")
    tok = os.environ.get("CRED_BROKER_ADMIN_TOKEN")
    if not (url and tok):
        return
    try:
        body = json.dumps({
            "instance_id": int(iid),
            "nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
        }).encode("utf-8")
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/register", data=body,
            headers={"Content-Type": "application/json", "X-Broker-Admin": tok})
        with urllib.request.urlopen(req, timeout=3):
            pass
        print(f">> broker: registered attach nonce for box {iid}")
    except Exception:
        pass
