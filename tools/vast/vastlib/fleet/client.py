"""vastlib.fleet.client — the JSON-over-unix-socket client for fleetd.

Why this exists
---------------
`fleetd` — the persistent fleet-supervision daemon (`tools/vast/FLEETD_DESIGN.md`)
— owns ALL box babysitting; agents TALK to it (watch/pause/park/resume/destroy)
instead of spawning and pkilling `supervise` / `job supervise` processes. This
module is the thin client half of that conversation plus the compat shims that
hand a legacy supervise invocation to the daemon when it is up (FLEETD_DESIGN
§5/§6).

It also HOLDS THE PROTOCOL. Before the port there were two independent copies of
every wire constant — `herdd.FLEET_PROTO_VERSION` / `fleetd.VERSION` and
`herdd.FLEET_UNIT_NAME` / `fleetd.UNIT_NAME` — with nothing comparing them.
`Server.handle` REFUSES a request whose `v` does not match, so a divergence is
not a degradation, it is a total client outage on a daemon that keeps running
and keeps looking healthy. They are collapsed here into one definition each,
below both consumers in the DAG: the CLI imports this module and so does the
daemon, so the dependency direction is no longer inverted (the daemon's protocol
used to live in the CLI). `test_vastlib_fleet_client.py` pins each collapsed
constant against BOTH original literals for as long as the flat files exist.

What is deliberately NOT here
-----------------------------
* **No response models.** The `watch` reply OMITS `standing` entirely when it is
  falsey so that a pre-2026-08-14 client receives a byte-identical payload
  (`test_standing_watch.py` pins it). A typed response model with a default
  `False` would serialize the key unconditionally and break that silently — the
  wire is a frozen contract, not a schema to normalize.
* **No socket for `fleet_watch_supervision` / `fleet_recoveries_in_flight`.**
  Both read `state.json` DIRECTLY, on purpose: one runs on the submit path
  (must cost nothing and must never fail a submit because a reconcile tick is
  slow), the other runs at the exact moment an operator types `fleet restart`,
  which is very often the moment the daemon is not answering. Routing either
  through `fleet_request` "for consistency" inverts both design decisions.
* **No caching of the socket path.** `fleet_sock_path` re-reads `FLEETD_SOCK`
  on EVERY call because `conftest.py`'s autouse fixture redirects that env var
  to a nonexistent path for the whole suite — the only thing standing between a
  fixture-id `destroy` intent and the LIVE daemon (measured leak, 2026-08-01).
  A snapshot at import time, or a read through a config layer that caches,
  re-arms that leak and no test would fail.
* **No error class.** The `nodaemon:` / `timeout:` / `socket:` / `refused:` /
  `malformed response:` prefixes are a de-facto typed enum expressed as string
  prefixes, parsed by `str.startswith` at four call sites (one of which slices
  `err[8:]` to strip `refused:`). They stay strings.
* **No `cmd_fleet_*` presentation.** The 14 argparse-facing dispatch functions
  are `cli/fleet/` (plan §8 step 6); they consume `fleet_request`,
  `_fleet_call_or_die` and `fleet_follow` from here.
* **No `Fleet`/`Server`.** The daemon side is `fleet/daemon.py`; the pure state
  folds are `fleet/rows.py` + `fleet/state.py`, which import the two on-disk
  filenames from here rather than keeping a second literal.

Provenance: moved from `herdd.py` (the `fleet` client block) plus the two
duplicated `fleetd.py` protocol constants, plan §8 step 5. Behavior-preserving;
the one non-textual line is `_TOOLS_DIR` (see its comment).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from vastlib.core import config, fmt, result
from vastlib.market import offers
from vastlib.storage import dashcache

if TYPE_CHECKING:                                   # pragma: no cover - typing only
    from collections.abc import Mapping

# --------------------------------------------------------------------------- #
# the wire protocol — ONE definition each (see the module docstring)
# --------------------------------------------------------------------------- #
# moved-from: herdd.FLEET_PROTO_VERSION
# moved-from: fleetd.VERSION
FLEET_PROTO_VERSION = 1
# moved-from: herdd.FLEET_UNIT_NAME
# moved-from: fleetd.UNIT_NAME
FLEET_UNIT_NAME = "vast-fleetd.service"
# moved-from: herdd.FLEET_SOCK_TIMEOUT_S
FLEET_SOCK_TIMEOUT_S = 15

# The two on-disk filenames under the state dir. FROZEN, and they were
# duplicated too: `herdd.fleet_state_path`/`fleet_journal_path` hardcoded the
# same literals `fleetd.STATE_NAME`/`JOURNAL_NAME` declare — an external reader
# that disagrees reads a file nobody writes.
#
# DELIBERATELY MARKER-LESS (ruled 2026-08-16, wave 6a; fleetd-reexports H4):
# `fleet/state.py` owns the `fleetd.STATE_NAME` / `fleetd.JOURNAL_NAME`
# mappings — they are on-disk names belonging to the state writer, and one flat
# name cannot have two rename targets. The literals stay here because the
# client-side path helpers below read them; only the rename claim moved.
STATE_NAME = "state.json"
JOURNAL_NAME = "journal.ndjsonl"

# `_git_rev_short` shells `git -C <dir>`, and in `herdd.py` that dir is
# `dirname(dirname(abspath(__file__)))` — the repo's `tools/` directory, which
# is correct only from `tools/vast/herdd.py`. Copied verbatim into
# `tools/vast/vastlib/fleet/client.py` the same expression yields
# `tools/vast/vastlib`, which is still inside the repo, so `git rev-parse` keeps
# WORKING and keeps returning a plausible answer — the failure is silent by
# construction. Hoisted to a module constant with the depth recomputed, the way
# `core.config._HERE` and `boxes.ssh._REPO_ROOT` are, and pinned by
# `test_vastlib_fleet_client.py::test_tools_dir_matches_herdd_computation`.
_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


# --------------------------------------------------------------------------- #
# paths — env read on EVERY call, never cached (see the module docstring)
# --------------------------------------------------------------------------- #
# moved-from: herdd.fleet_state_dir
def fleet_state_dir() -> str:
    """State/journal/socket directory (FLEETD_STATE_DIR overrides — the tests and
    a dry-run soak use their own).

    The formula itself moved DOWN to `core.config` on 2026-08-20 so that
    `market.hostrep` could resolve the same directory without importing this
    module: `fleet` sits above `market` in the ring order, so the edge is
    forbidden, and re-spelling the default in a second file is how the two would
    silently disagree the next time it changes. The name stays here — ~30 call
    sites bind it, and the tests monkeypatch it by module attribute."""
    return config.fleet_state_dir()


# moved-from: herdd.fleet_sock_path
def fleet_sock_path() -> str:
    return os.environ.get("FLEETD_SOCK") or os.path.join(fleet_state_dir(),
                                                         "fleetd.sock")


# moved-from: herdd.fleet_journal_path
def fleet_journal_path() -> str:
    return os.path.join(fleet_state_dir(), JOURNAL_NAME)


# moved-from: herdd.fleet_state_path
def fleet_state_path() -> str:
    return os.path.join(fleet_state_dir(), STATE_NAME)


# --------------------------------------------------------------------------- #
# state.json readers — no socket, by design
# --------------------------------------------------------------------------- #
# moved-from: herdd.fleet_watch_supervision
def fleet_watch_supervision(iid: object) -> tuple[str, dict[str, Any]]:
    """What is supervising box `iid` right now, for a submit-time advisory.

    Returns (level, detail) where level is one of:
      "policy"    a spend-capable watch (jobs/run/serve) — full ladder armed
      "lapsed"    an ADOPTED `bare` watch holding an INHERITED ceiling. This is
                  the dangerous one: it means a previous batch on this box
                  DRAINED, its jobs watch finished normally, and the safety net
                  re-adopted the box. The spend ceiling survived; the ladder
                  (bid rescue, eviction replacement, drain-park) did not.
      "bare"      an adopted `bare` watch on a provisional ceiling
      "none"      no watch — normal immediately after a launch, since the
                  documented order is rent -> submit -> arm
      "unknown"   fleetd state unreadable; we are NOT going to guess

    Reads `state.json` DIRECTLY — no socket, no API — for the same reason
    `fleet_recoveries_in_flight` does: this runs on the submit path, must cost
    nothing, and must not fail a submit because a daemon is slow. Every error
    is `unknown`, which advises and never blocks.

    Why this exists: box 47511739, 2026-08-12. Its `jobs` watch finished
    `drained` at 04:53 when K1 and M1 went terminal, the safety net adopted it
    `bare`, and three probe runs were then submitted onto a SPOT box with no bid
    rescue and no eviction replacement. fleetd raised exactly the right alarm
    and it went unread — so the notice belongs where the operator is looking,
    which is the submit they are typing.
    """
    try:
        with open(fleet_state_path()) as fh:
            st = json.load(fh)
    except Exception:
        return "unknown", {}
    try:
        want = str(iid)
        for w in (st.get("watches") or {}).values():
            if str(w.get("iid")) != want:
                continue
            prof = w.get("profile")
            # `ceiling_by_box` maps a box to its CEILING ID, not to a spend, so
            # it needs the second hop through `ceilings`. Reading it directly
            # printed the ceiling id as dollars — box 47939448 was reported as
            # "$47939448.00 of $1.50 spent ($0.00 left)" on a watch 6.5e-05 in.
            _cid = (st.get("ceiling_by_box") or {}).get(want)
            _ceil = (st.get("ceilings") or {}).get(_cid) or {}
            d = {"profile": prof, "budget_usd": w.get("budget_usd"),
                 "spend_usd": _ceil.get("spend_usd", w.get("spend_usd")),
                 "ceiling_source": w.get("ceiling_source"),
                 # A STANDING watch reads dormant between waves and is still
                 # fully armed — the opposite of the `lapsed` shape below, and
                 # worth saying so on the submit that is about to wake it.
                 "standing": bool(w.get("standing")),
                 "standing_dormant": bool(w.get("standing_dormant")),
                 "adopted": bool(w.get("adopted"))}
            if prof in ("run", "jobs", "serve"):
                return "policy", d
            if w.get("adopted") and w.get("ceiling_source") == "inherited":
                return "lapsed", d
            return "bare", d
        return "none", {}
    except Exception:
        return "unknown", {}


# moved-from: herdd.fleet_recoveries_in_flight
def fleet_recoveries_in_flight() -> list[Any]:
    """What a `fleet restart` right now would interrupt (recalibration
    2026-08-09, item C). Reads state.json DIRECTLY — no socket, no API — because
    the moment an operator types `fleet restart` is very often the moment the
    daemon is not answering, and a guard that needs a healthy daemon to warn you
    about restarting an unhealthy one is not a guard.

    `[]` on a missing or unreadable state file: an unknown state is not evidence
    of a recovery, and refusing a restart on a parse error would make a corrupt
    state file unrecoverable-by-restart, which is the one thing a restart is for.

    The pure fold lives in `fleet.rows.recoveries_in_flight`. Before the port it
    lived in `fleetd`, reached by a LAZY `import fleetd` inside this function
    because `fleetd` imports `herdd` at module scope — the one concrete import
    cycle the `fleet/` package exists to break, and it is broken: both sides now
    import the fold downward. The import stays function-local for a different
    and much smaller reason — `fleet.state` imports this module's path constants,
    so a module-scope `from vastlib.fleet import rows` would couple the two
    halves of the package at import time for a call one CLI guard makes. It is
    bound as a module attribute so the patch idiom survives (plan §8b).
    """
    try:
        with open(fleet_state_path()) as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return []
        from vastlib.fleet import rows
        return rows.recoveries_in_flight(state)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# the transport
# --------------------------------------------------------------------------- #
# moved-from: herdd.fleet_request
def fleet_request(op: str, _timeout: float = FLEET_SOCK_TIMEOUT_S,
                  _retries: int = 1,
                  **args: Any) -> result.Soft:  # noqa: ANN401 — arbitrary JSON args
    """One JSON-line request/response against the daemon socket.

    Error taxonomy (review B3) — only the LAST class is a real refusal:
      `nodaemon:` socket absent/refusing        -> caller falls back inline
      `timeout:` / `socket:` transport trouble  -> retried, then falls back
      `refused:` the DAEMON answered with an error -> a real decision; surface it
    A slow reconcile tick must never turn into a hard client failure.

    ONE `socket` binding, deliberately. `herdd`'s copy did `import socket as
    _socket` locally for the connect while its `except` clause named
    `socket.timeout` off the MODULE-level import — both resolved there only
    because `herdd` happens to import `socket` at top level. Carried over
    half-way that becomes a `NameError` raised inside the `try`, i.e. a
    transport blip turned into an exception on the exact path whose whole
    purpose is to degrade into a fallback.
    """
    path = fleet_sock_path()
    req = json.dumps({"v": FLEET_PROTO_VERSION, "op": op, "args": args},
                     sort_keys=True) + "\n"
    last = "unknown"
    for attempt in range(max(1, _retries + 1)):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(_timeout)
            s.connect(path)
        except (FileNotFoundError, ConnectionRefusedError, PermissionError) as e:
            return result.Soft(False, None, f"nodaemon:{type(e).__name__}")
        except OSError as e:
            return result.Soft(False, None, f"nodaemon:{e}")
        try:
            s.sendall(req.encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        except (socket.timeout, TimeoutError) as e:
            last = f"timeout:{e or 'no response'}"
            buf = b""
        except OSError as e:
            last = f"socket:{e}"
            buf = b""
        finally:
            try:
                s.close()
            except OSError:
                pass
        line = buf.split(b"\n", 1)[0].decode(errors="replace").strip() if buf else ""
        if not line:
            if attempt < _retries:
                time.sleep(0.5 * (attempt + 1))       # one backed-off retry
                continue
            return result.Soft(False, None,
                               last if last != "unknown" else "timeout:empty response")
        try:
            resp = json.loads(line)
        except ValueError:
            return result.Soft(False, None, f"malformed response: {line[:200]}")
        if not resp.get("ok"):
            return result.Soft(False, resp.get("data"),
                               "refused:" + str(resp.get("error") or "refused"))
        return result.Soft(True, resp.get("data"), None)
    return result.Soft(False, None, last)


# moved-from: herdd.fleet_daemon_up
def fleet_daemon_up() -> bool:
    ok, _d, _e = fleet_request("ping", _timeout=5)
    return ok


# moved-from: herdd._fleet_policy
def _fleet_policy(a: argparse.Namespace) -> dict[str, Any]:
    """The full parsed namespace as a JSON-safe policy dict, so the daemon can
    rebuild `argparse.Namespace(**policy)` and run the SAME tick functions with
    the SAME flags the operator typed."""
    out: dict[str, Any] = {}
    for k, v in vars(a).items():
        if k in ("func", "jobfunc", "cmd", "jobcmd", "wfcmd"):
            continue
        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
            out[k] = v
    return out


# moved-from: herdd._fleet_delegation_disabled
def _fleet_delegation_disabled(a: object) -> bool:
    # PYTEST_CURRENT_TEST: a test run must NEVER register a watch with the live
    # daemon (the portable supervise-driver tests fake the API, not the socket).
    # This is the SECOND live-fleet guard, independent of conftest's FLEETD_SOCK
    # redirect, and it is an env read evaluated per call on purpose — folding it
    # into a typed policy object resolved once is how it would be lost.
    return (bool(getattr(a, "no_fleet", False))
            or os.environ.get("FLEETD_DISABLE") == "1"
            or bool(os.environ.get("PYTEST_CURRENT_TEST"))
            or bool(getattr(a, "dry_run", False)))   # a dry-run never spends


# moved-from: herdd._fleet_delegate
def _fleet_delegate(a: argparse.Namespace, target: str, profile: str,
                    budget: float | None) -> bool:
    """Shared shim body: register the watch with a live daemon and report True
    (caller returns instead of holding a babysitter process open). False keeps
    the legacy inline loop (no daemon / opted out / dry-run)."""
    if _fleet_delegation_disabled(a):
        return False
    ok, data, err = fleet_request("watch", target=target, profile=profile,
                                  budget_usd=budget, policy=_fleet_policy(a),
                                  requester=_fleet_requester())
    if not ok:
        if not str(err).startswith("refused:"):
            # B3: no daemon, a timeout, a slow tick — ANY transport trouble falls
            # back to the legacy inline babysitter. Delegation must never be
            # strictly worse than the pre-fleetd world.
            print(f">> note: fleetd unreachable ({err}) — running the legacy "
                  f"inline supervisor for this invocation")
            return False
        sys.exit(f"error: fleetd refused the watch ({str(err)[8:]}); "
                 f"retry, or run inline with --no-fleet")
    if isinstance(data, dict) and data.get("redirected_from"):
        # The daemon resolved our id to the watch that already owns this box
        # (its ladder had rented it as a replacement). Follow the daemon's key,
        # or the confirmation and `--follow` both name a watch that is not the
        # one we just registered.
        print(f">> note: {target} is watch {data.get('target')}'s CURRENT box "
              f"(replacement/handoff) — the registration went to that watch, "
              f"keeping its accrued spend")
        target = data.get("target") or target
    print(f">> fleetd is up — registered watch {target} (profile={profile}, "
          f"budget={fmt.dollars(budget) if budget is not None else 'none'}). "
          f"NOTE: a `supervise` PROCESS is deprecated (FLEETD_DESIGN §6); the "
          f"daemon babysits from now on, and survives this shell.")
    print(f"   watch it: {os.path.basename(sys.argv[0])} fleet status | "
          f"{os.path.basename(sys.argv[0])} fleet log -f")
    if isinstance(data, dict) and data.get("note"):
        print(f"   {data['note']}")
    if getattr(a, "follow", False):
        # S7: scripts and runbooks assume `supervise` blocks in the foreground.
        sys.exit(fleet_follow(target, iid=getattr(a, "id", None)))
    return True


# moved-from: herdd._git_rev_short
def _git_rev_short() -> str | None:
    """This checkout's short HEAD, or None.

    Lives here because its only caller is `cmd_fleet_ping`'s version-skew line
    (the daemon runs a RELEASE checkout, so a skew means `fleet deploy`, not
    `fleet restart`). FLAGGED at port time rather than duplicated: `fleetd` has
    its own `git_rev` (fleet/daemon.py), and if a third caller appears this
    belongs in `core/` — not copied a third time.
    """
    try:
        p = subprocess.run(["git", "-C", _TOOLS_DIR,
                            "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() or None
    except Exception:
        return None


def rev_matches(a: str | None, b: str | None) -> bool:
    """Do two ABBREVIATED revs name the same commit?

    Never `==`. `git rev-parse --short` picks its length dynamically to stay
    unambiguous, so the same commit abbreviates to different strings in two
    checkouts — or in one checkout at two times, as the object set grows. The
    daemon stamps its `rev` once at startup and the client re-derives its own
    per call, so an `==` here reports skew for a commit that already matches
    (measured 2026-08-18: `38e76425` vs `38e76425e`, one commit, deploy VERIFIED
    green and the ping line red at the same instant). A skew alarm that fires on
    a correct deploy is worse than no alarm — it is the one the operator learns
    to skip. Prefix semantics are exactly git's own rule for resolving one."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return False
    return a.startswith(b) or b.startswith(a)


# moved-from: herdd._fleet_requester
def _fleet_requester() -> str:
    """Who asked — journaled with every daemon action so a destroy is always
    attributable."""
    try:
        return f"{os.environ.get('USER') or 'user'}@{socket.gethostname()}"
    except Exception:
        return "cli"


# --------------------------------------------------------------------------- #
# the best-effort announcements (called from boxes/ and launch/)
# --------------------------------------------------------------------------- #
# moved-from: herdd.fleet_operator_intent
def fleet_operator_intent(iid: object, kind: str,
                          reason: str | None = None) -> Any:  # noqa: ANN401 — daemon payload
    """B2: tell a live daemon what a human is about to do to this box BEFORE the
    vast PUT/DELETE lands. Workstation-local and label-agnostic (unlike
    runmeta's `stopping` event, which only exists for run:-labelled boxes).
    Best-effort: no daemon, no problem. Returns the daemon's reply or None.

    Why it matters: the jobs ladder deliberately reads "bid box stopped, no
    self-park event" as OUTBID and RESCUES it (SPOT_DESIGN §3.5). With an
    immortal daemon watch, a human's `herdd stop` would otherwise be
    resurrected minutes later and bill all night."""
    ok, data, _err = fleet_request("operator_intent", _timeout=5, _retries=0,
                                   target=str(iid), kind=kind, reason=reason,
                                   requester=_fleet_requester())
    return data if ok else None


def fleet_ticket_placed(target: object, job_id: object = None,
                        source: str | None = None,
                        announce: bool = True) -> Any:  # noqa: ANN401 — daemon payload
    """Tell a live daemon that a NON-TERMINAL ticket now sits in this box's
    queue, so a dormant STANDING watch re-arms without waiting on its own poll.

    Every path that places live work calls this — `job submit`, `job retarget`,
    `job requeue` — because the standing watch's re-arm rule is "a ticket", and
    until 2026-08-27 the only thing that could see one was a queue listing that
    is silent on a parked box and `unknown` on a B2 blip. It fired 0 times in 84
    drains; a retarget onto a drained box left it evicted and undefended.

    Best-effort in the strong sense: no daemon, an older daemon that answers
    `unknown op`, or a refusal all leave the CLI's own work done and unreported.
    Never a refusal — the daemon sets a flag, and the resume still requires the
    box to read LIVE."""
    ok, data, _err = fleet_request("ticket_placed", _timeout=5, _retries=0,
                                   target=str(target),
                                   job_id=None if job_id is None else str(job_id),
                                   source=source, requester=_fleet_requester())
    if ok and announce and (data or {}).get("woken"):
        print(f">> fleetd: the STANDING watch on {target} was dormant — this "
              f"ticket re-arms its ladder (bid defense, outbid rescue, eviction "
              f"replacement) on the next tick")
    return data if ok else None


# moved-from: herdd.fleet_note_operator_stop
def fleet_note_operator_stop(iid: object) -> Any:  # noqa: ANN401 — daemon payload
    """Print the dormancy note when parking a fleet-watched box by hand."""
    info = fleet_operator_intent(iid, "stop")
    if info and info.get("note"):
        print(f">> {info['note']}")
    return info


# moved-from: herdd.fleet_watch_best_effort
def fleet_watch_best_effort(target: object, profile: str = "bare",
                            budget_usd: float | None = None,
                            policy: Mapping[str, Any] | None = None) -> bool:
    """B1b: register a freshly launched box with the daemon so the fleet is
    never in a launch->watch gap. NEVER fatal — a missing daemon just means the
    safety net adopts the box on its own later."""
    ok, _data, err = fleet_request("watch", _timeout=5, _retries=0,
                                   target=str(target), profile=profile,
                                   budget_usd=budget_usd, policy=policy or {},
                                   requester=_fleet_requester())
    if ok:
        print(f">> registered {target} with fleetd (profile={profile})")
    elif not str(err).startswith("nodaemon:"):
        print(f">> note: fleetd registration skipped ({err})", file=sys.stderr)
    return ok


def print_bare_watch_hint(target: object, profile: str = "jobs") -> bool:
    """Say, at the seam that just produced a `bare` watch, that `bare` is not
    supervision — and name the command that makes it one.

    `bare` is observation + alarms + the auto-adopt cost ceiling; the ladder
    (`POLICY_PROFILES`) is what defends a bid, rescues an outbid box and rents
    an eviction replacement. The names do not carry that: `--fleet-watch` reads
    like supervision, and a box registered by it sat undefended through a 44%
    spot-floor rise until a human hand-raised the bid (2026-08-25).

    Deliberately NOT called from `fleet_watch_best_effort`. Three of its four
    callers must not print this: workflowctl registers `bare` so fleetd never
    fights its in-process supervisor, and a train box's upgrade is `--profile
    run`, not `jobs`. Which advice is right is the CALLER's knowledge, so the
    caller asks for it and names the profile.

    `HERDD_WATCH_HINT=0` suppresses it — `launch_jobs_box.sh` sets that
    because it arms the ladder itself two steps later. Returns whether it
    printed, so a test can pin the suppression."""
    if os.environ.get("HERDD_WATCH_HINT") == "0":
        return False
    standing = " --standing" if profile == "jobs" else ""
    after = ("AFTER the tickets exist (a jobs watch parks a box whose every "
             "ticket is terminal; an existing STANDING one wakes on the ticket "
             "itself)" if profile == "jobs" else "next")
    print(">> `bare` is observation + alarms ONLY — no bid defense, no outbid "
          "rescue, no eviction replacement.")
    print(f">>   Arm the spend-capable ladder {after}:")
    print(f">>     {os.path.basename(sys.argv[0]) or 'herdd.py'} fleet watch "
          f"{target} --profile {profile} --budget <USD>{standing}")
    return True


def print_jobs_ticket_hint(target: object) -> bool:
    """The supervision advice for a command that just placed a LIVE ticket on
    `target` — the bare-watch hint, but only where it is true.

    `job retarget` and `job requeue` printed "arm the spend-capable ladder AFTER
    the tickets exist" unconditionally, including onto a box already carrying a
    standing `jobs` watch that this very ticket just woke. That is advice to
    re-arm what is armed, and re-arming is not free: `fleet watch` states the
    whole watch, so a hand re-registration is how a cap gets granted twice.

    An unreadable fleetd state still prints — a missing daemon is the normal
    shape on a fresh box and the hint is exactly right there."""
    try:
        level, _d = fleet_watch_supervision(target)
    except Exception:
        level = "unknown"
    if level == "policy":
        return False
    return print_bare_watch_hint(target, "jobs")


# moved-from: herdd.fleet_follow
def fleet_follow(target: object, iid: object | None = None,
                 timeout: float | None = None) -> int:
    """S7: mirror the legacy foreground-blocking contract — tail the journal for
    this watch until it finishes, and return the exit code the inline loop would
    have used (3 == the jobs lane's unrecoverable box)."""
    path = fleet_journal_path()
    t0 = time.time()
    pos = os.path.getsize(path) if os.path.exists(path) else 0
    while timeout is None or time.time() - t0 < timeout:
        if not os.path.exists(path):
            time.sleep(2.0)
            continue
        with open(path) as f:
            f.seek(pos)
            for line in f:
                pos = f.tell()
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if str(rec.get("target") or "") != str(target) and \
                        str(rec.get("iid") or "") != str(iid or target):
                    continue
                print(f"{rec.get('ts_iso', '')} {rec.get('event')}: "
                      + json.dumps({k: v for k, v in rec.items()
                                    if k not in ("ts", "ts_iso", "event")},
                                   sort_keys=True))
                if rec.get("event") == "watch_finished":
                    return 3 if rec.get("verdict") == "unrecoverable" else 0
                if rec.get("event") == "watch_removed":
                    return 0
        time.sleep(2.0)
    return 0


# moved-from: herdd.fleet_delegate_supervise
def fleet_delegate_supervise(a: argparse.Namespace, run_id: str) -> bool:
    """`herdd supervise` -> `fleet watch run:<RUN> --profile run`."""
    return _fleet_delegate(a, f"run:{run_id}", "run", getattr(a, "budget", None))


# moved-from: herdd.fleet_delegate_job_supervise
def fleet_delegate_job_supervise(a: argparse.Namespace) -> bool:
    """`herdd job supervise` -> `fleet watch <IID> --profile jobs`."""
    return _fleet_delegate(a, str(a.id), "jobs", getattr(a, "budget", None))


# moved-from: herdd._fleet_call_or_die
def _fleet_call_or_die(op: str, **args: Any) -> Any:  # noqa: ANN401 — daemon payload
    """The CLI-side strict caller: the payload, or a distinct exit per error
    class. The `nodaemon` text embeds the resolved socket path and the install
    hint, and is asserted verbatim by the fleet CLI tests."""
    ok, data, err = fleet_request(op, **args)
    if ok:
        return data
    if str(err).startswith("refused:"):
        sys.exit(f"error: fleet {op} refused: {str(err)[8:]}")
    if str(err).startswith("nodaemon:"):
        sys.exit(f"error: fleetd is not running (socket {fleet_sock_path()}).\n"
                 f"  start it: {os.path.basename(sys.argv[0])} fleet install\n"
                 f"  (or, for a foreground/dry soak: "
                 f"FLEETD_DRY_RUN=1 python3 tools/vast/fleetd.py serve)")
    sys.exit(f"error: fleet {op} failed: {err}")


# --------------------------------------------------------------------------- #
# the two dashboard sections that could not live in storage/
#
# `storage.dashcache` owns the sqlite schema, the section registry and the
# scrubbers; these two writers sit here because their reads point UP out of that
# ring — `_dash_write_fleet` opens the daemon socket, `_dash_offer_query` builds
# a market query — and `storage` is BELOW both in the plan §5 DAG. They arrive
# through `DashDeps.write_fleet` / `DashDeps.offer_query` as injected callables,
# so no import edge is created in either direction: dashcache dispatches by
# name, and the composition root (plan §8 step 6) is what binds these two in.
# --------------------------------------------------------------------------- #
# moved-from: herdd._dash_write_fleet
def _dash_write_fleet(conn: sqlite3.Connection, *,
                      deps: dashcache.DashDeps | None = None
                      ) -> tuple[int, str | None]:
    """fleetd `status` + `spend` -- READ OPS ONLY.

    `fleetd.sock` accepts `destroy` and `pause` on the same transport as
    `status`, so the op strings here are frozen literals and this is the only
    place in the dashboard's supply chain that opens the socket at all.

    A DOWN daemon writes `daemon_up=0` with every other column NULL and DELETES
    the watch/alarm/spend rows: "unknown" must not be served as a stale watch
    table that claims boxes are being babysat when nothing is.

    `requester` (user@hostname on every watch row and alarm intent) is never
    selected -- it is PII on a page with no authentication.

    A watch whose target has not resolved to a box yet has `iid = NULL`; sqlite
    autoassigns it a small rowid, which cannot collide with a real 8-digit vast
    id. `target` is the durable identity for those rows -- read it, not `iid`.

    `deps` is threaded through to `dashcache._dash_scrub` only (the two
    `launch.spec` members that ring cannot import). The section registry calls
    this with `conn` alone, so the composition root binds `deps` when it wires
    the hook.
    """
    ok, data, err = fleet_request("status", _timeout=8, _retries=0)
    if not ok:
        with conn:
            conn.execute("DELETE FROM fleet")
            conn.execute("DELETE FROM fleet_watches")
            conn.execute("DELETE FROM fleet_alarms")
            conn.execute("DELETE FROM fleet_spend")
            conn.execute(
                "INSERT INTO fleet(key,daemon_up,api_ok,dry_run,tick_age_s,"
                "tick_stale,rev,version,spend_total_usd,n_watches,n_strays,"
                "n_alarms,n_sticky_alarms) "
                "VALUES('fleet',0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,"
                "NULL,NULL,NULL)")
            dashcache._dash_meta(conn, "fleet")
        return 0, err

    srows = data.get("rows") or []
    recs = data.get("alarm_records")
    if recs is None:                       # older daemon: message strings only
        recs = [{"key": f"legacy:{n}", "msg": m, "sticky": True}
                for n, m in enumerate(data.get("alarms") or [])]
    tick = data.get("tick_age_s")
    wrows, n_watches, n_strays = [], 0, 0
    for r in srows:
        stray = 1 if (r.get("state") == "UNWATCHED") else 0
        n_strays += stray
        n_watches += 1 - stray
        # Prefer the CEILING's cumulative spend: the watch's own counter reads
        # $0.00 on a box that just inherited a ceiling with most of it drawn,
        # and `budget_frac` off that number is a dashboard that says a box is
        # fine right up to the park. `.get(... ) or` so an older daemon (no
        # ceiling in its payload) still reports what it always did.
        spend = r.get("ceiling_spend_usd")
        if spend is None:
            spend = r.get("spend_usd")
        budget = r.get("budget_usd")
        wrows.append((
            dashcache._dash_int(r.get("iid")),
            str(r.get("target")) if r.get("target") is not None else None,
            None if stray else r.get("profile"), r.get("state"),
            spend, budget,
            (spend / budget) if (isinstance(spend, (int, float))
                                 and isinstance(budget, (int, float))
                                 and budget) else None,
            1 if r.get("paused") else 0, r.get("pause_left_s"),
            dashcache._dash_scrub(r.get("pause_reason"), deps=deps),
            1 if r.get("dormant") else 0, 1 if r.get("adopted") else 0,
            stray, dashcache._dash_scrub(r.get("last_action"), deps=deps),
        ))
    arows = [(str(r.get("key")), dashcache._dash_int(r.get("iid")),
              dashcache._dash_scrub(r.get("msg"), deps=deps),
              1 if r.get("sticky") else 0,
              r.get("since_ts"), r.get("age_s"), r.get("count"))
             for r in recs if r.get("key")]

    ok2, sp, _e2 = fleet_request("spend", _timeout=8, _retries=0)
    by_box = ((sp or {}).get("by_box") or {}) if ok2 else {}
    sprows = [(dashcache._dash_int(k), v) for k, v in sorted(by_box.items())
              if dashcache._dash_int(k) is not None]

    with conn:
        conn.execute("DELETE FROM fleet")
        conn.execute("DELETE FROM fleet_watches")
        conn.execute("DELETE FROM fleet_alarms")
        conn.execute("DELETE FROM fleet_spend")
        conn.execute(
            "INSERT INTO fleet(key,daemon_up,api_ok,dry_run,tick_age_s,"
            "tick_stale,rev,version,spend_total_usd,n_watches,n_strays,"
            "n_alarms,n_sticky_alarms) "
            "VALUES('fleet',1,?,?,?,?,?,?,?,?,?,?,?)",
            (None if data.get("api_ok") is None else
             (1 if data.get("api_ok") else 0),
             None if data.get("dry_run") is None else
             (1 if data.get("dry_run") else 0),
             tick,
             1 if (isinstance(tick, (int, float))
                   and tick > dashcache.DASH_TICK_STALE_S) else 0,
             data.get("rev"), dashcache._dash_int(data.get("version")),
             data.get("spend_total_usd"), n_watches, n_strays,
             len(arows), sum(1 for r in arows if r[3])))
        conn.executemany(
            "INSERT OR REPLACE INTO fleet_watches(iid,target,profile,state,"
            "spend_usd,budget_usd,budget_frac,paused,pause_left_s,pause_reason,"
            "dormant,adopted,stray,last_action) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", wrows)
        conn.executemany(
            "INSERT OR REPLACE INTO fleet_alarms(key,iid,msg,sticky,since_ts,"
            "age_s,count) VALUES(?,?,?,?,?,?,?)", arows)
        conn.executemany(
            "INSERT OR REPLACE INTO fleet_spend(iid,spend_usd) VALUES(?,?)",
            sprows)
        dashcache._dash_meta(conn, "fleet")
    return len(srows), None


# moved-from: herdd._dash_offer_query
def _dash_offer_query(gpu_name: str, num_gpus: int, kind: str) -> dict[str, Any]:
    """`build_search_query` for one market probe -- the SAME query builder the
    launch path uses, so the floor the dashboard prints is the floor a launch
    would actually find. Deliberately PERMISSIVE (rentable + verified only): a
    market survey wants the true floor, and the per-offer `cuda_max_good` /
    `reliability` columns let the UI apply our launch filters client-side."""
    return offers.build_search_query(argparse.Namespace(
        limit=dashcache.DASH_OFFER_LIMIT, type=kind, num_gpus=num_gpus,
        unverified=False, gpu=[gpu_name], gpu_ram=0, max_dph=None,
        host_disk=0, reliability=0, cuda=0, inet_down=0,
        machine=None, exclude_machines=None, host=None, geo=None))


def _dash_census_query() -> dict[str, Any]:
    """`build_search_query` for the GPU-CLASS census -- one page, every class.

    `any_gpu=True` is the point: it bypasses `GPU_DEFAULT_POLICY_TIERS`, whose
    hand-written allowlist is the very thing discovery exists to stop being the
    limit of what the board can show. Otherwise as permissive as the per-probe
    query (rentable + verified, all numeric floors zeroed) -- the bf16 and VRAM
    tests are applied by `_dash_discover_gpus` against the per-offer fields, so
    a class rejected there can say WHY."""
    return offers.build_search_query(argparse.Namespace(
        # `num_gpus=1` is the "any width" spelling: the builder emits
        # `num_gpus >= N`, and None is rejected by the API outright.
        limit=dashcache.DASH_DISCOVER_LIMIT, type="ondemand", num_gpus=1,
        unverified=False, gpu=None, any_gpu=True, gpu_ram=0, max_dph=None,
        host_disk=0, reliability=0, cuda=0, inet_down=0,
        machine=None, exclude_machines=None, host=None, geo=None))


# --------------------------------------------------------------------------- #
# The OTHER delegation shape: spawn a child `herdd supervise` process
# --------------------------------------------------------------------------- #
# `fleet_delegate_supervise` above hands a run to the DAEMON; `_supervise_argv`
# builds the argv for the pre-fleetd path, where `train --supervise` forks a
# child `herdd supervise` itself. Two spellings of "someone else babysits this
# run", so they belong in one module — `sup-run-lane.json` says the same from
# the other side ("`_supervise_argv` … builds the child-process argv for a
# delegated supervise — cli/ or fleet/client, not run_lane"), and `run_lane.py`
# declined it so the thin CLI driver stays a driver.
#
# `os.path.abspath(__file__)` IN THE FLAT MODULE WAS `tools/vast/herdd.py` —
# the script the child re-executes. Copied verbatim here it becomes
# `.../vastlib/fleet/client.py`, and the child would be launched as
# `python3 .../client.py supervise <run> …`: no `main()`, no parser, instant
# exit 0-or-1 with the run unsupervised and NOTHING said about it, because the
# parent only checks that Popen succeeded. Hence the module constant below,
# recomputed from `_TOOLS_DIR`'s depth and pinned by
# `test_vastlib_cli_helpers.py::test_supervise_argv_reexecutes_herdd`.
_HERDD_SCRIPT = os.path.join(_TOOLS_DIR, "vast", "herdd.py")


# moved-from: herdd._supervise_argv
def _supervise_argv(a: argparse.Namespace, run: str, budget: object,
                    max_bid: object, defend_at: object,
                    rescue_wait: object) -> list[str]:
    """Build the child `herdd supervise` argv from a `train --supervise` handoff
    (extracted so the flag passthrough is unit-testable without the launch path).
    The spot ceiling flags (--strict-ceiling/--handoff) and the tuning defaults are
    threaded through. F7: train's --wall-budget is in HOURS (operator-friendly); the
    child supervise's --wall-budget is in SECONDS, and without it the child defaults
    to 48h — feeding _handoff_candidate_ok's amortization inequality a fake horizon
    on a short run. Convert HOURS -> SECONDS here."""
    argv = [sys.executable, _HERDD_SCRIPT,
            "supervise", run, "--budget", str(budget)]
    if max_bid is not None:
        argv += ["--max-bid", str(max_bid)]
    # over-ceiling policy: strict-terminate | get-and-hold-only | handoff (DEFAULT).
    # Forward explicitly so the child never depends on a matching default.
    if getattr(a, "strict_ceiling", False):
        argv += ["--strict-ceiling"]
    elif not getattr(a, "handoff", True):
        argv += ["--no-handoff"]
    else:
        argv += ["--handoff"]                        # get-and-hold + migrate (default)
    if defend_at is not None:
        argv += ["--defend-at", str(defend_at)]
    if rescue_wait is not None:
        argv += ["--rescue-wait", str(rescue_wait)]
    if getattr(a, "wall_budget", None) is not None:
        argv += ["--wall-budget", str(a.wall_budget * 3600.0)]
    if getattr(a, "boot_health", False):
        argv += ["--boot-health"]                    # forward the opt-in watchdog
    if not getattr(a, "boot_sla", True):
        argv += ["--no-boot-sla"]                    # forward the SLA opt-out
    return argv
