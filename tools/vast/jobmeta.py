#!/usr/bin/env python3
"""jobmeta.py — generic B2-mediated job submission for vast boxes (JOBS_DESIGN.md).

An agent preps a folder (+ a `job-config.yaml`), `herdd job submit` bundles it
content-addressed to Backblaze B2, and a per-box `onstart/jobd.sh` daemon picks it
up, runs the entrypoint, and pushes results back. This module is the pure core +
injectable transport for that system — the CLI (`herdd job …`) and the box-side
`jobd.py` are thin wrappers over it.

It is a SIBLING of `runmeta.py`, not a copy: it reuses runmeta's *pure primitives*
(`now_ts`, `nonce`, `_actor_slug`, `event_key`, `_num`, the injectable rclone
`_default_runner`, `_bucket`) so there is ONE implementation of the clock/identity/
transport discipline, and adds the job-specific pieces that genuinely differ:

  * a different id field (`job_id`, not `run_id`) and a job-specific `_coerce`;
  * a different, frozen event set + terminal lattice + the `lost` derivation;
  * deterministic tar+zstd content-addressed bundling.

Design invariants inherited from runmeta (see its module docstring):
  * B2 has no CAS but is read-after-write consistent -> append-only immutable
    events, one object per event (`jobs/<JOB_ID>/events/<ts>-<actor>-<nonce>.json`),
    state is a PURE fold. Filename order is NOT a status oracle.
  * Liveness ("is the box that claimed this alive NOW?") is injected from vast,
    never inferred from event recency — a claimed/started job whose box IID is not
    live folds to `lost`.
  * frozen schema: objects are immutable, so any format shipped lives forever; the
    fold tolerates unknown events + unknown fields.

Module boundary (portable-test lane): `fold_events`, the validators, and the
deterministic bundler are PURE (no I/O / net). Transport is injectable: every B2
op takes a `runner` callable with runmeta's contract
`runner(args, input=None) -> (rc, stdout, stderr)`; tests pass an in-memory fake.
"""
from __future__ import annotations

import contextlib
import datetime
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

# Reuse runmeta's pure primitives — ONE copy of the clock/identity/transport
# discipline. runmeta imports nothing that shells out and never sys.exits.
_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)
import runmeta  # noqa: E402
from runmeta import (  # noqa: E402
    now_ts, ts_succ, nonce, _actor_slug, event_key, _num, _default_runner, _bucket,
)
# The `defend:` vocabulary is OWNED by bidpolicy (it is a bid-ladder input, not
# a jobs concept) and imported here so the submit-time validator and the ladder
# can never drift apart. bidpolicy is pure and imports neither jobmeta nor
# herdd, so this direction is acyclic.
from bidpolicy import DEFEND_CHEAP, DEFEND_DEAR, DEFEND_MODES  # noqa: E402,F401

# --- frozen schema (v1) ------------------------------------------------------
SCHEMA_VERSION = 1

# The job lifecycle event set. Frozen: objects are immutable so any event name
# shipped lives forever; the fold tolerates unknown events (readers stay tolerant).
EVENTS = frozenset({
    "submitted",          # CLI wrote the ticket (actor=cli:<host>)
    "claimed",            # jobd on the target box picked the ticket up
    "started",            # entrypoint launched (one per attempt — count = attempts)
    "heartbeat",          # ~60s liveness + log tail while running
    "checkpoint",         # mid-run checkpoint globs synced to checkpoints/ (crash safety)
    "preempted",          # daemon caught SIGTERM/SIGINT with this job running (non-terminal)
    "preempt_save",       # outcome of the preempt-forced local checkpoint attempt
                          #   (non-terminal; `result` field: complete | timeout |
                          #   no_piddir | no_live_pid | no_ckptdir | disabled).
                          #   ALWAYS emitted, including on every SKIP — the whole
                          #   point is that a safety net which quietly does nothing
                          #   is indistinguishable from one that works. `no_piddir`
                          #   means the trainer never armed the SIGUSR1 handler.
    "resumed",            # jobd picked the job back up after an interruption (park/
                          #   preempt/daemon death); precedes the attempt's `started`
    "results_uploaded",   # result globs landed on B2 (before the DONE marker)
    "retargeted",         # CLI moved the ticket to another box's queue (old ticket
                          #   deleted — the ONE delete in the system; see delete_ticket)
    "cancelled",          # operator `herdd job cancel`: a TERMINAL, NON-resumable
                          #   stop (distinct from `interrupted`, which resumes) — the
                          #   queue ticket is deleted and a CANCEL marker (see
                          #   write_cancel_marker) tells a running box's jobd to kill
                          #   the process tree. NEVER revived on a later box resume.
    "done",               # entrypoint exited rc==0
    "failed",             # entrypoint rc!=0, timeout, unmet `needs`, or restart cap
})
# TERMINAL is sticky and wins in the fold. `cancelled` is terminal too — an
# operator override — but a genuine `done`/`failed` (the entrypoint actually
# reached an outcome, results on B2) beats a late `cancelled` in the precedence
# below, so cancelling a job that finished a beat earlier reports the real result.
TERMINAL = frozenset({"done", "failed", "cancelled"})
_CORE_KEYS = ("ts", "event", "job_id")     # required for a valid event

# Ticket field stamped by `requeue_ticket`: the operator's explicit "run this
# again" on a job that already went terminal. jobd.sh reads the SAME key name off
# the ticket to override its results.DONE.json skip — keep the two in step.
REQUEUE_TICKET_MARK = "requeued_ts"

# `herdd job wait --until` target states: the folded statuses (TERMINAL),
# the display_status values (running/interrupted/queued), and the meta-state
# "terminal" (= any TERMINAL). Kept here (the pure core) so the CLI + tests
# share one source of truth.
WAIT_STATES = ("terminal", "done", "failed", "cancelled",
               "running", "interrupted", "queued")


def wait_decision(view, want):
    """Pure poll-decision for `herdd job wait`. Returns one of:
      'match'       — the job reached the requested state (stop, success)
      'unreachable' — the job is already TERMINAL and can never reach `want`
                      (e.g. --until done on a job that FAILED) -> stop, error
      'pending'     — keep polling
    `view` is a jobmeta.read_job() fold; `want` is one of WAIT_STATES."""
    st = view.get("status")
    disp = view.get("display_status")
    if want == "terminal":
        return "match" if st in TERMINAL else "pending"
    if want == st or want == disp:
        return "match"
    if st in TERMINAL:            # ended, but not as asked -> will never get there
        return "unreachable"
    return "pending"

# --- box-lifecycle event set (SEPARATE stream, keyed on instance_id) ----------
# jobd writes these to jobs/nodes/<IID>/events/ — a PER-BOX log, NOT the per-job
# event stream (job TERMINAL/EVENTS above stay frozen + untouched, so an old CLI
# folding a job ignores box events entirely and a new CLI folding an old job's
# log finds none — both directions are backward compatible). They record what
# the daemon did to the BOX, not to any one job:
#   jobd_up      daemon booted (optional; jobd does not currently emit it)
#   drained      queue drained but the box has no self-control key — the laptop
#                (job supervise / an operator) must park it (non-terminal net)
#   parked_self  jobd self-parked the box after the idle grace (v2.1). reason=
#                drained (had jobs, queue emptied) | no_job (none ever arrived).
# jobd ALSO writes purely observational events to this stream — `scratch_probe`
# (fs/RAM/tmpfs at boot), `asset_throughput`, and `gemm_probe` (the boot
# dense-bf16 GEMM ceiling, tools/vast/gemm_probe.py). They are deliberately NOT
# in BOX_EVENTS: that set names the LIFECYCLE states a fold reasons about, and a
# telemetry row must never be able to move a box's state.
BOX_EVENTS = frozenset({"jobd_up", "drained", "parked_self"})
BOX_TERMINAL = frozenset({"parked_self"})
_BOX_CORE_KEYS = ("ts", "event", "instance_id")

# JOB_ID = <yyyymmddThhmmss>-<slug>-<nonce4>; used raw in object keys + a vast
# label, so validated with the same charset rule runmeta uses for RUN_ID.
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# name -> slug (becomes part of JOB_ID): lowercase alnum + dashes, bounded.
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
# `runmeta.now_ts` shape, fixed-width: YYYYmmddTHHMMSSmmmZ. Only used to tell a
# real timestamp from junk when parsing it back OUT of an event key name.
TS_RE = re.compile(r"^\d{8}T\d{9}Z$")

VENV_CHOICES = ("none", "serve", "eval")
DEFAULT_TIMEOUT_S = 7200
# Resume attempts after interruption (per box). 5, not 2: spot evictions cluster
# and a checkpointing job resumes for near-free, so a low cap kills nearly-done
# runs. One-shot bundles (no `checkpoint_s`) should pin lower — a restart there
# repeats the whole job.
DEFAULT_MAX_RESTARTS = 5
MAX_TIMEOUT_S = 7 * 24 * 3600          # 7 days — anything longer is a `run`, not a job

# --- job directory allowlist (submit-time; CREDENTIAL_LIFECYCLE.md) -----------
# The B2 prefixes a job may declare in its `scope:`/assets. READS span shared
# inputs; WRITES are confined to jobs/ (the box's scoped write key enforces the
# same at the key level). This is a defense-in-depth gate ENFORCED ON THE LAPTOP
# at submit — a job requesting a prefix outside the allowlist is rejected before
# any upload, independent of the box key. Widen reads via B2_JOB_READ_PREFIXES.
DEFAULT_JOB_READ_PREFIXES = (
    "jobs/", "eval-env/", "base-models/", "train-env/", "build-cache/",
    "chainmine-data/", "runsets/", "checkpoints/",
)
JOB_WRITE_PREFIXES = ("jobs/",)        # box-scoped; not user-overridable
# The PUBLISH grant (2026-08-05). A training bundle's publish stage writes the
# named adapter to checkpoints/<RUN_NAME>/, which shares no prefix with jobs/ —
# and a B2 application key carries exactly ONE namePrefix. So a jobs box gets a
# SECOND prefix-scoped write key (`b2p`, namePrefix=checkpoints/) rather than a
# widened one; see B2_BOX_GRANTS and docs/plans/witness/g2_push/
# B2_PUBLISH_KEY_SCOPE_FIX_2026-08-05.md.
JOB_PUBLISH_PREFIXES = ("checkpoints/",)
SCOPE_WRITE_PREFIXES = JOB_WRITE_PREFIXES + JOB_PUBLISH_PREFIXES

# What the launcher actually mints for a jobs box, keyed by the rclone REMOTE the
# box config names (b2_sync.sh / onstart/jobd_boot.sh write these sections).
# Value = the write prefixes that remote's key may create objects under; an empty
# tuple means the remote holds a READ-ONLY key (writeFiles absent -> 403). This
# table is the grant model the submit-time write-scope preflight checks against,
# and herdd._ship_b2_env mints from it, so the two cannot drift.
B2_BOX_GRANTS = {
    "b2":   (),                     # bucket-wide READ key (listFiles,readFiles)
    "b2eu": (),                     # EU read replica — read-only by construction
    "b2w":  JOB_WRITE_PREFIXES,     # namePrefix=jobs/
    "b2p":  JOB_PUBLISH_PREFIXES,   # namePrefix=checkpoints/
}
BUNDLE_WARN_BYTES = 1 << 30            # 1 GiB — warn (stage big inputs separately)

# zstd frame magic (little-endian 0xFD2FB528) — decompress auto-detects a raw tar.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class JobmetaError(ValueError):
    pass


class QueueUnreadable(RuntimeError):
    """A queue LISTING failed — which is not the same fact as "the queue is empty".

    Deliberately NOT a JobmetaError: a dozen CLI paths catch that and turn it
    into a tidy `sys.exit`, which would re-hide the transport failure this type
    exists to expose. Every `except Exception` guard around a listing still
    catches it, and those guards already answer "unknown", which is correct.
    """


# --- id / slug / validation --------------------------------------------------
def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise JobmetaError(
            f"invalid JOB_ID {job_id!r}: must match {JOB_ID_RE.pattern}")
    return job_id


def slugify(name: str) -> str:
    """Coerce a job `name` into a JOB_ID-safe slug (lowercase, dashes). Raises if
    nothing usable survives."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40]
    s = s.strip("-")
    if not s or not SLUG_RE.match(s):
        raise JobmetaError(
            f"invalid job name {name!r}: slug must match {SLUG_RE.pattern} "
            f"(lowercase letters/digits/dashes, <=40 chars)")
    return s


def job_ts() -> str:
    """Compact UTC stamp for the JOB_ID prefix: YYYYMMDDTHHMMSS (no millis/Z —
    that is the event-key clock; this is a human-facing id prefix)."""
    return now_ts()[:15]          # now_ts() == YYYYMMDDTHHMMSSmmmZ; take the head


def mint_job_id(name: str, *, ts: str | None = None, nonce4: str | None = None) -> str:
    slug = slugify(name)
    ts = ts or job_ts()
    n4 = nonce4 or os.urandom(2).hex()
    return validate_job_id(f"{ts}-{slug}-{n4}")


# --- job-config (schema v1 — frozen; readers tolerate unknown fields) --------
def load_job_config(bundle_dir: str) -> dict:
    """Read the job config from a bundle folder. Prefers job-config.json (already
    canonical), else parses job-config.yaml. Raises if neither exists."""
    j = os.path.join(bundle_dir, "job-config.json")
    y = os.path.join(bundle_dir, "job-config.yaml")
    if os.path.isfile(j):
        with open(j) as fh:
            data = json.load(fh)
    elif os.path.isfile(y):
        with open(y) as fh:
            data = _parse_job_yaml(fh.read())
    else:
        raise JobmetaError(
            f"{bundle_dir}: no job-config.yaml (or job-config.json) found")
    if not isinstance(data, dict):
        raise JobmetaError("job-config must be a mapping at top level")
    return data


def _parse_job_yaml(text: str) -> dict:
    """Parse the job-config schema. Uses PyYAML when installed (herdd stays
    stdlib-only by design), else a targeted fallback for exactly this schema:
    top-level `key: value` scalars, a one-level nested map under `env:`/`needs:`
    (2-space indent), and a `- item` list under `results:`. Nested structures the
    flat herdd fallback cannot do, so this parser is job-config-specific."""
    try:
        import yaml  # optional real parser
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        pass

    out: dict = {}
    cur_key = None          # the key currently accumulating a nested block
    cur_kind = None         # "map" | "list" | "maplist"
    item_indent = -1        # dash indent of the open list-of-mappings item
    _KEY_RE = re.compile(r"^[A-Za-z_][\w.-]*:(\s|$)")
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0:
            cur_key = cur_kind = None
            if line.startswith("- ") or ":" not in line:
                raise JobmetaError(f"job-config.yaml: cannot parse line {raw!r}")
            k, v = line.split(":", 1)
            k, v = k.strip(), _strip_inline_comment(v)
            if v == "":
                out[k] = None            # header of a nested block; type TBD below
                cur_key = k
            else:
                out[k] = _yaml_scalar(v)
        else:                            # nested (belongs to cur_key)
            if cur_key is None:
                raise JobmetaError(f"job-config.yaml: unexpected indent {raw!r}")
            if line.startswith("- "):
                body = _strip_inline_comment(line[2:])
                if not isinstance(out.get(cur_key), list):
                    out[cur_key] = []
                if _KEY_RE.match(body):
                    # `- key: val` opens a list-of-mappings item (the `assets`
                    # schema); continuation keys arrive on deeper-indent lines.
                    k2, v2 = body.split(":", 1)
                    out[cur_key].append({k2.strip():
                                         _yaml_scalar(_strip_inline_comment(v2))})
                    cur_kind, item_indent = "maplist", indent
                else:
                    out[cur_key].append(_yaml_scalar(body))
                    cur_kind = "list"
            elif ":" in line:
                if (cur_kind == "maplist" and indent > item_indent
                        and isinstance(out.get(cur_key), list)
                        and out[cur_key] and isinstance(out[cur_key][-1], dict)):
                    k2, v2 = line.split(":", 1)
                    out[cur_key][-1][k2.strip()] = _yaml_scalar(
                        _strip_inline_comment(v2))
                else:
                    if not isinstance(out.get(cur_key), dict):
                        out[cur_key] = {}
                    k, v = line.split(":", 1)
                    out[cur_key][k.strip()] = _yaml_scalar(_strip_inline_comment(v))
            else:
                raise JobmetaError(f"job-config.yaml: cannot parse line {raw!r}")
    return out


def _split_flow(s: str) -> list:
    """Split a flow-list body on commas outside quotes."""
    parts, buf, q = [], [], None
    for ch in s:
        if q:
            buf.append(ch)
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _strip_inline_comment(v: str) -> str:
    """YAML drops `<whitespace>#...` trailing an UNQUOTED scalar; the fallback
    parser didn't, so `gpu_ram_gb: 24  # why` read as the string '24  # why'
    and crash-looped a live workflow controller on a no-PyYAML interpreter for
    6h (2026-07-30, run 2ed9: exit 5 -> systemd restart -> re-adopt -> exit 5,
    ~150 cycles, box never claimed its job). Quoted values keep their '#'."""
    v = v.strip()
    if v and v[0] in "\"'":
        end = v.find(v[0], 1)          # keep through the closing quote, drop
        return v[:end + 1] if end != -1 else v      # any trailing comment
    if v.startswith("#"):
        return ""
    m = re.search(r"\s#", v)
    return v[:m.start()].rstrip() if m else v


def _yaml_scalar(v: str):
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    if len(v) >= 2 and v[0] == "[" and v[-1] == "]":
        inner = v[1:-1].strip()          # flow list, e.g. require: [a, "*.b"]
        return [_yaml_scalar(x) for x in _split_flow(inner)] if inner else []
    if len(v) >= 2 and v[0] == "{" and v[-1] == "}":
        # Flow MAPPING, e.g. `tracks: {trainer.py: tools/<dir>/trainer.py}`.
        # The one map shape the fallback can represent INSIDE a list item (the
        # `assets` schema): a block map nested under a `- key:` item would be
        # mis-attached by the maplist branch above, so the flow form is the
        # supported spelling there. PyYAML parses it identically.
        inner = v[1:-1].strip()
        out = {}
        for part in _split_flow(inner):
            if ":" not in part:
                return v                 # not a mapping — hand back the raw text
            k, val = part.split(":", 1)
            out[str(_yaml_scalar(k))] = _yaml_scalar(val)
        return out
    low = v.lower()
    if low in ("null", "~", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if re.match(r"^-?\d+$", v):
        return int(v)
    return v


def _normalize_tracks(spec, where: str) -> dict:
    """Validate + normalize a `tracks:` mapping — the PROVENANCE declaration the
    staged-asset staleness preflight needs.

    `tracks` answers the one question a generic "is B2 stale?" check cannot:
    *what should this staged object contain?* B2 is the source of truth for
    corpora with no repo counterpart, so staleness is only decidable where the
    operator says "this remote object MIRRORS this repo file". Shape:

        {<remote path>: <repo-relative source path>}

    Both sides are relative and may not escape ('..', leading '/'). Under an
    `assets:` entry the remote path is relative to that asset's `b2:` prefix;
    at top level it is a full B2 key (which covers a job whose entrypoint pulls
    from B2 itself, with no `assets:` entry to hang the declaration on).
    Absent/empty => the preflight is a NO-OP for that asset (never a false alarm
    on a genuinely B2-native corpus)."""
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise JobmetaError(f"job-config: {where} must be a mapping "
                           f"{{remote path: repo-relative path}}")
    out = {}
    for k, v in spec.items():
        k, v = str(k).strip(), ("" if v is None else str(v).strip())
        if not k or not v:
            raise JobmetaError(
                f"job-config: {where} entry {k!r}: both the remote path and the "
                f"repo-relative source path are required")
        for label, p in (("remote path", k), ("source path", v)):
            if p.startswith("/") or p.startswith("b2:") or ".." in p.split("/"):
                raise JobmetaError(
                    f"job-config: {where} {label} {p!r} must be relative "
                    f"(no leading '/', no 'b2:' remote, no '..')")
        out[k.strip("/")] = v
    return out


#: `${NAME}` — the ONLY interpolation `assets[].b2` understands. Deliberately
#: not shell: no `$BARE`, no `${X:-default}`, no nesting. A default would let a
#: forgotten pin resolve to something plausible, which is the failure this is
#: meant to make impossible.
ASSET_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
#: Anything `${…}`-shaped, so a MALFORMED placeholder (`${lower}`, `${A-B}`)
#: is refused by name instead of shipping to a box as a literal prefix.
_ASSET_VAR_ANY_RE = re.compile(r"\$\{[^}]*\}")


def resolve_asset_vars(template: str, env, *, where: str) -> str:
    """Substitute `${VAR}` in an `assets[].b2` template from the SUBMIT env map.

    Submit-side only, and that is the whole design: the bundle tar carries the
    TEMPLATE, so `assets[].b2` stops pinning one artifact into the bundle hash,
    while the ticket jobd reads carries the RESOLVED prefix and nothing on the
    box changes. See tools/vast/ASSET_PARAMETERIZATION.md.

    Fail-closed, three ways — an unresolvable prefix must never reach a rented
    box as a plausible-looking string:
      * a variable absent from `env`               -> refuse, naming it
      * a variable whose value is EMPTY            -> refuse, naming it (the
        house convention for "required at submit" is `KEY: ""` in `env:`)
      * a `${…}` left over after substitution      -> refuse (malformed name)
    Pure; `env` is any mapping."""
    env = env or {}
    missing = []
    for name in ASSET_VAR_RE.findall(template):
        if not str(env.get(name) or ""):
            missing.append(name)
    if missing:
        raise JobmetaError(
            f"job-config: {where} {template!r} has no value for "
            f"{', '.join('${%s}' % m for m in sorted(set(missing)))} — a "
            f"parameterized asset prefix is resolved AT SUBMIT from the job's "
            f"`env:` block plus `--env K=V` (or `--artifact PREFIX=<slug>`, "
            f"which composes it from the modelkit registry). Pass a value; an "
            f"empty one is not one.")
    out = ASSET_VAR_RE.sub(lambda m: str(env[m.group(1)]), template)
    leftover = _ASSET_VAR_ANY_RE.search(out)
    if leftover:
        raise JobmetaError(
            f"job-config: {where} {template!r} carries a malformed placeholder "
            f"{leftover.group(0)!r} — only ${{NAME}} with NAME matching "
            f"[A-Z][A-Z0-9_]* is interpolated, and an uninterpolated one would "
            f"ship to the box as a literal B2 prefix")
    return out


def _normalize_receipt(spec, where: str) -> str:
    """Validate + normalize an asset's OPTIONAL `receipt:` — the name of the
    publisher's completeness marker INSIDE that asset's `b2:` prefix.

    WHY THIS IS NOT jobd's `.complete`. The two markers answer different
    questions and neither substitutes for the other. `.complete` is written by
    the BOX, holds a local byte total, and answers "did I already land these
    bytes?" — it cannot see a prefix that was never fully published, because a
    truncated publish pulls cleanly and the total it records is the truncation's.
    A `receipt` is written by the PUBLISHER, LAST, after the payload lands
    (tools/witness/jobs/*/b2_transport.sh), so its presence answers "is the
    remote prefix whole?" — the one question a restore that races a push has to
    ask BEFORE paying for 52 GiB.

    Shape: a relative path under the prefix (usually a bare filename). Not a
    glob, not absolute, no '..', and no tab/newline — `onstart/jobd.py` ships
    the asset spec to jobd.sh as TSV, where either would silently split a
    field into two. Empty/absent => the whole receipt mechanism is a NO-OP for
    that asset (every legacy declaration keeps its exact behaviour)."""
    if spec is None:
        return ""
    if not isinstance(spec, str):
        raise JobmetaError(f"job-config: {where} must be a string filename "
                           f"(got {spec!r})")
    r = spec.strip().strip("/")
    if not r:
        return ""
    if spec.startswith("b2:") or ".." in r.split("/"):
        raise JobmetaError(
            f"job-config: {where} {spec!r} must be a path relative to the "
            f"asset's b2: prefix (no leading '/', no 'b2:' remote, no '..')")
    if any(c in r for c in "\t\n\r"):
        raise JobmetaError(
            f"job-config: {where} {spec!r} must not contain a tab or newline "
            f"(the box-side asset spec is TSV)")
    if any(c in r for c in "*?["):
        raise JobmetaError(
            f"job-config: {where} {spec!r} must name ONE object, not a glob — "
            f"the marker is what makes the publish atomic, so 'any of these' "
            f"is not a completeness claim")
    return r


def _normalize_assets(raw: dict, env=None) -> list[dict] | None:
    """Validate + normalize the OPTIONAL `assets:` list (laptop-side, at submit).
    Each asset stages a big B2-resident input onto the box BEFORE the entrypoint,
    via jobd's ONE shared pull primitive (retries + integrity), replacing the five
    hand-rolled `b2pull()` mechanisms. Returns the canonical list (baked into the
    ticket JSON verbatim — box side reads JSON only, no YAML) or None when absent.

    asset := {
      name:     slug (SLUG_RE); the stable cache key -> /workspace/assets/<name>
      b2:       relative B2 prefix (no leading '/', no 'b2:' remote, no '..').
                May carry `${VAR}` placeholders, resolved HERE from `env` (the
                job's `env:` block after `--env`/`--artifact` folding) so one
                generic bundle can serve every artifact — `resolve_asset_vars`.
                A resolved asset also records `b2_template`; the resolved value
                is what the ticket, the preflights and jobd see.
      dest:     OPTIONAL. Default = the cache path (/workspace/assets/<name>).
                If given: absolute UNDER /workspace, or relative INSIDE the job
                workdir (no '..'). jobd symlinks dest -> cache when they differ.
      mode:     'copy' (default) | 'sync'
      require:  OPTIONAL list of relative globs that MUST match post-pull
      optional: bool (default false) — absence/failure tolerated, job still runs
      receipt:  OPTIONAL relative filename under `b2:` — a COMPLETENESS RECEIPT
                the publisher writes LAST (b2_transport.sh's PUSHED.json). Its
                PRESENCE gates the pull, its `files` count corroborates it, and
                it is excluded from the staged dir. See `_normalize_receipt`.
    }
    Schema stays UNFROZEN (readers tolerate unknown fields); version stays 1.
    """
    assets = raw.get("assets")
    if assets is None:
        return None
    if not isinstance(assets, list):
        raise JobmetaError("job-config: `assets` must be a list of mappings")
    out: list[dict] = []
    seen: set[str] = set()
    for i, a in enumerate(assets):
        if not isinstance(a, dict):
            raise JobmetaError(f"job-config: assets[{i}] must be a mapping")
        name = a.get("name")
        if not name or not SLUG_RE.match(str(name)):
            raise JobmetaError(
                f"job-config: assets[{i}].name must be a slug matching "
                f"{SLUG_RE.pattern} (got {name!r})")
        name = str(name)
        if name in seen:
            raise JobmetaError(f"job-config: duplicate asset name {name!r}")
        seen.add(name)

        b2 = a.get("b2")
        if not b2 or not isinstance(b2, str):
            raise JobmetaError(
                f"job-config: assets[{name!r}].b2 (a relative B2 prefix) is required")
        # Resolve BEFORE the shape checks, so a variable that expands to an
        # absolute path or a '..' escape is refused by the same rules a literal
        # prefix is — an env value is operator input, not a trusted constant.
        template = b2
        b2 = resolve_asset_vars(b2, env, where=f"assets[{name!r}].b2")
        if b2.startswith("/") or b2.startswith("b2:") or ".." in b2.split("/"):
            raise JobmetaError(
                f"job-config: assets[{name!r}].b2 {b2!r} must be a relative prefix "
                f"(no leading '/', no 'b2:' remote, no '..')")
        b2 = b2.strip("/")

        mode = a.get("mode", "copy")
        if mode not in ("copy", "sync"):
            raise JobmetaError(
                f"job-config: assets[{name!r}].mode must be 'copy' or 'sync' (got {mode!r})")

        optional = a.get("optional", False)
        if not isinstance(optional, bool):
            raise JobmetaError(
                f"job-config: assets[{name!r}].optional must be a bool (got {optional!r})")

        # `archive: true` means the box UNPACKS this asset, so for one moment
        # the archive and its expansion are both on disk and the disk estimate
        # must carry that peak (disksize.UNPACK_PEAK_FACTOR). Default False:
        # the overwhelmingly common asset is a B2 prefix that `asset_pull()`
        # rclone-copies into the cache with no second copy at any instant.
        # disksize also infers this from an archive suffix, so the flag is for
        # a tarball whose name does not say so.
        archive = a.get("archive", False)
        if not isinstance(archive, bool):
            raise JobmetaError(
                f"job-config: assets[{name!r}].archive must be a bool (got {archive!r})")

        require = a.get("require") or []
        if isinstance(require, str):
            require = [require]
        if not isinstance(require, list):
            raise JobmetaError(
                f"job-config: assets[{name!r}].require must be a list of globs")
        require = [str(g) for g in require]
        for g in require:
            if g.startswith("/") or ".." in g.split("/"):
                raise JobmetaError(
                    f"job-config: assets[{name!r}].require glob {g!r} must be "
                    f"relative (no leading '/', no '..')")

        norm = {"name": name, "b2": b2, "mode": mode,
                "optional": optional, "archive": archive, "require": require}
        # Provenance for a parameterized asset: the ticket then says both what
        # the bundle declared and what this submit resolved it to. Recorded ONLY
        # when a placeholder fired, so a literal declaration serializes exactly
        # as it always has.
        if template != b2:
            norm["b2_template"] = template
        # Absent `receipt` stays ABSENT from the ticket, not present-and-empty:
        # a legacy declaration must serialize byte-identically to before.
        receipt = _normalize_receipt(a.get("receipt"), f"assets[{name!r}].receipt")
        if receipt:
            norm["receipt"] = receipt
        tracks = _normalize_tracks(a.get("tracks"), f"assets[{name!r}].tracks")
        if tracks:
            norm["tracks"] = tracks
        dest = a.get("dest")
        if dest is not None:
            dest = str(dest)
            if dest.startswith("/"):
                if dest != "/workspace" and not dest.startswith("/workspace/"):
                    raise JobmetaError(
                        f"job-config: assets[{name!r}].dest {dest!r} must be "
                        f"absolute under /workspace or a relative path")
            elif ".." in dest.split("/"):
                raise JobmetaError(
                    f"job-config: assets[{name!r}].dest {dest!r} must not contain '..'")
            norm["dest"] = dest
        out.append(norm)
    return out


def _allowed_read_prefixes() -> tuple[str, ...]:
    """The read-prefix allowlist, overridable via B2_JOB_READ_PREFIXES (comma/
    space-separated). Each entry is normalized to a trailing '/'."""
    ov = os.environ.get("B2_JOB_READ_PREFIXES")
    if ov:
        return tuple(p if p.endswith("/") else p + "/"
                     for p in re.split(r"[,\s]+", ov.strip()) if p)
    return DEFAULT_JOB_READ_PREFIXES


def _under_any(path: str, prefixes) -> bool:
    return any(path == p or path.startswith(p) for p in prefixes)


def _norm_scope_list(v, field: str):
    if v is None:
        return None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        raise JobmetaError(f"job-config: scope.{field} must be a list of prefixes")
    out = []
    for p in v:
        p = str(p)
        if p.startswith("/") or ".." in p.split("/"):
            raise JobmetaError(
                f"job-config: scope.{field} prefix {p!r} must be relative, no '..'")
        out.append(p if p.endswith("/") else p + "/")
    return out


def validate_scope(raw: dict, assets) -> dict:
    """Normalize + policy-check the optional `scope:` block (read/write B2 prefix
    allowlists) at SUBMIT time. Defaults when absent: read = the prefixes the
    declared assets pull from (plus jobs/), write = jobs/.

    Two enforcement tiers (CREDENTIAL_LIFECYCLE.md):
      * WRITES are ALWAYS hard-enforced to the prefixes the box actually holds a
        scoped write key for — jobs/ (b2w) and checkpoints/ (b2p, the publish
        grant). The real security boundary; a scope.write outside SCOPE_WRITE_
        PREFIXES is always rejected, and there is deliberately no way to declare
        a bucket-wide write.
      * READS are checked against the allowlist ONLY under B2_JOB_SCOPE_STRICT
        (opt-in) — the future multi-tenant control. Permissive by default so
        today's cross-prefix reads (e.g. a checkpoints/ adapter) are unaffected.
    Also rejects an asset whose source escapes an EXPLICITLY declared read scope
    (a config inconsistency, always). Returns the normalized {read, write} dict."""
    raw_scope = raw.get("scope")
    if raw_scope is not None and not isinstance(raw_scope, dict):
        raise JobmetaError("job-config: `scope` must be a mapping with read/write")
    explicit_read = bool(raw_scope) and raw_scope.get("read") is not None
    read = _norm_scope_list((raw_scope or {}).get("read"), "read")
    write = _norm_scope_list((raw_scope or {}).get("write"), "write")
    if read is None:                                   # derive from declared assets
        srcs = set()
        for a in (assets or []):
            b2p = a.get("b2") or ""
            if b2p:
                srcs.add(b2p.split("/", 1)[0] + "/" if "/" in b2p else b2p + "/")
        read = sorted(srcs | {"jobs/"}) if srcs else ["jobs/"]
    if write is None:
        write = list(JOB_WRITE_PREFIXES)
    # WRITES: always confined to the prefixes a box key exists for (hard).
    for p in write:
        if not _under_any(p, SCOPE_WRITE_PREFIXES):
            raise JobmetaError(
                f"job-config: scope.write prefix {p!r} must be under one of "
                f"{list(SCOPE_WRITE_PREFIXES)} — a box holds a namePrefix-scoped "
                f"write key for those and nothing else")
    # READS: allowlist enforced only in strict mode (opt-in future control).
    if os.environ.get("B2_JOB_SCOPE_STRICT", "").lower() in ("1", "true", "yes"):
        allow_read = _allowed_read_prefixes()
        for p in read:
            if not _under_any(p, allow_read):
                raise JobmetaError(
                    f"job-config: scope.read prefix {p!r} not in the allowlist "
                    f"{list(allow_read)} (B2_JOB_SCOPE_STRICT) — widen via "
                    f"B2_JOB_READ_PREFIXES")
    # An asset outside an EXPLICITLY declared read scope is a config error (always).
    if explicit_read:
        for a in (assets or []):
            b2p = a.get("b2") or ""
            if b2p and not _under_any(b2p if b2p.endswith("/") else b2p + "/", read):
                raise JobmetaError(
                    f"job-config: asset b2 source {b2p!r} is outside the declared "
                    f"scope.read {read} — add it to scope.read or fix the source")
    return {"read": read, "write": write}


# --- B2 WRITE-SCOPE PREFLIGHT (submit/rehearse-time, $0, no network) ---------- #
# THE INCIDENT THIS CLOSES (docs/plans/witness/g2_push/V7_TRAIN_RUN_2026-08-05.md).
# v7 was the first bundle to ship a PUBLISH stage. Both arms trained to
# completion and then exited rc=15: `run.sh` published with
# `rclone copy … "b2:$B2_BUCKET/checkpoints/$RUN_NAME/"` — the `b2` remote holds
# the bucket-wide READ key, and the only write key on the box was namePrefix-
# scoped to `jobs/`. Every PutObject came back `403 not entitled`, and the
# bundle's own "a publish failure fails the job" contract marked two successful
# training runs failed. NOTHING upstream could see it: the rehearsal is DRY_RUN=1
# and never touches B2 by construction ("PIN, DON'T SIMULATE"), `--dry-run`
# submit doesn't either, and the destination is a shell string, not config.
#
# So this check reads the bundle's own text and answers, statically and for $0:
# for every B2 destination this bundle writes, will the box hold a key entitled
# to write there? Covered -> one confirmation line. NOT covered -> refuse before
# a box is rented. Undecidable (a destination assembled from a variable the
# scanner cannot resolve) -> a `note:` that never blocks — see
# `b2_write_scope_report` for why the three tiers land where they do.
B2_WRITE_VERBS = frozenset((
    "copy", "copyto", "sync", "move", "moveto", "rcat", "mkdir", "touch",
    "delete", "deletefile", "purge", "copyurl", "settier",
))
B2_READ_VERBS = frozenset((
    "cat", "ls", "lsf", "lsl", "lsd", "lsjson", "md5sum", "sha1sum", "hashsum",
    "size", "check", "tree", "about", "listremotes", "version",
))
_B2_SCAN_SUFFIXES = (".sh", ".bash", ".zsh", ".py", ".yaml", ".yml")
_B2_SCAN_MAX_BYTES = 2 << 20           # skip anything bigger; bundles hold code
_B2_REF_RE = re.compile(r"\b(b2[a-z0-9_]*):([^\s\"';|&)]*)")
_B2_VAR_RE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::[-=?+][^}]*)?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_B2_ASSIGN_RE = re.compile(
    r"(?:^|[;&|(]\s*|\s(?:then|else|do)\s+)\s*(?:export\s+|local\s+|declare\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S[^\n;|&]*)", re.M)
# `for f in …` binds a name too. Only the FULL map uses it (to prove a
# destination local), which is what keeps `/workspace/${_f}` out of the notes.
_B2_FOR_RE = re.compile(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^\n;]*)")
_B2_SH_RCLONE_RE = re.compile(
    r"(?:^|[;&|(`]\s*|\s)(?:\$\(\s*)?rclone\s+((?:-{1,2}\S+\s+)*)([a-z][a-z0-9]*)")
# `subprocess.run(["rclone", "copy", …])` and friends — the same command, spelled
# as a Python list. Scanned so a bundle cannot dodge the gate by using subprocess.
_B2_PY_RCLONE_RE = re.compile(
    r"[\"']rclone[\"']\s*,\s*(?:[^)\]\n]{0,120}?)[\"']([a-z][a-z0-9]*)[\"']")


def _b2_unquote(tok: str) -> str:
    tok = tok.strip()
    for pre in ("f", "rf", "fr", "r", "b", "u"):        # python string prefixes
        if tok[:len(pre) + 1].lower() == pre + '"' or \
           tok[:len(pre) + 1].lower() == pre + "'":
            tok = tok[len(pre):]
            break
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        tok = tok[1:-1]
    return tok.strip("\"'")


def _b2_assignments(text: str, *, b2_only: bool = True) -> dict:
    """{VAR: value} for assignments in one file. `b2_only` keeps just the ones
    whose value names a b2* remote — enough for the real shapes:
    `PUB_DEST="b2:$B2_BUCKET/checkpoints/$RUN_NAME/"` and `B2W="b2w:${B2_BUCKET}"`.
    The FULL map is used only to prove a destination is LOCAL (see
    `_b2_scan_text`); resolution always consults the b2 map first, so a variable
    with any b2-bearing assignment can never be explained away as local."""
    out = {}
    for rx in (_B2_ASSIGN_RE, _B2_FOR_RE):
        for m in rx.finditer(text):
            val = _b2_unquote(m.group(2))
            if b2_only and not _B2_REF_RE.search(val):
                continue
            out.setdefault(m.group(1), val)
    return out


def _b2_resolve(tok: str, variables: dict, depth: int = 4) -> str:
    """Substitute known b2-bearing variables into a token (bounded passes)."""
    tok = _b2_unquote(tok)
    for _ in range(depth):
        def _sub(m):
            name = m.group(1) or m.group(2)
            return variables.get(name, m.group(0))
        new = _B2_VAR_RE.sub(_sub, tok)
        if new == tok:
            break
        tok = new
    return tok


def _b2_commands(text: str):
    """(lineno, command) for each logical shell command: backslash continuations
    joined, then split on `;`, `&&`, `||`. Heuristic by design — the scanner's
    failure mode is an UNRESOLVED destination (a note), never a silent pass."""
    joined, lineno, buf, start = [], 0, "", 1
    for raw in text.splitlines():
        lineno += 1
        if not buf:
            start = lineno
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        joined.append((start, buf + stripped))
        buf = ""
    if buf:
        joined.append((start, buf))
    for ln, line in joined:
        for part in re.split(r"&&|\|\||;", line):
            part = part.strip()
            if part:
                yield ln, part


def _b2_dest_token(cmd: str) -> str:
    """The last positional token of a command — rclone's DESTINATION. Options,
    redirections and everything after them are dropped, so `rclone copy SRC DST
    2>/dev/null` yields DST and `rclone copy b2:…/a /local` yields `/local`
    (a DOWNLOAD, correctly not a B2 write)."""
    toks = cmd.split()
    cut = len(toks)
    for i, t in enumerate(toks):
        if t in ("|", ")", "{", "}") or re.match(r"^\d*[<>]|^&>", t):
            cut = i
            break
    toks = [t for t in toks[:cut] if t not in ("\\",)]
    return toks[-1] if toks else ""


def _b2_ref_scope(ref: str):
    """('b2w', 'jobs/') for a resolved `b2w:$B2_BUCKET/jobs/…` reference.
    The FIRST path segment after `remote:` is always the bucket (these are
    `type = s3` remotes), so the grant-bearing prefix is the SECOND. Returns
    (remote, prefix|None); prefix None == could not be decided statically."""
    m = _B2_REF_RE.search(ref)
    if not m:
        return None, None
    remote, rest = m.group(1), m.group(2)
    parts = [p for p in rest.split("/")]
    if len(parts) < 2 or not parts[1]:
        return remote, None                     # bucket root / bucket only
    seg = parts[1]
    if any(c in seg for c in "${}*?"):
        return remote, None                     # unresolved variable segment
    return remote, seg + "/"


def _b2_scan_text(text: str, *, python: bool):
    """Yield (lineno, verb, dest) for every rclone WRITE in one file's text.
    `dest` is resolved through the file's b2-bearing assignments; a destination
    the FULL assignment map proves is a plain local path is dropped (that is a
    DOWNLOAD, not a B2 write), and anything still unresolved is yielded as-is so
    the caller can report it `unknown` rather than pass it silently."""
    b2vars = _b2_assignments(text)
    allvars = _b2_assignments(text, b2_only=False)

    def _emit(lineno, verb, tok):
        dest = _b2_resolve(tok, b2vars)
        if _B2_REF_RE.search(dest):
            return lineno, verb, dest
        # No b2 remote in sight. It is a LOCAL destination — and provably so —
        # when every variable it names is assigned in this same file to a value
        # that never mentions a b2 remote (b2-bearing assignments were resolved
        # above, so they cannot reach here). Otherwise the destination comes from
        # somewhere the scanner cannot see: report it `unknown`, never drop it.
        names = [m.group(1) or m.group(2) for m in _B2_VAR_RE.finditer(dest)]
        if all(n in allvars for n in names):
            return None
        return lineno, verb, dest

    for lineno, cmd in _b2_commands(text):
        m = _B2_SH_RCLONE_RE.search(cmd)
        if not m:
            continue
        verb = m.group(2)
        if verb in B2_READ_VERBS or verb not in B2_WRITE_VERBS:
            continue
        hit = _emit(lineno, verb, _b2_dest_token(cmd))
        if hit:
            yield hit
    if not python:
        return
    for m in _B2_PY_RCLONE_RE.finditer(text):
        verb = m.group(1)
        if verb not in B2_WRITE_VERBS:
            continue
        tail = text[m.end():m.end() + 600]
        refs = list(_B2_REF_RE.finditer(tail))
        lineno = text.count("\n", 0, m.start()) + 1
        hit = _emit(lineno, verb, refs[-1].group(0) if refs else "")
        if hit:
            yield hit


def scan_b2_writes(bundle_dir: str, *, grants=None) -> list[dict]:
    """Every B2 write destination reachable in a bundle's own text, classified
    against the grant table the launcher mints (B2_BOX_GRANTS). PURE: no network,
    no B2, no creds.

    Walks `iter_bundle_files`, not the raw folder: a bundle's SHIPPED text is its
    own files plus its `includes:`, and a gate that walked only the folder would
    go blind the moment a B2-writing script moved into the shared dir.

    Findings: {file, line, verb, remote, prefix, dest, status, detail} with
      status='ok'        the remote's key is entitled to write that prefix;
      status='uncovered' it is NOT — this run would 403 at that line;
      status='unknown'   the destination could not be resolved statically.
    """
    grants = B2_BOX_GRANTS if grants is None else grants
    findings = []
    for rel, path in iter_bundle_files(bundle_dir):
        fn = os.path.basename(rel)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        if not fn.endswith(_B2_SCAN_SUFFIXES):
            try:                                    # extension-less entrypoint
                with open(path, "rb") as fh:
                    if fh.read(2) != b"#!":
                        continue
            except OSError:
                continue
        try:
            if os.path.getsize(path) > _B2_SCAN_MAX_BYTES:
                continue
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for lineno, verb, dest in _b2_scan_text(
                text, python=fn.endswith(".py")):
            remote, prefix = _b2_ref_scope(dest)
            if remote is None:
                if _B2_REF_RE.search(dest or "") or "$" in (dest or ""):
                    findings.append({
                        "file": rel, "line": lineno, "verb": verb,
                        "remote": None, "prefix": None, "dest": dest,
                        "status": "unknown",
                        "detail": "destination is not a resolvable b2 remote"})
                continue                     # a plain local destination
            allowed = grants.get(remote)
            base = {"file": rel, "line": lineno, "verb": verb,
                    "remote": remote, "prefix": prefix, "dest": dest}
            if allowed is None:
                findings.append(dict(base, status="uncovered",
                                     detail=f"no rclone remote {remote!r} is "
                                            f"configured on a jobs box"))
            elif prefix is None:
                findings.append(dict(base, status="unknown",
                                     detail="prefix not statically resolvable"))
            elif not allowed:
                findings.append(dict(base, status="uncovered",
                                     detail=f"remote {remote!r} holds the "
                                            f"bucket-wide READ key (no writeFiles)"))
            elif _under_any(prefix, allowed):
                findings.append(dict(base, status="ok",
                                     detail=f"{remote} key is namePrefix-scoped "
                                            f"to {allowed[0]}"))
            else:
                findings.append(dict(base, status="uncovered",
                                     detail=f"remote {remote!r} is namePrefix-"
                                            f"scoped to {list(allowed)}"))
    return findings


def b2_write_preflight(cfg, bundle_dir: str, *, grants=None) -> list[dict]:
    """THE shared submit-path seam for write-scope (sibling of `asset_preflight`):
    the static scan, plus the consistency check against an EXPLICITLY declared
    `scope.write`. `herdd job submit`, `jobmatrix submit` and `rehearse.sh` all
    call exactly this so no surface can drift in what it checks. Pure."""
    findings = scan_b2_writes(bundle_dir, grants=grants)
    raw_scope = (cfg or {}).get("scope") or {}
    declared = raw_scope.get("write") if isinstance(raw_scope, dict) else None
    if declared:
        declared = [p if str(p).endswith("/") else str(p) + "/" for p in declared]
        for f in findings:
            if f["status"] == "ok" and not _under_any(f["prefix"], declared):
                f["status"] = "undeclared"
                f["detail"] = (f"writes {f['prefix']} but scope.write declares "
                               f"{declared}")
    return findings


def b2_write_scope_report(findings, *, allow_unscoped=False):
    """(lines, refuse) for the write-scope preflight. Presentation policy:

      * 'uncovered' -> LOUD and refuse. The box will 403 at that line, which on
        a publish stage means a FULLY TRAINED arm is marked failed (the v7
        incident). It costs nothing to catch here and ~$3 to catch on a box.
        `--allow-unscoped-writes` is the deliberate opt-out for the one honest
        case: a single-key box (no minter configured), where `b2` IS bucket-wide
        read-write and the scoped grants do not apply.
      * 'undeclared' -> LOUD and refuse: the bundle writes a prefix its own
        `scope:` block says it does not. One of the two is wrong.
      * 'unknown' -> a `note:` line; NEVER refuses. A regex cannot resolve every
        runtime-assembled destination, and a static check that blocks on its own
        blind spots gets switched off — which is worse than a note.
      * 'ok' -> ONE confirmation line per distinct (remote, prefix). Deliberately
        not silent: a silent pass is indistinguishable from a check that never
        ran, and "the publish stage was never exercised" is exactly how v7 got
        here. Bundles write two or three distinct prefixes, so this is not noise.
    Pure. The caller prints the lines and refuses when `refuse` is True."""
    lines, refuse, seen = [], False, set()
    for f in (findings or []):
        st = f.get("status")
        where = f"{f.get('file')}:{f.get('line')}"
        if st in ("uncovered", "undeclared"):
            refuse = refuse or not allow_unscoped
            lines.append(
                f"!! B2 WRITE NOT ENTITLED: {where} `rclone {f.get('verb')}` -> "
                f"{f.get('dest')} — {f.get('detail')}. The box would get "
                f"403 (not entitled) AFTER doing the work. Route this write "
                f"through a remote that holds the prefix "
                f"({', '.join(r for r, p in sorted(B2_BOX_GRANTS.items()) if p)}), "
                f"or move the destination under a granted prefix.")
        elif st == "unknown":
            lines.append(
                f"note: B2 write scope UNVERIFIED at {where} "
                f"(`rclone {f.get('verb')}` -> {f.get('dest') or '?'}) — "
                f"{f.get('detail')}; proceeding without the check for this line.")
        elif st == "ok":
            key = (f.get("remote"), f.get("prefix"))
            if key not in seen:
                seen.add(key)
                lines.append(
                    f">> B2 write scope OK: {f.get('remote')}:…/{f.get('prefix')} "
                    f"({f.get('detail')})")
    if refuse:
        lines.append("   (grants: " + ", ".join(
            f"{r}={'/'.join(p) if p else 'READ-ONLY'}"
            for r, p in sorted(B2_BOX_GRANTS.items()))
            + "; --allow-unscoped-writes overrides on a single-key box.)")
    return lines, refuse


# --- `fleet watch --profile jobs` ordering guard ----------------------------- #
def jobs_watch_advice(job_ids, views) -> str | None:
    """Warn BEFORE arming a `jobs` watch whose first tick has nothing to do.

    THE INCIDENT (box 46648873, 2026-08-03). The jobs ladder's drain exit
    (`herdd.job_supervise_tick`) is "every ticket for this box is terminal =>
    the work is finished => park it". That is right at the END of a session and
    wrong at the START of the next one: a box RESUMED to run more work still
    carries yesterday's DONE tickets in its queue, so a `jobs` watch armed
    before the new job is submitted sees an all-terminal queue on its FIRST
    tick and parks the box you just resumed. It happened 4 seconds after
    `fleet watch 46648873 --profile jobs --budget 1`, and looked like a budget
    trip -- it was not; fleetd had accrued $0.0001 of a $1.00 cap.

    The empty-queue case is NOT the same bug and is deliberately allowed:
    arming a watch minutes before a wave submits is the normal launch order,
    and fleetd treats `queue_empty` as transient (JOBS_TRANSIENT_VERDICTS) and
    keeps the watch. It still gets a note, because the operator who meant to
    submit first wants to know.

    Pure: the caller does the B2 read. Returns None when a pending ticket
    exists (the watch has real work and the drain exit cannot fire).
    """
    pending = [v for v in (views or []) if v.get("status") not in TERMINAL]
    if pending:
        return None
    if job_ids:
        return (f"this box's queue holds {len(job_ids)} ticket(s) and EVERY ONE "
                f"is terminal ({', '.join(sorted(job_ids)[:3])}"
                f"{' ...' if len(job_ids) > 3 else ''}). A `jobs` watch reads "
                f"that as 'the work is finished' on its FIRST tick and PARKS the "
                f"box — within seconds, and it will look like a budget trip in "
                f"`fleet status`. SUBMIT THE JOB FIRST, then arm the watch; or "
                f"arm it with --keep, which suppresses the drain park.")
    return ("this box has no queued tickets — the watch will idle until one is "
            "submitted (harmless: `queue_empty` is transient and fleetd keeps "
            "the watch). Submitting first is still the tidier order.")


# --- submit-time vLLM sampler lint ------------------------------------------ #
# WHY THIS EXISTS: with the flashinfer sampler enabled (vLLM's default), vLLM
# JIT-compiles its sampling kernels at ENGINE STARTUP and the JIT #includes
# curand.h, which our baked images (train-t211-latest and its torch-runtime
# predecessors) do not ship. nvcc fails, the engine core never starts, and the
# job dies rc=1 BEFORE A SINGLE TOKEN -- after the box was rented, the assets
# staged and (often) a multi-GB pip install paid. It has now cost three box
# starts: modelzoo-reader-06's eval, and the S0.c pad smoke on sm_120
# (box 46656454, 2026-08-03). The remedy is one env var, which every canonical
# launcher sets (onstart/serve_vllm.sh, wave/gen_arm.sh,
# witness/gen_probe_resumable.pin_sampler_backend) -- but a bundle that
# builds its own engine, or ships a STALE vendored copy of one of those
# launchers, silently opts out. This turns that into a submit-time warning.
#
# The rule is deliberately blunt: a bundle file that starts a vLLM engine must
# MENTION the env var (any spelling: export, os.environ, a comment saying why it
# is delegated). Mentioning it is cheap; the failure it prevents is not.
_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"
#: shell: launching the OpenAI-compatible server. `.sh` only — a `.py` that
#: merely BUILDS a `vllm serve` argv (runsets/base-bakeoff/bakeoff_lib.py) or
#: names it in a docstring (jobs/v3-gate-e/gen_v3.py) launches nothing itself.
_VLLM_LAUNCH_SH_RE = re.compile(r"(?:^|[|;&(]|\bexec\s+)\s*vllm\s+serve\b")
#: python: an in-process engine in THIS process (vllm.LLM(...) / the engine
#: classes). An OpenAI-style client against an ALREADY-SERVED endpoint matches
#: nothing here, which is right — it starts no engine.
_VLLM_LAUNCH_PY_RE = re.compile(
    r"(?<![\w.])LLM\s*\(|AsyncLLMEngine|LLMEngine\s*\.")
_VLLM_SCAN_EXT = (".sh", ".py")
#: build outputs / staged corpora / caches: never launchers, sometimes enormous.
_VLLM_SCAN_SKIP_DIRS = {
    "__pycache__", ".git", "results", "out", ".dryrun", "assets-fixture",
    "checkpoints", "node_modules",
}
_VLLM_SCAN_MAX_BYTES = 2_000_000


def _strip_noncode(body: str, name: str) -> str:
    """Blank out the spans that document a launcher rather than being one.

    A `#` line-strip cannot see inside a STRING, so a module docstring that
    explains the harness reads as the harness. Measured 2026-08-26 on
    `jobs/v1415-p0-chat27b-gen/check_saturation.py`, which parses a log and a
    JSON artifact, constructs no engine, and was flagged anyway because its
    docstring says the harness drives vLLM via ``LLM().generate()``. A lint
    that fires on the recommended workflow is a lint that gets ignored.

    Python is tokenized so strings and comments both go; on a syntax error we
    fall back to the old line-based strip rather than skipping the file, since
    under-reporting this particular check costs a rented box.
    """
    if name.endswith(".py"):
        try:
            import io
            import tokenize
            drop = {tokenize.STRING, tokenize.COMMENT}
            if hasattr(tokenize, "FSTRING_START"):  # 3.12+ splits f-strings
                drop |= {tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE,
                         tokenize.FSTRING_END}
            out = []
            for tok in tokenize.generate_tokens(io.StringIO(body).readline):
                # Keep newlines so line structure (and any later slicing of
                # this text) survives; replace dropped spans with nothing.
                if tok.type in drop:
                    out.append("\n" * tok.string.count("\n"))
                else:
                    out.append(tok.string)
            return "".join(out)
        except (tokenize.TokenError, SyntaxError, IndentationError, ValueError):
            pass
    return "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))


def vllm_sampler_findings(bundle_dir: str, env: dict | None = None) -> list[str]:
    """Warn about bundle files that start a vLLM engine without pinning the
    sampler backend. Pure + offline: reads only files inside `bundle_dir`.

    `env` is the job-config `env:` block -- setting the var THERE covers the
    whole entrypoint, so it satisfies the check for every file in the bundle.
    """
    if (env or {}).get(_SAMPLER_ENV):
        return []
    out = []
    for root, dirs, files in os.walk(bundle_dir):
        dirs[:] = [d for d in dirs
                   if d not in _VLLM_SCAN_SKIP_DIRS and not d.startswith("data-")
                   and d != "data"]
        for nm in sorted(files):
            if not nm.endswith(_VLLM_SCAN_EXT):
                continue
            p = os.path.join(root, nm)
            try:
                if os.path.getsize(p) > _VLLM_SCAN_MAX_BYTES:
                    continue
                body = open(p, "r", encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if _SAMPLER_ENV in body:
                continue
            code = _strip_noncode(body, nm)
            rx = _VLLM_LAUNCH_SH_RE if nm.endswith(".sh") else _VLLM_LAUNCH_PY_RE
            if not rx.search(code):
                continue
            rel = os.path.relpath(p, bundle_dir)
            out.append(
                f"{rel} starts a vLLM engine but never mentions {_SAMPLER_ENV}. "
                f"The flashinfer sampler JIT-compiles at engine startup and needs "
                f"curand.h, which the baked images do not ship — the engine core "
                f"dies rc=1 before a single token. Set "
                f"`export {_SAMPLER_ENV}=\"${{{_SAMPLER_ENV}:-0}}\"` (or the "
                f"job-config `env:` key), or route the engine through "
                f"witness/gen_probe_resumable.py / onstart/serve_vllm.sh, "
                f"which pin it themselves. If this file is a VENDORED copy of one "
                f"of those, it is STALE — re-stage it.")
    return out


def validate_job_config(raw: dict, bundle_dir: str, *,
                        materialized: bool = False) -> tuple[dict, list[str]]:
    """Validate at SUBMIT time on the laptop (fail fast, before any upload).
    Returns (normalized_config, warnings). Raises JobmetaError on a hard fault.

    The normalized config is the canonical JSON that ships in the ticket — jobd
    reads THAT (no YAML parser box-side).

    `materialized=True` when `bundle_dir` is an EXTRACTED bundle rather than an
    authoring tree — see `resolve_includes`."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        raise JobmetaError("job-config must be a mapping")

    ver = raw.get("version", 1)
    if ver != 1:
        warnings.append(f"job-config version={ver!r} (expected 1) — proceeding")

    name = raw.get("name")
    if not name:
        raise JobmetaError("job-config: `name` is required")
    slug = slugify(str(name))

    # `includes:` — shared files overlaid at bundle time. Resolved HERE, before
    # the entrypoint check, so a bundle may name a shared file as its entrypoint
    # and so a bad include name is refused at submit rather than on a paid box.
    includes = _normalize_includes(raw.get("includes"))
    included = (resolve_includes(bundle_dir, includes, materialized=materialized)
                if includes else {})

    # `pinned_copies:` — the declared opposite of an include. Checked here so a
    # typo is refused at submit rather than reading as "no pin" forever.
    pinned = _normalize_pinned_copies(raw.get("pinned_copies"))
    both = sorted(set(pinned) & set(includes))
    if both:
        raise JobmetaError(
            f"job-config: {both} appear in BOTH `includes` and `pinned_copies`. "
            f"A file either follows the shared copy or is pinned to this "
            f"bundle's own — naming both leaves no rule for which wins, which "
            f"is exactly what these keys exist to settle.")
    for name in pinned:
        if not os.path.isfile(os.path.join(bundle_dir, name)):
            raise JobmetaError(
                f"job-config: `pinned_copies` names {name!r} but the bundle has "
                f"no such file. A pin asserts a local copy exists; if the file "
                f"was removed, drop the pin (or add `includes: [{name}]`).")

    entrypoint = raw.get("entrypoint")
    if not entrypoint or not isinstance(entrypoint, str):
        raise JobmetaError("job-config: `entrypoint` (a path in the bundle) is required")
    if entrypoint.startswith("/") or ".." in entrypoint.split("/"):
        raise JobmetaError(
            f"job-config: entrypoint {entrypoint!r} must be relative and not escape the bundle")
    ep_abs = os.path.join(bundle_dir, entrypoint)
    if not os.path.isfile(ep_abs) and entrypoint not in included:
        raise JobmetaError(f"job-config: entrypoint {entrypoint!r} not found in {bundle_dir}")

    timeout_s = raw.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool) or timeout_s <= 0:
        raise JobmetaError(f"job-config: timeout_s must be a positive int (got {timeout_s!r})")
    if timeout_s > MAX_TIMEOUT_S:
        raise JobmetaError(
            f"job-config: timeout_s={timeout_s} exceeds {MAX_TIMEOUT_S}s (7d) — "
            f"that is a `run`, not a job")

    env = raw.get("env") or {}
    if not isinstance(env, dict):
        raise JobmetaError("job-config: `env` must be a mapping")
    env = {str(k): "" if v is None else str(v) for k, v in env.items()}

    results = raw.get("results") or []
    if isinstance(results, str):
        results = [results]
    if not isinstance(results, list):
        raise JobmetaError("job-config: `results` must be a list of globs")
    results = [str(g) for g in results]
    for g in results:
        if g.startswith("/"):
            raise JobmetaError(f"job-config: result glob {g!r} must be relative")

    # mid-run checkpointing: every checkpoint_s seconds jobd syncs `checkpoints`
    # globs (default: the `results` globs) into jobs/<id>/checkpoints/ while the
    # entrypoint runs, so a timeout/OOM/box-death loses at most one interval of
    # work instead of everything. The on-disk relative paths are preserved under
    # that prefix, and a resuming box pulls them back from checkpoints/. The
    # separate prefix keeps jobs/<id>/results/ a single new-object write at
    # finalize (no B2 overwrite eventual-consistency window for downstream
    # readers). The trainer must actually WRITE periodic state (e.g. --save-steps)
    # for this to have anything to ship.
    checkpoint_s = raw.get("checkpoint_s")
    checkpoints = raw.get("checkpoints")
    if checkpoints is not None:
        if isinstance(checkpoints, str):
            checkpoints = [checkpoints]
        if not isinstance(checkpoints, list):
            raise JobmetaError("job-config: `checkpoints` must be a list of globs")
        checkpoints = [str(g) for g in checkpoints]
        for g in checkpoints:
            if g.startswith("/"):
                raise JobmetaError(f"job-config: checkpoint glob {g!r} must be relative")
        if checkpoint_s is None:
            checkpoint_s = 300          # globs given -> sensible default interval
    if checkpoint_s is not None:
        if isinstance(checkpoint_s, bool) or not isinstance(checkpoint_s, int) \
                or checkpoint_s <= 0:
            raise JobmetaError(
                f"job-config: checkpoint_s must be a positive int (got {checkpoint_s!r})")
        if checkpoint_s < 30:
            warnings.append(
                f"checkpoint_s={checkpoint_s} is aggressive — every pass re-lists "
                f"the workdir and hits B2; <30s is rarely useful")
        if not (checkpoints or results):
            raise JobmetaError(
                "job-config: checkpoint_s set but no `checkpoints` or `results` "
                "globs to sync")

    # `defend:` — how hard the bid ladder should fight to keep this job's box.
    # Read by bidpolicy's job-aware defense ceiling via the `submitted` event
    # (FLEET_REVIEW_2026-08-14 item 7); documented in JOBS_CONFIG.md.
    #
    #   dear   -> losing accumulated work is expensive, price it in
    #   cheap  -> the wall time already spent is not worth defending
    #
    # Derived when absent, because the two shapes we actually run already say
    # which they are: a job that checkpoints is training-shaped (resumable,
    # expensive to redo) => dear; a job that does not is bench-shaped
    # (deliberately checkpoint-free, "nothing worth resuming") => cheap. The
    # derived value is RESOLVED HERE and stored, so the supervisor reads one
    # field instead of re-deriving policy from a config it never sees, and an
    # old ticket without the field is distinguishable from one that chose.
    defend = raw.get("defend")
    if defend is not None:
        if not isinstance(defend, str) or defend.strip().lower() not in DEFEND_MODES:
            raise JobmetaError(
                f"job-config: defend must be one of {sorted(DEFEND_MODES)} "
                f"(got {defend!r})")
        defend = defend.strip().lower()
    else:
        defend = DEFEND_DEAR if checkpoint_s is not None else DEFEND_CHEAP

    needs = raw.get("needs") or {}
    if not isinstance(needs, dict):
        raise JobmetaError("job-config: `needs` must be a mapping")
    gpu = bool(needs.get("gpu", False))
    # needs.gpu_ram_gb: minimum GPU VRAM in GB the entrypoint requires. jobd
    # probes the box (nvidia-smi) and fails the job if the card is smaller — so
    # a job authored for a 96GB Blackwell won't silently OOM on a 32GB 5090.
    # Setting it implies needs.gpu (a VRAM floor is meaningless without a GPU).
    gpu_ram_gb = needs.get("gpu_ram_gb")
    if gpu_ram_gb is not None:
        if isinstance(gpu_ram_gb, bool) or not isinstance(gpu_ram_gb, int) or gpu_ram_gb <= 0:
            raise JobmetaError(
                f"job-config: needs.gpu_ram_gb must be a positive int GB (got {gpu_ram_gb!r})")
        gpu = True
    # needs.gpus: how many cards the entrypoint wants — jobd's scheduler assigns
    # exactly that many via CUDA_VISIBLE_DEVICES and runs jobs CONCURRENTLY while
    # cards are free (v2; v1 was one-job-per-box). int >= 1, or "all" (whole box —
    # resolved to the live card count box-side). Setting it implies needs.gpu.
    # Default: 1 when gpu is true (a plain GPU job gets one card, so 4 matrix
    # arms saturate a 4-GPU box), absent when gpu is false.
    gpus = needs.get("gpus")
    if gpus is not None:
        if isinstance(gpus, str):
            if gpus != "all":
                raise JobmetaError(
                    f"job-config: needs.gpus must be a positive int or \"all\" (got {gpus!r})")
        elif isinstance(gpus, bool) or not isinstance(gpus, int) or gpus <= 0:
            raise JobmetaError(
                f"job-config: needs.gpus must be a positive int or \"all\" (got {gpus!r})")
        gpu = True
    elif gpu:
        gpus = 1
    # needs.cpu_cores: how many CPU cores the entrypoint actually wants. A
    # COUNT, not a capability score — a run launching `--workers 64` needs 64
    # cores to be worth renting, and that question has nothing to do with how
    # fast each core is. Capability is handled at SELECTION time by ranking
    # (`market.offers`), never as a gate here: scoring a box needs a
    # distribution we do not have, and the GEMM lane's acceptance policy is
    # held unbuilt for exactly that reason (HOST_ACCEPTANCE_PROBE §5).
    #
    # Unlike gpu_ram_gb, it does NOT imply needs.gpu — a CPU-only bundle
    # (compile/search work) is the case this exists for, and `_job_shape`
    # already reads a falsy needs.gpu as "CPU".
    cpu_cores = needs.get("cpu_cores")
    if cpu_cores is not None:
        if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or cpu_cores <= 0:
            raise JobmetaError(
                f"job-config: needs.cpu_cores must be a positive int core count "
                f"(got {cpu_cores!r})")
    # needs.host_ram_gb: the HOST RAM (GB) the entrypoint needs, the CPU-side
    # twin of gpu_ram_gb. A CPU-shaped job is sized by memory far more often
    # than by cores — a bf16 CPU merge holds the whole base resident, so it
    # dies on a wide box with a small slice and runs fine on a narrow one with
    # a big slice, and cpu_cores cannot express that.
    #
    # MEASURED, NEVER AUTHORED — the same rule gpu_ram_gb carries (vram_facts).
    # A guess here buys either an OOM or a box twice the price of the one that
    # would have worked. The measurement comes from the peak-RSS harvest a run
    # banks (`host_ram_peak_gb`); a hand-written seed is legitimate only while
    # marked provisional and pending that number.
    #
    # Like cpu_cores it does NOT imply needs.gpu. Float, because RAM tiers are
    # not integers (a 126 GB slice is what a "128 GB" host rents out).
    host_ram_gb = needs.get("host_ram_gb")
    if host_ram_gb is not None:
        if isinstance(host_ram_gb, bool) or not isinstance(host_ram_gb, (int, float)):
            raise JobmetaError(
                f"job-config: needs.host_ram_gb must be a positive number of GB "
                f"(got {host_ram_gb!r})")
        host_ram_gb = float(host_ram_gb)
        if not host_ram_gb > 0:
            raise JobmetaError(
                f"job-config: needs.host_ram_gb must be a positive number of GB "
                f"(got {needs['host_ram_gb']!r})")
    # needs.cc_allow: the sm ARCHITECTURE allowlist this workload can run on,
    # e.g. [80, 86, 89, 90] for a bundle pinning flash_attention_2 (the baked
    # wheel ships no cubin for sm_100/sm_120) or [90] for one whose extension
    # was built at a single arch. This is a SELECTION-time statement — the
    # launcher turns it into `herdd launch --cc-allow`, which narrows the
    # offer search and stamps LAUNCH_CC_ALLOW so every eviction replacement
    # inherits it. It is NOT re-checked box-side; the entrypoint's own device
    # gate stays the backstop.
    #
    # ABSENT OR EMPTY MEANS UNCONSTRAINED, never "allow nothing" — the whole
    # pre-2026-08-19 behaviour, which is what an arch-agnostic bundle still
    # gets. Only a non-empty list narrows anything.
    cc_allow = needs.get("cc_allow")
    if cc_allow is not None:
        if isinstance(cc_allow, (str, bytes)) or not isinstance(cc_allow, (list, tuple)):
            raise JobmetaError(
                f"job-config: needs.cc_allow must be a list of sm levels, e.g. "
                f"[80, 86, 89, 90] (got {cc_allow!r})")
        levels: list[int] = []
        for it in cc_allow:
            s = str(it).strip().lower().removeprefix("sm_").removeprefix("sm")
            try:
                v = int(s)
            except (TypeError, ValueError):
                raise JobmetaError(
                    f"job-config: needs.cc_allow entries must be sm levels like "
                    f"90 or \"sm_90\" (got {it!r})") from None
            if v <= 0:
                raise JobmetaError(
                    f"job-config: needs.cc_allow entries must be positive "
                    f"(got {it!r})")
            # vast advertises `compute_cap` as sm x10 (800, 900, 1200) and both
            # spellings get hand-typed; the split is unambiguous because every
            # real sm level (75..120) is below 200 and every twin above it.
            # Same rule as vastlib.market.offers.parse_cc_allow, which re-parses
            # whatever the launcher passes through.
            if v >= 200:
                v //= 10
            if v not in levels:
                levels.append(v)
        cc_allow = sorted(levels)
        if cc_allow:
            gpu = True
    venv = needs.get("venv", "none")
    if venv is None:
        venv = "none"
    venv = str(venv)
    if venv not in VENV_CHOICES:
        raise JobmetaError(f"job-config: needs.venv must be one of {VENV_CHOICES} (got {venv!r})")

    # needs.scratch_gb / needs.disk_gb — the two disk knobs (velvet P4). They are
    # NOT interchangeable, and picking the wrong one is the whole point of having
    # both:
    #
    #   scratch_gb  ADDS a term to the derived estimate. It is for working state
    #               the config cannot possibly reveal — a ninja build tree of
    #               object files, per-worker compile worktrees and their PCHs,
    #               an unpacked toolchain. `assets:` sizes are measurable;
    #               whatever the entrypoint MAKES is not. Declaring it keeps the
    #               derivation live: assets grow, and the estimate grows with
    #               them, which a hand-typed total silently would not.
    #   disk_gb     OVERRIDES the estimate outright. An escape hatch for the case
    #               where the derivation is known wrong, not the normal knob.
    #
    # Reach for scratch_gb by default. disksize.py explains the derivation.
    scratch_gb = needs.get("scratch_gb")
    if scratch_gb is not None:
        if (isinstance(scratch_gb, bool) or not isinstance(scratch_gb, (int, float))
                or scratch_gb <= 0):
            raise JobmetaError(
                f"job-config: needs.scratch_gb must be a positive number of GB "
                f"(got {scratch_gb!r})")
        scratch_gb = float(scratch_gb)
    # needs.scratch_volatile: the author asserting this scratch is
    # RECONSTRUCTIBLE — a build tree, a compile shadow, an unpacked toolchain —
    # so it may live on a RAM-backed tmpfs instead of the allocated disk.
    # Measured on the audited box (BOX_SATURATION_AUDIT_2026-07-30 §3.1): 503 GiB
    # RAM with only 34-37 GB in use, and a 125 GB /dev/shm tmpfs that no script
    # in the tree touches. NOTE `/tmp` there is on the OVERLAY, not a tmpfs, so
    # "just use /tmp" would move nothing and cost the same disk.
    # Declaring this does not by itself move anything: disksize.plan_scratch_
    # placement still requires MEASURED box facts before it shrinks a disk.
    scratch_volatile = needs.get("scratch_volatile")
    if scratch_volatile is not None:
        if not isinstance(scratch_volatile, bool):
            raise JobmetaError(
                f"job-config: needs.scratch_volatile must be true or false "
                f"(got {scratch_volatile!r})")
        if scratch_volatile and not scratch_gb:
            raise JobmetaError(
                "job-config: needs.scratch_volatile has nothing to place — set "
                "needs.scratch_gb to say how much scratch the job makes")
    disk_gb = needs.get("disk_gb")
    if disk_gb is not None:
        if (isinstance(disk_gb, bool) or not isinstance(disk_gb, (int, float))
                or disk_gb <= 0):
            raise JobmetaError(
                f"job-config: needs.disk_gb must be a positive number of GB "
                f"(got {disk_gb!r})")
        disk_gb = float(disk_gb)

    # optional experiment-matrix association (set by jobmatrix.py submit; jobd
    # echoes exp_id/arm onto its lifecycle events so ANY arm's event log audits
    # back to its experiment even without the manifest). Absent = plain job.
    experiment = raw.get("experiment")
    if experiment is not None:
        if not isinstance(experiment, dict):
            raise JobmetaError("job-config: `experiment` must be a mapping")
        exp_id = experiment.get("exp_id")
        if not exp_id:
            raise JobmetaError("job-config: experiment.exp_id is required")
        validate_job_id(str(exp_id))
        arm = experiment.get("arm")
        if not arm or not JOB_ID_RE.match(str(arm)):
            raise JobmetaError(
                f"job-config: experiment.arm is required and must match "
                f"{JOB_ID_RE.pattern} (got {arm!r})")
        axes = experiment.get("axes") or {}
        if not isinstance(axes, dict):
            raise JobmetaError("job-config: experiment.axes must be a mapping")
        experiment = {"exp_id": str(exp_id), "arm": str(arm),
                      "axes": {str(k): str(v) for k, v in axes.items()}}

    # symlink safety: no symlink inside the bundle may resolve outside it.
    real_root = os.path.realpath(bundle_dir)
    for root, dirs, files in os.walk(bundle_dir):
        for nm in dirs + files:
            p = os.path.join(root, nm)
            if os.path.islink(p):
                tgt = os.path.realpath(p)
                if tgt != real_root and not tgt.startswith(real_root + os.sep):
                    raise JobmetaError(
                        f"job-config: symlink {p!r} escapes the bundle (-> {tgt!r})")

    # max_restarts: how many times jobd may pick the job back up after an
    # interruption (park/resume, preemption, daemon death) before declaring it
    # failed. Attempts are counted per box (a local crash-loop is bounded even
    # when the event log is unreachable). 0 = never resume (one attempt only).
    max_restarts = raw.get("max_restarts", DEFAULT_MAX_RESTARTS)
    if isinstance(max_restarts, bool) or not isinstance(max_restarts, int) \
            or max_restarts < 0:
        raise JobmetaError(
            f"job-config: max_restarts must be an int >= 0 (got {max_restarts!r})")

    # declarative big-input staging (N4): jobd pulls these onto the box BEFORE the
    # entrypoint via ONE shared retry+integrity primitive. Validated here, baked
    # into the ticket JSON verbatim.
    # `env` is the NORMALIZED map above (`env:` with `--env`/`--artifact`
    # already folded by the submit surface), so `${VAR}` in an asset prefix
    # resolves from exactly what the box will be handed.
    assets = _normalize_assets(raw, env)

    # top-level `tracks:` — provenance for B2 objects the ENTRYPOINT pulls
    # itself (no `assets:` entry to hang the declaration on; the jobmatrix DSL
    # has no assets field at all, so every matrix job's B2 reads live here).
    # Keys are full B2 keys. Consumed ONLY by the submit-time staleness
    # preflight; jobd ignores it.
    tracks = _normalize_tracks(raw.get("tracks"), "`tracks`")

    warnings.extend(vllm_sampler_findings(bundle_dir, env))

    # directory allowlist (submit-time policy gate; CREDENTIAL_LIFECYCLE.md).
    # ALWAYS validated (rejects an out-of-allowlist asset/scope before upload);
    # recorded in the ticket only when the job declares `scope:` or has assets, so
    # a plain job's ticket is unchanged.
    scope = validate_scope(raw, assets)

    cfg = {
        "version": 1,
        "name": slug,
        "entrypoint": entrypoint,
        "timeout_s": timeout_s,
        "env": env,
        "results": results,
        "needs": {"gpu": gpu, "venv": venv},
        "max_restarts": max_restarts,
    }
    if gpu_ram_gb is not None:
        cfg["needs"]["gpu_ram_gb"] = gpu_ram_gb
    if gpus is not None:
        cfg["needs"]["gpus"] = gpus
    if cpu_cores is not None:
        cfg["needs"]["cpu_cores"] = cpu_cores
    if host_ram_gb is not None:
        cfg["needs"]["host_ram_gb"] = host_ram_gb
    # Empty stays ABSENT from the ticket: `cc_allow: []` and no key at all mean
    # the same unconstrained thing, and recording an empty list would invite a
    # reader to treat it as "an allowlist that permits nothing".
    if cc_allow:
        cfg["needs"]["cc_allow"] = list(cc_allow)
    if scratch_gb is not None:
        cfg["needs"]["scratch_gb"] = scratch_gb
    if scratch_volatile is not None:
        cfg["needs"]["scratch_volatile"] = bool(scratch_volatile)
    if disk_gb is not None:
        cfg["needs"]["disk_gb"] = disk_gb
    if checkpoint_s is not None:
        cfg["checkpoint_s"] = checkpoint_s
        cfg["checkpoints"] = checkpoints if checkpoints is not None else list(results)
    cfg["defend"] = defend
    if experiment is not None:
        cfg["experiment"] = experiment
    if assets is not None:
        cfg["assets"] = assets
    if tracks:
        cfg["tracks"] = tracks
    if includes:
        # Provenance only — jobd never reads this. By the time a box sees the
        # bundle the shared files are already IN it, so the ticket records what
        # the bundle was assembled from, not an instruction to assemble it.
        cfg["includes"] = includes
    if pinned:
        # Provenance, like `includes` above — jobd never reads it. Recording it
        # in the ticket means the reason a bundle diverged travels with the run.
        cfg["pinned_copies"] = pinned
    if raw.get("scope") is not None or assets is not None:
        cfg["scope"] = scope
    return cfg, warnings


# --- shared bundle files (`includes:`) ---------------------------------------
# WHY THIS EXISTS. A bundle ships only its own folder, so every bundle that
# needed the shared launch-shape planner or the artifact-manifest writer got a
# COPY — and bundles are authored by forking the previous one, so the copies
# multiplied. Measured before this landed: `launch_plan.sh` byte-identical in 9
# bundles, `train_artifact.py` in 8. They were identical only by luck:
# test_autotune.py's bash-mirror parity check reads exactly ONE of the nine
# (repair-lifter-train's), so the other eight were unverified, and a
# cross-cutting edit ("bf16 is the training precision default, everywhere",
# d6730bce) had to find and touch every copy by hand.
#
# `includes:` lets a bundle NAME a shared file instead of carrying it. The file
# is overlaid into a staging copy at bundle time, so the tar a box receives is
# byte-identical to what it would have been with the copy in place — the
# migration is provable (see test_jobmeta_includes.py, which asserts the
# pre-migration bundle sha equals the post-migration materialized sha).
JOBCOMMON_DIR = os.path.join(_HERE, "jobcommon")


def _normalize_includes(raw, where: str = "`includes`") -> list[str]:
    """Normalize `includes:` to a sorted list of bare filenames. Each name is
    BOTH the file in JOBCOMMON_DIR and its destination path in the bundle — no
    renaming, because a shared file that arrives under a different name in
    different bundles is the drift this was built to remove."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise JobmetaError(f"job-config: {where} must be a list of file names")
    out = []
    for item in raw:
        name = str(item).strip()
        if not name:
            continue
        if name.startswith("/") or ".." in name.split("/") or "\\" in name:
            raise JobmetaError(
                f"job-config: {where} entry {name!r} must be a plain relative "
                f"name inside the shared dir (no absolute paths, no `..`)")
        out.append(name)
    dupes = {n for n in out if out.count(n) > 1}
    if dupes:
        raise JobmetaError(
            f"job-config: {where} lists {sorted(dupes)} more than once")
    return sorted(out)


def _normalize_pinned_copies(raw, where: str = "`pinned_copies`") -> dict[str, str]:
    """Normalize `pinned_copies:` to {filename: reason}.

    The counterpart to `includes:`. An include says "this bundle follows the
    shared file"; a pin says "this bundle keeps its OWN copy, on purpose, and
    here is why". Before this key existed the two were spelled identically —
    absence of an include — so a deliberately frozen bundle and a stale
    copy-paste were indistinguishable, which is how 7 of 8
    gen_probe_resumable.py copies drifted behind their source unnoticed.

    The reason is mandatory and free-form. It is the whole point: a pin with no
    reason is the ambiguity this key was added to remove."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise JobmetaError(
            f"job-config: {where} must be a mapping of filename -> reason "
            f"(got {type(raw).__name__})")
    out: dict[str, str] = {}
    for key, val in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if name.startswith("/") or ".." in name.split("/") or "\\" in name:
            raise JobmetaError(
                f"job-config: {where} entry {name!r} must be a plain relative "
                f"name inside the bundle (no absolute paths, no `..`)")
        reason = str(val).strip() if val is not None else ""
        if not reason:
            raise JobmetaError(
                f"job-config: {where}[{name!r}] needs a non-empty reason — say why "
                f"this bundle keeps its own copy (e.g. what it ran, or how it "
                f"differs). An unexplained pin is the state this key replaces.")
        out[name] = reason
    return dict(sorted(out.items()))


def _declared_pinned_copies(bundle_dir: str) -> dict[str, str]:
    """The bundle's `pinned_copies:`, or {} when the folder has no job-config.
    Same tolerance as `_declared_includes` — not every folder is a bundle."""
    if not any(os.path.isfile(os.path.join(bundle_dir, n))
               for n in ("job-config.json", "job-config.yaml")):
        return {}
    return _normalize_pinned_copies(load_job_config(bundle_dir).get("pinned_copies"))


def _declared_includes(bundle_dir: str) -> list[str]:
    """The bundle's `includes:`, or [] when the folder has no job-config at all.

    Not every directory these entry points see IS a job bundle — the jobd
    bootstrap bundle and a pile of test fixtures are plain folders — so a
    missing config means "declares no includes", not an error. A config that
    EXISTS and is malformed still raises."""
    if not any(os.path.isfile(os.path.join(bundle_dir, n))
               for n in ("job-config.json", "job-config.yaml")):
        return []
    return _normalize_includes(load_job_config(bundle_dir).get("includes"))


def resolve_includes(bundle_dir: str, includes=None, *,
                     materialized: bool = False) -> dict:
    """Map each declared include to its source path in JOBCOMMON_DIR.

    Raises when a name has no shared file (a typo would otherwise ship a bundle
    that is silently MISSING the file — the failure would land minutes into a
    paid run) and when the bundle also carries its own copy at that path. A
    local copy is refused even when it is byte-identical: that is the
    un-migrated state, and leaving both means the next edit has two places to
    land and no rule for which wins. To pin a private copy, drop the include.

    `materialized=True` inverts both rules, because the caller is holding a
    bundle that has ALREADY been through `materialize_bundle` — an extracted
    tar, not an authoring tree. There the include is SUPPOSED to be present as
    a file, so its presence is the success condition and its absence means a
    broken tar. Nothing is overlaid; each name maps to the file already in
    place. Without this the recovery paths refused every migrated bundle they
    were handed (`job retarget --reconstruct`), which is the worst possible
    moment to be strict: the ticket is already lost."""
    if includes is None:
        includes = _declared_includes(bundle_dir)
    if materialized:
        resolved = {}
        for name in includes:
            local = os.path.join(bundle_dir, name)
            if not os.path.isfile(local):
                raise JobmetaError(
                    f"job-config: includes: {name!r} is declared but MISSING "
                    f"from {bundle_dir} — this bundle was extracted from a tar "
                    f"that should already carry it, so the tar is incomplete")
            resolved[name] = local
        return resolved
    resolved = {}
    for name in includes:
        src = os.path.join(JOBCOMMON_DIR, name)
        if not os.path.isfile(src):
            avail = (sorted(os.listdir(JOBCOMMON_DIR))
                     if os.path.isdir(JOBCOMMON_DIR) else [])
            raise JobmetaError(
                f"job-config: includes: {name!r} is not in the shared dir "
                f"({JOBCOMMON_DIR}). Available: {', '.join(avail) or '(none)'}")
        local = os.path.join(bundle_dir, name)
        if os.path.exists(local):
            same = (os.path.isfile(local)
                    and open(local, "rb").read() == open(src, "rb").read())
            raise JobmetaError(
                f"job-config: {name!r} is declared in includes: AND present in "
                f"the bundle" + (" (byte-identical — this is the un-migrated "
                                 "state: delete the bundle's copy)" if same else
                                 " with DIFFERENT content: either delete the "
                                 "local copy to take the shared one, or drop "
                                 "the include to keep the local one"))
        resolved[name] = src
    return resolved


@contextlib.contextmanager
def materialize_bundle(src_dir: str):
    """Yield a directory holding `src_dir` plus its declared `includes:`.

    A bundle with no includes yields `src_dir` ITSELF — no copy, no temp dir, and
    (the point) a byte-identical content address to before this feature existed.
    A bundle with includes gets a staging copy; the temp dir lives only for the
    `with` body, so callers must finish reading before it closes."""
    includes = _declared_includes(src_dir)
    if not includes:
        yield src_dir
        return
    resolved = resolve_includes(src_dir, includes)
    tmp = tempfile.mkdtemp(prefix="jobbundle-")
    try:
        staged = os.path.join(tmp, os.path.basename(os.path.abspath(src_dir)) or "bundle")
        shutil.copytree(src_dir, staged, symlinks=True)
        for name, src in resolved.items():
            dest = os.path.join(staged, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(src, dest)
            # Assert it LANDED. An include that silently failed to copy would
            # produce a bundle that validates, uploads, and dies on the box.
            if not os.path.isfile(dest):
                raise JobmetaError(
                    f"internal: include {name!r} did not land in the staged bundle")
        yield staged
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def iter_bundle_files(bundle_dir: str):
    """Yield (relpath, abspath) for every file a bundle SHIPS — its own files
    plus its includes. Text-scanning gates (the B2 write preflight) walk this
    rather than the raw folder, so moving a file into the shared dir can never
    make a gate go blind to it."""
    skip = ("__pycache__", ".git", "node_modules")
    seen = set()
    for root, dirs, files in os.walk(bundle_dir):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for fn in sorted(files):
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, bundle_dir)
            seen.add(rel)
            yield rel, path
    try:
        resolved = resolve_includes(bundle_dir)
    except JobmetaError:
        return                      # validation reports it; scanning must not raise
    for name, src in sorted(resolved.items()):
        if name not in seen:
            yield name, src


# --- deterministic content-addressed bundling --------------------------------
def _norm_tarinfo(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """Zero every field that varies between two byte-identical folders so the tar
    hashes identically: mtime, uid/gid, uname/gname, and mode (canonicalized by
    type, NOT by the source exec bit — jobd dispatches the entrypoint by
    extension, so executability need not ride in the hash)."""
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.pax_headers = {}
    if ti.isdir():
        ti.mode = 0o755
    elif ti.issym() or ti.islnk():
        ti.mode = 0o777
    else:
        ti.mode = 0o644
    return ti


def deterministic_tar_bytes(src_dir: str) -> bytes:
    """Build a reproducible tar of `src_dir`: entries sorted by archive name,
    all volatile metadata zeroed. Two builds of byte-identical folders (any
    mtimes/owners/umask) produce identical bytes. PURE (reads the tree only)."""
    src = os.path.abspath(src_dir)
    if not os.path.isdir(src):
        raise JobmetaError(f"not a directory: {src_dir}")
    entries: list[tuple[str, str, bool]] = []
    for root, dirs, files in os.walk(src):     # os.walk does NOT follow dir symlinks
        dirs.sort()
        for d in sorted(dirs):
            full = os.path.join(root, d)
            entries.append((os.path.relpath(full, src), full, True))
        for f in sorted(files):
            full = os.path.join(root, f)
            entries.append((os.path.relpath(full, src), full, False))
    entries.sort(key=lambda e: e[0])
    buf = io.BytesIO()
    # GNU_FORMAT: deterministic long-name handling, no PAX float-time headers.
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for arc, full, isdir in entries:
            ti = tf.gettarinfo(full, arcname=arc)
            if ti is None:
                continue
            _norm_tarinfo(ti)
            if ti.isreg():
                with open(full, "rb") as fh:
                    tf.addfile(ti, fh)
            else:                              # dir / symlink: no payload
                tf.addfile(ti)
    return buf.getvalue()


def bundle_sha256(src_dir: str) -> str:
    """Content address of a bundle: sha256 of the DETERMINISTIC TAR bytes (not the
    compressed frame — so the address is invariant across zstd versions/levels).
    Two byte-identical folders share one address (dedupe).

    Materializes `includes:` first, so this address is of what a box RECEIVES.
    Both this and `write_bundle` must do that, and for a reason worth stating:
    workflowctl computes the address here and writes the blob there, in two
    separate calls (`workflowctl.py` `_upload_bundle`). Materializing in only
    one of them would hand it a sha that does not name its own bytes."""
    with materialize_bundle(src_dir) as staged:
        return hashlib.sha256(deterministic_tar_bytes(staged)).hexdigest()


def compress_tar(data: bytes) -> bytes:
    """zstd-compress the tar. zstandard module if present, else the `zstd` CLI."""
    try:
        import zstandard  # noqa
        return zstandard.ZstdCompressor(level=10).compress(data)
    except ImportError:
        pass
    if shutil.which("zstd"):
        p = subprocess.run(["zstd", "-q", "-10", "-c"], input=data,
                           stdout=subprocess.PIPE)
        if p.returncode == 0 and p.stdout:
            return p.stdout
    raise JobmetaError(
        "zstd unavailable: `pip install zstandard` or install the `zstd` CLI")


def decompress_zst(data: bytes) -> bytes:
    """Inverse of compress_tar. Auto-detects a raw (uncompressed) tar so a
    store-only fallback still round-trips."""
    if not data.startswith(_ZSTD_MAGIC):
        return data
    try:
        import zstandard  # noqa
        # stream_reader, NOT the one-shot .decompress(): the one-shot API
        # REQUIRES the content size in the frame header, and the `zstd` CLI
        # (compress_tar's fallback, used on any host without this module)
        # writes streaming frames that omit it. Mixing the two sides is the
        # normal case — a workstation with no zstandard builds the bundle,
        # a box image with zstandard extracts it — and the one-shot call
        # made that combination fail with "could not determine content size
        # in frame header" AFTER the box was rented. Streaming reads both.
        return zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(data)).read()
    except ImportError:
        pass
    if shutil.which("zstd"):
        p = subprocess.run(["zstd", "-q", "-d", "-c"], input=data,
                           stdout=subprocess.PIPE)
        if p.returncode == 0:
            return p.stdout
    raise JobmetaError(
        "zstd unavailable to decompress: install the `zstd` CLI or `zstandard`")


def write_bundle(src_dir: str, out_path: str) -> dict:
    """Deterministically tar+zstd `src_dir` to `out_path` (.tar.zst). Returns
    {sha256, tar_size, zst_size, path}. sha256 is the tar content address.

    Materializes `includes:` first — see `bundle_sha256` for why both entry
    points must."""
    with materialize_bundle(src_dir) as staged:
        tar = deterministic_tar_bytes(staged)
    sha = hashlib.sha256(tar).hexdigest()
    blob = compress_tar(tar)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(blob)
    return {"sha256": sha, "tar_size": len(tar), "zst_size": len(blob), "path": out_path}


def extract_bundle(blob_or_path, dest_dir: str, *, expect_sha: str | None = None) -> str:
    """Decompress + extract a bundle into `dest_dir` (created fresh — caller must
    ensure it is empty/clean). Verifies the decompressed tar's sha256 against
    `expect_sha` (integrity: the stored object name is <sha>.tar.zst). Rejects any
    member that would escape `dest_dir`. Returns the actual tar sha256."""
    if isinstance(blob_or_path, (bytes, bytearray)):
        blob = bytes(blob_or_path)
    else:
        with open(blob_or_path, "rb") as fh:
            blob = fh.read()
    tar = decompress_zst(blob)
    sha = hashlib.sha256(tar).hexdigest()
    if expect_sha is not None and sha != expect_sha:
        raise JobmetaError(
            f"bundle sha mismatch: expected {expect_sha}, got {sha} (corrupt download?)")
    os.makedirs(dest_dir, exist_ok=True)
    dest_real = os.path.realpath(dest_dir)
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as tf:
        for m in tf.getmembers():
            target = os.path.realpath(os.path.join(dest_dir, m.name))
            if target != dest_real and not target.startswith(dest_real + os.sep):
                raise JobmetaError(f"bundle member {m.name!r} escapes dest — refusing")
        tf.extractall(dest_dir)
    return sha


# --- events (envelope reuses runmeta primitives) -----------------------------
def make_event(job_id: str, event: str, actor: str, ts: str | None = None,
               **fields) -> dict:
    """Build a v1 job event. Unknown `event` values are allowed (fold tolerates)."""
    validate_job_id(job_id)
    ev = {
        "v": SCHEMA_VERSION,
        "ts": ts or now_ts(),
        "actor": actor,
        "event": event,
        "job_id": job_id,
        "nonce": fields.pop("nonce", None) or nonce(),
    }
    ev.update({k: v for k, v in fields.items() if v is not None})
    return ev


def _coerce(raw) -> dict | None:
    """Return a valid job event dict, or None (counted as a parse error). Mirrors
    runmeta._coerce but keys on `job_id` (the only reason it is not imported). One
    bad object NEVER breaks a fold."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    if not all(raw.get(k) for k in _CORE_KEYS):
        return None
    return raw


def _ev_order(e) -> tuple:
    """Total order on events: (ts, nonce). The SAME key `fold_events` sorts by, so
    "newer than" means one thing everywhere in this module."""
    return (e.get("ts", ""), e.get("nonce", ""))


def _status_lattice(types) -> str:
    """Fold a SET of event names to a status. Terminal precedence:
    `failed`/`done` > `cancelled` (a real entrypoint outcome outranks a late
    operator cancel). Set-based and therefore ORDER-BLIND — when `failed` AND
    `done` are both present only `_fold_status` (which sees the events, not just
    their names) can rank them; this set fold's failed-first answer is a
    tie-break for the degenerate same-instant case only. Callers fold status via
    `_fold_status`; this stays the shared non-terminal lattice."""
    if "failed" in types:                       # order-blind tie-break; see
        return "failed"                         #   _fold_status for the real rule
    if "done" in types:
        return "done"
    if "cancelled" in types:                    # operator override; loses only to a
        return "cancelled"                      #   real done/failed (results exist)
    if "started" in types:
        return "started"
    if "claimed" in types:
        return "claimed"
    if "submitted" in types:
        return "submitted"
    return "unknown"


def _fold_status(evs) -> str:
    """Fold a LIST of events to a status. Same lattice as `_status_lattice`,
    with the ONE conflict that lattice cannot rank resolved by ORDER: when the
    log holds both `failed` and `done`, the NEWER of the two (by `_ev_order`,
    the module's single total order) is the job's outcome.

    Why order and not a fixed precedence: a requeued job that then SUCCEEDS
    writes exactly this log — an early attempt's `failed`, a `resumed`, and a
    final `done`. A set-based failed-beats-done fold reported that job `failed`
    forever (measured 2026-08-06 on 20260806T082213-v11-qwen25c7b-chat-dec-
    train-aff8: `failed rc=1` at 09:34 on box 46962674, `done rc=0` at 21:05 on
    box 47011548 with all gates passed and the adapter published — and both
    `job status` and `job wait --until terminal` reported rc=1, nearly writing
    off a paid-for, completed run). No event is ever emitted after a genuine
    `done` except operator chatter, so the newest terminal word is the truth.

    A tie on (ts, nonce) — only possible for a literally duplicated event —
    falls back to the lattice's conservative failed-first answer."""
    types = {e.get("event") for e in evs}
    if "failed" in types and "done" in types:
        last_fail = max(_ev_order(e) for e in evs if e.get("event") == "failed")
        last_done = max(_ev_order(e) for e in evs if e.get("event") == "done")
        if last_done > last_fail:
            return "done"
        return "failed"
    return _status_lattice(types)


def fold_events(raw_events, live_iids=()) -> dict:
    """Fold an unordered multiset of job events into a view. Tolerant to
    missing/extra/duplicate/out-of-order objects (never mutates B2).

    `live_iids` is the INJECTED set of vast-live instance ids (the only source of
    'is the box alive now'). A non-terminal claimed/started job whose box IID is
    not live folds to display_status `lost` (never resurrected from recency).
    Terminal (done/failed) is sticky and wins; when BOTH exist the NEWER one is
    the outcome (`_fold_status` — a stale early-attempt `failed` must never
    outrank the final attempt's `done`)."""
    live = {str(x) for x in (live_iids or ())}
    evs: list[dict] = []
    parse_errors = 0
    for r in raw_events:
        e = _coerce(r)
        if e is None:
            parse_errors += 1
        else:
            evs.append(e)
    # deterministic display order (NOT a status oracle): (ts, nonce).
    evs.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))

    view = {
        "job_id": evs[-1]["job_id"] if evs else None,
        "status": "unknown", "display_status": "unknown", "live": False,
        "instance_id": None, "target_box": None,
        "exp_id": None, "arm": None,
        "bundle_sha256": None, "name": None, "entrypoint": None, "timeout_s": None,
        "defend": None, "gpu": None,
        "rc": None, "fail_reason": None,
        "reopened": False, "reopened_at": None,
        "prior_rc": None, "prior_fail_reason": None,
        "started_at": None, "ended_at": None,
        "last_event": None, "last_event_ts": None,
        "last_heartbeat_ts": None, "last_tail": None, "last_metrics": None,
        "last_checkpoint_ts": None, "n_checkpoints": 0,
        "attempts": 0, "n_preempted": 0, "last_resumed_ts": None,
        "results": None, "declared_globs": None,
        "terminal_boxes": [],
        "n_events": len(evs), "parse_errors": parse_errors,
    }
    if not evs:
        return view

    types = {e.get("event") for e in evs}
    status = _fold_status(evs)

    # --- the ONE un-stick: `resumed` newer than `failed` (operator requeue) ----
    # Terminal is sticky by default. The single exception is `herdd job requeue`
    # (see requeue_ticket): it re-mints the queue ticket for a TERMINAL-FAILED job
    # under the SAME JOB_ID and emits `resumed` (frozen vocabulary — no new event
    # kind). Without this rule the fold would keep reporting `failed` while a box
    # is demonstrably running the job again, and every reader downstream (`job
    # ls`/`status`/`wait`, supervise, the matrix) would treat the live re-run as
    # dead.
    #
    # Deliberately narrow, in three ways:
    #   * ONLY `failed` un-sticks. `done` (the entrypoint reached rc=0, results on
    #     B2) and `cancelled` (the explicit never-revive verdict — JOBS_DESIGN
    #     "Cancel") stay sticky UNCONDITIONALLY, so the rule fires only when
    #     `failed` is the sole terminal in the log. (Once the re-run's `done`
    #     lands the log holds BOTH terminals; this window closes and
    #     `_fold_status`'s newest-terminal rule reports the `done` — the fold
    #     must never regress to `failed` at the moment the job succeeds.)
    #   * ORDER decides. The comparison is on (ts, nonce) — the same total order
    #     the display sort uses — so an ordinary crash/preempt `resumed` that
    #     PRECEDED the failure leaves the job terminal.
    #   * The re-opened status is folded from the events at-or-after the
    #     re-opening `resumed` ONLY. A requeue puts the job back in the QUEUE, so a
    #     bare `resumed` folds to `submitted`; the new box's `claimed`/`started`
    #     take over as they land, and a second `failed` re-sticks it. Restricting
    #     the window is what keeps this monotone — the OLD attempt's `started`
    #     must never make a freshly-queued job look like it is running.
    reopen_at = None
    if types & TERMINAL == {"failed"}:
        last_fail = max((_ev_order(e) for e in evs if e.get("event") == "failed"),
                        default=None)
        last_res = max((_ev_order(e) for e in evs if e.get("event") == "resumed"),
                       default=None)
        if last_fail and last_res and last_res > last_fail:
            reopen_at = last_res
    if reopen_at is not None:
        post = [e for e in evs if _ev_order(e) >= reopen_at]
        status = _fold_status(post)
        if status == "unknown":         # bare `resumed` — re-queued, awaiting a claim
            status = "submitted"
        view["reopened"] = status not in TERMINAL
        # THE ATTEMPT BOUNDARY, published for readers of the B2 side-objects.
        # Everything under jobs/<id>/ that is a MUTABLE key rather than an event
        # — results/, log.txt, results.DONE.json — was written by whichever
        # attempt got there last, and a requeue does not clear any of it. So a
        # reader holding one of those objects needs to know when the current
        # attempt began before it may claim the object describes it. Set
        # whenever the un-stick fired, INCLUDING when the re-run has since gone
        # terminal (`reopened` is False then, but the boundary is still the
        # thing that dates the marker). See `classify_done_marker`.
        view["reopened_at"] = reopen_at[0] or None
    view["status"] = status

    # experiment association: the CLI stamps exp_id/arm on `submitted`, jobd
    # echoes them on its lifecycle events — take the last non-empty sighting so
    # the audit trail survives losing either side's events.
    for e in evs:
        if e.get("exp_id"):
            view["exp_id"] = e.get("exp_id")
        if e.get("arm"):
            view["arm"] = e.get("arm")

    sub = next((e for e in evs if e.get("event") == "submitted"), None)
    if sub:
        view["bundle_sha256"] = sub.get("bundle_sha256")
        view["name"] = sub.get("name")
        view["entrypoint"] = sub.get("entrypoint")
        view["timeout_s"] = _num(sub.get("timeout_s"))
        # `defend` is a bid-ladder input, not a jobs one: the fold carries it so
        # the supervisor can price a defense without reading the bundle. None on
        # any ticket submitted before 2026-08-14, which the ladder reads as "no
        # hint" and derives from, exactly as it did before the key existed.
        view["defend"] = sub.get("defend")
        # tri-state launch shape: True = GPU job, False = CPU-only bundle
        # (needs.gpu falsy), None = pre-2026-08-27 stream that never stamped it
        # — readers must render None as unknown, never as CPU.
        view["gpu"] = bool(sub["gpu"]) if "gpu" in sub else None
        view["target_box"] = sub.get("box")
        view["declared_globs"] = _num(sub.get("n_results_globs"))

    # Whose queue is the ticket in?  `submitted`.box is only the ORIGINAL target —
    # stale the moment the ticket moves — and TWO events move it.  BOTH have to be
    # folded here or the field names a box the job has left:
    #
    #   * `retargeted` — `retarget_ticket` MOVES the existing pointer (writes the
    #     new queue key, deletes the old).  Observed 2026-08-02 on
    #     20260715T081939-68b3b57b-generate-a0, which `job status` placed on
    #     44960616 while its ticket sat under 44967157.  Not a corner case: the
    #     boot-pull watchdog (27dc7fd2) retargets every ticket it reschedules.
    #   * `resumed` CARRYING A `box` FIELD — `requeue_ticket` RECONSTRUCTS a
    #     pointer on a new box for a terminal-FAILED job, and by design emits no
    #     new event kind (frozen vocabulary; see requeue_ticket step 4).  This one
    #     was missed until 2026-08-07: on 20260806T212132-v9-gemma4-dec-train-8818
    #     two operator requeues walked the ticket 47041615 -> 47042386 -> 47045282
    #     while `target_box` stayed pinned to 47041615, a box destroyed hours
    #     earlier, for the whole of a live 156-step training run.
    #
    # jobd's OWN `resumed` (kind=retarget/crash/preempt, actor `box:<iid>`) carries
    # no `box` — it reports a continuation on the box it is already on, not a move.
    # So "a `resumed` with a `box`" is exactly the ticket-moved discriminator, and
    # keying on the field rather than on `kind` keeps this true for any future
    # emitter that moves a ticket without inventing an event name.
    #
    # `instance_id` (claimed/started/heartbeat) still answers "where did it RUN";
    # this answers "whose queue is it in", which is the only truth for a job that
    # was moved before it ever ran.
    rets = [e for e in evs
            if e.get("box") and e.get("event") in ("retargeted", "resumed")]
    if rets:
        view["target_box"] = rets[-1].get("box")
        # requeue_ticket writes "-" for an unknown predecessor; don't launder that
        # placeholder into a box id.
        _from = rets[-1].get("from_box")
        if _from and _from != "-":
            view["retargeted_from"] = _from

    # A ticket move puts the job on a DIFFERENT box, so every box event before it
    # describes a machine this job is no longer on.  Bound the search at the newest
    # move: otherwise `instance_id` keeps naming the old box until the new one
    # emits its first claim, and since that box is typically destroyed by then it
    # is also not in `live` -> the job displays `interrupted` AGAINST A BOX IT HAS
    # ALREADY LEFT (measured 2026-08-06, 20260806T071847-fit-ladder-ea7c: the
    # `retargeted`->`claimed` gap was 5 s, and `target_box` had already flipped
    # while `instance_id` had not).  That disagreement is why "wait for job status
    # to report the new box id" is not by itself a usable gate -- the two fields
    # answer different questions and flip at different times.  With no box event
    # after the retarget, `instance_id` is genuinely UNKNOWN, and None says so;
    # `target_box` above is the field that answers "whose queue is it in".
    _ret_ts = rets[-1].get("ts") if rets else None
    boxev = [e for e in evs
             if e.get("event") in ("claimed", "started", "heartbeat") and e.get("instance_id")
             and (_ret_ts is None or (e.get("ts") or "") >= _ret_ts)]
    if boxev:
        view["instance_id"] = boxev[-1].get("instance_id")

    hbs = [e for e in evs if e.get("event") == "heartbeat"]
    if hbs:
        view["last_heartbeat_ts"] = hbs[-1].get("ts")
        view["last_tail"] = hbs[-1].get("tail")
        # host_metrics attribution happens BELOW, once `instance_id` is final --
        # a `resumed` event can still move it after this point.

    ckpts = [e for e in evs if e.get("event") == "checkpoint"]
    if ckpts:
        view["last_checkpoint_ts"] = ckpts[-1].get("ts")
        view["n_checkpoints"] = len(ckpts)

    # interruption bookkeeping (v2, revised 2026-08-09): a naive `started` tally
    # over-counts "attempts" the moment a job survives a preempt-resume — jobd
    # emits ONE `started` per actual entrypoint launch, including the relaunch
    # after every resume (crash OR preempt), so a single clean preempt/resume
    # already shows TWO `started`s. The old fold reported that as attempts=2,
    # n_preempted=0 whenever the interruption was signal-less (park/eviction:
    # jobd.sh's own trap almost never fires — vast delivers no SIGTERM on
    # eviction, see the boot-nonce block in onstart/jobd.sh) — an operator
    # reading `job status --json` saw "crashed once, restart budget consumed"
    # for a box that was cleanly evicted and resumed (drill 2026-08-09: durable
    # on-box counters read attempts=1/preempts=1 while this view read
    # attempts=2/n_preempted=0).
    #
    # jobd's `resumed` ALWAYS carries `kind` ("preempt"|"crash"|"retarget"|
    # "requeue") — it is emitted unconditionally on every resume path, unlike
    # the trap's `preempted`, which only exists when a signal actually landed.
    # So `resumed{kind:preempt}` is the reliable per-interruption signal; fold
    # against IT, not against the `started` relaunch it precedes:
    #   attempts     = count(started) - count(resumed kind=preempt), floored at
    #                  1 whenever any `started` exists (never negative, never
    #                  zero for a job that actually ran).
    #   n_preempted  = count(resumed kind=preempt) + any `preempted` events that
    #                  are NOT yet followed by one (still mid-interruption, no
    #                  resume landed) — so a trap-emitted `preempted` immediately
    #                  followed by its own `resumed{kind:preempt}` counts ONCE,
    #                  not twice.
    # `resumed` events without a `kind` (older/synthetic logs) are inert here —
    # conservative default, unchanged from the pre-fix fold — so this only
    # activates for streams that actually carry the field jobd emits today.
    started_evs = [e for e in evs if e.get("event") == "started"]
    preempt_resumes = [e for e in evs
                        if e.get("event") == "resumed" and e.get("kind") == "preempt"]
    view["attempts"] = len(started_evs) - len(preempt_resumes)
    if started_evs:
        view["attempts"] = max(view["attempts"], 1)
    _last_preempt_resume_order = max((_ev_order(e) for e in preempt_resumes),
                                      default=None)
    _pending_preempted = [
        e for e in evs if e.get("event") == "preempted"
        and (_last_preempt_resume_order is None
             or _ev_order(e) > _last_preempt_resume_order)]
    view["n_preempted"] = len(preempt_resumes) + len(_pending_preempted)
    res = [e for e in evs if e.get("event") == "resumed"]
    if res:
        view["last_resumed_ts"] = res[-1].get("ts")
        if res[-1].get("instance_id"):
            view["instance_id"] = res[-1].get("instance_id")

    # host_metrics (GPU util / cpu / net / disk) rides on heartbeats when the box
    # carries metrics_probe.py.  ATTRIBUTE IT -- "the latest heartbeat that has
    # metrics" is not the same as "THIS box's metrics", and across a box change the
    # difference is a dead machine's GPU numbers rendered under the live one's
    # status line: a healthy `gpu_util:100` for a box that has already died, or an
    # idle 0% for one that is busy.  Same family as defect (7) in
    # QWEN36_27B_LORA_TRAINABILITY (a resume line read off the prior box's log
    # tail), and NOT closed by that defect's mitigation: metrics fold independently
    # of `instance_id`, so gating a check on "job status reports the new box id"
    # still leaves this surface showing the old box.  Presence is not provenance.
    # Deliberately placed after the `resumed` fix-up above so it filters against
    # the FINAL box identity, not an intermediate one.
    metric_hbs = [e for e in evs
                  if e.get("event") == "heartbeat" and e.get("host_metrics")
                  and (view["instance_id"] is None
                       or e.get("instance_id") == view["instance_id"])]
    if metric_hbs:
        view["last_metrics"] = metric_hbs[-1].get("host_metrics")
        view["last_metrics_box"] = metric_hbs[-1].get("instance_id")
        view["last_metrics_ts"] = metric_hbs[-1].get("ts")

    # Which boxes are POISONED for this JOB_ID? A jobd that emits a terminal
    # event also writes $STATE_DIR/<JOB_ID>.terminal, which poll_once checks
    # before any B2 read — so that box skips a re-queued ticket forever, in
    # silence. `job requeue` refuses such a target; `job retarget` needs the same
    # evidence, and only the event's own attribution carries it (`instance_id`
    # above answers "where did it RUN", which a claim-time failure never sets).
    # NOT a complete census: the box also latches on paths that emit nothing
    # (remote-done, the restart/disk caps), so membership PROVES poison while
    # absence proves nothing.
    _tb: list[str] = []
    for e in evs:
        if e.get("event") not in TERMINAL:
            continue
        _b = e.get("instance_id")
        if not _b:
            _a = str(e.get("actor") or "")
            _b = _a[4:] if _a.startswith("box:") else None
        if _b and str(_b) not in _tb:
            _tb.append(str(_b))
    view["terminal_boxes"] = sorted(_tb)

    failed = [e for e in evs if e.get("event") == "failed"]
    done = [e for e in evs if e.get("event") == "done"]
    # Which terminal word stands? Same order rule as `_fold_status`: a `done`
    # NEWER than every `failed` (a requeued job that then succeeded) speaks for
    # rc; the stale failure is demoted to prior_* below, never reported as the
    # outcome of a job whose final attempt finished rc=0.
    _done_newer = bool(done) and (
        not failed
        or max(_ev_order(e) for e in done) > max(_ev_order(e) for e in failed))
    if failed and not _done_newer:
        view["fail_reason"] = failed[-1].get("reason")
        view["rc"] = _num(failed[-1].get("rc"))
        # A failure tail outranks the heartbeat tail because it is the most
        # diagnostic thing available -- but ONLY while that failure is still the
        # latest word. After a requeue/retarget the job runs again and streams
        # fresh heartbeats; pinning `last_tail` to the dead attempt's traceback
        # then makes a HEALTHY job read as a crashing one.
        #
        # That is not hypothetical: on 2026-08-06 `job logs` on
        # 20260806T082213-v11-...-aff8 printed a ChildFailedError from a
        # previous box while the current attempt was 55/156 steps in at
        # loss 0.2597, and it was believed. The newest heartbeat's tail was
        # correct all along -- this override buried it.
        #
        # So the failure tail wins only when no heartbeat is NEWER than it.
        # Timestamps are runmeta's zero-padded UTC stamps, lexically ordered.
        _f_ts = failed[-1].get("ts") or ""
        _hb_newer = bool(hbs) and (hbs[-1].get("ts") or "") > _f_ts
        if failed[-1].get("tail") and not _hb_newer:
            view["last_tail"] = failed[-1].get("tail")
    if _done_newer and _num(done[-1].get("rc")) is not None:
        view["rc"] = _num(done[-1].get("rc"))
    if _done_newer and failed:
        # the stale early-attempt failure survives as DIAGNOSIS (prior_*), with
        # the same field semantics the requeue un-stick uses -- never as the
        # job's outcome.
        view["prior_rc"] = _num(failed[-1].get("rc"))
        view["prior_fail_reason"] = failed[-1].get("reason")
    # cancel reason/rc surface only when `cancelled` is the winning terminal (a
    # real done/failed outranks it — see the precedence above), so a job that
    # finished a beat before the cancel still reports its genuine outcome.
    cancelled = [e for e in evs if e.get("event") == "cancelled"]
    if status == "cancelled" and cancelled:
        view["fail_reason"] = cancelled[-1].get("reason")
        if view["rc"] is None:
            view["rc"] = _num(cancelled[-1].get("rc"))

    for e in [e for e in evs if e.get("event") in ("results_uploaded", "done")]:
        if e.get("results") is not None:
            view["results"] = e.get("results")

    # A RE-OPENED job (requeue un-stick above) has not ended and has no current
    # rc: the prior attempt's outcome moves to `prior_rc`/`prior_fail_reason` so
    # the diagnosis survives, but nothing reports a live job as finished.
    if view["reopened"]:
        view["prior_rc"], view["rc"] = view["rc"], None
        view["prior_fail_reason"], view["fail_reason"] = view["fail_reason"], None

    starts = [e["ts"] for e in evs if e.get("event") in ("claimed", "started")]
    view["started_at"] = min(starts) if starts else None
    terms = [e["ts"] for e in evs if e.get("event") in TERMINAL]
    view["ended_at"] = None if view["reopened"] else (max(terms) if terms else None)
    view["last_event"] = evs[-1].get("event")
    view["last_event_ts"] = evs[-1].get("ts")

    # liveness derivation (I5-analog: vast is the only "alive now" source).
    # v2: a claimed/started job on a dead box displays `interrupted` (jobd
    # resumes it when the box comes back, or `job retarget` moves the ticket) —
    # the v1 `lost` dead-end no longer exists.
    iid = view["instance_id"]
    view["live"] = bool(iid is not None and str(iid) in live)
    if status in TERMINAL:
        view["display_status"] = status
    elif status in ("claimed", "started"):
        view["display_status"] = "running" if view["live"] else "interrupted"
    else:                                       # submitted / unknown: queued
        view["display_status"] = status
    return view


# --- software-epoch stamp ----------------------------------------------------
# A box has no repo: the bundle it receives is a tar with no .git, so nothing on
# the far side can name the checkout that produced it. Anything mined out of a
# run's telemetry later (throughput anchors above all) needs that name to know
# whether two measurements belong to the same software epoch — the training path
# only ever got FASTER, so comparing across a boundary silently understates the
# newer one.
#
# The stamp rides the ticket's `env`, which jobd sources into the entrypoint, so
# it costs no bundle byte and cannot move a bundle's content address. It is
# applied at SUBMIT, never in `validate_job_config`/`make_ticket`: those are
# pure and their output is pinned by tests, and a HEAD-dependent config would
# also make `submit_with_id`'s idempotent resubmit conflict after any commit.
TRAINER_REV_ENV = "DS_TRAINER_REV"
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


def repo_head_rev(repo_root: str | None = None) -> str | None:
    """The client checkout's HEAD, `-dirty`-suffixed when the tree is modified,
    or None outside a git repo / without git. Never raises: a missing rev is a
    telemetry hole, not a reason to refuse a submit.

    Deliberately UNCACHED — two git calls per submit is nothing, and a cache
    would let a long-lived process keep stamping a rev it has since moved off."""
    root = repo_root or _REPO_ROOT
    rev = None
    try:
        p = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and (p.stdout or "").strip():
            rev = p.stdout.strip()
            d = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20)
            if d.returncode == 0 and (d.stdout or "").strip():
                rev += "-dirty"
    except Exception:                                   # noqa: BLE001
        rev = None
    return rev


def stamp_trainer_rev(config: dict, repo_root: str | None = None) -> str | None:
    """Stamp the client HEAD into a resolved config's `env`, in place. Returns
    the value stamped, or None when it did not apply.

    A value already in the config WINS — an explicit pin is the operator saying
    which epoch this run belongs to, and a submit must not overwrite it."""
    if not isinstance(config, dict):
        return None
    env = config.get("env")
    if not isinstance(env, dict) or (env.get(TRAINER_REV_ENV) or "").strip():
        return None
    rev = repo_head_rev(repo_root)
    if rev:
        env[TRAINER_REV_ENV] = rev
    return rev


# --- ticket ------------------------------------------------------------------
def make_ticket(job_id: str, bundle_sha256: str, actor: str, config: dict,
                box) -> dict:
    """The tiny immutable submission ticket. Carries the CANONICAL config JSON so
    jobd never needs a YAML parser box-side."""
    validate_job_id(job_id)
    # Defence in depth for `${VAR}` asset prefixes: resolution is SUBMIT-side, so
    # an unresolved placeholder reaching the ticket would be a rented box pulling
    # a literal `${...}` key. `resolve_asset_vars` already refuses it; this is the
    # backstop at the one door every surface writes through.
    for a in ((config or {}).get("assets") or []):
        if "${" in str(a.get("b2") or ""):
            raise JobmetaError(
                f"job-config: asset {a.get('name')!r} would ship an UNRESOLVED "
                f"prefix {a.get('b2')!r} to the box — refusing to write the "
                f"ticket (tools/vast/ASSET_PARAMETERIZATION.md)")
    return {
        "v": SCHEMA_VERSION,
        "job_id": job_id,
        "bundle_sha256": bundle_sha256,
        "submitted_ts": now_ts(),
        "actor": actor,
        "box": box,
        "config": config,
    }


# --- transport seam (injectable runner; runmeta contract) --------------------
def _q(bucket, key):
    return f"b2:{_bucket(bucket)}/{key}"


def _wq(bucket, key):
    """Write-side remote path. On a box carrying an Option-1b split key
    (`B2_WRITE_KEY_ID` set — a scoped `[b2w]` remote whose namePrefix is `jobs/`)
    box-side writes go to `[b2w]`; everywhere else (workstation, or a box with a
    single bucket-wide key) it is the plain `[b2]` remote. Every box-side write is
    under `jobs/`, which the write key's namePrefix covers. See
    tools/vast/CREDENTIAL_LIFECYCLE.md."""
    remote = "b2w" if os.environ.get("B2_WRITE_KEY_ID") else "b2"
    return f"{remote}:{_bucket(bucket)}/{key}"


def emit_event(job_id, event, *, actor=None, runner=_default_runner, bucket=None,
               **fields) -> dict:
    """Append one immutable event object to jobs/<job_id>/events/. Best-effort:
    a transport failure returns the event with `_emitted=False` (a dying box's
    final emit can't crash the caller)."""
    actor = actor or _default_actor()
    ev = make_event(job_id, event, actor, **fields)
    key = f"jobs/{job_id}/events/{event_key(ev)}"
    body = json.dumps(ev, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _wq(bucket, key)], input=body)   # box-side write
    ev["_key"] = key
    ev["_emitted"] = (rc == 0)
    if rc != 0:
        ev["_error"] = (err or "").strip()
    return ev


def _default_actor():
    iid = os.environ.get("VAST_INSTANCE_ID") or os.environ.get("INSTANCE_ID")
    if iid:
        return f"box:{iid}"
    return f"cli:{os.environ.get('HOSTNAME') or runmeta._hostname()}"


def bundle_exists(sha, *, runner=_default_runner, bucket=None) -> bool:
    """Dedupe check: is jobs/bundles/<sha>.tar.zst already on B2?"""
    rc, out, _ = runner(["lsf", _q(bucket, f"jobs/bundles/{sha}.tar.zst")])
    return rc == 0 and bool((out or "").strip())


def upload_bundle(local_path, sha, *, runner=_default_runner, bucket=None):
    """Upload a bundle to its content-addressed key. Returns (ok, err).

    List-based `copy --include`, never `copyto`: copyto's per-key HeadObject
    intermittently 403s on B2 in hours-long windows (2026-07-11 it rejected
    every `job submit` in the window; same class as the jobd results-push
    rule and eval-env bake.sh's b2_put). The staged file is named
    `<sha>.tar.zst`, so a dir->dir copy filtered to that one name lands the
    exact content-addressed key."""
    src_dir = os.path.dirname(os.path.abspath(local_path)) or "."
    rc, _, err = runner(["copy", "--include", f"/{os.path.basename(local_path)}",
                         src_dir, _q(bucket, "jobs/bundles") + "/",
                         "--retries", "5"])
    return rc == 0, (err or "").strip()


def download_bundle(sha, local_path, *, runner=_default_runner, bucket=None):
    rc, _, err = runner(["copyto", _q(bucket, f"jobs/bundles/{sha}.tar.zst"), local_path])
    return rc == 0, (err or "").strip()


def write_ticket(ticket, *, runner=_default_runner, bucket=None):
    """Write the immutable ticket to jobs/queue/<box>/<job_id>.json. Returns
    (ok, key, err)."""
    box = ticket["box"]
    jid = ticket["job_id"]
    key = f"jobs/queue/{box}/{jid}.json"
    body = json.dumps(ticket, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _q(bucket, key)], input=body)
    return rc == 0, key, (err or "").strip()


def _ticket_identity(t: dict) -> dict:
    """The STABLE identity subset of a ticket — excludes the volatile
    `submitted_ts` + `actor` that `make_ticket` stamps fresh on every call, so
    two tickets minted for the same deterministic job_id at different times/by
    different actors are still comparable as "the same submission"."""
    return {
        "job_id": t.get("job_id"),
        "bundle_sha256": t.get("bundle_sha256"),
        "box": t.get("box"),
        "config": t.get("config"),
    }


def submit_with_id(job_id, config, box, *, bundle_sha256, actor=None,
                    runner=_default_runner, bucket=None) -> dict:
    """Idempotent submit for a DETERMINISTIC job_id (the workflow-controller
    submit path — distinct from `cmd_job_submit`'s random-nonce mint in
    herdd.py, which this never touches).

    Re-submitting the identical (job_id, bundle_sha256, box, config) is a
    no-op: no ticket write, no event, safe to retry from a crashed controller.
    Re-submitting the SAME job_id with DIFFERENT ticket bytes is a hard
    conflict (raises) — a deterministic id promises one fixed submission, so a
    changed body means the caller computed the id wrong, not that a resend is
    wanted. Never deletes or overwrites an existing ticket.
    """
    validate_job_id(job_id)
    actor = actor or _default_actor()
    existing = read_ticket(box, job_id, runner=runner, bucket=bucket)
    tk = make_ticket(job_id, bundle_sha256, actor, config, box)

    if existing is None:
        ok, key, err = write_ticket(tk, runner=runner, bucket=bucket)
        if not ok:
            raise JobmetaError(f"submit_with_id {job_id}: ticket write failed: {err}")
        emit_event(job_id, "submitted", actor=actor, runner=runner, bucket=bucket,
                   bundle_sha256=bundle_sha256, name=config.get("name"),
                   entrypoint=config.get("entrypoint"),
                   timeout_s=config.get("timeout_s"), box=box,
                   # how many results: globs the config declares — lets the
                   # durability classifier distinguish "declared nothing, empty
                   # manifest is fine" from "declared some, manifest empty =>
                   # HOLD" (parked_lifecycle). Absent on pre-2026-07-30 streams.
                   n_results_globs=len(config.get("results") or []),
                   # how hard the bid ladder may defend this job's box. Rides
                   # the `submitted` event for the same reason `timeout_s` does:
                   # the supervisor folds events, it never reads the bundle.
                   # Absent on pre-2026-08-14 streams -> the ladder derives.
                   defend=config.get("defend"),
                   # launch shape for readers that fold, never read the bundle
                   # (ls tags CPU-only jobs). Absent pre-2026-08-27 -> unknown.
                   gpu=bool((config.get("needs") or {}).get("gpu")))
        return {"status": "submitted", "job_id": job_id, "key": key}

    if _ticket_identity(existing) == _ticket_identity(tk):
        return {"status": "noop", "job_id": job_id}

    raise JobmetaError(
        f"deterministic JOB_ID {job_id} already submitted with different "
        f"ticket bytes (conflict)")


def delete_ticket(box, job_id, *, runner=_default_runner, bucket=None):
    """Remove a queue ticket. THE ONE delete in the job system — a ticket is a
    queue POINTER, not history (events/results/bundles are never deleted) — and
    leaving one behind would let its box double-run the job the moment it
    resumes. Returns (ok, err).

    "Used ONLY by retarget" was true when this was written and is not now. Four
    callers, all of them retiring a pointer whose box must not act on it again:
    `herdd cmd_job_retarget` (+ its stale-pointer sweep), `_job_cancel_writes`
    (cancel, and `job orphans --resolve` through it), `_retarget_pending_tickets`
    (fleetd's eviction/pull replacement — the one that moves tickets behind an
    operator's back), and `retarget_ticket`/`requeue_ticket` below. jobd is NOT
    among them and never will be: it only reads jobs/queue/<IID>/."""
    rc, _, err = runner(["deletefile", _q(bucket, f"jobs/queue/{box}/{job_id}.json")])
    return rc == 0, (err or "").strip()


# --- the dead-letter queue ---------------------------------------------------
# WHY THIS EXISTS (2026-08-26). jobd never deletes a ticket, so
# jobs/queue/<IID>/ accumulates every ticket ever submitted to that box. Nothing
# in ticket SELECTION ever read `submitted_ts` — it was written by
# `make_ticket` and read by no decision anywhere — so a ticket could sit
# unclaimed for a week and still be first in line. That is fine while the box
# is alive and working the queue in FIFO order. It stops being fine on the
# automatic paths: `_retarget_pending_tickets` (fleetd's eviction / pull-condemn
# replacement) moves EVERY key in the prefix to the successor box with no status
# and no age filter, and the successor is a fresh container whose
# $STATE_DIR/<jid>.terminal breadcrumbs do not exist. The only thing standing
# between a week-old ticket and a fresh GPU run is the per-job
# `results.DONE.json` probe — which a job that failed BEFORE its entrypoint ran
# never wrote.
#
# MEASURED on our own bucket the day this landed: of 34 stuck orphan tickets,
# 30 had no DONE marker, and 28 of those were an `a5t-cap` sweep whose work had
# already completed 20/20 arms elsewhere. Moving them onto a live box would have
# re-run a finished sweep on rented hardware.
#
# The DLQ is the retirement home for such a ticket. It is a MOVE, not a delete:
# `job cancel` deletes the queue pointer, and the pointer is the only place the
# frozen `config` (the whole resolved env the job would run with) is recorded —
# the `submitted` event carries `bundle_sha256`/`entrypoint`/`timeout_s` and NOT
# the env. So cancelling a ticket destroys the only record of what it would have
# done. Dead-lettering preserves it, under a key nothing claims from, with the
# reason attached and a deliberate `restore` path back.
DLQ_PREFIX = "jobs/dlq"

# Ticket field stamped by `write_dlq_entry`: why this ticket was retired, by
# whom, and what it was when it died. Its presence is what makes an entry a
# dead letter rather than a queue pointer that wandered into the wrong prefix.
DEAD_LETTER_MARK = "dead_letter"

# Default staleness bound for the REVIVAL paths (retarget / requeue / the bulk
# move). Not a TTL on the queue: nothing expires, nothing is swept, and a live
# box works its own backlog at any age. It is the age past which moving a ticket
# onto a *different* box needs an operator to say so out loud. Three days is
# chosen to sit above any real serial queue we run (the longest measured arm
# chain is well under a day) and below the week that the a5t-cap leftovers had
# been sitting.
STALE_TICKET_DAYS = 3.0


def dlq_key(box, job_id) -> str:
    """The dead-letter key for one ticket. Mirrors the queue layout exactly, so
    a restore is a move back with the box component unchanged."""
    return f"{DLQ_PREFIX}/{box}/{job_id}.json"


def ticket_age_days(ticket, *, now=None):
    """Age of a ticket in days from its `submitted_ts`, or None if unparseable.

    The first reader of `submitted_ts` in the system's history — it has been
    written into every ticket since the schema was minted and consulted by no
    decision until the DLQ. `now` is injected (a ts string or None) so the
    callers and tests share one clock and nothing here reads the wall clock
    twice in a comparison."""
    t0 = runmeta._ts_epoch(str(ticket.get("submitted_ts") or ""))
    if t0 is None:
        return None
    t1 = runmeta._ts_epoch(now or runmeta.now_ts())
    if t1 is None:
        return None
    return (t1 - t0) / 86400.0


def ticket_staleness(ticket, *, now=None, max_age_days=None):
    """PURE. `(stale, why)` for one ticket on a REVIVAL path.

    `stale=True` means "do not move this onto a box without an operator saying
    so"; it is never a claim that the work is unwanted, only that its age has
    outrun the window in which moving it is obviously right. An unparseable or
    absent `submitted_ts` is NOT stale — fail OPEN here on purpose: the bound
    exists to catch forgotten backlog, and turning a malformed timestamp into a
    refusal would break recovery of exactly the tickets most worth recovering.
    The caller that cares about provenance should say so separately."""
    limit = STALE_TICKET_DAYS if max_age_days is None else float(max_age_days)
    age = ticket_age_days(ticket, now=now)
    if age is None:
        return False, "submitted_ts absent or unparseable — age unknown, not gating"
    if age <= limit:
        return False, f"age {age:.1f}d within the {limit:g}d revival bound"
    return True, (f"ticket is {age:.1f}d old (bound {limit:g}d) — it was frozen at "
                  f"submit and carries that day's config and bundle_sha256, not "
                  f"today's bundle")


def bulk_move_verdict(ticket, status, *, now=None, max_age_days=None):
    """PURE. `(move, why)` for ONE ticket on a BULK move — fleetd's eviction /
    pull-condemn replacement and the handoff cutover, the paths that relocate a
    whole queue prefix without an operator in the loop.

    Two refusals, for two different reasons:

    * TERMINAL — a job that already reached done/failed/cancelled needs no
      successor. Today it is moved anyway and then skipped box-side by the
      `results.DONE.json` probe, which is a guard the mover cannot see and which
      a job that failed BEFORE its entrypoint ran never wrote. Filtering here
      closes that hole at the source instead of relying on a marker that is
      sometimes absent.
    * STALE — the ticket is frozen: it carries the config and bundle_sha256 of
      the day it was submitted, not today's bundle. Relocating a forgotten
      week-old ticket onto fresh hardware spends money on bytes nobody has
      looked at since. The operator can still move it deliberately.

    A skipped ticket is not destroyed — it stays in the old box's queue and
    surfaces as an orphan once that box is gone, which is the reviewable end
    state. `unknown` status moves: fail OPEN, since an unreadable event log
    must never silently abandon live work."""
    st = str(status or "unknown").lower()
    if st in TERMINAL:
        return False, f"job is already {st} — a finished job needs no successor"
    stale, why = ticket_staleness(ticket, now=now, max_age_days=max_age_days)
    if stale:
        return False, why
    return True, f"status {st}, {why}"


def write_dlq_entry(ticket, *, reason, actor=None, verdict=None, now=None,
                    runner=_default_runner, bucket=None):
    """Write a ticket to the dead-letter prefix, body preserved verbatim plus a
    `dead_letter` block. Returns (ok, key, err). Does NOT touch the queue — the
    caller composes (see `dead_letter_ticket`), so a failed DLQ write can never
    be mistaken for a retired ticket."""
    box, jid = ticket["box"], ticket["job_id"]
    entry = dict(ticket)
    entry[DEAD_LETTER_MARK] = {
        "ts": now or runmeta.now_ts(),
        "actor": actor,
        "reason": reason,
        "verdict": verdict,
        "source_key": f"jobs/queue/{box}/{jid}.json",
        "age_days_at_retirement": ticket_age_days(ticket, now=now),
    }
    key = dlq_key(box, jid)
    body = json.dumps(entry, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _q(bucket, key)], input=body)
    return rc == 0, key, (err or "").strip()


def read_dlq_entry(box, job_id, *, runner=_default_runner, bucket=None):
    """The dead-letter entry for one ticket, or None if there is none."""
    rc, out, _ = runner(["cat", _q(bucket, dlq_key(box, job_id))])
    if rc != 0 or not (out or "").strip():
        return None
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None


def list_dlq(box=None, *, runner=_default_runner, bucket=None):
    """(box, job_id) pairs in the dead-letter prefix. `box=None` lists all.

    Raises QueueUnreadable on a failed listing, for the same reason `list_queue`
    does: an unreadable bucket must never read as an empty DLQ."""
    if box is None:
        rc, out, err = runner(["lsf", "-R", _q(bucket, f"{DLQ_PREFIX}/")])
    else:
        rc, out, err = runner(["lsf", _q(bucket, f"{DLQ_PREFIX}/{box}/")])
    if rc != 0:
        raise QueueUnreadable(
            f"dead-letter listing FAILED (rc={rc}) — this is not an empty DLQ: "
            f"{(err or '').strip()[:400] or '<no stderr>'}")
    pairs = []
    for line in (out or "").splitlines():
        p = line.strip()
        if not p.endswith(".json"):
            continue
        p = p[:-5]
        if box is None:
            head, _, tail = p.partition("/")
            if tail:
                pairs.append((head, tail))
        else:
            pairs.append((str(box), p))
    return sorted(pairs)


def delete_dlq_entry(box, job_id, *, runner=_default_runner, bucket=None):
    """Remove a dead-letter entry (the restore path's last step). Returns
    (ok, err)."""
    rc, _, err = runner(["deletefile", _q(bucket, dlq_key(box, job_id))])
    return rc == 0, (err or "").strip()


def dead_letter_ticket(box, job_id, *, reason, actor=None, verdict=None,
                       now=None, runner=_default_runner, bucket=None) -> dict:
    """Retire a queue ticket into the DLQ. The ORDER is the safety property:

      1. read the ticket        — nothing to retire if it is already gone
      2. write the DLQ entry    — the frozen config is preserved BEFORE
                                  anything is destroyed
      3. delete the queue pointer

    A failure at step 2 leaves the queue untouched and returns `dlq_failed`, so
    the ticket is never destroyed without its record. This deliberately does NOT
    write a CANCEL marker or a `cancelled` event: retiring a POINTER is not the
    same act as ending a JOB, and conflating them is how a job that is still
    wanted somewhere else gets a sticky terminal event. Callers that mean
    "this job is over" call `_job_cancel_writes` as well — `job orphans
    --resolve` does exactly that."""
    src = read_ticket(box, job_id, runner=runner, bucket=bucket)
    if src is None:
        existing = read_dlq_entry(box, job_id, runner=runner, bucket=bucket)
        return {"status": "already_dead_lettered" if existing else "no_ticket",
                "job_id": job_id}
    ok, key, err = write_dlq_entry(src, reason=reason, actor=actor,
                                   verdict=verdict, now=now, runner=runner,
                                   bucket=bucket)
    if not ok:
        return {"status": "dlq_failed", "job_id": job_id, "err": err}
    del_ok, del_err = delete_ticket(box, job_id, runner=runner, bucket=bucket)
    return {"status": "dead_lettered", "job_id": job_id, "key": key,
            "ticket_deleted": del_ok,
            "delete_err": (None if del_ok else del_err)}


def restore_dlq_entry(job_id, box, new_box=None, *, actor=None,
                      runner=_default_runner, bucket=None) -> dict:
    """Put a dead-lettered ticket back on a queue — the deliberate inverse of
    `dead_letter_ticket`, and the reason the DLQ is a move rather than a delete.

    `new_box` defaults to the box the ticket died on, which is almost always
    wrong (that box is usually gone) — callers should pass a live one. The
    `dead_letter` block is stripped from the restored ticket but the retirement
    is left in the DLQ history by NOT deleting on a failed write."""
    entry = read_dlq_entry(box, job_id, runner=runner, bucket=bucket)
    if entry is None:
        return {"status": "no_entry", "job_id": job_id}
    target = str(new_box if new_box is not None else entry.get("box") or box)
    ticket = {k: v for k, v in entry.items() if k != DEAD_LETTER_MARK}
    if target != str(ticket.get("box")):
        ticket["retargeted_from"] = str(ticket.get("box"))
    ticket["box"] = target
    ok, key, err = write_ticket(ticket, runner=runner, bucket=bucket)
    if not ok:
        return {"status": "write_failed", "job_id": job_id, "err": err}
    emit_event(job_id, "retargeted", actor=actor, runner=runner, bucket=bucket,
               box=target, from_box=str(entry.get("box")),
               reason="dlq_restore")
    del_ok, del_err = delete_dlq_entry(box, job_id, runner=runner, bucket=bucket)
    return {"status": "restored", "job_id": job_id, "key": key, "box": target,
            "dlq_entry_deleted": del_ok,
            "delete_err": (None if del_ok else del_err)}


def retarget_ticket(job_id, old_box, new_box, *, actor=None,
                    runner=_default_runner, bucket=None) -> dict:
    """Move a queued ticket from old_box's queue to new_box's queue (the
    non-exiting core of herdd's `cmd_job_retarget`, for programmatic callers
    like the workflow controller's box-loss recovery). Rewrites `box` +
    `retargeted_from`, writes the new queue pointer, deletes the old, emits
    `retargeted`. The new box's jobd claims it fresh and a checkpointing job
    pulls its synced state back from jobs/<JOB_ID>/checkpoints/.

    A missing source ticket returns {'status': 'no_ticket'} rather than
    raising — on a controller retarget the ticket should exist, but a crash
    between box-launch and ticket-move must be a resumable no-op, not fatal.
    old_box == new_box is a no-op ({'status': 'noop'})."""
    old_box, new_box = str(old_box), str(new_box)
    if old_box == new_box:
        return {"status": "noop", "job_id": job_id}
    existing_new = read_ticket(new_box, job_id, runner=runner, bucket=bucket)
    src = read_ticket(old_box, job_id, runner=runner, bucket=bucket)
    if src is None:
        # already moved (idempotent resume) or never queued
        return {"status": "present" if existing_new else "no_ticket",
                "job_id": job_id}
    ticket = dict(src)
    ticket["box"] = new_box
    ticket["retargeted_from"] = old_box
    ok, key, err = write_ticket(ticket, runner=runner, bucket=bucket)
    if not ok:
        raise JobmetaError(f"retarget_ticket {job_id}: new ticket write failed: {err}")
    del_ok, del_err = delete_ticket(old_box, job_id, runner=runner, bucket=bucket)
    emit_event(job_id, "retargeted", actor=actor, runner=runner, bucket=bucket,
               box=new_box, from_box=old_box)
    return {"status": "retargeted", "job_id": job_id, "key": key,
            "old_ticket_deleted": del_ok, "delete_err": (None if del_ok else del_err)}


def requeue_ticket(job_id, new_box, config, bundle_sha256, *, old_box=None,
                   attempt=None, actor=None, runner=_default_runner,
                   bucket=None) -> dict:
    """Re-open a TERMINAL-FAILED job onto `new_box` under the SAME JOB_ID (the
    non-exiting core of herdd's `cmd_job_requeue`). The one-command form of the
    2026-07-30 manual recovery (dead-queue submit + checkpoint seed + retarget).

    Composed from the existing primitives, in this order:
      1. `make_ticket` — a FRESH queue ticket. The old one is consumed/absent by
         the time a job goes terminal, so unlike `retarget_ticket` (which MOVES an
         existing pointer) requeue must RECONSTRUCT one; `config`/`bundle_sha256`
         come from the caller, which is where the fail-closed sha check lives.
      2. `retargeted_from` — the SAME marker `retarget_ticket` writes, and the
         reason this is the retarget path rather than a fresh submit: jobd reads
         it off the ticket and pulls jobs/<JOB_ID>/checkpoints/ back on a fresh
         box whose restart_count is 0 (jobd.sh run_job_body, HANDOFF_DESIGN §4).
         So the B2 checkpoint seed of the manual recipe is free — same JOB_ID,
         same prefix, no copy.
      3. `requeued_ts` — the operator's explicit "run this again" on the ticket.
         An entrypoint-rc failure publishes results.DONE.json BEFORE it emits
         `failed`, and jobd skips any ticket whose DONE marker exists; this mark
         is what a requeue-aware jobd honours over that marker (exactly once —
         its own terminal breadcrumb is the latch).
      4. `resumed` — frozen vocabulary, NOT a new event kind (jobd.sh already
         emits `resumed` with kind=retarget for a cross-box continuation). Newer
         than the `failed`, so `fold_events` un-sticks the job — and newer BY
         CONSTRUCTION, not by hoping the clock ticked (see the ts step below).
      5. `delete_ticket` on the old box when one survived — same double-run
         hazard `retarget_ticket` guards against.

    Every B2 write is fail-closed except the old-ticket delete (reported, not
    raised — same contract as retarget). Caller MUST have verified the status and
    the bundle sha; this core does the writes, not the policy."""
    validate_job_id(job_id)
    new_box = str(new_box)
    old_box = str(old_box) if old_box else None
    actor = actor or _default_actor()

    # The `resumed` MUST sort strictly after the `failed` it un-sticks, and
    # `now_ts()` alone does not deliver that. Timestamps are millisecond
    # resolution, so a requeue issued in the same millisecond as the failure TIES
    # — and `_ev_order`'s (ts, nonce) tie-break then decides the un-stick on a
    # random nonce, i.e. a coin flip on whether the job re-opens at all. (Found
    # as an ~8% flake in test_requeue_ticket_reopens_the_fold, where the seeded
    # `failed` and the requeue land in the same millisecond ~16% of the time.)
    # Step past the newest event on record instead — one `lsf`, key names only.
    # This also settles the skewed-clock case in the right direction: a requeue
    # is the operator's LAST word by construction, never a race between their
    # clock and the box's.
    ts = now_ts()
    newest = latest_event_ts(job_id, runner=runner, bucket=bucket)
    if newest and newest >= ts:
        ts = ts_succ(newest)

    ticket = make_ticket(job_id, bundle_sha256, actor, config, new_box)
    if old_box:
        ticket["retargeted_from"] = old_box       # jobd checkpoint pull-back (2)
    ticket[REQUEUE_TICKET_MARK] = ts              # jobd DONE-marker override (3)
    ok, key, err = write_ticket(ticket, runner=runner, bucket=bucket)
    if not ok:
        raise JobmetaError(f"requeue_ticket {job_id}: ticket write failed: {err}")

    ev = emit_event(job_id, "resumed", actor=actor, runner=runner, bucket=bucket,
                    ts=ts,                        # provably newest — see above
                    kind="requeue", box=new_box, instance_id=new_box,
                    from_box=old_box or "-", retargeted_from=old_box or "-",
                    bundle_sha256=bundle_sha256,
                    **({"attempt": attempt} if attempt is not None else {}))
    if not ev.get("_emitted"):
        # The ticket is written but the fold still says `failed`: a box would run
        # the job while every reader called it dead. Loud, not silent.
        raise JobmetaError(
            f"requeue_ticket {job_id}: ticket written to jobs/queue/{new_box}/ but "
            f"the `resumed` event FAILED to emit ({ev.get('_error') or 'unknown'}) "
            f"— the fold is still stuck at `failed`; re-run the requeue")

    stale_deleted, stale_err = None, None
    if old_box and old_box != new_box and read_ticket(
            old_box, job_id, runner=runner, bucket=bucket) is not None:
        stale_deleted, stale_err = delete_ticket(old_box, job_id, runner=runner,
                                                 bucket=bucket)
    return {"status": "requeued", "job_id": job_id, "key": key, "box": new_box,
            "from_box": old_box, "requeued_ts": ts,
            "old_ticket_deleted": stale_deleted,
            "delete_err": (None if stale_deleted is not False else stale_err)}


def write_cancel_marker(job_id, *, actor=None, reason=None, runner=_default_runner,
                        bucket=None):
    """Write the tiny immutable CANCEL marker at jobs/<job_id>/CANCEL. A running
    box's jobd lsf-polls this object for each job it is running and, on seeing it,
    kills that job's process tree and records a terminal `cancelled` (the queue
    ticket deletion alone stops jobd RECLAIMING a job, but cannot stop an
    already-running entrypoint — this marker is how the cooperative kill lands).
    Best-effort, streamed via rcat (no HeadObject flake). Returns (ok, err)."""
    validate_job_id(job_id)
    body = json.dumps({
        "v": SCHEMA_VERSION, "ts": now_ts(),
        "actor": actor or _default_actor(),
        "reason": reason or "cancelled by operator",
    }, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _q(bucket, f"jobs/{job_id}/CANCEL")], input=body)
    return rc == 0, (err or "").strip()


def has_cancel_marker(job_id, *, runner=_default_runner, bucket=None) -> bool:
    """Does jobs/<job_id>/CANCEL exist? (The box-side jobd does this same probe in
    bash; exposed here for the laptop side + tests.)"""
    rc, out, _ = runner(["lsf", _q(bucket, f"jobs/{job_id}/CANCEL")])
    return rc == 0 and bool((out or "").strip())


def write_checkpoint_now_marker(job_id, *, actor=None, reason=None,
                                runner=_default_runner, bucket=None):
    """Write the CHECKPOINT_NOW flush marker at jobs/<job_id>/CHECKPOINT_NOW.

    Mirrors write_cancel_marker and rides the SAME box-side poll, but asks for the
    opposite of a stop: jobd fires one unfiltered checkpoint sync (whole declared
    glob, no --min-age), deletes the marker, and leaves the entrypoint running.

    Consume-and-delete is at-most-once by design — B2 has no CAS, so a lost delete
    race costs one extra flush and never a missed one.

    This CANNOT rescue an eviction: vast delivers no SIGTERM on a spot reclaim and
    the warning budget is single-digit seconds, far short of one marker poll. Use
    it before a park or a handoff. Returns (ok, err)."""
    validate_job_id(job_id)
    body = json.dumps({
        "v": SCHEMA_VERSION, "ts": now_ts(),
        "actor": actor or _default_actor(),
        "reason": reason or "checkpoint flush requested by operator",
    }, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _q(bucket, f"jobs/{job_id}/CHECKPOINT_NOW")],
                        input=body)
    return rc == 0, (err or "").strip()


def has_checkpoint_now_marker(job_id, *, runner=_default_runner, bucket=None) -> bool:
    """Is a flush still PENDING (the box has not consumed it yet)? Exposed so the
    CLI can report an un-consumed marker rather than silently overwriting it."""
    rc, out, _ = runner(["lsf", _q(bucket, f"jobs/{job_id}/CHECKPOINT_NOW")])
    return rc == 0 and bool((out or "").strip())


def read_ticket(box, job_id, *, runner=_default_runner, bucket=None):
    rc, out, _ = runner(["cat", _q(bucket, f"jobs/queue/{box}/{job_id}.json")])
    if rc != 0 or not (out or "").strip():
        return None
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None


def list_queue(box, *, runner=_default_runner, bucket=None) -> list[str]:
    """JOB_IDs queued for a box (jobs/queue/<box>/*.json).

    Raises QueueUnreadable when the listing itself fails. It used to answer `[]`,
    so a broken rclone config, a revoked key, a partition and a B2 outage all
    read as "this box has no work" — and fleetd stopped defending a box that had
    a live ticket (2026-08-22, box 48392137).
    """
    rc, out, err = runner(["lsf", _q(bucket, f"jobs/queue/{box}/")])
    if rc != 0:
        raise QueueUnreadable(
            f"queue listing for box {box} FAILED (rc={rc}) — this is not an "
            f"empty queue: {(err or '').strip()[:400] or '<no stderr>'}")
    return sorted(x.strip()[:-5] for x in (out or "").splitlines()
                  if x.strip().endswith(".json"))


def list_all_queued(*, runner=_default_runner, bucket=None) -> list[tuple[str, str]]:
    """(box, job_id) for every queued ticket (jobs/queue/<box>/<job_id>.json).

    Raises QueueUnreadable on a failed listing — same reason as `list_queue`."""
    rc, out, err = runner(["lsf", "-R", _q(bucket, "jobs/queue/")])
    if rc != 0:
        raise QueueUnreadable(
            f"fleet-wide queue listing FAILED (rc={rc}) — this is not an empty "
            f"queue: {(err or '').strip()[:400] or '<no stderr>'}")
    pairs = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line.endswith(".json"):
            continue
        parts = line.split("/")
        if len(parts) == 2:
            pairs.append((parts[0], parts[1][:-5]))
    return sorted(pairs)


def has_events(job_id, *, runner=_default_runner, bucket=None) -> bool:
    """Idempotency probe: does jobs/<job_id>/events/ already hold any object?
    jobd skips a ticket whose job already has events (survives daemon restarts)."""
    rc, out, _ = runner(["lsf", _q(bucket, f"jobs/{job_id}/events/")])
    return rc == 0 and bool((out or "").strip())


def event_cache_root(cache_dir=None) -> str:
    """The local mirror root of `jobs/` — `<XDG_CACHE_HOME>/vast-jobmeta` by
    default, `cache_dir` when a caller (or a test) pins one.

    ONE definition, because three readers derive paths under it and any drift
    between them silently splits the cache in half: `read_job`
    (`<root>/<job_id>/events/`), `read_box` (`<root>/nodes/<iid>/events/`) and
    `parked_lifecycle.job_ticket`, which re-reads `read_job`'s directory to get
    the raw bodies back. The layout deliberately MIRRORS the B2 key space with
    `jobs/` stripped, so a bulk `rclone copy b2:<bucket>/jobs/ <root>` lands
    every object in exactly the place the per-job readers already look.
    `vastlib.jobs.scan` depends on that property; do not reshape the tree
    without repointing all four."""
    return cache_dir or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "vast-jobmeta")


def event_cache_dir(job_id, cache_dir=None) -> str:
    """`read_job`'s per-job event directory under `event_cache_root`."""
    return os.path.join(event_cache_root(cache_dir), job_id, "events")


def read_job(job_id, *, runner=_default_runner, live_iids=(), bucket=None,
             cache_dir=None) -> dict:
    """Fold one job's event log into a view. Incremental local cache of the
    immutable event bodies (rclone copy), same as runmeta.read_run.

    ONE JOB PER CALL, ONE rclone SUBPROCESS PER CALL — that is the contract this
    function keeps for the box-side caller (jobd folds exactly one job and has
    no fleet to amortize over). It is also why calling it in a loop over a whole
    queue is a mistake: measured 2026-08-17 on 275 queued tickets, the loop cost
    138.1s of a 139.7s `job orphans`, ~0.5s of process-start + B2 auth + LIST per
    job against a cache that already held every body. Operator-side callers that
    need many jobs use `vastlib.jobs.scan.fold_many`, which learns the whole key
    set in ONE listing and fetches only what is missing."""
    validate_job_id(job_id)
    b = _bucket(bucket)
    dst = event_cache_dir(job_id, cache_dir)
    os.makedirs(dst, exist_ok=True)
    runner(["copy", f"b2:{b}/jobs/{job_id}/events/", dst,
            "--transfers", "16", "--checkers", "32", "--fast-list"])
    bodies = []
    try:
        for name in os.listdir(dst):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(dst, name), "rb") as fh:
                        bodies.append(fh.read())
                except OSError:
                    pass
    except OSError:
        pass
    return fold_events(bodies, live_iids)


def read_job_fresh(job_id, *, runner=_default_runner, live_iids=(),
                   bucket=None) -> dict:
    """`read_job` without the two things that can hide a late event, plus the one
    strongly-consistent probe that outranks the fold.

    A job event log is a set of immutable objects, so a fold can only ever be
    BEHIND — and during the 2026-07-30 frontier launch it was behind by minutes:
    jobs that had already failed on-box kept reporting `submitted live=False`, a
    snapshot that reads as "still queued on a dead box" and means "no box event
    has surfaced yet". Two contributors, both removed here:

      * `read_job` incrementally `rclone copy`s into a local cache with
         `--fast-list` (one recursive bucket listing). This reads each event key
         directly (`lsf` + per-key `cat`, `read_job_events`) and keeps nothing.
      * `view["live"]` is False whenever `instance_id` is None — which is ALWAYS
         true before a claim/started/heartbeat event folds, independent of the
         box. Callers must render that as n/a, not as False; `unclaimed` says so.

    `done_marker` is the part with real teeth: `jobs/<id>/results.DONE.json` is
    written LAST — so a `cat` hit is strong evidence the job finished even when
    no `done` event has surfaced. **Evidence about WHICH attempt, though**: the
    marker is a mutable key that survives a requeue, so it is dated here and
    graded by `classify_done_marker` into `done_marker_verdict`. A caller that
    reads the bare `done_marker` bool on a re-opened job is reading a dead
    attempt (see the block above `DONE_MARKER_CURRENT`).

    What this does NOT do: eliminate the LIST window. It narrows it.
    """
    view = fold_events(read_job_events(job_id, runner=runner, bucket=bucket),
                       live_iids)
    view["fresh"] = True
    view["unclaimed"] = view.get("instance_id") is None
    probe = probe_done_marker(job_id, reopened_at=view.get("reopened_at"),
                              runner=runner, bucket=bucket)
    view["done_marker"] = probe["present"]
    view["done_marker_ts"] = probe["ts"]
    view["done_marker_rc"] = probe["rc"]
    view["done_marker_box"] = probe["box"]
    view["done_marker_verdict"] = probe["verdict"]
    return view


def pull_results(job_id, dest, *, runner=_default_runner, bucket=None) -> list[str]:
    """Copy jobs/<job_id>/results/ -> dest. Returns the pulled file manifest."""
    b = _bucket(bucket)
    os.makedirs(dest, exist_ok=True)
    runner(["copy", f"b2:{b}/jobs/{job_id}/results/", dest,
            "--transfers", "16", "--checkers", "32", "--fast-list"])
    rc, out, _ = runner(["lsf", "-R", f"b2:{b}/jobs/{job_id}/results/"])
    if rc != 0:
        return []
    return sorted(x.strip() for x in (out or "").splitlines()
                  if x.strip() and not x.strip().endswith("/"))


def list_checkpoints(job_id, *, runner=_default_runner, bucket=None) -> list[str]:
    """List jobs/<job_id>/checkpoints/ — the prefix jobd's mid-run sync writes and
    a resuming box pulls back. Deliberately SEPARATE from `pull_results`, which
    reads jobs/<job_id>/results/ (a single new-object write at finalize).

    This exists so a reader can tell the two apart. A job that is still running,
    or was interrupted mid-run, has an EMPTY results/ and a fully populated
    checkpoints/ — so a 0-file `job pull` is the expected reading of a healthy
    job, not evidence of lost work. Probe, not validator: an unreadable listing
    returns [] rather than raising."""
    rc, out, _ = runner(["lsf", "-R", _q(bucket, f"jobs/{job_id}/checkpoints/")])
    if rc != 0:
        return []
    return sorted(x.strip() for x in (out or "").splitlines()
                  if x.strip() and not x.strip().endswith("/"))


def read_results_done(job_id, *, runner=_default_runner, bucket=None):
    """Tolerant read of the child's DONE marker (jobs/<job_id>/results.DONE.json,
    frozen key, JOBS_DESIGN.md). Returns the parsed dict, or None if the object
    is missing / empty / unparseable — this is a probe, not a validator."""
    rc, out, _ = runner(["cat", _q(bucket, f"jobs/{job_id}/results.DONE.json")])
    if rc != 0 or not (out or "").strip():
        return None
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# WHOSE ATTEMPT IS THE DONE MARKER?
#
# `results.DONE.json` is written on ANY terminal outcome — a run that exits
# rc!=0 publishes its partial results AND the marker BEFORE it emits `failed`
# (JOBS_DESIGN "jobd's DONE-marker override"). `job requeue`/`job retarget` then
# re-open the SAME JOB_ID and clear nothing: B2 events are append-only and the
# marker is a mutable key nobody rewrites until the next attempt finalizes. So
# for the whole life of the re-run, a marker describing the DEAD attempt sits at
# the live job's key.
#
# Read as "the job FINISHED" that is a false terminal on a healthy run. Measured
# live 2026-08-28 on 20260828T064840-v16-r64-8c87 (v16-r64): rc=3 on a resume
# guard, requeued, retargeted after a host_stop, and 42% into 273 steps at
# gpu_util 100% `job status --fresh` reported it finished. Its results/ held
# 57.7 KiB of the dead attempt's debris (no adapter, no optimizer.pt) while the
# live attempt's complete 2.1 GiB checkpoint-112 sat under checkpoints/. `job
# pull` would have handed over the debris and the arm would have been declared
# complete.
#
# The fix is to DATE the marker and compare it against the attempt boundary
# (`fold_events`'s `reopened_at`), which is why `verdict` is TRI-STATE: a marker
# that cannot be dated on a re-opened job is `unknown`, never `current`. Only a
# marker positively newer than the re-open may claim the job finished.
DONE_MARKER_CURRENT = "current"   # no re-open, or the marker postdates it
DONE_MARKER_STALE = "stale"       # written before the current attempt began
DONE_MARKER_UNKNOWN = "unknown"   # re-opened and the marker could not be dated


def _rclone_modtime_ts(mod):
    """rclone's RFC3339 ModTime -> a runmeta ts (`YYYYMMDDTHHMMSSmmmZ`, UTC), or
    None. Normalized so it orders lexicographically against event timestamps,
    which is the only comparison any caller makes."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})"
                 r"(?:\.(\d+))?\s*(Z|z|[+-]\d{2}:?\d{2})?$", str(mod or "").strip())
    if not m:
        return None
    y, mo, d, hh, mm, ss, frac, off = m.groups()
    try:
        dt = datetime.datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss),
                               tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    if off and off not in ("Z", "z"):
        sign = 1 if off[0] == "+" else -1
        oh, om = off[1:].replace(":", "")[:2], off[1:].replace(":", "")[2:4]
        dt -= datetime.timedelta(minutes=sign * (int(oh) * 60 + int(om or 0)))
    ms = int((frac or "0")[:3].ljust(3, "0"))
    return dt.strftime("%Y%m%dT%H%M%S") + f"{ms:03d}Z"


def results_done_mtime(job_id, *, runner=_default_runner, bucket=None):
    """B2 modification time of jobs/<job_id>/results.DONE.json as a runmeta ts,
    or None when the object is absent/unlistable/undatable.

    `lsjson` and not `lsf`/`lsl`: it is the only listing whose time field is
    unambiguously UTC and machine-shaped (`lsl` renders LOCAL time, which would
    silently mis-order the comparison by the workstation's offset). A probe, not
    a validator — every failure is None, and None is what routes a caller to the
    fail-closed `unknown` verdict rather than to a claim."""
    try:
        rc, out, _ = runner(["lsjson", _q(bucket, f"jobs/{job_id}/results.DONE.json")])
    except Exception:
        return None
    if rc != 0 or not (out or "").strip():
        return None
    try:
        rows = json.loads(out)
    except (ValueError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("ModTime"):
            ts = _rclone_modtime_ts(r.get("ModTime"))
            if ts:
                return ts
    return None


def done_marker_ts(marker, mtime_ts=None):
    """PURE. When was this DONE marker written? Its own `written_ts` stamp if it
    carries one, else the B2 mtime the caller looked up, else None.

    The stamp is preferred because it is the marker DESCRIBING ITSELF (jobd
    writes it in the same pass that builds the body) while an mtime describes the
    object — they agree in practice, and where they cannot, self-description
    wins. Markers written before 2026-08-28 carry no stamp at all, which is
    exactly why the mtime fallback exists and why neither may be required."""
    if isinstance(marker, dict):
        stamp = marker.get("written_ts")
        if isinstance(stamp, str) and TS_RE.match(stamp.strip()):
            return stamp.strip()
    return mtime_ts or None


def classify_done_marker(marker, marker_ts, reopened_at):
    """PURE. Does this DONE marker describe the CURRENT attempt? One of
    DONE_MARKER_{CURRENT,STALE,UNKNOWN}, or None when there is no marker.

    `reopened_at` is `fold_events`'s attempt boundary (None = the job was never
    re-opened, so there is only one attempt and the marker is necessarily its).
    A re-opened job needs a positive date to claim `current`: an undatable marker
    is `unknown`, because "we could not tell" and "it finished" are the two
    answers this whole seam exists to keep apart."""
    if marker is None:
        return None
    if not reopened_at:
        return DONE_MARKER_CURRENT
    if not marker_ts:
        return DONE_MARKER_UNKNOWN
    # Both are fixed-width UTC stamps: ordering is lexicographic by construction
    # (runmeta's `_ts_epoch` docstring states the same rule for event streams).
    return DONE_MARKER_CURRENT if marker_ts > reopened_at else DONE_MARKER_STALE


def probe_done_marker(job_id, *, reopened_at=None, runner=_default_runner,
                      bucket=None) -> dict:
    """Read the DONE marker and date it against the attempt boundary.

    `{present, ts, rc, box, verdict}`. The mtime lookup is issued ONLY for a
    re-opened job that has a marker — the single case where the answer can
    change — so the common path costs exactly the one `cat` it always did."""
    marker = read_results_done(job_id, runner=runner, bucket=bucket)
    out = {"present": marker is not None, "ts": None, "rc": None, "box": None,
           "verdict": None}
    if marker is None:
        return out
    if isinstance(marker, dict):
        out["rc"] = _num(marker.get("rc"))
        if marker.get("instance_id") is not None:
            out["box"] = str(marker.get("instance_id"))
    mtime = None
    ts = done_marker_ts(marker)
    if ts is None and reopened_at:
        mtime = results_done_mtime(job_id, runner=runner, bucket=bucket)
    out["ts"] = done_marker_ts(marker, mtime)
    out["verdict"] = classify_done_marker(marker, out["ts"], reopened_at)
    return out


# Arm-file sha re-read budgets for validate_generation_artifact. DEFENSE-IN-DEPTH:
# the ROOT CAUSE of the four false-fails (2026-07-15/16/19/20) was jobd's mid-run
# checkpoint loop OVERWRITING jobs/<id>/results/gens_*.jsonl (the e2 gen bundles
# list the arm files in BOTH `checkpoints:` and `results:` globs) before finalize
# overwrote the SAME key — and B2 object OVERWRITES are eventually consistent, so a
# `cat` moments after finalize could still return the STALE early empty/partial
# version (the arm read as sha-of-empty ~3s after results.DONE.json → false
# ARTIFACT_INVALID killed a fully-successful E2 generate stage). That is now fixed
# on the WRITE side: jobd's mid-run sync goes to jobs/<id>/checkpoints/, so this
# results/ key is written exactly ONCE at finalize as a NEW object (strong
# read-after-write). This retry-on-mismatch stays as a backstop — it still covers
# the rare preempt-then-resume path (the N1b results-glob preempt flush can put an
# earlier version at results/ before the resumed run re-publishes) and any future
# re-introduction of a results/ overwrite.
#
# 2026-07-19: retry-on-mismatch alone LOST the race a third time (run
# 1d82: the stale-empty read outlived the full ~158s budget; content surfaced
# ~90s after the validator gave up). The PRIMARY fix is now ORDERING, box-side:
# jobd's publish path ("publish verify" in onstart/jobd.sh) reads every
# published result back from B2 and requires its sha256 to match the local
# file BEFORE results.DONE.json is written — so by the time this validator can
# even start (it keys off DONE), each declared arm is readable at its final
# bytes. The budgets below remain as DEFENSE-IN-DEPTH only (pre-fix boxes,
# publish_verify_failed exhaustion) — never lengthen them again to chase this
# race.
#
# TWO budgets, split on WHAT the mismatching read contained (live recurrence
# 2026-07-16, E2 run 0d9d: the stale EMPTY checkpoint version persisted PAST
# the single ~18s linear budget and a fully-valid generate failed
# ARTIFACT_INVALID -> RETRY_EXHAUSTED):
#   * EMPTY/SHORT body (0 bytes, or fewer lines than the manifest arm's
#     declared `rows`) — a not-yet-consistent checkpoint version, never real
#     corruption (the finalize publish that minted the declared sha is
#     strictly longer). Waited out on the LONG exponential schedule
#     (2,4,8,16,32,32,32,32s = 158s worst case — overwrite windows can be
#     tens of seconds).
#   * NON-empty body at/above the declared rows whose sha simply differs —
#     genuine corruption; keeps failing FAST on the short linear schedule so
#     a real bad artifact isn't masked for minutes.
ARM_SHA_RETRIES = 3          # non-empty-mismatch re-reads AFTER the first read
ARM_SHA_BACKOFF_S = 3.0      # linear: 3s, 6s, 9s (~18s total)
ARM_SHA_STALE_RETRIES = 8            # empty/short re-reads AFTER the first read
ARM_SHA_STALE_BACKOFF_S = 2.0        # exponential base: 2, 4, 8, 16, 32, ...
ARM_SHA_STALE_BACKOFF_CAP_S = 32.0   # ... capped (sum of the 8 sleeps = 158s)


def _arm_read_is_stale(body, declared_rows) -> bool:
    """True iff a sha-mismatching arm read looks like an eventual-consistency
    STALE version rather than genuine corruption — an EMPTY body, or (when the
    manifest arm declares `rows`) a body with fewer lines than declared. Routes
    the read to the long ARM_SHA_STALE_* budget instead of the fast-fail one."""
    if not body:
        return True
    return bool(declared_rows) and len(body.splitlines()) < declared_rows


def _results_rel(p) -> str:
    """Normalize an arm-file path to the `results/<rest>` suffix the box's
    `.uploaded` list (and thus a `publish_verified` event's `files`) carries, so a
    manifest that records an ABSOLUTE box path
    (`/workspace/jobs/<id>/work/results/gens_a.jsonl`) and one that records the
    workdir-relative form (`results/gens_a.jsonl`) both collapse to the same key.
    Falls back to the basename when no `results/` component is present."""
    p = (p or "").strip().replace("\\", "/").lstrip("/")
    idx = p.find("results/")
    if idx != -1:
        return p[idx:]
    return os.path.basename(p)


def _parse_files_field(raw):
    """The verified-files list a box stamps on `publish_verified`. Accepts a JSON
    list (already parsed by the fold, or a JSON-string), or the comma-joined
    string the `emit --field files=a,b` convention produces. Anything else -> []."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if x is not None]
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
        except (ValueError, TypeError):
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def latest_event_ts(job_id, *, runner=_default_runner, bucket=None) -> str | None:
    """Newest event timestamp in jobs/<job_id>/events/, read from the KEY NAMES
    alone — ONE `lsf`, zero body reads, because the key already carries it
    (`<ts>-<actor>-<nonce>.json`, `runmeta.event_key`). Returns None when the
    listing is empty or unreadable; like `read_job_events`, a transport failure
    is a safe negative, never an exception.

    This is a strictly cheaper probe than folding the log, and the only thing a
    writer needs in order to place itself after every event on record."""
    try:
        rc, out, _ = runner(["lsf", _q(bucket, f"jobs/{job_id}/events/")])
    except Exception:
        return None
    if rc != 0:
        return None
    best = None
    for name in (out or "").splitlines():
        name = name.strip()
        if not name.endswith(".json"):
            continue
        ts = name.split("-", 1)[0]
        if TS_RE.match(ts) and (best is None or ts > best):
            best = ts
    return best


def read_job_events(job_id, *, runner=_default_runner, bucket=None) -> list:
    """Best-effort list+read of the RAW job event objects (jobs/<id>/events/),
    returning parsed event dicts (unfolded — `fold_events` drops unknown event
    types like `publish_verified`). Resilient by contract: an unreadable/absent
    listing, a transport error, or an unparseable body yields fewer/zero events,
    never an exception — so a caller can treat 'no events' as a safe negative."""
    try:
        rc, out, _ = runner(["lsf", _q(bucket, f"jobs/{job_id}/events/")])
    except Exception:
        return []
    if rc != 0 or not (out or "").strip():
        return []
    evs = []
    for name in (out or "").splitlines():
        name = name.strip()
        if not name.endswith(".json"):
            continue
        try:
            rc2, body, _ = runner(["cat", _q(bucket, f"jobs/{job_id}/events/{name}")])
        except Exception:
            continue
        if rc2 != 0 or not (body or "").strip():
            continue
        try:
            ev = json.loads(body)
        except (ValueError, TypeError):
            continue
        if isinstance(ev, dict):
            evs.append(ev)
    return evs


def _publish_verified_files(events):
    """Union of the `results/`-relative file suffixes covered by every
    `publish_verified` event in `events`, or None when the job has emitted no
    such event (older bundle, or publish-verify failed/absent -> no positive
    signal). A `publish_verify_failed` event is NOT a positive signal and is
    ignored here (the backstop handles that job)."""
    covered = None
    for e in events:
        if not isinstance(e, dict) or e.get("event") != "publish_verified":
            continue
        covered = (covered or set()) | {
            _results_rel(n) for n in _parse_files_field(e.get("files"))}
    return covered


def validate_generation_artifact(job_id, *, expect_kind, runner=_default_runner,
                                  bucket=None,
                                  manifest_path="results/artifact-manifest.json",
                                  sleep_fn=None) -> dict:
    """Controller's generate->score artifact acceptance (M2-T2, roadmap
    'Artifact binding'): reads the child's DONE marker, then downloads and
    validates its generation artifact manifest — schema version, `kind`, and
    every declared arm file's sha256. The returned `manifest_sha256` (sha of the
    raw manifest bytes) is what feeds the downstream asset cache name
    (`workflowmeta.input_ref_asset`). Raises JobmetaError on any acceptance
    failure — never returns a partially-validated result.

    `manifest_path` is WORKDIR-RELATIVE, the same frame as a bundle's
    `results:` globs: results land on B2 under
    `jobs/<job_id>/results/<workdir-relative-path>`, so the manifest is read
    at `jobs/<job_id>/results/` + `manifest_path`. The default matches the
    e2 bundles (which write into a local `results/` dir — hence the
    double `results/results` in the final key); a workflow stage's
    `ArtifactContract.manifest_path` is threaded here by
    `workflowctl._accept_stage_artifacts`.

    PRIMARY acceptance path: when the job carries a `publish_verified` event
    (the box's own publish-verify read each arm back from B2 and confirmed
    sha==local==manifest-declared before writing DONE) that COVERS every declared
    arm, the controller skips the per-arm body re-download entirely — validating
    manifest STRUCTURE only — because a controller re-read is redundant and races
    B2 cross-client read-after-write consistency (the 5x false-fail). Bundles
    without that event fall back to the per-arm re-hash loop below.

    In that fallback, a declared-vs-read sha MISMATCH is re-read before raising —
    B2 overwrite eventual-consistency can serve a stale pre-finalize arm version
    (see the constants' comment) — on one of TWO budgets picked per-read by
    `_arm_read_is_stale`: an EMPTY/SHORT body (always a stale checkpoint
    version, never corruption) waits out the long ARM_SHA_STALE_* exponential
    schedule (~158s), while a full-length wrong-bytes body exhausts only the
    fast ARM_SHA_RETRIES linear schedule (~18s) so genuine corruption isn't
    masked for minutes. A settled object matches on the first read (zero
    retries, byte-identical to the pre-retry behavior). `sleep_fn` defaults to
    `time.sleep` (injectable, same seam as workflowctl's `run_controller`)."""
    done = read_results_done(job_id, runner=runner, bucket=bucket)
    if done is None:
        raise JobmetaError(f"job {job_id}: results.DONE.json missing or unreadable")
    rc_field = done.get("rc")
    if rc_field is not None and rc_field != 0:
        raise JobmetaError(f"job {job_id}: results.DONE.json reports rc={rc_field!r}")

    key = f"jobs/{job_id}/results/{manifest_path}"
    rc, out, _ = runner(["cat", _q(bucket, key)])
    if rc != 0 or not (out or "").strip():
        raise JobmetaError(f"job {job_id}: artifact manifest missing")

    manifest_sha256 = hashlib.sha256(out.encode("utf-8")).hexdigest()
    try:
        manifest = json.loads(out)
    except ValueError:
        raise JobmetaError(f"job {job_id}: artifact manifest is not valid JSON")

    if manifest.get("v") != 1:
        raise JobmetaError(f"job {job_id}: manifest v!={manifest.get('v')!r} (want 1)")
    if manifest.get("kind") != expect_kind:
        raise JobmetaError(
            f"job {job_id}: manifest kind {manifest.get('kind')!r} != {expect_kind!r}")

    arms = manifest.get("arms") or {}

    # --- box publish-verify trust gate (THE 5x cross-client false-fail fix) -----
    # jobd's box-side "publish verify" already reads every uploaded result BACK
    # from B2 and requires sha256 == the local file == the manifest-declared sha
    # BEFORE writing results.DONE.json, then stamps a `publish_verified` event
    # listing the verified `results/…` paths. When that event covers every arm
    # here, the controller RE-downloading each 3.8 MB arm body to re-hash it is
    # both redundant (durability + correctness already proven at the source) and
    # ACTIVELY HARMFUL: the controller is a different B2 client/edge and its
    # re-read races B2 cross-client read-after-write consistency — it read a
    # genuinely-durable arm as empty (sha-of-empty) past the whole ~158s retry
    # budget and false-failed five fully-successful generate stages
    # (2026-07-15/16/19/20 + run 20260720T004614-e2-paired-55ca, whose OWN
    # box-side publish-verify PASSED). So: trust the positive box signal and skip
    # the body re-read; still enforce manifest STRUCTURE below. If there is no
    # `publish_verified` event (pre-fix bundle, or publish-verify failed/absent),
    # fall back to the per-arm re-download+re-hash loop as defense-in-depth.
    events = read_job_events(job_id, runner=runner, bucket=bucket)
    verified_files = _publish_verified_files(events)
    box_verified = False
    if verified_files is not None:
        want = {_results_rel(a.get("path")) for a in arms.values() if a.get("path")}
        # fully-trusted only when the box verified EVERY declared arm; a coverage
        # gap (an arm the box never confirmed) falls through to the backstop.
        box_verified = bool(want) and want <= verified_files

    if box_verified:
        for arm_name, arm in arms.items():
            p = arm.get("path")
            want_sha = arm.get("sha256")
            if not p:
                raise JobmetaError(
                    f"job {job_id}: arm {arm_name!r} missing/empty path")
            if not want_sha:
                raise JobmetaError(
                    f"job {job_id}: arm {arm_name!r} missing/empty sha256")
            for cnt in ("rows", "unique_ids"):
                if cnt in arm and not (isinstance(arm[cnt], int) and arm[cnt] > 0):
                    raise JobmetaError(
                        f"job {job_id}: arm {arm_name!r} {cnt}={arm[cnt]!r} "
                        f"not a positive int")
        return {"manifest_sha256": manifest_sha256, "manifest": manifest,
                "kind": manifest.get("kind")}

    sleep_fn = sleep_fn or time.sleep
    for arm_name, arm in arms.items():
        p = arm.get("path")
        want = arm.get("sha256")
        if not p or not want:
            continue
        declared_rows = arm.get("rows")
        stale_reads = mismatch_reads = 0
        while True:
            rc2, body, _ = runner(["cat", _q(bucket, f"jobs/{job_id}/results/{p}")])
            if rc2 != 0:
                raise JobmetaError(f"job {job_id}: declared arm file {p!r} missing")
            got = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if got == want:
                break
            if _arm_read_is_stale(body, declared_rows):
                stale_reads += 1
                if stale_reads > ARM_SHA_STALE_RETRIES:
                    raise JobmetaError(
                        f"job {job_id}: arm {arm_name!r} file {p!r} still "
                        f"empty/short of its declared content after "
                        f"{ARM_SHA_STALE_RETRIES} re-reads "
                        f"(declared {want[:12]} got {got[:12]})")
                sleep_fn(min(ARM_SHA_STALE_BACKOFF_S * 2 ** (stale_reads - 1),
                             ARM_SHA_STALE_BACKOFF_CAP_S))
            else:
                mismatch_reads += 1
                if mismatch_reads > ARM_SHA_RETRIES:
                    raise JobmetaError(
                        f"job {job_id}: arm {arm_name!r} file {p!r} sha256 mismatch "
                        f"(declared {want[:12]} got {got[:12]}, "
                        f"persisted across {ARM_SHA_RETRIES} re-reads)")
                sleep_fn(ARM_SHA_BACKOFF_S * mismatch_reads)

    return {"manifest_sha256": manifest_sha256, "manifest": manifest,
            "kind": manifest.get("kind")}


# --- asset B2-staleness preflight (submit-time drift guard, GAP 1) -----------
# The failure this closes (docs/plans/spot-resilient-eval-jobs.md, "Live-run
# results", 2026-07-12): an `assets:` entry pulls the LIVE B2 copy of a runset,
# but the operator edited the runset locally and never re-`build.sh`/re-staged —
# so the box ran a stale entrypoint (a pre-EVAL_ONLY train.sh that started
# TRAINING instead of the eval). The local rehearsal uses LOCAL fixtures and is
# structurally blind to B2 drift; this is the ONLY point that reads B2 and sees it.
#
# Signal = byte-identity of a SINGLE sentinel file (the runset's entrypoint,
# train.sh), local vs B2, via one `rclone cat` of a few-KB object — NOT a tree
# hash, NOT an mtime compare. Rationale (why it won't false-positive on
# immutable assets):
#   * Only MUTABLE prefixes are checked (runsets/*). base-models/, checkpoints/,
#     eval-env/ are content-addressed / write-once — never inspected, so an
#     immutable asset can never trip the warning.
#   * Even within runsets/*, IDENTICAL bytes are silent regardless of mtimes, so
#     a fresh `git clone` (which resets every file mtime to checkout time — the
#     reason an mtime compare WOULD false-positive) stays silent as long as the
#     staged bytes match what you hold locally.
# It answers exactly one question — "will the box run different bytes than the
# source I have here?" — which is the corruption the incident was.
MUTABLE_ASSET_PREFIXES = ("runsets/",)
# Candidate sentinel filenames, in priority order. The FIRST that exists in the
# local source is the one compared. A runset's entrypoint (train.sh, the file the
# box actually execs) is first, so an unstaged code change surfaces immediately.
ASSET_SENTINELS = ("train.sh", "run.sh", "entrypoint.sh", "MANIFEST.json")


def local_source_for_asset(b2_prefix, repo_root):
    """Best-effort map a MUTABLE B2 asset prefix back to the local directory that
    mirrors it, or None when the prefix is immutable-ish / unmapped (=> not
    checked). Convention (stage_run.sh / a runset build.sh): b2 'runsets/<name>'
    is synced from tools/vast/runsets/<name>/_build (the dir build.sh assembles),
    falling back to the runset dir itself when no _build is present."""
    p = (b2_prefix or "").strip("/")
    if not any((p + "/").startswith(m) for m in MUTABLE_ASSET_PREFIXES):
        return None
    if p.startswith("runsets/"):
        name = p[len("runsets/"):].split("/", 1)[0]
        if not name:
            return None
        base = os.path.join(repo_root, "tools", "vast", "runsets", name)
        build = os.path.join(base, "_build")
        if os.path.isdir(build):
            return build
        if os.path.isdir(base):
            return base
    return None


# --- tracked (declared-provenance) staleness --------------------------------
# The sentinel heuristic above answers "did the runset's ENTRYPOINT drift?".
# It cannot answer "did the PAYLOAD drift?", and on 2026-07-31 that gap shipped
# a half-stale trainer: b2:runsets/witness-lifter/train_proposer_lora.py held
# 64,593 B while tools/pipeline/ml_infra/train_proposer_lora.py at HEAD was
# 125,307 B (many commits, including a --quant semantics change). The sentinel
# (train.sh) was identical on both sides, so the preflight printed nothing — a
# GREEN light on a box about to run ~40 min-old-at-best code for 8 hours. Worse,
# the local dir the sentinel compares against (`runsets/<name>/_build`) is
# GITIGNORED: on a fresh clone it does not exist (=> 'skipped'), and here it held
# a copy staler than B2's.
#
# The fix is provenance, not cleverness: a `tracks:` mapping names the repo file
# each staged object mirrors, and this compares them. Absent `tracks:` nothing
# happens — a B2-native corpus can never trip it.
# WHICH hash: the remote decides, so ASK rather than assume. Our `b2:` remote is
# configured as rclone `type = s3` against B2's S3 endpoint (see b2_sync.sh), and
# the s3 backend serves MD5 (the ETag) while the NATIVE b2 backend serves SHA-1
# and refuses md5. Hard-coding either one silently degrades the check to a
# size-only compare on the other — measured 2026-07-31: `hashsum sha1` against
# the live remote answers "hash unsupported: hash type not supported". So try in
# order and take the first that answers. Either way it is a METADATA read: no
# download, no egress.
_HASH_ALGOS = (("md5", re.compile(r"^[0-9a-f]{32}$")),
               ("sha1", re.compile(r"^[0-9a-f]{40}$")))


def _b2_object_fingerprint(key, *, runner, bucket):
    """(kind, value, err) for one B2 object, download-free:

      ('md5'|'sha1', <hex>, None)  the remote's stored digest.
      ('size', <int>,        None) fallback when NO hash is available — e.g. a
                                   multipart upload, whose S3 ETag is a hash of
                                   part hashes and which rclone therefore
                                   reports as no hash at all.
      (None,   None,        '...') unreadable: creds absent, blip, or no object.
    """
    hash_err = "remote reported no usable hash for this object"
    for algo, pat in _HASH_ALGOS:
        try:
            rc, out, err = runner(["hashsum", algo, _q(bucket, key)])
        except Exception as e:
            return None, None, f"B2 read error: {e}"
        if rc == 0:
            tok = (out or "").strip().split()
            if tok and pat.match(tok[0].lower()):
                return algo, tok[0].lower(), None
        elif "not found" in ((err or "") + (out or "")).lower():
            hash_err = (err or "").strip() or "object not found"
            break                          # a missing object: stop probing algos
        else:
            hash_err = (err or "").strip() or f"B2 hashsum {algo} failed"
    try:                                   # degrade to size, still no download
        rc, out, err = runner(["size", "--json", _q(bucket, key)])
        if rc == 0 and (out or "").strip():
            n = json.loads(out).get("bytes")
            if isinstance(n, (int, float)) and n >= 0:
                return "size", int(n), None
    except Exception:
        pass
    return None, None, hash_err


def _local_fingerprint(path, algos=("md5", "sha1")):
    """({algo: hex}, size) for a local file, or (None, None) if unreadable. Every
    candidate digest in ONE pass — the file is read once whichever algo the
    remote turns out to speak."""
    hs = {a: hashlib.new(a) for a in algos}
    n = 0
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                for h in hs.values():
                    h.update(chunk)
                n += len(chunk)
    except OSError:
        return None, None
    return {a: h.hexdigest() for a, h in hs.items()}, n


def restage_hint(key, local_rel):
    """The exact command that re-stages `key` — named in the failure message so
    the operator never has to go find it. A runset with a build.sh gets that
    (it regenerates the corpus too); anything else gets the one-object copy."""
    parts = key.split("/")
    if len(parts) >= 2 and parts[0] == "runsets":
        return f"bash tools/vast/runsets/{parts[1]}/build.sh"
    return f"rclone copyto {local_rel} b2:$B2_BUCKET/{key}"


def collect_tracked(cfg) -> dict:
    """Flatten every provenance declaration in a validated job config into
    {full B2 key: repo-relative source path} — top-level `tracks:` plus each
    asset's `tracks:` (rebased onto that asset's `b2:` prefix). Pure."""
    out = {}
    for k, v in (((cfg or {}).get("tracks")) or {}).items():
        out[str(k).strip("/")] = str(v)
    for a in ((cfg or {}).get("assets") or []):
        prefix = (a.get("b2") or "").strip("/")
        for k, v in (a.get("tracks") or {}).items():
            key = f"{prefix}/{str(k).strip('/')}" if prefix else str(k).strip("/")
            out[key] = str(v)
    return out


def check_tracked_assets(tracked, *, repo_root, runner=_default_runner, bucket=None,
                         asset_names=None):
    """Compare each DECLARED-provenance B2 object against the repo file it
    mirrors. Returns findings shaped like `check_asset_staleness`' (plus
    kind='tracks', local=<repo-relative path>):

      status 'ok'      remote == local (sha1, or size when no remote hash)
             'stale'   they differ                     -> REFUSES a submit
             'broken'  the declared local file is absent from the checkout
                       (the declaration is wrong)      -> REFUSES a submit
             'unknown' remote unreadable (no creds offline, blip, absent object)
                       -> a NOTE; never blocks, so the portable lane and an
                          offline laptop are unaffected.

    Injected-runner pure; never raises; never writes to B2."""
    findings = []
    names = asset_names or {}
    for key in sorted(tracked or {}):
        local_rel = tracked[key]
        local_abs = os.path.join(repo_root, local_rel)
        base = {"name": names.get(key, "tracks"), "b2": key, "kind": "tracks",
                "local": local_rel, "src": local_abs,
                "restage": restage_hint(key, local_rel)}
        lhashes, lsize = _local_fingerprint(local_abs)
        if lhashes is None:
            findings.append(dict(base, status="broken", sentinel=os.path.basename(key),
                                 detail=(f"tracks: names {local_rel}, which is not "
                                         f"readable in this checkout — the "
                                         f"declaration is stale or the file moved")))
            continue
        rkind, rval, err = _b2_object_fingerprint(key, runner=runner, bucket=bucket)
        if rkind is None:
            findings.append(dict(base, status="unknown", sentinel=os.path.basename(key),
                                 detail=err or "B2 object not readable"))
            continue
        lval = lsize if rkind == "size" else lhashes.get(rkind)
        if rval == lval:
            findings.append(dict(base, status="ok", sentinel=os.path.basename(key),
                                 detail=f"{rkind} match ({rval})"))
        elif rkind != "size":
            findings.append(dict(base, status="stale", sentinel=os.path.basename(key),
                                 detail=(f"{rkind} B2 {str(rval)[:16]} != local "
                                         f"{str(lval)[:16]} ({lsize} B local)")))
        else:
            findings.append(dict(base, status="stale", sentinel=os.path.basename(key),
                                 detail=(f"size B2 {rval} B != local {lsize} B "
                                         f"(remote carries no usable hash)")))
    return findings


RUNSET_B2_PREFIX = "runsets"


def runset_preflight_cfg(runset, runset_cfg):
    """A probe config for `herdd train --runset NAME` — GAP 3, the last submit
    surface with no staleness gate at all.

    THE HOLE THIS CLOSES. `tracks:` was wired into `herdd job submit`
    (jobs-v2 bundles) and `jobmatrix submit` (the arm matrix), but NOT into
    `herdd train --runset`, which is the *older* and blunter launcher: it
    rents a box and points it at `b2:runsets/<NAME>/`, whose contents were
    pushed by that runset's `build.sh` at some unrecorded time in the past.
    Every one of the seven runsets stages `train_proposer_lora.py` that way and
    none of them checked it. That is the exact shape of the 2026-07-31 incident
    — B2 held a 64,593 B trainer while HEAD was 125,307 B (HEAD is 147,880 B
    today) — except that on this path there was not even a sentinel heuristic
    to be fooled. Nothing read B2 at all, so nothing could have noticed.

    Provenance, not cleverness: the runset declares in `config.yaml`

        tracks:
          train_proposer_lora.py: tools/pipeline/ml_infra/train_proposer_lora.py

    and this rebases those keys onto `runsets/<NAME>/` so `asset_preflight` can
    compare them against the repo. Pure; validates through `_normalize_tracks`,
    so a malformed declaration raises rather than silently checking nothing.
    A runset with no `tracks:` yields an empty probe (the caller reports that
    as an explicit gap — never as a pass)."""
    where = f"runsets/{runset}/config.yaml tracks:"
    tracks = _normalize_tracks((runset_cfg or {}).get("tracks"), where)
    prefix = f"{RUNSET_B2_PREFIX}/{str(runset).strip('/')}"
    return {"assets": [],
            "tracks": {f"{prefix}/{k}": v for k, v in tracks.items()}}


def asset_preflight(cfg, *, repo_root, runner=None, bucket=None):
    """THE shared submit-path seam: every staleness finding for one validated job
    config — declared-provenance (`tracks:`) plus the legacy runset sentinel.
    `herdd job submit`, `jobmatrix submit` and `herdd train --runset` (via
    `runset_preflight_cfg`) all call exactly this, so no surface can drift in
    what it checks.

    `runner=None` resolves `_default_runner` LATE rather than as a def-time
    default, which makes the MODULE ATTRIBUTE the real transport seam. The CLI
    paths have no injection point of their own, so without late binding the only
    way to test their wiring is to stub out this very function — i.e. to not
    test it. That is precisely the shape that let GAP 3 survive."""
    runner = runner or _default_runner
    tracked = collect_tracked(cfg)
    names = {}
    for a in ((cfg or {}).get("assets") or []):
        prefix = (a.get("b2") or "").strip("/")
        for k in (a.get("tracks") or {}):
            names[f"{prefix}/{str(k).strip('/')}" if prefix else str(k).strip("/")] = \
                a.get("name")
    return (check_tracked_assets(tracked, repo_root=repo_root, runner=runner,
                                 bucket=bucket, asset_names=names)
            + check_asset_staleness((cfg or {}).get("assets"), repo_root=repo_root,
                                    runner=runner, bucket=bucket)
            + check_asset_receipts((cfg or {}).get("assets"), runner=runner,
                                   bucket=bucket))


def measure_asset_bytes(assets, *, runner=_default_runner, bucket=None):
    """{asset_name: bytes} for `disksize.estimate_disk_gb` — velvet P2.

    One `rclone size --json` per DISTINCT asset prefix (the same shape
    `ensure_base_model.sh` already uses). A name is OMITTED, never zeroed, when
    the read fails or the payload is unparseable: the estimator reports an
    omitted name as unsized and downgrades its verdict to a lower bound, which
    is the only honest reading — a missing size silently treated as 0 is how you
    ship a confidently undersized box.

    Injected-runner pure; never raises. A base model staged as an asset resolves
    through the same call, so no special case is needed here.
    """
    out = {}
    for p in sorted({(a.get("b2") or "").strip("/"): a.get("name")
                     for a in (assets or []) if a.get("name")}.items(),
                    key=lambda kv: kv[0]):
        b2p, name = p
        if not b2p:
            continue
        try:
            rc, body, _ = runner(["size", "--json", _q(bucket, b2p)])
        except Exception:
            continue
        if rc != 0 or not (body or "").strip():
            continue
        try:
            n = json.loads(body).get("bytes")
        except (ValueError, TypeError, AttributeError):
            continue
        if isinstance(n, (int, float)) and n >= 0:
            out[name] = int(n)
    return out


def parse_receipt_body(text):
    """(complete, files) from a receipt body, both possibly None.

    `complete` is True/False only when the key is a real bool; `files` only when
    it is a non-negative int. Anything else — unparseable JSON, a bare
    `touch`-style empty marker, a schema we do not know — reads as (None, None),
    i.e. NO CLAIM. Pure; never raises. The asymmetry is deliberate and is the
    same house rule as the disk precheck's unreadable `df`: a marker we cannot
    read is not evidence of incompleteness, and refusing on it would strand a
    publish whose payload is perfectly whole."""
    try:
        d = json.loads(text)
    except Exception:
        return None, None
    if not isinstance(d, dict):
        return None, None
    complete = d.get("complete")
    complete = complete if isinstance(complete, bool) else None
    files = d.get("files")
    files = (files if isinstance(files, int) and not isinstance(files, bool)
             and files >= 0 else None)
    return complete, files


def check_asset_receipts(assets, *, runner=_default_runner, bucket=None):
    """Read each `receipt:`-declaring asset's completeness marker on B2 at SUBMIT
    time. Returns findings {name, b2, receipt, status, detail, files}:

      status 'missing'    the receipt object is not there, or says complete:false
                          -> the prefix is NOT published; refuse before renting
             'ok'         present, and (if it parses) claims completeness
             'unknown'    the read failed (no creds, a blip) -> a NOTE, never a block

    WHY THIS RUNS LAPTOP-SIDE TOO, when jobd re-checks it on the box: the box
    check is correct but expensive — by the time it fires, a machine is rented
    and billing. This one is $0 and answers the same question against the same
    object, so the overwhelmingly common failure (someone submits against a
    prefix whose publish never finished) is caught before any money moves.
    Injected-runner pure; never raises, never mutates B2."""
    findings = []
    for a in (assets or []):
        receipt = (a.get("receipt") or "").strip("/")
        if not receipt:
            continue
        name, b2p = a.get("name"), (a.get("b2") or "").strip("/")
        key = f"{b2p}/{receipt}" if b2p else receipt
        try:
            rc, out, err = runner(["cat", _q(bucket, key)])
        except Exception as e:
            findings.append({"name": name, "b2": b2p, "receipt": receipt,
                             "kind": "receipt", "status": "unknown", "files": None,
                             "detail": f"B2 read error: {e}"})
            continue
        if rc != 0:
            # A missing marker and a dead key are NOT the same answer, and only
            # one of them may block. rclone says "not found"/"doesn't exist" for
            # the former; anything else (403, DNS, no creds) is a blip.
            blob = ((err or "") + (out or "")).lower()
            if "not found" in blob or "doesn't exist" in blob or "does not exist" in blob:
                findings.append({
                    "name": name, "b2": b2p, "receipt": receipt, "kind": "receipt",
                    "status": "missing", "files": None,
                    "detail": "no completeness receipt at this prefix — the "
                              "publish never finished, or never happened"})
            else:
                findings.append({"name": name, "b2": b2p, "receipt": receipt,
                                 "kind": "receipt", "status": "unknown", "files": None,
                                 "detail": (err or "").strip() or "B2 read failed"})
            continue
        complete, files = parse_receipt_body(out or "")
        if complete is False:
            findings.append({
                "name": name, "b2": b2p, "receipt": receipt, "kind": "receipt",
                "status": "missing", "files": files,
                "detail": "the receipt is present and says complete: false"})
            continue
        findings.append({
            "name": name, "b2": b2p, "receipt": receipt, "kind": "receipt",
            "status": "ok", "files": files,
            "detail": (f"{files} file(s) published" if files is not None
                       else "present (no file count in the body)")})
    return findings


def check_asset_staleness(assets, *, repo_root, runner=_default_runner, bucket=None):
    """Compare each MUTABLE asset's B2 sentinel against the local source it mirrors
    (one small-object read, byte-identity). Returns a list of finding dicts:

      {name, b2, status, sentinel, detail}
        status 'stale'   B2 sentinel bytes differ from local  -> drift; re-stage
               'ok'      byte-identical                        -> silent
               'skipped' immutable prefix / no local source / no local sentinel
               'unknown' B2 read failed (creds absent, blip, or object missing)

    Injected-runner pure otherwise. NEVER raises on a transport failure and never
    mutates B2 — a network blip degrades to 'unknown' (a NOTE, not a block)."""
    findings = []
    for a in (assets or []):
        name = a.get("name")
        b2p = (a.get("b2") or "").strip("/")
        if a.get("tracks"):
            # An explicit provenance declaration SUPERSEDES the sentinel guess
            # for this asset: `check_tracked_assets` compares the named repo
            # files directly, so re-running the heuristic here could only add a
            # false alarm from the gitignored `_build` mirror.
            findings.append({"name": name, "b2": b2p, "status": "skipped",
                             "sentinel": None,
                             "detail": "covered by the asset's `tracks:` declaration"})
            continue
        src = local_source_for_asset(b2p, repo_root)
        if src is None:
            findings.append({"name": name, "b2": b2p, "status": "skipped",
                             "sentinel": None,
                             "detail": "immutable-ish or unmapped B2 prefix"})
            continue
        # first candidate sentinel that exists in the local source
        sentinel = local_text = None
        for cand in ASSET_SENTINELS:
            fp = os.path.join(src, cand)
            if os.path.isfile(fp):
                try:
                    with open(fp, encoding="utf-8", errors="surrogateescape") as fh:
                        local_text = fh.read()
                except OSError:
                    continue
                sentinel = cand
                break
        if sentinel is None:
            findings.append({"name": name, "b2": b2p, "status": "skipped",
                             "sentinel": None,
                             "detail": f"no sentinel {ASSET_SENTINELS} in {src}"})
            continue
        key = f"{b2p}/{sentinel}"
        try:
            rc, out, err = runner(["cat", _q(bucket, key)])
        except Exception as e:                 # misconfig / blip -> never block
            findings.append({"name": name, "b2": b2p, "status": "unknown",
                             "sentinel": sentinel, "src": src,
                             "detail": f"B2 read error: {e}"})
            continue
        if rc != 0:
            findings.append({"name": name, "b2": b2p, "status": "unknown",
                             "sentinel": sentinel, "src": src,
                             "detail": ((err or "").strip()
                                        or "B2 sentinel not readable "
                                           "(creds absent, or object missing)")})
            continue
        # Same universal-newline normalization on both sides (runner runs rclone
        # with text=True); a genuine code edit changes the text regardless.
        if out == local_text:
            findings.append({"name": name, "b2": b2p, "status": "ok",
                             "sentinel": sentinel, "src": src,
                             "detail": "byte-identical"})
        else:
            findings.append({"name": name, "b2": b2p, "status": "stale",
                             "sentinel": sentinel, "src": src,
                             "detail": (f"B2 {len(out)} chars vs local "
                                        f"{len(local_text)} chars — differ")})
    return findings


def asset_preflight_report(findings, *, strict=False, allow_stale=False):
    """Turn staleness findings into (surfaced_lines, refuse). Presentation policy:

      * kind='tracks' 'stale'/'broken' -> LOUD, and refuse=True REGARDLESS of
        `strict`. A `tracks:` mapping is an explicit operator contract ("this
        staged object mirrors that repo file"), so a mismatch is a known-wrong
        run, not a guess — it fails CLOSED before a box is rented.
        `allow_stale=True` (--allow-stale-assets) is the deliberate opt-out for
        "I really do want the bytes currently on B2".
      * heuristic (sentinel) 'stale' -> LOUD; refuse only under strict. Unchanged:
        it is an inference, and inferences do not get to block a submit.
      * 'unknown' -> a 'note:' line (graceful skip); NEVER refuses, even strict —
        a transport blip (or an offline laptop with no B2 creds) must not block.
      * kind='tracks' 'ok' -> ONE short confirmation line. Deliberately not
        silent: this whole preflight exists because a stale staged trainer was
        invisible, and a silent pass is indistinguishable from a check that
        never ran (the v7 matrix lane carried no asset check at all and looked
        exactly this quiet). A declared contract reports that it was honoured.
        Tracked assets are few and named by hand, so this cannot become noise.
      * heuristic 'ok' / 'skipped' -> silent (inferences stay out of the way).
    Pure. The CLI prints the lines and sys.exits when refuse is True."""
    lines = []
    refuse = tracked_hit = other_hit = False
    for f in (findings or []):
        st = f.get("status")
        tracked = f.get("kind") == "tracks"
        if f.get("kind") == "receipt":
            # A completeness receipt is a PUBLISHER contract, not a freshness
            # guess, so `missing` refuses regardless of --strict-assets. It is
            # also deliberately NOT cleared by --allow-stale-assets: that flag
            # means "run the staged bytes on purpose", and there is no coherent
            # reading of "run the bytes of a publish that never finished". The
            # escape is --no-asset-check, which skips the preflight entirely.
            if st == "missing":
                refuse = True
                lines.append(
                    f"!! ASSET INCOMPLETE: b2:{f['b2']} has no completeness "
                    f"receipt ({f['receipt']}) — {f.get('detail', '')}. The box "
                    f"would stage a TRUNCATED prefix. Re-run the publish (its "
                    f"marker is written last, so a prefix without one is not "
                    f"published), or drop `receipt:` if this prefix has no "
                    f"publisher.\n   (--no-asset-check skips the preflight.)")
            elif st == "unknown":
                lines.append(
                    f"note: asset {f.get('name')!r} receipt UNVERIFIED "
                    f"(b2:{f['b2']}/{f['receipt']}) — {f.get('detail', '')}; "
                    f"jobd re-checks it on the box.")
            else:
                lines.append(f">> asset receipt OK: b2:{f['b2']}/{f['receipt']} "
                             f"— {f.get('detail', '')}")
        elif tracked and st in ("stale", "broken"):
            refuse = refuse or not allow_stale
            tracked_hit = other_hit = True
            what = "STALE" if st == "stale" else "PROVENANCE BROKEN"
            lines.append(
                f"!! ASSET {what}: b2:{f['b2']} does not match the repo file it "
                f"declares it mirrors ({f.get('local', '?')}) — {f.get('detail', '')}. "
                f"The box would run the B2 bytes, not yours. Re-stage:\n"
                f"     {f.get('restage', '(re-stage the asset)')}")
        elif st == "stale":
            refuse = refuse or strict
            other_hit = True
            lines.append(
                f"!! ASSET STALE: b2:{f['b2']}/{f['sentinel']} differs from the "
                f"local source it mirrors ({f.get('src', '?')}) — {f.get('detail', '')}. "
                f"Re-stage with the runset build.sh before submitting "
                f"(the box would run the B2 bytes, not yours).")
        elif st == "unknown":
            lines.append(
                f"note: asset {f.get('name')!r} staleness UNVERIFIED "
                f"(b2:{f['b2']}) — {f.get('detail', '')}; proceeding without the check.")
        elif tracked and st == "ok":
            lines.append(
                f">> asset provenance OK: b2:{f['b2']} matches "
                f"{f.get('local', '?')} ({f.get('detail', '')})")
    if refuse and other_hit:
        # Suppressed on a RECEIPT-ONLY refusal: neither flag it names applies
        # there, and an escape hatch that does not work is worse than none.
        lines.append("   (pass --allow-stale-assets to run the STAGED bytes on "
                     "purpose, or --no-asset-check to skip the preflight entirely.)"
                     if tracked_hit else
                     "   (pass --no-asset-check to submit anyway, or drop "
                     "--strict-assets to downgrade to a warning.)")
    return lines, refuse


# --- EVAL_ENV_VER pin gate (submit-time; M4) ----------------------------------
# WHY THIS REFUSES INSTEAD OF REPORTING. A `needs.venv: eval` job runs against
# the BAKED eval-env, and an unpinned launch resolves `eval-env/LATEST` at boot
# — which can be OLDER than the env the bundle was preflighted against, because
# a deliberately-pinned bake does not advance LATEST. That is not hypothetical:
# wave A (docs/plans/witness/g2_push/FLOOR_DEGRADATION_2026-08-01.md) graded
# its FLOOR gate on an env predating `floor_restore.py`, so three revert-then-
# splice controls were read and DROPPED and the gate PASSed on 2 bytes-moving
# controls instead of 5 — with the correct counts visible in the artifacts. The
# old code imports cleanly (it simply never imports the module), so nothing
# on-box could see it; `results/env_identity.json` records the loaded version
# but only *reports*. The remedy was written up as "an operator rule, not code"
# (§5d row 5a.3) and an operator rule is exactly what failed. Hence: refuse.
#
# WHICH PIN ACTUALLY BINDS. jobd sources the job's `.job.env` inside the
# ENTRYPOINT subshell (`onstart/jobd.sh`), which runs AFTER `check_venv eval`
# has already provisioned the env via `fetch_eval_env.sh`. So on a cold box only
# the BOX launch env (`herdd launch/train --eval-env-ver` -> vast extra_env)
# selects the tarball; a job-config/`--env` pin reaches the entrypoint (where
# run.sh compares it against the loaded manifest and writes env_identity.json)
# but cannot steer the fetch. Both are accepted as a pin — an operator who names
# a version has made the choice reviewable either way — but the job-env-only
# case is called out loudly, because it documents rather than determines.
EVAL_ENV_PIN_KEY = "EVAL_ENV_VER"


def _pin_str(v):
    return "" if v is None else str(v).strip()


def eval_env_pin_report(cfg, box_env, *, box=None, box_env_known=True,
                        require_box_pin=False):
    """(lines, refuse) for the submit-time EVAL_ENV_VER gate. Pure.

    Fires only for `needs.venv: eval` — the jobs whose grading semantics come
    out of the baked env. `box_env` is the target box's launch env
    (vast `extra_env`, {} if it carries no pin); `box_env_known=False` says the
    record could not be read (soft API failure, or `--local`), which changes the
    wording but never the verdict: a submit with no visible pin is refused.

    Deliberately NO override flag. The two documented escapes are the two ways
    of naming the version — `job submit --env EVAL_ENV_VER=<ver>`, or launching
    the box with `--eval-env-ver <ver>` — both of which leave the choice in the
    ticket/instance record where a later reader can audit it.

    `require_box_pin=True` upgrades the job-pin-only case from a NOTE to a
    refusal. It is opt-in, and it is for ONE caller shape: a launcher that just
    rented the box itself and injected the pin into its launch env
    (`launch_jobs_box.sh`). There the box is COLD by construction — nothing has
    provisioned /workspace/eval yet — so a missing box pin is not a benign
    "already warm at the right version", it is proof the injection did not land,
    and the next thing that happens is jobd fetching eval-env/LATEST. Left OFF
    by default because the soft path is load-bearing for the other shape: a
    hand-resubmit onto a warm box, where the env is already unpacked and the
    fetch the box pin steers will never run again."""
    if ((cfg or {}).get("needs") or {}).get("venv") != "eval":
        return [], False
    job_pin = _pin_str(((cfg or {}).get("env") or {}).get(EVAL_ENV_PIN_KEY))
    box_pin = _pin_str((box_env or {}).get(EVAL_ENV_PIN_KEY)) if box_env_known else ""
    where = f"box {box}" if box else "the target box"
    if require_box_pin and not box_pin:
        seen = ("could not be read" if not box_env_known
                else f"carries no {EVAL_ENV_PIN_KEY}")
        return ([
            f"!! EVAL_ENV_VER NOT ON THE BOX: {where}'s launch env {seen}, and "
            f"this submit was made with --require-box-eval-pin — i.e. by a "
            f"caller that rented the box and believed it had injected the pin."
            f"\n   The box env is the ONLY pin fetch_eval_env.sh can see, so a "
            f"cold box with no box pin resolves eval-env/LATEST at boot"
            + (f" instead of the {job_pin!r} this job grades against"
               if job_pin else "") + "."
            f"\n   Refusing while the box is still empty. Relaunch it with "
            f"`--eval-env-ver "
            + (job_pin if job_pin else "<ver>") + "`."], True)
    if not job_pin and not box_pin:
        unknown = "" if box_env_known else (
            f"\n   ({where}'s launch env could not be read, so a box-side pin "
            f"cannot be confirmed either — name the version in the submit.)")
        return ([
            f"!! EVAL_ENV_VER UNPINNED: this job declares needs.venv: eval, and "
            f"neither its `env:` block nor {where}'s launch env names an "
            f"eval-env version. An unpinned box resolves eval-env/LATEST at "
            f"boot, which CAN be older than the env you preflighted (a pinned "
            f"bake does not advance LATEST) — that is how wave A graded FLOOR "
            f"on pre-fix code and passed on 2 of 5 bytes-moving controls."
            f"\n   Pin it: herdd job submit … --env EVAL_ENV_VER=<ver>"
            f"\n   (list versions: rclone lsf b2:$B2_BUCKET/eval-env/ ; the box "
            f"can also be launched with --eval-env-ver <ver>, which is the pin "
            f"that steers the fetch on a cold box).{unknown}"], True)
    if job_pin and box_pin and job_pin != box_pin:
        return ([
            f"!! EVAL_ENV_VER CONFLICT: the job pins {job_pin!r} but {where} was "
            f"launched with {box_pin!r}. The box env is what fetch_eval_env.sh "
            f"reads, so the job would GRADE on {box_pin!r} while its artifacts "
            f"claim {job_pin!r}. Resubmit with --env EVAL_ENV_VER={box_pin}, or "
            f"use a box launched with {job_pin}."], True)
    if box_pin:
        return ([f">> eval-env pin: {box_pin} ({where} launch env"
                 + (", job env agrees)" if job_pin else ")")], False)
    return ([
        f">> eval-env pin: {job_pin} (job env)",
        f"~~ NOTE: {where}'s launch env carries no EVAL_ENV_VER, so if its eval "
        f"venv is not already provisioned jobd fetches eval-env/LATEST BEFORE "
        f"the job env exists. The pin is recorded and compared on-box "
        f"(env_identity.json), not enforced by the fetch — launch with "
        f"--eval-env-ver {job_pin} for that."], False)


# --- needs.gpu_ram_gb gate (submit-time; measured, not modelled) --------------
# WHY. `needs.gpu_ram_gb` is hand-authored per bundle and the provenance is
# whatever the author had that day — measured for some, an explicit non-
# measurement for others ("**Not** a local measurement", phase1-cot-train's own
# README), and for the rest inherited by fork from the bundle it was copied
# from. It costs money in BOTH directions: too low and jobd schedules the job
# onto a card too small for it, too high and it gates to a bigger card class
# than the shape needs. tools/vast/vram_facts.py answers the same question from
# every past run's measured peak, so this compares the two.
#
# The asymmetry is deliberate: REFUSE only on what measurement PROVES wrong (a
# declared floor below a peak this exact shape has already been observed to
# reach), WARN on everything the estimate merely suggests. The estimate picks
# the right card class about two thirds of the time, which is plenty to advise
# with and nowhere near enough to block on.

# Bundle env -> vram_facts shape. Only these keys; a bundle that spells its
# shape some other way is reported as unreadable rather than guessed at.
_VRAM_ENV = {
    "base_slug": "BASE_SLUG", "quant_mode": "QUANT", "max_seq": "MAX_SEQ",
    "batch": "BATCH", "grad_checkpointing": "GRAD_CKPT",
    # The same env knob carries the GC FRACTION ("0.5"); anchors group on it
    # (vram_facts.gc_flag_class), so a query that drops it defaults to gc=full
    # and misses every fractional group.
    "grad_checkpointing_flag": "GRAD_CKPT",
    "ce_chunk_matmul": "CE_CHUNK_MATMUL", "target_modules": "TARGET_MODULES",
    "lora_r": "LORA_R", "packing": "PACKING", "fsdp": "FSDP",
}

# train_proposer_lora.py's own argparse defaults, for knobs a bundle leaves
# unset. Kept here rather than imported: the trainer pulls in torch.
# MUST track the trainer. `ce_chunk_matmul` flipped fp32 -> bf16 on 2026-08-10;
# a stale value here mis-keys every anchor group for bundles that leave
# CE_CHUNK_MATMUL unset, which is worse than not gating them at all.
_TRAINER_DEFAULTS = {"ce_chunk_matmul": "bf16", "packing": "off",
                     "grad_checkpointing": "on"}
_TRAINER_DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj")


def base_slug_from_assets(assets) -> str:
    """The base model a bundle stages, read off its `assets:` list.

    Most training bundles never set BASE_SLUG — they name the model once, as
    the B2 prefix of their `base` asset (`b2: base-models/qwen25-coder-7b-
    instruct`). Reading only the env missed five of them, including v7."""
    for a in (assets or []):
        b2 = str((a or {}).get("b2") or "")
        if b2.startswith("base-models/"):
            return b2.split("/", 1)[1].strip("/").split("/")[0]
    return ""


def window_ladder_from_env(env: dict) -> list[int]:
    """The rungs of a `WINDOW_LADDER`, ascending, or [].

    v10 authors no MAX_SEQ: it ships `WINDOW_LADDER: "20480,16384"` and its
    run.sh fit-probes the rungs largest-first ON THE BOX, taking the first that
    fits and exiting 15 if none does. Reading only MAX_SEQ made the gate silent
    on it — the largest training run in the campaign, 90 GB declared, and the
    one number a sizing check would most want to see."""
    raw = str((env or {}).get("WINDOW_LADDER") or "").strip()
    if not raw:
        return []
    out = []
    for part in raw.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return []                 # unreadable ladder -> no opinion
    return sorted(out)


def vram_shape_from_env(env: dict, assets=None) -> dict | None:
    """Read a training shape out of a bundle's `env:` (+ `assets:`), or None if
    it isn't one.

    A bundle with no identifiable base or no MAX_SEQ is an eval, a generation
    sweep, or a probe — not a training job with a predictable footprint — and
    must pass through untouched rather than be sized by a table that knows
    nothing about it."""
    env = dict(env or {})
    if not env.get("BASE_SLUG"):
        recovered = base_slug_from_assets(assets)
        if recovered:
            env["BASE_SLUG"] = recovered
    if not env.get("MAX_SEQ"):
        # A ladder bundle is sized on its SMALLEST rung, because that is the
        # one whose failure is unrecoverable: the box tries the rungs
        # largest-first and falls back, so a peak above the declared floor at
        # the top rung merely costs a fallback, while a peak above it at the
        # bottom rung means no rung fits and the run exits without training.
        # Refusing on the top rung would refuse a bundle designed to fall back.
        ladder = window_ladder_from_env(env)
        if ladder:
            env["MAX_SEQ"] = str(ladder[0])
    if not env.get("BASE_SLUG") or not env.get("MAX_SEQ"):
        return None
    out = {}
    for field, key in _VRAM_ENV.items():
        v = str(env.get(key, "")).strip()
        if v:
            out[field] = v
    for n in ("max_seq", "batch", "lora_r"):
        if n in out:
            try:
                out[n] = int(out[n])
            except ValueError:
                return None
    tm = out.get("target_modules")
    if tm and tm != "all-linear":
        out["target_modules"] = [x.strip() for x in tm.split(",") if x.strip()]
    elif not tm:
        out["target_modules"] = list(_TRAINER_DEFAULT_TARGETS)
    # An anchor records what the trainer RESOLVED; a bundle's env records only
    # what it overrode. Comparing the two without filling in the defaults makes
    # every bundle look like an unmeasured shape — which is what happened first
    # time round: all six training bundles came back UNCHECKED purely because
    # they leave CE_CHUNK_MATMUL and PACKING at the trainer's defaults.
    out.setdefault("ce_chunk_matmul", _TRAINER_DEFAULTS["ce_chunk_matmul"])
    out.setdefault("packing", _TRAINER_DEFAULTS["packing"])
    out.setdefault("grad_checkpointing", _TRAINER_DEFAULTS["grad_checkpointing"])
    gc = str(out["grad_checkpointing"]).strip().lower()
    if gc in ("auto", ""):
        gc = "on"                     # --grad-checkpointing auto -> on (trainer)
    try:
        on = float(gc) > 0            # a fraction: any nonzero checkpoints
    except ValueError:
        on = gc in ("on", "true", "yes")
    out["grad_checkpointing"] = on
    return out


def vram_requirement(cfg: dict, *, world_size=None) -> dict | None:
    """The measured per-card VRAM floor this config's training shape needs, or
    None when it is not a training shape (or vram_facts is unavailable).

    Extracted from `vram_gate_findings` so a READER can ask the sizing question
    with the same machinery the submit gate refuses on — `herdd search --job`
    derives its `--gpu-ram` filter from here. Two copies of this derivation
    would let the board a search shows drift away from what submit admits.
    `world_size` overrides `needs.gpus` for a caller ranking a different card
    count. PURE apart from reading the facts file."""
    shape = vram_shape_from_env((cfg or {}).get("env"), (cfg or {}).get("assets"))
    if shape is None:
        return None
    try:
        if _HERE not in sys.path:      # not `insert` unconditionally: this runs
            sys.path.insert(0, _HERE)  # once per matrix ARM, and would grow
        import vram_facts  # noqa: E402
    except ImportError:
        return None
    ws = world_size if world_size is not None else ((cfg or {}).get("needs")
                                                    or {}).get("gpus")
    shape["world_size"] = ws if isinstance(ws, int) else 1
    ladder = window_ladder_from_env((cfg or {}).get("env"))
    extra = {"ladder": ladder} if ladder else {}
    # The gate sizes the DECLARED shape; the box can re-decide part of it.
    # Only one re-decision moves VRAM the wrong way, and it is narrow:
    # launch_plan.sh flips grad-ckpt to `off` (higher peak) ONLY under
    # MODE=autotune with GRAD_CKPT unset/auto. An explicit `on` is honoured,
    # and an explicit `off` can only be flipped TO on (the VRAM-safety floor).
    # That unset case is also where `_TRAINER_DEFAULTS` fills in `on` — so the
    # gate would size the cheap posture while the box runs the expensive one,
    # measured at 20.87 -> 52.20 GB for a 7B at seq 4096.
    _env = {str(k).strip().upper(): str(v).strip().lower()
            for k, v in ((cfg or {}).get("env") or {}).items()}
    if (_env.get("MODE") == "autotune"
            and _env.get("GRAD_CKPT", "") in ("", "auto")):
        extra["autotune_may_disable_grad_ckpt"] = True
    try:
        # Pass the cached document: a matrix checks every arm, and each
        # required_gpu_ram_gb() would otherwise re-read and re-parse the file.
        r = vram_facts.required_gpu_ram_gb(facts=vram_facts.load_facts(), **shape)
    except vram_facts.Unmeasured as e:
        return {"status": "unmeasured", "detail": str(e),
                "probe_cmd": e.probe_cmd, "shape": shape, **extra}
    except Exception as e:                  # a broken facts file must not block
        return {"status": "skipped", "detail": str(e)}
    return dict(r, status="ok", shape=shape, **extra)


def vram_gate_findings(cfg: dict) -> dict | None:
    """Compare a config's declared `needs.gpu_ram_gb` against the measured
    estimate. Returns None when the gate does not apply (not a training shape,
    no declared floor, or vram_facts is unavailable). PURE apart from reading
    the facts file."""
    declared = ((cfg or {}).get("needs") or {}).get("gpu_ram_gb")
    if declared is None:
        return None
    r = vram_requirement(cfg)
    return None if r is None else dict(r, declared=declared)


def vram_gate_report(finding, *, allow_drift=False):
    """(lines, refuse) from a `vram_gate_findings` result. Pure.

    Refuses in exactly one case: the declared per-card floor is below a peak
    THIS SHAPE HAS ALREADY BEEN MEASURED TO REACH. That is not a prediction —
    it is a run that happened, so the declared floor is known-wrong and the job
    would be scheduled onto a card that cannot hold it. Everything else advises.
    `--allow-vram-drift` opts out."""
    lines, refuse = _vram_gate_lines(finding, allow_drift=allow_drift)
    if lines and (finding or {}).get("autotune_may_disable_grad_ckpt"):
        lines.append(
            "   (MODE=autotune with GRAD_CKPT unset: the box may pick `off`, "
            "which this estimate does NOT cover — it sizes the `on` default. "
            "Author GRAD_CKPT explicitly to make the gate binding.)")
    lad = (finding or {}).get("ladder")
    if lines and lad:
        rungs = ",".join(str(x) for x in reversed(lad))
        lines.append(
            f"   (sized on {lad[0]}, the SMALLEST rung of WINDOW_LADDER "
            f"{rungs} — the box probes largest-first and falls back, so only "
            f"the bottom rung failing to fit is unrecoverable)")
    return lines, refuse


def _vram_gate_lines(finding, *, allow_drift=False):
    if not finding:
        return [], False
    st = finding.get("status")
    declared = finding.get("declared")
    if st == "skipped":
        return [f"note: VRAM sizing check skipped ({finding.get('detail')})"], False
    if st == "unmeasured":
        lines = [f">> vram: gpu_ram_gb {declared} is UNCHECKED — no measured "
                 f"anchor for this shape ({finding.get('detail')})"]
        if finding.get("probe_cmd"):
            lines.append(f"   measure it: {finding['probe_cmd']}")
        return lines, False

    peak, req, klass = finding["gb"], finding["required_gb"], finding["card_class"]
    n, spread = finding["n"], finding["spread"]
    est = (f"measured peak {peak:.2f} GB (n={n}"
           + (f", spread {spread:.2f}" if spread else "")
           + (", EXTRAPOLATED" if finding.get("extrapolated") else "")
           + f") -> needs {req:.2f} GB, card class {klass}")

    if declared < peak:
        return ([f"!! vram: gpu_ram_gb {declared} is BELOW a peak this exact "
                 f"shape has already measured — {est}",
                 f"   runs that reached it: {', '.join(finding['runs'][:3])}",
                 f"   jobd schedules on cards >= {declared} GB, so this would "
                 f"place a {peak:.2f} GB job on a card that cannot hold it."],
                not allow_drift)
    if declared < req:
        return ([f">> vram: gpu_ram_gb {declared} is inside the headroom band — "
                 f"{est}. Fits every measured run, with "
                 f"{declared - peak:.2f} GB of reserved-pool margin instead of "
                 f"{finding['headroom_gb']:.2f}."], False)
    if declared > klass:
        return ([f">> vram: gpu_ram_gb {declared} gates to a bigger card than "
                 f"the shape needs — {est}. Lowering it to {klass} widens the "
                 f"offer pool."
                 + (f" (Caveat: identical declared shapes in this group have "
                    f"measured {spread:.2f} GB apart, so some margin is "
                    f"warranted.)" if spread > 1.0 else "")], False)
    return [f">> vram: gpu_ram_gb {declared} agrees with measurement — {est}"], False


# --- live-append tail snapshot (task #110) ------------------------------------ #
# THE DEFECT. jobd's periodic checkpoint pass ships with `rclone --min-age 45s`
# so it never copies a file the entrypoint is mid-write on. `--min-age` filters
# on MTIME and every append bumps it, so a file appended faster than the window
# never ages past it and is NEVER shipped — not "shipped one tick late".
#
# PROVEN, not theorised, on job 20260803T130435-frontier-wave-3a68 (rb3 PAD arm,
# 2026-08-03). Its ten attempt-1 `checkpoint` events read matched=16 / files=15
# for every pass from 13:10 to 13:35 — sixteen glob matches, fifteen shipped,
# the SAME one file skipped every time. That file was `results/gens_PAD.jsonl`,
# the arm being generated at ~15-20 s/chunk, i.e. permanently inside the 45 s
# window. Pass n=1 reads files=5/matched=11, which is exactly the five artifacts
# then older than 45 s. The bundle's `checkpoints:` globs DO cover
# `results/gens_*.jsonl` — the glob was never the problem, the age filter was.
# Cost that day: 864 of 1090 rows, ~25-30 min of GPU, regenerated from chunk 1.
#
# THE FIX, and why it is not "drop --min-age". A checkpoint taken mid-append can
# ship a PARTIAL FINAL LINE, and durability that trades bounded lost compute for
# a corrupt row is strictly worse than the defect. So the age window stays
# exactly as it is on the ordinary pass, and the files it skips get a second,
# narrower path: a SNAPSHOT cut at the last complete line.
#
# Nothing is snapshotted unless it is provably an append-only text log:
#   1. the bundle's own `checkpoints:` globs matched it (unchanged authority —
#      this never widens what a job checkpoints, only when);
#   2. the ordinary pass would skip it (younger than the age window);
#   3. it is not inside a `checkpoint-<N>/` dir — the trainer-save protocol owns
#      those and has a real completeness oracle (`_ckpt_write_complete`);
#   4. it is at most `max_bytes` (a big binary is not an append log);
#   5. it was seen in the PREVIOUS pass, has GROWN since, and the bytes it held
#      then are UNCHANGED now (sha256 over the old prefix). That is the append
#      proof: a rewritten file, an atomic replace, and a quiescent file all fail
#      it, so a whole-JSON artifact caught mid-write is never shipped;
#   6. the prefix holds a newline, no NUL byte, decodes as UTF-8, and its FIRST
#      and LAST complete lines each parse as JSON. That is the NDJSON test —
#      pretty-printed JSON fails it on line 1 (`{`), compact JSON fails 6's
#      newline clause, binary fails the NUL/UTF-8 clause.
# The snapshot is then bytes [0, last_newline], so the object that reaches B2
# holds a whole number of complete records and can never carry a torn tail.
# Readers stay tail-tolerant anyway (`ndjson_tail.py`, fd4c80ff) — belt and
# braces, because the LOCAL file after a hard kill still can be torn.
_CKPT_TAIL_MAX_BYTES = 128 * 1024 * 1024
_CKPT_TAIL_CHUNK = 1 << 20


def _ckpt_tail_read_state(path) -> dict:
    """Prior-pass observations: {rel: {"size": int, "sha": hex}}. An unreadable
    or malformed state file is an EMPTY history, never an error — the cost is
    one pass of baseline, and a state file is a cache, not a record."""
    try:
        with open(path) as fh:
            st = json.load(fh)
    except Exception:
        return {}
    if not isinstance(st, dict):
        return {}
    out = {}
    for rel, v in st.items():
        if isinstance(v, dict) and isinstance(v.get("size"), int) \
                and isinstance(v.get("sha"), str):
            out[str(rel)] = {"size": v["size"], "sha": v["sha"]}
    return out


def _ckpt_tail_write_state(path, state) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _ckpt_tail_hashes(fh, prev_size: int, size: int):
    """One streamed pass over [0, size): return (sha of the first `prev_size`
    bytes, sha of all `size` bytes). `prev_size` <= 0 or > size yields None for
    the first element (nothing to compare against)."""
    h = hashlib.sha256()
    prev_sha = None
    if prev_size == 0:
        prev_sha = hashlib.sha256().hexdigest()
    read = 0
    while read < size:
        want = min(_CKPT_TAIL_CHUNK, size - read)
        if prev_sha is None and read < prev_size < read + want:
            want = prev_size - read          # stop exactly on the old boundary
        buf = fh.read(want)
        if not buf:
            break
        h.update(buf)
        read += len(buf)
        if prev_sha is None and read == prev_size:
            prev_sha = h.hexdigest()
    return prev_sha, h.hexdigest(), read


def _ckpt_tail_last_newline(fh, size: int) -> int:
    """Offset one PAST the last b"\\n" strictly inside [0, size), or -1."""
    pos = size
    while pos > 0:
        start = max(0, pos - _CKPT_TAIL_CHUNK)
        fh.seek(start)
        buf = fh.read(pos - start)
        i = buf.rfind(b"\n")
        if i >= 0:
            return start + i + 1
        pos = start
    return -1


def _ckpt_tail_is_ndjson(data: bytes) -> bool:
    """`data` is a whole number of newline-terminated lines. Accept it only if
    it decodes as UTF-8 with no NUL byte and its first and last non-blank lines
    each parse as JSON — the discriminator that keeps an append-only NDJSON
    apart from a half-written JSON document or a binary."""
    if not data or b"\0" in data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    for l in (lines[0], lines[-1]):
        try:
            json.loads(l)
        except Exception:
            return False
    return True


def _ckpt_tail_in_checkpoint_dir(rel: str) -> bool:
    return any(re.fullmatch(r"checkpoint-\d+", part)
               for part in rel.replace("\\", "/").split("/")[:-1])


def ckpt_tail_snapshot(run_dir, rels, min_age_s, state_path, stage_dir,
                       now=None, max_bytes=_CKPT_TAIL_MAX_BYTES) -> dict:
    """Stage line-aligned snapshots of the actively-appended files among `rels`.

    `rels` are run-dir-relative paths the bundle's checkpoint globs already
    matched. Returns {"staged": [rel…], "skipped": {rel: reason}} and rewrites
    `state_path` with this pass's observations. Pure filesystem work: no B2, no
    network — jobd pushes `stage_dir` afterwards.

    The six admission rules are in this section's header comment. Every refusal
    is recorded with a reason so a pass that ships nothing can say why.
    """
    now = time.time() if now is None else now
    prev = _ckpt_tail_read_state(state_path)
    state, staged, skipped = {}, [], {}
    for rel in rels:
        rel = str(rel).strip()
        if not rel:
            continue
        src = os.path.join(run_dir, rel)
        try:
            st = os.stat(src)
        except OSError:
            continue                                   # vanished mid-pass
        if not os.path.isfile(src):
            continue
        size = st.st_size
        if _ckpt_tail_in_checkpoint_dir(rel):
            skipped[rel] = "checkpoint-dir"            # rule 3
            continue
        if size > max_bytes:
            skipped[rel] = "too-big"                   # rule 4
            continue
        # Record every candidate's observation even when it does not ship: a
        # file that is quiescent this pass and appended-to next needs a baseline
        # already on file, or the append proof can never be satisfied.
        old = prev.get(rel)
        prev_size = old["size"] if old else -1
        try:
            with open(src, "rb") as fh:
                prev_sha, full_sha, read = _ckpt_tail_hashes(
                    fh, prev_size if 0 <= prev_size <= size else -1, size)
                if read != size:
                    continue                           # raced a truncation
                state[rel] = {"size": size, "sha": full_sha}
                if now - st.st_mtime >= min_age_s:
                    skipped[rel] = "old-enough"        # rule 2: ordinary pass owns it
                    continue
                if old is None:
                    skipped[rel] = "no-baseline"       # rule 5
                    continue
                if size <= old["size"]:
                    skipped[rel] = "not-growing"       # rule 5
                    continue
                if prev_sha != old["sha"]:
                    skipped[rel] = "prefix-changed"    # rule 5: rewritten, not appended
                    continue
                cut = _ckpt_tail_last_newline(fh, size)
                if cut <= 0:
                    skipped[rel] = "no-complete-line"  # rule 6
                    continue
                fh.seek(0)
                data = fh.read(cut)
        except OSError:
            continue
        if len(data) != cut or not _ckpt_tail_is_ndjson(data):
            skipped[rel] = "not-ndjson"                # rule 6
            continue
        dst = os.path.join(stage_dir, rel)
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            tmp = f"{dst}.tmp{os.getpid()}"
            with open(tmp, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, dst)
        except OSError as exc:
            skipped[rel] = f"stage-failed: {exc}"
            continue
        staged.append(rel)
    _ckpt_tail_write_state(state_path, state)
    return {"staged": staged, "skipped": skipped}


# --- box-lifecycle stream (jobs/nodes/<IID>/events/) --------------------------
# A per-box append-only log with the SAME immutable-object + pure-fold discipline
# as the per-job stream, but keyed on instance_id. Separate namespace + separate
# event set, so nothing here can perturb the frozen per-job schema or its fold.
def make_box_event(instance_id, event, actor=None, ts=None, **fields) -> dict:
    iid = str(instance_id)
    ev = {
        "v": SCHEMA_VERSION,
        "ts": ts or now_ts(),
        "actor": actor or f"box:{iid}",
        "event": event,
        "instance_id": iid,
        "nonce": fields.pop("nonce", None) or nonce(),
    }
    ev.update({k: v for k, v in fields.items() if v is not None})
    return ev


def emit_box_event(instance_id, event, *, actor=None, runner=_default_runner,
                   bucket=None, **fields) -> dict:
    """Append one immutable box-lifecycle event to jobs/nodes/<IID>/events/.
    Best-effort (same contract as emit_event): a transport failure returns the
    event with `_emitted=False` so a parking box's final emit can't crash."""
    ev = make_box_event(instance_id, event, actor=actor, **fields)
    key = f"jobs/nodes/{ev['instance_id']}/events/{event_key(ev)}"
    body = json.dumps(ev, separators=(",", ":")) + "\n"
    rc, _, err = runner(["rcat", _wq(bucket, key)], input=body)   # box-side write
    ev["_key"] = key
    ev["_emitted"] = (rc == 0)
    if rc != 0:
        ev["_error"] = (err or "").strip()
    return ev


def _coerce_box(raw) -> dict | None:
    """Return a valid box event dict, or None. One bad object never breaks a fold."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    if not all(raw.get(k) for k in _BOX_CORE_KEYS):
        return None
    return raw


def fold_box_events(raw_events) -> dict:
    """Fold a box's lifecycle log into a view. Tolerant to missing/extra/duplicate/
    out-of-order objects (never mutates B2). `parked_self` is sticky-terminal (the
    box is parked until an operator resumes + re-attaches)."""
    evs, parse_errors = [], 0
    for r in raw_events:
        e = _coerce_box(r)
        if e is None:
            parse_errors += 1
        else:
            evs.append(e)
    evs.sort(key=lambda e: (e.get("ts", ""), e.get("nonce", "")))
    view = {
        "instance_id": evs[-1]["instance_id"] if evs else None,
        "parked": False, "parked_ts": None, "park_reason": None,
        "n_done": None, "n_failed": None, "idle_s": None,
        "drained_pending": False,
        "last_event": None, "last_event_ts": None,
        "n_events": len(evs), "parse_errors": parse_errors,
    }
    if not evs:
        return view
    view["last_event"] = evs[-1].get("event")
    view["last_event_ts"] = evs[-1].get("ts")
    park = [e for e in evs if e.get("event") == "parked_self"]
    if park:
        p = park[-1]
        view["parked"] = True
        view["parked_ts"] = p.get("ts")
        view["park_reason"] = p.get("reason")
        view["n_done"] = _num(p.get("n_done"))
        view["n_failed"] = _num(p.get("n_failed"))
        view["idle_s"] = _num(p.get("idle_s"))
    # `drained` without a later `parked_self` = the box asked the laptop to park it
    drained = [e for e in evs if e.get("event") == "drained"]
    if drained and not park:
        view["drained_pending"] = True
        view["park_reason"] = drained[-1].get("reason")
    return view


def read_box(instance_id, *, runner=_default_runner, bucket=None, cache_dir=None) -> dict:
    """Fold one box's lifecycle log into a view (incremental local cache of the
    immutable event bodies, same shape as read_job)."""
    iid = str(instance_id)
    b = _bucket(bucket)
    dst = os.path.join(event_cache_root(cache_dir), "nodes", iid, "events")
    os.makedirs(dst, exist_ok=True)
    runner(["copy", f"b2:{b}/jobs/nodes/{iid}/events/", dst,
            "--transfers", "16", "--checkers", "32", "--fast-list"])
    bodies = []
    try:
        for name in os.listdir(dst):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(dst, name), "rb") as fh:
                        bodies.append(fh.read())
                except OSError:
                    pass
    except OSError:
        pass
    return fold_box_events(bodies)


# --- tiny CLI (manual poking; box-side jobd.py is the real box entry) ---------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="jobmeta — B2 job metadata")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("sha"); ps.add_argument("dir")
    pf = sub.add_parser("fold"); pf.add_argument("job_id")
    a = ap.parse_args()
    if a.cmd == "sha":
        print(bundle_sha256(a.dir))
    elif a.cmd == "fold":
        print(json.dumps(read_job(a.job_id), indent=2))
