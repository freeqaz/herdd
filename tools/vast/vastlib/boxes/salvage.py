"""vastlib.boxes.salvage — instance -> instance disk salvage for an EVICTED box.

Why this module exists
----------------------
When fleetd's eviction ladder replaces a lost `jobs` box it retains the dead one
for a window so its unsynced disk can be recovered
(`herdd._job_retain_or_destroy`). The runbook that window pointed at
(`RETENTION_SALVAGE.md`) assumed reading that disk required RESUMING the box,
i.e. re-winning its GPUs. That assumption is false:
`DISK_ACCESS_FINDINGS_2026-08-05.md` establishes that vast.ai serves filesystem
access to an `exited` instance with **no GPU contract entered** — billing docs:
*"GPU charges begin when the instance reaches the `running` state"*.

Two server-orchestrated paths run against a stopped instance:

  survey            PUT /api/v0/instances/command/{id}/   (`ls`, `rm`, `du` only)
  disk -> disk      PUT /api/v0/commands/copy_direct/     (instance -> instance)

**`execute` is refused on a RUNNING instance.** OBSERVED 2026-08-05 against live
box 46866095: `400 invalid_args — "Execute command only avail on stopped
instances. Use ssh to run commands on running instances."` The research readout
this module was built from called that restriction stale; it is not, it is
enforced server-side today. It matters because the two ends of a salvage are in
OPPOSITE states by construction — the source is the stopped dead box, the
destination is a running box we own — so the two surveys need two different
transports (`execute` for the source, ssh for the destination). Verification is
not optional here, so a single-transport design would have reported
`salvage_unverifiable` on every real salvage: the fail-safe holding, and the
happy path never once occurring.

We build on **instance -> instance**, not `cloud copy`, for two reasons the
findings doc spells out: `cloud copy` needs a cloud connection registered on the
vast account (we have none — `GET /users/cloud_integrations/` -> `[]`) and it
STAGES OUR B2 CREDENTIALS ONTO THE HOST MACHINE, which is exactly what the
ephemeral/scoped-key model in `CREDENTIAL_LIFECYCLE.md` exists to avoid.
Instance -> instance has neither problem and it fits the ladder we already run:
fleetd rents a replacement box on eviction, so the replacement pulls the dead
box's disk and the push to B2 goes through OUR b2x with OUR keys.

OBSERVED end-to-end, 2026-08-05, against `exited` box 46861081 at $0 (no GPU
contract entered): `ls -1 /workspace/jobs` in 1.6 s, `ls -lR .../work` in 2.7 s
parsing to 6 sections with ZERO unaccounted lines, yielding
`out/checkpoint-100` and `out/checkpoint-50` at 0.98 GB / 12 files each — and a
`plan_salvage` of `nothing_newer` that matches an independent read of B2. The
`copy_direct` leg remains DOCUMENTED-not-OBSERVED: no evicted box since has held
anything worth moving.

The constraint that shapes everything here
------------------------------------------
**The race is HOST RECLAMATION, not GPU availability.** Box 46859541 was GONE
~30 min after its eviction (`{"instances": null}`, `execute` -> 404) — far inside
the 2h reap threshold and far inside a 3h retention window — taking an unsynced
`checkpoint-50` with it. A retention *window* is therefore the wrong instrument
on its own: it protects against the operator being slow, and the operator being
slow is not what loses the data. Salvage must fire within SECONDS of noticing an
eviction, which is why it is wired into the ladder rather than left to a runbook.

Losslessness
------------
`copy_direct` is FIRE-AND-FORGET: the API returns `{"success": true}` when the
transfer is *initiated* ("check instance status bar for progress updates (~30
seconds delayed)"), never when it finishes, and there is no completion callback.
So "the API said success" is NOT evidence that anything landed. Verification is
therefore a separate, mandatory phase: re-survey the DESTINATION with `execute
ls -l` and require the file-name set and per-file byte counts to match the source
survey exactly. Three outcomes, deliberately distinct:

  * `ok`           — name set and every size match. Only this counts as salvaged.
  * `partial`      — something landed, but not all of it. LOUD, never folded into
                     success: a truncated checkpoint that is *trusted* is worse
                     than no checkpoint at all, because a resume will load it.
  * `unverifiable` — we could not read the destination back. Also never success.

The parsing/policy/state-machine half of this module is PURE (parsing + policy +
a transport-injected orchestrator). Every side effect it takes is a callable
passed in, so the whole state machine is testable with no vast API, no ssh and
no B2 — the portable-lane discipline the rest of `tools/vast/` follows. The
`_salvage_*` / `_mk_salvage_*` block at the bottom is the other half: the real
transports and readers that SUPPLY those callables, moved down out of
`herdd.py` so that the injection seam has both of its sides in one file.

What is deliberately NOT here
-----------------------------
* **The tick drivers.** `_job_salvage_start`, `_job_salvage_advance`,
  `_job_salvage_sweep`, `_salvage_defer_until` and `SALVAGE_DEFER_GRACE_S` sit
  immediately after this code in `herdd.py` and are NOT part of it: they
  mutate the job-context dict, journal through `supervise.journal`, read
  `_job_replacement_knob`, and print the operator lines keyed off
  `LOUD_OUTCOMES`. They belong to `supervise/retention.py` and move with the
  knot (plan §8 step 4). This module answers "advance this record one step",
  never "when should a record exist".
* **`cmd_salvage` / `_add_salvage_args`** and the `salvage` argparse block —
  `cli/` at step 6. Note that they interpolate `SALVAGE_KEEP_N`,
  `SALVAGE_MAX_GB` and `SALVAGE_DEADLINE_S` into help text, so those constants
  are inputs to the §6 CLI-surface byte diff and must keep their values.
* **No transport of its own.** `boxes/remote.py` owns `execute`/ssh/copy and
  `storage/b2.py` owns rclone; both are called as module attributes so the
  suite's patch sites keep steering.
* **No repurposing of an `OUTCOME_*` string, ever.** They land in the fleetd
  journal and in `retained_boxes[].salvage.outcome`: a wire schema. Add, never
  repurpose. Same for `new_record`'s 15-key dict, which is persisted into
  fleetd's `state.json` (a plan §4 load-compat contract), and for
  `b2_salvage_prefix`'s `jobs/<JOB_ID>/salvage/<IID>/<ckpt>` — never
  `.../checkpoints/`, which the replacement job is a LIVE WRITER of.

Provenance: `tools/vast/salvage.py` absorbed whole (plan §3's list, §8 step 3),
plus the ten salvage transport-glue functions from `tools/vast/herdd.py`
(`_mk_salvage_dest_exec` .. `_salvage_dest_candidates`), 2026-08-16.
Behavior-preserving: bodies copied, annotations added, and the five
`collections.namedtuple` calls respelled as `typing.NamedTuple` classes (same
runtime object model — tuple subclass, `._fields`, positional construction —
with declared field types; the equivalence is pinned field-for-field in
`test_vastlib_boxes_salvage.py`). Every symbol carries its `# moved-from:`
marker (grammar: `vastlib/README.md` §2). The flat `salvage.py` and the
`herdd.py` copies stay live until steps 6-7; `herdd.py:71` still does
`import salvage`, and `test_salvage.py` still drives both halves there.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import subprocess
from typing import Any, Callable, Iterable, Mapping, MutableMapping, NamedTuple, Sequence

from vastlib.boxes import lifecycle, remote, ssh
from vastlib.core import api
from vastlib.storage import b2

#: One salvage record: the dict `new_record` builds, which fleetd persists into
#: `state.json` and the handoff journal. Kept as a plain mutable mapping rather
#: than a dataclass precisely because it is a PERSISTED shape — plan §4 freezes
#: it, and `supervise/state.py` has to stay dict-compatible with it.
Record = MutableMapping[str, Any]

#: The injected transports (see `advance`). Spelled with `...` argument lists
#: because the same callable is passed positionally here and by keyword in the
#: suite's fakes; the RESULT shape is what this module actually depends on, and
#: `boxes.remote`'s `Soft` NamedTuple satisfies each of them by construction.
ExecFn = Callable[..., tuple[bool, str, str | None]]
CopyFn = Callable[..., tuple[bool, str, str | None]]
PrepareFn = Callable[..., bool]
CopyStatusFn = Callable[..., str | None]
B2BytesFn = Callable[..., Mapping[str, int] | None]
PushFn = Callable[..., tuple[bool, str]]

# --- vocabulary ------------------------------------------------------------- #
# Frozen outcome names. They land in the fleetd journal and in
# `retained_boxes[].salvage.outcome`, so treat them as a wire schema: add, never
# repurpose. Kept DISTINCT for the same reason `retention_lost` is not folded
# into `reaped` — each one means something different about whether the mechanism
# worked, and collapsing them hides the failure rate.
# moved-from: salvage.OUTCOME_SALVAGED
OUTCOME_SALVAGED = "salvaged"                # copy verified complete on the dest
# moved-from: salvage.OUTCOME_PARTIAL
OUTCOME_PARTIAL = "salvaged_partial"         # landed short — DO NOT TRUST IT
# moved-from: salvage.OUTCOME_UNVERIFIABLE
OUTCOME_UNVERIFIABLE = "salvage_unverifiable"  # copy started, dest never readable
# moved-from: salvage.OUTCOME_NOTHING_NEWER
OUTCOME_NOTHING_NEWER = "nothing_newer"      # B2 already holds it all
# moved-from: salvage.OUTCOME_NOTHING_FOUND
OUTCOME_NOTHING_FOUND = "nothing_found"      # no checkpoint on the dead disk
# moved-from: salvage.OUTCOME_DEAD_GONE
OUTCOME_DEAD_GONE = "dead_box_gone"          # host reclaimed it before we looked
# moved-from: salvage.OUTCOME_DEST_NOT_READY
OUTCOME_DEST_NOT_READY = "dest_not_ready"    # replacement never became addressable
# moved-from: salvage.OUTCOME_COPY_REFUSED
OUTCOME_COPY_REFUSED = "copy_refused"        # copy_direct answered success:false
# moved-from: salvage.OUTCOME_DISABLED
OUTCOME_DISABLED = "salvage_disabled"        # switched off by knob

# moved-from: salvage.TERMINAL_OUTCOMES
TERMINAL_OUTCOMES = frozenset({
    OUTCOME_SALVAGED, OUTCOME_PARTIAL, OUTCOME_UNVERIFIABLE,
    OUTCOME_NOTHING_NEWER, OUTCOME_NOTHING_FOUND, OUTCOME_DEAD_GONE,
    OUTCOME_DEST_NOT_READY, OUTCOME_COPY_REFUSED, OUTCOME_DISABLED,
})
#: Outcomes where bytes we believe we needed are NOT safely in hand. Callers
#: print these loudly; `salvaged` and `nothing_newer` are the only quiet ones.
# moved-from: salvage.LOUD_OUTCOMES
LOUD_OUTCOMES = frozenset(TERMINAL_OUTCOMES) - {OUTCOME_SALVAGED,
                                                OUTCOME_NOTHING_NEWER,
                                                OUTCOME_DISABLED}

# --- knobs ------------------------------------------------------------------ #
# moved-from: salvage.SALVAGE_KEEP_N
SALVAGE_KEEP_N = 1          # newest N checkpoint dirs per job. One is the resume
                            # point; the older ones are dose-curve evidence that
                            # a healthy run already synced to B2 on its own.
# moved-from: salvage.SALVAGE_MAX_GB
SALVAGE_MAX_GB = 12.0       # refuse to initiate a transfer larger than this
                            # without an explicit override. A checkpoint is
                            # ~0.98 GB (adapter 323 MB + optimizer 646 MB +
                            # tokenizer 11 MB), so this is ~12 of them: generous
                            # for the real case, a fuse against `ls` returning
                            # something we misparsed as a terabyte.
# moved-from: salvage.SALVAGE_DEADLINE_S
SALVAGE_DEADLINE_S = 1800.0  # give up on a copy that has not verified in 30 min.
                            # Measured shape: ~1 GB host-to-host. The bound
                            # exists because there is no completion signal at
                            # all, so an un-deadlined wait is an infinite one.
# moved-from: salvage.SALVAGE_DEST_WAIT_S
SALVAGE_DEST_WAIT_S = 900.0  # bounded wait for the replacement to become
                            # addressable. Past this the outcome is
                            # `dest_not_ready` — reported, never hung on.
# moved-from: salvage.SALVAGE_ROOT
SALVAGE_ROOT = "/workspace/salvage"   # where salvaged bytes land on the DEST.
                            # Deliberately NOT under /workspace/jobs/<ID>/: the
                            # replacement is a LIVE WRITER there and its jobd
                            # resume pull-back reads that tree.
# moved-from: salvage.JOBS_ROOT
JOBS_ROOT = "/workspace/jobs"         # jobd's JOBS_DIR default (JOBD_ROOT/jobs)


# The five result shapes, respelled from `collections.namedtuple` to
# `typing.NamedTuple` (the `core.models.MarketRead` precedent): identical
# runtime object model — tuple subclass, `._fields`, `._replace`, positional and
# keyword construction, `==` against a bare tuple — with the field types
# declared, which is what lets strict mypy see through `plan.items` and
# `v.bytes_seen` instead of collapsing them to `Any`.
# moved-from: salvage.LsEntry
class LsEntry(NamedTuple):
    name: str
    size: int
    is_dir: bool


# moved-from: salvage.CkptDir
class CkptDir(NamedTuple):
    name: str
    step: int
    #: Total size of `files`. Shadows the builtin inside the class body only,
    #: exactly as the namedtuple field did.
    bytes: int
    files: dict[str, int]


# moved-from: salvage.SalvagePlan
class SalvagePlan(NamedTuple):
    action: str
    items: tuple[CkptDir, ...]
    bytes: int
    reason: str


# moved-from: salvage.Verification
class Verification(NamedTuple):
    status: str
    missing: tuple[str, ...]
    short: tuple[str, ...]
    bytes_seen: int
    bytes_expected: int
    reason: str


# moved-from: salvage._LS_LINE
_LS_LINE = re.compile(
    r"^(?P<mode>[-dlbcps][-rwxSsTt]{9}[+.@]?)\s+"
    r"\d+\s+\S+\s+\S+\s+"
    r"(?P<size>\d+)\s+"
    r"(?P<rest>.+)$")
# moved-from: salvage._CKPT
_CKPT = re.compile(r"^checkpoint-(\d+)$")


# --- parsing (PURE) --------------------------------------------------------- #

# moved-from: salvage._LS_ABSENT
_LS_ABSENT = re.compile(
    r"^ls: cannot (?:access|open directory) .*: No such file or directory\s*$")


# moved-from: salvage.parse_ls_l
def parse_ls_l(text: str | None) -> list[LsEntry]:
    """`ls -l` output -> [LsEntry]. See `parse_ls_l_strict` for the residual."""
    return parse_ls_l_strict(text)[0]


# moved-from: salvage.parse_ls_l_strict
def parse_ls_l_strict(text: str | None) -> tuple[list[LsEntry], int, int]:
    """`ls -l` output -> `(entries, residual, absent)`.

    `residual` is the count of non-blank lines that were neither a `total N`
    header nor a parsed entry, EXCLUDING the "no such file" errors counted in
    `absent`. It exists because a dropped line is NOT harmless in both
    directions:

      * for `plan_salvage`'s size fuse, under-reporting is the safe direction;
      * for `verify_salvage`, the SOURCE survey is the only oracle. A source map
        that silently lost a file cannot notice that the copy lost it too — a
        truncated `result_url` body, or one unparsed line, is enough to turn an
        incomplete copy into a reported `salvaged`, after which those bytes get
        pushed to B2 under that label.

    So callers that verify must refuse to trust a survey with `residual > 0`.
    `absent` is tracked separately because "the path is not there" is a real,
    useful answer (the copy has not landed), not a parse failure.
    """
    out: list[LsEntry]                    # annotation only: mypy cannot
    out, residual, absent = [], 0, 0     # infer an empty list's element type
    for raw in (text or "").splitlines():
        line = raw.rstrip("\n").strip()
        if not line:
            continue
        if line.startswith("total ") and line[6:].strip().isdigit():
            continue
        m = _LS_LINE.match(line)
        if not m:
            if _LS_ABSENT.match(line):
                absent += 1
            else:
                residual += 1
            continue
        rest = m.group("rest").strip()
        # strip the date field: "Aug  5 07:00 name" / "Aug  5  2026 name" /
        # (ls --full-time) "2026-08-05 07:00:00.000000000 +0000 name"
        name = _strip_ls_date(rest)
        if name is None:
            residual += 1
            continue
        if name in (".", ".."):
            continue
        if " -> " in name:                    # symlink: keep the link name only
            name = name.split(" -> ", 1)[0]
        out.append(LsEntry(name, int(m.group("size")),
                           m.group("mode").startswith("d")))
    return out, residual, absent


# moved-from: salvage._strip_ls_date
def _strip_ls_date(rest: str) -> str | None:
    """Drop `ls -l`'s date columns from the tail of a line, return the filename.

    Handles the two GNU shapes (`Mon DD HH:MM`, `Mon DD  YYYY`) and
    `--full-time`. Returns None when nothing recognisable is there — the caller
    then skips the line rather than inventing a name.
    """
    parts = rest.split()
    if len(parts) >= 4 and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]):
        drop = 3 if re.match(r"^[+-]\d{4}$", parts[2]) else 2
        return " ".join(parts[drop:]) or None
    if len(parts) >= 4:
        return " ".join(parts[3:]) or None
    return None


# moved-from: salvage.parse_ls_lr
def parse_ls_lr(text: str | None) -> dict[str, list[LsEntry]]:
    """`ls -lR` output -> {dirpath: [LsEntry]}. See `parse_ls_lr_strict`."""
    return parse_ls_lr_strict(text)[0]


# moved-from: salvage.parse_ls_lr_strict
def parse_ls_lr_strict(text: str | None
                       ) -> tuple[dict[str, list[LsEntry]], int, int]:
    """`ls -lR` output -> `(sections, residual, absent)`.

    GNU `ls -lR` emits a `<path>:` header before each directory's block. The
    first block may have no header (the argument itself when it is the only
    directory); it is filed under "". `residual`/`absent` aggregate across every
    block — see `parse_ls_l_strict` for why a verifier must not trust a survey
    with a non-zero residual.
    """
    out: dict[str, list[LsEntry]]        # annotations only, as above
    buf: list[str]
    out, cur = {}, ""
    buf = []
    residual = absent = 0

    def flush() -> None:
        nonlocal residual, absent
        ent, r, a = parse_ls_l_strict("\n".join(buf))
        out.setdefault(cur, []).extend(ent)
        residual += r
        absent += a

    for raw in (text or "").splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped.endswith(":") and not _LS_LINE.match(stripped) \
                and not _LS_ABSENT.match(stripped) \
                and ("/" in stripped or stripped in (".:", "..:")):
            flush()
            buf = []
            cur = stripped[:-1]
            out.setdefault(cur, [])
            continue
        buf.append(line)
    flush()
    return ({k: v for k, v in out.items() if k != "" or v}, residual, absent)


# moved-from: salvage.checkpoint_step
def checkpoint_step(name: str | None) -> int | None:
    """`checkpoint-50` -> 50; anything else -> None. The HF trainer's naming, and
    the same pattern jobd's prune/scrub keys off."""
    m = _CKPT.match((name or "").strip())
    return int(m.group(1)) if m else None


# moved-from: salvage.split_ckpt_rel
def split_ckpt_rel(rel: str | None
                   ) -> tuple[str | None, int | None, str | None]:
    """Split a path RELATIVE to a job's work root at its `checkpoint-<N>` segment.

    Returns `(ckpt_dir_rel, step, remainder)` or `(None, None, None)`.

    The checkpoint is NOT at a fixed depth. jobd's real tree is
    `work/out/checkpoint-50/...` (the training output dir sits between), and a
    multi-arm bundle produces `work/arms/<name>/checkpoint-50/...` — the same
    per-layout-root shape jobd's own prune groups by. Keying on the FIRST path
    segment therefore finds nothing at all on a real box, which is a silent
    `nothing_found` on a disk that is full of checkpoints. Measured against the
    live bucket 2026-08-05: 22,577 of 22,898 checkpoint objects sit under an
    `out/` level.

    The LAST `checkpoint-<N>` segment wins, so a checkpoint nested inside another
    (which should not happen) still resolves to the innermost one rather than
    merging two trees.
    """
    parts = [p for p in (rel or "").split("/") if p]
    for i in range(len(parts) - 1, -1, -1):
        step = checkpoint_step(parts[i])
        if step is not None:
            return "/".join(parts[:i + 1]), step, "/".join(parts[i + 1:])
    return None, None, None


# moved-from: salvage.ckpt_dirs_from_survey
def ckpt_dirs_from_survey(sections: Mapping[str, Sequence[LsEntry]] | None,
                          root: str) -> list[CkptDir]:
    """Fold a `parse_ls_lr` result rooted at `root` into [CkptDir], newest first.

    Only `checkpoint-<N>` directories are considered, AT ANY DEPTH under `root`
    (see `split_ckpt_rel`). `CkptDir.name` is the path relative to `root`, e.g.
    `out/checkpoint-50`. Nested subdirectories of a checkpoint are INCLUDED in
    the byte total and in the verified file set, keyed by their path relative to
    the checkpoint dir — a partial copy that dropped a nested file must not read
    as complete.
    """
    root = root.rstrip("/")
    dirs: dict[str, dict[str, Any]] = {}
    for path, entries in (sections or {}).items():
        p = (path or "").rstrip("/")
        if not p.startswith(root + "/"):
            continue
        head, step, sub = split_ckpt_rel(p[len(root) + 1:])
        if step is None:
            continue
        # `head` is None exactly when `step` is (split_ckpt_rel returns the
        # all-None triple or an all-populated one), and the line above has
        # already returned in that case — a correlation no annotation on a
        # 3-tuple return can express. Narrowing it with an assert would add a
        # runtime check the ported body never had.
        d = dirs.setdefault(head, {"step": step, "files": {}})  # type: ignore[arg-type]
        for e in entries:
            if e.is_dir:
                continue
            d["files"][f"{sub}/{e.name}" if sub else e.name] = e.size
    return sorted(
        (CkptDir(name, v["step"], sum(v["files"].values()), dict(v["files"]))
         for name, v in dirs.items()),
        key=lambda c: c.step, reverse=True)


# --- policy (PURE) ---------------------------------------------------------- #

# moved-from: salvage.plan_salvage
def plan_salvage(ckpts: Sequence[CkptDir] | None, *,
                 b2_bytes: Mapping[str, int] | None = None,
                 # Deliberately `Any`, not a union: both are read straight out of a
                 # PERSISTED record (`rec.get("keep_n")`), so the honest static
                 # type is "whatever state.json held". The body coerces with
                 # int()/float() exactly as it always did.
                 keep_n: Any = SALVAGE_KEEP_N,  # noqa: ANN401
                 max_gb: Any = SALVAGE_MAX_GB) -> SalvagePlan:  # noqa: ANN401
    """What (if anything) is worth pulling off the dead disk. PURE.

    `b2_bytes` maps `checkpoint-<N>` -> total bytes ALREADY on B2 under
    `jobs/<JOB_ID>/checkpoints/`. `None` means "we could not read B2" — and that
    is deliberately NOT treated as "B2 has nothing": an unreadable B2 makes us
    copy MORE, never less. The `optimized` requirement is served by copying the
    newest `keep_n` checkpoints rather than the whole disk (one is ~0.98 GB and
    a long run leaves many).

    A checkpoint counts as already-safe only when B2 holds it at >= the byte
    total we just measured on disk. A short object on B2 is a torn sync, which
    is precisely the case salvage exists for.
    """
    if not ckpts:
        return SalvagePlan("nothing", (), 0,
                           "no checkpoint-<N> directory on the dead disk — "
                           "nothing this tool knows how to salvage")
    have = b2_bytes if isinstance(b2_bytes, dict) else {}
    b2_known = isinstance(b2_bytes, dict)
    fresh = [c for c in ckpts if c.bytes > int(have.get(c.name, -1))]
    if not fresh:
        newest = ckpts[0]
        return SalvagePlan("nothing", (), 0,
                           f"B2 already holds every checkpoint on that disk "
                           f"(newest {newest.name}, {newest.bytes} B) — nothing "
                           f"was lost")
    pick = tuple(fresh[:max(1, int(keep_n))])
    total = sum(c.bytes for c in pick)
    cap = int(float(max_gb) * 1e9)
    if total > cap:
        return SalvagePlan(
            "refuse", (), total,
            f"planned salvage is {total / 1e9:.2f} GB, over the "
            f"{float(max_gb):.2f} GB fuse — raise --salvage-max-gb deliberately "
            f"or narrow --salvage-keep-n (a checkpoint is ~0.98 GB, so this is "
            f"either a very large model or a misparsed survey)")
    why = ", ".join(f"{c.name} ({c.bytes / 1e9:.2f} GB)" for c in pick)
    tail = "" if b2_known else " (B2 side UNREADABLE — copying anyway)"
    return SalvagePlan("copy", pick, total,
                       f"newest {len(pick)} of {len(fresh)} checkpoint(s) not "
                       f"safe on B2: {why}{tail}")


# moved-from: salvage.verify_salvage
def verify_salvage(src_files: Mapping[str, int] | None,
                   dst_files: Mapping[str, int] | None) -> Verification:
    """Did the copy land LOSSLESSLY? PURE, and fail-closed.

    `src_files` / `dst_files` are `{relpath: size}` from the two surveys.
    `dst_files is None` means the destination survey FAILED — that is
    `unverifiable`, never `ok`. This is the fail-safe the whole design turns on:
    the only thing worse than losing a checkpoint is believing you still have it.
    """
    if dst_files is None:
        return Verification("unverifiable", (), (), 0,
                            sum((src_files or {}).values()),
                            "could not read the destination back — the copy may "
                            "have landed, may be in flight, or may have failed; "
                            "treat these bytes as NOT salvaged")
    want = dict(src_files or {})
    got = dict(dst_files or {})
    expected = sum(want.values())
    seen = sum(min(got.get(k, 0), v) for k, v in want.items())
    missing = tuple(sorted(k for k in want if k not in got))
    short = tuple(sorted(k for k, v in want.items()
                         if k in got and got[k] != v))
    if not want:
        return Verification("unverifiable", (), (), 0, 0,
                            "the source survey listed no files, so there is "
                            "nothing to verify against")
    if not got:
        return Verification("partial", missing, short, 0, expected,
                            "destination path is EMPTY — the copy has not "
                            "landed (yet)")
    if missing or short:
        return Verification(
            "partial", missing, short, seen, expected,
            f"INCOMPLETE: {len(missing)} file(s) missing, {len(short)} at the "
            f"wrong size, {seen}/{expected} B present — a resume that loads "
            f"this would load a TORN checkpoint")
    return Verification("ok", (), (), seen, expected,
                        f"name set and all {len(want)} byte counts match "
                        f"({expected} B)")


# moved-from: salvage.dest_path
def dest_path(dead_iid: int | str, job_id: str | None, ckpt_name: str,
              root: str = SALVAGE_ROOT, flat: bool = False) -> str:
    """The `dst_path` handed to `copy_direct` — the directory the checkpoint
    lands INSIDE. Use `landed_path()` for where the bytes end up.

    Namespaced by the dead instance id so two salvages never collide, and kept
    entirely outside `/workspace/jobs/` because the destination's jobd owns that
    tree and reads it back on resume.

    **`copy_direct` does not `mkdir -p`.** OBSERVED 2026-08-05: a copy into
    `/workspace/salvage/<iid>/<job>/out/checkpoint-100` on a box with no
    `/workspace/salvage` failed with

        rsync: [Receiver] mkdir ".../out/checkpoint-100" (in C.46864611)
        failed: No such file or directory (2)

    while the API had already answered `success: true` — the fire-and-forget
    property biting exactly as designed for. rsync creates the FINAL component
    only, so every parent must already exist. Two layouts follow:

      * `flat=False` (preferred) — the readable nested tree. Requires a
        `prepare_dest` that can `mkdir -p` the parents, which in practice means
        an ssh-reachable RUNNING destination.
      * `flat=True` — one new component directly under an existing root, so it
        needs no directory creation at all. Uglier, and the only thing that works
        when the destination is STOPPED: `execute` allows `ls`/`rm`/`du` and
        there is no `mkdir` in that list.
    """
    if flat:
        slug = f"{job_id}-{ckpt_name}".replace("/", "_")
        return f"{root.rstrip('/')}-{dead_iid}-{slug}"
    parent = ckpt_name.rsplit("/", 1)[0] if "/" in (ckpt_name or "") else ""
    base = f"{root.rstrip('/')}/{dead_iid}/{job_id}"
    return f"{base}/{parent}" if parent else base


# moved-from: salvage.landed_path
def landed_path(target: str, ckpt_name: str | None) -> str:
    """Where the bytes actually ARE once `copy_direct` finishes.

    `copy_direct` is rsync WITHOUT a trailing slash on the source, so it copies
    the source directory *into* the destination rather than copying its contents
    *onto* it: `copy_direct(SRC=.../out/checkpoint-100, DST=X)` lands at
    `X/checkpoint-100`. OBSERVED 2026-08-05.

    This is the difference between a verified salvage and a permanent
    `salvaged_partial`: verifying at `X` finds one directory and zero files
    forever, which the verifier correctly refuses to call success — so the
    fail-safe would have masked a pure path bug as a data problem.
    """
    return (f"{target.rstrip('/')}/"
            f"{(ckpt_name or '').rstrip('/').rsplit('/', 1)[-1]}")


# moved-from: salvage.b2_salvage_prefix
def b2_salvage_prefix(job_id: str | None, dead_iid: int | str,
                      ckpt_name: str) -> str:
    """Where salvaged bytes go on B2: `jobs/<JOB_ID>/salvage/<IID>/<ckpt>/`.

    NEVER `jobs/<JOB_ID>/checkpoints/` — the replacement job is a LIVE WRITER
    there and its resume pull-back reads it. Salvage is evidence to inspect, not
    state to silently re-inject into a running job.
    """
    return f"jobs/{job_id}/salvage/{dead_iid}/{ckpt_name}"


# --- orchestrator (transports injected) ------------------------------------- #

# Declared in the flat module and never constructed anywhere in-tree (pinned as
# latent finding H14d by the port manifest: kept, not deleted, because deleting
# an unused public name is a behavior change in a port step). Its fields stay
# `Any` for the same reason — typing them concretely would invent a contract no
# call site has ever agreed to.
# moved-from: salvage.Step
class Step(NamedTuple):
    phase: Any
    outcome: Any
    plan: Any
    verification: Any
    detail: Any
    commands: Any


#: An error string that means the instance is really gone, as opposed to the
#: request merely not working this time. `herdd._classify_http` already draws
#: this line for HTTP; these are the shapes the salvage transport surfaces.
# moved-from: salvage._FATAL_SURVEY
_FATAL_SURVEY = ("404", "not_found", "not found", "no_such_instance",
                 "instance not found", "invalid instance")


# moved-from: salvage.survey_is_fatal
def survey_is_fatal(err: object) -> bool:
    """Does this survey error mean the disk is GONE, or just not readable now?

    THE distinction the whole retry story rests on. A poll timeout, a 5xx, or a
    transient network error must NOT be reported as `dead_box_gone` — that is an
    authoritative "everything on that disk is lost", and an operator who reads it
    will not re-run salvage. Only an answer that names the instance as absent
    earns that verdict; everything else is retried until the record's deadline.
    """
    low = str(err or "").lower()
    return any(tok in low for tok in _FATAL_SURVEY)


# moved-from: salvage.survey_dead_box
def survey_dead_box(dead_iid: int | str, *, execute: ExecFn,
                    jobs_root: str = JOBS_ROOT, job_id: object = None
                    ) -> tuple[dict[str, list[CkptDir]], str | None, bool]:
    """Survey the dead disk. Returns `(job_ckpts, err, fatal)`.

    `job_ckpts` is `{job_id: [CkptDir]}`. `err` is a string when the survey did
    not complete; `fatal` says whether that error means the instance is GONE
    (`dead_box_gone`) rather than merely unreadable this tick (retry).

    A per-job `ls -lR` that fails, or that comes back with UNPARSED lines, is an
    error too — not a job quietly skipped. Skipping it would report
    `nothing_found` ("the dead disk answered, but carried no checkpoint") over a
    disk that is full of checkpoints, which is exactly the authoritative false
    negative this module exists to avoid.

    `execute(iid, cmd) -> (ok, text, err)` is the injected transport (in
    production: `PUT /api/v0/instances/command/{id}/` plus its result poll). Only
    `ls`/`rm`/`du` are accepted by that endpoint, so this never asks for more.
    """
    ok, text, err = execute(dead_iid, f"ls -1 {jobs_root}")
    if not ok:
        return {}, (err or "the dead box did not answer a survey"), \
            survey_is_fatal(err)
    jids = [ln.strip() for ln in (text or "").splitlines()
            if ln.strip() and not ln.strip().startswith("total ")
            and "/" not in ln.strip() and not ln.strip().endswith(":")
            and not _LS_ABSENT.match(ln.strip())]
    if job_id is not None:
        jids = [j for j in jids if j == str(job_id)]
    out: dict[str, list[CkptDir]] = {}
    for jid in jids:
        root = f"{jobs_root}/{jid}/work"
        ok, text, err2 = execute(dead_iid, f"ls -lR {root}")
        if not ok:
            return {}, (f"survey of {root} failed: "
                        f"{err2 or 'no answer'}"), survey_is_fatal(err2)
        sections, residual, _absent = parse_ls_lr_strict(text)
        if residual:
            return {}, (f"survey of {root} came back with {residual} unparsed "
                        f"line(s) — a partial listing cannot be the oracle a "
                        f"byte-for-byte verification is checked against"), False
        ck = ckpt_dirs_from_survey(sections, root)
        if ck:
            out[jid] = ck
    return out, None, False


# moved-from: salvage.survey_dest_files
def survey_dest_files(dest_iid: int | str, path: str, *,
                      execute: ExecFn) -> dict[str, int] | None:
    # NOTE: `execute` here is the DESTINATION transport, which is NOT the same
    # thing as the dead box's. See `advance(dest_execute=...)` — vast refuses
    # `execute` on a running instance, and the destination is running by
    # construction.
    """`{relpath: size}` for a salvaged checkpoint on the destination, or None.

    Three distinct answers, and keeping them distinct is the point:

      * `None`  — could not read (transport failure, or output we could not fully
                  parse). `verify_salvage` turns this into `unverifiable`.
      * `{}`    — read fine, path is NOT THERE (`ls: cannot access …: No such
                  file or directory`) or is empty. The copy has not landed.
      * `{...}` — what is actually on disk.

    The `{}` case is decided on the `ls` ERROR TEXT, not on the transport's `ok`.
    The vast `execute` endpoint returns HTTP 200 with the command's stderr in the
    body — it never surfaces the command's exit code — so keying `None` off `ok`
    alone made `unverifiable` almost unreachable and collapsed a three-outcome
    design into two.
    """
    ok, text, _ = execute(dest_iid, f"ls -lR {path}")
    if not ok:
        return None
    sections, residual, absent = parse_ls_lr_strict(text)
    if residual:
        return None                 # output we cannot fully account for
    base = path.rstrip("/")
    files: dict[str, int] = {}
    for p, entries in sections.items():
        pp = (p or base).rstrip("/")
        if pp != base and not pp.startswith(base + "/"):
            continue
        sub = pp[len(base):].lstrip("/")
        for e in entries:
            if not e.is_dir:
                files[f"{sub}/{e.name}" if sub else e.name] = e.size
    if absent and not files:
        return {}                   # the path is not there: not landed (yet)
    return files


# --- the state machine (one bounded step per tick) -------------------------- #
#
# fleetd MUST NOT block. `copy_direct` is asynchronous with no completion signal
# and the destination may take minutes to boot, so salvage is a record advanced
# by the supervising tick rather than a blocking call — the same shape the
# retention sweep and the handoff state machine already use. One `advance()` per
# tick, bounded work, every wait carrying a deadline.

#: `actual_status` values a copy destination can be addressed at. `loading` is
#: excluded on purpose: the container is still materialising, and a copy into a
#: half-built rootfs is the kind of silent partial this module exists to refuse.
# moved-from: salvage.DEST_READY_STATES
DEST_READY_STATES = frozenset({"running"})


# moved-from: salvage.new_record
def new_record(dead_iid: int | str, *, now: float,
               dest_candidates: Iterable[Any] = (), job_id: str | None = None,
               keep_n: float | str = SALVAGE_KEEP_N,
               max_gb: float | str = SALVAGE_MAX_GB,
               deadline_s: float = SALVAGE_DEADLINE_S,
               dest_wait_s: float = SALVAGE_DEST_WAIT_S) -> dict[str, Any]:
    """A fresh salvage record, stamped at the moment eviction is NOTICED.

    Both deadlines start now, not when the destination becomes ready: the thing
    racing us is the host reclaiming the dead disk (~30 min observed, box
    46859541), so the clock that matters is wall-clock since the eviction.
    """
    return {"dead_iid": str(dead_iid), "job_id": job_id,
            "dest_candidates": [str(c) for c in dest_candidates],
            "dest_iid": None, "phase": "pending", "outcome": None,
            "items": [], "started_ts": float(now),
            "dest_deadline_ts": float(now) + float(dest_wait_s),
            "deadline_ts": float(now) + float(deadline_s),
            "keep_n": int(keep_n), "max_gb": float(max_gb),
            "attempts": 0, "detail": None, "bytes": 0, "b2": None}


#: Headroom multiplier on a destination's free disk. A salvage that fills the
#: landing box's disk kills whatever is running on it — and jobs boxes run near
#: full BY DESIGN (that is why jobd prunes), so "it is running" is not evidence
#: that it has room.
# moved-from: salvage.DEST_FREE_MARGIN
DEST_FREE_MARGIN = 1.2


# moved-from: salvage.pick_dest
def pick_dest(candidates: Iterable[Any] | None,
              statuses: Mapping[str, str] | None, *,
              free_gb: Mapping[str, float] | None = None,
              need_bytes: float = 0) -> str | None:
    """First candidate that is copy-ready AND has room. PURE.

    Ordered, so the caller's preference (the replacement box first) wins — but
    ANY box we already own is a valid landing zone, and taking a running one
    beats waiting on a booting one while the dead disk is being reclaimed.

    `free_gb` is `{iid: free_gb}` from the tick's instance snapshot. A candidate
    with UNKNOWN free space is accepted (we cannot prove it is full, and refusing
    every box on missing telemetry would disable salvage entirely); one that is
    known not to fit is skipped.
    """
    need = float(need_bytes or 0) / 1e9 * DEST_FREE_MARGIN
    for c in candidates or ():
        if (statuses or {}).get(str(c)) not in DEST_READY_STATES:
            continue
        have = (free_gb or {}).get(str(c))
        if have is not None and need and float(have) < need:
            continue
        return str(c)
    return None


# moved-from: salvage._finish
def _finish(rec: Record, outcome: str, detail: str) -> Record:
    rec["phase"] = "done"
    rec["outcome"] = outcome
    rec["detail"] = detail
    return rec


# moved-from: salvage.advance
def advance(rec: Record, *, now: float, execute: ExecFn, copy_direct: CopyFn,
            statuses: Mapping[str, str] | None,
            free_gb: Mapping[str, float] | None = None,
            dest_execute: ExecFn | None = None,
            prepare_dest: PrepareFn | None = None,
            copy_status: CopyStatusFn | None = None,
            b2_bytes: B2BytesFn | None = None,
            push_to_b2: PushFn | None = None,
            dry_run: bool = False) -> Record:
    """Advance ONE salvage record by one tick. Returns the record (mutated).

    Injected transports, all of them total (they report failure, never raise):

      execute(iid, cmd)        -> (ok, text, err)     survey the STOPPED dead box
      dest_execute(iid, cmd)   -> (ok, text, err)     survey the RUNNING destination
                                                      (defaults to `execute`)
      prepare_dest(iid, paths) -> bool                mkdir -p the parents; False
                                                      selects the flat layout
      copy_status(iid)         -> str | None          the source instance's
                                                      `status_msg`, where vast
                                                      reports rsync failures
      copy_direct(src, srcp,
                  dst, dstp)   -> (ok, msg, err)      initiate a host-to-host copy
      statuses                 -> {iid: actual_status} this tick's snapshot
      free_gb                  -> {iid: free_gb} | None   landing-zone headroom
      b2_bytes(job_id)         -> {ckpt: bytes} | None  what B2 already holds
      push_to_b2(dest, path,
                 prefix)       -> (ok, detail)        VERIFIED bytes -> our bucket

    A tick never blocks and never retries internally; the deadlines in the record
    are the only thing that ends a wait.
    """
    if rec.get("phase") == "done":
        return rec
    rec["attempts"] = int(rec.get("attempts", 0)) + 1
    if rec.get("phase") == "pending":
        return _advance_pending(rec, now=now, execute=execute,
                                copy_direct=copy_direct, statuses=statuses,
                                b2_bytes=b2_bytes, free_gb=free_gb,
                                prepare_dest=prepare_dest, dry_run=dry_run)
    return _advance_copying(rec, now=now,
                            execute=dest_execute or execute,
                            copy_status=copy_status,
                            push_to_b2=push_to_b2, dry_run=dry_run)


# moved-from: salvage._advance_pending
def _advance_pending(rec: Record, *, now: float, execute: ExecFn,
                     copy_direct: CopyFn, statuses: Mapping[str, str] | None,
                     b2_bytes: B2BytesFn | None,
                     free_gb: Mapping[str, float] | None = None,
                     prepare_dest: PrepareFn | None = None,
                     dry_run: bool = False) -> Record:
    # SURVEY FIRST, choose a destination second. The dead disk is the thing on a
    # countdown (~30 min observed), the destination is not; and the size of the
    # plan is what decides whether a candidate box has room for it.
    ckpts, err, fatal = survey_dead_box(rec["dead_iid"], execute=execute,
                                        job_id=rec.get("job_id"))
    if err is not None:
        rec["last_survey_error"] = err
        if fatal:
            return _finish(rec, OUTCOME_DEAD_GONE,
                           f"the dead box answered that it does not exist "
                           f"({err}) — a stopped spot instance and its disk can "
                           f"be reclaimed by the host within minutes (box "
                           f"46859541 was gone ~30 min after eviction); "
                           f"anything only on it is lost")
        if now >= rec.get("deadline_ts", 0):
            return _finish(rec, OUTCOME_UNVERIFIABLE,
                           f"never got a complete survey of the dead disk "
                           f"before the salvage deadline (last error: {err}). "
                           f"This is NOT evidence the disk was empty — re-run "
                           f"`herdd salvage` before destroying the box")
        # TRANSIENT. Stay pending and retry next tick. Reporting `dead_box_gone`
        # or `nothing_found` here would be an AUTHORITATIVE negative built on a
        # 6-second poll timeout, and an operator who reads "nothing to salvage"
        # does not re-run the command.
        return rec
    if not ckpts:
        return _finish(rec, OUTCOME_NOTHING_FOUND,
                       "the dead disk answered a COMPLETE survey, and carried "
                       "no checkpoint-<N> directory under /workspace/jobs/*/work")

    items: list[dict[str, Any]]          # annotations only: an empty list
    reasons: list[str]                   # literal has no inferable type
    items, planned, reasons = [], 0, []
    for jid, cks in sorted(ckpts.items()):
        have = b2_bytes(jid) if callable(b2_bytes) else None
        plan = plan_salvage(cks, b2_bytes=have, keep_n=rec.get("keep_n"),
                            max_gb=rec.get("max_gb"))
        reasons.append(f"{jid}: {plan.reason}")
        if plan.action == "refuse":
            return _finish(rec, OUTCOME_COPY_REFUSED, plan.reason)
        for c in plan.items:
            items.append({"job_id": jid, "name": c.name, "step": c.step,
                          "bytes": c.bytes, "files": dict(c.files),
                          "status": "queued", "b2": None})
            planned += c.bytes
    rec["items"] = items
    rec["bytes"] = planned
    if not items:
        return _finish(rec, OUTCOME_NOTHING_NEWER, "; ".join(reasons))

    dest = pick_dest(rec.get("dest_candidates"), statuses,
                     free_gb=free_gb, need_bytes=planned)
    if dest is None:
        if now >= rec.get("dest_deadline_ts", 0):
            return _finish(rec, OUTCOME_DEST_NOT_READY,
                           f"no candidate box was `running` with room for "
                           f"{planned / 1e9:.2f} GB inside the salvage "
                           f"destination window — the dead disk was never "
                           f"copied anywhere (it may already be reclaimed)")
        rec["items"] = []            # re-plan next tick against a fresh survey
        return rec
    rec["dest_iid"] = dest

    # `copy_direct` creates the FINAL path component only. Try to make the
    # readable nested tree; fall back to a flat name that needs no mkdir at all
    # (the only option against a STOPPED destination, where `execute` offers
    # `ls`/`rm`/`du` and nothing that creates a directory).
    nested = False
    if callable(prepare_dest):
        parents = sorted({dest_path(rec["dead_iid"], it["job_id"], it["name"])
                          for it in items})
        try:
            nested = bool(prepare_dest(dest, parents))
        except Exception:                                  # noqa: BLE001
            nested = False
    rec["dest_layout"] = "nested" if nested else "flat"
    for it in items:
        it["dest"] = dest_path(rec["dead_iid"], it["job_id"], it["name"],
                               flat=not nested)
        it["landed"] = landed_path(it["dest"], it["name"])

    if dry_run:
        return _finish(rec, OUTCOME_DISABLED,
                       "[dry-run] would copy " + "; ".join(reasons))

    started = 0
    for it in items:
        src = f"{JOBS_ROOT}/{it['job_id']}/work/{it['name']}"
        ok, msg, err2 = copy_direct(rec["dead_iid"], src, dest, it["dest"])
        it["status"] = "copying" if ok else "refused"
        it["copy_msg"] = msg if ok else (err2 or msg)
        started += 1 if ok else 0
    if not started:
        return _finish(rec, OUTCOME_COPY_REFUSED,
                       "vast refused every copy_direct for this box: "
                       + "; ".join(str(i.get("copy_msg")) for i in items))
    rec["phase"] = "copying"
    rec["detail"] = "; ".join(reasons)
    return rec


# moved-from: salvage._copy_status_soft
def _copy_status_soft(copy_status: CopyStatusFn | None,
                      iid: int | str | None) -> str | None:
    """The source instance's `status_msg`, squashed to one line. Never raises and
    never blocks a verdict — it only enriches the report."""
    if not callable(copy_status) or iid is None:
        return None
    try:
        msg = copy_status(iid)
    except Exception:                                      # noqa: BLE001
        return None
    if not msg:
        return None
    return " | ".join(ln.strip() for ln in str(msg).splitlines()
                      if ln.strip())[:400]


# moved-from: salvage._advance_copying
def _advance_copying(rec: Record, *, now: float, execute: ExecFn,
                     push_to_b2: PushFn | None,
                     copy_status: CopyStatusFn | None = None,
                     dry_run: bool = False) -> Record:
    pending = [it for it in rec.get("items", [])
               if it.get("status") in ("copying", "queued")]
    for it in pending:
        where = it.get("landed") or it["dest"]
        got = survey_dest_files(rec["dest_iid"], where, execute=execute)
        v = verify_salvage(it.get("files") or {}, got)
        it["verify"] = v.status
        it["verify_reason"] = v.reason
        it["bytes_seen"] = v.bytes_seen
        it["missing"] = list(v.missing)[:20]
        it["short"] = list(v.short)[:20]
        if v.status == "ok":
            it["status"] = "verified"
            if callable(push_to_b2) and not dry_run:
                prefix = b2_salvage_prefix(it["job_id"], rec["dead_iid"],
                                           it["name"])
                pok, pdetail = push_to_b2(rec["dest_iid"], where, prefix)
                it["b2"] = prefix if pok else None
                it["b2_detail"] = pdetail

    if all(it.get("status") == "verified" for it in rec.get("items", [])):
        n = len(rec["items"])
        on_b2 = [it for it in rec["items"] if it.get("b2")]
        where = (f"; {len(on_b2)}/{n} pushed to B2"
                 if callable(push_to_b2) else "")
        return _finish(rec, OUTCOME_SALVAGED,
                       f"{n} checkpoint(s), {rec.get('bytes', 0) / 1e9:.2f} GB, "
                       f"name set and every byte count verified on "
                       f"{rec['dest_iid']}{where}")
    if now < rec.get("deadline_ts", 0):
        return rec

    # Deadline. Whatever we have is what we get — and it is NOT success.
    verified = [it for it in rec["items"] if it.get("status") == "verified"]
    seen = sum(int(it.get("bytes_seen") or 0) for it in rec["items"])
    unreadable = [it for it in rec["items"]
                  if it.get("verify") == "unverifiable"]
    outcome = OUTCOME_UNVERIFIABLE if len(unreadable) == len(rec["items"]) \
        else OUTCOME_PARTIAL
    # vast reports a failed host-to-host transfer in the SOURCE instance's
    # `status_msg`, and nowhere else — the API's `success: true` is only "the
    # request was accepted". Pulling it into the record is the difference
    # between an opaque `salvaged_partial` and one that names the rsync error.
    why = _copy_status_soft(copy_status, rec.get("dead_iid"))
    rec["copy_status_msg"] = why
    return _finish(
        rec, outcome,
        f"salvage deadline passed with {len(verified)}/{len(rec['items'])} "
        f"checkpoint(s) verified ({seen}/{rec.get('bytes', 0)} B present). "
        f"DO NOT RESUME FROM THE UNVERIFIED COPIES — a short checkpoint loads "
        f"without complaint and trains from torn weights"
        + (f". vast's transfer status on the source box: {why}" if why else ""))


# --- the transports the injection seam is fed with (herdd's half) --------- #
#
# Everything above this line is pure or transport-injected. Everything below it
# is the real I/O: the closures `advance()` is handed, the B2/instance readers
# that answer its questions, and the two snapshot folds the tick supplies. They
# moved down out of `herdd.py` so that both sides of the injection seam are
# readable in one file — the flat split (policy here, transports 8,000 lines
# away) is what made the seam hard to check.
#
# The three `os.environ` reads below (`B2_BUCKET` x2, `SALVAGE_ENABLED`) are
# kept as direct reads, exactly as they were: `core.config` deliberately ships
# no accessors for the ~70 stray env sites (see its "deliberately NOT here"),
# and inventing one here would add a rung to a precedence that currently has
# none. All three stay listed in `config.ENV_SITES_TODO`; plan §9 owns them.

# moved-from: herdd._mk_salvage_dest_exec
def _mk_salvage_dest_exec(statuses: Mapping[str, str] | None) -> ExecFn:
    """Destination survey transport, chosen by the box's state.

    `execute` (no GPU contract, works on a stopped box) and ssh (works on a
    running one) are strictly complementary — neither covers both. Unknown state
    tries `execute` first and falls back to ssh on vast's own refusal, so a
    stale snapshot degrades instead of failing.
    """
    def _exec(iid: int | str, command: str) -> tuple[bool, str, str | None]:
        if (statuses or {}).get(str(iid)) == "running":
            return remote._ssh_exec_soft(iid, command)
        ok, text, err = remote._vast_execute_soft(iid, command)
        if not ok and "only avail on stopped" in str(err):
            return remote._ssh_exec_soft(iid, command)
        return ok, text, err
    return _exec

# moved-from: herdd._mk_salvage_prepare_dest
def _mk_salvage_prepare_dest(statuses: Mapping[str, str] | None) -> PrepareFn:
    """`mkdir -p` the destination parents -> True when the nested layout is usable.

    `copy_direct` creates the FINAL path component only (OBSERVED: `rsync:
    [Receiver] mkdir "…/out/checkpoint-100" failed: No such file or directory`
    AFTER the API answered `success: true`). A STOPPED destination cannot be
    prepared at all — `execute` offers `ls`/`rm`/`du` and nothing that makes a
    directory — so returning False there selects the flat, mkdir-free layout
    rather than failing the salvage.
    """
    def _prep(iid: int | str, parents: Sequence[str] | None) -> bool:
        if (statuses or {}).get(str(iid)) != "running" or not parents:
            return False
        quoted = " ".join(shlex.quote(p) for p in parents)
        ok, _, err = remote._ssh_exec_soft(
            iid, f"mkdir -p {quoted} && echo MKDIR_OK")
        if not ok:
            print(f"!! salvage: could not prepare {iid}'s destination dirs "
                  f"({err}) — falling back to the flat layout")
        return bool(ok)
    return _prep

# moved-from: herdd._salvage_copy_status
def _salvage_copy_status(iid: int | str) -> str | None:
    """The instance's `status_msg`. vast reports a failed host-to-host transfer
    THERE and nowhere else, so this is the only way a partial salvage can say
    WHY. Best-effort: None on any read failure."""
    ok, d, _ = api.request_soft("GET", f"v0/instances/{iid}/", retries=1)
    if not ok or not isinstance(d, dict):
        return None
    inst = d.get("instances")
    return (inst or {}).get("status_msg") if isinstance(inst, dict) else None

# moved-from: herdd._salvage_b2_bytes
def _salvage_b2_bytes(job_id: str | None) -> dict[str, int] | None:
    """`{checkpoint-<N>: bytes}` already on B2 under `jobs/<JOB_ID>/checkpoints/`.

    `None` on any failure (no bucket, no rclone, listing error) — and
    `salvage.plan_salvage` reads `None` as "unknown", which makes it copy MORE,
    never less. An unreadable B2 must never be mistaken for an empty one.
    """
    bucket = os.environ.get("B2_BUCKET")
    if not bucket or not job_id:
        return None
    rc, out, _ = b2._rclone_soft(["lsf", "-R", "--files-only", "--format", "sp",
                                  "--separator", "|",
                                  f"b2:{bucket}/jobs/{job_id}/checkpoints/"])
    if rc != 0:
        return None
    # Keys must match `CkptDir.name` from the box-side survey, which is the path
    # relative to the job's `work/` dir (`out/checkpoint-50`). That is the SAME
    # relative path B2 holds, because jobd pushes `$wdir/work` -> the job's
    # `checkpoints/` prefix wholesale. Splitting on the first segment instead
    # would key everything under `out` and make every checkpoint look
    # already-synced — a silent `nothing_newer` on a disk that still had the
    # only copy.
    sizes: dict[str, int] = {}
    for line in (out or "").splitlines():
        if "|" not in line:
            continue
        size, path = line.split("|", 1)
        head, step, _ = split_ckpt_rel(path.strip())
        if step is None:
            continue
        try:
            # `head`/`step` are None together; see ckpt_dirs_from_survey.
            sizes[head] = sizes.get(head, 0) + int(size)  # type: ignore[index,arg-type]
        except ValueError:
            continue
    return sizes

# moved-from: herdd._salvage_push_script
def _salvage_push_script(local_path: str, bucket: str, prefix: str) -> str:
    """Base64-shipped remote push script: salvaged bytes -> our own bucket with
    our own keys, run ON the destination box.

    Ships as a FILE (`bash /tmp/...`) rather than an inline command for the same
    reason `_job_cancel_kill_script` does: the remote argv must not contain
    strings that a later `pgrep`/`pkill` could self-match. Destination is
    `jobs/<JOB_ID>/salvage/<IID>/<ckpt>/` — never `checkpoints/`, which the
    replacement job is a LIVE WRITER of.
    """
    return (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f"SRC={shlex.quote(local_path)}\n"
        f"DST={shlex.quote(f'b2:{bucket}/{prefix}')}\n"
        '[ -d "$SRC" ] || { echo "salvage-push: $SRC absent"; exit 2; }\n'
        'if command -v b2x >/dev/null 2>&1; then\n'
        '  b2x push "$SRC" "$DST/" && { echo "salvage-push: b2x ok"; exit 0; }\n'
        'fi\n'
        'rclone copy --fast-list "$SRC" "$DST/" && '
        '{ echo "salvage-push: rclone ok"; exit 0; }\n'
        'echo "salvage-push: FAILED"; exit 1\n')

# moved-from: herdd._salvage_push_to_b2
def _salvage_push_to_b2(dest_iid: int | str, local_path: str,
                        prefix: str) -> tuple[bool, str]:
    """Push a VERIFIED salvaged checkpoint from the destination box to B2.

    Returns `(ok, detail)`. Best-effort by design: the bytes are already safe on
    a live box we own at this point, so a failed push downgrades the report, not
    the outcome. Called only after `verify_salvage` returned `ok` — pushing an
    unverified copy is how a torn checkpoint gets laundered into durable storage.
    """
    bucket = os.environ.get("B2_BUCKET")
    if not bucket:
        return False, "B2_BUCKET not set — salvaged bytes stay on the box only"
    try:
        i = lifecycle._get_instance(dest_iid)
    except SystemExit:
        return False, f"destination {dest_iid} is not listed any more"
    host, port, _ = ssh._pick_ssh_endpoint(i)
    if not (host and port):
        return False, (f"no ssh endpoint on {dest_iid} "
                       f"(status={i.get('actual_status')})")
    b64 = base64.b64encode(
        _salvage_push_script(local_path, bucket, prefix).encode()).decode("ascii")
    remote = (f"echo {b64} | base64 -d > /tmp/salvage_push.sh && "
              f"bash /tmp/salvage_push.sh; rc=$?; rm -f /tmp/salvage_push.sh; "
              f"exit $rc")
    r = subprocess.run(["ssh", "-p", str(port), f"root@{host}",
                        "-o", "StrictHostKeyChecking=accept-new",
                        "-o", "LogLevel=ERROR", remote],
                       capture_output=True, text=True)
    detail = ((r.stdout or "") + (r.stderr or "")).strip()[:400]
    return r.returncode == 0, detail or f"ssh exited {r.returncode}"

# moved-from: herdd._salvage_enabled
def _salvage_enabled(jc: Mapping[str, Any]) -> bool:
    """Salvage is ON by default and disarmed by `SALVAGE_ENABLED=0` (or
    `--no-salvage`). It costs bandwidth on ~1 GB and nothing else — no GPU
    contract is entered on either box — so the default is the safe one."""
    v = getattr(jc.get("a"), "salvage", None)
    if v is False:
        return False
    env = os.environ.get("SALVAGE_ENABLED")
    return env not in ("0", "false", "no")

# moved-from: herdd._salvage_statuses
def _salvage_statuses(jc: Mapping[str, Any]) -> dict[str, str]:
    """`{iid: actual_status}` from the tick's instance snapshot — what
    `salvage.pick_dest` needs to choose a landing zone."""
    return {str(i.get("id")): (i.get("actual_status") or "").lower()
            for i in (jc.get("instances") or [])}

# moved-from: herdd._salvage_free_gb
def _salvage_free_gb(jc: Mapping[str, Any]) -> dict[str, float]:
    """`{iid: free_gb}` from the tick's instance snapshot.

    A jobs box runs near full BY DESIGN — that is what jobd's prune is for — so
    "it is `running`" is not evidence it has room for a ~1 GB checkpoint, and a
    salvage that fills the landing box's disk kills whatever is running on it.
    A box whose telemetry is missing is simply ABSENT from this map, which
    `pick_dest` reads as "unknown, allow": refusing every candidate on missing
    telemetry would disarm salvage entirely.
    """
    out: dict[str, float] = {}
    for i in (jc.get("instances") or []):
        try:
            total = float(i.get("disk_space") or i.get("disk_gb") or 0)
            used = float(i.get("disk_util") or i.get("disk_used_gb") or 0)
        except (TypeError, ValueError):
            continue
        if total > 0:
            out[str(i.get("id"))] = max(0.0, total - used)
    return out

# moved-from: herdd._salvage_dest_candidates
def _salvage_dest_candidates(jc: Mapping[str, Any], new_iid: int | str | None,
                             old: int | str | None) -> list[str]:
    """Ordered landing zones for a salvage copy: the REPLACEMENT first (it is the
    box that will want the state), then any other live box this watch owns.

    Taking a box that is already `running` beats waiting on one that is still
    pulling an image, because the clock we are racing is the host reclaiming the
    dead disk — not the replacement's boot."""
    cands = [str(new_iid)] if new_iid is not None else []
    for i in (jc.get("instances") or []):
        iid = str(i.get("id"))
        if iid in (str(old), *cands):
            continue
        if (i.get("actual_status") or "").lower() == "running":
            cands.append(iid)
    return cands
