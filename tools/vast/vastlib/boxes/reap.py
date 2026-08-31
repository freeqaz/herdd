"""vastlib.boxes.reap — the automatic destroyer, and the two clocks it destroys by.

Why this module is separate from `boxes.lifecycle`
--------------------------------------------------
`cmd_reap` is the only function in this package that a machine runs unattended
against a live fleet: `herdd-reaper.timer` fires `herdd.py reap -y` every 15
minutes from the repo working tree and DESTROYS boxes. Everything that decides
*whether* a box dies therefore lives here, apart from the primitives that
perform the killing (`boxes.lifecycle`), so the policy can be read, tested and
reviewed as one page.

Two lanes, two clocks:

* **Stopped lane** — a parked box is destroyed once it has been idle past
  `--idle-hours` (default `REAP_IDLE_H_DEFAULT` = 2 h, owner policy 2026-07-21),
  unless its label carries a keep token. Vast records no stop timestamp, so the
  age comes from `_idle_secs_map`'s first-observed-stopped ledger.
* **Live lane** — a box in a guard zombie verdict is fed to
  `parked_lifecycle.zombie_action` (destroy / park / alarm), but only after
  `_zombie_confirm_map` proves NO PROGRESS for `REAP_ZOMBIE_CONFIRM_S`
  (default 900 s) across five independent signals.

The two ledger files are a byte-frozen contract
-----------------------------------------------
Both clocks live entirely in JSON under `XDG_CACHE_HOME`:

    <XDG_CACHE_HOME|~/.cache>/herdd/idle-ledger.json
        {"<iid>": <float first-seen-stopped epoch>}
    <XDG_CACHE_HOME|~/.cache>/herdd/zombie-ledger.json
        {"<iid>": {"first": float, "verdict": str,
                   "pull": {...}, "hb": float|None,
                   "inet": float|None, "disk": float|None}}

Changing either PATH or either key shape does not fail — it silently resets
every live box's clock on the workstation that runs the timer: every parked box
restarts its 2 h fuse at 0, every zombie restarts its 900 s confirmation. Both
paths are read from `os.environ` at IMPORT time, so a test that wants a
temporary ledger patches the MODULE ATTRIBUTE (`reap._IDLE_LEDGER`), not the
environment variable. `test_vastlib_boxes_reap.py` pins both the path formula
and both key shapes.

The five reap knobs and the one that is different
-------------------------------------------------
Read at RUN time from `os.environ`, string-lowered, `"0"`/`"no"`/`"off"` = off:

  `HERDD_REAP`             global kill switch — returns immediately
  `HERDD_REAP_IDLE_H`      float; falls back to `REAP_IDLE_H_DEFAULT` on
                             ValueError
  `HERDD_REAP_ZOMBIE`      disables the live lane
  `HERDD_REAP_STALL`       legacy spelling; EITHER being off disables it
  `HERDD_REAP_DURABILITY`  skips the B2 reads in the advisory

None of those five goes through `core.config._boot_knob`. The sixth,
`REAP_ZOMBIE_CONFIRM_S` (in `_zombie_confirm_map`), DOES — it gets the full
CLI > env > yaml > constant precedence. That asymmetry is inherited, not
designed, and it is deliberate here: routing the other five through `_boot_knob`
would add a yaml rung to a kill switch that today is env-only, which is a
behavior change (plan v1 §S5 puts the knob-resolver unification out of scope).
The asymmetry has an operational edge: the systemd unit carries NO
`EnvironmentFile`, so all five reach `cmd_reap` only via `config.load_env()`'s
walk-up-from-CWD `.env` discovery — which is why the unit's
`WorkingDirectory=$REPO_ROOT` is load-bearing, and why moving when `load_env`
runs relative to `cmd_reap` would silently disarm the kill switch.

What is deliberately NOT here
-----------------------------
* **No keep-token grammar.** `labels._reap_kept` / `labels._keep_retention_info`
  own it, and this module never re-derives it. A second copy is what produced
  the 2026-08-02 un-revoked-key bug; `parked_lifecycle` still carries its own
  fallback predicate, which is the same defect one level down.
* **No mutation primitives.** Destroy is `lifecycle._destroy_and_revoke`, park
  is `lifecycle.stop_box` — reached by module attribute so the suite's patches
  keep steering them.
* **No health classification.** `boxes.health` owns `gather_fleet_health`, the
  guard verdicts and `_jobd_ever_stamped`. `cmd_reap` calls the gather inside a
  bare `try/except` that degrades to `health={}` — so a bug in that module
  quietly reduces reap to the idle lane instead of erroring. Same shape for the
  jobs fold and the `parked_lifecycle` import.
* **No `_dash_reap_threshold_s`.** `storage.dashcache` re-derives this policy
  for the dashboard display; it is a SECOND reader of `REAP_IDLE_H_DEFAULT` and
  has to be kept in step by hand.

Exit codes are a scripting contract: `sys.exit(2)` on a preview that found
candidates (and on `--json` with candidates), `sys.exit(<msg>)` on a sweep that
left a box billing. `cmd_reap`'s argparse surface is exactly
`(a.idle_hours, a.yes, a.json)` — `test_guard.py` constructs that Namespace by
hand at 17 call sites, so a new required attribute breaks all of them.

Provenance: moved from `tools/vast/herdd.py` (plan §8 step 3, 2026-08-16) —
the reap block (`REAP_IDLE_H_DEFAULT`:5231, `_ZOMBIE_*`:5346-5361,
`_zombie_confirm_map`:5364, `_reap_durability_advisory`:5467, `cmd_reap`:5519)
plus the idle ledger (`_IDLE_LEDGER`:2310, `_idle_secs_map`:2315), which sits
textually in the ls-snapshot block and which `cli/ls` must keep reading from
HERE — two ledgers would fork the 2 h clock. Behavior-preserving: bodies
verbatim, annotations and cross-module attribute paths added. Step 3 is
ADD-ONLY; `herdd.py` keeps its copies (and the live timer keeps running them)
until step 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Mapping, Sequence

from vastlib.boxes import health as boxhealth
from vastlib.boxes import lifecycle
from vastlib.core import config, fmt, labels, models

import bidpolicy

# `boxes.health` is aliased because `_zombie_confirm_map`'s first PARAMETER is
# named `health`, as is a local in `cmd_reap` — both verbatim from `herdd.py`,
# where no such module existed. The alias is the same module object, so
# `monkeypatch.setattr(vastlib.boxes.health, "gather_fleet_health", ...)` still
# steers the calls below.


# --------------------------------------------------------------------------- #
# CROSS-RING SEAM — new code, no `moved-from:` marker (README §2 rule 7).
# `_fold_fleet_jobs` is the fleet-wide jobd fold. It lives in `vastlib.jobs.view`
# (plan §5, step 5) and `boxes` sits BELOW `jobs` in the §5 DAG, so this ring may
# never import it — import-linter rejects the edge whether the import is written
# at module level or inside a function (grimp reads the whole AST). A COPY here
# would be the exact fork this refactor exists to kill.
#
# So the seam is an INJECTION SLOT, exactly as this comment said it would be at
# step 3: `vastlib.jobs.view` binds `_FOLD_FLEET_JOBS` at ITS import (bottom of
# that file), and the composition roots — `cli.main`, `fleet.daemon` — import it.
# Landed 2026-08-16 with the step-5 jobs port.
#
# Unbound is still a NotImplementedError rather than a silent `{}`: both call
# sites below already wrap this in `try/except`, so a `{}` would degrade reap to
# the idle lane INVISIBLY. The raise degrades identically but says why.
# --------------------------------------------------------------------------- #

#: Bound by `vastlib.jobs.view` at import. Not a `Callable` default and not a
#: fallback — a wrong binding must be loud, not empty.
_FOLD_FLEET_JOBS: Any = None   # noqa: ANN401 — Callable[[Any, Any], dict[str, list[Any]]]


def _fold_fleet_jobs(live_iids: Any,  # noqa: ANN401 — set[int]|set[str], caller-shaped
                     prog: Any = None) -> Any:  # noqa: ANN401 — progress cb / fold map
    """The fleet-wide jobd fold, through the injection slot above.

    `import vastlib.jobs.view` binds it. Until something does, this raises —
    which both call sites swallow into "no jobs fold", the same outcome a B2
    failure produces.
    """
    impl = _FOLD_FLEET_JOBS
    if impl is None:
        raise NotImplementedError(
            "_fold_fleet_jobs: no implementation bound — `import vastlib.jobs.view` "
            "(it binds boxes.reap._FOLD_FLEET_JOBS at import), rebind this module "
            "attribute, or stub it in your test with monkeypatch.setattr")
    return impl(live_iids, prog)


# --------------------------------------------------------------------------- #
# The idle clock. `cli/ls` MUST read this same ledger — two copies fork the 2 h
# fuse, and the fork is invisible until a box that `ls` calls old is called
# young by `reap` (or the other way round).
# --------------------------------------------------------------------------- #

# moved-from: herdd._IDLE_LEDGER
_IDLE_LEDGER = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "herdd", "idle-ledger.json")


# moved-from: herdd._idle_secs_map
def _idle_secs_map(instances: Any, live_ids: Any) -> dict[str, float]:  # noqa: ANN401 — raw vast rows
    """box-id(str) -> seconds a STOPPED box has been idle. Vast's instance
    object carries no stop timestamp (only start_date = creation), so we keep
    our own ledger of when each box was FIRST observed stopped, refreshed on
    every ls: a box seen live (or gone) clears its entry, a newly-stopped box
    stamps `now`. Idle is thus measured since herdd first saw it stopped —
    on the very first sighting a long-stopped box reads ~0 and self-corrects.
    Best-effort: {} if the ledger can't be read/written."""
    now = time.time()
    try:
        with open(_IDLE_LEDGER) as fh:
            led = json.load(fh)
        if not isinstance(led, dict):
            led = {}
    except Exception:
        led = {}
    live = {str(x) for x in live_ids}
    present = {str(i.get("id")) for i in instances}
    stopped = present - live
    changed = False
    for k in list(led):                      # forget live/destroyed boxes
        if k not in present or k in live:
            del led[k]; changed = True       # noqa: E702 — verbatim body (plan §7.4)
    for k in stopped:                        # stamp newly-stopped boxes
        if k not in led:
            led[k] = now; changed = True     # noqa: E702 — verbatim body (plan §7.4)
    if changed:
        try:
            os.makedirs(os.path.dirname(_IDLE_LEDGER), exist_ok=True)
            tmp = _IDLE_LEDGER + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(led, fh)
            os.replace(tmp, _IDLE_LEDGER)
        except Exception:
            pass
    return {k: max(0.0, now - led[k]) for k in stopped if k in led}


# moved-from: herdd.REAP_IDLE_H_DEFAULT
REAP_IDLE_H_DEFAULT = 2.0   # owner policy 2026-07-21: idle > 2h => destroy


# --------------------------------------------------------------------------- #
# The zombie confirmation clock
# --------------------------------------------------------------------------- #

# moved-from: herdd._ZOMBIE_LEDGER
_ZOMBIE_LEDGER = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "herdd", "zombie-ledger.json")

# Progress thresholds for the confirm clock's ENV-SETUP signals (2026-08-02).
# During env-setup the pull-byte fold is flat BY DEFINITION (pull finished) and
# the jobd heartbeat is absent BY DEFINITION (jobd not up yet), so without a
# third signal a perfectly healthy install is indistinguishable from a stall —
# the shape that would have auto-destroyed a jobs box mid-provision. What IS
# observable from the API alone: `inet_down_billed` (cumulative box download,
# KB — measured 2026-08-02: 16358262.6 ≈ the 16.3 GB the invoice billed) moves
# while onstart pulls weights/env/bootstrap tars, and `disk_usage` (GB) moves
# while it unpacks. Thresholds are deliberately above idle noise (jobd
# heartbeats are tiny UPLOADS) but far below any real provisioning step.
# moved-from: herdd._ZOMBIE_INET_EPS_KB
_ZOMBIE_INET_EPS_KB = 50_000     # ≥ ~50 MB new download traffic = alive
# moved-from: herdd._ZOMBIE_DISK_EPS_GB
_ZOMBIE_DISK_EPS_GB = 0.5        # ≥ 0.5 GB new disk usage = alive


# moved-from: herdd._zombie_confirm_map
def _zombie_confirm_map(health: Any,  # noqa: ANN401 — {iid: BoxHealth-shaped dict}
                        by_iid: Any,  # noqa: ANN401 — {iid: raw vast instance}
                        now: float | None = None) -> dict[str, dict[str, Any]]:
    """{iid(str): {"confirmed": bool, "since_s": int, "note": str}} for every
    box in a zombie verdict, persisting sightings across reap invocations in
    _ZOMBIE_LEDGER (same first-observed pattern as the idle ledger).

    Why: an AUTOMATIC destroyer must be more conservative than the manual one.
    `guard --fix` fires a human at the 1500 s age deadline; a 20+ min image
    pull on a bad-network host is a legitimate boot that crosses that deadline
    mid-pull, and destroying it wastes the pull. So the sweep condemns on
    NO PROGRESS, not on age alone: a box is `confirmed` only once the SAME
    verdict has persisted >= REAP_ZOMBIE_CONFIRM_S (default 900 s ≈ one extra
    timer period) since its first ledger sighting, and ANY progress signal
    resets that clock —

      * the verdict changed (different shape: restart the evidence),
      * docker-pull bytes advanced (`status_msg` folded through
        parse_pull_progress; the per-layer high-water map is persisted so the
        comparison survives vast's tail-window truncation),
      * the jobd heartbeat epoch advanced (jobd wrote SOMETHING — not dead),
      * the box's cumulative download counter (`inet_down_billed`, KB) advanced
        past _ZOMBIE_INET_EPS_KB, or its `disk_usage` (GB) advanced past
        _ZOMBIE_DISK_EPS_GB — the ENV-SETUP liveness signals (2026-08-02): once
        the pull is done and before jobd stamps, those two are the only
        API-visible evidence that onstart is really provisioning, and without
        them a healthy long install (weight pull, env build) is
        indistinguishable from a stall and would be auto-condemned.
      * `cpu_util` is above `CPU_BUSY_UTIL` (2026-08-21). A dedicated CPU box
        — compile/search work, no model endpoint — is FLAT ON ALL FIVE signals
        above while burning cores: nothing pulls, jobd never stamps, the
        download counter sits still and disk does not grow. It was therefore
        confirmable as a zombie while doing exactly the work it was rented for.
        Unlike the others this is a LEVEL, not a counter: the question is "busy
        now", so nothing is persisted for it and the ledger shape is unchanged.

    Effective floor before any automatic action: deadline + up to one
    timer period of sighting latency + 900 s confirmation ≈ 40-55 min of
    provable no-progress. Ledger entries for boxes no longer in a zombie
    verdict (recovered, destroyed, gone) are dropped. Best-effort throughout:
    an unreadable/unwritable ledger yields confirmed=False for everything —
    degrading to alarm-only, never to a faster destroy."""
    now = time.time() if now is None else now
    confirm_s = config._boot_knob("REAP_ZOMBIE_CONFIRM_S", cast=float)
    try:
        with open(_ZOMBIE_LEDGER) as fh:
            led = json.load(fh)
        if not isinstance(led, dict):
            led = {}
    except Exception:
        led = {}
    out, nxt = {}, {}
    for iid_s, h in (health or {}).items():
        v = h.get("verdict")
        # `boxes.health` unified the two verdict frozensets into `GuardVerdict`
        # and exposes the string-side membership question as a predicate; this
        # is the same test `v not in _GUARD_ZOMBIE_VERDICTS` made, including the
        # "unknown verdict is not a zombie" arm.
        if not boxhealth.verdict_is_zombie(v):
            continue
        inst = (by_iid or {}).get(str(iid_s)) or {}
        prev: Any = led.get(str(iid_s)) if isinstance(led.get(str(iid_s)), dict) \
            else None
        try:
            pull = boxhealth.parse_pull_progress(inst.get("status_msg") or "",
                                                (prev or {}).get("pull"))
        except Exception:
            pull = (prev or {}).get("pull") or {}
        hb_age = (h.get("evidence") or {}).get("jobd_hb_age_s")
        hb = (now - hb_age) if hb_age is not None else None
        inet = models._num_dph(inst.get("inet_down_billed"))     # KB, cumulative
        disk = models._num_dph(inst.get("disk_usage"))           # GB
        # A LEVEL, unlike the four counters above — so the test is "busy now",
        # not "advanced since last sighting", and nothing about it is persisted.
        cpu = (h.get("evidence") or {}).get("cpu_util")
        note = None
        if prev is None:
            note = "first sighting"
        elif prev.get("verdict") != v:
            note = f"verdict changed ({prev.get('verdict')} -> {v})"
        elif int(pull.get("total_bytes") or 0) > \
                int((prev.get("pull") or {}).get("total_bytes") or 0):
            note = "pull bytes advancing"
        elif hb is not None and prev.get("hb") is not None \
                and hb > float(prev["hb"]) + 1.0:
            note = "jobd heartbeat advancing"
        elif inet is not None and prev.get("inet") is not None \
                and inet > float(prev["inet"]) + _ZOMBIE_INET_EPS_KB:
            note = "box download traffic advancing (env-setup/pull alive)"
        elif disk is not None and prev.get("disk") is not None \
                and disk > float(prev["disk"]) + _ZOMBIE_DISK_EPS_GB:
            note = "disk usage advancing (unpack/install alive)"
        elif isinstance(cpu, (int, float)) and cpu > boxhealth.CPU_BUSY_UTIL:
            note = f"cpu {cpu:.2f} busy (compute alive)"
        first = now if note else float(prev.get("first") or now)
        nxt[str(iid_s)] = {"first": first, "verdict": v, "pull": pull,
                           "hb": hb if hb is not None else
                           (prev or {}).get("hb"),
                           # carry the last READ value forward on a failed read
                           # so one missing sample can't fake progress or reset
                           "inet": inet if inet is not None else
                           (prev or {}).get("inet"),
                           "disk": disk if disk is not None else
                           (prev or {}).get("disk")}
        since = max(0.0, now - first)
        out[str(iid_s)] = {
            "confirmed": since >= confirm_s,
            "since_s": int(round(since)),
            "note": note or f"no progress for {fmt._age_str(since)} "
                            f"(confirm at {int(confirm_s)}s)"}
    try:
        os.makedirs(os.path.dirname(_ZOMBIE_LEDGER), exist_ok=True)
        tmp = _ZOMBIE_LEDGER + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(nxt, fh)
        os.replace(tmp, _ZOMBIE_LEDGER)
    except Exception:
        pass
    return out


# moved-from: herdd._reap_durability_advisory
def _reap_durability_advisory(inst: models.Payload, pal: fmt._Pal,
                              jobs_by_box: Any = None) -> None:  # noqa: ANN401 — folded jobs map
    """ADVISORY (P2, docs/plans/parked-box-lifecycle.md §11a-R2): compute and
    print the durability verdict for a box the time policy is about to destroy.
    Changes NOTHING about the destroy decision — the caps are `<= 0` until the
    owner arms them — but the verdict lands on stdout, which the reaper's
    systemd unit persists to the user journal, satisfying "read the journal on
    the next natural fleet churn". Soft: a broken advisory must never block the
    reap. HERDD_REAP_DURABILITY=0 skips the B2 reads.

    `jobs_by_box` is the ALREADY-FOLDED fleet job map. Pass it: the fold is
    fleet-wide (`jobmeta.list_all_queued` + a per-job view), so folding it
    per candidate made an N-box reap do N full fleet folds against B2. None
    means "fold one now", kept only so a caller with no map still works."""
    if os.environ.get("HERDD_REAP_DURABILITY", "1").strip().lower() \
            in ("0", "no", "off"):
        return
    # Function-local by design (verbatim): `parked_lifecycle` is an absorbed
    # sibling that imports back into this tree, and the local import is what
    # keeps that from being a cycle.
    import parked_lifecycle as _pl
    iid = inst.get("id")
    lab = inst.get("label") or ""
    try:
        rid = models._label_value(lab, "run")
        if rid:
            # a handoff twin (`run:<ID>:handoff`) shares the PRIMARY's event
            # stream — evidence lives at runs/<ID>/; whether the verdict
            # applies to THIS box is the emitter rule's job, not the lookup's
            rid = rid.split(":")[0]
            evd = _pl.gather_run_evidence(rid)
            emitter = evd.pop("terminal_emitter_iid")
            verdict, reasons = _pl.classify_box_run_durability(
                box_iid=iid, terminal_emitter_iid=emitter, **evd)
        elif lifecycle._box_is_jobd(iid):
            fold = jobs_by_box if jobs_by_box is not None else _fold_fleet_jobs(set())
            jobs = fold.get(str(iid)) or []
            tickets = []
            for v in jobs:
                if not v.get("job_id"):
                    continue
                t = _pl.job_ticket(v.get("job_id"))
                t["id"] = v.get("job_id")
                tickets.append(t)
            verdict, reasons = _pl.classify_jobs_durability(tickets=tickets)
        else:
            verdict, reasons = _pl.UNKNOWN, ["no run:<RID> label and not a "
                                             "jobs box — no recorded channel "
                                             "to check"]
    except Exception as e:
        verdict, reasons = "UNKNOWN", [f"advisory failed: {type(e).__name__}: {e}"]
    line = (f"       durability={verdict} (advisory, does not gate this "
            f"destroy) — {reasons[0] if reasons else ''}")
    print(pal.red(line) if verdict == "UNSYNCED" else line)


# moved-from: herdd.cmd_reap
def cmd_reap(a: argparse.Namespace) -> None:
    """Idle-box reaper (owner policy 2026-07-21 — supersedes park-by-default):
    DESTROY every STOPPED box that has sat idle past --idle-hours (default 2h,
    env HERDD_REAP_IDLE_H), unless its label carries a `keep` token — opt a
    box out with `herdd label <ID> keep:<why>` (or append `:keep` to the
    existing label). Bare `reap` previews (exit 2 when it would destroy);
    `-y` executes. Idle age comes from the same first-observed-stopped ledger
    `ls` shows, so a long-parked box first seen by THIS machine starts its 2h
    clock at first sighting — the reaper never destroys on a blind guess.
    Each candidate also gets a DURABILITY advisory line (is its work provably
    on B2?) — advisory only while the lifecycle caps are unset
    (parked-box-lifecycle §11a-R2); HERDD_REAP_DURABILITY=0 skips it.
    Live boxes get the ZOMBIE lane (2026-08-02, after 46633685 — an on-demand
    serve box dead in `loading` 31 min — was seen by this reaper and declined
    while billing; generalizes the 2026-07-30 stall sweep): every box in a
    guard zombie verdict is fed to the graded `parked_lifecycle.zombie_action`
    policy — DESTROY only with the provably-workless proof (jobs box, jobd
    never stamped JOBD_STATUS: readable B2 absence), PARK when death is
    measured but worklessness is not provable (non-jobs loading stall; running
    jobs box whose heartbeat was read and is stale — parking ends the GPU-rate
    bleed, keeps the disk, and hands the box to THIS reaper's 2 h idle lane),
    ALARM for everything weaker (keep labels, unreadable evidence per I3, and
    ZOMBIE_TICKET_UNCLAIMED where jobd is alive). An automatic action further
    requires CONFIRMATION — the same verdict persisted >= REAP_ZOMBIE_CONFIRM_S
    (default 900 s) across passes with no progress on ANY signal: docker-pull
    bytes, jobd heartbeat, box download traffic (`inet_down_billed`), or disk
    usage (`_zombie_confirm_map`). The last two are the env-setup liveness
    signals (2026-08-02 phase split): a running jobs box mid-provision has flat
    pull bytes and no heartbeat by definition, and without them a healthy long
    install would confirm as a stall and be destroyed. So the automatic trigger
    is strictly later than manual `guard --fix`, a slow-but-moving image pull
    is never condemned (and is GPU-unbilled anyway — invoice-verified), and a
    demonstrably-downloading env-setup is never condemned either. Same preview/-y/keep
    semantics; HERDD_REAP_ZOMBIE=0 (or the legacy HERDD_REAP_STALL=0)
    disables the live lane; manual `guard --fix -y` is unchanged.
    Meant for a scheduler — tools/vast/reaper_install.sh installs a systemd
    user timer running `reap -y` every 15 min; HERDD_REAP=0 disables every
    invocation globally (campaign kill switch)."""
    if os.environ.get("HERDD_REAP", "1").strip().lower() in ("0", "no", "off"):
        print("reap disabled (HERDD_REAP=0)."); return   # noqa: E702 — verbatim body (plan §7.4)
    hours = a.idle_hours
    if hours is None:
        try:
            hours = float(os.environ.get("HERDD_REAP_IDLE_H", "")
                          or REAP_IDLE_H_DEFAULT)
        except ValueError:
            hours = REAP_IDLE_H_DEFAULT
    thresh = max(0.0, hours * 3600.0)
    pal = fmt._Pal(fmt._color_on())
    ins = lifecycle._instances()
    live = [i.get("id") for i in ins
            if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES]
    idle = _idle_secs_map(ins, live)
    by_iid = {str(i.get("id")): i for i in ins}

    # ONE fleet-wide job fold for the whole command. It is the expensive read
    # here (`jobmeta.list_all_queued` + a view per non-terminal job), and both
    # consumers below — the stall lane's health gather and the per-candidate
    # durability advisory — used to fold it independently, the advisory once
    # PER BOX. Memoized, so a reap with no jobs box and no stall never pays.
    _fold_memo: dict[str, Any] = {}

    def _jobs_fold() -> Any:  # noqa: ANN401 — {iid: [folded job views]}
        if "v" not in _fold_memo:
            try:
                _fold_memo["v"] = _fold_fleet_jobs(set(live))
            except Exception:
                _fold_memo["v"] = {}
        return _fold_memo["v"]

    rows, reap_ids = [], []
    for i in ins:
        iid = i.get("id")
        sec = idle.get(str(iid))
        if sec is None:                       # live box: the stall lane below
            continue
        lab = i.get("label") or ""
        if labels._reap_kept(lab):
            verdict = "KEEP"
        elif sec >= thresh:
            verdict = "REAP"; reap_ids.append(iid)   # noqa: E702 — verbatim body (plan §7.4)
        else:
            verdict = f"WAIT ({fmt._age_str(thresh - sec)} left)"
        rows.append({"iid": iid, "label": lab, "idle_s": sec,
                     "storage_day": models._storage_day(i), "verdict": verdict})

    # Live lane: graded automatic action on every zombie-verdict box (see the
    # docstring). Everything live and healthy stays guard/fleetd territory.
    zombie_rows, zdestroy_ids, zpark_ids = [], [], []
    _off = ("0", "no", "off")
    if (os.environ.get("HERDD_REAP_ZOMBIE", "1").strip().lower() not in _off
            and os.environ.get("HERDD_REAP_STALL", "1").strip().lower()
            not in _off):
        import parked_lifecycle as _pl
        try:
            health = boxhealth.gather_fleet_health(ins, _jobs_fold())
        except Exception:
            health = {}
        confirm = _zombie_confirm_map(health, by_iid)
        for iid_s, h in sorted((health or {}).items()):
            if not boxhealth.verdict_is_zombie(h.get("verdict")):  # see _zombie_confirm_map
                continue
            inst = by_iid.get(str(iid_s)) or {}
            lab = inst.get("label") or ""
            evd = h.get("evidence") or {}
            kept = labels._reap_kept(lab)
            # The B2 never-ran listing is read only where it can license a
            # destroy: an unkept jobs-lane box. Everyone else never pays it.
            stamped = (boxhealth._jobd_ever_stamped(iid_s)
                       if (evd.get("is_jobs_box") and not kept) else None)
            c = confirm.get(str(iid_s)) or {}
            action, why = _pl.zombie_action(
                verdict=h.get("verdict"),
                is_jobs_box=evd.get("is_jobs_box"),
                jobd_ever_stamped=stamped,
                jobd_hb_read=evd.get("jobd_hb_age_s") is not None,
                label_kept=kept,
                confirmed=bool(c.get("confirmed")))
            zombie_rows.append({"iid": inst.get("id") or iid_s, "label": lab,
                                "verdict": h.get("verdict"),
                                "reason": h.get("reason"),
                                "action": action, "why": why,
                                "confirm": c.get("note"),
                                "zombie_for_s": c.get("since_s")})
            if action == _pl.ZOMBIE_DESTROY:
                zdestroy_ids.append(inst.get("id") or iid_s)
            elif action == _pl.ZOMBIE_PARK:
                zpark_ids.append(inst.get("id") or iid_s)

    if getattr(a, "json", False):
        print(json.dumps({"idle_hours": hours, "rows": rows,
                          "reap": reap_ids, "zombie_rows": zombie_rows,
                          "zombie_destroy": zdestroy_ids,
                          "zombie_park": zpark_ids}, indent=2))
        sys.exit(2 if (reap_ids or zdestroy_ids or zpark_ids) else 0)
    print(f"== herdd reap — stopped + idle>{hours:g}h => destroy · "
          f"{len(rows)} stopped box(es), {len(reap_ids)} to reap ==")
    for r in sorted(rows, key=lambda r: -(r["idle_s"] or 0)):
        stor = (f"${r['storage_day']:.2f}/day"
                if r["storage_day"] is not None else "$?/day")
        line = (f"  {'!!' if r['verdict'] == 'REAP' else '  '} {r['iid']}  "
                f"idle {fmt._age_str(r['idle_s'])} · {stor} · "
                f"{r['label'] or '(no label)'}  -> {r['verdict']}")
        print(pal.red(line) if r["verdict"] == "REAP" else line)
        if r["verdict"] == "REAP":
            _reap_durability_advisory(by_iid.get(str(r["iid"])) or {}, pal,
                                      jobs_by_box=_jobs_fold())
    if not rows:
        print("no stopped boxes — nothing to reap.")
    if zombie_rows:
        print(f"\n-- zombie boxes ({len(zombie_rows)}; "
              f"{len(zdestroy_ids)} => destroy, {len(zpark_ids)} => park) --")
        for r in zombie_rows:
            act = r["action"]
            mark = "!!" if act != "alarm" else "  "
            line = (f"  {mark} {r['iid']}  {r['label'] or '(no label)'}  "
                    f"{r['reason']}  -> {act.upper() if act != 'alarm' else 'alarm'}"
                    f" ({r['why']}"
                    + (f"; {r['confirm']}" if r.get("confirm") else "") + ")")
            print(pal.red(line) if act != "alarm" else line)
    doomed = (list(reap_ids)
              + [i for i in zdestroy_ids if i not in reap_ids]
              + [i for i in zpark_ids if i not in reap_ids])
    if not doomed:
        return
    if not a.yes:
        nact = len(reap_ids) + len(zdestroy_ids)
        print(f"\n[preview] reap WOULD DESTROY {nact} + PARK {len(zpark_ids)} "
              f"box(es): {doomed}")
        print("  re-run `herdd reap -y` to execute; opt a box out with "
              "`herdd label <ID> keep:<why>`.")
        sys.exit(2)
    failed = []
    if reap_ids:
        failed += lifecycle._destroy_and_revoke(reap_ids, ins, "reap_idle_destroy",
                                                noun="idle ")
    if zdestroy_ids:
        failed += lifecycle._destroy_and_revoke(zdestroy_ids, ins, "reap_zombie_destroy",
                                                noun="zombie ")
    for iid in zpark_ids:
        # Park, not destroy: stops the GPU-rate bleed, keeps the disk, and
        # lands the box in THIS reaper's stopped lane (2 h fuse, keep-label
        # escape). Intent goes to fleetd FIRST so its bid ladder reads the
        # stop as deliberate (dormant watch), never as OUTBID-to-rescue.
        lifecycle._emit_stopping_intent(iid, "reap_zombie_park", instances=ins)
        try:
            lifecycle.fleet_operator_intent(iid, "stop", reason="reap_zombie_park")
        except Exception:
            pass
        ok, err = lifecycle.stop_box(iid)
        if ok:
            print(f"parked zombie {iid} (idle reaper finishes in 2h unless "
                  f"kept/resumed)")
        else:
            print(f"FAILED to park zombie {iid}: {err}", file=sys.stderr)
            failed.append(iid)
    if failed:
        sys.exit(f"error: could not sweep {failed} — still billing, retry!")
    print(f"reaped {len(doomed)} box(es) "
          f"({len(reap_ids)} idle, {len(zdestroy_ids)} zombie-destroyed, "
          f"{len(zpark_ids)} zombie-parked); billing ended.")


# --------------------------------------------------------------------------- #
# `guard`'s two shared helpers — the SAME graded policy, presented for a human
# --------------------------------------------------------------------------- #
# Homed here, not in `boxes/health.py` and not in `cli/guard.py`:
#
# * `health.py` declines both BY NAME in its "deliberately NOT here" section
#   ("`_guard_evidence_bits`, `_guard_fix_plan`, `cmd_guard`, the ls scream and
#   fleetd's alarm text are all consumers") and sends the fix plan HERE in the
#   same breath: "`guard --fix`'s plan (and `parked_lifecycle.zombie_action` …)
#   live in `boxes/reap.py`". That is the ruling this section follows.
# * `cli/guard.py` could own them — `guard` is the only command that reaches
#   either — but `_guard_fix_plan` IS policy: it is the one place `guard --fix`
#   and the automatic reaper are proven to grade a zombie the same way, and the
#   reason that matters is a destroy that cannot be undone (46682313, killed by
#   an UNGRADED `guard --fix -y` 90 s after its co-resident twin cleared the
#   identical verdict to OK). Policy that expensive does not live in the
#   argparse layer.
#
# DUPLICATE, RULED 2026-08-16 (wave 6a): `vastlib/cli/guard.py` landed its own
# copies of both in the same wave (flagged by `gen_rename_table.py --check`).
# Bodies were identical. THIS is the home — `health.py`'s explicit routing
# quoted above, and a graded destroy is policy, not argparse — so `cli/guard.py`
# deleted its copies (and their markers) and calls
# `boxes_reap._guard_evidence_bits` / `._guard_fix_plan`. One target per name;
# `gen_rename_table.py --check` is clean on this pair.
#
# `_guard_evidence_bits` rides along because it renders the same twelve-key
# evidence dict the plan grades, and splitting the pair puts the explanation and
# the decision in different rings.


# moved-from: herdd._guard_evidence_bits
def _guard_evidence_bits(ev: Mapping[str, Any]) -> str:
    """Compact ` · `-joined measured-evidence string for one guard row (phase /
    age / last jobd heartbeat / unclaimed ticket age), skipping absent fields.
    The phase bit carries the cost profile: `loading` is GPU-unbilled,
    `env-setup` bills full GPU (the 2026-08-02 boot-phase split)."""
    bits = []
    if ev.get("phase"):
        cost = {"loading": " (GPU unbilled)", "env-setup": " (BILLED)"}
        bits.append(f"phase {ev['phase']}{cost.get(ev['phase'], '')}")
    if ev.get("boot_age_s") is not None:
        bits.append(f"box age {fmt._age_str(ev['boot_age_s'])}")
    if ev.get("jobd_hb_age_s") is not None:
        bits.append(f"jobd hb {fmt._age_str(ev['jobd_hb_age_s'])} ago")
    elif ev.get("is_jobs_box"):
        bits.append("jobd hb absent")
    if ev.get("pyhalf") is True:
        bits.append("pyhalf BROKEN (box's own beacon)")
    if ev.get("ticket_age_s") is not None:
        bits.append(f"ticket queued {fmt._age_str(ev['ticket_age_s'])}")
    if ev.get("pull_active") is not None:
        bits.append("pull ADVANCING" if ev["pull_active"] else "pull inert")
    return " · ".join(bits)


# moved-from: herdd._guard_fix_plan
def _guard_fix_plan(
        rows: Sequence[Mapping[str, Any]],
        by_iid: Mapping[str, models.Payload],
) -> list[tuple[Mapping[str, Any], Any, Any]]:
    """[(BoxHealth-dict, action, why)] for every zombie row, using the SAME
    graded policy the automatic reaper uses (`parked_lifecycle.zombie_action`).

    Why `guard --fix` is graded at all (2026-08-03): it was the ONLY ungraded
    destroyer left. The reap lane already refused to destroy a GPU-unbilled
    box without a no-progress confirmation, but `guard --fix -y` destroyed
    every ZOMBIE_* verdict on the spot — and that is the command that killed
    46682313 (fleetd journal: `operator_intent_destroy reason=
    guard_zombie_destroy`, 08:15:06Z), 90 s after its co-resident twin on the
    same image cleared the identical verdict to OK. Worse, the loop was closed:
    fleetd's alarm text for that verdict literally offered `fix: herdd guard
    --fix`, so the control plane diagnosed a false positive and then handed
    over the irreversible remedy.

    `confirmed=True` is passed deliberately: a human running `guard` IS the
    confirmation step (they are looking at the box now), which is why guard
    stays the fast lever for the EXPENSIVE running-but-dead shapes. What it can
    no longer do is destroy something the policy says to park."""
    import parked_lifecycle as _pl
    plan = []
    for h in rows:
        iid = h.get("iid")
        ev = h.get("evidence") or {}
        lab = (by_iid.get(str(iid)) or {}).get("label") or ""
        kept = labels._reap_kept(lab)
        stamped = (boxhealth._jobd_ever_stamped(iid)
                   if (ev.get("is_jobs_box") and not kept) else None)
        action, why = _pl.zombie_action(
            verdict=h.get("verdict"), is_jobs_box=ev.get("is_jobs_box"),
            jobd_ever_stamped=stamped,
            jobd_hb_read=ev.get("jobd_hb_age_s") is not None,
            label_kept=kept, confirmed=True)
        plan.append((h, action, why))
    return plan
