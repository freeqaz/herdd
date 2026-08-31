"""vastlib.jobs.risk — what a job queue would LOSE, priced in hours and seconds.

Why this exists
---------------
Cluster C23 of `herdd.py`: the pure metrics the handoff, the bid ladder and
the checkpoint watchdog all price their decisions against. Every function here
takes `now` as a parameter, none of them reads a clock, an env var, a socket or
a file, and the only stdlib touch in the whole cluster is the timestamp parse
(which now lives in `core.fmt`). That is why plan §5 names this module the
package's first to reach 100% typed and tested: it is the one place where a
number that decides whether to park a box on top of running work can be pinned
by a unit test instead of by a live fleet.

The three conventions this module is not allowed to "clean up"
--------------------------------------------------------------
* **Tri-state `None` means UNKNOWN, in three different directions.**
  `_job_eta_s`, `_job_pct`, `_jobs_min_running_eta_s`, `_jobs_work_horizon_h`,
  `_jobs_remaining_wall_h`, `_jobs_prior_runtime_h` and `_jobs_defend_hint`
  return `None` for "no estimate exists", and it must never be coerced to `0.0`
  (about to finish), to a large number (plenty of time) or to `"cheap"` (nothing
  worth defending). `_jobs_work_at_risk_h` returns `0.0` and never `None` — a
  deliberate UNDER-statement, which is why it is only ever a price and never the
  protection. `_jobs_unresumable_running` returns `0` and never `None`. There is
  no `or 0.0` anywhere in this file and adding one re-opens defect #67 (the
  2026-08-08 22:17Z incident, which read a hang detector as a work estimate and
  inflated a projected saving ~5x).
* **The `isinstance(x, bool)` rejection is asymmetric on purpose.**
  `_jobs_ckpt_stale` (checkpoint_s) and `_jobs_remaining_wall_h` (timeout_s)
  reject a `bool` before the numeric test; `_ckpt_watchdog_alarm` reads the same
  `checkpoint_s` field and does NOT. `isinstance(True, int)` is True, so
  `checkpoint_s: true` behaves differently in the 1.5x path and the 3x path.
  Ported as found (plan §7.4: no expectation changes).
* **Four inline copies of the started_at/last_resumed_ts precedence rule.**
  `_attempt_start_epoch`, `_ckpt_watchdog_alarm` (which additionally folds in
  `last_checkpoint_ts`), `_jobs_prior_runtime_h` and `_jobs_remaining_wall_h`
  each re-derive it; the docstrings claim they are unified and they are not.
  Deduping mid-port makes the diff unreviewable and risks the
  checkpoint-vs-attempt distinction — it is filed as a post-cutover cleanup.

The alarm strings `_ckpt_watchdog_alarm` returns are asserted SUBSTRINGS, not
opaque values (`"NO checkpoint"`, `"checkpoint_sync_failed"`, the job id). They
carry em dashes and split across implicit concatenation; a reflow that moves a
space across a concatenation boundary breaks the asserts.

What is deliberately NOT here
-----------------------------
* **`_ts_to_epoch`.** It lives in `core.fmt` and is called module-attribute-style
  (`fmt._ts_to_epoch(...)`). `boxes/health.py` needs it too, and `boxes` sits
  BELOW `jobs` in the §6 import DAG — owning it here would invert an edge
  (integrator ruling, 2026-08-16). Same for `_hms_secs`, which `_tqdm_points`
  consumes.
* **The rest of the tqdm progress-parser cluster** — `_step_rate`,
  `_job_progress`, `_job_cell`. They format for display; `jobs/view.py` gets them
  at plan step 5 and will import `_step_delta_s`, `_tqdm_points` and `_TQDM_RE`
  from here (same ring, no DAG problem). `_TQDM_RE` + `_tqdm_points` land in this
  module rather than in `view` because `_job_eta_s` and `_job_pct` cannot be
  ported without them, and a second copy of the bar regex is exactly the drift
  this refactor exists to remove (integrator ruling, 2026-08-16).
* **`classify_job_box_stop` / `_job_primary_evicted`.** Textually adjacent in
  `herdd.py` and pure, but they classify an EVICTION, not a job's work; they
  travel with `supervise/job_lane.py`.
* **The consumers of every number produced here** — `mk_handoff_state`,
  `_handoff_candidate_ok`, `_handoff_fence_hold`, `handoff_poll` — are in
  `bidpolicy.py` (Zone S) and are coupled to this module by dict KEY, not by
  import: `job_supervise_tick` copies these outputs into `jc` under the frozen
  names `work_at_risk_h` / `running_unresumable` / `min_running_eta_s` /
  `ckpt_stale` / `remaining_wall_h` / `timeout_ceiling_h`. No import edge is
  needed, but those key names and the None-vs-0.0 conventions above are a frozen
  contract.
* **No default `now`.** Nothing here calls `time.time()`, and a typed port must
  not introduce one — determinism under test is the whole reason this cluster is
  separable.

Provenance: verbatim-with-types move of 17 symbols from `tools/vast/herdd.py`
(plan §8 step 3, 2026-08-16), each carrying its `# moved-from:` marker. ADD-ONLY:
`herdd.py` keeps its live copies until step 6, and
`tools/vast/test_vastlib_jobs_risk.py` pins the two implementations against each
other so a rebase that edits one and not the other fails loudly. Bodies are
copied; annotations were added and two mechanical changes were forced by strict
typing, each documented at its site.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vastlib.core import fmt

import bidpolicy

#: One parsed tqdm bar: `(pct, step, total, elapsed_s, rate, unit)`. Named so
#: the shape is stated once instead of six times; not a runtime construct.
TqdmPoint = tuple[int, int, int, int, float, str]


# tqdm progress line in a heartbeat log tail:
#   ` 83%|████████▎ | 572/688 [1:55:27<23:29, 12.15s/it]`
# moved-from: herdd._TQDM_RE
_TQDM_RE = re.compile(
    r"(\d{1,3})%\|[^|\n]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<[^\],]*,\s*"
    r"([0-9.]+)\s*(s/it|it/s)\]")

#: Below this, the 1-second resolution of tqdm's elapsed stamp cannot resolve a
#: consecutive-step delta (a 0.08 s/it eval bar would read 0 or 1), so the
#: cumulative figure is the honest one.
# moved-from: herdd._STEP_DELTA_FLOOR_S
_STEP_DELTA_FLOOR_S = 2.0

# moved-from: herdd.CKPT_STALL_MULT
CKPT_STALL_MULT = 3          # alarm when a running job goes this x checkpoint_s silent


# moved-from: herdd._tqdm_points
def _tqdm_points(tail: str | None) -> list[TqdmPoint]:
    """PURE. Every TRAINING progress bar in a log tail, oldest first, as
    `(pct, step, total, elapsed_s, rate, unit)`.

    Two bars in the same tail are the raw material for a consecutive-step delta
    (`_step_rate`), which is the only honest step time available without SSH.

    DESCRIBED bars are dropped. `Loading weights: 100%|…| 339/339 [00:00<00:00,
    9415.48it/s]` and `Map:  50%|…| 2/4 [00:01<00:01, 1.20it/s]` ride in the same
    heartbeat as the training bar and match the same regex; read as training
    steps they turn a booting job into a finished one, and mixed into a delta
    pairing they produce nonsense. The dashboard's `parseTail` learned this the
    hard way (`dashboard/lib/job-events.test.ts`, "a DESCRIBED tqdm bar is not a
    training step"). The discriminator is tqdm's own `{desc}: ` separator, so
    the test is "does the same-line prefix end in a colon" rather than "is the
    prefix non-empty" — a tail is a fixed-size byte capture and can begin
    mid-line, and the training bar is often the thing it begins in the middle
    of."""
    # TYPING-FORCED (behavior-identical): the original re-read the parameter
    # `tail` inside the loop after matching on `tail or ""`. Under `str | None`
    # mypy cannot know the loop body is unreachable when `tail` is None (an
    # empty pattern space yields no matches), so the `or ""` is hoisted into a
    # local and the loop reads that. Same object whenever the loop runs.
    text = tail or ""
    pts: list[TqdmPoint] = []
    for m in _TQDM_RE.finditer(text):
        cut = max(text.rfind("\r", 0, m.start()), text.rfind("\n", 0, m.start()))
        if text[cut + 1:m.start()].rstrip().endswith(":"):
            continue                           # described bar, not a train step
        pct, step, total, elapsed, rate, unit = m.groups()
        try:
            el = fmt._hms_secs(elapsed)
        except ValueError:
            continue
        pts.append((int(pct), int(step), int(total), el, float(rate), unit))
    return pts


# moved-from: herdd._step_delta_s
def _step_delta_s(pts: Sequence[TqdmPoint] | None) -> float | None:
    """PURE. Seconds per training step from the CONSECUTIVE-STEP delta of two
    tqdm bars, or None when the tail has no usable pair.

    The numeric core of `_step_rate` (which formats the same number for display),
    factored out so the handoff ETA and the `ls` phase column can never disagree
    about how fast a job is going. tqdm's OWN rate is deliberately not a fallback
    here: on a resume it is an attempt-wide aggregate that read 13x fast on box
    47021787 (see `_step_rate`), and a 13x-fast rate is a 13x-short ETA — which
    on this path decides whether to park a box on top of running work."""
    if not pts or len(pts) < 2:
        return None
    _pct, step, total, el, _rate, _unit = pts[-1]
    for p in reversed(pts[:-1]):
        if p[1] < step and p[3] < el and p[2] == total:
            dt = (el - p[3]) / (step - p[1])
            return dt if dt >= _STEP_DELTA_FLOOR_S else None
    return None


# moved-from: herdd._ckpt_watchdog_alarm
def _ckpt_watchdog_alarm(view: object, now: float, *, mult: float = CKPT_STALL_MULT) -> str | None:
    """PURE. Missed-checkpoint watchdog (SPOT_DESIGN §3.7). Given a folded job
    `view` and the current epoch `now`, return a loud alarm string when a RUNNING
    job that is supposed to be checkpointing has gone dark, else None. Two triggers:

      1. EXPLICIT: the box published a `checkpoint_sync_failed` event (a peer's
         box-side signal, surfaced via the fold's last_event / an explicit flag) —
         the key/transport died but compute keeps burning.
      2. SILENCE: the job declares `checkpoint_s` yet no `checkpoint` event has
         landed for > mult x checkpoint_s since the later of its last checkpoint /
         resume / start. A dead ephemeral key ALSO kills event publish, so the
         silence path (not just the explicit event) is what catches the incident
         class — the watchdog must never depend on the failure event arriving.

    Returns None (never fires) unless the job is display_status==running, so a
    queued or terminal job is silent. checkpoint_s absent -> no silence alarm (the
    job opted out of checkpointing); the explicit failure still fires regardless.

    NOTE (port): unlike `_jobs_ckpt_stale`, this reader does NOT reject a `bool`
    `checkpoint_s` — `isinstance(True, int)` is True, so `checkpoint_s: true`
    reaches the numeric comparison here and is skipped there. The asymmetry is
    real and is carried verbatim (plan §7.4)."""
    if not isinstance(view, dict) or view.get("display_status") != "running":
        return None
    job = view.get("job_id") or "?"
    if view.get("last_event") == "checkpoint_sync_failed" \
            or view.get("checkpoint_sync_failed"):
        return (f"{job}: box reported checkpoint_sync_failed — checkpoint transport "
                f"is dead (bad/rotated B2 key?) while the job keeps burning compute; "
                f"results will NOT survive a preemption. Fix the key or retarget.")
    cps = view.get("checkpoint_s")
    if not isinstance(cps, (int, float)) or cps <= 0:
        return None
    epochs = [e for e in (fmt._ts_to_epoch(view.get(k)) for k in
                          ("last_checkpoint_ts", "last_resumed_ts", "started_at"))
              if e is not None]
    if not epochs:
        return None
    age = now - max(epochs)
    if age > mult * cps:
        n = view.get("n_checkpoints", 0)
        return (f"{job}: NO checkpoint for {int(age)}s (> {mult}x checkpoint_s={int(cps)}s); "
                f"{n} checkpoint(s) so far — sync may be silently dead (dead key / hung "
                f"box). Preemption now would lose all progress since the last sync.")
    return None


# moved-from: herdd._attempt_start_epoch
def _attempt_start_epoch(v: Mapping[str, Any]) -> float | None:
    """PURE. Epoch of the CURRENT attempt's start for a folded job view, or None.

    `max` because either stamp may be absent and the LATER one is this attempt:
    jobd re-execs the entrypoint under a fresh `timeout $JOB_TIMEOUT_S` on every
    restart, while the fold's `started_at` is min(claimed, started) — the FIRST
    attempt, arbitrarily stale after a preemption or a requeue. Same precedence
    as `_ckpt_watchdog_alarm` and `_jobs_remaining_wall_h`, extracted so the four
    work-awareness readers below cannot drift from it.

    NOTE (port): that last sentence is aspirational — `_ckpt_watchdog_alarm`,
    `_jobs_remaining_wall_h` and `_jobs_prior_runtime_h` each re-inline the same
    comprehension rather than calling this. All four copies are ported as found;
    unifying them is a post-cutover cleanup, not a port."""
    e = [x for x in (fmt._ts_to_epoch(v.get(k))
                     for k in ("started_at", "last_resumed_ts")) if x is not None]
    return max(e) if e else None


# moved-from: herdd._job_eta_s
def _job_eta_s(v: object, now: float | None = None) -> float | None:
    """PURE. Estimated SECONDS until this job finishes, or None.

    STRICT TRI-STATE (task #67): None means "no estimate exists" and is never
    coerced to 0 (about to finish) or to a large number (plenty of time). Every
    caller has to handle the third state explicitly — the fence hold ignores a
    None, and the ARM-side horizon refuses on one.

    Sources, in order: the training bar's remaining steps x the consecutive-step
    delta. `total`/`step` and the delta all come from `_tqdm_points`, i.e. from
    the SAME estimator `ls` shows, which exists because tqdm's own rate lies by
    13x on a resume. A job with no training bar (a non-training entrypoint, a job
    whose heartbeat has not landed yet) has no ETA, and that is the honest
    answer, not a large one. `now` is accepted for signature symmetry with the
    other readers and deliberately unused: the tqdm elapsed clock is the job's
    own, and mixing it with wall time would double-count the heartbeat lag."""
    if not isinstance(v, dict) or v.get("display_status") != "running":
        return None
    pts = _tqdm_points(v.get("last_tail") or "")
    if not pts:
        return None
    _pct, step, total, _el, _rate, _unit = pts[-1]
    dt = _step_delta_s(pts)
    if dt is None or not total or step is None or step > total:
        return None
    return max(0.0, (total - step) * dt)


# moved-from: herdd._job_pct
def _job_pct(v: object) -> int | None:
    """PURE. Completion percent from the training bar, or None (tri-state, same
    rule as `_job_eta_s`). Advisory only — see HANDOFF_WARN_PCT.

    NOTE (port): alone among the readers here it does NOT gate on
    `display_status`; a terminal job's last bar still yields a percent."""
    if not isinstance(v, dict):
        return None
    pts = _tqdm_points(v.get("last_tail") or "")
    return pts[-1][0] if pts else None


# moved-from: herdd._jobs_work_at_risk_h
def _jobs_work_at_risk_h(views: Iterable[object] | None, now: float) -> float:
    """PURE. Hours of compute a migration off this box would DISCARD.

    Per RUNNING ticket: time since its last checkpoint, or — when it has never
    checkpointed — time since the current attempt began. The MAX across tickets,
    not the sum: they share one box and would be redone in parallel on the next
    one, so what is at risk is wall time, not billed box-hours summed twice.

    0.0 when nothing is running or nothing is readable. That is an
    UNDER-statement of the overhead rather than an over-statement, which is why
    it is only ever a price (`_handoff_candidate_ok`) and never the thing that
    protects the work — `_handoff_fence_hold` is."""
    worst = 0.0
    for v in views or ():
        if not isinstance(v, dict) or v.get("display_status") != "running":
            continue
        if v.get("n_checkpoints"):
            base = fmt._ts_to_epoch(v.get("last_checkpoint_ts")) or _attempt_start_epoch(v)
        else:
            base = _attempt_start_epoch(v)
        if base is None:
            continue
        worst = max(worst, max(0.0, (now - base) / 3600.0))
    return worst


# moved-from: herdd._jobs_unresumable_running
def _jobs_unresumable_running(views: Iterable[object] | None) -> int:
    """PURE. How many RUNNING tickets have NO checkpoint to resume from.

    The count that blocks the fence (defect #62). `n_checkpoints` is the jobd
    event fold's own counter, so 0/absent means what it says: nothing of this
    attempt is on B2, and parking the box under it discards the attempt. The
    2026-08-08 incident fenced a ticket at `n_checkpoints: 0` and lost the cell
    plus a 23.8 GB base re-pull.

    NOTE (port): takes NO `now` — the only reader in this cluster that does not,
    and `job_supervise_tick` calls it with one argument. Adding one for symmetry
    would break that call site."""
    n = 0
    for v in views or ():
        if isinstance(v, dict) and v.get("display_status") == "running" \
                and not v.get("n_checkpoints"):
            n += 1
    return n


# moved-from: herdd._jobs_ckpt_stale
def _jobs_ckpt_stale(views: Iterable[object] | None, now: float, *,
                     mult: float | None = None) -> bool:
    """PURE. True when a RUNNING ticket DECLARES `checkpoint_s` and has not synced
    within `mult` x that interval — its "resumable" claim is stale, so treat it
    as unproven and hold the fence.

    Narrower than `_ckpt_watchdog_alarm` (which alarms at 3x and also fires on an
    explicit sync failure): this one only has to decide whether to park a box in
    the next few seconds, so it uses the tighter HANDOFF_CKPT_FRESH_MULT. A job
    that declares no interval opted out of checkpointing and is covered by
    `_jobs_unresumable_running` instead.

    NOTE (port): `mult=None` is a SENTINEL resolved inside the body, so the
    policy constant is read at CALL time — unlike `_ckpt_watchdog_alarm`, whose
    `mult=CKPT_STALL_MULT` binds at def time. Do not turn it into a real
    default. The constant is read as `bidpolicy.HANDOFF_CKPT_FRESH_MULT`
    (module-attribute form, plan §8(b)) where `herdd.py` read its own
    from-imported copy of the same object — identical value, and the late bind
    now lands on the module that owns the policy."""
    mult = bidpolicy.HANDOFF_CKPT_FRESH_MULT if mult is None else mult
    for v in views or ():
        if not isinstance(v, dict) or v.get("display_status") != "running":
            continue
        cps = v.get("checkpoint_s")
        if not isinstance(cps, (int, float)) or isinstance(cps, bool) or cps <= 0:
            continue
        base = fmt._ts_to_epoch(v.get("last_checkpoint_ts")) or _attempt_start_epoch(v)
        if base is None:
            continue
        if (now - base) > mult * cps:
            return True
    return False


# moved-from: herdd._jobs_min_running_eta_s
def _jobs_min_running_eta_s(views: Iterable[object] | None, now: float) -> float | None:
    """PURE. The TIGHTEST estimated seconds-to-finish across RUNNING tickets, or
    None when none of them yields an estimate. The fence hold reads this: what
    matters at the moment of the park is the job closest to being done, because
    that is the one whose loss is least excusable."""
    etas = [e for e in (_job_eta_s(v, now) for v in views or ()
                        if isinstance(v, dict)) if e is not None]
    return min(etas) if etas else None


# moved-from: herdd._jobs_work_horizon_h
def _jobs_work_horizon_h(views: Iterable[object] | None, now: float, *,
                         wall_remaining_h: float | None = None) -> float | None:
    """PURE. The handoff's WORK-AT-RISK horizon in hours, or None for UNKNOWN.

    This is the number the amortization is priced against, and it is NOT
    `_jobs_remaining_wall_h`. That one reads `timeout_s`, which is a HANG
    DETECTOR — jobmeta's outer bound on an attempt before it is killed — and on
    2026-08-08 22:17Z it was read as a work estimate: the eval ticket declared
    `timeout_s: 36000` (10 h) and had been running ~345 s, so the handoff priced
    its migration against `remaining_wall_h: 9.904` when the real remaining work
    was one to two hours. That inflated the projected saving roughly 5x, and 5x
    is the difference between a migration that pays for itself and one that does
    not.

    So: the horizon is the SUM of per-ticket ETAs (the box burns until the LAST
    ticket is done), capped by the timeout ceiling and by any --wall-budget
    remainder, and it is None unless EVERY pending ticket yields an estimate:

      * a RUNNING ticket contributes `_job_eta_s`; no ETA -> the whole horizon is
        unknown, because the unmeasured ticket could be the long one.
      * a QUEUED ticket has not started, so nothing about it has been measured at
        all -> unknown. Its `timeout_s` is not an estimate of its length; it is
        the ceiling we are refusing to treat as one.

    Under the SAFE-OFF posture that makes a jobs-lane handoff refuse on most
    boxes, which is the intended answer: the eval and training entrypoints this
    fleet runs publish a tqdm bar, and the ones that do not have not told us
    anything we could price a voluntary second rental against. UNKNOWN refuses;
    it does not assume the maximum."""
    ceiling = _jobs_remaining_wall_h(views, now, wall_remaining_h=wall_remaining_h)
    etas: list[float] = []
    for v in views or ():
        if not isinstance(v, dict):
            continue
        eta = _job_eta_s(v, now) if v.get("display_status") == "running" else None
        if eta is None:
            return None                      # one unmeasured ticket poisons the sum
        etas.append(eta)
    if not etas:
        return None                          # empty queue: nothing left to save on
    work_h = sum(etas) / 3600.0
    return work_h if ceiling is None else min(ceiling, work_h)


# moved-from: herdd._jobs_defend_hint
def _jobs_defend_hint(views: Iterable[object] | None) -> str | None:
    """PURE. The queue's lost-work hint for the bid ladder — `"dear"`,
    `"cheap"`, or None when no pending ticket carries one.

    DEAR WINS. The box runs the whole queue, so an eviction destroys every
    pending ticket's work at once; one job that says its progress is expensive
    is enough to make the box worth defending at the dearer price. Mixing is
    rare in practice (a queue is normally one bundle's tickets) and the safe
    direction when it happens is to defend, not to discard.

    None (no ticket says anything — every ticket submitted before 2026-08-14)
    leaves the derivation to `bidpolicy.resolve_defend`, which reads it off
    `checkpoint_s` exactly as the ladder did before the key existed."""
    seen: set[str] = set()
    for v in views or ():
        if not isinstance(v, dict):
            continue
        d = v.get("defend")
        if isinstance(d, str) and d.strip().lower() in bidpolicy.DEFEND_MODES:
            seen.add(d.strip().lower())
    if bidpolicy.DEFEND_DEAR in seen:
        return bidpolicy.DEFEND_DEAR
    return bidpolicy.DEFEND_CHEAP if seen else None


# moved-from: herdd._jobs_prior_runtime_h
def _jobs_prior_runtime_h(views: Iterable[object] | None, now: float) -> float | None:
    """PURE. Hours of work an eviction would destroy outright — the LONGEST
    running ticket's accumulated wall time, or None when nothing is running or
    no attempt timestamp is readable.

    Only meaningful for an un-checkpointed job, and only consumed when that job
    is explicitly `defend: dear`: with nothing synced there is no "since the
    last checkpoint" and the replacement restarts the ticket from zero, so the
    whole elapsed attempt is the lost work.

    `last_resumed_ts` outranks `started_at` for the same reason it does in
    `_jobs_remaining_wall_h`: the fold's `started_at` is the FIRST attempt, and
    after a preemption or requeue the work before the resume is already gone —
    counting it would bill the ceiling twice for the same lost hours. MAX, not
    sum: the tickets ran concurrently on one box, so wall time does not add.

    NOTE (port): the `elapsed > 0` test is STRICT, so a zero or negative elapsed
    contributes nothing and can leave the result None even with a running ticket
    present (pinned by test_defend_hint.py)."""
    best: float | None = None
    for v in views or ():
        if not isinstance(v, dict) or v.get("display_status") != "running":
            continue
        starts = [e for e in (fmt._ts_to_epoch(v.get(k))
                              for k in ("started_at", "last_resumed_ts"))
                  if e is not None]
        if not starts:
            continue
        elapsed = (now - max(starts)) / 3600.0
        if elapsed > 0:
            best = elapsed if best is None else max(best, elapsed)
    return best


# moved-from: herdd._jobs_remaining_wall_h
def _jobs_remaining_wall_h(views: Iterable[object] | None, now: float, *,
                           wall_remaining_h: float | None = None) -> float | None:
    """PURE. The jobs-lane TIMEOUT CEILING in HOURS — the outer bound on how much
    longer this box can be asked to work, measured off the queue, never assumed.

    NOT the work estimate. `timeout_s` is a hang detector, and reading it as an
    ETA is defect #67 (see `_jobs_work_horizon_h`, which is what the handoff
    prices against and which uses this only as a CAP). It stays exactly as
    landed for that role, and for anything else that wants a runway bound:

        min( --wall-budget remainder (when the operator set one),
             sum over pending tickets of their remaining timeout_s )

    `timeout_s` is the one runway number the queue genuinely knows: jobmeta
    validates it as a positive int on every job config (DEFAULT_TIMEOUT_S when
    unset, refused over MAX_TIMEOUT_S) and it rides the `submitted` event into
    the fold. It is a per-ATTEMPT budget — jobd re-execs the entrypoint under a
    fresh `timeout $JOB_TIMEOUT_S` on each restart (onstart/jobd.sh) — so a
    RUNNING ticket's clock starts at `last_resumed_ts` when there is one and at
    `started_at` otherwise. That precedence matters: the fold's `started_at` is
    min(claimed, started), i.e. the FIRST attempt, which after a preemption or a
    requeue is arbitrarily stale. Per-ticket remainders are clamped at >= 0 so a
    straggler already past its own timeout contributes nothing rather than
    subtracting from a neighbour's genuine runway. A ticket that is not running
    yet has spent none of its budget, so it contributes the whole timeout_s. It
    is a SUM and not a max because the box keeps burning until the LAST ticket
    is done.

    Returns None when nothing yields a bound — an empty queue, a ticket with no
    readable timeout_s, a running ticket with no attempt timestamp — even if a
    --wall-budget remainder is available, because that bounds our SPEND, not the
    work: savings only accrue while there is something left to run.

    Failing closed here is the point. This number feeds the handoff horizon,
    whose contract is that any missing input refuses because the cost cannot be
    bounded; the flat `24.0` this replaced violated that contract from OUTSIDE
    the function. fleetd's JOBS_POLICY_DEFAULTS seed `wall_budget=None`, so
    EVERY jobs watch under the daemon priced its migration against a full day of
    runway nobody had measured — and on 2026-08-08 that armed a voluntary
    handoff on a running, healthy box roughly 90 s from the end of a cell, at
    `n_checkpoints: 0`, discarding the work. The refusal direction is safe by
    construction: a handoff is a VOLUNTARY cost optimisation, so refusing costs
    at most a missed saving and never the workload."""
    bounds: list[float] = []
    for v in views or ():
        if not isinstance(v, dict):
            continue
        to = v.get("timeout_s")
        if not isinstance(to, (int, float)) or isinstance(to, bool) or to <= 0:
            continue                                  # no declared runway to read
        if v.get("display_status") != "running":
            bounds.append(float(to))                  # queued/interrupted: unspent
            continue
        # `max` because either field may be absent and the LATER one is the
        # current attempt's start (same precedence as _ckpt_watchdog_alarm).
        starts = [e for e in (fmt._ts_to_epoch(v.get(k))
                              for k in ("started_at", "last_resumed_ts"))
                  if e is not None]
        if not starts:
            continue                                  # burning, but since when?
        bounds.append(max(0.0, float(to) - (now - max(starts))))
    if not bounds:
        return None
    queue_h = sum(bounds) / 3600.0
    return queue_h if wall_remaining_h is None else min(wall_remaining_h, queue_h)
