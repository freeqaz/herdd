"""vastlib.cli._ls_render — the fleet snapshot: one read, two renderings.

Why this exists
---------------
`cli-surface.json` **H3**: five helpers behind `herdd ls` are reachable from
MORE THAN ONE command, so no `cli/<command>.py` can own them. `_gather_ls_data`,
`_market_map` and `_stale_image_ids` are called by `ls` **and** by `dash-cache`
(the sqlite snapshot the dashboard reads); `_render_ls` and `_render_minimal`
are the two renderings of the dict the first three produce, and they must stay
next to it or the "pure over `data`" contract below is only a comment. A naive
one-module-per-command split duplicates all five.

`storage/dashcache.py` already names this module's territory from the other
side: `_gather_ls_data`, `_job_cell` and `_ACTIVE_JOB_STATES` are "cli/ls
territory", `cli` sits ABOVE `storage` in the plan §5 DAG, and so they arrive
there as `DashDeps` injections **permanently**. `cli/dash_cache.py` builds that
`DashDeps` from THIS module. (`_job_cell` itself is the exception: it landed in
`jobs/view.py` with the fold that produces its input, so both readers share one
definition rather than a copy.)

The two contracts this module freezes
-------------------------------------
1. **`_render_ls` is pure over `data`.** It renders a `_gather_ls_data` dict —
   or a *snapshot* of one loaded off disk — into a list of logical lines, and it
   performs no I/O of its own. That is what guarantees the stale paint (drawn
   from the snapshot the moment `ls` starts) and the fresh paint (drawn when the
   network read lands) have the same shape; a network call sneaked in here makes
   the stale frame lie.
2. **`--minimal` is a FROZEN TSV contract** (plan §4). Its column ORDER, its
   header line and its emptiness convention (empty cell = N/A) are read by agent
   consumers and by the dashboard, and reordering a column silently re-labels
   every value downstream of it. The tuple lives in `_MINIMAL_COLS` below and is
   pinned column-by-column by
   `test_vastlib_cli_helpers.py::test_minimal_tsv_column_order_is_frozen`;
   `test_disk_sizing.py` independently asserts on the `disk_gb`/`disk_used_gb`
   pair. Adding a column APPENDS; nothing is ever inserted or renamed.

What is deliberately NOT here
-----------------------------
* **`cmd_ls` itself**, the snapshot cache (`_ls_snapshot_save/_load`,
  `_LS_SNAPSHOT`) and `fleet_daemon_banner`. They are `ls`-private — one command
  reaches them — and they move with `cli/ls.py`. The banner is also an I/O call
  (it pings fleetd), which is exactly what contract 1 keeps out of the renderer:
  `cmd_ls` computes it and passes it in as `banner`.
* **`_rates` / `_money` / `_ACTIVE_JOB_STATES`-adjacent price arithmetic.**
  `_rates` is `core.models`' (it is dict-accessor arithmetic over an instance
  payload) and `_money` is `core.fmt`'s. Only `_ACTIVE_JOB_STATES` stayed with
  the renderers, because "which job states make a box ACTIVE" is a display
  grouping, not a policy: `boxes.reap` and `fleet` decide liveness from
  `bidpolicy.LIVE_STATES` and from health verdicts, never from this tuple.
* **The guard verdict sets.** `_GUARD_ZOMBIE_VERDICTS` / `_GUARD_VERDICT_SHORT`
  collapsed into `boxes.health.GuardVerdict` (plan §5); this module asks the
  lattice its two questions (`verdict_is_zombie`, `verdict_short`) instead of
  re-deriving membership. The scream BANNER stays here — the enum absorbed set
  membership and the short-tag table, deliberately not either caller's
  rendering.

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 6 (`cli/`) of `docs/plans/vast-tooling-refactor-v2.md`. Every symbol
carries its `# moved-from:` marker. Step 6 is ADD-ONLY for `herdd.py`, so the
flat copies stay live until 6d and the help-tree/CLI-surface diff test compares
the two while both exist.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any, Callable, Mapping, Sequence

import imageref

from vastlib.boxes import health, lifecycle, reap
from vastlib.core import fmt, labels, models
from vastlib.fleet import client as fleet_client
from vastlib.jobs import view
from vastlib.market import pricing

import bidpolicy
import jobmeta

Payload = models.Payload


# moved-from: herdd._market_map
def _market_map(instances: Sequence[Payload],
                enabled: bool = True,
                prog: Any = None) -> dict[str, models.MachineMarket]:  # noqa: ANN401 — fmt._Progress | None
    """machine_id(str) -> {"offers":[...], "max_gpus":int}, one parallel soft
    POST per UNIQUE machine. {} when disabled (--no-spot) or on failure — a
    machine with no live read is simply absent (prices fall back to the
    instance's own stale fields; availability shows unknown)."""
    out: dict[str, models.MachineMarket] = {}
    if not enabled:
        return out
    # The set is annotated (rather than inlined into `sorted`) only so the
    # element type is `Any` and not `Any | None`: `Mapping.get` is Optional-typed
    # even behind the truthiness filter, and `sorted` rejects an Optional key.
    mid_set: set[Any] = {i.get("machine_id") for i in instances if i.get("machine_id")}
    mids = sorted(mid_set)
    if not mids:
        return out
    if prog:
        prog.add(len(mids))

    def probe(mid: object) -> list[models.MachineRow] | None:
        try:
            return pricing._machine_offers_soft(mid)
        finally:
            if prog:
                prog.tick()

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(mids))) as ex:
        for mid, offers in zip(mids, ex.map(probe, mids)):
            if offers is not None:
                out[str(mid)] = {"offers": offers,
                                 "max_gpus": max((o["g"] for o in offers),
                                                 default=0)}
    return out


# non-terminal job states that make a live box "active" (has work bound to it)
# moved-from: herdd._ACTIVE_JOB_STATES
_ACTIVE_JOB_STATES = ("running", "claimed", "submitted")


# moved-from: herdd._stale_image_ids
def _image_check_ids(instances: Sequence[Payload],
                     prog: Any = None) -> dict[str, list[Any]]:  # noqa: ANN401 — fmt._Progress | None
    """`{'stale': [...], 'unresolved': [...]}` over stamped boxes.

    `stale`      the stamped launch digest (HERDD_IMAGE_DIGEST) no longer
                 matches the registry tag — an env/image push landed after
                 launch, and a park/resume will NOT pick it up.
    `unresolved` the tag did not resolve AT ALL, so the box was not checked.

    The second list is the whole point of the split. Resolution degrades to
    None on a missing $REGISTRY_AUTH_SECRET (registry.example.com refuses
    anonymous reads), and a None used to be folded into "not stale" — so an
    unset secret rendered byte-identically to a clean fleet. "Could not check"
    must never look like "checked, fine".

    Digest resolution is a network round-trip per unique image (creds-ful
    `skopeo inspect` on our R2 registry), so images resolve in parallel; order
    follows the instance
    list.
    """
    stamped = [(i, models._instance_env(i).get(imageref.IMAGE_DIGEST_ENV))
               for i in instances]
    stamped = [(i, s) for i, s in stamped if s]
    imgs = sorted({models._instance_image(i) for i, _ in stamped})
    if not imgs:
        return {"stale": [], "unresolved": []}
    if prog:
        prog.add(len(imgs))

    def resolve(img: str) -> Any:  # noqa: ANN401 — imageref returns a digest str | None
        try:
            return imageref.image_tag_digest(img)
        finally:
            if prog:
                prog.tick()

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(6, len(imgs))) as ex:
        digs = dict(zip(imgs, ex.map(resolve, imgs)))
    stale: list[Any] = []
    unresolved: list[Any] = []
    for i, s in stamped:
        cur = digs.get(models._instance_image(i))
        if not cur:
            unresolved.append(i.get("id"))
        elif cur != s:
            stale.append(i.get("id"))
    return {"stale": stale, "unresolved": unresolved}


def _stale_image_ids(instances: Sequence[Payload],
                     prog: Any = None) -> list[Any]:  # noqa: ANN401 — fmt._Progress | None
    """Just the stale half of `_image_check_ids`, for callers that only paint
    the row tag. `ls` uses the full dict — it has a banner for both halves."""
    return _image_check_ids(instances, prog)["stale"]


def _start_epoch(i: Payload) -> float | None:
    """The box's billing anchor (`start_date`), tolerant to the garbage the
    API occasionally serves — the same guard `fleet.rows.reconcile_rows`
    carries. None = no usable anchor, render nothing."""
    raw: Any = i.get("start_date")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _fleet_budget_map(prog: Any = None) -> dict[str, Any]:  # noqa: ANN401 — fmt._Progress | None
    """`{"by_box": {iid: {"spend_usd", "budget_usd"}}, "total_usd": float|None}`
    from one fleetd `status` read. {} on machines with no fleetd, a down
    daemon, or any socket failure — `ls` renders supervision, never requires
    it. Spend prefers the CEILING's cumulative counter, same rule as the
    dashboard's `_dash_write_fleet`: a watch's own counter reads $0.00 on a
    box that inherited a ceiling with most of it drawn."""
    if prog:
        prog.add(1)
    try:
        if not (os.path.exists(fleet_client.fleet_sock_path())
                or os.path.isdir(fleet_client.fleet_state_dir())):
            return {}
        ok, data, _err = fleet_client.fleet_request(
            "status", _timeout=8, _retries=0)
        if not ok:
            return {}
        by_box: dict[str, Any] = {}
        for r in (data.get("rows") or []):
            iid = r.get("iid")
            if iid is None:
                continue
            spend = r.get("ceiling_spend_usd")
            if spend is None:
                spend = r.get("spend_usd")
            by_box[str(iid)] = {"spend_usd": spend,
                                "budget_usd": r.get("budget_usd")}
        return {"by_box": by_box, "total_usd": data.get("spend_total_usd")}
    except Exception:
        return {}
    finally:
        if prog:
            prog.tick()


# moved-from: herdd._gather_ls_data
def _gather_ls_data(no_spot: bool = False,
                    prog: Any = None) -> dict[str, Any]:  # noqa: ANN401 — fmt._Progress | None
    """One full fleet read for the ls view: instances first, then the three
    slow network phases (B2 jobs fold | spot floors | stale-image digests)
    fanned out concurrently. `prog` (a _Progress) is ticked per parallel unit
    so the caller's spinner can show live `done/total`. Returns a JSON-safe
    dict shared by the renderer and the snapshot cache."""
    ins = lifecycle._instances()
    live = [i.get("id") for i in ins
            if (i.get("actual_status") or "").lower() in bidpolicy.LIVE_STATES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        fj = ex.submit(view._fold_fleet_jobs, set(live), prog)
        fs = ex.submit(_market_map, ins, not no_spot, prog)
        fi = ex.submit(_image_check_ids, ins, prog)
        fb = ex.submit(_fleet_budget_map, prog)
        jobs_by_box, market, img_check, budgets = (
            fj.result(), fs.result(), fi.result(), fb.result())
    stale_ids = img_check["stale"]
    unchecked_ids = img_check["unresolved"]
    idle_secs = reap._idle_secs_map(ins, live)
    # Durable zombie-sweep: classify every box so `ls` can scream. Bounded extra
    # B2 reads (only running jobs-boxes with no fresh in-fold heartbeat); a
    # read-failure never breaks the listing.
    try:
        fleet_health = health.gather_fleet_health(ins, jobs_by_box)
    except Exception:
        fleet_health = {}
    return {"ts": time.time(), "no_spot": bool(no_spot), "instances": ins,
            "live_ids": live, "jobs_by_box": jobs_by_box,
            "market": market, "stale_ids": stale_ids,
            "unchecked_image_ids": unchecked_ids,
            "idle_secs": idle_secs, "health": fleet_health,
            # fleetd budget/spend per watched box; absent on old snapshots and
            # {} without a daemon — both render as "nothing to say".
            "fleet_watch": budgets.get("by_box") or {},
            "fleet_spend_total": budgets.get("total_usd")}


# moved-from: herdd._render_ls
def _render_ls(data: Mapping[str, Any],
               pal: fmt._Pal,
               banner: str | None = None,
               cols: int | None = None) -> list[str]:
    """Render one fleet snapshot to a list of logical lines. Pure over `data`
    (the _gather_ls_data / snapshot dict) so the stale paint and the fresh
    paint are guaranteed the same shape. `cols` (terminal width) drives the
    reflow: when the one-line table overflows it, box rows collapse to a
    two-line compact layout (identity+price up top; gpu/ssh/image dimmed
    below) and long ids keep their distinctive TAIL when ellipsized."""
    ins = data.get("instances") or []
    lines: list[str] = []
    if banner:
        lines.append(banner)
    if not ins:
        lines.append("no instances.")
        return lines
    # Durable zombie-sweep scream: any ZOMBIE_* box gets a LOUD line at the TOP
    # of the view (before the healthy table) so a routine `ls` cannot miss it.
    # Healthy fleets add nothing here — normal output stays stable.
    fleet_health = data.get("health") or {}
    zombies = [h for h in fleet_health.values()
               if health.verdict_is_zombie(h.get("verdict"))]
    if zombies:
        zombies.sort(key=lambda h: -(h.get("age_s") or 0))
        lines.append(pal.red(pal.bold(
            f"!! {len(zombies)} ZOMBIE BOX(ES) — dead boot / dead jobd, burning "
            f"schedule or GPU. Sweep: herdd guard --fix (graded: it PARKS a "
            f"GPU-unbilled loading stall, destroys only a billing dead box)")))
        for h in zombies:
            short = health.verdict_short(h.get("verdict"))
            age = fmt._age_str(h.get("age_s") or 0)
            lines.append(pal.red(
                f"   !! {h.get('iid')}  {short} {age}: {h.get('reason')}"))
        lines.append("")
    # Boot-phase visibility (2026-08-02 split): a BOOTING box gets one yellow
    # line naming WHICH phase it is in, because the correct response differs —
    # `loading` (image pull / vast-side standup) is GPU-unbilled and deserves
    # patience; `env-setup` (running, onstart/jobd bootstrap) bills full GPU
    # and deserves scrutiny. Healthy steady-state fleets add nothing here.
    booting = [h for h in fleet_health.values()
               if h.get("verdict") == health.GUARD_BOOTING]
    if booting:
        booting.sort(key=lambda h: -(h.get("age_s") or 0))
        for h in booting:
            ph = (h.get("evidence") or {}).get("phase") or "boot"
            # `cost` is re-bound far below in the RETAINED footer, where it can
            # be None; widened here so one function scope has one type for it.
            cost: str | None = ("BILLED full GPU" if ph == "env-setup"
                                else "GPU unbilled" if ph == "loading" else "")
            age = fmt._age_str(h.get("age_s") or 0)
            lines.append(pal.yellow(
                f"   ·· {h.get('iid')}  booting[{ph}] {age}"
                + (f" — {cost}" if cost else "")))
        lines.append("")
    # A LOADING_SLOW box used to appear here as a `!! ZOMBIE` scream (it was
    # classified a stall). It is an advisory now, so it gets its own yellow
    # line — over the deadline is still worth SEEING, it is just not worth
    # destroying (2026-08-03). Silence would be the wrong overcorrection.
    slow = [h for h in fleet_health.values()
            if h.get("verdict") == health.GUARD_LOADING_SLOW]
    if slow:
        slow.sort(key=lambda h: -(h.get("age_s") or 0))
        for h in slow:
            lines.append(pal.yellow(
                f"   ~~ {h.get('iid')}  loading-slow {fmt._age_str(h.get('age_s') or 0)} "
                f"— past the deadline but the pull is STILL ADVANCING; GPU "
                f"unbilled. Let it finish, or `herdd stop` (recoverable)"))
        lines.append("")
    live = set(data.get("live_ids") or [])
    jobs_by_box = data.get("jobs_by_box") or {}
    market = data.get("market") or {}
    stale_ids = list(data.get("stale_ids") or [])
    stale_set = set(stale_ids)
    # Absent from an OLD cached snapshot (pre-2026-08-21) — treated as "nothing
    # to say", not as "all checked", because an old snapshot genuinely knows
    # nothing about which boxes went unchecked.
    unchecked_ids = list(data.get("unchecked_image_ids") or [])
    idle_secs = data.get("idle_secs") or {}
    # fleetd budget/spend per watched box — absent from old snapshots and from
    # fleets with no daemon, both of which render as "nothing to say".
    fleet_watch = data.get("fleet_watch") or {}
    # Uptime is measured against the GATHER's clock, not the paint's, so the
    # renderer stays pure over `data` and a --cached view ages honestly with
    # its banner instead of silently inflating every box's uptime.
    data_ts = data.get("ts") or time.time()
    budget = cols if cols is not None else 10 ** 9

    def _ell(s: str, n: int) -> str:
        return s if len(s) <= n else s[: max(1, n - 1)] + "…"

    def _ell_left(s: str, n: int) -> str:
        return s if len(s) <= n else "…" + s[-(max(1, n - 1)):]

    active: list[tuple[Payload, list[Any], list[Any]]] = []
    running: list[tuple[Payload, list[Any], list[Any]]] = []
    suspended: list[tuple[Payload, list[Any], list[Any]]] = []
    for i in ins:
        jobs = jobs_by_box.get(str(i.get("id")), [])
        act = [v for v in jobs if v.get("display_status") in _ACTIVE_JOB_STATES]
        if i.get("id") not in live:
            suspended.append((i, jobs, []))
        elif act:
            active.append((i, jobs, act))
        else:
            running.append((i, jobs, []))

    def price_bits(i: Payload) -> tuple[str, float | None, str, float | None]:
        # which billing mode this box IS (reserved/on-demand vs interruptible
        # spot), the rate for that mode, and the counterfactual other rate.
        reserved, spot, avail = models._rates(i, market)
        is_bid = bool(i.get("is_bid"))
        if is_bid:
            return "spot", spot, "on-dem", reserved
        return "on-dem", reserved, "spot", spot

    def fields(i: Payload) -> dict[str, Any]:
        mode, billed, alt_mode, alt = price_bits(i)
        reserved, spot, avail = models._rates(i, market)
        is_live = i.get("id") in live
        util = i.get("gpu_util")
        cpu = i.get("cpu_util")
        sec = idle_secs.get(str(i.get("id")))
        # Which number earns the one utilization column. A dedicated CPU box
        # (compile/search work, no model endpoint) has a GPU pinned at 0, and
        # showing that 0 — in the yellow this column paints "you are paying for
        # an idle GPU" — describes a box that is in fact saturated. So when the
        # GPU is idle and the CPU is not, the CPU figure wins the cell.
        # `kind` rather than parsing the string back: the caller used to
        # recover the number with float(txt.rstrip("%")), which a "cpu 19.98"
        # would have crashed on.
        # Narrowed to `float | None` HERE rather than carried as a pair of
        # booleans: a bool computed elsewhere does not narrow the value for a
        # type checker, so `cpu > CPU_BUSY_UTIL` reads as `Any | None > float`
        # under mypy --strict even when the guard makes it safe.
        gpu_u: float | None = (util if is_live
                               and isinstance(util, (int, float)) else None)
        cpu_u: float | None = (cpu if is_live
                               and isinstance(cpu, (int, float)) else None)
        show_cpu = (cpu_u is not None and cpu_u > health.CPU_BUSY_UTIL
                    and (gpu_u is None or gpu_u < 30))
        if show_cpu and cpu_u is not None:
            util_txt, util_kind = f"cpu {cpu_u:.0f}", "cpu"
        elif gpu_u is not None:
            util_txt, util_kind = f"{gpu_u:.0f}%", "gpu"
        else:
            util_txt, util_kind = "", ""
        # Uptime + accrued-cost estimate, LIVE boxes only: `start_date` is the
        # billing anchor (creation — the API has no stop timestamp, so on a
        # stopped box it measures nothing worth printing). The dollar figure is
        # the same UPPER BOUND `fleet spend --reconcile` quotes: dph x age,
        # which over-counts loading windows and any parked stretch — hence `≈`.
        start = _start_epoch(i)
        up_s = ((data_ts - start)
                if (is_live and start and data_ts > start) else None)
        dph = models._num_dph(i.get("dph_total"))
        est = (dph * up_s / 3600.0) if (dph and up_s) else None
        # fleetd watch economics for this box: `$spent/$cap`, or `Σ$spent` for
        # a watch with no cap. Empty = not watched (the banner screams if the
        # whole daemon is down, so silence here means just this box).
        wtch = fleet_watch.get(str(i.get("id"))) or {}
        sp, bg = wtch.get("spend_usd"), wtch.get("budget_usd")
        # two decimals, not fmt.dollars' three: these are totals, not rates
        if isinstance(sp, (int, float)) and isinstance(bg, (int, float)) and bg:
            bud_txt, bud_frac = f"${sp:.2f}/${bg:.2f}", sp / bg
        elif isinstance(sp, (int, float)):
            bud_txt, bud_frac = f"Σ${sp:.2f}", None
        else:
            bud_txt, bud_frac = "", None
        return {
            "id": str(i.get("id")),
            "st": str(i.get("actual_status") or i.get("cur_state") or "?"),
            "gpu": f"{i.get('num_gpus', '?')}x {i.get('gpu_name', '?')}",
            "util": util_txt,
            "util_kind": util_kind,
            "mode": mode,
            "billed": fmt._money(billed),
            "res": fmt._money(reserved),
            "spot": fmt._money(spot),
            "avail": avail,       # True / False / None(unknown)
            "badge": ("✓" if avail is True else
                      "✗ taken" if avail is False else "?"),
            "stor": (fmt._money(models._storage_day(i)) + "/day"
                     if models._storage_day(i) is not None else ""),
            "up": (("up " + fmt._age_str(up_s)
                    + (f" ≈${est:.2f}" if est is not None else ""))
                   if up_s is not None else ""),
            "up_sec": up_s,
            "bud": bud_txt,
            "bud_frac": bud_frac,
            "idl": (("idle " + fmt._age_str(sec)) if sec is not None else ""),
            "idl_sec": sec,
            "ssh": f"{i.get('ssh_host', '-')}:{i.get('ssh_port', '-')}",
            "img": fmt._image_short(models._instance_image(i)),
            "lbl": i.get("label") or "",
        }

    F = {id(i): fields(i) for i, _, _ in active + running + suspended}
    w = {k: max([len(f[k]) for f in F.values()] + [1])
         for k in ("id", "st", "gpu", "util", "mode", "billed", "res",
                   "spot", "badge", "stor", "idl", "ssh", "img", "up", "bud")}

    def idle_cell(f: Mapping[str, Any], pad: int = 0) -> str:
        # idle age heats up with duration: dim < 2h < yellow < 24h < red.
        # Pad BEFORE coloring so the suspended grid stays aligned.
        if not f["idl"]:
            return " " * pad
        txt = f"{f['idl']:<{pad}}" if pad else f["idl"]
        sec = f["idl_sec"] or 0
        return (pal.red(txt) if sec >= 86400
                else pal.yellow(txt) if sec >= 7200 else pal.dim(txt))

    def up_cell(f: Mapping[str, Any], pad: int = 0) -> str:
        # `up 7h3m ≈$4.71` — context, not an accusation, so it stays dim.
        # Pad BEFORE coloring so the live grid stays aligned.
        if not f["up"]:
            return " " * pad
        return pal.dim(f"{f['up']:<{pad}}" if pad else f["up"])

    def bud_cell(f: Mapping[str, Any], pad: int = 0) -> str:
        # fleetd watch spend vs cap, heating up as the cap approaches:
        # dim < 60% < yellow < 90% < red (red also covers an over-drawn cap).
        # A capless watch (`Σ$…`) has no fraction and stays dim.
        if not f["bud"]:
            return " " * pad
        txt = f"{f['bud']:<{pad}}" if pad else f["bud"]
        frac = f["bud_frac"]
        if frac is None:
            return pal.dim(txt)
        return (pal.red(txt) if frac >= 0.90
                else pal.yellow(txt) if frac >= 0.60 else pal.dim(txt))

    def util_cell(f: Mapping[str, Any], pad: bool = True) -> str:
        # GPU busy-ness of a live box: bold ≥90% (saturated — earning its
        # keep), plain in between, yellow <30% (paying compute for an idle
        # GPU). Blank pad for stopped rows keeps the shared grid aligned.
        txt = f"{f['util']:>{w['util']}}" if pad else f["util"]
        if not f["util"]:
            return txt
        # A CPU-shaped box is working, so it never wears the idle-GPU yellow —
        # that colour is an accusation, and here it would be a false one.
        if f.get("util_kind") == "cpu":
            return pal.bold(txt)
        u = float(f["util"].rstrip("%"))
        return (pal.bold(txt) if u >= 90
                else pal.yellow(txt) if u < 30 else txt)

    st_color: dict[str, Callable[[str], str]] = {
        "running": pal.green, "loading": pal.yellow,
        "created": pal.yellow}
    job_color: dict[str, Callable[[str], str]] = {
        "running": pal.green, "claimed": pal.yellow,
        "submitted": pal.yellow, "interrupted": pal.yellow,
        "failed": pal.red, "done": pal.dim}

    def mode_word(mode: str, pad: int = 0) -> str:
        # spot = interruptible → yellow (can be evicted); on-dem = reserved →
        # left in the default weight. Pad BEFORE coloring so alignment holds.
        tail = " " * max(0, pad - len(mode))
        return (pal.yellow(mode) if mode == "spot" else mode) + tail

    def avail_badge(f: Mapping[str, Any], pad: int = 0) -> str:
        # can this stopped box actually resume right now? (machine has a
        # rentable offer of >= its GPU count). `resume ✓` green = gpus free,
        # `resume ✗ taken` red = another renter holds them, dim `?` unknown.
        txt = f"{f['badge']:<{pad}}" if pad else f["badge"]
        cf = (pal.green if f["avail"] is True
              else pal.red if f["avail"] is False else pal.dim)
        return pal.dim("resume ") + cf(txt)

    def rate_pair(f: Mapping[str, Any], pad: bool = True) -> str:
        """`on-dem $0.673 · spot $0.274`, cells right-aligned to the fleet-wide
        money widths when `pad`. Dimmed when the machine is taken — those
        rates are what resuming WOULD cost, not something you can click now."""
        res = f"{f['res']:>{w['res']}}" if pad else f["res"]
        spt = f"{f['spot']:>{w['spot']}}" if pad else f["spot"]
        if f["avail"] is False:
            return pal.dim(f"on-dem {res} · spot {spt}")
        return (pal.dim("on-dem ") + res + pal.dim(" · ")
                + pal.yellow("spot") + " " + spt)

    def econ_seg(i: Payload, compact: bool) -> str:
        """The money segment, so it's unambiguous which billing mode applies.
        LIVE box: `<mode> <billed-rate>`, the billed rate being the ACTIVE
        cost → bold-green. STOPPED box: compute isn't billed, so lead with
        `storage $X/day` (the real burn → bold-green) + idle age, then whether
        it can resume + BOTH resume rates (reserved `on-dem` / interruptible
        `spot`). Every cell pads to its fleet-wide width in the wide layout so
        the suspended rows read as one aligned grid."""
        f = F[id(i)]
        if i.get("id") in live:
            seg = (mode_word(f["mode"], w["mode"]) + " "
                   + pal.bgreen(f"{f['billed']:>{w['billed']}}"))
            if compact:
                # compact head keeps only rate + budget (the two live-money
                # facts); uptime rides the muted info line in box_lines.
                if f["bud"]:
                    seg += "  " + bud_cell(f)
                return seg
            return "  ".join(x for x in (
                seg,
                up_cell(f, pad=w["up"]) if w["up"] > 1 else "",
                bud_cell(f, pad=w["bud"]) if w["bud"] > 1 else "") if x)
        # compact drops the "storage " word (the SUSPENDED group already says
        # so) to claw back width on very narrow terminals.
        if f["stor"]:
            stor = pal.bgreen(f["stor"] if compact
                              else f"{f['stor']:>{w['stor']}}")
            if not compact:
                stor = pal.dim("storage ") + stor
        else:
            stor = ""
        if compact:
            # compact head: just storage/idle; availability + both resume
            # rates ride the muted line added in box_lines, to save width.
            return "  ".join(x for x in (stor, idle_cell(f)) if x)
        return "  ".join(x for x in (
            stor, idle_cell(f, pad=w["idl"]),
            avail_badge(f, pad=w["badge"]), rate_pair(f),
            # a parked box can still sit under a fleetd watch — its drawn
            # budget is what a resume inherits, so it stays visible here.
            bud_cell(f) if f["bud"] else "") if x)

    def resume_prices(f: Mapping[str, Any]) -> str:
        """Availability + the two resume rates, for the compact extra line of
        a stopped box."""
        return avail_badge(f) + "  " + rate_pair(f, pad=False)

    def box_lines(i: Payload, compact: bool) -> list[str]:
        # per-column hue map (scan-at-a-glance): id bold-white, status
        # green/yellow/dim, gpu magenta, mode word (spot=yellow), active cost
        # bold-green, label cyan, image blue, ssh dim.
        f = F[id(i)]
        is_live = i.get("id") in live
        bullet = pal.green("●") if is_live else pal.dim("○")
        idf = f"{f['id']:>{w['id']}}"
        if is_live:
            idf = pal.bold(idf)
        stf = st_color.get(f["st"], pal.dim)(f"{f['st']:<{w['st']}}")
        econ = econ_seg(i, compact)
        tag = ("  " + pal.red("STALE-IMAGE")) if i.get("id") in stale_set else ""
        if not compact:
            # stopped rows carry no ssh endpoint (it dies on park — a resume
            # gets a fresh host:port) and no util pad: less ink, same info.
            gpuf = pal.magenta(f"{f['gpu']:<{w['gpu']}}")
            if is_live:
                gpuf += " " + util_cell(f)
            imgf = pal.blue(f"{f['img']:<{w['img']}}")
            lblf = pal.cyan(f["lbl"]) if f["lbl"] else ""
            segs = [f"  {bullet} {idf}", stf, gpuf, econ + " "]
            if is_live:
                segs.append(pal.dim(f"{f['ssh']:<{w['ssh']}}"))
            segs += [imgf, lblf]
            return [("  ".join(s for s in segs if s) + tag).rstrip()]
        # compact: identity + money up top (label stays prominent — it names
        # the run/serve), gpu/ssh/image demoted to a muted second line.
        head = f"  {bullet} {idf}  {stf}  {econ}"
        if f["lbl"]:
            avail = budget - fmt._visw(head) - fmt._visw(tag) - 2
            if avail >= 4:               # else no room — drop label, no overflow
                head += "  " + pal.cyan(_ell(f["lbl"], avail))
        head += tag
        sep = pal.dim(" · ")
        gput = f["gpu"] + (f" {f['util']}" if f["util"] else "")
        # uptime (+ ≈cost) is context, so in compact it joins the muted line
        mid = ([f["up"]] if f["up"] else []) \
            + ([f["ssh"]] if is_live else [])   # no ssh on stopped rows
        fixed = 6 + len(gput) + sum(len(x) + 3 for x in mid) + 3
        img = _ell(f["img"], max(6, budget - fixed))
        if fixed + len(img) <= max(8, budget):
            info = (pal.magenta(f["gpu"])
                    + ((" " + util_cell(f, pad=False)) if f["util"] else ""))
            for x in mid:
                info += sep + pal.dim(x)
            info += sep + pal.blue(img)
        else:                       # pathological narrowness: single dim cut
            info = pal.dim(_ell(" · ".join([gput] + mid + [f["img"]]),
                                max(8, budget - 6)))
        rows = [head, f"      {info}"]
        # stopped box: the two resume rates ride a third muted line if they fit
        if not is_live:
            rp = resume_prices(f)
            if 6 + fmt._visw(rp) <= budget:
                rows.append(f"      {rp}")
        return rows

    def prog_seg(v: Mapping[str, Any],
                 keys: Sequence[str] = ("pct", "rate", "toks", "ckpt")) -> str:
        """Colored ` · `-joined training-progress bits for one job view;
        `keys` lets the compact layout shed the least-important bits first."""
        pg = view._job_progress(v)
        bits = []
        if "pct" in keys and "pct" in pg:
            bits.append(pal.bold(f"{pg['pct']}%")
                        + pal.dim(f" {pg['step']}/{pg['total']}"))
            if "rate" in keys and "rate" in pg:
                bits.append(pg["rate"])
        if "toks" in keys and "toks" in pg:
            bits.append(fmt._fmt_toks(pg["toks"]))
        if "ckpt" in keys and "ckpt" in pg:
            bits.append(pal.dim("ckpt ") + pal.bcyan(f"{pg['ckpt']}"))
        return pal.dim(" · ").join(bits)

    def job_rows(act: Sequence[Any], compact: bool) -> list[str]:
        out = []
        wj = max((len(v.get("job_id") or "?") for v in act), default=1)
        wn = max((len(v.get("name") or v.get("entrypoint") or "")
                  for v in act), default=0)
        # `cpu` chip: the fold's tri-state launch shape — ONLY an explicit
        # False earns the tag (a pre-stamp stream folds to None = unknown,
        # and unknown must never read as CPU). Column reserved fleet-wide so
        # mixed boxes keep the name column aligned.
        wc = 3 if any(v.get("gpu") is False for v in act) else 0
        for v in sorted(act, key=lambda v: v.get("job_id") or ""):
            stt = v.get("display_status") or "?"
            cf = job_color.get(stt, pal.dim)
            nm = v.get("name") or v.get("entrypoint") or ""
            jid = v.get("job_id") or "?"
            is_cpu = v.get("gpu") is False
            seg = prog_seg(v)
            if not compact:
                nmf = pal.bold(f"{nm:<{wn}}") if nm else " " * wn
                chip = ((pal.bcyan("cpu") if is_cpu else " " * wc) + " "
                        if wc else "")
                row = (f"      {pal.dim('└─')} {pal.dim(f'{jid:<{wj}}')}  "
                       f"{cf(f'{stt:<11}')} {chip}{nmf}")
                if seg:
                    row += "  " + seg
                out.append(row.rstrip())
                continue
            row = f"      {pal.dim('└─')} {cf(stt)}"
            if is_cpu:
                row += " " + pal.bcyan("cpu")
            row += "  "
            row += pal.bold(nm) if nm else ""
            row = row.rstrip()
            rem = budget - fmt._visw(row) - 2
            if rem >= 10:                 # id tail is the distinctive part
                row += "  " + pal.dim(_ell_left(jid, rem))
            out.append(row)
            # progress rides its own line, shedding bits until it fits
            for keys in (("pct", "rate", "toks", "ckpt"),
                         ("pct", "rate", "ckpt"), ("pct", "ckpt"), ("pct",)):
                s = prog_seg(v, keys)
                if s and 9 + fmt._visw(s) <= budget:
                    out.append(" " * 9 + s)
                    break
                if not s:
                    break
        return out

    def last_line(jobs: Sequence[Any], compact: bool) -> str | None:
        term = [v for v in jobs if v.get("status") in jobmeta.TERMINAL]
        if not term:
            return None
        last = max(term, key=lambda v: v.get("ended_at") or "")
        jid = last.get("job_id") or "?"
        if compact:
            jid = _ell_left(jid, max(12, budget - 24))
        return pal.dim(f"        last job: {jid} "
                       f"({last.get('display_status')})")

    def build(compact: bool) -> list[tuple[str, str, Callable[[str], str], list[str]]]:
        gs: list[tuple[str, str, Callable[[str], str], list[str]]] = []
        for name, desc, cf, entries, with_jobs in (
                ("ACTIVE", "live, jobs attached", pal.bgreen, active, True),
                ("RUNNING", "live, no jobs attached", pal.bcyan, running,
                 False),
                ("SUSPENDED", "stopped — still billing storage until "
                 "destroyed", pal.byellow, suspended, False)):
            if not entries:
                continue
            rows = []
            for i, jobs, act in entries:
                rows.extend(box_lines(i, compact))
                if with_jobs:
                    rows.extend(job_rows(act, compact))
                else:
                    ln = last_line(jobs, compact)
                    if ln:
                        rows.append(ln)
            gs.append((name, desc, cf, rows))
        return gs

    gs = build(False)
    table_w = max((fmt._visw(ln) for _, _, _, rows in gs for ln in rows),
                  default=0)
    if cols is not None and table_w > cols:
        gs = build(True)
        table_w = max((fmt._visw(ln) for _, _, _, rows in gs for ln in rows),
                      default=0)
    width = table_w if cols is None else min(max(table_w, 44), cols)

    # ── NAME · desc ───────── group headers, ruled to the table width
    first = True
    for name, desc, cf, rows in gs:
        if not first:
            lines.append("")
        first = False
        d = _ell(desc, max(4, width - len(name) - 9))   # 9 = frame + min rule
        text_w = len(name) + len(d) + 7        # "── NAME · desc "
        rule = "─" * max(2, width - text_w)
        lines.append(pal.dim("── ") + cf(name) + pal.dim(f" · {d} {rule}"))
        lines.extend(rows)

    sep = pal.dim(" · ")

    def emit_meta(parts: Sequence[str]) -> None:
        """Greedy-pack `·`-joined (possibly colored) parts into lines no wider
        than the budget, so the footer reflows instead of overflowing."""
        cur: list[str] = []
        curw = 0
        for p in parts:
            pw = fmt._visw(p)
            add = pw + (3 if cur else 0)
            if cur and curw + add > budget - 2:
                lines.append("  " + sep.join(cur)); cur, curw = [], 0  # noqa: E702 — verbatim body (plan §7.4)
                add = pw
            cur.append(p); curw += add   # noqa: E702 — verbatim body (plan §7.4)
        if cur:
            lines.append("  " + sep.join(cur))

    # fleet economics: what's actually leaving your wallet right now. This is
    # the WHOLE footer — the symbol legend lives in `ls --help`, not here.
    live_hr = 0.0
    for i, _, _ in active + running:
        _, billed, _, _ = price_bits(i)
        if billed is not None:
            live_hr += billed
    idle_day = sum(d for d in (models._storage_day(i) for i, _, _ in suspended)
                   if d is not None)
    n_live, n_idle = len(active) + len(running), len(suspended)
    lines.append("")
    econ = []
    if n_live:
        econ.append(pal.dim("live compute ") + pal.bgreen(fmt.dollars(live_hr))
                    + pal.dim(f"/hr ×{n_live}"))
    if n_idle:
        econ.append(pal.dim("idle storage ") + pal.byellow(fmt.dollars(idle_day))
                    + pal.dim(f"/day ×{n_idle}"))
    tot = data.get("fleet_spend_total")
    if isinstance(tot, (int, float)) and tot > 0:
        # fleetd's cumulative watched-spend counter — accrues from watch
        # ADOPTION, so it under-counts unwatched heads (fleet spend --reconcile
        # is the audit); still the one number that says what supervision saw.
        econ.append(pal.dim("fleetd watched spend Σ") + f"${tot:.2f}")
    if data.get("no_spot"):
        econ.append(pal.dim("spot rates stale (--no-spot)"))
    emit_meta(econ)
    if stale_ids:
        ids = " ".join(str(x) for x in stale_ids)
        lines.append(pal.yellow(
            f"  warn: {len(stale_ids)} box(es) run a STALE image — the "
            f"registry tag moved since launch (new env push). Rotate when "
            f"convenient: herdd destroy {ids} -y, then relaunch "
            f"(a resume keeps the OLD env)."))
    if unchecked_ids:
        # NOT a quieter STALE — the opposite kind of statement. These boxes were
        # not compared at all, and the usual cause is a missing credential, so
        # printing nothing would render an unset secret as a clean fleet.
        ids = " ".join(str(x) for x in unchecked_ids)
        hint = ("set REGISTRY_AUTH_SECRET (it lives in .env, which "
                "load_env does not read) and re-run"
                if any(imageref.r2_secret_missing(models._instance_image(i))
                       for i in ins if i.get("id") in set(unchecked_ids))
                else "registry lookup failed — retry, or check skopeo/network")
        lines.append(pal.yellow(
            f"  warn: {len(unchecked_ids)} box(es) could NOT be checked for a "
            f"stale image — the tag did not resolve, so these are UNKNOWN, not "
            f"fresh: {ids}. Fix: {hint}."))
    # Retained-for-salvage boxes (self-expiring `keep:...-until-<TS>` label —
    # the automatic eviction ladder holds the box it replaced). Read off the
    # LABEL, not from fleetd, so it still shows after the watch that created it
    # has ended. Nobody chose to rent these, so `ls` names them and their cost.
    ret = []
    for i in ins:
        info = labels._keep_retention_info(i.get("label"))
        if info:
            ret.append((i.get("id"), info, models._storage_day(i)))
    for iid, info, sd in ret:
        left = info["left_s"]
        cost = (fmt.dollars(max(0.0, left) / 86400.0 * sd) if sd else None)
        lines.append(pal.yellow(
            f"  note: {iid} is RETAINED ({info['reason']}) — "
            + (f"{fmt._age_str(left)} left"
               if left > 0 else "window CLOSED; `herdd reap` takes it next pass")
            + (f", ~{cost} more storage" if cost else "")
            + (f", {fmt.dollars(sd)}/day. " if sd else ". ")
            + f"Salvage it (`herdd start {iid}` then push to B2) or end it "
              f"now: herdd destroy {iid} -y"))
    # Oversized allocations: storage bills on the ALLOCATED disk, so a box using
    # a small fraction of it burns the difference for nothing. `ls` reported the
    # dollar cost but never the GB, which is exactly why the 2026-07-21 audit's
    # 8.9x oversizing was invisible. Threshold is deliberately loose (< 40%
    # used, and only above a floor allocation) so normal headroom never nags.
    over = []
    for i in ins:
        frac = models._disk_frac(i)
        alloc, used = models._disk_gb(i)
        if frac is not None and frac < 0.40 and (alloc or 0) >= 60:
            over.append((i.get("id"), alloc, used, frac, models._storage_day(i)))
    if over:
        waste_day = sum(
            (sd or 0) * (1 - f) for _, _, _, f, sd in over)
        worst = min(over, key=lambda t: t[3])
        lines.append(pal.yellow(
            f"  warn: {len(over)} box(es) OVERSIZED on disk — "
            f"~{fmt.dollars(waste_day)}/day pays for unused allocation "
            f"(worst: {worst[0]} uses {worst[2]:.0f} of {worst[1]:.0f} GB, "
            f"{worst[3] * 100:.0f}%). Size --disk to observed usage on the "
            f"next launch; a running box's allocation cannot be shrunk."))
    # Un-ssh-able boxes: read straight off each instance's stored onstart (the
    # list payload already carries it — zero extra network). Worth a footer
    # because the alternative discovery path is a bare `Permission denied
    # (publickey)` at the exact moment you need a shell, which reads like a
    # local key problem and is not (see ssh_access_warning).
    nossh = [i.get("id") for i in ins if models.instance_ssh_install(i) == "none"]
    if nossh:
        lines.append(pal.yellow(
            f"  warn: {len(nossh)} box(es) install NO ssh key in their onstart "
            f"({' '.join(str(x) for x in nossh)}) — expect "
            f"`Permission denied (publickey)`. The onstart is fixed at create "
            f"time, so a resume cannot repair it: reach them over B2 "
            f"(`job submit` / `runs`), or destroy + relaunch. "
            f"Why: herdd ssh <id>."))
    return lines


# `phase` (2026-08-02 boot-phase split): loading = image pull / vast-side
# standup (GPU unbilled) · env-setup = running, onstart/jobd bootstrap
# (BILLED full GPU) · up = workload evidence seen. Empty = N/A/unknowable
# (non-jobs running boxes have no workload contract the API can check).
#
# FROZEN COLUMN ORDER (plan §4). Hoisted out of `_render_minimal`'s body to a
# module constant so the contract has a name a test can hold: agent consumers
# and the dashboard read this TSV positionally, so inserting or renaming a
# column silently re-labels every value after it. Append only.
#
# NO `moved-from:` MARKER, deliberately (README §2 rule 7): in `herdd.py` this
# tuple was a LOCAL named `cols` inside `_render_minimal`, so there is no
# top-level `herdd.<name>` for the rename table to key on, and emitting one
# would put a phantom entry (`herdd._render_minimal.cols`) in a table whose
# whole job is to rewrite real attribute references. The value is byte-compared
# against the flat renderer's header line by
# `test_vastlib_cli_helpers.py::test_minimal_tsv_column_order_is_frozen`, which
# is the provenance link that actually binds.
_MINIMAL_COLS = ("state", "id", "status", "gpus", "gpu", "gpu_util", "mode",
                 "hourly", "storage_day", "disk_gb", "disk_used_gb", "idle",
                 "avail", "ondemand", "spot", "stale", "label", "jobs", "phase",
                 # `cpu_util` appended 2026-08-21: a dedicated CPU box reads
                 # gpu_util 0 and looked idle to every consumer of this table.
                 "cpu_util",
                 # appended 2026-08-27: `uptime` = age since the billing anchor
                 # (start_date, live boxes only); `spend_usd`/`budget_usd` =
                 # this box's fleetd watch economics (empty = unwatched, or no
                 # daemon); `cpu_jobs` yes = every ACTIVE job on the box
                 # declared needs.gpu false (pre-stamp streams fold to unknown
                 # and never earn the tag).
                 "uptime", "spend_usd", "budget_usd", "cpu_jobs")


def _filter_boxes(data: Mapping[str, Any], ids: Sequence[Any]) -> dict[str, Any]:
    """A shallow copy of one gather dict narrowed to the named instance ids.
    Filtering at RENDER time, never at gather: the snapshot cache stays
    full-fleet, so a filtered `ls` cannot poison the next unfiltered one."""
    want = {str(x) for x in ids}
    out = dict(data)
    out["instances"] = [i for i in (data.get("instances") or [])
                        if str(i.get("id")) in want]
    out["live_ids"] = [x for x in (data.get("live_ids") or [])
                       if str(x) in want]
    for k in ("jobs_by_box", "idle_secs", "health", "fleet_watch"):
        out[k] = {kk: v for kk, v in (data.get(k) or {}).items() if kk in want}
    for k in ("stale_ids", "unchecked_image_ids"):
        out[k] = [x for x in (data.get(k) or []) if str(x) in want]
    return out


# moved-from: herdd._render_minimal (row construction; the TSV spelling stays
# in _render_minimal so the two views share one code path and cannot drift)
def _minimal_rows(data: Mapping[str, Any]) -> list[dict[str, str]]:
    """The `--minimal` table as data: one dict per box keyed by `_MINIMAL_COLS`,
    every value the exact string the TSV prints (empty = N/A)."""
    ins = data.get("instances") or []
    live = set(data.get("live_ids") or [])
    market = data.get("market") or {}
    jobs_by_box = data.get("jobs_by_box") or {}
    idle_secs = data.get("idle_secs") or {}
    stale = set(data.get("stale_ids") or [])
    fleet_health = data.get("health") or {}
    fleet_watch = data.get("fleet_watch") or {}
    data_ts = data.get("ts") or time.time()

    def num(x: object) -> str:
        return f"{x:.4f}" if isinstance(x, (int, float)) else ""

    def gb(x: object) -> str:
        # GB to one decimal — disk sizes are whole-GB allocations and the used
        # figure arrives as a float with float32 noise (1.100000023841858).
        return f"{x:.1f}" if isinstance(x, (int, float)) else ""

    rows: list[dict[str, str]] = []

    # job cell = name:status, extended with :NN%:rate:ckptN when training
    # progress is parseable — see _job_cell (shared with dash-cache).
    job_cell = view._job_cell

    for i in ins:
        iid = i.get("id")
        is_live = iid in live
        jobs = jobs_by_box.get(str(iid), [])
        act = [v for v in jobs if v.get("display_status") in _ACTIVE_JOB_STATES]
        state = "active" if (is_live and act) else \
                "running" if is_live else "suspended"
        reserved, spot, avail = models._rates(i, market)
        is_bid = bool(i.get("is_bid"))
        mode = "spot" if is_bid else "on-dem"
        hourly = (spot if is_bid else reserved) if is_live else None
        sd = models._storage_day(i)
        d_alloc, d_used = models._disk_gb(i)
        util = i.get("gpu_util")
        cpu = i.get("cpu_util")
        jn = ",".join(sorted(job_cell(v) for v in act)) if act else ""
        start = _start_epoch(i)
        up_s = ((data_ts - start)
                if (is_live and start and data_ts > start) else None)
        wtch = fleet_watch.get(str(iid)) or {}
        rows.append(dict(zip(_MINIMAL_COLS, (
            # strict: a value tuple that drifts from the column tuple must
            # raise here, not silently re-label every later column.
            state,
            str(iid),
            str(i.get("actual_status") or i.get("cur_state") or "?"),
            str(i.get("num_gpus") or "?"),
            str(i.get("gpu_name") or "?"),
            (f"{util:.0f}" if is_live
             and isinstance(util, (int, float)) else ""),
            mode,
            num(hourly),
            num(sd) if not is_live else "",
            # allocated / actually-used GB. Reported for LIVE boxes too: the
            # allocation bills the same either way, and a live box is where an
            # oversized `--disk` can still be corrected on the next launch.
            gb(d_alloc),
            gb(d_used),
            fmt._age_str(idle_secs[str(iid)]) if str(iid) in idle_secs else "",
            ("" if avail is None else "yes" if avail else "no"),
            num(reserved),
            num(spot),
            "yes" if iid in stale else "",
            i.get("label") or "",
            jn,
            ((fleet_health.get(str(iid)) or {}).get("evidence") or {})
            .get("phase") or "",
            # APPENDED, never inserted — this TSV is read positionally.
            (f"{cpu:.2f}" if is_live
             and isinstance(cpu, (int, float)) else ""),
            fmt._age_str(up_s) if up_s is not None else "",
            num(wtch.get("spend_usd")),
            num(wtch.get("budget_usd")),
            ("yes" if act and all(v.get("gpu") is False for v in act) else ""),
        ), strict=True)))
    return rows


# moved-from: herdd._render_minimal
def _render_minimal(data: Mapping[str, Any]) -> str:
    """Token-efficient, color-free, stable TSV for agent consumers. One header
    line documenting the columns, then one row per box. Empty cell = N/A.
    Rates are bare floats ($/hr); storage is $/day; `disk_gb`/`disk_used_gb`
    are bare GB floats (allocated vs actually used — storage bills on the
    ALLOCATED number, so the pair is the oversizing signal); avail is
    yes/no/? — all trivially parseable, far smaller than `--json`'s raw
    vast dump."""
    lines = ["\t".join(_MINIMAL_COLS)]
    lines += ["\t".join(r[c] for c in _MINIMAL_COLS)
              for r in _minimal_rows(data)]
    return "\n".join(lines)
