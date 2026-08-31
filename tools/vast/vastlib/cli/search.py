"""`herdd search` — query the offer market, and the 16-flag filter block it shares.

Two things live here:

* `run` / `add_parser` — the `search` command itself: one `market.offers`
  query, printed as `fmt.fmt_offer` lines or raw JSON.
* **`add_search_filters(p)`** — the flag block applied to THREE parsers:
  `search`, `launch` and `supervise`. It is here rather than in `cli/_args.py`
  because it is a command's flag block, not composition plumbing (`_args.py`
  says the same in its own "NOT here" section).

`--job` (2026-08-28)
--------------------
Ranks the board by expected TRAINING tokens per dollar for one job instead of
by price. Two derivations, both REUSED rather than re-implemented: the shape
comes from `market.train_value` (which reads `train_rates`' measured anchors)
and the `--gpu-ram` floor from `jobmeta.vram_requirement`, the same call the
submit VRAM gate refuses on — a second copy of either would let this board
drift away from what submit accepts. `search` is the only parser that gets the
flag: `launch` and `supervise` PICK a box, and picking on a rate is a spending
decision this build does not make. Without `--job` every byte of the output is
what it was.

Why the block must stay one function
------------------------------------
`market.offers.build_search_query` reads these flags back off the Namespace **by
argparse DEST name** — `any_gpu`, `gpu_ram`, `max_dph`, `host_disk`,
`inet_down`, `exclude_machines`, … A per-parser copy of the block is a silent
divergence: `launch` would accept a filter `search` does not, or vice versa, and
nothing would fail until an offer query came back with the wrong shape. One
definition, three call sites, and the CLI-surface byte diff proves all three
render identically.

The `--inet-down` help interpolates `core.config._BOOT_KNOB_DEFAULTS`, i.e. the
VASTLIB copy of the knob table, per cli-surface.json hazard H4: the flat file
still holds its own copy during the add-only wave, and importing the wrong one
renders the same today and drifts silently later.

What is deliberately NOT here
-----------------------------
* The GPU alias/normalization tables, the preferred-GPU policy tiers and the
  inet floor logic. Those are `market.offers`; this module only declares the
  flags that steer them.
* `--offer` / `--offer-machine` / pricing flags. Those belong to `launch`,
  which is the only command that can act on a pinned offer.

Provenance: moved from `tools/vast/herdd.py` (`cmd_search`,
`add_search_filters`, parser block in `main()`), plan §8 step 6, 2026-08-16,
behavior-preserving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any, NamedTuple

from vastlib.cli import _args, _docs
from vastlib.core import config, fmt, models
from vastlib.market import offers as market_offers
from vastlib.market import train_value

if TYPE_CHECKING:                         # Zone S flat leaf; see `_job_context`
    from train_rates import Family


# moved-from: herdd.cmd_search
def run(a: argparse.Namespace) -> None:
    # `--job` first: it derives the VRAM floor the query itself filters on, so
    # it has to run before the offers are fetched. Without it every line below
    # is exactly what it was — this lane's default order is price-ascending and
    # every other lane depends on that.
    job = _job_context(a)
    offers = market_offers.search_offers(a)
    # Asking about CPU shape re-sorts by capability per dollar. Only then: the
    # default order is price-ascending and every other lane depends on it.
    # Offers with no score sort LAST rather than as zero — a missing term is an
    # unknown box, not a bad one.
    want_cpu = bool(getattr(a, "cpu_cores", 0) or getattr(a, "cpu_ghz", 0))
    dropped: list[tuple[dict[str, Any], str]] = []
    if want_cpu:
        # The floor refuses silicon we have MEASURED to be pathological. It
        # cannot refuse an offer we have never measured (see `cpu_floor_verdict`
        # — that is most of the market), and it is armed only here, on a
        # CPU-shaped ask, never on an ordinary GPU launch.
        ratio = _floor_ratio(a)
        if ratio > 0:
            kept: list[dict[str, Any]] = []
            for o in offers:
                ok, why = market_offers.cpu_floor_verdict(o, ratio=ratio)
                if ok:
                    kept.append(o)
                else:
                    dropped.append((o, why))
            offers = kept
        # Measured capability per dollar, with `cpu_score` as the fallback for
        # rows nothing has measured. The two are different units, so they cannot
        # share a sort key: measured offers rank ABOVE unmeasured ones outright,
        # and each group is ordered within itself. Unmeasured sorts last rather
        # than as zero — a missing term is an unknown box, not a bad one.
        def _rank(o: dict[str, Any]) -> tuple[int, float]:
            v = market_offers.cpu_value(o)
            if v is not None:
                return (2, v)
            s = market_offers.cpu_score(o)
            dph = models._num_dph(o.get("dph_total")) or 0.0
            return (1, s / dph) if (s and dph > 0) else (0, 0.0)
        offers = sorted(offers, key=_rank, reverse=True)
    # The job ask is the LAST word on order: it is the whole point of --job, and
    # a CPU-shaped ask alongside it is a floor on the box, not the question.
    rows: list[train_value.OfferValue] = []
    if job is not None:
        rows = train_value.rank_offers(offers, job.env, job.assets)
        offers = [v.offer for v in rows]
    if a.json:
        print(json.dumps(offers, indent=2)); return  # noqa: E702 — verbatim body (plan §7.4)
    if job is not None:
        for line in _job_header(job):
            print(line)
    print(f"{len(offers)} offers ({a.type}"
          f"{', by measured CPU work/$' if want_cpu else ''}"
          f"{', by expected tokens/$' if job is not None else ''}):")
    want_ram = bool(getattr(a, "host_ram", 0))
    explain = bool(getattr(a, "explain", False))
    for i, o in enumerate(offers):
        line = "  " + fmt.fmt_offer(o)
        if job is not None:
            line += "  " + _tpd_bracket(rows[i], explain)
        if want_cpu:
            line += "  " + _cpu_bracket(o)
        if want_ram:
            # The SLICE, not the host figure `--host-ram` filtered on. Printing
            # cpu_ram raw is how a "768 GB" box gets picked for a job that then
            # gets 96 and OOMs.
            got = market_offers.effective_host_ram_gb(o)
            line += (f"  [ram {got:g}G slice]" if got is not None
                     else "  [ram ?G — offer carries no cpu_ram/gpu_frac]")
        print(line)
    if want_ram:
        print("  note: --host-ram filters the HOST's cpu_ram; the bracket is "
              "the SLICE (cpu_ram x gpu_frac), which is what you actually get.")
    if want_cpu:
        _cpu_notes(dropped, _floor_ratio(a))
    if job is not None:
        _job_notes(job, rows)


def _floor_ratio(a: argparse.Namespace) -> float:
    """The armed floor ratio. `--any-cpu` disarms; `--min-cpu-perf` overrides."""
    if getattr(a, "any_cpu", False):
        return 0.0
    override = getattr(a, "min_cpu_perf", None)
    if override is not None:
        return max(0.0, float(override))
    return market_offers.CPU_PERF_FLOOR_RATIO


def _cpu_bracket(o: dict[str, Any]) -> str:
    """One offer's CPU line. Says WHICH tier the number came from, always — a
    measured rate and a cores-times-GHz prior must never look alike.

    Both arms print when both are known: they are separate axes and a cheap
    wide box can rank well on throughput while being the slowest thing on the
    board for ONE compile. Collapsing that to a single number is the thing the
    two-arm table exists to stop.
    """
    cores = models._num_dph(o.get("cpu_cores_effective"))
    p = market_offers.cpu_perf(o)
    if p:
        thr = market_offers.cpu_throughput(o)
        val = market_offers.cpu_value(o)
        tier = p["tier"]
        if tier == "model":
            tier += f":n={p['n']}" + (f",spread={p['spread']:g}x"
                                      if p.get("spread") else "")
        lat = market_offers.cpu_perf(o, arm=market_offers.FLOOR_ARM)
        return (f"[cpu {cores:g}t {p['rate']:.3g}/t {tier}"
                + (f" work/$={val:.3g}" if val else "")
                + (f" thr={thr:.3g}" if thr else "")
                + (f" compile={lat['rate']:.3g}/s" if lat else "") + "]")
    s = market_offers.cpu_score(o)
    if s:
        ghz = models._num_dph(o.get("cpu_ghz"))
        return f"[cpu {cores:g}t@{ghz:g}GHz UNMEASURED score={s:g}]"
    return "[cpu ? — offer carries no cores/GHz]"


def _cpu_notes(dropped: list[tuple[dict[str, Any], str]], ratio: float) -> None:
    table = market_offers.cpu_calibration()
    if table:
        print(f"  note: measured rows are ranked on {table['units']} per thread "
              f"per $/hr, from {table['n_machines']} machines / "
              f"{table['n_models']} CPU models measured "
              f"({table['generated']}). UNMEASURED rows fall back to "
              f"GHz*cores, which is BLIND TO IPC and over-rates old silicon — "
              f"they sort below every measured row rather than as zero.")
    else:
        print("  note: no calibration table — every row is GHz*cores, which is "
              "BLIND TO IPC. Run `hostfacts.py calibrate --write`.")
    if not ratio:
        print("  note: CPU floor DISARMED — measured-slow offers are included.")
        return
    floor = market_offers.cpu_perf_floor(ratio=ratio)
    if floor is None:
        return
    # Name the ARM. The floor and the ranking read different ones on purpose,
    # so an unlabelled rate here would invite comparing it to the `/t` figure
    # in each row -- different units, different question.
    arm = market_offers.FLOOR_ARM
    print(f"  note: floor {ratio:g}x {arm} fleet median ({floor:.3g}/s, a "
          f"SERIAL single-compile rate — not the per-thread figure rows show) "
          f"dropped {len(dropped)} MEASURED-slow offer(s); unmeasured offers "
          f"are never dropped. --any-cpu disarms, --min-cpu-perf RATIO retunes.")
    for o, why in dropped[:5]:
        print(f"    - {o.get('machine_id')} {str(o.get('cpu_name') or '?')[:40]}"
              f": {why}")


class _JobCtx(NamedTuple):
    """What `--job` resolved, kept together so the header can PRINT it. An agent
    that cannot see which family and which floor were derived cannot tell a
    ranking from a coincidence."""

    name: str
    env: dict[str, Any]
    assets: object
    family: Family | None
    world_size: int
    need: dict[str, Any] | None
    floor_gb: float | None  # the derived floor, whether or not it was applied
    floor_applied: bool


def _job_context(a: argparse.Namespace) -> _JobCtx | None:
    """Read the job's shape off its bundle and arm the VRAM filter, or None.

    The floor comes from `jobmeta.vram_requirement` — the SAME derivation the
    submit gate refuses on — so the board this prints and the board submit will
    accept cannot drift. An explicit `--gpu-ram` wins: the operator asking a
    different question is not a bug to correct.
    """
    path = getattr(a, "job", None)
    if not path:
        return None
    # Zone S leaf, imported here rather than at module scope: `--job` is the
    # only path that needs it, and `add_search_filters` is shared with two other
    # parsers that must stay importable without the tools/vast bootstrap.
    import jobmeta  # noqa: PLC0415

    bundle = path if os.path.isdir(path) else (os.path.dirname(path) or ".")
    try:
        cfg = jobmeta.load_job_config(bundle)
    except Exception as e:                                     # noqa: BLE001
        sys.exit(f"--job {path}: {e}")
    env = dict(cfg.get("env") or {})
    assets = cfg.get("assets")
    gpus = (cfg.get("needs") or {}).get("gpus")
    ws = gpus if isinstance(gpus, int) and gpus >= 1 else 1
    need = jobmeta.vram_requirement(cfg)
    floor = None
    if (need or {}).get("status") == "ok":
        floor = models._num_dph((need or {}).get("required_gb"))
    applied = bool(floor and not getattr(a, "gpu_ram", 0))
    if applied and floor:
        a.gpu_ram = float(floor)
    return _JobCtx(str(cfg.get("name") or os.path.basename(os.path.abspath(bundle))),
                   env, assets,
                   train_value.family_for(env, assets, world_size=ws),
                   ws, need, floor, applied)


def _job_header(job: _JobCtx) -> list[str]:
    """What was RESOLVED, before any row. Two things an operator has to be able
    to check by eye: which training family the env mapped to (a mis-mapped one
    silently ranks a job nobody ran) and which VRAM floor is filtering."""
    fam = job.family
    slug = fam.slug() if fam is not None else None
    head = [f"job {job.name}: "
            + (f"[{slug}] w={job.world_size}" if slug else
               "env maps to NO training family (eval/probe/generation shape?) — "
               "rows are unmeasured and the order is price")]
    st = (job.need or {}).get("status")
    if st == "ok":
        n = (job.need or {}).get("n")
        head.append(
            f"  vram: >= {job.floor_gb:.4g} GB/card (measured peak "
            f"{(job.need or {}).get('gb')} + {(job.need or {}).get('headroom_gb')} "
            f"headroom, n={n}, card class {(job.need or {}).get('card_class')})"
            + (" — applied as the --gpu-ram filter" if job.floor_applied else
               "  NOT applied: your --gpu-ram wins"))
    elif st == "unmeasured":
        head.append(f"  vram: UNMEASURED for this shape, no floor applied "
                    f"({(job.need or {}).get('detail')})")
    elif st == "skipped":
        head.append(f"  vram: sizing skipped, no floor applied "
                    f"({(job.need or {}).get('detail')})")
    else:
        head.append("  vram: not a sized training shape — no floor applied")
    return head


def _tpd_bracket(v: train_value.OfferValue, explain: bool = False) -> str:
    """One offer's rate verdict. Always names the SOURCE tier: a measured rate
    and a stale-stack floor must never look alike on the page, and `-` must
    never look like a slow box."""
    if v.est is None:
        return "[tok/s ?  tok/$ ?  UNMEASURED]"
    body = (f"[tok/s {v.est.tok_s:.0f}  "
            f"tok/$ {train_value.human_tokens(v.tok_per_dollar)}  {v.src}")
    if explain:
        body += (f" n={v.est.n} spread={v.est.spread:.2f}x "
                 f"op={v.est.op_point}: {v.est.why}")
    return body + "]"


def _job_notes(job: _JobCtx, rows: list[train_value.OfferValue]) -> None:
    if job.family is None:
        print("  note: no training family for this bundle's env, so no rate was "
              "looked up. Order is the market's (price); nothing was dropped.")
        return
    print("  note: rows are ranked on MEASURED tokens per dollar for THIS job's "
          "shape at each offer's own card count — tier first (meas > prov, a "
          "prov rate is a stale-stack FLOOR), then tok/$ = tok/s x 3600 / dph. "
          "Steady-state only: boot, base pull and cold compile are excluded, so "
          "it ranks, it does not budget. A card class we have never measured on "
          "this shape sorts LAST by price, never as zero, and is never dropped.")
    unmeasured = train_value.unmeasured_cells(rows)
    if not unmeasured:
        return
    print(f"  note: {len(unmeasured)} card cell(s) unmeasured for this shape. "
          f"Every real run is a benchmark here — running the job banks the "
          f"anchor:")
    for name, n in unmeasured[:5]:
        hint = train_value.probe_hint(job.family, name)
        if hint:
            # The card COUNT is part of the cell (eff_batch carries world_size),
            # so a 1x row and a 4x row of the same card are two probes.
            print(f"    - {f'{n}x ' if n > 1 else ''}{hint}")


# moved-from: herdd.add_search_filters
def add_search_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", action="append", help="GPU name/alias (repeatable): 4090, a100, h100, '2080 Ti'")  # noqa: E501 — verbatim help text (plan §7.4)
    p.add_argument("--any-gpu", dest="any_gpu", action="store_true",
                   help="disable the default preferred-GPU policy (with no --gpu, "
                        "auto-pick is limited to bf16-capable cards, >=32 GB "
                        "first) and consider ANY card matching the filters — may "
                        "land on pre-Ampere silicon with no bf16")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--gpu-ram", type=float, default=0,
                   help="min GPU RAM in GB, as the card is MARKETED — `48` "
                        "matches a 48 GB card even though it advertises "
                        "49140 MiB (see gpu_ram_floor_mib)")
    p.add_argument("--max-dph", type=float, default=None, help="max total $/hr")
    p.add_argument("--host-disk", type=int, default=0, help="min disk GB available on host")
    # The two floors below come from `core.config` rather than a literal: the
    # /admin market snapshot labels offers launch-reachable with the same
    # numbers, and two copies drift into a printed floor nothing can rent.
    p.add_argument("--reliability", type=float,
                   default=config.LAUNCH_RELIABILITY_MIN)
    p.add_argument("--cuda", type=float, default=config.LAUNCH_CUDA_MAX_GOOD,
                   help="min cuda_max_good (default 12.8 — the CUDA-12 driver "
                        "floor the cu129 image is rented at, config."
                        "LAUNCH_CUDA_MAX_GOOD; 0 disables, raise it explicitly "
                        "for a lane on a cu13x image). Server-enforced on a "
                        "SEARCH; best-effort under --offer (the offer `id` filter "
                        "resolves nothing, so a missed pin warns and defers to the "
                        "on-box probe)")
    p.add_argument("--inet-down", type=float, default=None,
                   help="min advertised download Mbps (default: the "
                        "LAUNCH_INET_DOWN_MBPS knob, "
                        f"{int(config._BOOT_KNOB_DEFAULTS['LAUNCH_INET_DOWN_MBPS'])} — "
                        "slow image pulls on slow-NIC hosts dominate slow "
                        "boots, owner directive 2026-08-03; 0 disables)")
    p.add_argument("--any-inet", dest="any_inet", action="store_true",
                   help="disable the default inet-down floor on auto-pick "
                        "(escape hatch — may land on a host that pulls the "
                        "image for 30+ min; the boot SLA will replace it)")
    p.add_argument("--exclude-machine", dest="exclude_machines", action="append",
                   type=int, default=None, metavar="ID",
                   help="never pick these machine_id(s) (repeatable; the "
                        "boot-SLA/pull-watchdog relaunch lanes use this for "
                        "host rotation)")
    p.add_argument("--machine", action="append", type=int, default=None, metavar="ID",
                    help="restrict to vast machine_id(s) (repeatable)")
    p.add_argument("--host", action="append", type=int, default=None, metavar="ID",
                    help="restrict to vast host_id(s) (repeatable)")
    p.add_argument("--geo", action="append", default=None, metavar="CC",
                    help="restrict to 2-letter country code(s), e.g. US (repeatable). "
                         "Pin near the B2 bucket (us-west-004) to speed the base-model pull.")
    # CPU shape. Both are LOOSE floors whose job is to bound the fetch, not to
    # pick a box — ranking happens client-side. That ranking is now MEASURED
    # (`offers.cpu_perf`, from `cpu_calibration.json`), and the distribution a
    # threshold was waiting on exists: 53 machines, 41 CPU models, fleet spread
    # 7.07x. Hence the floor below, armed by owner decision 2026-08-27.
    p.add_argument("--cpu-cores", dest="cpu_cores", type=float, default=0,
                   help="min CPU cores (cpu_cores_effective — the SLICE, not "
                        "the host's advertised cpu_cores). For CPU-shaped work "
                        "(compile/search) where a worker count is the real "
                        "requirement")
    p.add_argument("--cpu-ghz", dest="cpu_ghz", type=float, default=0,
                   help="min advertised CPU clock in GHz. A floor only: many "
                        "slow cores beat few fast ones for parallel compiles, "
                        "so use it to exclude, not to prefer")
    p.add_argument("--host-ram", dest="host_ram", type=float, default=0,
                   help="min HOST RAM in GB. Filters on cpu_ram, which is the "
                        "WHOLE MACHINE's memory — the slice you rent is "
                        "cpu_ram x gpu_frac, so this bounds the search rather "
                        "than guaranteeing the share (the auto-pick lanes "
                        "check the slice too). The axis CPU-shaped work is "
                        "really sized by: a bf16 CPU merge holds the whole "
                        "base resident")
    p.add_argument("--type", choices=["ondemand", "bid"], default="ondemand")
    p.add_argument("--unverified", action="store_true", help="include unverified hosts")
    p.add_argument("--limit", type=int, default=20)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser],
               add_cmd: _args.AddCmd) -> argparse.ArgumentParser:
    ps = add_cmd(sub, "search", "search GPU offers",
                 _docs.DOC_README, _docs.DOC_TRAINING)
    add_search_filters(ps)
    # Deliberately NOT in `add_search_filters`: that block is shared with
    # `launch` and `supervise`, and only this lane reads these. A flag on a
    # parser whose command ignores it is worse than no flag — wiring the floor
    # into the lanes that SPEND is a separate decision (`pick_offers` takes it
    # as an opt-in argument for exactly that reason), not a side effect of
    # sharing a parser block.
    ps.add_argument("--min-cpu-perf", dest="min_cpu_perf", type=float,
                    default=None, metavar="RATIO",
                    help="retune the measured-CPU floor, as a fraction of the "
                         f"fleet median {market_offers.FLOOR_ARM} rate — a "
                         "SERIAL single-compile rate, not the per-thread figure "
                         "rows are ranked on (default "
                         f"{market_offers.CPU_PERF_FLOOR_RATIO:g}). Only offers "
                         "we have MEASURED can be refused — an unmeasured one "
                         "cannot be shown to be slow, so it is ranked last, "
                         "never dropped. 0 disables")
    ps.add_argument("--any-cpu", dest="any_cpu", action="store_true",
                    help="disarm the measured-CPU floor entirely (escape "
                         "hatch: keeps offers we have measured to be "
                         "pathologically slow, e.g. a cheap-and-wide box whose "
                         "throughput per dollar still wins)")
    # Also deliberately NOT in the shared block: `launch` and `supervise` PICK a
    # box, and picking on a rate is a spending decision this build does not
    # make. Here it only orders a page the market already returned.
    ps.add_argument("--job", metavar="PATH", default=None,
                    help="rank offers by expected TRAINING tokens per dollar "
                         "for this job — a bundle dir or its job-config.yaml. "
                         "Derives the job's shape from `env:`/`assets:` and the "
                         "--gpu-ram floor from the same measured anchors the "
                         "submit VRAM gate refuses on (an explicit --gpu-ram "
                         "wins). Card classes with no anchor at this shape are "
                         "shown last, never dropped and never scored zero")
    ps.add_argument("--explain", action="store_true",
                    help="with --job: show how each rate was derived (anchor "
                         "count, spread, operating point)")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=run)
    return ps
