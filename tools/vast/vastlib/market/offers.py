"""vastlib.market.offers — which box to rent: GPU names, policy, and the search.

Why this exists
---------------
Everything between "a job says it needs a 48 GB bf16 card" and "here is the
offer id to launch" lives here, and it lives in ONE module because every defect
this ring has ever had was two copies of the same rule drifting apart:

* the vast `gpu_name` filter is EXACT MATCH, so a spelling the alias table does
  not know is indistinguishable from an empty market (`GPU_ALIASES`,
  `normalize_gpu`);
* a declared VRAM need is a marketing GB number and vast advertises the usable
  framebuffer, so the GB->MiB floor carries a measured tolerance band — and
  when `build_search_query` and `pick_offers` each carried their own, the whole
  48 GB card class became unrentable (`gpu_ram_floor_mib`, `VRAM_SEARCH_TOLERANCE`);
* an unnamed GPU must never be taken as "cheapest offer that fits", or a
  pre-Ampere card with no bf16 gets rented for a bf16 training job
  (`GPU_DEFAULT_POLICY_TIERS`, `_gpu_policy_tiers`);
* an auto-picked host needs a download floor or the image pull eats the boot
  SLA (`_inet_floor`).

The hard/soft split is a CONTRACT, not a style. `search_offers` goes through
`api.request`, which `sys.exit`s on an API error — right for a one-shot CLI.
`_search_offers_soft`, `pick_offers` and the `_offer_*_soft` rungs must never
exit: they run inside supervise loops where an API blip is not a verdict.
Neither may be "simplified" into the other.

Two query builders, on purpose
------------------------------
`build_search_query` (argparse-driven, the CLI lane) and `pick_offers`
(kwargs-driven, the automatic lanes) build near-identical bundles bodies by
hand. They are NOT unified here. They have drifted before — that is the 0.96
vs exact VRAM factor above — and the repair was one shared floor helper plus
`test_gpu_ram_floor.py`'s parity assertion, not one builder. Unifying them is a
separate, testable change; a verbatim port is not the place for it.

What is deliberately NOT here
-----------------------------
* **No pricing.** On-demand references, min-bid reads, the C17 chunk floors and
  the `bidpolicy` bid arithmetic are `market.pricing`. This module answers
  "which offers exist"; that one answers "what would they cost".
* **No launching.** Picking an offer is not creating an instance
  (`launch/`), and no bid is ever PLACED from `market/` (`boxes.lifecycle`).
* **No third query builder.** `hosts.py` has its own (`type: "ask"`, GET with a
  quoted `q`, no inet floor, no policy tiers). It is an absorbed sibling with
  its own port step; it is NOT unified here, and it still calls
  `herdd.normalize_gpu` at that path until it moves.
* **No `fmt_offer`.** Offer rendering is `core.fmt` — it reads eleven offer
  fields but it is presentation, and `market/` must stay importable by anything
  that never prints.
* **No `_gpu_rate_soft`.** Tokens-per-dollar ranking travels with the eviction
  walkers into `supervise/replacement.py`; `_replacement_candidate_class` here
  answers only "which cards MAY replace this one", never "which is best value".

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 3 (`market/`) of `docs/plans/vast-tooling-refactor-v2.md`. Every symbol
carries its `# moved-from:` marker. Step 3 is ADD-ONLY, so `herdd.py` keeps
its own copies until step 6 and both are live meanwhile.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vastlib.core import api, config, models
from vastlib.market import hostrep as hostrep_mod

# The value type is `str | list[str]` and the annotation is the only thing
# added: mypy would otherwise join the two branches to `object` and lose the
# `isinstance(v, list)` narrowing `normalize_gpu` depends on.
# moved-from: herdd.GPU_ALIASES
GPU_ALIASES: dict[str, str | list[str]] = {
    "3090": "RTX 3090", "rtx3090": "RTX 3090",
    "4090": "RTX 4090", "rtx4090": "RTX 4090",
    "5090": "RTX 5090", "rtx5090": "RTX 5090",
    "a100": ["A100 SXM4", "A100 PCIE", "A100X"],
    "a100-80": ["A100 SXM4", "A100 PCIE"],
    "h100": ["H100 SXM", "H100 PCIE", "H100 NVL"],
    "h200": ["H200", "H200 NVL"],
    "b200": "B200", "b300": "B300",
    # RTX PRO 6000 Blackwell (96 GB) — vast lists both a workstation and a
    # "server" SKU; match either.
    "rtxpro6000": ["RTX PRO 6000 WS", "RTX PRO 6000 S"],
    "pro6000": ["RTX PRO 6000 WS", "RTX PRO 6000 S"],
    "6000blackwell": ["RTX PRO 6000 WS", "RTX PRO 6000 S"],
    "l40": ["L40", "L40S"], "l40s": "L40S",
    "a6000": "RTX A6000", "rtxa6000": "RTX A6000",
    "a5000": "RTX A5000", "rtxa5000": "RTX A5000",
    "a4000": "RTX A4000", "rtxa4000": "RTX A4000",
    "a40": "A40",
    # Ada workstation line. vast spells these with NO SPACE before "Ada"
    # ("RTX 6000Ada"), which nobody writes by hand and which the exact-match
    # gpu_name filter will not forgive — see normalize_gpu.
    "6000ada": "RTX 6000Ada", "rtx6000ada": "RTX 6000Ada",
    "5880ada": "RTX 5880Ada", "rtx5880ada": "RTX 5880Ada",
    "5000ada": "RTX 5000Ada", "rtx5000ada": "RTX 5000Ada",
    "4500ada": "RTX 4500Ada", "rtx4500ada": "RTX 4500Ada",
    "4000ada": "RTX 4000Ada", "rtx4000ada": "RTX 4000Ada",
    "4080": "RTX 4080", "3080": "RTX 3080",
}


# moved-from: herdd.normalize_gpu
def normalize_gpu(names: Iterable[str]) -> list[str]:
    """Human GPU spellings -> the exact `gpu_name` strings vast indexes on.

    The API's gpu_name filter is EXACT MATCH — case- AND whitespace-sensitive,
    with no fuzzy fallback — and an unmatched name is not an error, it is an
    empty result. Verified live 2026-08-16: `RTX 6000Ada` returns offers,
    `RTX 6000 Ada` / `rtx6000ada` / `6000ada` all return ZERO, and so does
    `l40s` (only `L40S` works). So a spelling this table does not know looks
    exactly like an empty market, and a launcher asking for a card that is
    plentifully in supply just never rents one.

    Lookup is therefore two-pass: the lowercased name, then the lowercased name
    with all whitespace removed. The second pass is what lets one alias entry
    cover `RTX 6000 Ada`, `rtx 6000ada` and `RTX6000Ada` at once. Hyphens are
    NOT stripped — `a100-80` is a distinct key from `a100`. A name in no alias
    family is passed through unchanged, so an un-aliased SKU still works when
    spelled the way vast spells it."""
    out = []
    for n in names:
        low = n.lower()
        v = GPU_ALIASES.get(low)
        if v is None:
            v = GPU_ALIASES.get("".join(low.split()), n)
        out.extend(v if isinstance(v, list) else [v])
    # de-dup preserving order
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)   # noqa: E702 — verbatim body (plan §7.4)
    return res


# moved-from: herdd.gpu_family_names
def gpu_family_names(name: str | None) -> list[str]:
    """Every vast `gpu_name` in the SAME alias family as `name` — the inverse
    index over GPU_ALIASES, e.g. "H100 NVL" -> ["H100 SXM", "H100 PCIE",
    "H100 NVL"]. `[name]` when the name is in no family (unknown SKU, or one we
    never aliased); `[]` for an empty name.

    Added 2026-08-16. Automatic re-rent lanes used to pin a replacement to the
    primary's EXACT vast gpu_name string, which is not a requirement — it is a
    typo-level accident of which SKU the host happened to list. "H100 NVL" and
    "H100 SXM" are the same 80 GB card for every purpose this fleet has, and
    pinning the string is how a one-offer candidate set gets built."""
    if not name:
        return []
    want = str(name).strip()
    fam = []
    for v in GPU_ALIASES.values():
        names = v if isinstance(v, list) else [v]
        if want in names:
            for n in names:
                if n not in fam:
                    fam.append(n)
    return fam or [want]


# Default preferred-GPU policy (owner directive 2026-08-03, WIDENED 2026-08-07)
# — THE one place the policy lives; update these tiers when the market shifts.
#
# What this exists to prevent, and the ONLY thing it exists to prevent: when the
# caller does NOT name a GPU (no --gpu, no --machine/--host/--offer pin),
# "cheapest offer that fits the filters" must not be taken literally. On
# 2026-08-03 it picked a Quadro RTX 8000 — Turing sm_75, NO bf16 — for a bf16
# training job, and the box had to be destroyed before it wasted the run. The
# cuda_max_good floor can't catch this (it measures the host DRIVER, not the
# silicon: a Turing card behind a new driver reports 13.0). So the allowlist
# draws exactly one line: **bf16-capable silicon, i.e. Ampere or newer**.
# Pre-Ampere (Turing/Volta/Pascal) is excluded by construction — a card not
# listed here can never be auto-picked.
#
# It is NOT an architecture preference. The tiers were Blackwell-only from
# 2026-08-03 to 2026-08-07, which made an H100 or A100 unreachable without
# --any-gpu even when it was the best value on the board; owner ruling
# 2026-08-07 removed that restriction ("a100, h100, etc all work and can be
# exceptional value") and the off-policy annotation it fed on the admin market
# page went with it. Anything on this list is a card we will happily rent; the
# choice between them is PRICE, which is what the tier search already does.
#
# The auto-pick searches these tiers in order and takes the cheapest offer from
# the first tier with any match — so within tier 0 an exceptionally cheap H200
# beats a dearer RTX 5090 on merit. The tier split is VRAM, not vintage: tier 0
# is >=32 GB (a 7B bf16 training footprint fits), tier 1 is the smaller/older
# tail that should only win when tier 0 is dry. Escape hatches unchanged: an
# explicit --gpu <name>, or --any-gpu for unrestricted cheapest-overall.
# moved-from: herdd.GPU_DEFAULT_POLICY_TIERS
GPU_DEFAULT_POLICY_TIERS: tuple[tuple[str, ...], ...] = (
    # tier 0 — >=32 GB and bf16: every card here is one we run real work on,
    # ranked against each other on price alone
    # Supply verified against the live bundles API 2026-08-07 (bid/on-demand
    # counts, unfiltered). Present: L40S 30/35, RTX 6000Ada 28/30, RTX A6000
    # 11/14, RTX 5880Ada 7/7, L40 5/5, A800 PCIE 5/5. Returned ZERO offers in
    # both modes: A40, A100X — kept anyway because they are real SKUs and an
    # allowlist entry with no supply costs nothing, but do not expect them.
    ("RTX 5090", "RTX PRO 6000 WS", "RTX PRO 6000 S", "RTX PRO 5000",
     "RTX PRO 4500", "B200", "B300",
     "H200", "H200 NVL", "H100 SXM", "H100 PCIE", "H100 NVL",
     "A100 SXM4", "A100 PCIE", "A100X", "A800 PCIE",
     "L40S", "L40", "A40", "RTX A6000", "RTX 6000Ada", "RTX 5880Ada"),
    # tier 1 — bf16-capable but <32 GB or an older generation (fallback when
    # tier 0 is dry; entries are still subject to any --gpu-ram floor)
    ("RTX 5080", "RTX 5070 Ti", "RTX 5070", "RTX PRO 4000",
     "RTX 4090", "RTX 4080", "RTX 3090", "RTX A5000"),
)


# moved-from: herdd._gpu_policy_tiers
def _gpu_policy_tiers(a: argparse.Namespace) -> tuple[tuple[str, ...], ...] | None:
    """The gpu_name allowlist tiers the default policy imposes on this search,
    or None when the policy is bypassed: an explicit --gpu, a --machine/--host
    pin (operator chose the hardware), or the --any-gpu escape hatch.
    (`exclude_machines` host-rotation does NOT bypass — it is not a pin.)"""
    if getattr(a, "gpu", None) or getattr(a, "any_gpu", False):
        return None
    if getattr(a, "machine", None) or getattr(a, "host", None):
        return None
    return GPU_DEFAULT_POLICY_TIERS


# moved-from: herdd._inet_floor
def _inet_floor(explicit: object, *, pinned: bool = False,
                any_inet: bool = False) -> float | None:
    """Effective advertised-download floor (Mb/s) for an offer search, or None.

    Owner directive 2026-08-03 (the 39-minute serve image pull on an 805 Mb/s
    host, box 46682177, while a 26689 Mb/s host pulled the same image in
    minutes): AUTO-PICKED offers get a default `inet_down >=
    LAUNCH_INET_DOWN_MBPS` (1000; env/herdd.yaml-tunable — relaxed from
    2000 on 2026-08-03: the rating is a weak predictor in both directions per
    the netprobe experiment, and 2000 excluded too much supply). Slow boots are
    dominated by the image pull and advertised inet_down is the best pick-time
    predictor available. Precedence:

      * an explicit value (CLI --inet-down / a caller-passed number) is used
        verbatim — 0 disables the filter entirely;
      * `pinned` (--machine/--host/--offer — the operator chose the hardware)
        or `any_inet` (the escape hatch) suppress the DEFAULT floor;
      * otherwise the knob applies.

    inet_down is a whole-machine Ookla snapshot and hosts shape per-TCP-flow
    (memory: vast-per-flow-image-layering), so this floor PREVENTS most slow
    pulls but guarantees nothing — the boot SLA (BOOT_SLA_S, enforced by the
    owning lifecycle) is the backstop for the hosts that slip through."""
    if explicit is not None:
        try:
            return float(explicit) or None       # type: ignore[arg-type]  # 0 = explicitly no floor
        except (TypeError, ValueError):
            return None
    if pinned or any_inet:
        return None
    return config._boot_knob("LAUNCH_INET_DOWN_MBPS") or None


# moved-from: herdd._inet_floor_for
def _inet_floor_for(a: argparse.Namespace) -> float | None:
    """`_inet_floor` fed from a search-args namespace (getattr-safe: relaunch
    and probe namespaces predate the flags)."""
    return _inet_floor(getattr(a, "inet_down", None),
                       pinned=bool(getattr(a, "machine", None)
                                   or getattr(a, "host", None)),
                       any_inet=getattr(a, "any_inet", False))


# A declared VRAM need is a MARKETING GB number; vast advertises the USABLE
# framebuffer. The gap is a hardware constant, so the search floor gets a
# tolerance band. `1 - VRAM_SEARCH_TOLERANCE` is that band; see
# `gpu_ram_floor_mib` for the measurement it is drawn from.
# moved-from: herdd.VRAM_SEARCH_TOLERANCE
VRAM_SEARCH_TOLERANCE = 0.99


# moved-from: herdd.gpu_ram_floor_mib
def gpu_ram_floor_mib(gpu_ram_gb: object) -> int:
    """THE one place a declared per-card VRAM need (GB) becomes the vast
    `gpu_ram` search floor (MiB). Every offer query goes through here.

    WHY THIS IS NOT `gb * 1024`, and why nobody should "clean up" the
    tolerance: a card's declared size is its MARKETING capacity, but vast
    advertises the USABLE framebuffer, which is always a little smaller. A
    "48 GB" RTX A6000 / RTX 6000Ada / RTX 5880Ada advertises **49140 MiB =
    47.99 GiB**, and 49140 < 48*1024 = 49152. A naive floor therefore excludes,
    by 12 MiB, the exact card class the declaration was written to admit —
    which is what it did: `--gpu a6000 --gpu-ram 48` returned 0 offers against
    a market with 4 rentable A6000s (reproduced 2026-08-16), so every bundle
    declaring `needs.gpu_ram_gb: 48` was unable to rent a 48 GB card at all.

    The band is 1%, from a full live-board survey (1465 verified on-demand
    offers, 2026-08-16). The carve-out is a fixed ~0.5% at EVERY size and never
    exceeded 0.53%: 8151/8192 (RTX 5060), 24467/24576 (RTX PRO 4000),
    32607/32768 (RTX 5090), 48935/49152 (RTX PRO 5000), 81559/81920 (H100),
    97887/98304 (RTX PRO 6000), 183359/184320 (B200). 1% is that constant
    doubled — the TIGHTEST band that admits every class on the board with ~2x
    margin. It is deliberately tighter than the 0.96 this replaced (which
    predates the survey and was reasoned from two data points): the
    declaration is a FLOOR for a real VRAM requirement, so slack is OOM risk,
    and 0.96 would hand a job declaring 48 a card with 46.1 GiB.

    What this does NOT do, on purpose: admit an ECC-capacity card into a
    higher class. L40 / L40S / an ECC-on RTX A6000 advertise 46068 MiB
    (44.99 GiB) — GDDR6 ECC costs 1/16 of capacity, so those cards really do
    not hold 48 GiB and a job declaring 48 must not land on one. They are
    reachable from an honest `gpu_ram_gb: 45` or lower. If a bundle wants
    L40S supply, the fix is re-deriving its measured need (`vram_facts.py`),
    never widening this band.

    Callers that pass an ADVERTISED value rather than a nominal class (the
    eviction-replacement probe derives its floor from the primary's own
    `gpu_ram`) get the same band, which there means "at most 1% smaller than
    the card we lost" — strictly tighter than the 4% they had before."""
    # `gpu_ram_gb` is `object` for the same reason `models._gpu_ram_gb`'s
    # `raw` is: the except clause below IS the contract for every non-numeric
    # value, so the coercion is deliberately not narrowed first.
    try:
        nominal = float(gpu_ram_gb) * 1024.0   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    if nominal <= 0:
        return 0
    return int(nominal * VRAM_SEARCH_TOLERANCE)


def filter_host_ram(rows: Iterable[Mapping[str, Any]], host_ram_gb: object,
                    ) -> tuple[list[dict[str, Any]], int]:
    """`(kept, dropped)` — rows narrowed to those whose SLICE clears the floor.

    Three-way, not two, and that is the whole point. A row is measurably big
    enough (kept, in market order), measurably too small (DROPPED — the server
    -side `cpu_ram` bound cannot see this, because it reads the whole host),
    or UNMEASURABLE (kept, but ranked after every measured row).

    Unknown is never a drop and never a zero. Refusing what we cannot measure
    empties the market on ignorance; scoring it zero is the same mistake with
    extra steps. Ranking it last means a box we can prove fits always beats one
    we cannot — and when nothing is measurable the caller still gets a box."""
    allrows = [dict(r) for r in rows]
    try:
        need = float(host_ram_gb)  # type: ignore[arg-type]  # except IS the contract
    except (TypeError, ValueError):
        return allrows, 0
    if need <= 0:
        return allrows, 0
    fits, unknown, small = [], [], 0
    for r in allrows:
        got = effective_host_ram_gb(r)
        if got is None:
            unknown.append(r)
        elif got >= need:
            fits.append(r)
        else:
            small += 1
    return fits + unknown, small


def _host_ram_note(host_ram_gb: object, dropped: int) -> None:
    """Say when the RAM floor bit, for `_cc_allow_note`'s reason: a
    shape-emptied market must not read as a price-emptied one."""
    if dropped:
        print(f">> note: host-RAM floor ({host_ram_gb} GB of SLICE, i.e. "
              f"cpu_ram x gpu_frac) excluded {dropped} offer(s) whose share of "
              f"their host was too small")


def _cpu_perf_note(ratio: object, dropped: int) -> None:
    """Say when the measured-CPU floor bit, for `_host_ram_note`'s reason: a
    shape-emptied market must not read as a price-emptied one — and here the
    difference matters more, because this floor rejects on OUR measurement
    rather than on anything the offer said about itself."""
    if dropped:
        print(f">> note: measured-CPU floor ({ratio}x the fleet median "
              f"per-thread rate) excluded {dropped} offer(s) we have measured "
              f"to be slower than that; offers we have never measured are kept "
              f"and ranked last")


def host_ram_floor_mib(host_ram_gb: object) -> int:
    """A declared host-RAM need (GB) as the vast `cpu_ram` search floor (MiB).

    **Deliberately `gb * 1024` with NO tolerance, unlike `gpu_ram_floor_mib` —
    do not "clean up" the asymmetry into symmetry.** That band exists because
    a card's advertised framebuffer is a fixed ~0.5% below its marketing
    capacity, so a naive floor excludes a whole DISCRETE class by 12 MiB.
    Host RAM has no classes: a slice is whatever the host had left, the
    values are continuous, and there is no cliff for a tolerance to step over.
    Slack here buys nothing but an OOM on a job that declared its real need.

    Soft on junk for `gpu_ram_floor_mib`'s reason: unusable input means NO
    floor, never a floor of zero smuggled in as a filter."""
    try:
        mib = float(host_ram_gb) * 1024.0   # type: ignore[arg-type]  # except IS the contract
    except (TypeError, ValueError):
        return 0
    return int(mib) if mib > 0 else 0


def effective_host_ram_gb(offer: Mapping[str, Any] | None) -> float | None:
    """Host RAM this OFFER actually gets (GB), or None when nothing can tell.

    The CPU-RAM twin of `models.effective_cores`, and it exists because vast
    does NOT publish one: there is a `cpu_cores_effective` but no
    `cpu_ram_effective`, and `cpu_ram` is the WHOLE MACHINE's memory. A
    1-of-8-GPU slice of a 768 GB host is ~96 GB, so a floor read straight off
    `cpu_ram` over-admits by 1/gpu_frac — it would buy a "768 GB" box for a job
    needing 128 and hand it 96.

    None, never 0, when `cpu_ram` or `gpu_frac` is missing or unusable. A box
    we cannot measure is an UNKNOWN box, not a bad one, and a caller must rank
    it last or refuse it with that reason rather than treat it as empty — the
    same house rule as the disk precheck's "a measurement we could not take is
    not evidence". This is not hypothetical: every `num_gpus=0` row on the
    board is a `resource_type: disk` VOLUME listing carrying `cpu_ram: 0`."""
    o = offer or {}
    ram = models._num_dph(o.get("cpu_ram"))
    if not ram or ram <= 0:
        return None
    frac = models._num_dph(o.get("gpu_frac"))
    if frac is None or frac <= 0:
        # A whole-host rental has no fraction to apply. Anything else is a
        # slice of unknown size, and guessing 1.0 would over-admit exactly the
        # way reading `cpu_ram` raw does.
        return None
    return round(ram / 1024.0 * frac, 2)


# Box-env key carrying the ARCHITECTURE ALLOWLIST the launch was made under: a
# comma-separated list of sm levels ("80,86,89,90"), stamped at create time and
# read back off `extra_env` by the supervise lanes. Same channel, and the same
# reasoning, as `disksize.LAUNCH_DISK_ENV` and `ENTRY_FLOOR`.
#
# It exists because a replacement was architecture-BLIND. Twice in two days a
# rehost honoured the VRAM floor and landed on an RTX PRO 6000 (sm_120), where
# the baked flash_attn 2.8.3 has no kernel image — the import succeeds and the
# first forward dies (2026-08-17, worked around by hand with `--gpu h200
# --max-replacements 0`; 2026-08-18, a pk2 A100 outbid and replaced, caught
# after the swap by a bundle gate). Which silicon a workload can RUN on is a
# statement about the workload, so it belongs on the launch and has to survive
# every hop — re-deriving it from whatever card the last box happened to be is
# the same defect the disk stamp fixed.
#
# Absent = no constraint. Making workloads arch-TOLERANT is the primary fix and
# is not this; this only keeps a replacement inside what the launch declared.
LAUNCH_CC_ALLOW_ENV = "LAUNCH_CC_ALLOW"

#: Rows to fetch when an allowlist is active. The `compute_cap` filter is
#: applied CLIENT-SIDE (see `pick_offers`), so a `limit=1` query would hand back
#: the single cheapest offer on the board and then drop it — an allowlist that
#: empties every search is indistinguishable from an empty market. Over-fetch,
#: filter, then trim to the caller's `limit`. One request either way.
CC_ALLOW_SCAN_LIMIT = 64


def parse_cc_allow(raw: object) -> tuple[int, ...]:
    """PURE. An sm allowlist from a stamp, a CLI string or a restored list:
    `"80,86,89,90"` / `[80, "sm_90"]` -> `(80, 86, 89, 90)`. `()` for absent or
    unparseable input, which every caller reads as NO CONSTRAINT.

    A value of 200 or more is read as the `compute_cap` spelling of the same
    thing and divided by 10 — vast advertises sm x10 (800, 890, 1200) and both
    numbers get typed by hand. The split is unambiguous over every sm level that
    exists: 75/80/86/89/90/100/120 are all below 200, their compute_cap twins
    (750...1200) all above it."""
    if raw is None:
        return ()
    items: list[object]
    if isinstance(raw, str):
        items = [p for p in raw.replace(";", ",").split(",")]
    elif isinstance(raw, (int, float)):
        items = [raw]
    else:
        if not isinstance(raw, Iterable):
            return ()
        items = list(raw)
    out: list[int] = []
    for it in items:
        s = str(it).strip().lower().removeprefix("sm_").removeprefix("sm")
        try:
            v = int(float(s))
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        v = v // 10 if v >= 200 else v
        if v not in out:
            out.append(v)
    return tuple(sorted(out))


# The sm level of each vast `gpu_name` we alias. STATIC silicon facts, not a
# preference — this table answers "would a --gpu NAME and a --cc-allow LIST
# intersect to zero offers?" BEFORE the search runs, which is the difference
# between a launcher that refuses in a second and one that reports an empty
# market and leaves the operator guessing. A name absent here is UNKNOWN and
# never blocks anything: `arch_changed`'s doctrine, an alarm on ignorance is one
# nobody reads.
GPU_NAME_SM: dict[str, int] = {
    # Ampere GA100 (8.0) / GA10x (8.6)
    "A100 SXM4": 80, "A100 PCIE": 80, "A100X": 80, "A800 PCIE": 80,
    "RTX 3090": 86, "RTX 3080": 86, "A40": 86,
    "RTX A6000": 86, "RTX A5000": 86, "RTX A4000": 86,
    # Ada (8.9)
    "RTX 4090": 89, "RTX 4080": 89, "L40S": 89, "L40": 89,
    "RTX 6000Ada": 89, "RTX 5880Ada": 89, "RTX 5000Ada": 89,
    "RTX 4500Ada": 89, "RTX 4000Ada": 89,
    # Hopper (9.0)
    "H100 SXM": 90, "H100 PCIE": 90, "H100 NVL": 90, "H200": 90, "H200 NVL": 90,
    # Blackwell datacenter (10.0) vs consumer/workstation Blackwell (12.0) —
    # DIFFERENT architectures for kernel purposes, which is exactly why the
    # allowlist is per-sm and not per-vendor-generation.
    "B200": 100, "B300": 100,
    "RTX 5090": 120, "RTX 5080": 120, "RTX 5070 Ti": 120, "RTX 5070": 120,
    "RTX PRO 6000 WS": 120, "RTX PRO 6000 S": 120,
    "RTX PRO 5000": 120, "RTX PRO 4500": 120, "RTX PRO 4000": 120,
}


def gpu_alias_sm(alias: str | None) -> tuple[int, ...]:
    """PURE. The sm levels a `--gpu` alias can land on — `"h100"` -> `(90,)`,
    `"rtxpro6000"` -> `(120,)`, `"l40"` -> `(89,)`. `()` when the alias is
    unknown to `GPU_ALIASES`/`GPU_NAME_SM`, which callers must read as "cannot
    tell", never as "no architecture"."""
    if not alias:
        return ()
    sms = []
    for name in normalize_gpu([str(alias)]):
        sm = GPU_NAME_SM.get(name)
        if sm is not None and sm not in sms:
            sms.append(sm)
    return tuple(sorted(sms))


def gpu_alias_conflicts(alias: str | None,
                        allow: Sequence[int] | None) -> tuple[int, ...]:
    """PURE. The sm levels `alias` resolves to when NOT ONE of them is inside
    `allow` — i.e. the card-name filter and the architecture allowlist are an
    AND that intersects to zero offers. `()` means no conflict, or that we
    cannot tell (unknown alias, empty allowlist).

    Reported BEFORE a search, because the symptom otherwise is "no offers match
    filters", which reads as a thin market rather than as two filters that can
    never both hold."""
    allow = tuple(allow or ())
    if not allow:
        return ()
    sms = gpu_alias_sm(alias)
    if not sms:
        return ()
    return () if any(s in allow for s in sms) else sms


def offer_sm(row: Mapping[str, Any] | None) -> int | None:
    """The sm level of an offer (or instance) row from its `compute_cap`, which
    vast publishes as sm x10. None when the row carries no usable value —
    UNKNOWN, never 0: a floor of zero would admit everything."""
    try:
        cc = int(float((row or {}).get("compute_cap")))  # type: ignore[arg-type]
    except (TypeError, ValueError, AttributeError):
        return None
    return cc // 10 if cc > 0 else None


def cc_allow_ok(row: Mapping[str, Any] | None,
                allow: Sequence[int] | None) -> bool:
    """Whether one offer row is inside an sm allowlist. No allowlist = True.

    An UNKNOWN `compute_cap` is EXCLUDED while an allowlist is active. The whole
    point of the list is that one architecture cannot run the workload, and a
    row that declines to say which architecture it is must not be the one that
    smuggles an sm_120 in."""
    if not allow:
        return True
    sm = offer_sm(row)
    return sm is not None and sm in tuple(allow)


def filter_cc_allow(rows: Iterable[Mapping[str, Any]],
                    allow: Sequence[int] | None,
                    ) -> tuple[list[dict[str, Any]], int]:
    """`(kept, dropped)` — `rows` narrowed to the allowlist. Pure; `dropped` is
    what a caller says out loud so an operator can see the filter bite."""
    allrows = list(rows)
    keep = [dict(r) for r in allrows if cc_allow_ok(r, allow)]
    return keep, len(allrows) - len(keep)


def arch_label(row: Mapping[str, Any] | None) -> str:
    """A human name for a box/offer's architecture: `"H200 (sm_90)"`, or just
    the gpu_name when the row carries no compute_cap. `"?"` when neither."""
    name = str((row or {}).get("gpu_name") or "").strip()
    sm = offer_sm(row)
    if name and sm:
        return f"{name} (sm_{sm})"
    return name or (f"sm_{sm}" if sm else "?")


def arch_changed(old_row: Mapping[str, Any] | None,
                 new_row: Mapping[str, Any] | None) -> bool:
    """Did a box swap cross an ARCHITECTURE boundary? False when it did not, and
    ALSO false when nothing here can tell — this drives an alarm, and an alarm
    that fires on ignorance is one nobody reads.

    `compute_cap` decides when both rows carry it. Otherwise the gpu_name ALIAS
    FAMILY does: an instance body does not advertise compute_cap at all, and
    "H100 NVL" replacing "H100 SXM" is the same card class for every purpose
    this fleet has, while "RTX PRO 6000 WS" replacing "A100 PCIE" is not."""
    a, b = offer_sm(old_row), offer_sm(new_row)
    if a is not None and b is not None:
        return a != b
    an = str((old_row or {}).get("gpu_name") or "").strip()
    bn = str((new_row or {}).get("gpu_name") or "").strip()
    if not an or not bn or an == bn:
        return False
    return not (set(gpu_family_names(an)) & set(gpu_family_names(bn)))


def container_disk_floor_gb(raw: object) -> float:
    """PURE. A `disk_space` search floor in GB, or 0.0 for absent/garbage.

    GB on both sides — an offer's `disk_space` is GB, unlike `gpu_ram`'s MiB —
    so this is a coercion, not a conversion, and exists so the CLI lane and the
    launch-time error hint cannot disagree about what the floor was."""
    try:
        v = float(raw)  # type: ignore[arg-type]  # the TypeError is the guard
    except (TypeError, ValueError):
        return 0.0
    return v if v > 0 else 0.0


# moved-from: herdd.build_search_query
def cpu_score(offer: Mapping[str, Any] | None) -> float | None:
    """GHz·cores for an offer — a RANKING PRIOR, not a capability claim.

    Why a score at all: a raw core count is the wrong unit for choosing a
    CPU box. Many slow cores beat few fast ones for parallel compiles, so a
    cores-only floor throws away cheap wide boxes; a GHz-only floor throws away
    the wide ones. Multiplying keeps both visible and ranks them against each
    other. Measured on the live board 2026-08-21, at one price ($0.012/hr):

        AMD EPYC 7713    256 cores @ 3.7GHz -> 952
        AMD EPYC 7343     64 cores @ 3.9GHz -> 252
        AMD Ryzen 5950X   32 cores @ 5.7GHz -> 182

    **What it cannot see: IPC.** A Broadwell Xeon and a Zen 3 EPYC at equal
    clock are not equal cores, and nothing in the offer payload says so — this
    over-rates old silicon, and the error grows with the age of the part. It is
    therefore a tie-break over price, never a reason to reject an offer, and it
    is not wired to any gate. The correction is measurement, not a hand-written
    IPC table: `hostfacts.py` banks per-MACHINE throughput from work we already
    run — and as of 2026-08-27 that correction is BUILT and this is the
    FALLBACK, not the ranking. See `cpu_perf` below; this is what an offer gets
    when nothing like it has ever been measured.

    `cpu_cores_effective`, not `cpu_cores` — an offer is a slice of a host.
    None when either term is missing: no term, no score, and a caller that
    sorts on it must put those last rather than treat them as zero.
    """
    o = offer or {}
    cores = models._num_dph(o.get("cpu_cores_effective"))
    ghz = models._num_dph(o.get("cpu_ghz"))
    if not cores or not ghz or cores <= 0 or ghz <= 0:
        return None
    return round(cores * ghz, 1)


# --------------------------------------------------------------------------- #
# measured CPU capability: what `cpu_score` is a stand-in for
# --------------------------------------------------------------------------- #

#: Reject a MEASURED offer below this fraction of the fleet median rate. A
#: ratio, not an absolute, so it re-derives itself when the kernel or the fleet
#: moves instead of pinning a number to one probe version.
#:
#: Deliberately low. It is a floor against pathological silicon, not a quality
#: bar: for embarrassingly parallel compiles a cheap wide old box can still be
#: the best throughput per dollar on the board, and a bar near the median throws
#: away exactly those. What earns the floor is that per-core rate is the axis
#: the value ranking CANNOT see — it sets the latency of a single compile, and
#: the search lanes have serial phases. At 0.35 against the 2026-08-27 table
#: this rejects one measured machine in 53 (a Xeon E5-2699 v4 at 0.29x).
#: The arm each question is asked of. They are different arms ON PURPOSE.
#:
#: THROUGHPUT ranks on `pyops`, whose rate is measured all-core and therefore
#: already carries the box's own scaling losses — the one arm you may multiply
#: by a slice width.
#:
#: The FLOOR asks a latency question ("will ONE compile be painfully slow
#: here?"), and `compile_tu` measures exactly that, serially, on the real
#: toolchain. It shipped on `pyops` (2026-08-27) only because the compile arm
#: then had 2 hosts; recovering it took that to 60, the same fleet, so the
#: floor now reads the workload instead of a proxy for it. The two rank the
#: fleet differently (Spearman 0.673) with a systematic desktop-vs-server
#: split, so this is not a cosmetic swap — see `hostfacts.CALIBRATION_ARMS`.
THROUGHPUT_ARM = "pyops"
FLOOR_ARM = "compile_tu"

#: Refuse a measured box below this fraction of the FLOOR ARM's fleet median.
#: 0.60 sits in a real gap in the measured distribution rather than on a knife
#: edge: four machines land at 0.44-0.57x and the next is 0.71x, so anything in
#: 0.57-0.71 refuses the same four and 0.60 clears the highest refused by 0.03
#: and the lowest kept by 0.11. Those four are three decade-old Xeons (E5-2690
#: v3, E5-2673 v4, E5-2699 v4) and the Xeon 6952P, a modern part that is simply
#: very wide and very slow per thread — which is precisely the box the
#: throughput ranking is happiest to buy and a serial compile suffers most on.
CPU_PERF_FLOOR_RATIO = 0.60

_CALIBRATION: dict[str, Any] | None = None
_CALIBRATION_LOADED = False


def _calibration_blob(reload: bool = False) -> dict[str, Any] | None:
    """The whole tracked table, either schema, or None. Cached; never raises.

    Lazy because `vastlib` must not need `hostfacts` (or its rclone/B2 world)
    to import — `hostfacts` returns the favour with its own lazy `herdd`
    import. A missing table is not an error: it reads as "nothing measured",
    which is the state every consumer here already handles.
    """
    global _CALIBRATION, _CALIBRATION_LOADED
    if reload:
        _CALIBRATION_LOADED = False
    if not _CALIBRATION_LOADED:
        _CALIBRATION_LOADED = True
        try:
            import hostfacts  # noqa: PLC0415

            # `hostfacts` is an unannotated stdlib-only script that also has to
            # run on a box with no vastlib, so the boundary is untyped by
            # design; the shape assertion is `load_calibration`'s own guard.
            blob: object = hostfacts.load_calibration()  # type: ignore[no-untyped-call]
            _CALIBRATION = blob if isinstance(blob, dict) else None
        except Exception:                                     # noqa: BLE001
            _CALIBRATION = None
    return _CALIBRATION


def cpu_calibration(reload: bool = False,
                    arm: str = THROUGHPUT_ARM) -> dict[str, Any] | None:
    """One ARM of the tracked table, in the single-arm shape the readers below
    take (`by_machine` / `by_model` / `fleet_median`), or None if that arm is
    not measured. A pre-`arms` table answers for any arm — see
    `hostfacts.calibration_arm`."""
    blob = _calibration_blob(reload)
    if blob is None:
        return None
    try:
        import hostfacts  # noqa: PLC0415

        got: object = hostfacts.calibration_arm(blob, arm)  # type: ignore[no-untyped-call]
    except Exception:                                         # noqa: BLE001
        return None
    return got if isinstance(got, dict) else None


_CPU_NAME_TRIM = re.compile(r"\b(processor|cpu)\b|\b\d+th gen\b|@.*$", re.I)
_CPU_NAME_PUNCT = re.compile(r"[^a-z0-9]+")


def cpu_name_key(name: object) -> str:
    """`cpu_name` reduced to what identifies the PART, for the model tier.

    Some hosts advertise the `/proc/cpuinfo` string the probe banks
    (`Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz`) and some a marketing one
    (`Xeon® E5-2699 v4 `) — 26 of 229 offers on a live board. Same silicon,
    no join, so the model tier silently missed them.

    Stripped: registered/trademark marks, the `(R)`/`(TM)` spellings, a
    `13th Gen` prefix, the `Processor`/`CPU` noise words, a trailing `@ 2.20GHz`
    (a base clock, not an identity — the same part is listed with and without
    it), the vendor prefix, and all punctuation/spacing.

    NOT stripped, ever: the model number and its suffix. They are the whole
    identity. An early cut of this reduced `7352` and `7402` to the same key,
    which would have reported an EPYC Rome part as a measured Genoa one — a
    silent family tier, which this module refuses on purpose (see `cpu_perf`).
    `test_cpu_name_key_never_merges_two_distinct_parts` pins that.
    """
    s = _CPU_NAME_TRIM.sub(" ", str(name or "").lower().replace("(r)", " ")
                           .replace("(tm)", " ").replace("®", " ")
                           .replace("™", " "))
    s = _CPU_NAME_PUNCT.sub(" ", s).strip()
    for vendor in ("intel ", "amd "):
        if s.startswith(vendor):
            s = s[len(vendor):]
    return s.strip()


def _model_index(table: Mapping[str, Any]) -> dict[str, Any]:
    """`cpu_name_key` -> row, built once per table object.

    A key that two DIFFERENT table entries reduce to is dropped from the index
    rather than resolved: the normalisation was supposed to be identity-
    preserving, so a collision means it is not, and guessing between two
    measured parts is worse than reporting the offer unmeasured.
    """
    cached = _MODEL_INDEX.get(id(table))
    if cached is not None:
        return cached
    idx: dict[str, Any] = {}
    clash: set[str] = set()
    for name, row in (table.get("by_model") or {}).items():
        k = cpu_name_key(name)
        if not k:
            continue
        if k in idx and idx[k] is not row:
            clash.add(k)
        idx[k] = row
    for k in clash:
        idx.pop(k, None)
    _MODEL_INDEX[id(table)] = idx
    return idx


_MODEL_INDEX: dict[int, dict[str, Any]] = {}


def cpu_perf(offer: Mapping[str, Any] | None,
             table: Mapping[str, Any] | None = None,
             arm: str = THROUGHPUT_ARM) -> dict[str, Any] | None:
    """Measured rate for this offer on `arm`, with its PROVENANCE, or None.

    Two tiers, and the name of each is exactly how strong it is:

      `machine`  this machine_id has been measured. No extrapolation at all.
      `model`    other machines with this cpu_name have. Carries `n` and
                 `spread` so a caller can see how far the generalisation is
                 being asked to stretch.

    Both are string joins — a vast offer carries `machine_id` and a `cpu_name`,
    the latter matched exactly and then through `cpu_name_key` for hosts that
    advertise the marketing spelling. There is no family tier and no silent
    fall back to `cpu_score`: an offer nothing resembles is UNMEASURED, which
    callers rank last rather than guess at. Mixing a measured rate with a
    modelled prior in one number is how a ranking stops meaning anything.
    """
    t = table if table is not None else cpu_calibration(arm=arm)
    o = offer or {}
    if not t:
        return None
    mid = str(o.get("machine_id") or "")
    hit = (t.get("by_machine") or {}).get(mid)
    if hit and hit.get("rate"):
        return {"rate": float(hit["rate"]), "tier": "machine", "n": 1,
                "spread": None}
    name = str(o.get("cpu_name") or "").strip()
    hit = (t.get("by_model") or {}).get(name)
    if not (hit and hit.get("rate")):
        hit = _model_index(t).get(cpu_name_key(name))
    if hit and hit.get("rate"):
        return {"rate": float(hit["rate"]), "tier": "model",
                "n": int(hit.get("n_machines") or 1), "spread": hit.get("spread")}
    return None


def cpu_throughput(offer: Mapping[str, Any] | None,
                   table: Mapping[str, Any] | None = None) -> float | None:
    """Predicted work-rate for the SLICE this offer sells, or None.

    `rate * cpu_cores_effective`. Both terms count THREADS, not physical cores:
    vast advertises a "48-Core" EPYC 7K62 as `cpu_cores: 96`, and the probe's
    width comes from the cgroup quota, which is also threads. Because the two
    sides agree the product is sound — but "per core" here is per THREAD and is
    not a clean IPC claim, and an SMT box's rate is correspondingly halved.

    The measured rate is already an ALL-CORE figure (`per_core_s` derives from
    the all-core count), so scaling losses are in it once. Do not apply a
    scaling factor on top; that would charge for them twice.
    """
    p = cpu_perf(offer, table)
    cores = models._num_dph((offer or {}).get("cpu_cores_effective"))
    if not p or not cores or cores <= 0:
        return None
    return float(p["rate"]) * cores


def cpu_value(offer: Mapping[str, Any] | None,
              table: Mapping[str, Any] | None = None) -> float | None:
    """Predicted work per dollar-hour, or None when either term is unknown.

    `dph_total` on both rental types: in bid mode it is the CURRENT
    INTERRUPTIBLE price, which is what a winning bid actually pays (the
    `build_search_query` comment below documents that at length).
    """
    thr = cpu_throughput(offer, table)
    dph = models._num_dph((offer or {}).get("dph_total"))
    if thr is None or not dph or dph <= 0:
        return None
    return thr / dph


def filter_cpu_perf(rows: Iterable[Mapping[str, Any]],
                    ratio: object,
                    table: Mapping[str, Any] | None = None,
                    ) -> tuple[list[dict[str, Any]], int]:
    """`(kept, dropped)` — rows narrowed to those the measured floor allows.

    Three-way for the same reason `filter_host_ram` is: measured-and-fast
    (kept), measured-and-slow (DROPPED), or UNMEASURED (kept, and ranked after
    every measured row). Unknown is never a drop — 70% of the cheap market is
    unmeasured, and refusing what we cannot measure empties the board on
    ignorance rather than on evidence.

    Market order is preserved within each group, so this narrows a
    cheapest-first page without reordering it.
    """
    allrows = [dict(r) for r in rows]
    try:
        r = float(ratio)  # type: ignore[arg-type]  # except IS the contract
    except (TypeError, ValueError):
        return allrows, 0
    if r <= 0 or cpu_perf_floor(table, r) is None:
        return allrows, 0
    fast, unknown, slow = [], [], 0
    for row in allrows:
        keep, why = cpu_floor_verdict(row, table, r)
        # Read "is this measured?" off the verdict rather than asking again.
        # A second `cpu_perf` call would have to name the same arm to agree,
        # and the two silently disagreeing is exactly how a row gets sorted as
        # unknown while being judged as measured.
        if why == "unmeasured":
            unknown.append(row)
        elif keep:
            fast.append(row)
        else:
            slow += 1
    return fast + unknown, slow


def cpu_perf_floor(table: Mapping[str, Any] | None = None,
                   ratio: float = CPU_PERF_FLOOR_RATIO) -> float | None:
    """The absolute rate below which a MEASURED offer is refused, or None.

    None when there is no table or no median — no measurement, no floor. A gate
    that fires on ignorance is one that empties the board.
    """
    t = table if table is not None else cpu_calibration(arm=FLOOR_ARM)
    med = models._num_dph((t or {}).get("fleet_median"))
    if not t or not med or med <= 0 or ratio <= 0:
        return None
    return med * ratio


def cpu_floor_verdict(offer: Mapping[str, Any] | None,
                      table: Mapping[str, Any] | None = None,
                      ratio: float = CPU_PERF_FLOOR_RATIO
                      ) -> tuple[bool, str]:
    """`(keep, why)` for one offer against the floor.

    UNMEASURED OFFERS ARE KEPT. An offer nothing resembles cannot be shown to be
    below the floor, and 70% of the cheap market is unmeasured today — a gate
    that dropped those would delete the board and call it selectivity. It is the
    same contract `effective_host_ram_gb` states: a box we cannot measure is an
    UNKNOWN box, not a bad one. Unknown ranks last; only MEASURED-AND-SLOW is
    refused.
    """
    floor = cpu_perf_floor(table, ratio)
    p = cpu_perf(offer, table, arm=FLOOR_ARM)
    if floor is None or p is None:
        return True, "unmeasured" if floor is not None else "no calibration"
    if p["rate"] >= floor:
        return True, p["tier"]
    return False, (f"measured {p['rate']:.3g} < floor {floor:.3g} "
                   f"({ratio:g}x {FLOOR_ARM} fleet median, via {p['tier']})")


def build_search_query(a: argparse.Namespace) -> dict[str, Any]:
    # In bid mode filter and sort on min_bid (the spot floor — what a bid must
    # clear) so --max-dph means "max I pay". NOTE the bid view's dph_total is
    # the CURRENT INTERRUPTIBLE price (min_bid + the storage sliver), and its
    # dph_base equals min_bid — NEITHER is the on-demand rate (API-verified
    # 2026-08-06; an earlier comment here claimed dph_total was the on-demand
    # list price, and that wrong claim seeded the doc 50 R1 razor-thin-bid
    # defect family — see _offer_ondemand_ref for the correct reference).
    price_field = "dph_total" if a.type == "ondemand" else "min_bid"
    q: dict[str, Any] = {
        "limit": a.limit,
        "type": a.type,
        "rentable": {"eq": True},
        "num_gpus": {"gte": a.num_gpus},
        "order": [[price_field, "asc"]],
    }
    if not a.unverified:
        q["verified"] = {"eq": True}
    if a.gpu:
        q["gpu_name"] = {"in": normalize_gpu(a.gpu)}
    if a.gpu_ram:
        q["gpu_ram"] = {"gte": gpu_ram_floor_mib(a.gpu_ram)}  # API is MiB
    if a.max_dph is not None:
        q[price_field] = {"lte": a.max_dph}
    # CONTAINER-DISK floor. `--disk N` is a REQUEST, not a guarantee: a machine
    # advertising less than N does not refuse the rental, it hands back a
    # SMALLER container (`pick_offers`'s `disk_gb` carries the same floor for the
    # automatic re-rent lanes, and says so at length). This lane had only
    # `--host-disk`, which nothing derives from `--disk`, so an auto-picked
    # cheapest offer silently downgraded the allocation.
    _disk_floor = max(container_disk_floor_gb(a.host_disk),
                      container_disk_floor_gb(getattr(a, "disk", None)))
    if _disk_floor:
        q["disk_space"] = {"gte": _disk_floor}
    # CPU-shape floors. Server-side only to bound the fetch — the box is CHOSEN
    # by `cpu_score` ranking below, never by these. `cpu_cores_effective` and
    # not `cpu_cores` for the reason `models` gives at length: an offer is a
    # SLICE of a host, and the raw field is the whole machine (measured
    # 2026-08-21: three live boxes advertising 256/64/256 cores were all 32-core
    # slices).
    if getattr(a, "cpu_cores", 0):
        q["cpu_cores_effective"] = {"gte": a.cpu_cores}
    if getattr(a, "cpu_ghz", 0):
        q["cpu_ghz"] = {"gte": a.cpu_ghz}
    # HOST-RAM floor. `cpu_ram` is the WHOLE MACHINE's memory in MiB, so this is
    # a NECESSARY condition and not a sufficient one — the slice you rent is
    # `cpu_ram * gpu_frac` (`effective_host_ram_gb`). Filtering the host field
    # can never exclude a viable offer, which is what makes it the right server
    # -side bound; the sufficient half is client-side, where a missing value can
    # read as UNKNOWN instead of as zero.
    if getattr(a, "host_ram", 0):
        q["cpu_ram"] = {"gte": host_ram_floor_mib(a.host_ram)}  # API is MiB
    if a.reliability:
        q["reliability"] = {"gte": a.reliability}
    if a.cuda:
        q["cuda_max_good"] = {"gte": a.cuda}
    # Default download floor for auto-picks (owner directive 2026-08-03); an
    # explicit --inet-down wins (0 disables), pins + --any-inet bypass.
    _inet = _inet_floor_for(a)
    if _inet:
        q["inet_down"] = {"gte": _inet}
    if getattr(a, "machine", None):
        q["machine_id"] = {"in": a.machine}
    elif getattr(a, "exclude_machines", None):
        # Host rotation for the relaunch lanes (supervise/babysit boot-health
        # condemn): skip every machine a prior attempt already failed on so the
        # replacement lands on a DIFFERENT host. Mutually exclusive with an
        # explicit --machine pin (an operator pin wins; excluding a pinned
        # machine would just search nothing).
        q["machine_id"] = {"notin": list(a.exclude_machines)}
    if getattr(a, "host", None):
        q["host_id"] = {"in": a.host}
    if getattr(a, "geo", None):
        # vast exposes each host's country as a 2-letter code in `geolocation`;
        # `in` matches any listed. OPT-IN ONLY, no default (owner directive
        # 2026-08-05): `--geo US` used to be the standing advice because a US
        # host sits near the us-west-004 B2 bucket, but the bandwidth gate
        # (`inet_down >= LAUNCH_INET_DOWN_MBPS`, applied above) measures the
        # thing that advice was proxying for, and the container image is now
        # small enough (t211 + per-flow layering) that a fast non-US host boots
        # inside the 600 s SLA. Keep this flag for real geography constraints
        # (B2 locality, residency), not as a bandwidth heuristic.
        q["geolocation"] = {"in": [g.upper() for g in a.geo]}
    return q


# moved-from: herdd.search_offers
def search_offers(a: argparse.Namespace) -> list[dict[str, Any]]:
    tiers = _gpu_policy_tiers(a)
    q = build_search_query(a)
    # `--cc-allow` narrows CLIENT-SIDE, for `pick_offers`' reasons — including
    # the over-fetch, without which the cheapest row on the board is fetched and
    # then dropped. The launch that follows STAMPS this list into the box env,
    # so a search that ignored it would hand back an offer its own stamp forbids.
    allow = parse_cc_allow(getattr(a, "cc_allow", None))
    want = int(q.get("limit") or 0) or 1
    if allow:
        q = dict(q, limit=max(want, CC_ALLOW_SCAN_LIMIT))

    def _narrow(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not allow:
            return rows
        kept, n = filter_cc_allow(rows, allow)
        _cc_allow_note(allow, n)
        return kept[:want]

    if tiers is None:
        d = api.request("POST", "v0/bundles/", q)
        return _narrow(d.get("offers", []) or [])
    # Default GPU policy (GPU_DEFAULT_POLICY_TIERS): no name/pin given — search
    # tier by tier so cheapest-within-preferred beats cheapest-overall.
    for tier in tiers:
        d = api.request("POST", "v0/bundles/", dict(q, gpu_name={"in": list(tier)}))
        offers = _narrow(d.get("offers", []) or [])
        if offers:
            return offers
    return []


# moved-from: herdd.pick_cheapest_offer
def pick_cheapest_offer(**kw: Any) -> dict[str, Any] | None:  # noqa: ANN401 — see pick_offers
    """The single cheapest qualifying offer, or None. Thin alias for
    `pick_offers(limit=1)[0]` — kept as the name every caller (workflow
    controller, understudy, replacement, relaunch lanes) already binds and
    monkeypatches. See `pick_offers` for the whole contract."""
    offers = pick_offers(limit=1, **kw)
    return offers[0] if offers else None


# moved-from: herdd.pick_offers
def pick_offers(*, gpu: Sequence[str] = (), num_gpus: int = 1,
                gpu_ram_gb: object = None, disk_gb: object = None,
                host_ram_gb: object = None,
                max_dph: float | None = None,
                rental: str = "bid", verified: bool = True,
                inet_down: object = None,
                exclude_machines: Iterable[object] | None = None,
                geo: Iterable[str] | None = None, any_gpu: bool = False,
                any_inet: bool = False, cuda: float | None = None,
                cc_allow: Sequence[int] | None = None,
                hostrep: bool = True,
                min_cpu_perf: float | None = None,
                limit: int = 1) -> list[dict[str, Any]]:
    """Argparse-free primitive: search offers and return the qualifying ones
    CHEAPEST-FIRST (at most `limit`), or [] on no match / any API error. Builds
    the same bundles query as `build_search_query` (bid mode sorts/filters on
    `min_bid`, on-demand on `dph_total`) without needing a fake argparse
    namespace — the API already sorts ascending, so the first offer is the
    cheapest. Soft by contract — never sys.exits; the workflow controller
    reuses this to choose a launch offer.

    `limit` (2026-08-16) is what makes a CANDIDATE SET possible. Every caller
    of `pick_cheapest_offer` gets a sample of ONE, and a per-offer safety rail
    evaluated against a sample of one is a coin flip: on 2026-08-16 the
    eviction-replacement rung read exactly one H100 NVL offer whose bid floor
    sat at 91.5% of its own on-demand price, `bid_decision` correctly refused
    to hold that machine on spot, and the ladder bought the on-demand box —
    while a $0.40 H200 NVL spot offer sat unqueried on the same market. Walk a
    list, apply the rail per candidate, and the rail selects instead of
    vetoing. Cost is one API request either way.

    `disk_gb` (2026-08-16) is the CONTAINER-DISK floor in GB — a hard fit
    requirement, not a preference, on every lane that will hand `--disk N` to
    the launch it makes from this pick. See the filter below for what happens
    without it.

    `host_ram_gb` is the HOST-RAM floor in GB, the axis a CPU-shaped job is
    actually sized by. It filters TWICE and has to: the query bounds on
    `cpu_ram`, which is the whole machine's memory, and the survivors are then
    checked on the SLICE (`effective_host_ram_gb`). Offers whose slice cannot be
    measured are kept but ranked LAST — never dropped and never read as zero, so
    an unmeasurable market still yields a box.

    `cc_allow` (2026-08-18) is the sm ARCHITECTURE allowlist — the axis the
    replacement lane shipped without, which is how an evicted A100 was rehosted
    onto an sm_120 RTX PRO 6000 whose flash_attn has no kernel image for it.
    Applied CLIENT-SIDE after the query (the bundles filter shape for
    `compute_cap` is not established in this codebase, and a wrong server-side
    filter fails silently as an empty market), which is why an active allowlist
    over-fetches `CC_ALLOW_SCAN_LIMIT` rows before trimming back to `limit`. A
    row with NO `compute_cap` is dropped — see `cc_allow_ok`.

    `rental` accepts EITHER the vast-native `"ondemand"` or the workflow's
    frozen `"on-demand"` (workflow.RENTAL_CHOICES) spelling — both normalize to
    the API's `"ondemand"` type and to sorting/filtering on `dph_total`. Anything
    else (i.e. `"bid"`) is the interruptible market, sorted on `min_bid`.

    With no `gpu` name the default preferred-GPU policy applies
    (GPU_DEFAULT_POLICY_TIERS — bf16-capable only, never pre-Ampere): tiers are
    searched in order and the cheapest offers of the first non-empty tier win.
    `any_gpu=True` restores the unrestricted cheapest-overall pick.

    `hostrep` (2026-08-20) is the DURABLE host-reputation layer, and this is
    where it binds for every automatic lane at once — launch, the eviction and
    pull replacements, the workflow controller, the understudy — because they
    all arrive here and none of them should have to remember a policy. Blocked
    machines are excluded IN THE QUERY (a blocked host filling a cheapest-first
    page would otherwise crowd out usable offers), and the survivors are
    reordered by reputation-adjusted price, so the returned list is no longer
    strictly cheapest-first when we hold evidence against the cheap host. That
    is the point: owner directive 2026-08-20, "a cheap host that doesn't work is
    not worth using for us". `hostrep=False` (and `$VAST_HOSTREP_DISABLE=1`)
    restores the pure cheapest-first pick — see `vastlib.market.hostrep`.

    `min_cpu_perf` (2026-08-27) is the MEASURED-CPU floor as a fraction of the
    fleet median, and it is OPT-IN here while `herdd search` arms it by
    default. The asymmetry is deliberate: this picker backs the eviction
    replacement, the pull/SLA reschedule and the workflow controller — lanes
    that spend money to rescue a run — and a floor armed by default there could
    refuse a GPU rescue over a CPU the job never needed. A caller doing
    CPU-shaped work passes it; nothing else pays for it. Unmeasured offers are
    kept and ranked last either way (`filter_cpu_perf`)."""
    is_ondemand = rental in ("ondemand", "on-demand")
    price_field = "dph_total" if is_ondemand else "min_bid"
    if hostrep:
        # Union, never replace: an explicit exclusion is the caller's evidence
        # about THIS attempt and outranks nothing, but must not be dropped.
        exclude_machines = hostrep_mod.with_blocked(exclude_machines)
    cc_allow = tuple(cc_allow or ())
    want = max(1, int(limit or 1))
    # A host-RAM floor narrows CLIENT-SIDE too (the slice check), so it needs
    # cc_allow's over-fetch for cc_allow's reason: fetching exactly `want` and
    # then dropping rows hands back fewer offers than asked for, or none.
    _overfetch = bool(cc_allow) or bool(host_ram_gb) or bool(min_cpu_perf)
    q: dict[str, Any] = {
        "limit": max(want, CC_ALLOW_SCAN_LIMIT) if _overfetch else want,
        "type": "ondemand" if is_ondemand else "bid",
        "rentable": {"eq": True},
        "num_gpus": {"gte": num_gpus},
        "order": [[price_field, "asc"]],
    }
    if verified:
        q["verified"] = {"eq": True}
    if gpu:
        q["gpu_name"] = {"in": normalize_gpu(gpu)}
    if gpu_ram_gb:
        # Same floor as `build_search_query` — see `gpu_ram_floor_mib` for why
        # it is not `gb * 1024` and where the tolerance comes from. This site
        # carried its own 0.96 from 2026-07-15 (the E2 controller looping
        # need_box forever on a market full of matching 5090s); the two
        # translations then drifted, and the search path's exact floor made the
        # whole 48 GB class unrentable. One helper so they cannot drift again.
        q["gpu_ram"] = {"gte": gpu_ram_floor_mib(gpu_ram_gb)}  # API is MiB
    if disk_gb:
        # CONTAINER-DISK floor (`disk_space`, GB on an offer — NOT MiB like
        # gpu_ram), the same filter the CLI's `--host-disk` sets. Automatic
        # lanes that re-rent on our behalf MUST pass it: the launch that
        # follows asks vast for a fixed `--disk`, and a machine advertising
        # less than that does not refuse the rental — it hands back a smaller
        # container. That is how the eviction/pull ladder minted 47845159
        # (23 GB) and 47845212 (47 GB) on 2026-08-16 against a 50 GB request,
        # both of which then died or were destroyed before they could
        # (GPU_BENCH_RESULTS.md Part 2 §5: all cheap A100 PCIe supply that
        # night was one host, 67231, advertising 18/23/33/47/128/275 GB).
        q["disk_space"] = {"gte": float(disk_gb)}  # type: ignore[arg-type]
    if host_ram_gb:
        # Same floor as `build_search_query` — one helper, for the reason
        # `gpu_ram_floor_mib` gives at length (the two query builders are
        # hand-maintained copies and their VRAM translations silently diverged
        # for a month). Whole-host bound only; the slice check is below.
        q["cpu_ram"] = {"gte": host_ram_floor_mib(host_ram_gb)}  # API is MiB
    if max_dph is not None:
        q[price_field] = {"lte": max_dph}
    # Download-bandwidth floor (Mb/s). Cheap hosts shape per-TCP-flow to
    # 1-16 MB/s, so a cold multilayer image pull can blow past the boot
    # deadline (found live 2026-07-15: a 5090 box never finished pulling
    # train-vast-latest in 1200s though it pulls in ~127s on a fast host).
    # 2026-08-03: with inet_down=None the LAUNCH_INET_DOWN_MBPS knob applies
    # by default (any_inet=True bypasses); explicit 0 disables. When the
    # DEFAULT floor empties the market, a second unfloored pass runs — this
    # picker feeds relaunch/understudy/workflow lanes that must not fail for
    # want of a fast host (a slow host under the boot SLA beats no host).
    _floor = _inet_floor(inet_down, any_inet=any_inet)
    if _floor:
        q["inet_down"] = {"gte": _floor}
    if exclude_machines:
        # Skip machines a prior attempt already failed on (retry host-diversity)
        # so a retry doesn't loop on the same slow/broken cheapest host.
        q["machine_id"] = {"notin": list(exclude_machines)}
    if cuda:
        # Host DRIVER floor (cuda_max_good), the same filter the CLI `--cuda`
        # lane sets. Automatic lanes that re-rent on our behalf (the eviction
        # replacement, the pull/SLA reschedule) MUST pass this: they re-launch
        # the primary's own image, and an image whose CUDA runtime outruns the
        # host driver boots into Error-804 instead of training (memory
        # vast-cuda-driver-floor). The number is config.LAUNCH_CUDA_MAX_GOOD —
        # it tracks the image, not a card. The `--geo` lane is deliberately NOT the
        # symmetric knob here — see the geo note below.
        q["cuda_max_good"] = {"gte": cuda}
    if geo:
        # 2-letter country codes, same filter shape as the CLI --geo lane.
        # NO DEFAULT (owner directive 2026-08-05): the 2026-07-20 US pin was a
        # proxy for "hosts that can pull the image fast", and the direct
        # measurement — inet_down >= LAUNCH_INET_DOWN_MBPS above, plus the 600 s
        # boot SLA and the pull watchdog behind it — now does that job on its
        # own. Geography stays available as an explicit operator choice (B2
        # locality, data residency); it is no longer an implicit bandwidth
        # heuristic. Supersession record:
        # docs/plans/witness/g2_push/FLEETD_AUTOREPLACE_2026-08-05.md.
        q["geolocation"] = {"in": [g.upper() for g in geo]}
    # Default GPU policy: no gpu name and no escape hatch -> tiered allowlist
    # search (see GPU_DEFAULT_POLICY_TIERS; owner directive 2026-08-03).
    if gpu or any_gpu:
        queries = [q]
    else:
        queries = [dict(q, gpu_name={"in": list(t)})
                   for t in GPU_DEFAULT_POLICY_TIERS]
    if _floor and inet_down is None:
        # unfloored fallback pass (default floor only — an explicit floor is a
        # real constraint and stays hard)
        queries = queries + [{k: v for k, v in qq.items() if k != "inet_down"}
                             for qq in queries]
    dropped = 0
    for qq in queries:
        ok, d, _ = api.request_soft("POST", "v0/bundles/", qq)
        if not ok or not isinstance(d, dict):
            continue                       # soft: a tier-read blip, try the next
        offers = d.get("offers") or []
        if cc_allow:
            offers, n = filter_cc_allow(offers, cc_allow)
            dropped += n
        if host_ram_gb:
            offers, n = filter_host_ram(offers, host_ram_gb)
            _host_ram_note(host_ram_gb, n)
        if min_cpu_perf:
            offers, n = filter_cpu_perf(offers, min_cpu_perf)
            _cpu_perf_note(min_cpu_perf, n)
        if _overfetch:
            offers = offers[:want]
        if offers:
            _cc_allow_note(cc_allow, dropped)
            return _hostrep_rerank(offers, price_field, hostrep)
    _cc_allow_note(cc_allow, dropped)
    return []


def _hostrep_rerank(offers: Sequence[Mapping[str, Any]], price_field: str,
                    on: bool) -> list[dict[str, Any]]:
    """Reputation-adjusted order, and SAY so when it changed the pick.

    Fail-open by construction: any error inside the reputation layer returns the
    market's own order. A ranking preference must never be able to make a launch
    impossible — the whole point is to get a box that boots, not to get no box.
    """
    rows = [dict(o) for o in offers]
    if not on:
        return rows
    try:
        ranked, notes = hostrep_mod.rank_offers(rows, price_field)
    except Exception:
        return rows
    for line in notes:
        print(line)
    return [dict(o) for o in ranked]


def _cc_allow_note(cc_allow: Sequence[int], dropped: int) -> None:
    """Say how many offers the architecture allowlist removed. Silence would
    make an arch-emptied market read as a price-emptied one, which is the class
    of confusion the container-disk floor's own 'name the bound' note exists to
    prevent."""
    if cc_allow and dropped:
        print(f">> note: architecture allowlist "
              f"(sm {','.join(str(s) for s in cc_allow)}) excluded {dropped} "
              f"offer(s), including any whose compute_cap the market did not "
              f"advertise")


# moved-from: herdd._search_offers_soft
def _search_offers_soft(a: argparse.Namespace) -> list[dict[str, Any]]:
    """search_offers() that returns [] instead of sys.exit on any API error.
    Applies the same default preferred-GPU policy (GPU_DEFAULT_POLICY_TIERS)
    when the captured spec / CLI gave no gpu name — an eviction relaunch or
    handoff pick must not land on pre-Ampere silicon either. The default
    inet-down floor (2026-08-03) applies too, but with an automatic UNFLOORED
    retry: a rescue/relaunch pick must never fail outright for want of a fast
    host — a slow host under the boot SLA beats no host at all. An EXPLICIT
    --inet-down stays hard (a real constraint, not a default)."""
    tiers = _gpu_policy_tiers(a)
    allow = parse_cc_allow(getattr(a, "cc_allow", None))

    def _offers_for(ns: argparse.Namespace) -> list[dict[str, Any]]:
        q = build_search_query(ns)
        want = int(q.get("limit") or 0) or 1
        if allow:
            q = dict(q, limit=max(want, CC_ALLOW_SCAN_LIMIT))
        for qq in ([q] if tiers is None else
                   [dict(q, gpu_name={"in": list(t)}) for t in tiers]):
            ok, d, err = api.request_soft("POST", "v0/bundles/", qq)
            if not ok or not isinstance(d, dict):
                continue
            offers = d.get("offers", []) or []
            if allow:
                offers, n = filter_cc_allow(offers, allow)
                _cc_allow_note(allow, n)
                offers = offers[:want]
            if offers:
                return list(offers)
        return []

    offers = _offers_for(a)
    if offers:
        return offers
    if getattr(a, "inet_down", None) is None and _inet_floor_for(a):
        a2 = argparse.Namespace(**dict(vars(a)))
        a2.inet_down = 0
        offers = _offers_for(a2)
        if offers:
            print(f">> note: no offer clears the default inet-down floor "
                  f"({config._boot_knob('LAUNCH_INET_DOWN_MBPS'):g} Mb/s) — picking "
                  f"unfloored (slow-pull risk; the boot SLA still enforces the "
                  f"come-online deadline)")
            return offers
    return []


#: Row cap for `_offer_machine_scan_soft`. Deliberately far above the search
#: `--limit` (20): this is a haystack scan for one known id, not a pick.
# moved-from: herdd.OFFER_SCAN_LIMIT
OFFER_SCAN_LIMIT = 512


# moved-from: herdd._offer_machine_scan_soft
def _offer_machine_scan_soft(a: argparse.Namespace) -> dict[str, Any] | None:
    """Recover a pinned `--offer`'s ROW by scanning, not by filtering: one soft
    `v0/bundles/` POST built from `build_search_query(a)` with the (dead) `id`
    key deliberately absent, matched on `o["id"] == a.offer` in Python.

    A RUNG, not the fix. The same reshuffling that kills the id filter means the
    pinned chunk id may genuinely not be in ANY current listing, and the query
    still carries the caller's own filters (gpu/ram/max-dph/verified), so a miss
    is expected and cheap. When it hits, the row carries everything the pricing
    ladder wants — `machine_id`, `min_bid`, `num_gpus`, `cuda_max_good`.

    Returns the offer dict, or None."""
    try:
        want = int(getattr(a, "offer", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    mid = getattr(a, "offer_machine", None)
    # getattr-safe rebuild rather than `build_search_query(a)` straight: the
    # internal launch namespaces (handoff understudy, jobs replacement) predate
    # several search flags, and an AttributeError here would take a LAUNCH down.
    q_ns = argparse.Namespace(
        limit=getattr(a, "limit", 0) or 0,
        type=getattr(a, "type", "bid"),
        num_gpus=getattr(a, "num_gpus", 1) or 1,
        unverified=getattr(a, "unverified", False),
        gpu=getattr(a, "gpu", None), gpu_ram=getattr(a, "gpu_ram", None),
        max_dph=getattr(a, "max_dph", None),
        host_disk=getattr(a, "host_disk", 0),
        reliability=getattr(a, "reliability", 0),
        cuda=getattr(a, "cuda", 0),
        inet_down=getattr(a, "inet_down", None),
        any_inet=getattr(a, "any_inet", False),
        # An --offer-machine pin narrows the haystack to one machine, which is
        # the difference between a lucky hit and a real one.
        machine=(getattr(a, "machine", None) or ([int(mid)] if mid else None)),
        host=getattr(a, "host", None), geo=getattr(a, "geo", None),
        exclude_machines=getattr(a, "exclude_machines", None))
    q = dict(build_search_query(q_ns))
    q.pop("id", None)
    q["limit"] = max(int(q.get("limit") or 0), OFFER_SCAN_LIMIT)
    ok, d, _ = api.request_soft("POST", "v0/bundles/", q, retries=2)
    if not ok or not isinstance(d, dict):
        return None
    for o in d.get("offers") or []:
        try:
            if int(o.get("id")) == want:
                return o                      # type: ignore[no-any-return]
        except (TypeError, ValueError):
            continue
    return None


# moved-from: herdd._offer_cuda_soft
def _offer_cuda_soft(offer_id: object, row: object = None) -> float | None:
    """Soft cuda_max_good for a pinned `--offer`, or None when nothing resolves
    the row.

    `row` — an offer dict already recovered by `_offer_machine_scan_soft` — is
    the path that actually works. The id-filtered fallback below is retained as
    a rung and is expected to return nothing: the `id` filter is dead in every
    view (see `_offer_pricing_soft`). Callers must keep degrading to a warning
    on None; the on-box `ensure_cuda_init` probe is the remaining gate."""
    if isinstance(row, dict) and row.get("cuda_max_good") is not None:
        try:
            return float(row["cuda_max_good"])
        except (TypeError, ValueError):
            return None
    try:
        oid = int(offer_id)                   # type: ignore[call-overload]
    except (TypeError, ValueError):
        oid = offer_id
    for typ in ("bid", "ondemand"):
        q = {"limit": 1, "type": typ, "id": {"in": [oid]}}
        ok, d, _ = api.request_soft("POST", "v0/bundles/", q, retries=2)
        if ok and isinstance(d, dict):
            offers = d.get("offers") or []
            if offers and offers[0].get("cuda_max_good") is not None:
                try:
                    return float(offers[0]["cuda_max_good"])
                except (TypeError, ValueError):
                    return None
    return None


# moved-from: herdd._replacement_candidate_class
def _replacement_candidate_class(primary: models.Payload | None,
                                 gpu_name: str | None = None,
                                 ) -> tuple[tuple[str, ...], float | None]:
    """(gpu_names, gpu_ram_gb) — the MINIMUM-REQUIREMENTS candidate class for a
    replacement rental (owner ruling 2026-08-16: "minimum requirements + best
    tokens-per-dollar", never a SKU pin).

    Hard requirements, and only these:

      * bf16-capable and not pre-Ampere — the GPU_DEFAULT_POLICY_TIERS
        allowlist, flattened. Flattened, NOT tiered: tiering ranks by
        preference and the ruling ranks by tokens-per-dollar, so a tier walk
        would hide a cheaper-per-token card behind a nominally nicer one. The
        primary's own alias family is unioned in so a SKU we never listed can
        still replace itself.
      * per-card VRAM at or above the primary's. UPGRADES ARE ALLOWED and
        downgrades are not: a faster/bigger card that wins on tokens-per-dollar
        is the better deal AND finishes sooner, which is less eviction
        exposure. `gpu_ram_gb=None` when the primary's VRAM is unreadable —
        then the class narrows to the primary's own alias FAMILY, because
        without the floor a name-open search could hand back a smaller card.

    num_gpus / cuda / inet / the price ceiling are the caller's arguments to
    `pick_offers` and are unchanged by this function."""
    primary = primary or {}
    name = (gpu_name if gpu_name is not None
            else (primary.get("gpu_name") or "")).strip()
    fam = gpu_family_names(name)
    ram_gb = models._gpu_ram_gb(primary.get("gpu_ram") or primary.get("gpu_totalram"))
    if ram_gb is None:
        # No VRAM floor to enforce => a name-open class could DOWNGRADE us.
        # Fall back to the primary's own family: still wider than the exact-SKU
        # pin this replaced, and never smaller than the card we lost.
        return tuple(fam), None
    names = []
    for tier in GPU_DEFAULT_POLICY_TIERS:
        for n in tier:
            if n not in names:
                names.append(n)
    for n in fam:
        if n not in names:
            names.append(n)
    return tuple(names), ram_gb

