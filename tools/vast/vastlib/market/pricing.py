"""vastlib.market.pricing — what a box costs: market reads and the bid number.

Why this exists
---------------
Every price this fleet acts on is READ FROM THE MARKET, not taken off the
object we already hold, and each of those reads has a defect behind it:

* a BID-view offer row's `dph_total` is the current interruptible price and its
  `dph_base` equals `min_bid`, so NOTHING on a bid row carries the on-demand
  rate. Using `dph_total` as the clamp reference put bids a rounding unit over
  their own floor and lost two understudy boxes inside an hour — the doc 50 R1
  family (`_offer_ondemand_ref`, `_market_ondemand_soft`).
* a machine lists offers per CHUNK (1/2/4 GPUs) and `min_bid` is per-chunk
  while our standing bid is whole-instance, so a bare `min()` read the 1-GPU
  floor for a 2-GPU box and vast underbid-parked it (defect D5 —
  `_market_chunk_floors`, `_market_min_bid_soft`).
* "the read failed" and "vast answered and we are no longer listed" used to
  collapse into one `None`, which is why an outbid produced no observable at
  all (defect D7 — `MarketRead`, `_market_bid_listed_soft`).

So the contracts here are narrow on purpose and the tri-states are load-bearing.
`ok=False` is IGNORANCE and must never advance eviction state (SPOT_DESIGN §5
rule 1); `ok=True, listed=False` is EVIDENCE. Widening one of these into the
other is how a live box gets parked.

What is deliberately NOT here
-----------------------------
* **No decisions.** `bidpolicy` (Zone S) computes every bid rail and
  `ladder_core` owns the two lanes' state transitions. `_auto_bid_price` is a
  pure delegation to `bidpolicy._bid_target` precisely so the "launch price ==
  steady-state target" invariant (SPOT_DESIGN §3.2) is enforced by a call
  rather than by a comment — `test_bid_cushion.py` asserts that identity. Do
  not reintroduce local arithmetic here.
* **No bid PLACEMENT.** This module computes the number; `boxes.lifecycle`
  performs the PUT (`_put_bid_soft` / `set_bid`).
* **No ladder implementation.** The five `ladder_core` names below are ALIASES,
  identity-preserving, exactly as `herdd.py` re-exports them — one copy of
  the state transitions, addressable from both. `ladder_core.py` stays a live
  flat file this step; its `def`s move at plan step 7, and `num_dph` becomes
  public `core.models.num_dph` there (this module uses `models._num_dph`, which
  is already that alias).
* **No result-shape NamedTuple for `_offer_pricing_soft`.** `core/result.py`
  anticipates one here for its shape-J triple; it is deliberately not built,
  because that function is MEASURED DEAD (below) and a verbatim port is not the
  place to grow API surface for a rung nothing may depend on.
* **No offer SEARCH.** Query building, GPU aliasing and the policy tiers are
  `market.offers`; this module never builds a search.

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 3 (`market/`) of `docs/plans/vast-tooling-refactor-v2.md`. Every symbol
carries its `# moved-from:` marker. Step 3 is ADD-ONLY, so `herdd.py` keeps
its own copies until step 6 — `bid_echo_probe.py` still reaches
`herdd._market_min_bid_read` by attribute and must keep working.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import ladder_core

from vastlib.core import api, models

import bidpolicy

# The self-floor guard's echo window and the seam resets live in `ladder_core`
# (2026-08-14, FLEET_REVIEW item 1) — ONE copy of the state transitions both
# supervise lanes used to hand-twin. Re-exported into THIS module's namespace on
# purpose, exactly as `herdd.py` re-exports them and for the same reason: the
# consumers (fleetd, bid_echo_probe, workflowctl) and the suite both address
# these by attribute, and the ALIAS IDENTITY is what keeps one implementation.
# See AUTOBID_DESIGN.md §"One core, two lanes".
#
# The annotations are the only thing added, and they are what lets mypy's
# strict `disallow_untyped_calls` see through an untyped Zone-adjacent leaf at
# every call site without a per-call ignore (same trick, same reason, as
# `core/models.py`'s `_num_dph`). `Callable[..., Any]` rather than a written-out
# signature because these are aliases, not a re-declaration of the contract —
# `ladder_core` remains the place that states it.
# moved-from: herdd.BID_HISTORY_MAX
BID_HISTORY_MAX: int = ladder_core.BID_HISTORY_MAX
# moved-from: herdd._note_standing_bid
_note_standing_bid: Callable[..., Any] = ladder_core.note_standing_bid
# moved-from: herdd._hist_field
_hist_field: Callable[..., Any] = ladder_core.hist_field
# moved-from: herdd._bid_history_for
_bid_history_for: Callable[..., Any] = ladder_core.bid_history_for
# moved-from: herdd._self_floor_reset
_self_floor_reset: Callable[..., Any] = ladder_core.self_floor_reset


# moved-from: herdd._auto_bid_price
def _auto_bid_price(min_bid: object, on_demand: object = None) -> float | None:
    """The LAUNCH bid $/hr. Identical by construction to the steady-state
    defend/decay target — SPOT_DESIGN §3.2's "launch price == steady-state target"
    invariant (P2, 2026-07-18: a launch price one $0.001 grid step above the
    target made the first supervised poll decay the fresh bid back down on ~9% of
    floors). None when the floor is unknown or unwinnable.

    Since 2026-08-08 that invariant is enforced by DELEGATION rather than by a
    comment: see `bidpolicy._bid_target` for the four rails (preference, cost cap,
    survival cushion, hard on-demand/max-bid clamps). Keeping the cushion here and
    only here would have been useless anyway — the decay ladder would have walked
    a freshly cushioned launch bid straight back down to the old target within
    BID_DECAY_POLLS ticks.

    Razor-thin floors (the M4-T2 launch stall — NOT the autobid audit's "D7",
    which is the blind `outbid` classifier; the collision is a numbering
    accident, see AUTOBID_AUDIT_2026-08-08 §D7): when floor and on-demand sit within
    BID_ONDEMAND_EPS of each other, the clamp would emit a bid BELOW the floor —
    vast underbid-parks the box at launch and it never schedules (the M4-T2
    fixed-underbid stall, now machine-made). `_bid_target` raises back to the floor
    while that still bids strictly under on-demand; else None (an unwinnable offer
    — callers skip it / require --price)."""
    mb = models._num_dph(min_bid)
    if mb is None:
        return None
    # ONE implementation, not a second copy (2026-08-08). The two used to be
    # hand-kept in sync — "EXACTLY _bid_target's arithmetic" was a comment, not a
    # call — and the audit's cushion rail is exactly the kind of change that would
    # have landed in one and not the other. `max_bid=None` here because a launch
    # has no standing ceiling yet; the caller's own rails (--max-bid, the
    # replacement `max_dph`) still bind downstream.
    #
    # Called by ATTRIBUTE on the Zone S module (never `from bidpolicy import
    # _bid_target`), so a test that patches `bidpolicy._bid_target` still steers
    # this call. The ignore is the price of that: an unannotated Zone S def is an
    # untyped call under mypy strict, and annotating it here would mean a second
    # statement of a contract bidpolicy already owns.
    return bidpolicy._bid_target(mb, None, models._num_dph(on_demand))  # type: ignore[no-any-return,no-untyped-call]


# THE ONE HOME for the sticky clamp reference, ported 2026-08-16 (plan §8 step 6
# leftovers). Both supervise lanes call it — the jobs lane twice per tick
# (`job_lane` :834 announce, :1306 ladder) and the run lane once — which is why
# `supervise/job_lane.py` declared it a raising seam rather than porting a
# second copy: two clamps that can drift is the defect, not the duplication.
# `ladder_core.box_swap_reset(..., reset_sticky_on_demand=…)` clears the memo it
# writes, and that flag's one `False` caller (the run-lane handoff) is one of the
# six pinned lane divergences — so the KEY NAME `on_demand_last` is a contract
# between this function and `ladder_core`, not a local detail.
# moved-from: herdd._sticky_on_demand
def _sticky_on_demand(st: MutableMapping[str, Any],
                      fresh: float | None) -> float | None:
    """Remember the last on-demand price we successfully read for this box, and
    hand it back when the live probe fails (autobid audit 2026-08-08).

    A failed probe used to mean `on_demand=None`, which silently DISABLES the
    on-demand clamp in `_bid_target` AND drops `_default_max_bid` onto the
    3.0x-median-floor fallback. The two together let the defend ladder walk a
    standing bid straight past the machine's on-demand price. Four boxes are on
    record having done exactly that, with the on-demand price the very next
    `bid_over_preferred_ceiling` event recorded:

        44962074  bid $0.1501  on-demand $0.1067   (1.41x)
        44965461  bid $0.1501  on-demand $0.1067   (1.41x)
        46177923  bid $0.2278  on-demand $0.1520   (1.50x)
        47018759  bid $1.2100  on-demand $1.1707   (1.03x)  <- the v11 chat arm,
                                                               then ondemand_displaced

    Every one of those is pure waste: on-demand outranks every interruptible bid
    at any price, so a bid above on-demand buys nothing a bid just under it does
    not already buy.

    Staleness is the right trade here and not a compromise. The on-demand LIST
    price is the stable half of the market — it is precisely why the ceiling is
    anchored to it rather than to the floor (the 2026-07-12 J1 anti-ratchet) —
    while the FLOOR is the thing that spikes. A minutes-old on-demand number is a
    far better clamp than no clamp; the value is scoped to one box's state, so it
    cannot leak across machines, and a single successful probe replaces it.

    NOT applied to the launch/handoff paths on purpose: there the doctrine is
    already "no clamp beats a wrong one" (`_offer_ondemand_ref`), because a fresh
    box has no history to be stale about — the candidate machine may never have
    been probed at all."""
    if fresh and fresh > 0:
        st["on_demand_last"] = fresh
        return fresh
    # `st` is the live supervise context (`jc` / the run-lane state dict), typed
    # `MutableMapping[str, Any]` for the persistence reasons `job_lane`'s header
    # states, so this `.get` is an `Any` widened to the declared `float | None`.
    # Deliberately NOT narrowed: a cast or an `isinstance` guard here would be a
    # BEHAVIOR change — whatever the ladder wrote under `on_demand_last` is what
    # must come back, and inventing a `None` for a value we did store is exactly
    # the unclamping this function exists to prevent.
    return st.get("on_demand_last")


# Cap on per-machine on-demand market probes in one _handoff_pick_offer pass
# (one soft bundles POST per distinct machine; offers arrive min_bid-ascending
# so the qualifier is normally within the first few).
# moved-from: herdd.HANDOFF_ODPROBE_MAX
HANDOFF_ODPROBE_MAX = 8

# Same cap for the RUN lane's eviction relaunch (2026-08-08): offers arrive
# min_bid-ascending, so the affordable one is normally the first, and each probe
# is one soft POST per distinct machine.
# moved-from: herdd.RELAUNCH_ODPROBE_MAX
RELAUNCH_ODPROBE_MAX = 5


# moved-from: herdd._offer_ondemand_ref
def _offer_ondemand_ref(offer: object, num_gpus: int | None = None) -> float | None:
    """The ON-DEMAND reference price for a BID-view offer row — for the
    `_auto_bid_price` clamp and the §2.3 handoff candidate filter. NEVER the
    offer's own `dph_total`: on a bid-view bundles row that field is the
    CURRENT INTERRUPTIBLE price (min_bid + the storage sliver — API-verified
    2026-08-06: min_bid 0.2667 / dph_total 0.2711 on a machine whose on-demand
    view lists dph_total 0.5111), and bid-view `dph_base` equals min_bid, so
    NOTHING on the bid row carries the on-demand rate. Using dph_total as the
    clamp lands the bid a rounding unit over its own floor — the razor-thin
    bids that lost understudy 46909754 45 minutes after launch ($1.071 over a
    $1.0667 floor) and understudy 46934673 ($0.401, 2026-08-06). This is the
    doc 50 R1 defect; the eviction-replacement path was fixed 2026-08-05, this
    helper closes the launch and handoff paths with the same source:
    `_market_ondemand_soft` (dph_base off the machine's on-demand offers).

    None = market unreadable. That DISABLES the clamp in `_auto_bid_price`
    (full 1.20x cushion, still bounded by max_bid/ceiling rails) and makes the
    §2.3 candidate filter refuse (missing input) — no clamp beats a wrong one."""
    if not isinstance(offer, dict):
        return None
    return _market_ondemand_soft(offer.get("machine_id"),
                                 num_gpus or offer.get("num_gpus"))


# moved-from: herdd._offer_pricing_soft
def _offer_pricing_soft(offer_id: object,
                        typ: str = "bid") -> tuple[float | None, float | None, Any]:
    """Soft (min_bid, dph_total, machine_id) for a single pinned --offer id: one
    POST v0/bundles/ filtered by id.

    MEASURED DEAD, 2026-08-09 (task #72). The `id` filter returns HTTP 200 with
    ZERO rows in **every** view — bid and ondemand alike — for offers that are
    live and rentable in the same query without it. The GET-by-id endpoints
    (`v0/bundles/<id>`, `v0/asks/<id>`, `v0/offers/<id>`) are 404. Worse, chunk
    ids RESHUFFLE between two identical unfiltered queries, so an id pinned a
    minute ago may name nothing at all by the time it is used.

    So this function's real contract is "(None, None, None), almost always". It
    is kept as a rung rather than deleted because it costs one soft POST and the
    filter may come back; nothing may DEPEND on it resolving. The rungs that do
    work are `_offer_machine_scan_soft` (id matched in Python over an unfiltered
    query) and, given a machine, `_market_min_bid_soft` + `_market_ondemand_soft`
    — which is why `--offer-machine` exists.

    The older docstring here claimed the bid view was REQUIRED for a bid floor
    while the ondemand view still resolved a machine_id. Both halves described a
    world where the id filter answers at all."""
    try:
        oid = int(offer_id)                   # type: ignore[call-overload]
    except (TypeError, ValueError):
        oid = offer_id
    q = {"limit": 1, "type": typ, "rentable": {"eq": True}, "id": {"in": [oid]}}
    ok, d, _ = api.request_soft("POST", "v0/bundles/", q, retries=2)
    if not ok or not isinstance(d, dict):
        return None, None, None
    offers = d.get("offers") or []
    if not offers:
        return None, None, None
    o = offers[0]
    return (models._num_dph(o.get("min_bid")), models._num_dph(o.get("dph_total")),
            o.get("machine_id"))


# moved-from: herdd._machine_offers_soft
def _machine_offers_soft(machine_id: object) -> list[models.MachineRow] | None:
    """Current rentable offers on one machine, or None on any failure. Each
    offer carries the GPU-config count (`g`), the on-demand GPU rate (`base` =
    dph_base) AND the spot floor (`bid` = min_bid) — so a single no-`type`
    bundles POST answers reserved price, spot price, AND capacity at once."""
    if not machine_id:
        return None
    q = {"limit": 64, "rentable": {"eq": True},
         "machine_id": {"in": [machine_id]}, "order": [["dph_total", "asc"]]}
    ok, d, err = api.request_soft("POST", "v0/bundles/", q, retries=2)
    if not ok or not isinstance(d, dict):
        return None
    offers: list[models.MachineRow] = []
    for o in d.get("offers", []):
        g = o.get("num_gpus")
        if g is None:
            continue
        offers.append({"g": int(g), "base": models._num_dph(o.get("dph_base")),
                       "bid": models._num_dph(o.get("min_bid"))})
    return offers


# moved-from: herdd._market_min_bid_read
def _market_min_bid_read(machine_id: object,
                         num_gpus: int | None = None) -> models.MarketRead:
    """`_market_min_bid_soft` with its evidence intact — returns a `MarketRead`.

    `ok=False` is IGNORANCE (no machine_id, HTTP failure, unparseable body) and
    must never advance eviction state (SPOT_DESIGN §5 rule 1: transient !=
    eviction). `ok=True, listed=False` is EVIDENCE: vast answered, and the
    machine we are supposedly renting a chunk of lists no rentable bid offer at
    all. On 2026-08-08 that transition — floor $1.315789 at 23:02:19, nothing at
    23:03:12 — was the entire visible trace of box 47214941's displacement, and
    it was being discarded."""
    if not machine_id:
        return models.MarketRead(False, False, None)
    q = {"limit": 64, "type": "bid", "rentable": {"eq": True},
         "machine_id": {"in": [machine_id]}, "order": [["min_bid", "asc"]]}
    ok, d, err = api.request_soft("POST", "v0/bundles/", q, retries=2)
    if not ok or not isinstance(d, dict):
        return models.MarketRead(False, False, None)
    offers = [o for o in d.get("offers", [])
              if models._num_dph(o.get("min_bid")) is not None]
    if not offers:
        return models.MarketRead(True, False, None)
    floors, scaled = _market_chunk_floors(offers, num_gpus)
    return models.MarketRead(True, True, min(floors), floors=floors, scaled=scaled)


# moved-from: herdd._market_bid_listed_soft
def _market_bid_listed_soft(machine_id: object,
                            num_gpus: int | None = None) -> bool | None:
    """Tri-state: does this machine still list ANY rentable bid offer?

      True  — yes (so it is still purchasable and we were not displaced off it)
      False — vast ANSWERED and it lists none: positive evidence of displacement
      None  — the read failed, or there was no machine_id. Ignorance.

    The eviction classifier's missing observable (defect D7). Deliberately a
    SEPARATE probe from the per-tick floor read rather than a widened return
    contract on it: this question is only asked on an eviction tick, and
    `_market_min_bid_soft`'s two-state contract is depended on by the decay path,
    where a mis-read floor has already underbid-parked a live box once
    (defect D5, handoff-canary-3 2026-07-15)."""
    r = _market_min_bid_read(machine_id, num_gpus)
    return r.listed if r.ok else None


# moved-from: herdd._market_chunk_floors
def _market_chunk_floors(offers: Sequence[Mapping[str, Any]],
                         num_gpus: int | None = None) -> tuple[list[float], bool]:
    """PURE. `(floors, scaled)`: EVERY candidate per-instance floor a machine's
    bid offers imply for OUR GPU-count chunk, plus whether the number had to be
    synthesized by per-GPU rescale (no exact-count offer listed).

    Keeping the rows apart matters (review 2026-08-10, F3): on a machine we are
    a tenant of, one query returns both our rented chunk (min_bid = the echo of
    our own bid) and any free sibling chunk (a genuine floor). A min() collapse
    can hide a genuine sibling floor that rose ABOVE our bid behind our own
    lower echo — the self-floor guards therefore filter rows, not the scalar.

    TYPING NOTE (the trap, not a defect): every `_num_dph` below is `float |
    None` to a type checker and is arithmetic here with no None guard. That is
    safe ONLY because the sole caller, `_market_min_bid_read`, pre-filters the
    rows to non-None `min_bid`. The ignores mark the dependency; do NOT
    "harden" this helper to swallow None — that would silently turn an
    unreadable row into a floor."""
    g = None
    try:
        g = int(num_gpus) if num_gpus else None
    except (TypeError, ValueError):
        g = None
    if g:
        exact = [models._num_dph(o["min_bid"]) for o in offers
                 if (o.get("num_gpus") or 0) == g]
        if exact:
            return sorted(exact), False       # type: ignore[arg-type]
        per_gpu = [models._num_dph(o["min_bid"]) / o["num_gpus"] for o in offers
                   if (o.get("num_gpus") or 0) > 0]
        if per_gpu:
            return [round(min(per_gpu) * g, 4)], True
    return [min(models._num_dph(o["min_bid"]) for o in offers)], False  # type: ignore[type-var,list-item]


# moved-from: herdd._market_chunk_floor
def _market_chunk_floor(offers: Sequence[Mapping[str, Any]],
                        num_gpus: int | None = None) -> float:
    """PURE. The collapsed single floor (min of `_market_chunk_floors`) — the
    legacy two-state contract for callers that want one number.

    ZERO callers repo-wide as of the port (2026-08-16) — `_market_chunk_floors`
    is read only by `_market_min_bid_read`. Ported anyway because deleting a
    documented contract is a step-6 decision with the owner, not a side effect
    of moving a file. Do not invent a use for it."""
    floors, _scaled = _market_chunk_floors(offers, num_gpus)
    return min(floors)


# moved-from: herdd._market_min_bid_soft
def _market_min_bid_soft(machine_id: object,
                         num_gpus: int | None = None) -> float | None:
    """Live market floor for our box's machine (SPOT_DESIGN §3.2 / ground-truth #5):
    the min `min_bid` across current bid offers on `machine_id` **for our GPU-count
    chunk**. The instance object has no live min_bid, so this is the one extra soft
    GET per tick that feeds defend/rescue. None on no machine_id or ANY failure — a
    failed offers read must never advance eviction state, so None simply disables
    both bid actions.

    `num_gpus` matters (defect D5, live canary handoff-canary-3 2026-07-15): a
    machine lists offers per CHUNK (1/2/4 GPUs), and min_bid is per-chunk while
    our standing bid + the PUT price are whole-instance. The old bare min() read
    the 1-GPU chunk floor ($0.1333) for a 2-GPU instance (real floor $0.2667),
    so decay "corrected" the bid to $0.16 and vast instantly underbid-parked the
    box. Exact-chunk match first; else scale the best per-GPU floor by our count;
    `num_gpus=None` keeps the min-across-chunks read (launch-time probes).

    CAVEAT the callers must know about (incident 2026-08-08, task #73): on a
    chunk we are ALREADY THE TENANT OF, the listed `min_bid` is the price to
    displace the current tenant — us. So this can hand back our own last PUT
    dressed as "the market", and multiplying it is a ~10%/poll (now 100%/poll)
    ratchet toward `max_bid`. `bidpolicy.market_floor_is_self` is the guard, and
    BOTH lanes now apply it, tenant-gated: the jobs ladder inline in
    `job_supervise_tick`, the run lane in `_self_floor_guard` (called from
    `_observe`, so the suppression reaches defend AND `floor_samples`)."""
    return _market_min_bid_read(machine_id, num_gpus).min_bid


# moved-from: herdd._market_ondemand_soft
def _market_ondemand_soft(machine_id: object,
                          num_gpus: int | None = None) -> float | None:
    """Live ON-DEMAND price (dph_base) for our box's machine + GPU config, the
    ceiling anchor (AUTOBID_DESIGN). A RUNNING bid instance's own `dph_total` is
    the PAID bid (see _do_bid_move), NOT on-demand — so read it from the market:
    `_machine_offers_soft` returns `base`=dph_base (on-demand) per offer. Pick the
    offer matching this box's GPU count (else the smallest covering it), mirroring
    `_rates`. None on no machine_id / read failure / no base — which disables the
    on-demand clamp and falls the ceiling back to the median-floor path."""
    if not machine_id:
        return None
    offers = _machine_offers_soft(machine_id)
    if not offers:
        return None
    bases = None
    if num_gpus:
        match = next((o for o in offers if o.get("g") == num_gpus), None)
        if match is None:
            cover = [o for o in offers if o.get("g", 0) >= num_gpus]
            match = min(cover, key=lambda o: o["g"]) if cover else None
        if match and match.get("base") is not None:
            bases = match["base"]
    if bases is None:                                 # no GPU-count match -> min base seen
        cand = [o["base"] for o in offers if o.get("base") is not None]
        bases = min(cand) if cand else None           # type: ignore[type-var]
    return bases

