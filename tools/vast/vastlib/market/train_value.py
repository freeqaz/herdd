"""vastlib.market.train_value — expected TRAINING tokens per dollar for an offer.

The ranking half of `herdd search --job`: given a job's `env:`/`assets:` and a
page of live offers, it asks `train_rates` what that exact training shape has
been MEASURED to do on each card class and turns the answer into a sort key.

Why here and not in `offers.py`
-------------------------------
`offers.py` answers "which offers exist" and says so in its own header ("No
`_gpu_rate_soft`" — tokens-per-dollar ranking is not part of the market query).
This module is that ranking, kept separate for the same reason `cpu_value` and
`build_search_query` are different surfaces: a filter narrows the board, a value
orders it, and only the filter may ever change what we pay for.

Three rules it inherits, all load-bearing
-----------------------------------------
* **Unmeasured is an unknown box, not a bad one.** A card with no anchor in the
  job's family gets `None`, ranks below every measured row, and is never dropped
  and never scored zero — a zero is a claim (`offers.py:946`, `cpu_value`).
* **A ranking signal never touches a price we pay.** Nothing here writes a bid,
  a filter or a query; it only orders rows that were already returned.
* **`train_rates` is optional.** It is imported lazily, exactly as `offers.py`
  imports the calibration blob, so a checkout without it — or a broken anchor
  file — degrades this whole module to "everything is unmeasured", i.e. today's
  price order, rather than breaking `search`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple

from vastlib.core import models

if TYPE_CHECKING:                         # Zone S flat leaf; see `_rates()`
    from train_rates import Family, RateEstimate

# The tier is the OUTER sort term, so a measured reading is never displaced by a
# faster provisional one. `provisional` is a stale-stack anchor and staleness is
# one-directional here (every lever landed made training faster), so it is a
# FLOOR on that card — usable to rank, not to compare against a measured peer.
TIER_RANK = {"measured": 2, "provisional": 1}
TIER_SRC = {"measured": "meas", "provisional": "prov"}

_RATES: ModuleType | None = None
_RATES_LOADED = False


def _rates() -> ModuleType | None:
    """`train_rates`, or None. Cached; never raises.

    Lazy for `offers._calibration_blob`'s reason: `vastlib` must not need a Zone
    S leaf to import, and an absent one reads as "nothing measured" — a state
    every consumer here already handles.
    """
    global _RATES, _RATES_LOADED
    if not _RATES_LOADED:
        _RATES_LOADED = True
        try:
            import train_rates  # noqa: PLC0415

            _RATES = train_rates
        except Exception:                                     # noqa: BLE001
            _RATES = None
    return _RATES


def family_for(env: Mapping[str, Any] | None, assets: object = None,
               *, world_size: int = 1) -> Family | None:
    """The job's `Family` at a given card count, or None.

    `world_size` is a parameter and not a constant because `eff_batch` includes
    it: ranking a 4-card offer with a 1-card family names a job nobody ran.
    """
    mod = _rates()
    if mod is None:
        return None
    fam: Family | None = mod.family_from_env(dict(env or {}),
                                             world_size=max(1, int(world_size)),
                                             assets=assets)
    return fam


def offer_num_gpus(offer: Mapping[str, Any] | None) -> int:
    n = models._num_dph((offer or {}).get("num_gpus"))
    return int(n) if n and n >= 1 else 1


def offer_gpu_ram_gb(offer: Mapping[str, Any] | None) -> float | None:
    """The offer's PER-CARD framebuffer in GB, as `fmt_offer` prints it.

    None when the field is missing — which must stay "unknown", not "0 GB": a
    zero would make every operating point fail the fit filter and silently turn
    a measured card unmeasured.
    """
    mib = models._num_dph((offer or {}).get("gpu_ram"))
    return mib / 1024.0 if mib and mib > 0 else None


def offer_dph(offer: Mapping[str, Any] | None) -> float | None:
    """The price the row already displays. `dph_total` on BOTH rental types —
    in bid mode that is the current interruptible price, which is what a winning
    bid pays (`offers.build_search_query` documents this at length)."""
    return models._num_dph((offer or {}).get("dph_total"))


class OfferValue(NamedTuple):
    """One offer with its rate verdict. `est`/`tok_per_dollar` are None for an
    UNMEASURED cell — the two are always None together, and neither is ever 0."""

    offer: dict[str, Any]
    family: Family | None
    est: RateEstimate | None
    tok_per_dollar: float | None
    dph: float | None

    @property
    def src(self) -> str:
        return TIER_SRC.get(self.est.tier, "?") if self.est else "-"


def value_for(offer: Mapping[str, Any], env: Mapping[str, Any] | None,
              assets: object = None) -> OfferValue:
    """Price this one offer for this one job. Never raises."""
    row = dict(offer or {})
    mod = _rates()
    dph = offer_dph(row)
    ws = offer_num_gpus(row)
    fam = family_for(env, assets, world_size=ws)
    if mod is None or fam is None:
        return OfferValue(row, fam, None, None, dph)
    try:
        est: RateEstimate | None = mod.rate_for_offer(
            fam, str(row.get("gpu_name") or ""), ws, offer_gpu_ram_gb(row))
        tpd = mod.tokens_per_dollar(est, dph) if (est and dph) else None
    except Exception:                                         # noqa: BLE001
        return OfferValue(row, fam, None, None, dph)
    return OfferValue(row, fam, est, tpd, dph)


def rank_key(v: OfferValue) -> tuple[int, float]:
    """Sort key for `sorted(..., reverse=True)` — the CPU lane's exact shape.

    A measured tok/$ and "no measurement at all" are not the same unit, so they
    cannot share a sort slot: the tier separates them into groups and the second
    term only ever orders WITHIN a group. For the unmeasured group that term is
    the negated price, which under `reverse=True` puts the cheapest unknown box
    first and an offer with no readable price last (rather than as free).
    """
    if v.est is not None and v.tok_per_dollar is not None:
        return (TIER_RANK.get(v.est.tier, 1), v.tok_per_dollar)
    return (0, -v.dph if v.dph and v.dph > 0 else float("-inf"))


def rank_offers(offers: Iterable[Mapping[str, Any]],
                env: Mapping[str, Any] | None,
                assets: object = None) -> list[OfferValue]:
    """Every offer, priced and ordered. Same length as the input, always —
    ranking is not filtering, and an unmeasured card is never dropped."""
    return sorted((value_for(o, env, assets) for o in offers),
                  key=rank_key, reverse=True)


def unmeasured_cells(rows: Sequence[OfferValue]) -> list[tuple[str, int]]:
    """The distinct `(card, count)` cells nothing has measured for this job, in
    the order they appear — one probe line each is useful, one per offer is noise.

    Keyed on the COUNT as well as the name because a family carries `world_size`
    (`eff_batch` includes it): a 1-card A100 cell can be measured while the
    2-card one is not, and naming just the card would report a measured cell as
    unmeasured.
    """
    out: list[tuple[str, int]] = []
    for v in rows:
        cell = (str(v.offer.get("gpu_name") or "?"), offer_num_gpus(v.offer))
        if v.est is None and cell not in out:
            out.append(cell)
    return out


def probe_hint(family: Family | None, gpu_name: str) -> str | None:
    """What to run to turn an unmeasured cell into an anchor, or None."""
    mod = _rates()
    if mod is None or family is None:
        return None
    try:
        hint: str = mod.probe_hint(family, gpu_name)
    except Exception:                                         # noqa: BLE001
        return None
    return hint


def human_tokens(n: float | None) -> str:
    """`20.1M`. A tok/$ figure spans six orders of magnitude across the board,
    so a raw float is unreadable and an exponent is worse."""
    if n is None:
        return "?"
    x = float(n)
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.1f}{unit}"
    return f"{x:.0f}"
