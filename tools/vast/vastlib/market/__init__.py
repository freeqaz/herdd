"""vastlib.market — what a box costs and which one to rent: offers and pricing.

Why this layer exists
---------------------
Offer search is one of the clusters the mapping found cleanly separable today:
it reads the market, normalizes GPU names, applies the policy tiers and the
inet floors, and returns a choice. Pricing is its twin — on-demand references,
min-bid reads, the chunk floors, and the glue that drives the Zone S bid
policy. Neither needs to know that boxes have lifecycles, so neither should be
able to reach the code that does.

Planned modules (plan §5)
-------------------------
  offers.py   `build_search_query`, `search_offers`, `pick_cheapest_offer`,
              GPU-name normalization, the policy tiers, the inet floors.
  pricing.py  on-demand references, min-bid reads, the C17 chunk floors, and
              the bid-ladder glue to `bidpolicy` (Zone S) and `ladder_core`.

What is deliberately NOT here
-----------------------------
* **No decisions.** `bidpolicy.py` stays the pure decision layer and
  `ladder_core.py` stays the one copy of the two lanes' state transitions.
  This package reads their verdicts and performs the I/O around them; a rule
  that lives in both places is the exact defect class those leaves exist to
  kill.
* No bid *placement*. `_put_bid_soft` and friends are box mutation and live in
  `boxes.lifecycle`; the 24 patch sites on that name repoint there, not here.
* No launch orchestration — picking an offer is not creating an instance.
  `launch/` composes this ring; this ring never composes `launch/`.

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 3 (the separable rings) as verbatim moves from `herdd.py`.
"""

from __future__ import annotations
