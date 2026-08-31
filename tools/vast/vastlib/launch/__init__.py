"""vastlib.launch — turning an intent to rent into a running instance.

Why this layer exists
---------------------
`_do_launch` is the single most consequential path in the tool: it mints
credentials, splits secrets out of the launch env, picks an offer, chooses a
bid price, creates the instance, and then watches the boot. Today that is one
function doing five jobs with module-global `_MINTED_*` state threaded through
it. Separating the *spec* (pure, testable, no network) from the *drive* (the
five-phase effectful sequence) is what makes the launch path reviewable.

Planned modules (plan §5)
-------------------------
  spec.py    `_build_launch_spec`, the secrets split, the B2/R2 key mint. The
             `_MINTED_*` module globals become a `MintLedger` instance owned by
             the launch context — mutable process state becomes an object with
             a lifetime, which is what makes the mint auditable.
  launch.py  `_do_launch` decomposed into preflight -> offer pick -> bid price
             -> create -> post-launch watch.

What is deliberately NOT here
-----------------------------
* No offer search and no price arithmetic of its own — `market.offers` and
  `market.pricing` are the source, called down into. A second copy of the
  price floor here is exactly the twin-implementation defect the refactor is
  meant to remove.
* No supervision. What happens to the box *after* the post-launch watch hands
  off — eviction replacement, rebid ladders, boot SLA — is `supervise/`.
* No credential storage. Minting is here; the broker (`credbroker*`) is
  deliberately NOT absorbed into `vastlib` (box-side coupling, audited
  separately, plan §3).

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 3 as verbatim moves from `herdd.py`.
"""

from __future__ import annotations
