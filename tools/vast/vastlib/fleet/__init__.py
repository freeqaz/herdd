"""vastlib.fleet — fleetd, fully decomposed: the daemon, its seam, and its state.

Why this layer exists
---------------------
`fleetd.py` (4,176 lines) holds 49 live `herdd.<attr>` references across 33
distinct names — and 21 of the 49 sit inside `class Hooks` (165 lines), its
documented sole I/O seam. The rest are pure predicates and constants
(`LIVE_STATES`, `_num_dph`, `_disk_gb`, the guard verdict sets). A typed API
client, a parsed instance model and one guard enum absorb roughly 90% of that
coupling, which is why full decomposition is affordable in the same effort
(owner decision, plan §0.4). Verified during the mapping: fleetd re-implements
no policy — only verdict *presentation* and cost arithmetic are thinly
duplicated, and both collapse into `core` and `boxes.health`.

Planned modules (plan §5)
-------------------------
  client.py  `fleet_request`, socket and state paths, watch helpers — the CLI
             side of the daemon protocol. Both `cli` and `daemon` import this,
             so the protocol constants live HERE and the direction is no
             longer inverted (today the daemon's protocol lives in the CLI).
  hooks.py   `Protocol FleetHooks`, typed from the existing `Hooks` class;
             the default implementation binds `vastlib`.
  rows.py    the pure row builders (ceiling / retention / stray / reconcile),
             verbatim, typed.
  state.py   `state.json` and journal persistence. **The schema is FROZEN**
             (hard constraint, plan §0.4): it is round-trip tested against a
             scrubbed live snapshot captured in step 0, and a deploy that
             cold-re-adopts instead of loading is a failed cutover.
  daemon.py  the tick loop, the unix-socket `Server`, and `main`.
  deploy.py  the 441-line release-checkout deploy block, generalized to
             "deploy a script from an audited checkout as a systemd user
             unit" — reused by plan §9 to give the reaper the same treatment
             and retire the every-15-min live-tree execution hazard.

What is deliberately NOT here
-----------------------------
* **No policy.** The daemon decides *when* to look; what a verdict means is
  `boxes.health.GuardVerdict`, what a box costs is `core.models`, and what to
  do about an eviction is `supervise/`. fleetd's job is scheduling and
  persistence.
* No systemd unit text for anything but its own deploy path, and no live-fleet
  side effects at import time — `conftest.py` redirects `FLEETD_SOCK` because
  the test suite once emitted real destroy-intents into the running daemon.
  That guard is a frozen contract (plan §4) and this module must keep it
  bitable: its target has to EXIST for the fixture to be non-vacuous.
* No `.env` key renames, ever, on this branch: the live daemon's `.env` is a
  SYMLINK to the repo `.env` and is hot-reloaded, so a rename lands on a
  running daemon.

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 5 as a decomposition of `fleetd.py` — which is untouched until then.
"""

from __future__ import annotations
