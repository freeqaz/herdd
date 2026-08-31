"""vastlib.core — the shared kernel: everything the rest of the package stands on.

Why this layer exists
---------------------
The fan-in measurement that shaped the split found the kernel by counting
inbound calls: config/HTTP (112), instance lookup and labels (53), the render
atoms (46). Those are the names that every other cluster reaches for, and they
are also the ones that made `herdd.py` impossible to import piecemeal. Pull
them out first and every other module below becomes independently testable.

Planned modules (plan §5)
-------------------------
  result.py   `Soft[T]` — a NamedTuple('ok', 'data', 'err'), tuple-unpack
              compatible with all 39 existing `*_soft` call shapes, plus
              `ok()`/`err()` helpers. The raising/soft pairs collapse to one
              implementation.
  config.py   `vastconf` absorbed verbatim: `.env` discovery, `herdd.yaml`
              merge order, the `_boot_knob` resolver (CLI > env > yaml >
              constant), `allow_local_gpu`, the disk defaults. The ~70 stray
              `os.environ.get` sites route through it name-preserving, at the
              same precedence.
  api.py      `VastClient`: `request` / `request_soft` / `_classify_http`,
              retry and rate-limit. Module-level default-client functions are
              kept deliberately, because that is the shape the 59 existing
              `request_soft` patch sites bind to. The only `urllib` importer
              in the package besides `boxes.remote`'s result poller.
  models.py   pydantic v2 at the API boundary: `Offer`, `Instance`,
              `MachineRow`, `MarketRead`, `BidState`. The accessors (`_num_dph`,
              `_disk_gb`, `_storage_day`, `_label_value`) become properties.
              Raw-dict passthrough is preserved via `model_extra` so unknown
              vast.ai fields survive a round trip.
  labels.py   the label grammar: `run:` / `serve:` / `keep:<why>|until`, and
              reap-keep parsing (`test_label_grammar.py` retargets here).
  fmt.py      pure render atoms: `dollars`, `_age_str`, `_money`, `_fmt_toks`,
              `_visw`, `_hms_secs`, `Pal`, `Progress`.

What is deliberately NOT here
-----------------------------
* Nothing that performs a *fleet* action. `core` reads and formats; deciding
  to stop, destroy, bid or launch a box belongs one ring up.
* No policy. `bidpolicy.py` (Zone S) stays the sole decision layer for bids and
  `ladder_core.py`'s state transitions land in `market.pricing`, not here.
* No import of any sibling ring. `core` may import stdlib, pydantic, and Zone S
  flat modules by bare name — nothing else. This is the one rule that keeps the
  DAG acyclic, and import-linter fails the build if it is broken.

Provenance: skeleton created 2026-08-16, plan §8 step 1. Contents arrive in
step 2 (`core/` + `supervise/journal.py`) as verbatim moves from `herdd.py`
and `vastconf.py`, each carrying its `# moved-from:` marker.
"""

from __future__ import annotations
