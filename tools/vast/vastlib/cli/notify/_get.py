"""The one authed GET the `notify` group is built out of.

Why this module exists
----------------------
All three `notify` subcommands are the same call with a different path, and
`inbox` additionally needs the exit code that says the HIDDEN endpoint is gone.
Both are group-private, so `cli-surface.json` proposes `cli/notify/__init__.py`
as their home — but that file has to import the three command modules to build
the parser, and those modules have to import these two symbols, which is a
package import cycle. Splitting the two shared symbols into a leaf the command
modules import instead breaks it with no re-export games and no
partially-initialized-module ordering to get right.

What is deliberately NOT here
-----------------------------
* **Any write.** Nothing under `notify` writes, and in particular nothing ever
  PUTs `seen_through_at` (NOTIFY_DESIGN D3: our cursor is ours, and marking the
  feed seen would both be useless for dedup and step on a console UI that may
  want it later).
* The renderers and the path constants. They are `tools/vast/notify.py`, the
  flat leaf that `fleetd`'s poll tick shares — one parser for both readers.
* HTTP. `core.api.request_soft` owns the transport, the auth header and the
  `(ok, data, err)` taxonomy; this is eight lines of shape adaptation on top,
  turning that triple into the `(payload, err)` the three commands read.

Provenance: moved from `tools/vast/herdd.py::_notify_get` and
`::NOTIFY_GONE_RC`, plan §8 step 6 (cli-surface.json H2 — the fourth
dispatcher).
"""

from __future__ import annotations

from typing import Any

from vastlib.core import api

#: `notify inbox` exit code when the HIDDEN inbox endpoint answers 404. Distinct
#: from the generic error exit on purpose: this endpoint is commented out of
#: vast's published OpenAPI spec, so its disappearance is an EXPECTED end state
#: that a caller should be able to test for, not a bug to page on.
# moved-from: herdd.NOTIFY_GONE_RC
NOTIFY_GONE_RC = 3


# moved-from: herdd._notify_get
def _notify_get(path: str) -> tuple[Any, Any]:  # noqa: ANN401 — server payload / error string
    """One authed GET against the notification API. Returns (payload, err);
    payload is None on failure. Read-only by construction — nothing under
    `notify` writes, and in particular nothing ever PUTs `seen_through_at`
    (NOTIFY_DESIGN D3: our cursor is ours, and marking the feed seen would both
    be useless for dedup and step on a console UI that may want it later)."""
    ok, data, err = api.request_soft("GET", path)
    return (data if ok else None), err
