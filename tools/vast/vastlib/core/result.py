"""vastlib.core.result — the return-shape vocabulary of the `*_soft` family.

Why this exists
---------------
`herdd.py` carries 40 top-level `*_soft` definitions (39 distinct names —
`_put_label_soft` is defined twice, the earlier def being dead). "Soft" is a
contract, not a type: *this function never raises; it reports failure in its
return value*. What it is NOT, today, is one shape. An AST sweep of every
`return` in all 40 (plan §8 step 2 mapping pass) found roughly ten distinct
shapes, of which only **4 of 39** are the canonical `(ok, data, err)` triple
the plan names. The rest disagree on arity (2 vs 3), on slot order (`(ok, err)`
vs `(value, err)`), on what slot 2 *means* (an error string in one function, a
payload in the next), and five families carry no result channel at all.

This module gives each of those shapes a name, so that the later port steps can
assign a type mechanically instead of re-deriving the contract from the body,
and so that the shapes that are positionally identical but semantically
inverted stop being a coin flip for the reader.

Everything here is a `typing.NamedTuple`. That is the whole design:
`NamedTuple` subclasses `tuple`, so `Soft(True, x, None) == (True, x, None)`,
`ok, data, err = soft_fn()` keeps working, and `soft_fn()[0]` keeps working
(`_put_bid_soft(iid, target)[0]` is a live call site). Adopting one of these
types at a port site is therefore a pure annotation change with no call-site
churn — that compatibility is the point, and `test_vastlib_core_result.py`
asserts it. A dataclass would have broken integer indexing and `==` against
the bare tuples the tests assert on (`_put_state_soft(...) == (True, None)`).

The in-tree precedent is `herdd.MarketRead` — a `collections.namedtuple`
whose `__new__.__defaults__` lets 3-positional construction survive after the
result was widened to 5 fields. This module generalizes that trick's intent.

The shape taxonomy (measured, not assumed)
------------------------------------------
Read: shape → the type in this module → the `herdd.py` functions that return
it. Later port steps annotate from this table.

  A  `(ok, data, err)`            → `Soft`        4 fns
     `request_soft`, `_vast_execute_soft`, `_vast_copy_direct_soft`,
     `_ssh_exec_soft`.
     `err` is None on success. On failure `data` is None for `request_soft`
     but `""` for the other three — preserve that asymmetry (see `err()`).

  B  `(ok, err)`                  → `OkErr`       4 names / 5 defs
     `_put_state_soft`, `_put_label_soft` (both defs), `_put_bid_soft`,
     `_destroy_soft`. Also the three suffix-less soft-by-contract primitives
     `stop_box` / `destroy_box` / `set_bid`, which a name-based inventory
     misses and the workflow controller depends on.
     The PUT trio demote vast's HTTP-200-with-`{"success": false}` into
     `(False, str(msg))`, so `err` can hold API prose, not just transport
     text. `_destroy_soft` returns `(True, None)` for the 404/already-gone
     and dry-run cases, so ok=True does not imply an action was taken.

  C  `(ok, payload)`              → `OkData`      1 fn
     `_wait_states_soft` → `(True, st)` / `(False, last_status)`.
     **Positionally identical to B, semantically inverted** — see the traps.

  D  `(value, err)`, DATA FIRST   → `ValueErr`    1 fn
     `_api_key_soft` → `(k, None)` / `(None, "config: …")`. The mirror image
     of B. Its `"config:"` prefix is a typed channel, not free text (traps).

  E  `(rc, stdout, stderr)`       → `ProcResult`  1 fn (+1 closure)
     `_rclone_soft` → `(r.returncode, r.stdout, r.stderr)` /
     `(127, "", "rclone not found on PATH")`; plus the `_runner` closure
     inside `_read_run_soft`. `rc == 0` is the success test, and `stderr` is
     populated on successful runs too. 18 seam patches ride on this arity.

  F  bare optional value          → no type here  13 fns — the largest class
     `_offer_machine_scan_soft` (dict|None), `_offer_cuda_soft` (float|None),
     `_machine_offers_soft` (list[dict]|None), `_get_instance_soft`
     (dict|None), `_jobd_status_line_soft` (str|None), `_scratch_probe_soft`
     (dict|None), `_status_marker_soft` (str|None), `_market_min_bid_soft`
     (float|None), `_market_ondemand_soft` (float|None),
     `_handoff_synced_epoch_soft` (int|None), `_gpu_rate_soft` (float|None),
     `_jobd_status_soft` (str|None), `_jobd_heartbeat_epoch_soft`
     (float|None). These collapse "read failed" and "answered, nothing there"
     into one None — deliberately in some, a known defect in others (G).

  G  tri-state `bool | None`      → no type here  2 fns
     `_jobd_status_pyhalf_soft` (its two None arms are indistinguishable ON
     PURPOSE — FAILCLOSED_DESIGN §3) and `_market_bid_listed_soft`, which is
     the explicit repair of F's ignorance/evidence conflation (defect D7,
     AUTOBID_AUDIT_2026-08-08 §4/§6): True = listed, False = vast ANSWERED
     and lists nothing (positive displacement evidence), None = read failed.

  H  bare `bool`, no err channel  → no type here  3 fns
     `attach_ssh_key_soft` (False on every failure, incl. a swallowed
     exception), `_serve_self_park_soft` (False on every read/parse failure —
     "fail toward RESCUE"), `_stop_instance_soft` (prints the error and
     returns only ok).

  I  empty container on failure   → no type here  7 fns
     `_instances_soft` / `_raw_events_soft` / `_raw_job_events_soft` /
     `_search_offers_soft` (`[]`), `_read_spec_soft` (`{}`),
     `_box_lifecycle_soft` (a defaulted dict), `_read_run_soft` (a view dict
     that tags its own failure IN-BAND via `_cache_stale` / `_read_error` /
     forced `status="unknown"` — the one function carrying error state inside
     the data). Failure is indistinguishable from an empty success.

  J  positional data 3-tuple, NO ok slot → no type here  2 fns
     `_offer_pricing_soft` → `(min_bid, dph_total, machine_id)` /
     `(None, None, None)`; `_serve_status_line_soft` → `(token, epoch_ts,
     detail)` / `(None, None, None)`. These LOOK like shape A and are not:
     slot 0 is a value, not an ok flag.

  K  pure side effect, returns None → no type here  1 fn
     `_emit_launched_soft`. Here `_soft` means only "never raises".

  4 + 4 + 1 + 1 + 1 + 13 + 2 + 3 + 7 + 2 + 1 = 39 distinct names.

Two semantic traps (the ones that will silently mislabel a port)
----------------------------------------------------------------
1. **`_ssh_exec_soft`'s ok=True is transport-only.** vast never surfaces the
   remote command's exit code, so ok=True means "ssh connected and ran
   something"; the remote command may have failed, and its *stderr is inside
   `data`* (`(r.stdout or "") + (r.stderr or "")`). Same class of lie in
   `_vast_copy_direct_soft`, where ok=True means "vast accepted the request"
   (fire-and-forget) and the data slot holds a human MESSAGE string
   (`str(data.get("msg") or "copy initiated")`), not a payload.
2. **`_wait_states_soft`'s slot 2 is DATA, not an error.** It returns
   `(True, st)` / `(False, last_status)` — a status string in both arms.
   Typing it `OkErr` would relabel a perfectly good status as an error
   string, which is why `OkData` exists as a separate name for an identical
   positional shape.

What is deliberately NOT here
-----------------------------
* **No `*_soft` function.** This module is types + helpers only, with zero
  imports from any other vastlib module — `core.result` is the bottom of the
  bottom ring. Each soft function moves with its home module in a later step
  (api, boxes.lifecycle, boxes.remote, storage.b2, …) and picks its type up
  from the table above.
* **No per-function payload types** for shapes F/G/H/I/J. `_offer_pricing_soft`'s
  `(min_bid, dph_total, machine_id)` and `_serve_status_line_soft`'s
  `(token, epoch_ts, detail)` are domain shapes, not result shapes; they get
  NamedTuples in their *home* modules (`market.pricing`, `boxes.health`) where
  the field names mean something. Naming them here would put market vocabulary
  in the kernel.
* **No invariant assertions.** An `assert err is None if ok` would be the
  obvious thing to add and it would be WRONG: `fleetd.Hooks.park/resume/
  destroy` return `(True, "dry-run")` on the dry-run path — err-slot-non-None
  on success, mixed into the same channel as real shape-B returns. These
  types describe the tree that exists.
* **No typed error object.** `err` stays a `str | None` because
  `herdd._classify_http` accepts the err STRING and parses `HTTP (\\d{3})`
  out of it, `_destroy_soft` and `_confirm_gone` test `"HTTP 404" in err`,
  `_start_busy` regex-searches it, and a `"config:"` prefix routes to "fatal".
  The err slot is a semi-structured, load-bearing channel; boxing it is a
  behavior change and out of scope for a behavior-preserving port.
* **No `Generic[T]`.** `class Soft(NamedTuple, Generic[T])` is a TypeError on
  Python 3.10 (generic NamedTuple arrived in 3.11) and the repo floor is 3.10
  (`ruff.toml target-version`, `mypy.ini python_version`). The concrete form
  below is the documented fallback: `data` is `Any`, and a call site that
  knows better narrows it at the seam.

Provenance: new in the vastlib package, plan §8 step 2 (`core/`). Nothing here
is a move, so nothing here carries a `# moved-from:` marker (vastlib/README.md
§2 rule 7): these are new names for shapes that existed only as bare tuple
literals. The shape table above is the mapping the rename table cannot express.
"""

from __future__ import annotations

from typing import Any, NamedTuple

__all__ = [
    "OkData",
    "OkErr",
    "ProcResult",
    "Soft",
    "ValueErr",
    "err",
    "ok",
]


class Soft(NamedTuple):
    """Shape A — the canonical `(ok, data, err)` triple.

    `request_soft`, `_vast_execute_soft`, `_vast_copy_direct_soft`,
    `_ssh_exec_soft`. On success `err` is None; on failure `data` is None
    (`request_soft`) or `""` (the exec/copy/ssh trio) — never assume one.

    ok=True does not always mean the *remote* work succeeded; see the module
    docstring's trap 1 before believing it for the ssh/copy pair.
    """

    ok: bool
    #: The payload on success. Untyped on purpose: across the four shape-A
    #: functions this is a parsed JSON dict, a raw JSON string, command output,
    #: or a human "copy initiated" message. Narrow it at the call site.
    data: Any
    #: None on success. On failure a semi-structured, string-matched channel:
    #: "HTTP 404 …", "network …", "error …", "config: …". See _classify_http.
    err: str | None

    def as_pair(self) -> OkErr:
        """Drop the payload — the shape-B view of a shape-A result.

        For call sites that only ever asked "did it work, and if not why".
        Equal to the bare `(ok, err)` tuple, so it can be returned straight
        out of a shape-B function that delegates to a shape-A one.
        """
        return OkErr(self.ok, self.err)

    def value_or(self, default: Any = None) -> Any:  # noqa: ANN401 — see `data`
        """The payload on success, `default` otherwise — the shape-F view.

        The lossy direction of the taxonomy, and the reason shape F exists:
        it collapses "the read failed" and "there was nothing there" into one
        value. Only use it where the caller genuinely cannot act on the
        difference.
        """
        return self.data if self.ok else default


class OkErr(NamedTuple):
    """Shape B — `(ok, err)`; slot 2 is an ERROR, None on success.

    `_put_state_soft`, `_put_label_soft`, `_put_bid_soft`, `_destroy_soft`,
    and the suffix-less `stop_box` / `destroy_box` / `set_bid`.

    `err` may carry vast's own prose (the PUT trio demote an HTTP 200 with
    `{"success": false}` into `(False, str(msg))`), and it is NOT guaranteed
    None when ok is True — fleetd's dry-run hooks return `(True, "dry-run")`.
    """

    ok: bool
    err: str | None


class OkData(NamedTuple):
    """Shape C — `(ok, payload)`; slot 2 is DATA, not an error.

    `_wait_states_soft` only: `(True, st)` on reaching a wanted state,
    `(False, last_status)` on timeout. Positionally identical to `OkErr`,
    which is exactly why it has its own name — see the module docstring's
    trap 2.
    """

    ok: bool
    #: A payload in BOTH arms (a vast instance status string, here).
    data: Any


class ValueErr(NamedTuple):
    """Shape D — `(value, err)`, the mirror of `OkErr`: the DATA comes first.

    `_api_key_soft` only: `(k, None)` / `(None, "config: VASTAI_API_KEY not
    set (env or .env)")`. There is no ok flag — `value is None` is the failure
    test.

    The `"config:"` prefix on `err` is load-bearing, not cosmetic:
    `_classify_http` string-matches it to return "fatal", which is what stops
    a missing key from being retried forever as a transient blip.
    """

    value: Any
    err: str | None


class ProcResult(NamedTuple):
    """Shape E — the POSIX `(rc, stdout, stderr)` triple, no ok slot.

    `_rclone_soft` (`(127, "", "rclone not found on PATH")` when the binary is
    absent) and the `_runner` closure inside `_read_run_soft`.

    `rc == 0` is the success test — NOT truthiness, which is inverted here.
    `stderr` is routinely non-empty on a successful run (rclone's transfer
    stats go there), so an `if stderr:` failure test is a bug.
    """

    rc: int
    stdout: str
    stderr: str


def ok(data: Any = None) -> Soft:  # noqa: ANN401 — mirrors Soft.data
    """Build a successful shape-A result: `(True, data, None)`.

    `err` is None on success everywhere in shapes A and B, and is never `""`.
    The empty string appears only as a *data*-slot failure value, so this
    helper never puts one in the err slot.
    """
    return Soft(True, data, None)


def err(message: str, data: Any = None) -> Soft:  # noqa: ANN401 — mirrors Soft.data
    """Build a failed shape-A result: `(False, data, message)`.

    `data` defaults to None, matching `request_soft`. The exec/copy/ssh trio
    return `""` in the data slot on failure and must pass `data=""`
    explicitly: both are falsy, so `if not data` guards agree, but anything
    that calls a string method on the payload does not.

    `message` is the semi-structured err channel — keep the existing prefixes
    ("HTTP <code> …", "network …", "error …", "config: …"), because
    `_classify_http` and three call sites match on them.
    """
    return Soft(False, data, message)
