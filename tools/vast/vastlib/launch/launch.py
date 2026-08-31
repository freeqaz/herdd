"""vastlib.launch.launch — the five-phase create sequence: intent in, box out.

Why this exists
---------------
`_do_launch` is the most consequential path in the tool — it is where money
starts being spent — and it is the one function every other launcher delegates
to (`cmd_launch`, `cmd_train`, the workflow arm launcher) precisely so there is
ONE assembly of the vast create body. It runs five phases in a fixed order:

  1. preflight       image gate, `--eval-env-ver` validation. FAIL-CLOSED and
                     FIRST: nothing network-touching may precede it.
  2. offer pick      pinned `--offer` (machine scan, cuda floor) or a search.
  3. bid price       the auto-bid ladder, then the second half — the None
                     re-check and the >= on-demand waste warning, which land
                     much later, at body assembly.
  4. env + body      env assembled with `setdefault` throughout, onstart, HF
                     token, ssh inject, `--jobs`, image digest stamp.
  5. create + watch  dry-run early return, preflight, PUT, runmeta, fleet
                     watch, ssh attach, boot health.

Ported MONOLITHIC on purpose (plan §7.4). The phase boundaries above are real
and named in plan §5, but every one of them reads or mutates the shared
`argparse.Namespace` and the `dph`/`on_demand`/`entry_floor`/`offer_id` locals,
and phase 3 is textually interleaved with phase 2 and then resumes inside phase
4. A decomposition is admissible only when each helper is a verbatim slice AND
stdout order, exit codes and Namespace mutations are bit-identical; splitting
this port and proving that at the same time is two changes, so it is one.

Three properties that look like accidents and are not
-----------------------------------------------------
* **`a` IS MUTATED IN PLACE.** `a.image` (the resolved image) and `a.price`
  (the auto-derived bid) are written back onto the caller's Namespace, and
  `cmd_train` READS `a.price` back after the call. Threading copies through a
  refactor would break the caller silently.
* **`sys.exit` lives inside this library function**, at eight sites, and
  `test_lifecycle.py` / `test_launch_eval_env_pin.py` assert on the message
  TEXT. Raising a typed error instead is a behavior change, not a cleanup.
* **The prologue's ORDER is load-bearing and tested.** `test_lifecycle.py`
  asserts that a launch refused by the image/`--eval-env-ver` gate never
  searched for an offer. Hoisting the offer pick passes type-check and fails
  that test for exactly the right reason.

What is deliberately NOT here
-----------------------------
* **No `cmd_launch`.** The four-line CLI wrapper becomes `cli/launch.py`'s
  `run(a)` at plan §8 step 6; keeping it out of this module is what makes that
  a move rather than a second entry point. (`test_dash_cache.py`'s banned-name
  list resolves `_do_launch`/`cmd_launch` against `herdd` and must be
  retargeted when they leave it, or it goes green against an empty surface —
  plan §7.3's target-exists meta-test covers that, not this file.)
* **No offer search, no price arithmetic, no B2 mint of its own.**
  `market.offers`, `market.pricing` and `launch.spec` own those; this module
  calls down into them BY MODULE ATTRIBUTE (plan §8b) so the monkeypatch idiom
  survives the port.
* **No supervision.** What happens to the box after the boot watch — eviction
  replacement, rebid ladders, relaunch — is `supervise/`.
* **`workflowctl.py`:915 and :1034 hand-replicate this function's ssh + hf
  prelude** ("build ssh + hf exactly like _do_launch"). `workflows/` is
  absorbed at plan §8 step 5; that agent should call down here instead of
  re-diverging a third copy.

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 3 (`launch/`) of `docs/plans/vast-tooling-refactor-v2.md`. Every symbol
carries its `# moved-from:` marker. Step 3 is ADD-ONLY, so `herdd.py` keeps
its own copies until step 6 and both are live meanwhile — the deferred seam
sites listed in `.port_manifests/launch.json` still patch `herdd` and still
steer the flat copy, which is correct until their CALLERS move.

Step 4 amendment: three of this module's ten raising seams are now REBOUND to
`boxes.lifecycle` — `_launch_preflight`, `launch_instance`,
`_emit_launched_soft`. That closes the money path (`_do_launch` can now reach a
real `PUT v0/asks/`), so the offline lane depends on conftest's mutation guard
and on `launch_instance` calling `api.request_soft` by MODULE ATTRIBUTE. Seven
seams still raise; the banner below names what blocks each one.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import disksize
import imageref

from vastlib.boxes import health, lifecycle, ssh
from vastlib.core import config, fmt, models
from vastlib.launch import spec
from vastlib.market import offers, pricing

import bidpolicy

# --------------------------------------------------------------------------- #
# CROSS-RING / NOT-YET-PORTED SEAM — new code, no `moved-from:` marker
# (README §2 rule 7).
#
# `_do_launch`'s verbatim body calls ten names whose definitions were not in
# `vastlib` when this module landed. THREE OF THEM ARE NOW REBOUND (plan §8
# step 4) and SEVEN still raise:
#
#   REBOUND — `_launch_preflight`, `launch_instance`, `_emit_launched_soft`
#     now live in `boxes.lifecycle` (orchestrator ruling at rev a1f2c8a5:
#     `market.json`'s "-> launch/ and cli/" pointer is overruled,
#     `launch.json`'s "-> boxes/" pointer is confirmed for the first two and
#     extended to the third; the reasoning is written down in that module's
#     CREATE-half section banner). They are bound BELOW by module-level
#     assignment. Those three are exactly the names that gate and perform the
#     `PUT v0/asks/` — the money path through this file is now real code, and
#     `launch_instance` reaches `api.request_soft` by module attribute, so
#     conftest's mutation guard still sees it.
#   STILL RAISING (2, ring above) — `compose_jobs_launch_env` (jobs/bundle.py)
#     and `fleet_watch_best_effort` (fleet/client.py) live in rings ABOVE this
#     one. BOTH DEFINITIONS NOW EXIST (plan §8 step 5 landed `jobs/bundle.py`
#     and `fleet/client.py`) and BOTH SEAMS STAY RAISING ANYWAY: the import
#     DIRECTION is the blocker, not the timing. `launch` may not import `jobs`
#     or `fleet` at module scope or inside a function — import-linter reads the
#     AST — so unlike the three rebinds above, these cannot be closed here at
#     all. `cli/_compose.py::bind()` closes them, and it is called when a
#     COMMAND RUNS rather than when `cli` is imported, so the census this
#     module publishes does not change under a test file's import order — read
#     that module's docstring before patching either name in a test. Their
#     stubs carry
#     the REAL signatures (`compose_jobs_launch_env`'s was corrected when
#     `jobs/bundle.py` landed; see its docstring) so a binding that lands the
#     real function cannot silently narrow the call.
#   REBOUND (5, plan §8 step 6) — `_require_image`, `image_login_arg`,
#     `_mask_image_login`, `hf_token_text` and `hf_login_snippet` now live in
#     `launch/spec.py`, one module over in the same ring: they are the
#     credentials a box is handed plus the gate in front of them, which is that
#     module's subject (`cli-surface.json` H3 — four commands reach them, so no
#     `cli/<command>.py` could own them; its `core/config.py` proposal for the
#     two `hf_*` helpers was overruled because a credential MINT below every
#     importer is the wrong direction). They no longer wait on the `imageref`
#     absorption: `image_login_arg` calls `registry.mint_token` directly and
#     nothing in the five needs `imageref` at all. Bound BELOW by module-level
#     assignment, same as the three above.
#
# WHY ASSIGNMENT AND NOT A DELEGATING `def` (decided once, here): a module-level
# `name = lifecycle.name` keeps `monkeypatch.setattr(launch, "<name>", …)`
# working, which `test_vastlib_launch.py::_wire` and
# `test_dry_run_returns_without_launching` both depend on. The cost is the
# other direction: a patch of `lifecycle.<name>` is NOT seen through `launch`,
# because the binding is captured at import. Tests that mean to steer
# `_do_launch` patch `launch`; tests that mean to characterize the function
# patch `lifecycle`. A delegating def would make both work and would also put a
# second frame on the money path — not worth it for a rebind that step 6
# deletes anyway.
#
# The seven that remain RAISE rather than no-op. A silent stub would let a test
# drive `_do_launch` to a green "launched" that never resolved an image and
# never PUT anything — the vacuous-pass failure mode plan §7.3 exists to kill.
# Nothing in `vastlib` is wired to a CLI before step 6, so the raise costs no
# live path.
# --------------------------------------------------------------------------- #

_SEAM_HINT = ("not ported yet — rebind this module attribute, or stub it in "
              "your test with monkeypatch.setattr")


def compose_jobs_launch_env(env: dict[str, str], onstart: str | None, *,
                            dry_run: bool = False,
                            key_base: str | None = None,
                            no_idle_park: bool = False,
                            idle_park_grace: object = None,
                            no_job_deadline: object = None,
                            timeout_s: float | None = None,
                            bootstrap_stager: Callable[..., str] | None = None,
                            ) -> tuple[str, str]:
    """SEAM for `herdd.compose_jobs_launch_env` -> `vastlib.jobs.bundle`
    (plan §8 step 5 — the DEFINITION landed there; this stays a seam).

    The ONE `--jobs` launch-body builder, shared with
    `workflowctl.build_box_resolver` so the manual CLI path and the workflow
    path compose an IDENTICAL jobs box — including the unique per-launch mint
    nonce (the 2026-07-12 box-44566398 no-revoke discipline). Mutates `env` in
    place and returns `(onstart, sha)`.

    WHY THIS ONE DOES NOT GET THE `= bundle.compose_jobs_launch_env` REBIND the
    three `boxes.lifecycle` names above got: `jobs` sits in the ring ABOVE
    `launch`, so `vastlib.launch` may never import `vastlib.jobs` — at module
    scope OR inside a function, since import-linter reads the AST and a deferred
    import does not dodge it. The direction is the problem, not the timing, and
    it does not improve when `jobs/bundle.py` exists (it does, as of step 5).
    The `cli/` composition root — which may import both rings — binds this name
    at startup at step 6; that is the same injectable-default idiom
    `workflowctl.build_box_resolver` already uses for `jobs_composer`.

    THE SIGNATURE IS THE REAL ONE, corrected 2026-08-16 when `jobs/bundle.py`
    landed. The stub as first written declared only
    `(env, onstart, *, dry_run, no_idle_park, idle_park_grace, no_job_deadline)`
    and DROPPED `key_base`, `timeout_s` and `bootstrap_stager` — all three of
    which live callers pass (`workflowctl.py:903` injects `bootstrap_stager`;
    `timeout_s` is what sizes the minted key's lifetime; `key_base` is the
    per-launch mint nonce). A composition root that bound the real function
    behind the narrow stub would have type-checked clean and broken the workflow
    lane at runtime.
    """
    raise NotImplementedError(f"compose_jobs_launch_env: {_SEAM_HINT} (plan §8 step 5 "
                              f"— defined in vastlib.jobs.bundle, an upper ring; "
                              f"bound by the cli composition root at step 6)")


# --------------------------------------------------------------------------- #
# REBOUND at plan §8 step 4 — the three that gate and perform the PUT. Their
# real bodies are in `boxes.lifecycle`; these are the module attributes
# `_do_launch` calls and tests patch. See the banner above for why this is an
# assignment and not a delegating def, and note there is no `moved-from:`
# marker on a rebind: the marker lives on the definition, in `lifecycle.py`.
#
# Refuses to launch a second box for a run that already has a vast instance
# labelled `run:<ID>` (a live twin is a double-writer; a stopped twin still
# bills disk and vast may restart it). Last gate before the PUT, and it
# `sys.exit`s — the caller of `_do_launch` inherits that exit.
_launch_preflight = lifecycle._launch_preflight

# THE MONEY MOVE. `PUT v0/asks/<offer>/` with a prepared body; returns
# `(ok, contract_id, err)` and never sys.exits. A bid launch allocates a real,
# BILLABLE contract while the bid is still pending, so any response carrying
# `new_contract` is a rented box. The body reaches `api.request_soft` by module
# attribute, which is what keeps conftest's mutation guard in front of it.
launch_instance = lifecycle.launch_instance

# Best-effort runmeta `launched` event for ANY `run:`-labelled box — the
# recording half of the dashboard's empty-column problem. Never raises.
_emit_launched_soft = lifecycle._emit_launched_soft

# REBOUND at plan §8 step 6 — the image gate and the four credential helpers.
# Definitions (and their `moved-from:` markers) live in `launch/spec.py`; these
# are the module attributes `_do_launch` calls and `test_vastlib_launch.py`
# patches. `_require_image` `sys.exit`s when no image resolved — deliberately no
# stock fallback — and `_mask_image_login` is the only thing that makes
# `image_login_arg`'s result printable, so the pair is bound together.
_require_image = spec._require_image
canonical_default_image = spec.canonical_default_image
image_pin_verdict = spec.image_pin_verdict
hf_token_text = spec.hf_token_text
hf_login_snippet = spec.hf_login_snippet
image_login_arg = spec.image_login_arg
_mask_image_login = spec._mask_image_login


def fleet_watch_best_effort(target: object, profile: str = "bare",
                            budget_usd: float | None = None,
                            policy: Mapping[str, Any] | None = None) -> bool:
    """SEAM for `herdd.fleet_watch_best_effort` -> `vastlib.fleet.client`
    (plan §8 step 5).

    B1b: registers a freshly launched box with the daemon so the fleet is never
    in a launch->watch gap. Never fatal — no daemon just means the safety net
    adopts the box on a later tick.

    THE SIGNATURE IS THE REAL ONE, corrected 2026-08-16 when `cli/_compose.py`
    started binding this name — the same correction `compose_jobs_launch_env`
    got, in the opposite direction. As first written the stub declared
    `budget_usd: object`, `policy: object` and `-> Any`, all WIDER than
    `fleet.client`'s `float | None` / `Mapping[str, Any] | None` / `bool`, so the
    binding assignment was the first thing that ever compared the two and mypy
    rejected it. A wide stub is not harmless: it type-checks call sites that the
    real function would refuse, which is a runtime failure deferred to whoever
    wires the seam.
    """
    raise NotImplementedError(f"fleet_watch_best_effort: {_SEAM_HINT} (plan §8 step 5 "
                              f"— blocked on vastlib.fleet.client, an upper ring)")


# --------------------------------------------------------------------------- #
# The launch itself.
# --------------------------------------------------------------------------- #

# moved-from: herdd._do_launch
def _do_launch(a: argparse.Namespace) -> tuple[Any, Any, Any]:  # noqa: ANN401 — (cid, offer_id, dph), all vast-shaped
    """Core of `launch`: pick offer, assemble body, launch. Returns
    (cid, offer_id, dph) — dph is the paid rate (bid price, or the picked
    offer's on-demand dph_total); offer_id/dph come from the picked offer or the
    explicit --offer/--price. On --dry-run prints the body and returns
    (None, offer_id, None). Extracted so cmd_train can reuse the exact launch
    path and read offer_id/dph back for its runmeta `launched` event without
    scraping stdout; cmd_launch is a thin wrapper with byte-identical CLI output."""
    # FAIL CLOSED on the image before ANY work — offer search, key mint, B2
    # staging. There is no stock fallback (see _require_image), and refusing
    # here costs nothing, whereas refusing after the POST costs a rented box.
    a.image = _require_image(getattr(a, "image", None), "launch")

    # And fail closed on a STALE pin for the same reason, in the same place:
    # renting the wrong image is only discovered once results off it are
    # already banked. Soft when the canonical pin is unreadable, silent when
    # --image was named explicitly. See spec.image_pin_verdict.
    # The local default is re-read from config, not taken from cli/launch:
    # `cli` is a ring ABOVE this one and importing it here is illegal. Same
    # value, same source (`herdd.yaml`), no ring violation.
    _local_pin = config.load_herdd_config().get("default_image")
    _stale_pin = image_pin_verdict(a.image, _local_pin,
                                   canonical_default_image())
    if _stale_pin:
        sys.exit(f"error: {_stale_pin}")

    # --eval-env-ver: SUGAR over --env, not a second mechanism, and validated
    # HERE for the same reason the image is — before any offer search or key
    # mint, where refusing is free. The flag existed only on `train`, but
    # `launch --eval-env-ver <v>` is what docs 97/98, eval-env/bake.sh and
    # jobmeta's own refusal text all tell an operator to run, and it exited 2
    # `unrecognized arguments` — so the one documented way to pin a jobs box
    # was a command that does not parse. (`train` reaches here with this unset
    # and EVAL_ENV_VER already folded into `env=`, so it is inert there.)
    _eev = getattr(a, "eval_env_ver", None)
    if _eev is not None:
        _eev = str(_eev).strip()
        if not _eev:
            sys.exit("error: --eval-env-ver was given an empty value. An "
                     "unpinned box resolves eval-env/LATEST at boot, which can "
                     "be older than the env you preflighted — refusing rather "
                     "than launching a box that silently picks its own.")
        for _kv in (a.env or []):
            _k, _s, _v = str(_kv).partition("=")
            if _s and _k == "EVAL_ENV_VER" and _v != _eev:
                sys.exit(f"error: --eval-env-ver {_eev} contradicts "
                         f"--env EVAL_ENV_VER={_v}. One box, one baked env: "
                         f"say it once.")

    # pick an offer: explicit --offer, else cheapest from the search filters
    dph: float | None = None
    on_demand: float | None = None   # offer's on-demand list price (for the bid clamp)
    entry_floor: float | None = None  # the PRE-RENT market floor — the last
                                     # uncontaminated read this box will ever give
                                     # us (post-rent its min_bid reflects our own
                                     # bid back, #73). Stamped into the box env
                                     # below for the defense controller.
    if a.offer:
        offer_id = a.offer
        # A pinned --offer resolves NOTHING through vast's `id` filter (measured
        # 2026-08-09, task #72: HTTP 200 / zero rows in every view; the
        # GET-by-id endpoints 404; chunk ids reshuffle between two identical
        # queries). So the row, if we get one at all, comes from a SCAN — an
        # unfiltered query matched on the id in Python — and the reliable
        # pricing input is a MACHINE, which `--offer-machine` lets the operator
        # hand us directly. One scan, reused by both the cuda gate and the bid
        # pricing below.
        _row = offers._offer_machine_scan_soft(a) if (
            getattr(a, "cuda", 0) or (a.type == "bid" and a.price is None)) else None
        _picked_row = _row
        _machine_pin = getattr(a, "offer_machine", None)
        if a.type != "bid" and _row is not None and dph is None:
            # Adjacent open item, now free: a pinned ON-DEMAND launch recorded
            # its `launched` event with dph=None, because the search path is the
            # only place `dph` was ever read off an offer. The scan row is an
            # ondemand-view row here (`build_search_query` carries `a.type`), so
            # its `dph_total` is the rate actually billed.
            dph = models._num_dph(_row.get("dph_total"))
        # A pinned --offer skips the search, so the cuda_max_good floor must be
        # enforced HERE or a pinned host silently dodges it — the exact class that
        # killed waves A/C 2026-07-30 (a stock vllm-0.24 torch-cu130 venv on a 12.9
        # driver, "NVIDIA driver ... too old" after boot). The floor itself is
        # config.LAUNCH_CUDA_MAX_GOOD, which tracks the IMAGE's CUDA runtime.
        # --cuda 0 disables.
        _floor = getattr(a, "cuda", 0) or 0
        if _floor:
            _cm = offers._offer_cuda_soft(a.offer, row=_row)
            if _cm is None:
                print(f">> warning: cuda_max_good UNVERIFIABLE for pinned offer "
                      f"{a.offer} (floor {_floor:g}) — vast's offer `id` filter "
                      f"returns no rows in any view and the id was not in the "
                      f"scanned listing either; the on-box ensure_cuda_init probe "
                      f"is the remaining gate. `--machine <ID>` searches instead of "
                      f"pinning and keeps the filter enforced server-side.",
                      file=sys.stderr)
            elif _cm < _floor:
                sys.exit(f"error: pinned offer {a.offer} has cuda_max_good {_cm:g} < "
                         f"required floor {_floor:g} (the image's CUDA runtime — "
                         f"Error-804 risk). "
                         f"Pick another host, or pass --cuda {_cm:g} (or --cuda 0) to "
                         f"override for a lane that truly doesn't need it.")
        # fetch its floor+on-demand so bid pricing
        # stays automatic (no more "--type bid requires --price" — AUTOBID_DESIGN)
        if a.type == "bid" and a.price is None:
            # Pinned-offer autobid ladder (owner directive 2026-08-03: agents
            # never hand-price — "pass --price" on a pinned offer violated the
            # autobid design). Rungs, cheapest-and-most-specific first; only when
            # ALL of them fail do we error, naming them.
            tried: list[str] = []
            # TYPING-FORCED SPLIT of the verbatim `mb = machine_id = None`: the
            # two names take different types below and a chained assignment
            # cannot carry two annotations. Same values, same order.
            mb: float | None = None
            machine_id: Any = None
            if _machine_pin:
                tried.append(f"--offer-machine {_machine_pin}")
                machine_id = _machine_pin
            if _row is not None:
                tried.append("offer row (machine scan)")
                mb = models._num_dph(_row.get("min_bid"))
                machine_id = machine_id or _row.get("machine_id")
            if mb is None:
                # The id-filtered rung. Expected to return nothing — kept because
                # it costs one soft POST and the filter may come back.
                tried.append("offer row (id filter — dead, see _offer_pricing_soft)")
                mb, _bid_view_total, _mid = pricing._offer_pricing_soft(a.offer)
                machine_id = machine_id or _mid
            # CHUNK SIZE comes off the recovered row when we have one, NOT off
            # `--num-gpus` (defect D5, handoff-canary-3): a machine lists a floor
            # per 1/2/4-GPU chunk, and pricing a pinned 2-GPU chunk against the
            # 1-GPU floor is the underbid that vast parks on arrival.
            _g = (_row or {}).get("num_gpus") or getattr(a, "num_gpus", None)
            if mb is None and machine_id:
                # The two reads that are VERIFIED working: the same per-machine
                # queries the spot views (`ls`, defend/rescue) price from.
                tried.append(f"market floor (machine {machine_id}, {_g}x GPU)")
                mb = pricing._market_min_bid_soft(machine_id, _g)
            if on_demand is None and machine_id:
                on_demand = pricing._market_ondemand_soft(machine_id, _g)
            entry_floor = mb
            a.price = pricing._auto_bid_price(mb, on_demand)
            if a.price is not None:
                print(f"       (auto bid price {fmt.dollars(a.price)} = "
                      f"{bidpolicy.BID_TARGET_MULT:g}x floor {fmt.dollars(mb)}, "
                      f"capped below on-demand "
                      f"{fmt.dollars(on_demand)}; offer {a.offer}"
                      + (f", machine {machine_id}" if machine_id else "") + ")")
            else:
                sys.exit(
                    f"error: could not auto-price pinned offer {a.offer} — "
                    f"tried: {'; '.join(tried)}; none yielded a bid floor.\n"
                    f"       This is NOT evidence the offer is gone: vast's offer "
                    f"`id` filter returns zero rows for live offers in every view, "
                    f"so re-pinning changes nothing.\n"
                    f"       Working escape hatches, in order:\n"
                    f"         --offer-machine <MACHINE_ID>  keep the pin, price it "
                    f"off the machine's live market reads\n"
                    f"         --machine <MACHINE_ID>        drop the pin and SEARCH "
                    f"restricted to that machine (server-side filters stay enforced)\n"
                    f"       --price is a last resort (autobid is the design — "
                    f"agents should not hand-price).")
    else:
        # NAMESPACE-FORCED RENAME of the verbatim local `offers`: that name is
        # the `market.offers` MODULE here, and shadowing it would break the
        # module-attribute call idiom the patch sites depend on. Local only —
        # same value, same order, nothing observable changes.
        found = offers.search_offers(a)
        if not found:
            hint = (" (default GPU policy limits auto-pick to bf16-capable "
                    "cards >=32 GB, cheapest first; pass --gpu <name> or "
                    "--any-gpu to widen)" if offers._gpu_policy_tiers(a) else "")
            if getattr(a, "inet_down", None) is None and offers._inet_floor_for(a):
                hint += (f" (default inet-down floor "
                         f"{config._boot_knob('LAUNCH_INET_DOWN_MBPS'):g} Mb/s is "
                         f"active — --inet-down 0 or --any-inet to widen)")
            # NAME THE DISK BOUND, same reason the replacement lane's refusal
            # does: a market emptied by the container-disk floor looks exactly
            # like a price problem, and the operator hunts the wrong one.
            _floor = offers.container_disk_floor_gb(getattr(a, "disk", None))
            if _floor:
                hint += (f" (the search requires >= {_floor:g}G of "
                         f"container disk, from --disk — hosts advertising "
                         f"less hand back a smaller container rather than "
                         f"refusing, so this floor is not optional)")
            sys.exit("error: no offers match filters; loosen the search or "
                     "pass --offer" + hint)
        best = found[0]
        _picked_row = best
        offer_id = best["id"]
        dph = models._num_dph(best.get("dph_total"))
        entry_floor = models._num_dph(best.get("min_bid"))
        print("picked: " + fmt.fmt_offer(best))
        if a.type == "bid":
            # In a BID search `best["dph_total"]` is the interruptible price
            # (~min_bid + storage), NOT on-demand — using it as the clamp made
            # every auto-priced spot launch bid its own floor + a rounding
            # unit, the lowest-priority bid the market can hold (doc 50 R1
            # family; see _offer_ondemand_ref). Read the real on-demand rate
            # off the machine's on-demand offers; None = no clamp. Probed for
            # explicit --price too: the "price >= on-demand" escape-hatch warn
            # below needs the same reference.
            on_demand = pricing._offer_ondemand_ref(best, getattr(a, "num_gpus", None))
        if a.type == "bid" and a.price is None:
            # default bid = BID_TARGET_MULT x min_bid, clamped below on-demand
            a.price = pricing._auto_bid_price(best.get("min_bid", 0), on_demand)
            print(f"       (auto bid price {fmt.dollars(a.price)} = "
                  f"{bidpolicy.BID_TARGET_MULT:g}x floor "
                  f"{fmt.dollars(best.get('min_bid'))}, capped below "
                  f"on-demand {fmt.dollars(on_demand)})")

    env: dict[str, str] = {}
    for kv in a.env or []:
        if "=" not in kv:
            sys.exit(f"error: --env expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        env[k] = v

    # --eval-env-ver, already validated in the prologue (empty / contradicting
    # an explicit --env both refuse there, before any network work).
    if _eev:
        env["EVAL_ENV_VER"] = _eev
    for p in a.port or []:
        env[f"-p {p}:{p}"] = "1"
    if a.jupyter:
        env["-p 8080:8080"] = "1"

    # cred-broker identity (docs/plans/cred-broker-buildout.md §2.1): EVERY box
    # gets a fresh nonce — the broker later verifies it via vast's extra_env
    # readback (_instance_env), so the workstation stores nothing. Broker URL
    # ships only when configured workstation-side; TS_AUTHKEY (tailnet join) is
    # useless without a broker to talk to, so it is gated on BOTH being set.
    # setdefault throughout: an explicit --env override always wins.
    # NOTE precise no-broker invariant: a BARE launch adds only the nonce, but
    # the role lanes (--jobs below, cmd_train, job attach) also ship CRED_ROLE
    # and — when a key was minted — B2_KEY_EXPIRES_AT with NO broker gating, by
    # design: the B2-mediated lane (§2.4) needs them broker-URL-less, the nonce
    # is the only gate. Box-side consumers are guarded; near key expiry a
    # broker-less box logs a throttled refresh attempt, nothing more.
    env.setdefault("BOX_IDENTITY_NONCE", secrets.token_hex(16))
    # Pre-rent floor -> immutable box env (AUTOBID_DESIGN "Next iteration" §1).
    # The instance env reads back on any daemon after any restart
    # (_instance_env), so the jobs tick seeds jc["entry_floor"] with no state
    # handoff — the same channel the identity nonce rides. Best-effort: an
    # explicit --price pinned launch may never read a floor, and that lane
    # simply ships no ENTRY_FLOOR.
    if entry_floor is not None and entry_floor > 0:
        env.setdefault("ENTRY_FLOOR", f"{entry_floor:.4f}")
    # The `--disk` we ASKED FOR -> immutable box env, same channel and the same
    # reason as ENTRY_FLOOR. `disk_space` on the instance body is what vast
    # DELIVERED, which a host with less to give silently rounds down; the
    # supervise lanes anchor a replacement's size on this stamp so a shortfall
    # is not inherited by every hop after it (disksize.LAUNCH_DISK_ENV).
    _want_disk = offers.container_disk_floor_gb(getattr(a, "disk", None))
    if _want_disk:
        env.setdefault(disksize.LAUNCH_DISK_ENV, f"{_want_disk:g}")
    # the search floor cannot bind a pinned --offer, so check the row we hold
    _warn_disk_shortfall(_picked_row, _want_disk)
    # The ARCHITECTURE allowlist we launched under -> the same immutable box env,
    # for the same reason: which silicon the workload can run on is a statement
    # about the WORKLOAD, and the replacement lane has to read it off a box that
    # may already be gone. Absent = unconstrained, exactly as before
    # (offers.LAUNCH_CC_ALLOW_ENV).
    _want_cc = offers.parse_cc_allow(getattr(a, "cc_allow", None))
    if _want_cc:
        env.setdefault(offers.LAUNCH_CC_ALLOW_ENV,
                       ",".join(str(s) for s in _want_cc))
    # the client-side search filter cannot bind a pinned --offer either
    _warn_cc_mismatch(_picked_row, _want_cc)
    if os.environ.get("CRED_BROKER_URL"):
        env.setdefault("CRED_BROKER_URL", os.environ["CRED_BROKER_URL"])
        if os.environ.get("TS_AUTHKEY"):
            env.setdefault("TS_AUTHKEY", os.environ["TS_AUTHKEY"])

    onstart: str | None = None
    if a.onstart:
        onstart = open(a.onstart).read() if os.path.isfile(a.onstart) else a.onstart

    # HuggingFace token: land it on the box so model-weight pulls authenticate
    # (full bandwidth, gated repos). Pass it via container env (so existing
    # onstarts that read $HF_TOKEN keep working) AND prepend a secret-free
    # install snippet that also writes the token file + /etc/environment, so a
    # custom/absent onstart and later SSH sessions are covered too.
    if not a.no_hf_token:
        tok = hf_token_text(a.hf_token)
        if tok:
            env.setdefault("HF_TOKEN", tok)   # don't clobber an explicit --env HF_TOKEN
            onstart = hf_login_snippet() + (onstart or "")
            print("       (injecting HuggingFace token for full-speed weight pulls)")
        elif a.hf_token:
            sys.exit("error: --hf-token given but empty")

    # robust ssh: bake the pubkey into authorized_keys via onstart AND repair the
    # file's ownership/mode every boot (image- and host-agnostic). ON BY DEFAULT
    # since 2026-07-31 — it costs nothing and every box eventually needs a shell;
    # `--no-ssh` opts out. See ssh_authorized_keys_snippet() for why an append is
    # not enough.
    pub = ssh.pub_key_text(getattr(a, "ssh_key_file", None)) \
        if getattr(a, "ssh", True) else None
    if pub:
        onstart = ssh.with_ssh_inject(onstart, pub=pub)
    elif getattr(a, "ssh", True):
        print("!! WARN: no ssh pubkey found (~/.ssh/id_ed25519.pub, "
              "~/.ssh/id_rsa.pub) — this box will NOT be ssh-able. "
              "`ssh-keygen -t ed25519` first, or pass --no-ssh to silence.",
              file=sys.stderr)

    # provision-time jobd (--jobs): the box starts the job daemon at boot and
    # begins polling jobs/queue/<IID>/ immediately — no separate `job attach`.
    # Ship B2 creds as container env (the daemon reads them; NEVER the laptop
    # VASTAI_API_KEY — the box self-parks with vast's per-instance scoped key)
    # and stage the (over-cap) daemon files to B2 for a tiny onstart pull+exec.
    if getattr(a, "jobs", False):
        # Shared with workflowctl.build_box_resolver via compose_jobs_launch_env
        # so the manual CLI path and the workflow launch path compose an IDENTICAL
        # jobs box — the UNIQUE per-launch mint-key nonce (2026-07-12 box-44566398
        # no-revoke discipline) and every jobs env key live in the ONE helper.
        onstart, sha = compose_jobs_launch_env(
            env, onstart, dry_run=a.dry_run,
            no_idle_park=a.no_idle_park,
            idle_park_grace=a.idle_park_grace,
            no_job_deadline=a.no_job_deadline)
        print(f"       (--jobs: jobd starts at boot, polls jobs/queue/<IID>/; "
              f"idle self-park {'OFF (--no-idle-park)' if a.no_idle_park else 'ON'})")

    # stamp the image's CONTENT digest so `ls` can flag this box as stale after
    # a later env/image push moves the tag (best-effort; our registries only)
    dg = imageref.image_tag_digest(a.image)
    if dg:
        env.setdefault(imageref.IMAGE_DIGEST_ENV, dg)
        print(f"       (image digest {dg[:19]}… stamped — `ls` will flag this "
              f"box STALE-IMAGE after the next env push)")

    body: dict[str, Any] = {
        "image": a.image,
        "disk": a.disk,
        "runtype": a.runtype,
        "label": a.label or "herdd",
    }
    if env:
        body["env"] = env
    if onstart:
        obytes = len(onstart.encode("utf-8"))
        if obytes > 16384:
            print(f"!! WARN: onstart is {obytes} B > Vast's 16384 B cap — it may be "
                  f"truncated (trim --onstart, or rely on the baked image)",
                  file=sys.stderr)
        body["onstart"] = onstart
    if a.type == "bid":
        if a.price is None:
            # auto-derivation above failed (offer read returned no min_bid) — the
            # only path that still needs an explicit number
            sys.exit("error: could not auto-price the bid (offer returned no min_bid); "
                     "pass --price, or launch without --offer to auto-pick")
        # explicit --price is an escape hatch: warn (never silently clamp) if it
        # reaches on-demand, where the bid is pure waste (on-demand outranks it)
        if on_demand and on_demand > 0 and a.price >= on_demand:
            print(f"!! WARN: --price {fmt.dollars(a.price)} >= on-demand {fmt.dollars(on_demand)} "
                  f"— a bid at/above on-demand is pure waste (on-demand outranks every "
                  f"bid); consider --type ondemand", file=sys.stderr)
        body["price"] = a.price
        dph = a.price          # paid rate for cmd_train's runmeta launched event
    if a.template_id:
        body["template_id"] = a.template_id

    # private-registry pull creds (Vast can only set these at launch). Auto-derived
    # from REGISTRY_AUTH_SECRET when --image is on our registry; --login overrides; opt out
    # with --no-registry-login.
    if not a.no_registry_login:
        login = image_login_arg(a.image, a.login)
        if login:
            body["image_login"] = login
            print(f"       (attaching image_login for private pull: {_mask_image_login(login)})")
    elif a.login:
        sys.exit("error: --login and --no-registry-login are mutually exclusive")

    if a.dry_run:
        shown = dict(body)
        if "image_login" in shown:
            shown["image_login"] = _mask_image_login(shown["image_login"])
        # never dump secret VALUES (B2 keys, HF/LLM tokens) — keys stay visible
        if isinstance(shown.get("env"), dict):
            shown["env"] = {k: ("<redacted>" if spec._is_secret_env(k, v) else v)
                            for k, v in shown["env"].items()}
        print(json.dumps({"offer": offer_id, "body": shown}, indent=2)); return None, offer_id, None  # noqa: E702 — verbatim body (plan §7.4)

    _launch_preflight(body["label"], a.force)
    ok, cid, err = launch_instance(offer_id, body)
    if not ok:
        sys.exit(f"error: {err}")
    print(f"launched instance {cid}")
    # record the box in runmeta before anything else can fail (see docstring)
    _emit_launched_soft(a, body, cid, offer_id, dph)
    # ...and its machine identity, which that one skips for any box without a
    # `run:` label — i.e. the entire jobs lane. The mapping dies with the box.
    lifecycle.record_box_identity_soft(cid)
    if getattr(a, "fleet_watch", False):
        # B1b: close the launch->watch gap. `bare` = observation + alarms + no
        # money moves; upgrade with a real `fleet watch --profile ... --budget`.
        fleet_watch_best_effort(cid, "bare",
                                policy={"launched_label": body.get("label")})
    # attach ssh via API too (best-effort; onstart injection already covers it)
    if pub:
        ssh.attach_ssh_key_soft(cid, pub=pub)
    if a.wait:
        if getattr(a, "boot_health", False):
            _launch_boot_health_watch(cid, a.wait)
        else:
            lifecycle._wait(cid, target="running", timeout=a.wait)
            ssh._print_ssh(cid)
    return cid, offer_id, dph


def _warn_disk_shortfall(row: Mapping[str, Any] | None, want_gb: float) -> float | None:
    """Does the offer we are about to rent advertise the `--disk` we asked for?
    Returns the advertised GB, or None when the row is missing/unreadable.
    PURE apart from the print — no API read, so it is free and deterministic.

    A machine advertising less container disk than the request does not refuse
    the rental, it CLAMPS the allocation and boots. Nothing said so, and the
    first symptom was an asset pull dying minutes later on a box already billing
    (`insufficient_disk: 12GB free < 24GB required`, box 48006308 asked 50 and
    got 13, 2026-08-18). `build_search_query`'s floor is the prevention; this is
    the receipt, and it is the ONLY check on the `--offer`/`--machine` pin path,
    which skips that search entirely."""
    if not want_gb:
        return None
    got = models._num_dph((row or {}).get("disk_space"))
    if got is None:
        return None
    if got + 1e-9 < want_gb:
        print(f"!! WARN: this offer advertises {got:g}G of container disk but "
              f"--disk asked for {want_gb:g}G. Vast will NOT refuse — it hands "
              f"back a {got:g}G container, and anything staging more than that "
              f"dies on the box after it starts billing. Pick a host with the "
              f"space, or shrink --disk.", file=sys.stderr)
    return got


def _warn_cc_mismatch(row: Mapping[str, Any] | None,
                      allow: Sequence[int]) -> int | None:
    """Is the offer we are about to rent inside the `--cc-allow` list? Returns
    its sm level, or None when the row does not say. PURE apart from the print.

    Twin of `_warn_disk_shortfall`, and it exists for the same reason: the
    narrowing is applied to the SEARCH, and a `--offer`/`--machine` pin skips
    the search. This warns rather than refuses — the stamp is what binds the
    replacement lane, and an operator pinning a specific box has already made
    the hardware choice."""
    if not allow:
        return None
    sm = offers.offer_sm(row)
    if sm is None or sm in tuple(allow):
        return sm
    print(f"!! WARN: this offer is sm_{sm}, which is NOT in --cc-allow "
          f"({','.join(str(s) for s in allow)}). The box will still carry the "
          f"allowlist in its env, so its REPLACEMENTS will be constrained to "
          f"hardware this one is not — pick an in-list host, or fix the list.",
          file=sys.stderr)
    return sm


# moved-from: herdd._launch_boot_health_watch
def _launch_boot_health_watch(cid: int | str, wait_s: float) -> None:
    """`launch --boot-health --wait`: run boot_health_watch instead of a bare
    _wait. On 'running' print ssh (healthy). On 'slow' print the condemnation
    (host, machine_id, measured MB/s) + a destroy hint and exit NONZERO — the
    bare launch CLI stays single-shot (no auto-relaunch loop here; that lives in
    supervise/babysit/workflow). 'deadline'/'gone' also exit nonzero."""
    min_mbps = config._boot_knob("BOOT_MIN_MBPS")
    window_s = config._boot_knob("BOOT_MBPS_WINDOW_S", cast=int)
    poll_s = config._boot_knob("BOOT_HEALTH_POLL_S", cast=int)
    samp = health.BootThroughputSampler(min_mbps=min_mbps, window_s=window_s,
                                        deadline_s=wait_s, start_t=time.time())
    verdict = health.boot_health_watch(cid, min_mbps=min_mbps, window_s=window_s,
                                       poll_s=poll_s, deadline_s=wait_s,
                                       get_instance=health._get_instance_soft,
                                       sampler=samp)
    if verdict == "running":
        print(f"instance {cid} is running.")
        ssh._print_ssh(cid)
        return
    inst = health._get_instance_soft(cid) or {}
    machine_id = inst.get("machine_id")
    host = inst.get("public_ipaddr") or inst.get("host_id")
    if verdict == "slow":
        mbps = samp.last_mbps if samp.last_mbps is not None else 0.0
        print(f"!! CONDEMNED: instance {cid} (machine {machine_id}, host {host}) "
              f"pulled at {mbps:.2f} MB/s < {min_mbps:g} MB/s over {window_s}s "
              f"({samp.phase} phase) — slow host.", file=sys.stderr)
        print(f"   destroy hint: herdd destroy {cid} -y  "
              f"(then relaunch excluding machine {machine_id})", file=sys.stderr)
        sys.exit(3)
    print(f"!! instance {cid} did not reach running ({verdict}) within {wait_s}s "
          f"(machine {machine_id}, host {host}) — check: herdd show {cid}  "
          f"(destroy if wedged: herdd destroy {cid} -y)", file=sys.stderr)
    sys.exit(4)
