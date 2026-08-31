"""vastlib.core.models — typed views over the vast.ai API's payload dicts.

Why this module exists
----------------------
A vast instance body and a bundles offer row are untyped JSON, and every place
`herdd.py` reached into one directly is a place a field name, a UNIT or a
VIEW could be read wrong. Three of those readings have cost real money or a
real credential:

  * `dph_total` vs `dph_base`. On an instance the standing bid is `dph_base`;
    `dph_total` is that bid PLUS storage. Seeding `jc["last_bid"]` from
    `dph_total` put our own bid one storage-sliver above the number vast echoes
    back as `min_bid`, so the exact-equality self-floor test could not
    recognise our own bid and the ratchet defended against itself
    (2026-08-08, box 47214941). `_instance_standing_bid` is the one reader.
  * `verified` vs `verification`. The bundles SEARCH FILTER key is the boolean
    `verified`; the response spells it `verification: "verified"`. Reading
    `offer["verified"]` yields None on every row, which would have published
    "unverified" for a wholly verified result set (`_dash_verified`).
  * negative sentinels. A loading box reports `disk_usage: -1`; passed through
    it reads as "uses -1 of 120 GB", so every booting box trips an oversizing
    warning and any average built on the field is poisoned (`_disk_gb`).

So the readers live together, once, with their units and their traps written
down next to them. Everything here is PURE: dict in, scalar/tuple out. No HTTP,
no clock, no fleet action — which is what keeps `core` at the bottom of the DAG
and lets the whole module be tested from a literal.

What is deliberately NOT here
-----------------------------
* **pydantic `Offer` / `Instance` / `BidState` classes.** Plan §5 calls for
  them, and `requirements.txt` already pins pydantic for it — but a model is
  only worth what parses INTO it, and nothing does yet: the port is ADD-ONLY
  (plan §8), so every caller still reads raw dicts out of `herdd.py`. Landing
  classes now would add an unexercised second representation and a
  round-trip-lossiness risk to a step whose whole contract is "behavior
  preserving". They arrive with the parse, at `core/api.py`'s boundary, and the
  design constraint recorded for that step is `ConfigDict(extra="allow")` plus
  a `.raw` escape hatch — unknown-field passthrough is load-bearing here, not
  cosmetic: `gpu_ram` was added to `_JOB_PRIMARY_SHAPE_KEYS` only on
  2026-08-16, `hf["primary_shape"]` persists a raw instance dict into a FROZEN
  journal schema, and `ls --json` / `show --json` / the dash-cache writers
  print the payload straight through. `BidState` additionally overlaps
  `supervise/state.py` and must be designed with it, not ahead of it.
* **The render half.** `fmt_offer` (offer row -> ls line) and `_image_short`
  are `core/fmt.py`; `_image_short` already landed there in wave 1. Reading 11
  offer fields does not make a formatter a model.
* **Anything that performs I/O.** `_machine_offers_soft` BUILDS the MachineRow
  shape below, but it does so with a bundles POST, so it belongs in `market/`.
  Only the shape it emits is described here, because `_rates` consumes it.
* **Policy that happens to read a payload.** `_replacement_candidate_class`
  needs `GPU_DEFAULT_POLICY_TIERS` + `gpu_family_names` (`market/offers.py`);
  `_is_jobs_box` is a health question (`boxes/health.py`); `ssh_access_warning`
  and `_ssh_endpoints` are `boxes/ssh.py` (plan §5 names endpoints there). The
  pure unit accessor `_gpu_ram_gb` and the onstart classifier
  `instance_ssh_install` DO live here — they read a field and answer about the
  field, and the policy above them imports them.
* **The label GRAMMAR.** `_label_value` is the accessor; the two token rules it
  applies (a value stops at the first whitespace GROUP boundary, and truncates
  at the first `keep` TOKEN) belong to `core/labels.py` and are called there.
  A second copy of those rules is precisely how the 2026-08-02
  `run-<RID>:keep` un-revoked-B2-key bug happened.
* **`LIVE_STATES`.** It lives in `bidpolicy.py` (Zone S, moved there 2026-07-30
  with the pure predicates that read it) and stays there; an `Instance.is_live`
  property is the natural absorber of the ~21 `actual_status`-lower-in-set
  sites, but it must READ the Zone S set, never re-declare it.

Provenance: verbatim-with-types move from `tools/vast/herdd.py`, plan §8
step 2 (`core/`) of `docs/plans/vast-tooling-refactor-v2.md`. Each symbol
carries its `# moved-from:` marker. Step 2 was ADD-ONLY, so `herdd.py` kept
its own copies and `tools/vast/test_vastlib_core_models.py` pinned the two
implementations against each other, input for input; step 6d deleted the flat
copies and that file's parity half with them (its docstring records what
stayed and why).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, TypedDict

import ladder_core

from vastlib.core import labels

# A raw vast.ai payload dict — an instance body, a bundles offer row, or one of
# the supervise context dicts that carries them. `Any` on the value side is the
# honest type: this is JSON off the wire whose field set is discovered, not
# declared (see the module docstring on why unknown-field passthrough matters).
# Reading is all these accessors do, so `Mapping` rather than `dict` — it keeps
# a caller free to hand in a read-only view without a copy.
Payload = Mapping[str, Any]


#: A price as a float, or None for anything that isn't one — `ladder_core`'s
#: copy, aliased rather than duplicated (the core needs it and must not import
#: herdd). Kept under this name because fleetd, workflowctl, bid_echo_probe
#: and the suite all call it as `herdd._num_dph`; `test_ladder_core.py`
#: asserts the ALIAS IDENTITY (`is`), so this must stay a binding and never
#: become a wrapper. The annotation is the only thing added — it is what lets
#: mypy's strict `disallow_untyped_calls` see through an untyped Zone-adjacent
#: leaf without a per-call `# type: ignore`.
#:
#: STEP 7, and it stays an alias. Plan §5 says "num_dph's def lands here
#: (public) at step 7 when ladder_core shims" — that antecedent never fired.
#: `ladder_core.py` was never ported: it has no vastlib home, no manifest maps
#: it, and the rename-table generator names it as a module that STAYS a flat
#: sibling. Moving the def here anyway would INVERT the dependency (a Zone S
#: flat leaf importing vastlib), which is the one direction this architecture
#: has never taken and the one import-linter cannot see — and `ladder_core` is
#: not in the jobd bundle, so the flat-bundle runtime test that stands in for
#: that contract would not catch a regression either. The alias costs nothing
#: and all three identity pins (test_ladder_core.py:93,
#: test_vastlib_core_models.py, test_vastlib_market.py) hold. Making `num_dph`
#: public here regardless is an owner call, not a refactor step.
# moved-from: herdd._num_dph
_num_dph: Callable[[object], float | None] = ladder_core.num_dph


# Every onstart we compose carries this marker line when the authorized_keys
# repair is present, so `ssh`/`ls` can tell an ssh-able box from one born
# without it by reading the instance's stored `onstart` — no box access needed.
# It lives here rather than in `boxes/ssh.py` because it is the payload TELL
# that `instance_ssh_install` classifies on; the snippet composer and
# `ssh_access_warning` import it from here when `boxes/ssh.py` lands, so there
# is one copy of the string that decides whether a box is reachable.
# moved-from: herdd.SSH_INJECT_MARKER
SSH_INJECT_MARKER = "# herdd-ssh-key v2"


# moved-from: herdd.instance_ssh_install
def instance_ssh_install(i: Payload | None) -> str:
    """Classify how (if at all) an instance installs our ssh key, from its
    stored onstart alone. The instance list already ships `onstart`, so this
    costs nothing — no probe, no box access.

      "v2"     — the current snippet: installs the key AND repairs
                 ownership/mode every boot. Expected to work.
      "legacy" — a pre-2026-07-31 append-only inject. Usually works, but has no
                 defence against vast writing authorized_keys as its own host
                 user (the StrictModes failure — see the snippet docstring).
      "none"   — nothing installs the key. Un-ssh-able for the box's whole
                 life, because the onstart is fixed at create time.
    """
    o = (i or {}).get("onstart") or ""
    if SSH_INJECT_MARKER in o:
        return "v2"
    if "authorized_keys" in o:
        return "legacy"
    return "none"


# moved-from: herdd.instance_has_ssh_inject
def instance_has_ssh_inject(i: Payload | None) -> bool:
    """True when this instance's stored onstart carries the current repair."""
    return instance_ssh_install(i) == "v2"


# moved-from: herdd._label_value
def _label_value(label: str | None, prefix: str) -> str | None:
    """PURE. Value of a `<prefix>:<value>` instance label, or None.

    The accessor half of the label grammar: `core/labels.label_value` applies
    the two token rules (stop at the first whitespace GROUP boundary; truncate
    at the first `keep` TOKEN, never at a substring) and this is the name every
    caller — `_destroy_and_revoke`'s key naming, `_launch_preflight`'s dup
    guard, `fleetd._resolve_iid` through `_instance_run_label` — binds to.

    Delegated rather than copied ON PURPOSE. The rules were learned by
    `_reap_kept` the hard way, and a drifted second copy of them is exactly the
    2026-08-02 bug: `run:<RID>` read as `<RID>:keep` after fleetd stamped its
    park token, so `_destroy_and_revoke` minted the revoke name
    `run-<RID>:keep` and the ephemeral B2 key actually named `run-<RID>` was
    never revoked — a credential outliving the box it was issued for. The full
    history, including why the `run:<ID>:handoff` suffix must SURVIVE the
    truncation, is on `labels.label_value`.

    Called module-attribute-style (`labels.label_value`) so the seam a test
    patches is the one the caller resolves at call time (plan §8b)."""
    return labels.label_value(label, prefix)


# moved-from: herdd._instance_run_label
def _instance_run_label(i: Payload) -> str | None:
    """RUN_ID carried by a vast instance's run:<id> label, or None.

    Tolerates appended label tokens (`run:<id>:keep`) — see `_label_value`.

    Takes `i` UNGUARDED, exactly as the original did: `i.get(...)` raises
    AttributeError on None where `_instance_serve_label`'s `(i or {})` returns
    None. The asymmetry is untested in `test_label_grammar.py` (which covers
    `{}` and `{"label": None}` only), and the port keeps it rather than picking
    a behavior no caller has been observed to rely on — a widening here is a
    behavior change, and this step is contractually behavior-preserving. It is
    the `Instance` model's job to make the question moot.
    """
    return _label_value(i.get("label") or "", "run")


# moved-from: herdd._instance_serve_label
def _instance_serve_label(i: Payload | None) -> str | None:
    """SERVE_ID carried by a vast instance's serve:<id> label, or None.
    Same grammar as run: (launch_serve.sh labels every box `serve:<SERVE_ID>`)."""
    return _label_value((i or {}).get("label") or "", "serve")


# moved-from: herdd._instance_env
def _instance_env(i: Payload) -> dict[str, Any]:
    """The env we passed at launch, read back from the instance record
    (`extra_env` is a list of [K, V] pairs on the wire; tolerate a dict too).
    {} when absent — older boxes were launched before any stamping."""
    ee = i.get("extra_env") or []
    if isinstance(ee, dict):
        return dict(ee)
    out: dict[str, Any] = {}
    for kv in ee:
        if isinstance(kv, (list, tuple)) and len(kv) == 2:
            out[kv[0]] = kv[1]
    return out


# moved-from: herdd.effective_cores
def effective_cores(o: object) -> float | None:
    """CPU cores this OFFER actually gets, as a float, or None.

    A vast offer is a SLICE of a host, and `cpu_cores` is the whole machine's
    count: a 384-core host offering 1 of its 8 GPUs hands you 48 cores, not 384.
    The slice fraction is `gpu_frac` (gpus_rented / gpus_on_host), and vast
    already publishes the product as `cpu_cores_effective` — prefer it, and fall
    back to the multiplication when a row omits it.

    This matters for EVAL boxes specifically, where scoring (a CPU compile+diff
    loop) was measured at 78-85% of arm wall clock on 2026-08-16: the advertised
    128/384 core counts on recent 5090 hosts resolved to 16-55 usable cores, so
    picking on the advertised number picks the wrong box. Guidance for choosing
    a host — never configuration; the box sizes its own concurrency at runtime
    from its cpuset/quota. And it must be computed PER OFFER: more GPUs usually
    means more cores, so it cannot be inferred from the GPU model or count."""
    if not isinstance(o, dict):
        return None
    v = o.get("cpu_cores_effective")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v)
    cores, frac = o.get("cpu_cores"), o.get("gpu_frac")
    if (isinstance(cores, (int, float)) and not isinstance(cores, bool)
            and isinstance(frac, (int, float)) and not isinstance(frac, bool)
            and cores > 0 and frac > 0):
        return float(cores) * float(frac)
    return None


# moved-from: herdd._instance_image
def _instance_image(i: Payload) -> str:
    """Image ref a vast instance runs, or ''."""
    # The `or` chain's static type is the payload's `Any`; the runtime value is
    # whatever vast sent. Declared `str` because that is the contract every
    # caller (the ls column, `_image_short`, the stale-image guard) relies on,
    # and a coercion here would CHANGE behavior on a non-str field rather than
    # preserve it.
    image: str = i.get("image_uuid") or i.get("image") or ""
    return image


class MachineRow(TypedDict):
    """One rentable offer on a machine, as `_machine_offers_soft` synthesizes it.

    NOT a vast payload shape — a three-field projection built by `market/`
    (`{"g": num_gpus, "base": dph_base, "bid": min_bid}`) so a single no-`type`
    bundles POST answers reserved price, spot price AND capacity at once.
    Modelled here because `_rates` below is its only consumer and the field
    names are two-letter; the builder stays in `market/` because it does I/O.

    `base`/`bid` are `_num_dph`-coerced, so either can be None on a row whose
    price field was junk — which is why `_rates` tests each one before adding
    the disk sliver.
    """

    g: int
    base: float | None
    bid: float | None


class MachineMarket(TypedDict):
    """`_market_map`'s per-machine value: the machine's rows plus the largest
    GPU count any of them offers (the availability test `_rates` applies)."""

    offers: list[MachineRow]
    max_gpus: int


# moved-from: herdd._rates
def _rates(i: Payload,
           market: Mapping[str, MachineMarket] | None) -> tuple[float | None,
                                                                float | None,
                                                                bool | None]:
    """(reserved_total, spot_total, available) $/hr for one instance.

    Picks the machine offer matching this box's GPU count (else the smallest
    offer that covers it) and adds the box's own disk portion, so both prices
    are apples-to-apples hourly totals for THIS box's config. `available` is
    True/False when the live market was read (machine has a rentable offer of
    >= the needed GPUs), or None when unknown (--no-spot / read failed).
    Reserved falls back to dph_total and spot to the stale min_bid when the
    machine isn't in `market`."""
    dph = _num_dph(i.get("dph_total"))
    base = _num_dph(i.get("dph_base"))
    disk = (dph - base) if (dph is not None and base is not None) else 0.0
    n = i.get("num_gpus") or 0
    reserved, spot, avail = dph, None, None
    mkt = market.get(str(i.get("machine_id"))) if market else None
    if mkt:
        avail = bool(n) and mkt["max_gpus"] >= n
        match = next((o for o in mkt["offers"] if o["g"] == n), None)
        if match is None:
            cover = [o for o in mkt["offers"] if o["g"] >= n]
            match = min(cover, key=lambda o: o["g"]) if cover else None
        if match:
            m_base, m_bid = match["base"], match["bid"]
            if m_base is not None:
                reserved = m_base + disk
            if m_bid is not None:
                spot = m_bid + disk
    if spot is None:
        mb = _num_dph(i.get("min_bid"))
        spot = (mb + disk) if mb is not None else None
    return reserved, spot, avail


# moved-from: herdd._storage_day
def _storage_day(i: Payload) -> float | None:
    """$/day a box bills for its disk while stopped. Vast gives this directly
    as storage_total_cost ($/hr); fall back to the dph disk delta. None if
    neither is present."""
    s = _num_dph(i.get("storage_total_cost"))
    if s is None:
        dph, base = _num_dph(i.get("dph_total")), _num_dph(i.get("dph_base"))
        s = (dph - base) if (dph is not None and base is not None) else None
    return s * 24 if s is not None else None


# moved-from: herdd._disk_gb
def _disk_gb(i: Payload) -> tuple[float | None, float | None]:
    """PURE. `(allocated_gb, used_gb)` for a box, either element None when the
    vast API omits it OR reports a negative sentinel. Storage bills on the
    ALLOCATED size, so `allocated` is what costs money and `used` is the only
    evidence of whether that was the right number (2026-07-21 audit: a 160 GB
    allocation billing $4.62/day while using 18 GB — 8.9x oversized, and
    invisible because `ls` reported the dollar cost and never the GB). Both
    fields ride every `GET v1/instances/` response already, so reading them
    costs nothing.

    NEGATIVES ARE UNKNOWN, NOT ZERO: a box still `loading` reports
    `disk_usage: -1` until the container is provisioned (observed live on box
    46246859). Passing that through reads as "uses -1 of 120 GB, -1% used" —
    a fully-wasted allocation — which would make every booting box trip an
    oversizing warning and would poison any average built on the field."""
    def pos(v: object) -> float | None:
        n = _num_dph(v)
        return n if (n is not None and n >= 0) else None
    return pos(i.get("disk_space")), pos(i.get("disk_usage"))


# moved-from: herdd._disk_frac
def _disk_frac(i: Payload) -> float | None:
    """PURE. `used/allocated` as a 0..1 float, or None when either side is
    unknown or the allocation is zero. This is the oversizing signal: the
    fraction of paid-for disk that is actually carrying anything."""
    alloc, used = _disk_gb(i)
    if alloc is None or used is None or alloc <= 0:
        return None
    return used / alloc


# moved-from: herdd._dash_verified
def _dash_verified(offer: Payload) -> int:
    """PURE. 1/0 host-verification flag for one bundle offer.

    The bundles response spells this `verification: "verified"`, NOT the
    `verified` boolean the SEARCH FILTER uses (`{"verified": {"eq": true}}`).
    Reading `offer["verified"]` therefore yields None for every row and would
    have published "unverified" for a wholly verified result set."""
    if offer.get("verified") is True:
        return 1
    return 1 if str(offer.get("verification") or "").lower() == "verified" else 0


# moved-from: herdd._instance_standing_bid
def _instance_standing_bid(inst: object) -> float | None:
    """The STANDING BID on a bid instance = `dph_base`, NOT `dph_total`.

    `dph_total` is the bid PLUS storage, and mixing the two is the same
    doc-50-R1 field confusion that has now cost money on three lanes. Verified
    against live instances 2026-08-09 (read-only) and against the incident box's
    own `show` snapshot:

        47218938  dph_base 2.944   diskHour 0.13722   dph_total 3.08122
                  (`instance.gpuCostPerHour` == dph_base == the price fleetd
                   had just PUT at 23:28:00Z)
        47205562  min_bid 0.13333  dph_base 0.200     dph_total 0.24630
        47226953  min_bid 0.33333  dph_base 0.667     dph_total 0.71978

    Why it matters beyond tidiness: `jc["last_bid"]` seeded from `dph_total` is
    one storage-sliver ABOVE the number vast will report back as the chunk's
    `min_bid`, so `market_floor_is_self` — which is an exact-equality test by
    design — could not recognise our own bid. On 2026-08-08 that is exactly the
    gap the ratchet walked through: bid 2.697 (dph_total) vs floor 2.562
    (dph_base), read as a market $0.135 below us and defended to 2.818.

    None when the field is absent; callers keep their existing `dph_total`
    fallback. Deliberately NOT applied to `launch_dph_anchor` or the spend
    accrual — those price the BILL, which is `dph_total` and correctly so."""
    if not isinstance(inst, dict):
        return None
    return _num_dph(inst.get("dph_base"))


# The evidence-carrying result of one market floor read. `ok=False` is
# IGNORANCE (no machine_id, HTTP failure, unparseable body) and must never
# advance eviction state (SPOT_DESIGN §5 rule 1: transient != eviction);
# `ok=True, listed=False` is EVIDENCE that vast answered and the machine lists
# no rentable bid offer at all — and that is the *only* observable an outbid
# produces (defect D7, AUTOBID_AUDIT_2026-08-08 §4/§6).
#
# `floors` keeps the per-chunk candidate floors for our GPU count APART
# (review 2026-08-10, F3): the min() collapse could hide a genuine sibling
# floor behind our own echo, so the self-floor guards filter ROWS, not the
# collapsed scalar. `scaled` is True when no offer matched our exact GPU count
# and min_bid was synthesized by per-GPU rescale of a different chunk size
# (F8) — such a number can never match our bid history.
#
# Spelled as a `typing.NamedTuple` rather than the original
# `collections.namedtuple` + `MarketRead.__new__.__defaults__ = ((), False)`
# two-liner: same runtime object model (tuple subclass, `._fields`,
# tuple-unpack, 3-positional construction with the same two defaults), but the
# field types are declared and the `__defaults__` poke — which mypy cannot
# type — is expressed as ordinary default values. `test_vastlib_core_models.py`
# pins the runtime object model field-for-field — `._fields`, the two defaults,
# tuple-unpack and the 3-positional call shape that `_market_min_bid_read`
# uses. (It compared them against `herdd.MarketRead` until step 6d made that
# name this class.)
# moved-from: herdd.MarketRead
class MarketRead(NamedTuple):
    ok: bool
    listed: bool
    min_bid: float | None
    floors: Sequence[float] = ()
    scaled: bool = False


# moved-from: herdd._gpu_ram_gb
def _gpu_ram_gb(raw: object) -> float | None:
    """Per-card VRAM in GB from a vast `gpu_ram` field, or None when the value
    is missing or not plausibly a card.

    Vast reports MB on offers (`fmt_offer` divides by 1024) and the instance
    schema is the same family, but this number becomes a search FLOOR: read a
    megabyte value as gigabytes and the floor is 0, which silently widens the
    candidate class to every smaller card on the market. So the unit is
    inferred and anything implausible is UNKNOWN rather than a floor of zero —
    a missing floor narrows the class to the primary's own family, which is
    safe, while a wrong one licenses a downgrade."""
    # `raw` is `object` because the function is deliberately total over
    # whatever the payload carried — the except clause below IS the contract
    # for every non-numeric field, so the coercion is not narrowed first.
    try:
        v = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if v >= 1024:                     # MB, the documented unit
        return v / 1024.0
    if v >= 4:                        # already GB (no card we run is under 4)
        return v
    return None


# moved-from: herdd._job_primary_inst
def _job_primary_inst(jctx: Payload) -> Payload | None:
    """The primary box's instance dict from the tick's instance snapshot, or None."""
    instances: Sequence[Payload] = jctx.get("instances", []) or []
    for i in instances:
        if str(i.get("id")) == str(jctx.get("iid")):
            return i
    return None


# instance-dict keys the understudy sizing reads; snapshotted into
# hf["primary_shape"] at ARM so a transient instance-API miss mid-handoff can't
# size a 1-GPU default-image understudy for a multi-GPU job (2026-07-18 review P5).
# `gpu_ram` (2026-08-16) is per-card VRAM in MB — the minimum-requirements
# floor a replacement candidate class is built from. Without it the class falls
# back to the primary's alias family (`_replacement_candidate_class`), so a
# watch whose ARM snapshot predates this key degrades, it does not misfire.
# moved-from: herdd._JOB_PRIMARY_SHAPE_KEYS
# `compute_cap` (2026-08-18) is sm x10 — the exact channel the arch-change
# alarm prefers over the gpu_name alias family. Instance bodies do not always
# carry it, so the alarm degrades to names; snapshotting it costs one key.
_JOB_PRIMARY_SHAPE_KEYS = ("gpu_name", "gpu_ram", "num_gpus", "disk_space",
                           "image_uuid", "compute_cap")


# moved-from: herdd._job_primary_shape
def _job_primary_shape(jctx: Payload, hf: Payload | None = None) -> Payload | None:
    """Primary sizing dict for understudy fit: the live instance dict when the
    primary is in this tick's snapshot, else the shape cached at ARM. None only
    when both are missing (pre-ARM, or a reconcile-adopt whose ARM never ran)."""
    cached: Payload | None = (hf or {}).get("primary_shape")
    return _job_primary_inst(jctx) or cached or None
