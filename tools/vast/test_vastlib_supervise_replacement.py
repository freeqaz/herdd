"""`vastlib.supervise.replacement` — the ported effectful drivers.

Why this file exists
--------------------
Ten flat suites drive this code today (`test_eviction_replacement.py`,
`test_pull_watchdog.py`, `test_boot_sla.py`, `test_rebid_ladder.py`,
`test_defense_wiring.py`, `test_ladder_latch_hygiene.py`,
`test_spot_breakeven_wiring.py`, `test_job_retarget.py`,
`test_eviction_blindspot.py`, `test_ondemand_ref.py`) and every one of them
stays UNEDITED through this step (plan §8's add-only amendment): they steer the
LIVE flat copies in `herdd.py` via `monkeypatch.setattr(v, ...)`, so not one
of their assertions reaches `vastlib`.

That is a hole with a shape. **Five run-lane symbols in this module get their
only coverage from `test_supervise.py`, whose 426 references are pinned unedited
by the §8 amendment** — `_supervise_boot_sla`, `_handoff_understudy_body`,
`_handoff_pick_offer`, `_has_relaunched_after_last_evicted` and `_relaunch`
would otherwise land with ZERO tests against the ported copy. They are the first
half of this file. The second half is what
`.port_manifests/sup-replacement.json` names as the port-time priority — the
peer-49bc0103 disk + TTL block (`test_eviction_replacement.py:1915-1989`) — and
the third is the four hazards a typed rewrite is most likely to quietly lose:
the frozen ten-key `_sel` decision record (H1), the refusal-dedup latch (H5),
and two of the doc-50 dollar guards (H4).

No expectation is changed (plan §7.4): where an assertion mirrors a flat one it
mirrors it exactly, and where it is new it is new because the flat suite could
not reach the ported copy at all.

What is deliberately NOT here
-----------------------------
* **No policy arithmetic.** `bidpolicy.replacement_decision` / `rebid_ladder` /
  `bid_decision` / `_handoff_candidate_ok` are Zone S and the flat suites own
  their numbers; re-asserting a ceiling here would put a second copy of a
  money constant in the tree. What is pinned is the ORDER of operations, the
  record shape, and which market read feeds which field.
* **No re-test of the jobs-lane drivers the flat suites already cover
  directly** (`_job_replacement_offers`, `_job_rebid_ladder`,
  `_job_pull_condemn`, `_job_boot_sla_tick`, ...). Those migrate with their
  callers at steps 6-7 via the rename table; duplicating them now would double
  the maintenance without adding a single new fact.
* **No network, no B2, no subprocess.** `api.request_soft` is stubbed in every
  test that can reach it (conftest's guard lets GETs through to the REAL API —
  it refuses mutations, it is not an offline switch), every destroy/PUT seam is
  stubbed, and `journal._sup_emit` / `_job_handoff_emit` — the only B2 writers
  on these paths — are stubbed wherever they are reachable.
  `journal._job_ladder_journal` is left REAL: it is pure in-memory and the
  `_sel` test reads the queue it appends to.

Provenance: created 2026-08-16 alongside `vastlib/supervise/replacement.py`,
plan §8 step 4.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pytest

VAST_DIR = Path(__file__).resolve().parent
if str(VAST_DIR) not in sys.path:                      # entry-script sys.path shape
    sys.path.insert(0, str(VAST_DIR))

import bidpolicy as bp                                 # noqa: E402
import imageref                                        # noqa: E402
import ladder_core                                     # noqa: E402

from vastlib.boxes import health, lifecycle, ssh       # noqa: E402
from vastlib.core import api, config, models           # noqa: E402
from vastlib.jobs import risk                          # noqa: E402
from vastlib.launch import launch as launchmod         # noqa: E402
from vastlib.launch import spec as launch_spec         # noqa: E402
from vastlib.market import offers as market_offers     # noqa: E402
from vastlib.market import pricing                     # noqa: E402
from vastlib.storage import b2                         # noqa: E402
from vastlib.supervise import job_lane, journal        # noqa: E402
from vastlib.supervise import replacement as R         # noqa: E402
from vastlib.supervise import run_lane                 # noqa: E402

NOW = 3_000_000.0


@pytest.fixture(autouse=True)
def _no_knob_env(monkeypatch):
    """Every knob here resolves namespace > `JOB_<NAME>` env > yaml > default.
    A developer with one exported would silently re-price the windows below, so
    the env rung is cleared rather than trusted."""
    for k in ("JOB_EVICTED_TTL_S", "JOB_REPLACEMENT_VERIFIED",
              "JOB_MAX_REPLACEMENTS", "JOB_REPLACE_CEILING_MULT",
              "REPLACEMENT_CUDA_FLOOR", "BOOT_SLA_S", "BOOT_MAX_HOST_RETRIES",
              "B2_BUCKET"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def _no_live_api(monkeypatch):
    """Belt to conftest's mutation guard, which passes GETs THROUGH to the real
    API. `_relaunch`'s first act is a `GET v1/instances/`, so an unstubbed test
    here would query vast for real."""
    monkeypatch.setattr(api, "request_soft",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("unstubbed api.request_soft")))


def _offer(oid=1, machine=7, min_bid=0.10, dph=0.1005, gpu="RTX PRO 6000",
           ngpu=2, disk=200.0):
    return {"id": oid, "machine_id": machine, "min_bid": min_bid,
            "dph_total": dph, "gpu_name": gpu, "num_gpus": ngpu,
            "disk_space": disk}


def _st(**kw):
    st = {"run_id": "R1", "instance_id": 41, "present": True,
          "actual_status": "loading", "excluded_machines": [],
          "spend_usd": 0.0, "relaunch_count": 0, "max_relaunch": 3,
          "remaining_wall_h": 6.0, "on_demand": 1.0, "last_bid": 0.5,
          "dph_total": 0.5, "max_bid": 2.0, "launch_spec": {}}
    st.update(kw)
    return st


def _a(**kw):
    ns = argparse.Namespace(dry_run=False, boot_sla=True, budget=None,
                            num_gpus=2, exclude_machines=[])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _jc(**kw):
    jc = {"a": _a(), "iid": "41", "instances": [], "dry_run": False,
          "now": NOW, "spend_usd": 0.0, "launch_disk_gb": 50}
    jc.update(kw)
    return jc


# --------------------------------------------------------------------------- #
# 1. The peer-49bc0103 disk + TTL block — the manifest's port-time priority,
#    mirroring test_eviction_replacement.py:1915-1989.
# --------------------------------------------------------------------------- #
def test_offer_disk_gb_reads_disk_space_as_gb():
    assert R._offer_disk_gb({"disk_space": 47.0}) == 47.0
    assert R._offer_disk_gb({"disk_space": "23"}) == 23.0


@pytest.mark.parametrize("offer", [None, {}, {"disk_space": None},
                                   {"disk_space": 0}, {"disk_space": "n/a"}])
def test_offer_disk_gb_is_none_when_unreadable(offer):
    """None, never 0.0 — an unreadable size must not read as "no disk" and let
    the caller rank a machine it cannot use (defect #67's shape)."""
    assert R._offer_disk_gb(offer) is None


def test_note_evicted_machine_writes_the_set_and_the_ts_sidecar():
    """BOTH halves: `evicted_machines` stays the historical set every existing
    reader and the journal print use; `evicted_machine_ts` is the str-keyed
    sidecar the TTL is computed from. The asymmetry is persisted in state.json."""
    jc = _jc()
    R._job_note_evicted_machine(jc, 7, bp.EVICTION_OUTBID, NOW)
    assert jc["evicted_machines"] == {7}
    assert jc["evicted_machine_ts"] == {"7": {"ts": NOW,
                                              "class": bp.EVICTION_OUTBID}}


def test_note_evicted_machine_ignores_an_unknown_machine():
    jc = _jc()
    R._job_note_evicted_machine(jc, None, bp.EVICTION_OUTBID, NOW)
    assert "evicted_machines" not in jc and "evicted_machine_ts" not in jc


def test_excluded_machines_expires_only_the_market_state_classes():
    """`outbid` and `host_stop` describe a market state and age out at
    EVICTED_EXCLUSION_TTL_S; `host_failure` and `ondemand_displaced` describe a
    machine and never do."""
    jc = _jc()
    for m, cls in ((1, bp.EVICTION_OUTBID), (2, bp.EVICTION_HOST_STOP),
                   (3, bp.EVICTION_HOST_FAILURE), (4, bp.EVICTION_ONDEMAND)):
        R._job_note_evicted_machine(jc, m, cls, NOW)
    assert R._job_excluded_machines(jc, NOW + 60) == {1, 2, 3, 4}
    aged = NOW + R.EVICTED_EXCLUSION_TTL_S + 1
    assert R._job_excluded_machines(jc, aged) == {3, 4}


def test_excluded_machines_degrades_to_permanent_without_the_sidecar():
    """A watch restored from a pre-2026-08-16 state.json has `evicted_machines`
    and no `evicted_machine_ts`. Absence means PERMANENT — silently
    un-excluding a broken host is the failure direction that costs money."""
    jc = _jc(evicted_machines={9})
    assert R._job_excluded_machines(jc, NOW + 10 * R.EVICTED_EXCLUSION_TTL_S) == {9}


def test_excluded_machines_keeps_pull_bad_machines_forever():
    """That host failed to pull our image; no TTL applies, and the set is
    unioned with the evicted one rather than replacing it."""
    jc = _jc(pull_bad_machines={5})
    R._job_note_evicted_machine(jc, 1, bp.EVICTION_OUTBID, NOW)
    aged = NOW + R.EVICTED_EXCLUSION_TTL_S + 1
    assert R._job_excluded_machines(jc, aged) == {5}


def test_excluded_machines_ttl_is_knobbable_per_watch():
    jc = _jc(a=_a(evicted_ttl_s=60.0))
    R._job_note_evicted_machine(jc, 1, bp.EVICTION_OUTBID, NOW)
    assert R._job_excluded_machines(jc, NOW + 61) == set()


# --------------------------------------------------------------------------- #
# 2. Run-lane symbols with NO coverage from the untouched flat suite.
# --------------------------------------------------------------------------- #
def test_has_relaunched_after_last_evicted_looks_only_at_the_tail(monkeypatch):
    evs = [{"event": "relaunched"}, {"event": "evicted"}, {"event": "tick"}]
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda r: evs)
    assert R._has_relaunched_after_last_evicted("R1") is False
    monkeypatch.setattr(launch_spec, "_raw_events_soft",
                        lambda r: evs + [{"event": "relaunched"}])
    assert R._has_relaunched_after_last_evicted("R1") is True


def test_has_relaunched_with_no_eviction_scans_the_whole_log(monkeypatch):
    monkeypatch.setattr(launch_spec, "_raw_events_soft",
                        lambda r: [{"event": "relaunched"}])
    assert R._has_relaunched_after_last_evicted("R1") is True
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda r: [])
    assert R._has_relaunched_after_last_evicted("R1") is False


def test_handoff_pick_offer_returns_the_first_qualifier(monkeypatch):
    """Offers arrive min_bid-ascending, so the first qualifier is the cheapest."""
    monkeypatch.setattr(market_offers, "_search_offers_soft",
                        lambda a: [_offer(oid=1, machine=1, min_bid=0.9),
                                   _offer(oid=2, machine=2, min_bid=0.10)])
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    seen = []
    monkeypatch.setattr(bp, "_handoff_candidate_ok",
                        lambda *a, **k: seen.append(a) or (len(seen) == 2))
    assert R._handoff_pick_offer(_st(), _a())["id"] == 2


def test_handoff_pick_offer_reads_on_demand_from_the_market_not_the_row(monkeypatch):
    """doc-50 R1. `dph_total` on a BID row is the interruptible price; the §2.3
    filter must see the machine's real on-demand rate."""
    o = _offer(oid=1, machine=3, min_bid=0.10, dph=0.1005)
    monkeypatch.setattr(market_offers, "_search_offers_soft", lambda a: [o])
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda off, n=None: 4.0)
    got = {}
    monkeypatch.setattr(bp, "_handoff_candidate_ok",
                        lambda pd, mb, od, *a, **k: got.update(od=od) or True)
    R._handoff_pick_offer(_st(), _a())
    assert got["od"] == 4.0 and got["od"] != o["dph_total"]


def test_handoff_pick_offer_refuses_past_the_probe_budget(monkeypatch):
    """The run lane RETURNS None past HANDOFF_ODPROBE_MAX (it does not keep the
    priced prefix the way `_replacement_spot_walk` does) — one of the six pinned
    lane divergences. Probes are memoized per machine, so the budget counts
    MACHINES, not rows."""
    offers = [_offer(oid=i, machine=i) for i in range(pricing.HANDOFF_ODPROBE_MAX + 3)]
    monkeypatch.setattr(market_offers, "_search_offers_soft", lambda a: offers)
    probes = []
    monkeypatch.setattr(pricing, "_offer_ondemand_ref",
                        lambda o, n=None: probes.append(o) or 4.0)
    monkeypatch.setattr(bp, "_handoff_candidate_ok", lambda *a, **k: False)
    assert R._handoff_pick_offer(_st(), _a()) is None
    assert len(probes) == pricing.HANDOFF_ODPROBE_MAX


def test_handoff_understudy_body_rejects_a_non_qualifying_offer(monkeypatch):
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    monkeypatch.setattr(bp, "_handoff_candidate_ok", lambda *a, **k: False)
    assert R._handoff_understudy_body(_st(), _a(), _offer()) == (
        None, None, "candidate_reject")


def test_handoff_understudy_body_refuses_an_unwinnable_floor(monkeypatch):
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    monkeypatch.setattr(bp, "_handoff_candidate_ok", lambda *a, **k: True)
    monkeypatch.setattr(pricing, "_auto_bid_price", lambda mb, od=None: None)
    assert R._handoff_understudy_body(_st(), _a(), _offer()) == (
        None, None, "no_price")


def test_handoff_understudy_body_stamps_the_two_box_side_contracts(monkeypatch):
    """T4b: HANDOFF_EPOCH (strictly above the primary's) and HANDOFF_TTL_S (the
    dead-man deadline) are what onstart/train.sh reads."""
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    monkeypatch.setattr(bp, "_handoff_candidate_ok", lambda *a, **k: True)
    monkeypatch.setattr(pricing, "_auto_bid_price", lambda mb, od=None: 0.12)
    monkeypatch.setattr(R, "_relaunch_body",
                        lambda st, a, bid, label=None, key_name=None:
                        ({"label": label, "key_name": key_name, "env": {"X": "1"}}, []))
    body, bid, missing = R._handoff_understudy_body(_st(), _a(), _offer(), epoch=3)
    assert bid == 0.12 and missing == []
    assert body["label"] == "run:R1:handoff"
    assert body["env"] == {"X": "1", "HANDOFF_EPOCH": "3",
                           "HANDOFF_TTL_S": str(bp.HANDOFF_TTL_S)}


def test_handoff_understudy_body_mints_a_fresh_key_name_every_call(monkeypatch):
    """Mirrors test_lifecycle.py:876-877. `b2_mint_key.mint()` is
    revoke-then-mint BY NAME and the primary holds `run-<RUN>`, so a
    deterministic (or memoized) nonce would revoke the primary's live key
    mid-run — the box-44566398 incident."""
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    monkeypatch.setattr(bp, "_handoff_candidate_ok", lambda *a, **k: True)
    monkeypatch.setattr(pricing, "_auto_bid_price", lambda mb, od=None: 0.12)
    monkeypatch.setattr(R, "_relaunch_body",
                        lambda st, a, bid, label=None, key_name=None:
                        ({"key_name": key_name}, []))
    names = {R._handoff_understudy_body(_st(), _a(), _offer())[0]["key_name"]
             for _ in range(6)}
    assert len(names) > 1
    assert all(n.startswith("run-R1-h") for n in names)


def _boot_sla_seams(monkeypatch, *, destroy=(True, None), gone=True):
    monkeypatch.setattr(R, "_boot_deadline_backoff", lambda base, kills: base)
    monkeypatch.setattr(R, "_confirm_gone", lambda iid, tries=6: gone)
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, dry_run=False, tries=4: destroy)
    emitted = []
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda run_id, ev, **f: emitted.append((ev, f)) or {})
    return emitted


def test_supervise_boot_sla_is_opt_out_and_only_arms_pre_running(monkeypatch):
    monkeypatch.setattr(R, "_boot_deadline_backoff", lambda base, kills: base)
    assert R._supervise_boot_sla(_st(), _a(boot_sla=False)) is None
    st = _st(actual_status="running")
    assert R._supervise_boot_sla(st, _a(), now=lambda: NOW) is None
    assert "boot_sla_armed_iid" not in st


def test_supervise_boot_sla_clears_the_latch_when_the_milestone_lands(monkeypatch):
    """loading -> running IS the milestone; the consecutive-kill counter resets
    with it, so a later slow boot starts from the base deadline again."""
    monkeypatch.setattr(R, "_boot_deadline_backoff", lambda base, kills: base)
    st = _st(actual_status="running", boot_sla_armed_iid=41, boot_sla_kills=2)
    assert R._supervise_boot_sla(st, _a(), now=lambda: NOW) is None
    assert st["boot_sla_armed_iid"] is None and st["boot_sla_kills"] == 0


def test_supervise_boot_sla_holds_without_a_clock(monkeypatch):
    """No start_date, no verdict — and a failed instance poll is not a verdict
    either. Both are "keep supervising", never "destroy"."""
    _boot_sla_seams(monkeypatch)
    st = _st()
    assert R._supervise_boot_sla(st, _a(), get_instance=lambda i: None,
                                 now=lambda: NOW) is None
    assert R._supervise_boot_sla(st, _a(), get_instance=lambda i: {},
                                 now=lambda: NOW) is None
    assert st["boot_sla_armed_iid"] == 41       # armed either way


def test_supervise_boot_sla_condemns_excludes_and_relaunches(monkeypatch):
    emitted = _boot_sla_seams(monkeypatch)
    relaunched = []
    monkeypatch.setattr(R, "_relaunch",
                        lambda st, a: relaunched.append(st) or "relaunched")
    st, a = _st(), _a()
    inst = {"start_date": NOW - 9000, "machine_id": 77, "inet_down": 900}
    assert R._supervise_boot_sla(st, a, get_instance=lambda i: inst,
                                 now=lambda: NOW) == "condemned"
    assert [e for e, _ in emitted] == ["boot_sla_condemned"]
    assert emitted[0][1]["phase"] == "image-pull"
    assert emitted[0][1]["suspect"] == "host"
    # the machine is excluded, and the exclusion is pushed onto the namespace
    # the next search query is built from
    assert st["excluded_machines"] == [77] and a.exclude_machines == [77]
    assert st["boot_sla_kills"] == 1 and st["instance_id"] is None
    assert st["boot_sla_armed_iid"] is None and relaunched


def test_supervise_boot_sla_retries_a_failed_destroy_next_tick(monkeypatch):
    """Never relaunch over a box that may still be alive: a failed destroy (or
    an unconfirmed one) returns 'condemned' with the SLA state untouched."""
    _boot_sla_seams(monkeypatch, destroy=(False, "HTTP 500"))
    monkeypatch.setattr(R, "_relaunch",
                        lambda st, a: pytest.fail("relaunched over a live husk"))
    st = _st()
    inst = {"start_date": NOW - 9000, "machine_id": 77}
    assert R._supervise_boot_sla(st, _a(), get_instance=lambda i: inst,
                                 now=lambda: NOW) == "condemned"
    assert st["instance_id"] == 41 and st.get("boot_sla_kills") is None


def test_supervise_boot_sla_stops_at_the_relaunch_guardrail(monkeypatch):
    """The kill counts against --max-relaunch, and the guardrail is checked
    BEFORE the destroy — a watch out of relaunches keeps its box."""
    _boot_sla_seams(monkeypatch)
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda *a, **k: pytest.fail("destroyed past the guardrail"))
    st = _st(relaunch_count=3, max_relaunch=3)
    inst = {"start_date": NOW - 9000, "machine_id": 77}
    assert R._supervise_boot_sla(st, _a(), get_instance=lambda i: inst,
                                 now=lambda: NOW) == "stop_fatal"
    assert st["last_error"] == "max_relaunch (boot SLA kills)"


def _relaunch_seams(monkeypatch, *, instances=(), emitted=None):
    monkeypatch.setattr(api, "request_soft",
                        lambda m, p, **k: (True, {"instances": list(instances)}, None))
    monkeypatch.setattr(journal, "_sup_emit",
                        lambda run_id, ev, **f: (emitted if emitted is not None
                                                 else []).append((ev, f)) or {})
    monkeypatch.setattr(R, "_reset_run_markers", lambda run_id, dry_run=False: None)
    monkeypatch.setattr(R, "_confirm_gone", lambda iid, tries=6: True)
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda iid, dry_run=False, tries=4: (True, None))


def test_relaunch_adopts_a_live_twin_before_launching_anything(monkeypatch):
    """Idempotence across a supervisor restart: a live non-husk twin is adopted,
    never duplicated. The `relaunched` event is emitted once — the adopt
    backfill is suppressed when the log already has one after the eviction."""
    emitted = []
    twin = {"id": 99, "dph_total": 0.42, "offer_id": 7}
    _relaunch_seams(monkeypatch, instances=[twin], emitted=emitted)
    monkeypatch.setattr(launch_spec, "_raw_events_soft", lambda r: [])
    monkeypatch.setattr(lifecycle, "live_run_instances",
                        lambda run_id=None, instances=None: [twin])
    monkeypatch.setattr(lifecycle, "launch_instance",
                        lambda oid, body: pytest.fail("launched over a live twin"))
    st = _st(husk_id=None)
    assert R._relaunch(st, _a()) == "relaunched"
    assert st["instance_id"] == 99 and st["husk_id"] == 99
    assert st["dph_total"] == 0.42 and st["relaunch_count"] == 1
    assert [e for e, _ in emitted] == ["relaunched"]
    assert emitted[0][1]["adopted"] is True


def test_relaunch_refuses_before_destroying_the_husk_on_a_missing_secret(monkeypatch):
    """SPOT_DESIGN §3.1: never trade a recoverable stopped box for a fresh one
    launched with absent creds. The refusal is BEFORE the destroy."""
    emitted = []
    _relaunch_seams(monkeypatch, emitted=emitted)
    monkeypatch.setattr(lifecycle, "live_run_instances",
                        lambda run_id=None, instances=None: [])
    monkeypatch.setattr(lifecycle, "_destroy_soft",
                        lambda *a, **k: pytest.fail("destroyed the husk anyway"))
    monkeypatch.setattr(R, "_relaunch_body",
                        lambda st, a, bid, label=None, key_name=None:
                        ({}, ["REGISTRY_AUTH_SECRET"]))
    st = _st(husk_id=41)
    assert R._relaunch(st, _a()) == "stop_fatal"
    assert st["last_error"] == "missing_secret_env:REGISTRY_AUTH_SECRET"
    assert [e for e, _ in emitted] == ["relaunch_refused"]


def test_relaunch_filters_affordability_on_the_floor_not_the_rank_price(monkeypatch):
    """The on-demand clamp can only LOWER a bid, so --max-bid is pre-filtered
    against the offer's own FLOOR. A floor over --max-bid is unaffordable by
    definition; an unclamped rank price over it is not."""
    _relaunch_seams(monkeypatch)
    monkeypatch.setattr(lifecycle, "live_run_instances",
                        lambda run_id=None, instances=None: [])
    monkeypatch.setattr(R, "_relaunch_body",
                        lambda st, a, bid, label=None, key_name=None: ({"b": bid}, []))
    monkeypatch.setattr(market_offers, "_search_offers_soft",
                        lambda a: [_offer(oid=1, min_bid=9.0)])
    st = _st(husk_id=None, max_bid=1.0)
    assert R._relaunch(st, _a()) == "stop_budget"
    assert st["last_error"] == "no_offer_under_max_bid (max_bid=1.0)"


def test_relaunch_prices_the_winner_through_bid_target_and_emits_after_the_put(
        monkeypatch):
    """The `relaunched` event is emitted STRICTLY after the contract exists, and
    `last_bid` is the price we PUT (standing-bid semantics), not `dph_total`."""
    emitted = []
    _relaunch_seams(monkeypatch, emitted=emitted)
    monkeypatch.setattr(lifecycle, "live_run_instances",
                        lambda run_id=None, instances=None: [])
    monkeypatch.setattr(R, "_relaunch_body",
                        lambda st, a, bid, label=None, key_name=None: ({"b": bid}, []))
    monkeypatch.setattr(market_offers, "_search_offers_soft",
                        lambda a: [_offer(oid=5, min_bid=0.10)])
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    monkeypatch.setattr(bp, "_bid_target", lambda mb, maxb, od: 0.12)
    put = []
    monkeypatch.setattr(lifecycle, "launch_instance",
                        lambda oid, body: put.append((oid, body)) or (True, 777, None))
    monkeypatch.setattr(__import__("vastlib.boxes.ssh", fromlist=["x"]),
                        "attach_ssh_key_soft", lambda cid: True)
    st = _st(husk_id=None)
    assert R._relaunch(st, _a()) == "relaunched"
    assert put == [(5, {"b": 0.12})]
    assert st["instance_id"] == st["husk_id"] == 777
    assert st["last_bid"] == 0.12 and st["dph_total"] == 0.12
    assert [e for e, _ in emitted] == ["relaunched"]
    assert emitted[0][1]["bid_price"] == 0.12 and emitted[0][1]["offer_id"] == 5


def test_relaunch_never_emits_on_a_transient_put_failure(monkeypatch):
    """"No phantom runs": a transient failure emits nothing and returns 'noop'
    so the next tick retries; a terminal one stops the run."""
    emitted = []
    _relaunch_seams(monkeypatch, emitted=emitted)
    monkeypatch.setattr(lifecycle, "live_run_instances",
                        lambda run_id=None, instances=None: [])
    monkeypatch.setattr(R, "_relaunch_body",
                        lambda st, a, bid, label=None, key_name=None: ({"b": bid}, []))
    monkeypatch.setattr(market_offers, "_search_offers_soft",
                        lambda a: [_offer(oid=5, min_bid=0.10)])
    monkeypatch.setattr(pricing, "_offer_ondemand_ref", lambda o, n=None: 4.0)
    monkeypatch.setattr(bp, "_bid_target", lambda mb, maxb, od: 0.12)
    monkeypatch.setattr(lifecycle, "launch_instance",
                        lambda oid, body: (False, None, "HTTP 429"))
    monkeypatch.setattr(api, "_classify_http", lambda err: "transient")
    assert R._relaunch(_st(husk_id=None), _a()) == "noop"
    assert emitted == []
    monkeypatch.setattr(api, "_classify_http", lambda err: "fatal")
    assert R._relaunch(_st(husk_id=None), _a()) == "stop_fatal"
    assert emitted == []


# --------------------------------------------------------------------------- #
# 3. Frozen wire contract + the doc-50 dollar guards.
# --------------------------------------------------------------------------- #
#: The ten keys `fleet_report.py:125-135` schemas. Spread into BOTH the B2
#: `eviction_replacement_decision` event and the `fleet log` ladder journal —
#: renaming or dropping one silently breaks `fleet log` rendering (H1).
SEL_KEYS = {"disk_floor_gb", "disk_blocked", "spot_candidates", "spot_survivors",
            "ranked_by", "spot_gpu", "spot_machine", "spot_ondemand",
            "ondemand_candidates", "ondemand_gpu"}


def _eviction_seams(monkeypatch, *, spot=(), od=()):
    monkeypatch.setattr(job_lane, "_job_sup_inst",
                        lambda jc, iid: {"id": iid, "machine_id": 7})
    monkeypatch.setattr(R, "_job_replacement_offers",
                        lambda jctx, excl=None, rental="bid", **k:
                        list(od if rental == "ondemand" else spot))
    monkeypatch.setattr(R, "_job_replacement_offer",
                        lambda jctx, excl=None, **k: None)
    emitted = []
    monkeypatch.setattr(journal, "_job_handoff_emit",
                        lambda jc, ev, **f: emitted.append((ev, f)) or {})
    return emitted


def test_eviction_decision_record_carries_all_ten_sel_keys(monkeypatch):
    """H1, both destinations. `disk_floor_gb` / `disk_blocked` are the pair the
    peer-49bc0103 change added and the pair most likely to be dropped by a
    "tidy up the decision dict" refactor."""
    emitted = _eviction_seams(monkeypatch)
    jc = _jc()                                    # no launch anchor -> refusal
    assert R._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "why") is False
    b2 = [f for e, f in emitted if e == "eviction_replacement_decision"]
    assert len(b2) == 1 and SEL_KEYS <= set(b2[0])
    ladder = [f for n, f in jc["ladder_journal"]
              if n == "eviction_replacement_decision"]
    assert len(ladder) == 1 and SEL_KEYS <= set(ladder[0])
    # ...and the values, on the empty-market path the incident produced
    assert b2[0]["disk_floor_gb"] == 50 and b2[0]["disk_blocked"] is False
    assert b2[0]["spot_candidates"] == 0 and b2[0]["ranked_by"] == "price"


def test_eviction_refusal_dedups_on_the_reason_string(monkeypatch):
    """H5. A stuck eviction re-runs this every ~50s; box 47398836 wrote 79
    byte-identical refusals in 66 min. The DECISION is still re-made every tick
    — only the announcement dedups — and a changed reason announces again."""
    emitted = _eviction_seams(monkeypatch)
    jc = _jc()
    for _ in range(3):
        assert R._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "why") is False
    assert len([e for e, _ in emitted if e == "eviction_replacement_decision"]) == 1
    assert jc["replacement_refused"]
    jc["replacement_refused"] = "a different bound"
    R._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "why")
    assert len([e for e, _ in emitted if e == "eviction_replacement_decision"]) == 2


def test_eviction_excludes_the_lost_machine_before_the_search(monkeypatch):
    """The one host choice we have positive evidence against. Recorded with its
    class so the TTL applies, and BEFORE the offer search, not after the
    launch."""
    seen = []
    _eviction_seams(monkeypatch)
    monkeypatch.setattr(R, "_job_replacement_offers",
                        lambda jctx, excl=None, **k: seen.append(excl) or [])
    jc = _jc()
    R._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "why")
    assert seen and all(e == [7] for e in seen)
    assert jc["evicted_machines"] == {7}


def test_spot_walk_prices_against_the_market_never_the_bid_row(monkeypatch):
    """doc-50 / H4. On a BID offer `dph_total` is the CURRENT INTERRUPTIBLE
    price (~min_bid + 0.5%); using it makes every candidate look like a machine
    whose on-demand rate sits a tenth of a cent over its own floor — `thin` by
    construction, and $3.4741/hr on 2026-08-05."""
    o = _offer(oid=1, machine=11, min_bid=0.80, dph=0.804)
    monkeypatch.setattr(pricing, "_market_ondemand_soft", lambda mid, n=None: 2.40)
    cands = R._replacement_spot_walk([o], 3.0, 2)
    assert len(cands) == 1
    assert cands[0].ondemand == 2.40 and cands[0].ondemand != o["dph_total"]


def test_spot_walk_probes_once_per_machine_and_drops_the_unpriced_tail(monkeypatch):
    """Bounded, and the bound DROPS the tail rather than guessing its price —
    the jobs-lane half of the pinned probe-budget divergence (it keeps the
    priced prefix; the run lane's `_handoff_pick_offer` returns None)."""
    probes = []
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, n=None: probes.append(mid) or 2.40)
    offers = [_offer(oid=1, machine=11), _offer(oid=2, machine=11),
              _offer(oid=3, machine=12), _offer(oid=4, machine=13)]
    cands = R._replacement_spot_walk(offers, 3.0, 2, max_probes=2)
    assert probes == [11, 12]
    assert [c.offer["id"] for c in cands] == [1, 2, 3]


def test_ondemand_walk_may_read_dph_total(monkeypatch):
    """The doc-50 ban is on reading `dph_total` off a BID row. On an
    ONDEMAND-type offer that field IS the on-demand price, so no probe is
    needed — and no probe must happen."""
    monkeypatch.setattr(pricing, "_market_ondemand_soft",
                        lambda mid, n=None: pytest.fail("probed the on-demand rung"))
    cands = R._replacement_ondemand_walk([_offer(oid=1, dph=1.60)], 2)
    assert [(c.price, c.ondemand) for c in cands] == [(1.60, 1.60)]


def test_replacement_rank_is_all_or_nothing_on_measured_rates(monkeypatch):
    """Mixing a measured tok/s against an assumed one ranks money on an
    assumption, and the assumption always favours the class we happened to
    measure. One unmeasured candidate reverts the WHOLE set to cheapest-first —
    which reproduces the pre-2026-08-16 pick exactly."""
    C = R.ReplacementCandidate
    cheap = C(_offer(oid=1), 0.50, 1.0, 100.0, 200.0, None)
    fast = C(_offer(oid=2), 1.00, 2.0, 900.0, 900.0, None)
    assert [c.offer["id"] for c in R._replacement_rank([cheap, fast])] == [2, 1]
    unmeasured = C(_offer(oid=3), 0.75, 1.5, None, None, None)
    assert [c.offer["id"]
            for c in R._replacement_rank([cheap, fast, unmeasured])] == [1, 3, 2]


def test_launch_job_replacement_ceiling_binds_on_the_repicked_offer(monkeypatch):
    """doc-50 R3. The rail did NOT bind here on 2026-08-05: the internal re-pick
    ran with no ceiling and bought a $3.4741/hr on-demand box against a $2.164
    ceiling the decision record claimed to respect. A rail that binds only in
    the pure decision is not a rail."""
    pricey = _offer(oid=9, dph=3.4741)
    monkeypatch.setattr(R, "_job_replacement_offer",
                        lambda jctx, excl=None, **k: pricey)
    monkeypatch.setattr(launchmod, "_do_launch",
                        lambda ns: pytest.fail("rented over the ceiling"))
    jc = _jc()
    assert R._launch_job_replacement(jc, [], rental="ondemand",
                                     max_dph=2.164) == (None, None, "over_ceiling")
    assert "over the $2.164 replacement ceiling" in jc["last_error"]


def test_launch_job_replacement_returns_the_realized_ondemand_rate(monkeypatch):
    """doc-50 R4. `_do_launch` reads a price off SEARCHED offers only, and this
    lane always PINS one — so the on-demand rung returned None and the journal
    recorded `ondemand @ $None/hr` while a real meter ran."""
    monkeypatch.setattr(launchmod, "_do_launch", lambda ns: ("NEW", 9, None))
    jc = _jc()
    cid, dph, reason = R._launch_job_replacement(
        jc, [], offer=_offer(oid=9, dph=1.603), rental="ondemand", max_dph=2.0)
    assert (cid, dph, reason) == ("NEW", 1.603, None)


def test_launch_job_replacement_names_the_disk_bound_on_an_empty_market(monkeypatch):
    """`no qualifying offer` alone sends the operator hunting for a price
    problem that isn't there — on host 67231's A100 book the bound was disk
    every time."""
    monkeypatch.setattr(R, "_job_replacement_offer", lambda jctx, excl=None, **k: None)
    jc = _jc(launch_disk_gb=110)
    assert R._launch_job_replacement(jc, []) == (None, None, "no_offer")
    assert ">= 110G of container disk" in jc["last_error"]


def test_launch_job_replacement_answers_whether_disk_is_the_bound(monkeypatch):
    """Pins that the pull-condemn refusal names the disk floor AND says whether
    that floor is what emptied the market."""
    monkeypatch.setattr(R, "_job_replacement_offer", lambda jctx, excl=None, **k: None)
    # The UNFLOORED probe finds boxes — they are just all too small.
    monkeypatch.setattr(R, "_job_replacement_offers",
                        lambda jctx, excl=None, **k: [_offer(machine=67231, disk=23.0),
                                                      _offer(machine=67232, disk=47.0)])
    jc = _jc(launch_disk_gb=70)
    assert R._launch_job_replacement(jc, []) == (None, None, "no_offer")
    err = jc["last_error"]
    assert ">= 70G of container disk" in err        # the pre-existing half
    assert "DISK FLOOR" in err                      # ...and the answer
    assert "47G" in err and "67232" in err          # the biggest candidate, named


def test_launch_job_replacement_stays_quiet_when_disk_is_not_the_bound(monkeypatch):
    """Negative control: a market emptied by price must not get a disk verdict
    appended."""
    monkeypatch.setattr(R, "_job_replacement_offer", lambda jctx, excl=None, **k: None)
    monkeypatch.setattr(R, "_job_replacement_offers",
                        lambda jctx, excl=None, **k: [_offer(machine=67233, disk=500.0)])
    jc = _jc(launch_disk_gb=70)
    assert R._launch_job_replacement(jc, []) == (None, None, "no_offer")
    assert jc["last_error"].endswith("requires >= 70G of container disk)")
    assert "DISK FLOOR" not in jc["last_error"]


def test_launch_job_replacement_sizes_from_the_workload_anchor(monkeypatch):
    """task #69: `launch_disk_gb` is the watch's IMMUTABLE anchor, so one
    under-sized hop cannot propagate down the chain (driftr3 went
    110 -> 110 -> 60 GB and died on its own disk guard)."""
    got = []
    monkeypatch.setattr(launchmod, "_do_launch",
                        lambda ns: got.append(ns) or ("NEW", 9, 0.5))
    jc = _jc(launch_disk_gb=110, instances=[{"id": "41", "disk_space": 60,
                                             "disk_usage": 5.0, "num_gpus": 2}])
    R._launch_job_replacement(jc, [], offer=_offer(), rental="bid", price=0.12)
    assert got[0].disk == 110
    assert jc["last_replacement_disk_gb"] == 110


def test_replacement_disk_need_falls_back_loudly(monkeypatch):
    """`known=False` is the third element for exactly one reason: the launcher
    says so out loud, and the search still carries the number because the launch
    is going to ask vast for it either way."""
    gb, why, known = R._replacement_disk_need(_jc(launch_disk_gb=None), {})
    assert known is False and gb and why
    gb, _why, known = R._replacement_disk_need(_jc(launch_disk_gb=110), {})
    assert (gb, known) == (110, True)


def test_replacement_verified_parses_its_bool_by_hand(monkeypatch):
    """H10. `_rebid_knob` coerces with `type(default)(v)` and `bool("0")` is
    True, so routing this through it would silently ignore every disable."""
    jc = _jc()
    assert R._job_replacement_verified(jc) is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("JOB_REPLACEMENT_VERIFIED", off)
        assert R._job_replacement_verified(jc) is False
    monkeypatch.setenv("JOB_REPLACEMENT_VERIFIED", "1")
    assert R._job_replacement_verified(jc) is True
    # the namespace outranks the env
    assert R._job_replacement_verified(
        _jc(a=_a(replacement_verified=False))) is False


def test_gpu_rate_soft_is_none_first_class(monkeypatch):
    """A rate we have not measured must never be guessed — an invented tok/s
    silently re-ranks real money. Module absent, class unmeasured and any error
    all answer None, never 0.0."""
    assert R._gpu_rate_soft(None) is None
    assert R._gpu_rate_soft("") is None

    class _Boom:
        @staticmethod
        def rate_for(name, num_gpus=1):
            raise RuntimeError("no table")

    monkeypatch.setitem(sys.modules, "gpu_rates", _Boom)
    assert R._gpu_rate_soft("H200 NVL", 2) is None

    class _Zero:
        @staticmethod
        def rate_for(name, num_gpus=1):
            return 0.0

    monkeypatch.setitem(sys.modules, "gpu_rates", _Zero)
    assert R._gpu_rate_soft("H200 NVL", 2) is None


def test_cross_module_calls_stay_late_bound():
    """Plan §8(b): every cross-module reference in this ring is resolved as a
    module ATTRIBUTE at call time. A `from … import _do_launch` would bind at
    import time, and the 659 patch sites that steer these drivers would go
    vacuously green while a real launch happened."""
    import inspect
    src = inspect.getsource(R)
    for bad in ("from vastlib.launch.launch import",
                "from vastlib.core.models import",
                "from vastlib.supervise.journal import",
                "from vastlib.market.offers import"):
        assert bad not in src
    for attr in ("_do_launch", "_num_dph", "_job_handoff_emit", "pick_offers"):
        assert not hasattr(R, attr), f"{attr} is bound as a module global"


# --------------------------------------------------------------------------- #
# 9. The notify-S2b re-port (peers 830579df..d5b0b773, applied 2026-08-16).
#
#    Two seams land in THIS module: the extracted defense inputs the rescue
#    quote's ceiling shares with the re-bid rung, and the `exclusion_class`
#    that keeps a notification from shortening a MACHINE exclusion. The flat
#    suite (`test_notify_policy.py`) proves the same properties against the
#    still-live `herdd` copies and stays UNEDITED; this is the parallel net
#    for the package copies.
# --------------------------------------------------------------------------- #
def test_defense_inputs_returns_the_six_and_is_None_safe():
    """The six numbers `bidpolicy.defense_ceiling` reads, off one tick's `jc`.
    With no `pending_views` and no fresh `p_alt` every consumer must be
    byte-identical to its pre-defense self, so `p_alt` is None and the horizon
    is None — NOT 0.0 (defect #67's shape: an UNKNOWN horizon collapsed to zero
    prices a migration against no remaining work)."""
    di = R._job_defense_inputs(_jc(), NOW)
    assert set(di) == {"p_alt", "remaining_h", "ckpt_interval_h", "defend",
                       "prior_runtime_h", "setup_h"}
    assert di["p_alt"] is None and di["remaining_h"] is None
    assert di["ckpt_interval_h"] == 0.0        # a WIDTH, legitimately zero
    assert di["setup_h"] == bp.SPOT_SETUP_H


def test_defense_inputs_takes_the_widest_checkpoint_interval():
    jc = _jc(pending_views=[{"checkpoint_s": 900}, {"checkpoint_s": 3600},
                            {"checkpoint_s": None}])
    assert R._job_defense_inputs(jc, NOW)["ckpt_interval_h"] == 1.0


def test_defense_inputs_survives_a_horizon_read_that_raises(monkeypatch):
    """A ceiling read never kills a tick. `_jobs_work_horizon_h` walks live job
    views; a malformed one used to take the whole watch down with it."""
    monkeypatch.setattr(risk, "_jobs_work_horizon_h",
                        lambda v, n: (_ for _ in ()).throw(KeyError("eta")))
    assert R._job_defense_inputs(_jc(pending_views=[{}]),
                                 NOW)["remaining_h"] is None


def test_defense_cap_is_None_without_a_fresh_p_alt():
    """No fresh replacement-market read means no DERIVABLE defense — and None
    here is what keeps the notification-priced quote at its undefended ceiling.
    A 0.0 would refuse every rescue on every box that never polled p_alt."""
    assert R._job_defense_cap(_jc(), NOW) is None


def test_defense_cap_is_the_scalar_defense_ceiling_derives():
    """The verifier's state: `p_alt` $0.60 against 20 h of work left. The rung
    and the rescue must read ONE number, so this is asserted against
    `bidpolicy.defense_ceiling` directly rather than against a copy of its
    arithmetic."""
    jc = _jc(p_alt=0.60, p_alt_ts=NOW,
             pending_views=[{"job_id": "j1", "eta_s": 20 * 3600}])
    cap = R._job_defense_cap(jc, NOW)
    di = R._job_defense_inputs(jc, NOW)
    assert cap == bp.defense_ceiling(**di)[0]
    assert cap is not None and cap > 0


def test_defense_cap_never_raises_out_of_a_tick(monkeypatch):
    monkeypatch.setattr(bp, "defense_ceiling",
                        lambda **k: (_ for _ in ()).throw(ValueError("boom")))
    assert R._job_defense_cap(_jc(p_alt=0.60, p_alt_ts=NOW), NOW) is None


def test_the_rebid_rung_derives_its_defense_through_the_shared_extractor(
        monkeypatch):
    """Review round 2's actual fix: ONE derivation of the six numbers, with two
    callers. A ceiling assembled from two derivations is not the same ceiling
    even when it agrees today, so the rung must go THROUGH `_job_defense_inputs`
    — patched here, and the patch has to land."""
    seen = []
    real = R._job_defense_inputs
    monkeypatch.setattr(R, "_job_defense_inputs",
                        lambda jc, now: seen.append(now) or real(jc, now))
    monkeypatch.setattr(lifecycle, "_put_bid_soft",
                        lambda iid, price: (True, None))
    jc = _jc(last_bid=0.45, max_bid=2.999, launch_dph_anchor=1.20,
             rebid_rungs=0)
    R._job_rebid_ladder(jc, _a(budget=5.0), "41", 1.01, 3.0,
                        bp.EVICTION_OUTBID, NOW)
    assert seen == [NOW], "the rung derived its own six numbers again"


def test_the_exclusion_is_keyed_on_the_bare_class_when_one_is_supplied(
        monkeypatch):
    """F2/M2. A notification REFINES the class we act on; it must not shorten
    how long we remember that this machine took our box. `unknown` is permanent
    in `EVICTED_TTL_CLASSES` and `outbid` ages out at 30 min, so a row that
    refined `unknown -> outbid` un-excluded the machine and let the very next
    replacement probe re-rent it."""
    _eviction_seams(monkeypatch)
    jc = _jc()
    R._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "why",
                            exclusion_class=bp.EVICTION_UNKNOWN)
    assert jc["evicted_machine_ts"]["7"]["class"] == bp.EVICTION_UNKNOWN
    aged = NOW + R.EVICTED_EXCLUSION_TTL_S + 1
    assert R._job_excluded_machines(jc, aged) == {7}, "still excluded"


def test_no_exclusion_class_means_the_eviction_class_verbatim(monkeypatch):
    """`None` = "same as `eviction_class`", which is every pre-S2b caller and
    every test. The default has to be byte-identical or the S2b boundary moved
    for callers that never opted in."""
    _eviction_seams(monkeypatch)
    jc = _jc()
    R._job_eviction_replace(jc, None, bp.EVICTION_OUTBID, "why")
    assert jc["evicted_machine_ts"]["7"]["class"] == bp.EVICTION_OUTBID
    aged = NOW + R.EVICTED_EXCLUSION_TTL_S + 1
    assert R._job_excluded_machines(jc, aged) == set(), "the TTL still expires"


def test_both_box_swap_sites_retire_the_latch_by_module_attribute():
    """A swap retires the CONSUMED set too — it is keyed to a box we no longer
    hold, and the new box has a new instance id, so nothing in it can ever
    match again.

    Both swap sites live in THIS module and the reset lives in `job_lane`, so
    the call has to be written as a module ATTRIBUTE: a `from … import` would
    bind at import time and every `monkeypatch.setattr(job_lane, …)` steering
    these paths — here and in the 659 flat patch sites — would go vacuously
    green. Asserted on the source because both sites sit at the END of a
    several-hundred-line driver whose success path rents a real box."""
    import inspect
    for fn in (R._job_pull_condemn, R._job_eviction_replace):
        assert "job_lane._job_notify_box_swap_reset(jc)" in inspect.getsource(fn), (
            f"{fn.__name__} must reach the reset by module attribute")


def test_the_box_swap_reset_drops_all_three_notify_keys():
    """The unit both swap sites call. `notify_consumed_ids` is the one that
    survives a return-to-live and a deploy-gate flap; a BOX SWAP is the only
    thing that may retire it."""
    jc = _jc(notify_matched={"iid": "41", "event_id": "e1"},
             notify_consumed_ids=["e1"], notify_quote_said=True)
    job_lane._job_notify_box_swap_reset(jc)
    assert "notify_matched" not in jc and "notify_consumed_ids" not in jc
    assert "notify_quote_said" not in jc


# =============================================================================
# _serve_self_park_soft — marker parse + freshness
# =============================================================================
# MIGRATED from `test_supervise.py`'s MIGRATION-BLOCKED group (step 6 leftovers).
# It was blocked on TWO things and both are gone: `SERVE_SELF_PARK_FRESH_S` now
# has a vastlib home, and `job_lane._serve_self_park_soft` is no longer a raising
# stub — the definition landed HERE, with the rest of the serve cluster, and the
# jobs tick reaches it as `replacement._serve_self_park_soft(...)`.
#
# The `_rclone_soft` seam is stubbed at `storage.b2`, its owning module, which is
# also the only I/O this group can reach: no network, no B2, no subprocess.

def _mark_env(monkeypatch, line, rc=0):
    monkeypatch.setenv("B2_BUCKET", "bkt")
    monkeypatch.setattr(b2, "_rclone_soft", lambda args: (rc, line, ""))


def _iso(now, ago_s):
    import datetime as _dt
    return _dt.datetime.fromtimestamp(
        now - ago_s, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_self_park_fresh_marker_is_true(monkeypatch):
    now = time.time()
    _mark_env(monkeypatch, f"SELF_PARKED {_iso(now, 60)} max_hours\n")
    assert R._serve_self_park_soft("sv1") is True


def test_self_park_legacy_failed_max_hours_wire_is_true(monkeypatch):
    """Pre-2026-08-02 serve_vllm.sh wrote `FAILED <ts> max_hours` for the same
    event — a mid-upgrade box must still classify as a self-park."""
    now = time.time()
    _mark_env(monkeypatch, f"FAILED {_iso(now, 60)} max_hours\n")
    assert R._serve_self_park_soft("sv1") is True


def test_self_park_STALE_marker_is_false(monkeypatch):
    """The marker outlives the park that wrote it: an hour-old self-park must
    never explain away a LATER genuine eviction."""
    now = time.time()
    _mark_env(monkeypatch,
              f"SELF_PARKED {_iso(now, R.SERVE_SELF_PARK_FRESH_S + 120)} "
              f"max_hours\n")
    assert R._serve_self_park_soft("sv1") is False


@pytest.mark.parametrize("line,rc", [
    ("READY 2026-08-02T00:00:00Z m1\n", 0),   # healthy marker: not a park
    ("FAILED 2026-08-02T00:00:00Z oom\n", 0),  # real failure: rescue, not park
    ("", 0),                                   # empty
    ("SELF_PARKED garbage max_hours\n", 0),    # unparseable ts
    ("SELF_PARKED 2026-08-02T00:00:00Z\n", 1),  # read failed
])
def test_self_park_fails_toward_rescue(monkeypatch, line, rc):
    _mark_env(monkeypatch, line, rc=rc)
    assert R._serve_self_park_soft("sv1") is False


def test_self_park_no_serve_id_or_bucket_is_false(monkeypatch):
    monkeypatch.delenv("B2_BUCKET", raising=False)
    assert R._serve_self_park_soft(None) is False
    assert R._serve_self_park_soft("sv1") is False


def test_the_max_age_s_keyword_survived_the_move():
    """The flat signature is `(serve_id, *, max_age_s=SERVE_SELF_PARK_FRESH_S)`.
    The keyword is not decoration: it is the knob that makes the freshness
    window testable at all, and `job_lane`'s stub docstring called it out as
    "part of the real signature" precisely so the port could not quietly drop
    it into a hardcoded constant."""
    import inspect
    sig = inspect.signature(R._serve_self_park_soft)
    p = sig.parameters["max_age_s"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default == R.SERVE_SELF_PARK_FRESH_S == 3600
