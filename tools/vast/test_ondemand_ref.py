"""The doc 50 R1 defect family, closed at the LAUNCH and HANDOFF paths.

On a BID-view vast bundles row, `dph_total` is the CURRENT INTERRUPTIBLE
price (min_bid + the storage sliver) and `dph_base` EQUALS min_bid — neither
carries the machine's on-demand rate (API-verified 2026-08-06: min_bid 0.2667 /
dph_total 0.2711 on a machine whose ON-DEMAND view lists dph_total 0.5111).
Feeding a bid row's `dph_total` to `_auto_bid_price` as "on-demand" clamps the
bid onto its own floor: `min(1.2 x floor, dph_total - 0.001)` == the lowest
priority bid the machine can hold. That is how understudy 46909754 was bid
$1.071 over a $1.0667 floor (lost 45 min later) and understudy 46934673 a
razor-thin $0.401 on 2026-08-06 while the design target is 1.2x the floor.

Every fixture offer here uses the REAL bid-view shape (dph_total ~= min_bid,
memory rule synthetic-repro-must-use-real-producer-output) — the pre-fix tests
missed the defect precisely because their fake offers carried dph_total = 5x
min_bid, so the wrong clamp never bound.

Each test FAILS against the reintroduced defect (verified by stashing the
herdd fix and re-running: 8 failed — including the exact razor price, 1.07
where the design says 1.28 — then 8 passed with the fix back).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageref  # noqa: E402
from vastlib.core import fmt  # noqa: E402
from vastlib.launch import launch  # noqa: E402
from vastlib.market import offers as market_offers, pricing  # noqa: E402
from vastlib.supervise import replacement  # noqa: E402


# A REAL-shape bid-view offer: dph_total is floor + storage sliver, dph_base
# is the floor again, and the machine's true on-demand rate (1.60) is NOT on
# the row — it only exists market-side.
_BID_OFFER = {"id": 999, "machine_id": 777, "num_gpus": 1,
              "min_bid": 1.0667, "dph_total": 1.0713, "dph_base": 1.0667}
_MARKET_OD = 1.60


def _patch_market(monkeypatch, od=_MARKET_OD, home=pricing):
    """`home` is the module the SUBJECT resolves `_market_ondemand_soft` through:
    `vastlib.market.pricing` for every subject in this file. The parameter
    survives the step-6e migration because it is the seam-placement knob — the
    flat arm is gone, but a future subject in another module would use it."""
    calls = []
    monkeypatch.setattr(home, "_market_ondemand_soft",
                        lambda mid, g=None: (calls.append((mid, g)) or od))
    return calls


def _handoff_st(**over):
    # primary paid 3.20/hr on a 3.20 on-demand machine (pref ceiling 1.60),
    # 10h wall left — the _BID_OFFER candidate genuinely qualifies at its full
    # cushioned target (1.173 <= 2.40; savings (3.20-1.173)*10 >> the
    # 2x-window overhead).
    st = {"run_id": "r1", "dph_total": 3.20, "last_bid": 3.20,
          "on_demand": 3.20, "remaining_wall_h": 10.0,
          "launch_spec": {"image": "reg/img:tag", "disk": 100,
                          "runtype": "ssh_direct", "env": {"RUN_ID": "r1"},
                          "secret_env_keys": []}}
    st.update(over)
    return st


def test_offer_ondemand_ref_reads_market_not_bid_row(monkeypatch):
    calls = _patch_market(monkeypatch)
    assert pricing._offer_ondemand_ref(_BID_OFFER) == _MARKET_OD
    # probed for THIS machine and the offer's own GPU chunk
    assert calls == [(777, 1)]
    assert pricing._offer_ondemand_ref(None) is None


def test_understudy_bid_is_full_cushion_not_razor_thin(monkeypatch):
    """_handoff_understudy_body must bid 1.2x the floor under the REAL market
    on-demand — not min(1.2x, dph_total - eps) == the floor + a rounding unit
    (the 46909754 shape: this test bids $1.28, the defect bid $1.071)."""
    # MIGRATED (was MIGRATION-BLOCKED, step 6e batch B7): the blocker is gone.
    # `_relaunch_body` landed in `vastlib.supervise.replacement` — the same
    # module as `_handoff_understudy_body`, so the :2643 -> :219 call reaches a
    # real body and subject + seam both repoint (seam `_market_ondemand_soft` at
    # its home `vastlib.market.pricing`, which is `_patch_market`'s default).
    _patch_market(monkeypatch)
    a = argparse.Namespace(dry_run=True)
    body, bid, missing = replacement._handoff_understudy_body(
        _handoff_st(), a, dict(_BID_OFFER))
    # 2026-08-08 cushion audit: floor/on-demand here is 1.0667/1.60 = 0.667,
    # above the ~0.59 crossover, so the SURVIVAL CUSHION (1.10 x floor =
    # $1.173) prices this rather than the 0.65 x on-demand cost cap ($1.04).
    # It was $1.28 (1.2 x floor) before. What this test pins is unchanged:
    # the number must come from the MARKET on-demand read, never the bid
    # row's $1.0713 dph_total, which would price it at $1.070 (razor-thin).
    assert bid == 1.173
    assert body is not None and body["price"] == 1.173


def test_understudy_body_refuses_when_market_od_unreadable(monkeypatch):
    # None from the market read must REFUSE (missing §2.3 input), never fall
    # back to the bid row's dph_total.
    _patch_market(monkeypatch, od=None)
    a = argparse.Namespace(dry_run=True)
    body, bid, reason = replacement._handoff_understudy_body(
        _handoff_st(), a, dict(_BID_OFFER))
    assert body is None and reason == "candidate_reject"


def test_pick_offer_filter_rejects_razor_only_candidate(monkeypatch):
    """A candidate that fits under the primary's preferred ceiling ONLY
    because the wrong clamp compressed its target onto the floor must be
    rejected once the REAL on-demand is read: primary od 0.50 -> pref ceiling
    0.75 x 0.50 = 0.375; floor 0.32 -> true target 1.2x = 0.384 > 0.375. The
    defect passed it (target razor-clamped onto its own dph_total).

    Fixture history: written with floor 0.22 against the old 0.50-frac $0.25
    ceiling; still rejected through the 2026-08-08 2.00x era (2.0 x 0.22 =
    0.44 > 0.375); floor raised 0.22 -> 0.32 when BID_TARGET_MULT returned to
    1.20 (owner ruling 2026-08-09) because 1.2 x 0.22 = 0.264 would fit under
    the 0.375 line — same reject branch, recomputed fixture."""
    razor = {"id": 1, "machine_id": 5, "num_gpus": 1,
             "min_bid": 0.32, "dph_total": 0.3211, "dph_base": 0.32}
    _patch_market(monkeypatch, od=0.60)
    monkeypatch.setattr(market_offers, "_search_offers_soft", lambda a: [razor])
    st = _handoff_st(dph_total=0.60, last_bid=0.60, on_demand=0.50)
    assert replacement._handoff_pick_offer(st, argparse.Namespace()) is None


def test_pick_offer_probe_cap_refuses_not_spins(monkeypatch):
    # more distinct machines than HANDOFF_ODPROBE_MAX and no qualifier ->
    # bounded probes, then None (get-and-hold), never an unbounded probe walk.
    calls = _patch_market(monkeypatch, od=None)
    offers = [{"id": i, "machine_id": i, "num_gpus": 1,
               "min_bid": 0.22, "dph_total": 0.2211}
              for i in range(pricing.HANDOFF_ODPROBE_MAX + 5)]
    monkeypatch.setattr(market_offers, "_search_offers_soft", lambda a: offers)
    assert replacement._handoff_pick_offer(_handoff_st(), argparse.Namespace()) is None
    assert len(calls) == pricing.HANDOFF_ODPROBE_MAX


def test_job_understudy_prices_full_cushion(monkeypatch):
    """The jobs-lane understudy launch (_launch_job_understudy) is the path
    that actually bid $1.071 (46909754) and $0.401 (46934673) — it must price
    1.2x the floor under the MARKET on-demand, and pass the same reference to
    the §2.3 re-check."""
    _patch_market(monkeypatch)
    seen = {}

    def fake_do_launch(ns):
        seen["price"] = ns.price
        return "iid-u1", None, ns.price
    monkeypatch.setattr(launch, "_do_launch", fake_do_launch)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    jctx = {"iid": "1", "last_bid": 3.20, "dph": 3.20, "on_demand": 3.20,
            "remaining_wall_h": 10.0, "instances": [], "dry_run": True,
            "now": 0.0}
    hf = {"chosen_offer": dict(_BID_OFFER)}
    iid, dph, reason = replacement._launch_job_understudy(jctx, hf, epoch=1)
    assert reason is None, jctx.get("last_error")
    assert seen["price"] == 1.173            # not 1.070 (razor) — cushioned


def test_do_launch_search_path_bid_clamps_to_market_od(monkeypatch):
    """The default auto-priced spot LAUNCH (search path) must bid 1.2x the
    floor: with the defect it bid min(1.2x, bid-row dph_total - 0.001) ==
    floor + storage sliver on EVERY auto-priced spot launch."""
    _patch_market(monkeypatch)
    monkeypatch.setattr(market_offers, "search_offers", lambda a: [dict(_BID_OFFER)])
    monkeypatch.setattr(fmt, "fmt_offer", lambda o: "offer-999")
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = argparse.Namespace(
        offer=None, type="bid", price=None, env=None, port=None, jupyter=False,
        onstart=None, no_hf_token=True, hf_token=None, ssh=False,
        ssh_key_file=None, jobs=False, image="img:tag", disk=40,
        runtype="ssh_direct", label=None, template_id=None,
        no_registry_login=True, login=None, dry_run=True, wait=None,
        force=False, num_gpus=1)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        launch._do_launch(ns)
    assert ns.price == 1.173                 # cushioned, not 1.070


def test_do_launch_pinned_path_never_takes_bid_view_dph_total(monkeypatch):
    """The pinned --offer path must clamp on the MARKET on-demand read, never on
    the bid row's own `dph_total`.

    2026-08-09: the rung that recovers the row is `_offer_machine_scan_soft`
    (an unfiltered query matched on the id in Python), not `_offer_pricing_soft`
    — vast's `id` filter answers HTTP 200 with zero rows in every view, so the
    old version of this test faked a response the live API cannot produce."""
    calls = _patch_market(monkeypatch)
    monkeypatch.setattr(market_offers, "_offer_machine_scan_soft",
                        lambda a: dict(_BID_OFFER, id=555))
    monkeypatch.setattr(
        pricing, "_offer_pricing_soft",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the id filter is dead; nothing may depend on it")))
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    ns = argparse.Namespace(
        offer=555, type="bid", price=None, env=None, port=None, jupyter=False,
        onstart=None, no_hf_token=True, hf_token=None, ssh=False,
        ssh_key_file=None, jobs=False, image="img:tag", disk=40,
        runtype="ssh_direct", label=None, template_id=None,
        no_registry_login=True, login=None, dry_run=True, wait=None,
        force=False, num_gpus=1)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        launch._do_launch(ns)
    assert ns.price == 1.173
    assert calls and calls[0][0] == 777      # clamp came from the market read


# ---------------------------------------------------------------------------
# spot_breakeven — the spot-vs-on-demand switch rule (pure, advisory, unwired)
# ---------------------------------------------------------------------------
import bidpolicy  # noqa: E402


def test_spot_breakeven_livelock_lane_never_wins():
    # v11 eval lane, measured 2026-08-06: lifetimes 11-13 min, setup 11m35s —
    # a full cycle banked ZERO rows. At ANY price ratio spot must lose.
    ok, spot_cost, od = bidpolicy.spot_breakeven(
        spot_dph=0.05, ondemand_dph=1.00, setup_h=0.193,
        expected_lifetime_h=0.19)
    assert ok is False and spot_cost == float("inf")


def test_spot_breakeven_contested_lane_flips_to_ondemand():
    # L = 1 h under contention, setup 11m35s: threshold = 1 - 0.193 = 0.807.
    # A typical 0.53 ratio still wins; a 0.85 ratio loses.
    ok, spot_cost, od = bidpolicy.spot_breakeven(
        spot_dph=0.53, ondemand_dph=1.00, setup_h=0.193,
        expected_lifetime_h=1.0)
    assert ok is True and spot_cost < 1.00
    ok2, _, _ = bidpolicy.spot_breakeven(
        spot_dph=0.85, ondemand_dph=1.00, setup_h=0.193,
        expected_lifetime_h=1.0)
    assert ok2 is False


def test_spot_breakeven_checkpoint_loss_term_binds():
    # Same L=1h lane but a 20-min SAVE_STEPS interval adds E[loss] = 10 min:
    # overhead 0.36h -> threshold 0.64 -> the 0.53 ratio still wins, 0.7 loses.
    ok, _, _ = bidpolicy.spot_breakeven(
        spot_dph=0.70, ondemand_dph=1.00, setup_h=0.193,
        expected_lifetime_h=1.0, ckpt_interval_h=1 / 3)
    assert ok is False


def test_spot_breakeven_calm_market_spot_wins():
    # 24 h lifetime: threshold ~0.992 — the measured 2.2x gap (ratio 0.45)
    # wins comfortably.
    ok, spot_cost, _ = bidpolicy.spot_breakeven(
        spot_dph=0.427, ondemand_dph=0.936, setup_h=0.193,
        expected_lifetime_h=24.0)
    assert ok is True and spot_cost < 0.45


def test_spot_breakeven_refuses_unknowns():
    assert bidpolicy.spot_breakeven(
        spot_dph=None, ondemand_dph=1.0, setup_h=0.2,
        expected_lifetime_h=1.0) == (None, None, None)
    assert bidpolicy.spot_breakeven(
        spot_dph=0.5, ondemand_dph=0, setup_h=0.2,
        expected_lifetime_h=1.0) == (None, None, None)


def test_job_understudy_inherits_the_eval_env_pin(monkeypatch):
    """A handoff understudy is a production box, and the eval-env pin lives in
    the BOX launch env because that is the only thing jobd's boot-time
    `check_venv eval` fetch can read (a job-level pin arrives after the fetch).
    This lane kept its own HANDOFF_* env and dropped the inherited pin, which
    is why the omission read as deliberate.

    Companion to the replacement-lane tests in test_eviction_replacement.py;
    measured failure 2026-08-16, box 47889345."""
    _patch_market(monkeypatch)
    seen = {}

    def fake_do_launch(ns):
        seen["env"] = list(ns.env or [])
        return "iid-u1", None, ns.price
    monkeypatch.setattr(launch, "_do_launch", fake_do_launch)
    monkeypatch.setattr(imageref, "image_tag_digest", lambda img: None)
    jctx = {"iid": "1", "last_bid": 3.20, "dph": 3.20, "on_demand": 3.20,
            "remaining_wall_h": 10.0, "dry_run": True, "now": 0.0,
            "instances": [{"id": 1, "actual_status": "running",
                           "extra_env": [["EVAL_ENV_VER", "20260816-1813-3c0a5f5b"]]}]}
    hf = {"chosen_offer": dict(_BID_OFFER)}
    _iid, _dph, reason = replacement._launch_job_understudy(jctx, hf, epoch=1)
    assert reason is None, jctx.get("last_error")
    assert "EVAL_ENV_VER=20260816-1813-3c0a5f5b" in seen["env"]
    assert any(s.startswith("HANDOFF_EPOCH=") for s in seen["env"]), \
        "inheriting the pin must not displace the lane's own env"
