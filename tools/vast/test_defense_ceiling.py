"""The job-aware ONE-SHOT defense (AUTOBID_DESIGN "Next iteration", owner
direction 2026-08-09): `defense_ceiling` = B_max = p_alt x (1 + (S+L)/R), and
what it does to `rebid_ladder`'s shape when a fresh replacement-market read
exists.

The two worked examples here are THE ones from the design doc — if they drift,
the doc and the code have diverged and one of them is wrong."""

import bidpolicy
from bidpolicy import defense_ceiling, rebid_ladder


# --------------------------------------------------------------------------- #
# defense_ceiling: the pure formula
# --------------------------------------------------------------------------- #

def test_design_doc_worked_example_hours_to_go():
    # R=4h at p_alt $0.45, L=0.05h (checkpoint_s 360): barely above the
    # replacement price — with hours to go, moving is nearly as good as holding.
    price, basis = defense_ceiling(p_alt=0.45, remaining_h=4.0,
                                   ckpt_interval_h=0.1)
    assert price == 0.477
    assert basis == "remaining"


def test_design_doc_worked_example_near_completion():
    # R=0.5h: defense gets MORE valuable near completion — the inverse of a
    # static multiple.
    price, basis = defense_ceiling(p_alt=0.45, remaining_h=0.5,
                                   ckpt_interval_h=0.1)
    assert price == 0.669
    assert basis == "remaining"
    # and strictly above the hours-to-go price for the same market
    assert price > 0.477


def test_unknown_remaining_falls_back_to_the_policy_prior():
    # No R -> SPOT_PRIOR_LIFETIME_H (0.643h), the same "what the policy already
    # asserts" prior replacement_decision's breakeven rung uses. Basis says so.
    price, basis = defense_ceiling(p_alt=0.45)
    assert price == 0.585                     # 0.45 x (1 + 0.193/0.643)
    assert basis == "prior"
    assert defense_ceiling(p_alt=0.45, remaining_h=0) == (0.585, "prior")
    assert defense_ceiling(p_alt=0.45, remaining_h=-1) == (0.585, "prior")


def test_no_replacement_market_read_is_no_licence():
    assert defense_ceiling(p_alt=None) == (None, None)
    assert defense_ceiling(p_alt=0) == (None, None)
    assert defense_ceiling(p_alt=-0.5) == (None, None)
    assert defense_ceiling(p_alt="garbage") == (None, None)


def test_setup_override_is_respected():
    # A lane with a heavier boot defends harder: S=0.5h, R=4h, L=0.
    price, basis = defense_ceiling(p_alt=0.45, remaining_h=4.0, setup_h=0.5)
    assert price == 0.506                     # 0.45 x (1 + 0.5/4)
    assert basis == "remaining"


# --------------------------------------------------------------------------- #
# rebid_ladder with a fresh p_alt: one shot, defense-bounded
# --------------------------------------------------------------------------- #

def _ladder(**kw):
    args = dict(last_bid=0.40, market_min_bid=0.42, on_demand=1.00,
                max_bid=None, rungs_used=0, launch_dph_anchor=0.40,
                eviction_class=bidpolicy.EVICTION_OUTBID)
    args.update(kw)
    return rebid_ladder(**args)


def test_one_meaningful_rebid_at_the_cushioned_market_target():
    # p_alt $0.60, 30 min left, ckpt 360s: B_max = 0.60 x 1.486 = $0.892 —
    # roomy, so the binding ceiling is still 2x the launch anchor min'd with
    # the hard 0.75 x od line. The rung jumps straight to the cushioned target
    # for the CURRENT floor (1.2 x 0.42 = 0.504), not a +25% walk.
    dec = _ladder(p_alt=0.60, remaining_h=0.5, ckpt_interval_h=0.1)
    assert dec.action == "rebid"
    assert dec.price == 0.504
    assert dec.rungs_left == 0                # ...and that was the one shot
    assert "defense cap $0.892" in dec.reason


def test_the_one_shot_is_actually_one_shot():
    # Same market, rung already used: the plain ladder (3 rungs) would keep
    # walking; the job-aware ladder stops and hands the replacement rung the
    # decision.
    dec = _ladder(p_alt=0.60, remaining_h=0.5, ckpt_interval_h=0.1,
                  rungs_used=1)
    assert dec.action == "stop"
    assert "one-shot job-aware defense already spent" in dec.reason
    # contrast: without p_alt the SAME state still rebids (pre-defense shape)
    dec2 = _ladder(rungs_used=1)
    assert dec2.action == "rebid"


def test_defense_ceiling_binds_and_names_itself():
    # Cheap replacement ($0.45) and 4h of runway: B_max $0.477. Standing bid
    # $0.47 leaves less than a cent of headroom, so the ladder refuses and the
    # reason says the REPLACEMENT is the rational defense — that refusal IS the
    # controller working, not a failure.
    dec = _ladder(last_bid=0.47, market_min_bid=0.30, p_alt=0.45,
                  remaining_h=4.0, ckpt_interval_h=0.1)
    assert dec.action == "stop"
    assert dec.ceiling == 0.477
    assert "JOB-AWARE defense ceiling" in dec.reason
    assert "replacement rung IS the rational defense" in dec.reason


def test_p_alt_alone_licenses_a_ceiling():
    # No launch anchor and no --max-bid used to be a hard "cannot derive a
    # ceiling" refusal. A replacement-market read IS a ceiling derivation.
    dec = _ladder(launch_dph_anchor=None, p_alt=0.60, remaining_h=0.5,
                  ckpt_interval_h=0.1)
    assert dec.action == "rebid"
    assert dec.price == 0.504
    # ...and with neither anchor nor p_alt the refusal still stands
    dec2 = _ladder(launch_dph_anchor=None)
    assert dec2.action == "stop"
    assert "cannot derive a re-bid ceiling" in dec2.reason


def test_hard_od_ceiling_survives_a_generous_defense():
    # Near-completion panic pricing cannot buy past the structural rail: B_max
    # = 5.0 x (1 + 0.243/0.1) = $17.15, but 0.75 x od $1.00 still caps the
    # ladder's ceiling at $0.75. Non-negotiable bound, per the design.
    dec = _ladder(last_bid=0.70, market_min_bid=0.72, launch_dph_anchor=2.0,
                  p_alt=5.0, remaining_h=0.1, ckpt_interval_h=0.1)
    assert dec.ceiling == 0.75
    if dec.action == "rebid":
        assert dec.price <= 0.75


def test_ondemand_displacement_still_outranks_the_defense():
    # No p_alt makes an on-demand claim biddable — that refusal is class 1.
    dec = _ladder(eviction_class=bidpolicy.EVICTION_ONDEMAND,
                  p_alt=0.60, remaining_h=0.5)
    assert dec.action == "stop"
    assert "on-demand claim" in dec.reason
