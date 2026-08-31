"""An empty account is not an outbid, and the alarm must not say it is.

2026-08-25: seven boxes stopped within minutes — four card classes, three
lanes, on-demand and bid alike — because the vast account ran out of credit.
Every relaunch returned HTTP 400 `insufficient_credit`. fleetd journalled
`eviction_class: outbid`, which routes an operator to the bid ladder (where it
then reports "no standing bid to raise") instead of to the billing page.

Insolvency's market shadow is indistinguishable from displacement: the machine
stops listing our chunk either way. So it can only be told apart by the
control-plane refusal, and that is what `account_credit_ok` carries.
"""
import pytest

from bidpolicy import (
    EVICTION_NO_CREDIT,
    EVICTION_OUTBID,
    EVICTION_UNKNOWN,
    classify_eviction,
    credit_ok_from_error,
)

CREDIT_ERR = ("replacement launch failed: error: HTTP 400 on PUT "
              "v0/asks/37158949/: {'error': 'insufficient_credit', "
              "'msg': 'Your account lacks credit; see the billing page.'}")


# --------------------------------------------------------------------------- #
# credit_ok_from_error — tri-state, and never True
# --------------------------------------------------------------------------- #
def test_the_marker_reads_false():
    assert credit_ok_from_error(CREDIT_ERR) is False


@pytest.mark.parametrize("err", [
    None, "", "replacement launch failed: no affordable replacement",
    "unlaunchable: no qualifying offer seen at any price",
    "HTTP 500 on PUT v0/asks/1/: {'error': 'server_error'}",
])
def test_everything_else_reads_none_never_true(err):
    # Not seeing the marker is not evidence of solvency. If this ever returns
    # True, a funded-account claim gets asserted that was never observed.
    assert credit_ok_from_error(err) is None


# --------------------------------------------------------------------------- #
# the arm, on the exact shape the incident produced
# --------------------------------------------------------------------------- #
def _incident_shape(**over):
    """An ON-DEMAND box, stopped, whose machine now lists no bid offers — the
    2026-08-25 shape, which classifies `outbid` on market evidence alone."""
    kw = dict(present=True, actual_status="exited", market_min_bid=None,
              on_demand=1.158, last_bid=None, market_listed=False,
              is_bid=False, notify=None)
    kw.update(over)
    return kw


def test_the_incident_shape_still_reads_outbid_without_the_signal():
    # The defect itself, pinned: absent the credit signal nothing has changed,
    # which is what makes the fix's tri-state default safe.
    assert classify_eviction(**_incident_shape()) == EVICTION_OUTBID


def test_the_credit_signal_reclassifies_it():
    assert classify_eviction(
        **_incident_shape(), account_credit_ok=False) == EVICTION_NO_CREDIT


def test_insolvency_outranks_a_genuine_risen_floor():
    # Both causes "true" at once: the floor really did rise AND we cannot pay.
    # Insolvency wins — no bid rung can buy back a box the account cannot fund,
    # so routing to the ladder would waste the operator's next move.
    assert classify_eviction(present=True, actual_status="exited",
                             market_min_bid=0.90, last_bid=0.45,
                             market_listed=True, is_bid=True,
                             account_credit_ok=False) == EVICTION_NO_CREDIT


def test_it_does_not_fire_on_a_live_box():
    # Ordering guard: the live-state debounce still comes first, so a transient
    # credit blip cannot declare an eviction on a box that never went down.
    assert classify_eviction(present=True, actual_status="running",
                             account_credit_ok=False) == EVICTION_UNKNOWN


def test_none_leaves_every_arm_bit_identical():
    shapes = [
        _incident_shape(),
        dict(present=True, actual_status="exited", market_min_bid=0.90,
             last_bid=0.45, market_listed=True, is_bid=True, notify=None),
        dict(present=True, actual_status="exited", market_min_bid=None,
             on_demand=1.0, last_bid=1.05, market_listed=None, is_bid=True,
             notify=None),
        dict(present=False, actual_status="exited"),
        dict(present=True, actual_status="running"),
    ]
    for kw in shapes:
        assert (classify_eviction(**kw)
                == classify_eviction(**kw, account_credit_ok=None)), kw


def test_end_to_end_from_the_raw_driver_error():
    # The wiring contract: the string the launcher stores must carry a stopped
    # box all the way to the class, with no hand-built intermediate.
    assert classify_eviction(
        **_incident_shape(),
        account_credit_ok=credit_ok_from_error(CREDIT_ERR)) == EVICTION_NO_CREDIT
