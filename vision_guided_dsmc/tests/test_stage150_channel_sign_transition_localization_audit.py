import numpy as np
import pytest

from vgdsmc.stage150_channel_sign_transition_localization_audit import (
    NO_TRANSITION, OPPOSING_NO_SINGLE_SIDE, OUTER_SIDE_DOMINANCE_MIN, SINGLE_SIDE,
    channel_sign_transition_metrics, classify_channel_sign_transition, validate_stage150_design,
)


def _metrics():
    dominant = np.array([-0.069427648879256731, -0.062306596511182977, -0.048164895063305457, -0.046554304290548687, -0.035624223271658984])
    parent = np.array([-0.065730703750408903, -0.045005698926767979, -0.038995271013813837, -0.028636898781650211, -0.019052011035461724])
    fine = np.array([0.0062655553375603745, 0.0021739721596047423, 0.008439527497165117])
    coarse = np.array([0.0043610410121524, -0.003396086379121477, 0.0009649546330309233])
    return channel_sign_transition_metrics(dominant, parent, fine, coarse)


def test_stage150_frozen_design_accepts_only_registered_values():
    validate_stage150_design()
    with pytest.raises(ValueError): validate_stage150_design(kn0=9.0)
    with pytest.raises(ValueError): validate_stage150_design(limiter="vanleer")
    with pytest.raises(ValueError): validate_stage150_design(physical_parameter_retuning=True)


def test_stage150_parent_channel_flips_but_dominant_does_not():
    m = _metrics()
    assert m["parent"]["channel_sign_transition"] is True
    assert m["dominant"]["channel_sign_transition"] is False


def test_stage150_outer_interval_decomposition_closes_exactly():
    m = _metrics()
    assert m["parent"]["outer_increment_closure"] < 1e-12
    assert m["dominant"]["outer_increment_closure"] < 1e-12
    assert m["maximum_identity_or_provenance_error"] < 1e-12


def test_stage150_parent_outer_contributions_oppose_without_75pct_dominance():
    m = _metrics()["parent"]
    assert m["outer_sign_product"] < 0
    assert m["stronger_outer_absolute_share"] < OUTER_SIDE_DOMINANCE_MIN
    assert m["outer_cancellation_fraction"] > 0.5


def test_stage150_observed_route_is_two_sided_asymmetry_audit():
    assert classify_channel_sign_transition(metrics=_metrics()) == OPPOSING_NO_SINGLE_SIDE


def test_stage150_alternate_routes_remain_preregistered():
    m = _metrics(); m["parent"]["stronger_outer_absolute_share"] = 0.9
    assert classify_channel_sign_transition(metrics=m) == SINGLE_SIDE
    m = _metrics(); m["parent"]["channel_sign_transition"] = False
    assert classify_channel_sign_transition(metrics=m) == NO_TRANSITION
