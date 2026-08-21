import numpy as np
import pytest

from vgdsmc.stage153_left_channel_endpoint_balance_audit import (
    COARSE_CANCELLATION,
    COARSE_ENDPOINT_CANCELLATION_MIN,
    COARSE_SINGLE_CHANNEL,
    SINGLE_CHANNEL_DOMINANCE_MIN,
    STAGE152_RECORD_BLOCKER,
    classify_left_channel_endpoint_balance,
    left_channel_endpoint_balance_metrics,
    validate_stage153_design,
)


def _metrics():
    dominant_outer = np.array([0.0035605261840368774, -0.005465040509444852])
    parent_outer = np.array([-0.010362502411820462, 0.004792443873094243])
    combined_outer = dominant_outer + parent_outer
    fine = np.array([0.0062655553375603745, 0.0021739721596047423])
    coarse = np.array([0.0043610410121524, -0.003396086379121477])
    return left_channel_endpoint_balance_metrics(
        stage152_dominant_outer=dominant_outer,
        stage152_parent_outer=parent_outer,
        stage152_combined_outer=combined_outer,
        stage150_dominant_outer=dominant_outer,
        stage150_parent_outer=parent_outer,
        stage150_fine_components=fine,
        stage150_coarse_components=coarse,
    )


def test_stage153_frozen_design_rejects_retuning_and_cross_kn_extension():
    validate_stage153_design()
    with pytest.raises(ValueError):
        validate_stage153_design(kn0=0.1)
    with pytest.raises(ValueError):
        validate_stage153_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage153_design(cross_knudsen_extension_permitted=True)


def test_stage153_parent_provenance_and_scale_identity_close():
    m = _metrics()
    assert m["maximum_identity_or_provenance_error"] == 0.0
    assert np.allclose(m["coarse_minus_fine_components"], m["expected_component_scale_increments"])


def test_stage153_fine_endpoint_reinforces_but_coarse_endpoint_opposes():
    m = _metrics()
    assert m["fine"]["channel_sign_product"] == 1
    assert m["coarse"]["channel_sign_product"] == -1
    assert np.isclose(m["fine"]["cross_channel_cancellation_fraction"], 0.0)


def test_stage153_coarse_endpoint_cancellation_is_material():
    m = _metrics()
    assert m["coarse"]["cross_channel_cancellation_fraction"] >= COARSE_ENDPOINT_CANCELLATION_MIN
    assert np.isclose(m["coarse"]["cross_channel_cancellation_fraction"], 0.8756041271003984)
    assert np.isclose(m["combined_coarse_to_fine_magnitude_ratio"], 0.1143375187005501)


def test_stage153_no_single_coarse_channel_reaches_dominance_guard():
    m = _metrics()
    assert m["coarse"]["stronger_absolute_share"] < SINGLE_CHANNEL_DOMINANCE_MIN
    assert m["coarse"]["stronger_channel"] == "dominant"
    assert np.isclose(m["coarse"]["stronger_absolute_share"], 0.5621979364498008)


def test_stage153_only_parent_channel_changes_endpoint_sign():
    m = _metrics()
    assert m["dominant_endpoint_sign_transition"] is False
    assert m["parent_endpoint_sign_transition"] is True


def test_stage153_observed_route_is_coarse_cancellation_without_single_channel_dominance():
    assert classify_left_channel_endpoint_balance(metrics=_metrics()) == COARSE_CANCELLATION


def test_stage153_single_channel_alternate_route_is_preregistered():
    m = _metrics()
    m["coarse"]["stronger_absolute_share"] = 0.80
    assert classify_left_channel_endpoint_balance(metrics=m) == COARSE_SINGLE_CHANNEL


def test_stage153_record_failure_blocks_interpretation():
    assert classify_left_channel_endpoint_balance(metrics=_metrics(), stage152_record_ok=False) == STAGE152_RECORD_BLOCKER
