import numpy as np
import pytest

from vgdsmc.stage152_left_outer_channel_balance_audit import (
    MATERIAL_CANCELLATION_MIN,
    MIXED_CANCELLATION,
    PARENT_CHANNEL_DOMINANCE_MIN,
    PARENT_DOMINANT,
    RECORD_BLOCKER,
    classify_left_outer_channel_balance,
    left_outer_channel_balance_metrics,
    validate_stage152_design,
)


def _metrics():
    dominant = np.array([0.0035605261840368774, -0.005465040509444852])
    parent = np.array([-0.010362502411820462, 0.004792443873094243])
    combined = np.array([-0.006801976227783585, -0.0006725966363506086])
    return left_outer_channel_balance_metrics(dominant, parent, combined)


def test_stage152_frozen_design_rejects_retuning_and_cross_kn_extension():
    validate_stage152_design()
    with pytest.raises(ValueError):
        validate_stage152_design(kn0=0.1)
    with pytest.raises(ValueError):
        validate_stage152_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage152_design(cross_knudsen_extension_permitted=True)


def test_stage152_exact_recombination_closes():
    m = _metrics()
    assert m["maximum_identity_or_provenance_error"] == 0.0
    assert np.isclose(m["combined_left"], m["dominant_left"] + m["parent_left"])


def test_stage152_left_channels_are_opposed():
    m = _metrics()
    assert m["left_channel_sign_product"] == -1
    assert m["right_channel_sign_product"] == -1


def test_stage152_parent_channel_is_stronger_but_below_single_channel_guard():
    m = _metrics()
    assert m["left_parent_absolute_share"] > m["left_dominant_absolute_share"]
    assert m["left_parent_absolute_share"] < PARENT_CHANNEL_DOMINANCE_MIN
    assert np.isclose(m["left_parent_absolute_share"], 0.7442707124011598)
    assert np.isclose(m["left_parent_to_dominant_magnitude_ratio"], 2.9103851161885275)


def test_stage152_left_balance_is_materially_cancellation_dominated():
    m = _metrics()
    assert m["left_cancellation_fraction"] >= MATERIAL_CANCELLATION_MIN
    assert np.isclose(m["left_cancellation_fraction"], 0.5114585751976803)
    assert np.isclose(m["left_net_l1_share"], 0.4885414248023197)


def test_stage152_observed_route_is_mixed_two_channel_balance():
    assert classify_left_outer_channel_balance(metrics=_metrics()) == MIXED_CANCELLATION


def test_stage152_parent_dominant_alternate_route_is_preregistered():
    m = _metrics()
    m["left_parent_absolute_share"] = 0.80
    assert classify_left_outer_channel_balance(metrics=m) == PARENT_DOMINANT


def test_stage152_record_failure_blocks_interpretation():
    assert classify_left_outer_channel_balance(metrics=_metrics(), stage151_record_ok=False) == RECORD_BLOCKER
