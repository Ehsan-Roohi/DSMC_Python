import numpy as np
import pytest

from vgdsmc.stage151_outer_interval_asymmetry_audit import (
    COMBINED_SIDE_DOMINANCE_MIN,
    DISTRIBUTED,
    OPPOSED_RECOMBINED_DOMINANCE,
    classify_outer_interval_asymmetry,
    outer_interval_asymmetry_metrics,
    validate_stage151_design,
)


def _metrics():
    dominant = np.array([0.0035605261840368774, -0.005465040509444852])
    parent = np.array([-0.010362502411820462, 0.004792443873094243])
    fine = np.array([0.0062655553375603745, 0.0021739721596047423])
    coarse = np.array([0.0043610410121524, -0.003396086379121477])
    return outer_interval_asymmetry_metrics(dominant, parent, fine, coarse)


def test_stage151_frozen_design_accepts_only_registered_values():
    validate_stage151_design()
    with pytest.raises(ValueError):
        validate_stage151_design(kn0=9.0)
    with pytest.raises(ValueError):
        validate_stage151_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage151_design(physical_parameter_retuning=True)


def test_stage151_component_outer_vectors_are_opposed_on_both_sides():
    m = _metrics()
    assert m["both_sides_cross_channel_opposed"] is True
    assert m["sidewise_channel_sign_products"] == [-1, -1]
    assert m["channel_outer_vector_cosine"] < -0.8


def test_stage151_component_asymmetry_orientations_are_opposite():
    m = _metrics()
    assert m["opposite_asymmetry_orientation"] is True
    assert m["dominant"]["absolute_asymmetry_index"] < 0.0
    assert m["parent"]["absolute_asymmetry_index"] > 0.0


def test_stage151_recombined_outer_increment_is_left_dominant():
    m = _metrics()
    assert m["combined"]["stronger_side"] == "left"
    assert m["combined"]["stronger_absolute_share"] >= COMBINED_SIDE_DOMINANCE_MIN
    assert m["combined"]["side_sign_product"] > 0


def test_stage151_exact_recombination_closes():
    m = _metrics()
    assert m["maximum_identity_or_provenance_error"] < 1e-12
    assert np.isclose(m["combined_outer_increments"][0], -0.006801976227783585)
    assert np.isclose(m["combined_outer_increments"][1], -0.0006725966363506086)


def test_stage151_observed_route_is_left_outer_balance_audit():
    assert classify_outer_interval_asymmetry(metrics=_metrics()) == OPPOSED_RECOMBINED_DOMINANCE


def test_stage151_distributed_alternate_route_remains_preregistered():
    m = _metrics()
    m["combined"]["stronger_absolute_share"] = 0.70
    assert classify_outer_interval_asymmetry(metrics=m) == DISTRIBUTED
