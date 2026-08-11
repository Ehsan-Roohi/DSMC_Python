import numpy as np
import pytest

from vgdsmc.stage107_limiter_activation_colocation_audit import (
    ACTIVATION_DEFINITION,
    COLOCATION_ENRICHMENT_GUARD,
    COLOCATION_OVERLAP_GUARD,
    PAIR_WEIGHT_SHARE_GUARD,
    _limiter_intervention_maps,
    _support_colocation_metrics,
    stage107_decision,
    validate_stage107_design,
)


def test_stage107_design_is_frozen():
    validate_stage107_design()
    with pytest.raises(ValueError):
        validate_stage107_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage107_design(colocation_enrichment_guard=1.01)
    with pytest.raises(ValueError):
        validate_stage107_design(stage67_run_id=-1)


def test_linear_field_has_no_minmod_intervention():
    ny = nx = 12
    nq = 3
    y, x = np.mgrid[:ny, :nx]
    base = (2.0 * x + 3.0 * y).astype(float)
    field = np.stack([base + 0.1 * k for k in range(nq)], axis=-1)
    weight = np.array([0.2, 0.3, 0.5])
    maps = _limiter_intervention_maps(field, weight, wall_band_cells=2)
    assert np.allclose(maps["intervention_fraction"], 0.0)
    assert np.allclose(maps["zeroed_velocity_weight_fraction"], 0.0)


def test_local_extremum_activates_minmod_zeroing():
    ny = nx = 12
    nq = 2
    x = np.arange(nx, dtype=float)
    profile = -((x - 5.5) ** 2)
    field2 = np.repeat(profile[None, :], ny, axis=0)
    field = np.stack([field2, 2.0 * field2], axis=-1)
    weight = np.array([0.4, 0.6])
    maps = _limiter_intervention_maps(field, weight, wall_band_cells=2)
    assert float(np.max(maps["intervention_fraction"])) > 0.0
    assert float(np.max(maps["zeroed_velocity_weight_fraction"])) > 0.0
    assert np.isfinite(maps["intervention_fraction"]).all()


def test_support_colocation_exact_high_support_is_detected():
    a = np.ones((8, 8), dtype=float)
    common = np.zeros((8, 8), dtype=bool)
    common[:2, :] = True
    a[common] = 10.0
    pair_weight = np.ones_like(a)
    pair_weight[common] = 5.0
    metrics, high = _support_colocation_metrics(a, common, pair_weight)
    assert metrics["inside_to_outside_enrichment"] > 1.0
    assert metrics["upper_quartile_overlap_coefficient"] == 1.0
    assert np.count_nonzero(high & common) == np.count_nonzero(common)


def test_stage107_decision_strong_and_negative_routes():
    strong = {
        "joint_intervention_colocation": {
            "inside_to_outside_enrichment": COLOCATION_ENRICHMENT_GUARD + 0.1,
            "upper_quartile_overlap_coefficient": COLOCATION_OVERLAP_GUARD + 0.1,
            "stage106_pair_gradient_weight_share_in_high_intervention_support": PAIR_WEIGHT_SHARE_GUARD + 0.1,
        }
    }
    assert stage107_decision(strong, True) == (
        "stage107_limiter_intervention_colocated_stage108_limiter_severity_correction_amplitude_coupling_audit"
    )

    negative = {
        "joint_intervention_colocation": {
            "inside_to_outside_enrichment": 1.0,
            "upper_quartile_overlap_coefficient": 0.2,
            "stage106_pair_gradient_weight_share_in_high_intervention_support": 0.3,
        }
    }
    assert stage107_decision(negative, True) == (
        "stage107_no_material_limiter_colocation_stage108_unlimited_gradient_smoothness_audit"
    )


def test_stage107_decision_nonfinite_is_blocker():
    metrics = {
        "joint_intervention_colocation": {
            "inside_to_outside_enrichment": 10.0,
            "upper_quartile_overlap_coefficient": 1.0,
            "stage106_pair_gradient_weight_share_in_high_intervention_support": 1.0,
        }
    }
    assert stage107_decision(metrics, False) == (
        "stage107_nonfinite_limiter_colocation_blocker_without_retuning"
    )


def test_activation_definition_is_precursor_not_solver_retuning():
    assert "stage67_pre_replay_state" in ACTIVATION_DEFINITION
