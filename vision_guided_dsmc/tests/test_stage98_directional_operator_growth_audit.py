import numpy as np
import pytest

from vgdsmc import stage98_directional_operator_growth_audit as stage98
from vgdsmc.stage90_single_condition_reconstruction_solver_ab_audit import muscl_correction_divergence


def test_stage98_frozen_design_accepts_defaults():
    stage98.validate_stage98_design()


@pytest.mark.parametrize(
    "override",
    [
        {"grid": (32, 32)},
        {"kn0": 5.0},
        {"rule": (32, 64)},
        {"radial_scale": 1.5},
        {"limiter": "none"},
        {"boundary_slope": "one_sided"},
        {"diagnostic_steps": 20},
        {"directional_dominance_share": 0.5},
        {"material_directional_growth_ratio": 1.5},
        {"material_cancellation_ratio": 0.75},
        {"stage97_run_id": 1},
    ],
)
def test_stage98_frozen_design_rejects_retuning(override):
    with pytest.raises(ValueError):
        stage98.validate_stage98_design(**override)


def test_directional_components_close_retained_zero_boundary_operator():
    rng = np.random.default_rng(20260810)
    field = rng.normal(size=(5, 6, 7))
    vx = np.array([-3.0, -1.5, -0.2, 0.0, 0.4, 1.7, 2.8])
    vy = np.array([2.2, -1.1, 0.0, 0.7, -2.6, 1.4, -0.3])
    dx = 0.125
    dy = 0.2
    corr_x, corr_y = stage98.muscl_correction_components(field, vx, vy, dx, dy)
    retained = muscl_correction_divergence(field, vx, vy, dx, dy, False)
    assert np.allclose(corr_x + corr_y, retained, rtol=0.0, atol=1.0e-14)


def test_directional_components_are_conservative_by_axis():
    rng = np.random.default_rng(98)
    field = rng.normal(size=(5, 6, 4))
    vx = np.array([-2.0, -0.5, 0.5, 2.0])
    vy = np.array([1.0, -1.0, 2.0, -2.0])
    corr_x, corr_y = stage98.muscl_correction_components(field, vx, vy, 0.1, 0.15)
    assert np.max(np.abs(np.sum(corr_x, axis=(0, 1)))) < 1.0e-12
    assert np.max(np.abs(np.sum(corr_y, axis=(0, 1)))) < 1.0e-12


def _trace(final_x, final_y, x_growth, y_growth, cancellation=0.9):
    return {
        "x_directional_abs_share": {"final": final_x},
        "y_directional_abs_share": {"final": final_y},
        "x_weighted_abs": {"final_to_first_ratio": x_growth},
        "y_weighted_abs": {"final_to_first_ratio": y_growth},
        "weighted_abs_cancellation_ratio": {"minimum": cancellation},
    }


def test_decision_blocks_decomposition_mismatch_without_retuning():
    payload = {"phi": _trace(0.8, 0.2, 3.0, 1.2), "psi": _trace(0.8, 0.2, 2.5, 1.1)}
    assert stage98.stage98_decision(payload, 1.0e-6, 0.0) == (
        "stage98_decomposition_or_replay_mismatch_blocker_without_retuning"
    )


def test_decision_blocks_parent_replay_mismatch_without_retuning():
    payload = {"phi": _trace(0.8, 0.2, 3.0, 1.2), "psi": _trace(0.8, 0.2, 2.5, 1.1)}
    assert stage98.stage98_decision(payload, 0.0, 1.0e-6) == (
        "stage98_decomposition_or_replay_mismatch_blocker_without_retuning"
    )


def test_decision_routes_common_x_dominance_and_growth():
    payload = {"phi": _trace(0.75, 0.25, 3.0, 1.2), "psi": _trace(0.70, 0.30, 2.2, 1.1)}
    assert stage98.stage98_decision(payload, 0.0, 0.0) == (
        "stage98_x_dominant_growing_operator_stage99_x_signed_lobe_localization_audit"
    )


def test_decision_routes_common_y_dominance_and_growth():
    payload = {"phi": _trace(0.25, 0.75, 1.1, 2.4), "psi": _trace(0.30, 0.70, 1.2, 2.1)}
    assert stage98.stage98_decision(payload, 0.0, 0.0) == (
        "stage98_y_dominant_growing_operator_stage99_y_signed_lobe_localization_audit"
    )


def test_decision_routes_material_cross_axis_cancellation():
    payload = {"phi": _trace(0.55, 0.45, 2.5, 2.0, 0.45), "psi": _trace(0.52, 0.48, 2.2, 2.3, 0.60)}
    assert stage98.stage98_decision(payload, 0.0, 0.0) == (
        "stage98_material_cross_axis_cancellation_stage99_signed_cancellation_localization_audit"
    )


def test_decision_routes_mixed_directional_growth_without_causal_claim():
    payload = {"phi": _trace(0.58, 0.42, 2.6, 2.0, 0.85), "psi": _trace(0.48, 0.52, 2.4, 2.2, 0.82)}
    assert stage98.stage98_decision(payload, 0.0, 0.0) == (
        "stage98_mixed_directional_growth_stage99_interior_velocity_sector_audit"
    )
