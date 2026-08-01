import numpy as np
import pytest

from vgdsmc.stage37_low_kn_transport_audit import (
    STAGE36_LOW_KN_ENDPOINTS,
    STAGE37_CFL,
    STAGE37_GRID,
    STAGE37_LIMITER_THETA,
    muscl_flux_divergence_reduced,
    positivity_blend_reduced,
    stage37_decision,
    validate_stage37_design,
)
from vgdsmc.velocity_quadrature_audit import VelocityQuadrature


def small_quadrature():
    return VelocityQuadrature(
        name="test_four_velocity",
        vx=np.array([-1.0, 1.0, -1.0, 1.0]),
        vy=np.array([-1.0, -1.0, 1.0, 1.0]),
        vz=np.zeros(4),
        weight=np.ones(4),
        family="test",
    )


def row(q_error, velocity_error, sign=0.8, converged=True):
    return {
        "converged": converged,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": velocity_error,
            "sign_agreement": sign,
        },
    }


def test_stage37_design_accepts_only_preregistered_screen():
    validate_stage37_design(
        STAGE37_GRID,
        STAGE37_CFL,
        STAGE37_LIMITER_THETA,
        16000,
        2e-5,
    )


def test_stage37_design_rejects_grid_or_numerical_retuning():
    with pytest.raises(ValueError):
        validate_stage37_design((36, 36), STAGE37_CFL, 1.5, 16000, 2e-5)
    with pytest.raises(ValueError):
        validate_stage37_design(STAGE37_GRID, 0.25, 1.5, 16000, 2e-5)
    with pytest.raises(ValueError):
        validate_stage37_design(STAGE37_GRID, STAGE37_CFL, 1.7, 16000, 2e-5)


def test_stage37_design_rejects_nonpositive_stopping_controls():
    with pytest.raises(ValueError):
        validate_stage37_design(STAGE37_GRID, STAGE37_CFL, 1.5, 0, 2e-5)
    with pytest.raises(ValueError):
        validate_stage37_design(STAGE37_GRID, STAGE37_CFL, 1.5, 16000, 0.0)


def test_positivity_blend_preserves_positive_candidate():
    old = np.full((2, 3, 4), 2.0)
    candidate = np.full((2, 3, 4), 1.0)
    result = positivity_blend_reduced(old, candidate, 1e-6)
    np.testing.assert_allclose(result, candidate)


def test_positivity_blend_limits_entire_cell_convexly():
    old = np.ones((1, 1, 3))
    candidate = np.array([[[-1.0, 0.5, 2.0]]])
    result = positivity_blend_reduced(old, candidate, 0.1)
    assert result.min() >= 0.1
    theta = (1.0 - 0.1) / (1.0 - (-1.0))
    np.testing.assert_allclose(result, old + theta * (candidate - old))


def test_positivity_blend_rejects_invalid_shapes_and_floor():
    with pytest.raises(ValueError):
        positivity_blend_reduced(np.ones((2, 2)), np.ones((2, 2)), 1e-6)
    with pytest.raises(ValueError):
        positivity_blend_reduced(np.ones((1, 1, 2)), np.ones((1, 1, 3)), 1e-6)
    with pytest.raises(ValueError):
        positivity_blend_reduced(np.ones((1, 1, 2)), np.ones((1, 1, 2)), 0.0)


def test_muscl_divergence_is_zero_for_constant_state_and_matching_boundaries():
    quadrature = small_quadrature()
    distribution = np.full((3, 4, quadrature.point_count), 0.75)
    left = np.full((3, quadrature.point_count), 0.75)
    right = np.full((3, quadrature.point_count), 0.75)
    bottom = np.full((4, quadrature.point_count), 0.75)
    top = np.full((4, quadrature.point_count), 0.75)
    divergence = muscl_flux_divergence_reduced(
        distribution, left, right, bottom, top, quadrature, 0.25, 1.0 / 3.0
    )
    np.testing.assert_allclose(divergence, 0.0, atol=1e-14)


def test_muscl_divergence_rejects_mismatched_quadrature():
    quadrature = small_quadrature()
    with pytest.raises(ValueError):
        muscl_flux_divergence_reduced(
            np.ones((2, 2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            quadrature,
            0.5,
            0.5,
        )


def test_stage36_low_kn_endpoints_are_retained_exactly():
    assert STAGE36_LOW_KN_ENDPOINTS["24x24"]["predicted_qav"] == pytest.approx(
        0.07460201631883724
    )
    assert STAGE36_LOW_KN_ENDPOINTS["36x36"]["qav_relative_error"] == pytest.approx(
        0.013542796517709883
    )
    assert STAGE36_LOW_KN_ENDPOINTS["profile_change_24x24_to_36x36"] == pytest.approx(
        0.24659632808808882
    )


def test_stage37_decision_advances_high_resolution_confirmation_only_if_both_improve():
    first = row(0.04, 0.90, 0.8)
    muscl = row(0.03, 0.70, 0.8)
    assert stage37_decision(first, muscl) == (
        "muscl_screen_positive_stage38_high_resolution_confirmation"
    )


def test_stage37_decision_retains_mixed_and_negative_outcomes():
    first = row(0.04, 0.90, 0.8)
    mixed = row(0.03, 1.10, 0.8)
    assert stage37_decision(first, mixed) == (
        "muscl_screen_mixed_stage38_transport_collision_interaction_audit"
    )
    negative = row(0.05, 1.00, 0.7)
    assert stage37_decision(first, negative) == (
        "muscl_screen_negative_stage38_collision_model_audit"
    )


def test_stage37_decision_preserves_nonconvergence_as_blocker():
    first = row(0.04, 0.90, 0.8, converged=False)
    muscl = row(0.03, 0.70, 0.8)
    assert stage37_decision(first, muscl) == (
        "transport_audit_nonconvergence_stage38_numerical_stability"
    )
