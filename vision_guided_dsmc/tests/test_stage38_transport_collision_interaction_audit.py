import numpy as np
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig, sidewall_temperature_profile
from vgdsmc.reduced_spherical_solver import discrete_maxwellian
from vgdsmc.stage38_transport_collision_interaction_audit import (
    STAGE37_COMPLETED_ENDPOINT,
    STAGE38_CFL,
    STAGE38_GRID,
    STAGE38_LIMITER_THETA,
    explicit_collision_substep,
    first_order_transport_reduced,
    stage38_decision,
    validate_stage38_design,
)
from vgdsmc.velocity_quadrature_audit import spherical_product


def row(q_error, velocity_error, sign=0.8, converged=True):
    return {
        "converged": converged,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": velocity_error,
            "sign_agreement": sign,
        },
    }


def small_case():
    cfg = LinearSidewallConfig(
        nx=3,
        ny=3,
        nv=5,
        velocity_extent=4.0,
        kn0=0.1,
        cold_hot_ratio=0.1,
        max_steps=10,
        cfl=STAGE38_CFL,
        tolerance=1e-4,
        check_interval=1,
        minimum_steps=1,
    )
    quadrature = spherical_product(4, 4, 8, 4.0, "stage38_test_spherical")
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.repeat(sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1)
    distribution = discrete_maxwellian(rho, zero, zero, zero, temperature, quadrature)
    return cfg, quadrature, distribution


def test_stage38_design_accepts_only_frozen_stage37_controls():
    validate_stage38_design(
        STAGE38_GRID, STAGE38_CFL, STAGE38_LIMITER_THETA, 16000, 2e-5
    )


def test_stage38_design_rejects_grid_cfl_or_limiter_retuning():
    with pytest.raises(ValueError):
        validate_stage38_design((36, 36), STAGE38_CFL, 1.5, 16000, 2e-5)
    with pytest.raises(ValueError):
        validate_stage38_design(STAGE38_GRID, 0.25, 1.5, 16000, 2e-5)
    with pytest.raises(ValueError):
        validate_stage38_design(STAGE38_GRID, STAGE38_CFL, 1.7, 16000, 2e-5)


def test_stage38_design_rejects_nonpositive_stopping_controls():
    with pytest.raises(ValueError):
        validate_stage38_design(STAGE38_GRID, STAGE38_CFL, 1.5, 0, 2e-5)
    with pytest.raises(ValueError):
        validate_stage38_design(STAGE38_GRID, STAGE38_CFL, 1.5, 16000, 0.0)


def test_stage37_completed_endpoint_is_retained_exactly():
    endpoint = STAGE37_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 30708041761
    assert endpoint["tests_passed"] == 44
    assert endpoint["artifact_id"] == 8822338075
    assert endpoint["artifact_sha256"] == (
        "2981365dd782c617eba73deb9e54d3ec31909acdc862d6d786934879f6b87152"
    )
    assert endpoint["rows"]["muscl_lie_explicit"]["predicted_qav"] == pytest.approx(
        0.04324014823991044
    )
    assert endpoint["comparison"]["qav_error_ratio_muscl_to_first_order"] == pytest.approx(
        10.533009250390585
    )


def test_explicit_collision_substep_preserves_shape_mass_and_positivity():
    cfg, quadrature, distribution = small_case()
    before_mass = np.sum(distribution * quadrature.weight[None, None, :], axis=-1)
    updated = explicit_collision_substep(distribution, cfg, quadrature, 1e-4)
    after_mass = np.sum(updated * quadrature.weight[None, None, :], axis=-1)
    assert updated.shape == distribution.shape
    assert np.isfinite(updated).all()
    assert updated.min() >= cfg.positivity_floor
    np.testing.assert_allclose(after_mass, before_mass, rtol=1e-12, atol=1e-12)


def test_explicit_collision_substep_rejects_nonpositive_time():
    cfg, quadrature, distribution = small_case()
    with pytest.raises(ValueError):
        explicit_collision_substep(distribution, cfg, quadrature, 0.0)


def test_first_order_transport_preserves_shape_and_positivity():
    cfg, quadrature, distribution = small_case()
    maximum_speed = max(np.max(np.abs(quadrature.vx)), np.max(np.abs(quadrature.vy)))
    dx = 1.0 / cfg.nx
    dy = 1.0 / cfg.ny
    dt = cfg.cfl * min(dx / maximum_speed, dy / maximum_speed)
    transported = first_order_transport_reduced(
        distribution, cfg, quadrature, dt, dx, dy
    )
    assert transported.shape == distribution.shape
    assert np.isfinite(transported).all()
    assert transported.min() >= cfg.positivity_floor


def test_stage38_decision_advances_only_for_full_muscl_rescue():
    first_lie = row(0.04, 0.90, 0.8)
    muscl_lie = row(0.40, 2.80, 1.0)
    first_strang = row(0.04, 0.90, 0.8)
    muscl_strang = row(0.03, 0.70, 0.9)
    assert stage38_decision(first_lie, muscl_lie, first_strang, muscl_strang) == (
        "coupling_rescues_muscl_stage39_high_resolution_confirmation"
    )


def test_stage38_decision_separates_splitting_and_partial_rescue_paths():
    first_lie = row(0.04, 0.90, 0.8)
    muscl_lie = row(0.40, 2.80, 1.0)
    first_strang = row(0.03, 0.70, 0.8)
    muscl_strang = row(0.08, 1.10, 1.0)
    assert stage38_decision(first_lie, muscl_lie, first_strang, muscl_strang) == (
        "splitting_dominates_stage39_collision_time_integration_audit"
    )
    first_strang = row(0.04, 0.90, 0.8)
    muscl_strang = row(0.20, 1.20, 1.0)
    assert stage38_decision(first_lie, muscl_lie, first_strang, muscl_strang) == (
        "partial_muscl_rescue_stage39_limiter_flux_consistency_audit"
    )


def test_stage38_decision_retains_negative_and_nonconvergent_endpoints():
    first_lie = row(0.04, 0.90, 0.8)
    muscl_lie = row(0.40, 2.80, 1.0)
    first_strang = row(0.05, 1.00, 0.8)
    muscl_strang = row(0.45, 3.00, 1.0)
    assert stage38_decision(first_lie, muscl_lie, first_strang, muscl_strang) == (
        "no_coupling_rescue_stage39_collision_model_or_benchmark_audit"
    )
    muscl_strang = row(0.20, 1.20, 1.0, converged=False)
    assert stage38_decision(first_lie, muscl_lie, first_strang, muscl_strang) == (
        "stage38_nonconvergence_stage39_numerical_stability_audit"
    )
