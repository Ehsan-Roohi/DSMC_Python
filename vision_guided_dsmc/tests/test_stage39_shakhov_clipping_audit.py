import numpy as np
import pytest

from vgdsmc.stage39_shakhov_clipping_audit import (
    STAGE39_CFL,
    STAGE39_CLIP_FLOOR,
    STAGE39_GRID,
    positivity_blend_with_theta,
    shakhov_equilibrium_variant,
    stage39_decision,
    validate_stage39_design,
)
from vgdsmc.reduced_spherical_solver import discrete_maxwellian, macroscopic
from vgdsmc.velocity_quadrature_audit import spherical_product


def test_validate_stage39_design_is_fixed():
    validate_stage39_design(STAGE39_GRID, STAGE39_CFL, STAGE39_CLIP_FLOOR, 10, 1e-4)
    with pytest.raises(ValueError):
        validate_stage39_design((12, 12), STAGE39_CFL, STAGE39_CLIP_FLOOR, 10, 1e-4)
    with pytest.raises(ValueError):
        validate_stage39_design(STAGE39_GRID, 0.1, STAGE39_CLIP_FLOOR, 10, 1e-4)
    with pytest.raises(ValueError):
        validate_stage39_design(STAGE39_GRID, STAGE39_CFL, 0.01, 10, 1e-4)


def test_prandtl_one_raw_and_clipped_equilibria_are_identical():
    q = spherical_product(4, 4, 8, 4.0, "stage39_test")
    shape = (2, 3)
    rho = np.ones(shape)
    zero = np.zeros(shape)
    temperature = np.full(shape, 0.7)
    distribution = discrete_maxwellian(rho, zero, zero, zero, temperature, q)
    fields = macroscopic(distribution, q)
    clipped, clipped_diag = shakhov_equilibrium_variant(
        fields, q, 1.0, pointwise_clip=True
    )
    raw, raw_diag = shakhov_equilibrium_variant(
        fields, q, 1.0, pointwise_clip=False
    )
    np.testing.assert_allclose(clipped, raw, rtol=1e-13, atol=1e-14)
    assert clipped_diag["raw_multiplier_minimum"] == pytest.approx(1.0)
    assert raw_diag["raw_multiplier_below_clip_floor_fraction"] == 0.0


def test_positivity_blend_reports_limited_theta():
    old = np.ones((2, 2, 3))
    candidate = old.copy()
    candidate[0, 0, 0] = -2.0
    limited, theta = positivity_blend_with_theta(old, candidate, 1e-6)
    assert np.min(limited) >= 1e-6
    assert theta[0, 0] < 1.0
    assert theta[1, 1] == pytest.approx(1.0)


def _row(q_error, v_error, sign, converged=True, activation=1e-3):
    return {
        "converged": converged,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": v_error,
            "sign_agreement": sign,
        },
        "collision_diagnostics": {
            "mean_raw_multiplier_below_clip_floor_fraction": activation,
        },
    }


def test_stage39_decision_paths():
    clipped = _row(0.10, 1.0, 0.8)
    raw_rescue = _row(0.08, 0.8, 0.8)
    assert stage39_decision(clipped, raw_rescue) == (
        "pointwise_clipping_identified_stage40_high_resolution_confirmation"
    )
    raw_partial = _row(0.08, 1.1, 0.9)
    assert stage39_decision(clipped, raw_partial) == (
        "clipping_material_but_incomplete_stage40_independent_benchmark"
    )
    raw_worse = _row(0.12, 1.2, 0.8)
    assert stage39_decision(clipped, raw_worse) == (
        "clipping_not_primary_stage40_independent_benchmark_or_full_boltzmann"
    )
    inactive = _row(0.10, 1.0, 0.8, activation=0.0)
    assert stage39_decision(inactive, raw_worse) == (
        "clipping_inactive_stage40_independent_benchmark"
    )
    not_converged = _row(0.10, 1.0, 0.8, converged=False)
    assert stage39_decision(not_converged, raw_worse) == (
        "stage39_nonconvergence_stage40_numerical_stability_audit"
    )
