import math
import numpy as np
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig
from vgdsmc.stage35_coordinate_covariance import (
    CoordinateSystem,
    SQRT2,
    build_stage35_coordinate_systems,
    coordinate_tau_prefactor,
    covariance_metrics,
    scaled_discrete_maxwellian,
    scaled_macroscopic,
    scaled_wall_incoming,
    solve_scaled_coordinate_case,
    stage35_decision,
    transform_quadrature_from_c0,
    validate_stage35_design,
)
from vgdsmc.velocity_quadrature_audit import spherical_product


def small_systems():
    base = spherical_product(4, 4, 4, 4.0, "small_c0")
    c0 = CoordinateSystem("c0", base, 1.0)
    zeta = CoordinateSystem(
        "zeta",
        transform_quadrature_from_c0(base, SQRT2, "small_zeta"),
        SQRT2,
    )
    return c0, zeta


def test_coordinate_transform_preserves_physical_nodes_and_volume():
    c0, zeta = build_stage35_coordinate_systems()
    assert zeta.quadrature.point_count == c0.quadrature.point_count
    assert np.allclose(SQRT2 * zeta.quadrature.vx, c0.quadrature.vx)
    assert np.allclose(SQRT2 * zeta.quadrature.vy, c0.quadrature.vy)
    assert np.allclose(SQRT2 * zeta.quadrature.vz, c0.quadrature.vz)
    assert np.allclose(
        SQRT2**3 * zeta.quadrature.weight,
        c0.quadrature.weight,
    )


def test_scaled_maxwellians_obey_distribution_jacobian():
    c0, zeta = small_systems()
    shape = (2, 3)
    rho = np.full(shape, 1.2)
    u = np.full(shape, 0.03)
    v = np.full(shape, -0.02)
    w = np.zeros(shape)
    temperature = np.full(shape, 0.7)
    f_c0 = scaled_discrete_maxwellian(rho, u, v, w, temperature, c0)
    g_zeta = scaled_discrete_maxwellian(rho, u, v, w, temperature, zeta)
    assert np.allclose(g_zeta / SQRT2**3, f_c0, rtol=2e-14, atol=1e-14)


def test_scaled_macroscopic_fields_are_coordinate_invariant():
    c0, zeta = small_systems()
    rho = np.array([[0.9, 1.1]])
    u = np.array([[0.02, -0.01]])
    v = np.array([[-0.03, 0.04]])
    w = np.array([[0.0, 0.01]])
    temperature = np.array([[0.4, 0.8]])
    f_c0 = scaled_discrete_maxwellian(rho, u, v, w, temperature, c0)
    g_zeta = scaled_discrete_maxwellian(rho, u, v, w, temperature, zeta)
    fields_c0 = scaled_macroscopic(f_c0, c0)
    fields_zeta = scaled_macroscopic(g_zeta, zeta)
    for name in ("rho", "u", "v", "w", "T", "qx", "qy", "qz"):
        assert np.allclose(fields_c0[name], fields_zeta[name], rtol=2e-13, atol=2e-13)


def test_tau_and_time_coordinate_ratios_are_sqrt_two():
    c0, zeta = small_systems()
    for kn0 in (0.1, 1.0, 10.0):
        assert coordinate_tau_prefactor(kn0, zeta) / coordinate_tau_prefactor(kn0, c0) == pytest.approx(SQRT2)


def test_diffuse_wall_states_transform_covariantly():
    c0, zeta = small_systems()
    cfg = LinearSidewallConfig(nx=4, ny=5, kn0=1.0)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.linspace(1.0, 0.1, cfg.ny)[:, None] * np.ones((1, cfg.nx))
    f_c0 = scaled_discrete_maxwellian(rho, zero, zero, zero, temperature, c0)
    g_zeta = scaled_discrete_maxwellian(rho, zero, zero, zero, temperature, zeta)
    walls_c0 = scaled_wall_incoming(f_c0, cfg, c0)
    walls_zeta = scaled_wall_incoming(g_zeta, cfg, zeta)
    for state_c0, state_zeta in zip(walls_c0, walls_zeta):
        assert np.allclose(
            state_zeta / SQRT2**3,
            state_c0,
            rtol=5e-13,
            atol=5e-14,
        )


def test_short_solver_trajectories_are_covariant():
    c0, zeta = small_systems()
    cfg = LinearSidewallConfig(
        nx=4,
        ny=4,
        kn0=1.0,
        max_steps=20,
        check_interval=5,
        minimum_steps=100,
        tolerance=1e-12,
    )
    result_c0 = solve_scaled_coordinate_case(cfg, c0)
    result_zeta = solve_scaled_coordinate_case(cfg, zeta)
    metrics = covariance_metrics(result_c0, result_zeta, zeta)
    assert metrics["iteration_count_match"]
    assert metrics["convergence_flag_match"]
    assert metrics["dt_ratio_zeta_to_c0"] == pytest.approx(SQRT2)
    assert metrics["tau_prefactor_ratio_zeta_to_c0"] == pytest.approx(SQRT2)
    assert metrics["maximum_covariance_error"] < 2e-10


def test_stage35_design_is_fixed():
    validate_stage35_design((0.1, 1.0, 10.0), (12, 12), 9000, 3e-5)
    with pytest.raises(ValueError):
        validate_stage35_design((0.1, 1.0), (12, 12), 9000, 3e-5)
    with pytest.raises(ValueError):
        validate_stage35_design((0.1, 1.0, 10.0), (16, 16), 9000, 3e-5)


def test_stage35_decision_requires_exact_covariance():
    passing_covariance = {
        "iteration_count_match": True,
        "convergence_flag_match": True,
        "maximum_covariance_error": 1e-11,
        "dt_ratio_zeta_to_c0": SQRT2,
        "tau_prefactor_ratio_zeta_to_c0": SQRT2,
    }
    rows = [
        {
            "c0_converged": True,
            "zeta_converged": True,
            "covariance": dict(passing_covariance),
        }
        for _ in range(3)
    ]
    assert stage35_decision(rows).startswith("coordinate_covariance_passes")
    rows[1]["covariance"]["maximum_covariance_error"] = 1e-4
    assert stage35_decision(rows).startswith("coordinate_covariance_fails")


def test_invalid_coordinate_scale_is_rejected():
    base = spherical_product(4, 4, 4, 4.0, "small")
    with pytest.raises(ValueError):
        transform_quadrature_from_c0(base, 0.0, "invalid")
