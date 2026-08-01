import numpy as np
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig, sidewall_temperature_profile
from vgdsmc.reduced_spherical_solver import (
    bottom_heat_flux,
    discrete_maxwellian,
    macroscopic,
    run_stage30,
    shakhov_equilibrium,
    solve_reduced_case,
    wall_incoming,
    wall_mass_balance_error,
)
from vgdsmc.velocity_quadrature_audit import cartesian_midpoint, spherical_product


def _matched_spherical():
    return spherical_product(16, 12, 24, 5.0, "spherical_matched_r16_mu12_phi24")


def test_discrete_maxwellian_conserves_density_on_both_rules():
    rho = np.array([[0.8, 1.2]])
    zero = np.zeros_like(rho)
    temperature = np.array([[0.1, 1.0]])
    for quadrature in (cartesian_midpoint(19, 5.0), _matched_spherical()):
        f = discrete_maxwellian(rho, zero, zero, zero, temperature, quadrature)
        fields = macroscopic(f, quadrature)
        np.testing.assert_allclose(fields["rho"], rho, rtol=2e-13, atol=2e-13)


def test_spherical_cold_temperature_is_accurate():
    quadrature = _matched_spherical()
    rho = np.ones((1, 1))
    zero = np.zeros_like(rho)
    f = discrete_maxwellian(rho, zero, zero, zero, 0.1 * rho, quadrature)
    measured = float(macroscopic(f, quadrature)["T"][0, 0])
    assert abs(measured - 0.1) / 0.1 < 3.0e-4


def test_spherical_shifted_maxwellian_recovers_velocity_sign():
    quadrature = _matched_spherical()
    rho = np.ones((1, 1))
    zero = np.zeros_like(rho)
    target_v = np.full_like(rho, 0.005)
    f = discrete_maxwellian(rho, zero, target_v, zero, 0.1 * rho, quadrature)
    measured = float(macroscopic(f, quadrature)["v"][0, 0])
    assert measured > 0.0
    assert abs(measured - 0.005) < 2.0e-4


def test_shakhov_equilibrium_is_finite_positive_and_density_conservative():
    quadrature = spherical_product(8, 6, 8, 5.0, "test_spherical")
    rho = np.ones((2, 2))
    zero = np.zeros_like(rho)
    temperature = np.array([[0.1, 0.3], [0.6, 1.0]])
    f = discrete_maxwellian(rho, zero, zero, zero, temperature, quadrature)
    fields = macroscopic(f, quadrature)
    equilibrium = shakhov_equilibrium(fields, quadrature, 2.0 / 3.0)
    assert np.isfinite(equilibrium).all()
    assert np.min(equilibrium) > 0.0
    np.testing.assert_allclose(
        macroscopic(equilibrium, quadrature)["rho"], fields["rho"], rtol=2e-12, atol=2e-12
    )


def test_diffuse_wall_states_have_zero_net_mass_flux():
    quadrature = spherical_product(8, 6, 8, 5.0, "test_spherical")
    cfg = LinearSidewallConfig(nx=4, ny=4, kn0=1.0, cold_hot_ratio=0.1)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.repeat(sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1)
    f = discrete_maxwellian(rho, zero, zero, zero, temperature, quadrature)
    left, right, bottom, top = wall_incoming(f, cfg, quadrature)
    errors = [
        wall_mass_balance_error(f[:, 0], left, quadrature.vx, quadrature.vx > 0.0, quadrature),
        wall_mass_balance_error(f[:, -1], right, -quadrature.vx, quadrature.vx < 0.0, quadrature),
        wall_mass_balance_error(f[0], bottom, quadrature.vy, quadrature.vy > 0.0, quadrature),
        wall_mass_balance_error(f[-1], top, -quadrature.vy, quadrature.vy < 0.0, quadrature),
    ]
    assert max(errors) < 2.0e-14


def test_bottom_heat_flux_is_finite():
    quadrature = spherical_product(8, 6, 8, 5.0, "test_spherical")
    cfg = LinearSidewallConfig(nx=4, ny=4, kn0=1.0, cold_hot_ratio=0.1)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.repeat(sidewall_temperature_profile(cfg)[:, None], cfg.nx, axis=1)
    f = discrete_maxwellian(rho, zero, zero, zero, temperature, quadrature)
    _, _, bottom, _ = wall_incoming(f, cfg, quadrature)
    flux = bottom_heat_flux(f, bottom, quadrature)
    assert flux.shape == (cfg.nx,)
    assert np.isfinite(flux).all()


def test_small_reduced_solver_run_is_finite():
    quadrature = spherical_product(6, 6, 8, 4.0, "small_spherical")
    cfg = LinearSidewallConfig(
        nx=4,
        ny=4,
        kn0=1.0,
        cold_hot_ratio=0.1,
        max_steps=4,
        check_interval=1,
        minimum_steps=4,
        tolerance=1.0e-30,
    )
    result = solve_reduced_case(cfg, quadrature)
    assert result["iterations"] == 4
    assert np.isfinite(result["T"]).all()
    assert np.isfinite(result["bottom_heat_flux"]).all()
    assert result["wall_mass_balance_relative_error"] < 2.0e-13
    assert result["minimum_distribution"] > 0.0


def test_stage30_rejects_retuned_physical_case(tmp_path):
    cfg = LinearSidewallConfig(nx=4, ny=4, kn0=0.5, cold_hot_ratio=0.1, max_steps=2)
    with pytest.raises(ValueError, match="fixed to Kn0=1"):
        run_stage30(tmp_path, cfg)
