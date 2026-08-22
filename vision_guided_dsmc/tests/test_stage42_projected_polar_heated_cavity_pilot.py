from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig
from vgdsmc.stage41_projected_polar_operator_audit import (
    mapped_polar_quadrature,
    projected_macroscopic,
    projected_maxwellian,
    wall_mass_balance_error,
)
from vgdsmc.stage42_projected_polar_heated_cavity_pilot import (
    STAGE41_COMPLETED_ENDPOINT,
    STAGE42_GRID,
    STAGE42_KNUDSEN,
    STAGE42_MAX_ITERATIONS,
    STAGE42_RATIO,
    STAGE42_RULE,
    STAGE42_SOURCE_RELAXATION,
    STAGE42_TOLERANCE,
    bottom_wall_heat_flux,
    projected_wall_incoming,
    stage42_decision,
    steady_source_iteration_step,
    validate_stage42_design,
)


def _small_cfg() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=4,
        ny=4,
        kn0=0.1,
        cold_hot_ratio=0.1,
        viscosity_exponent=0.5,
        prandtl=2.0 / 3.0,
        max_steps=2,
        cfl=0.2,
        tolerance=2.0e-5,
        check_interval=1,
        minimum_steps=1,
        positivity_floor=1.0e-30,
    )


def test_stage41_endpoint_is_exactly_retained() -> None:
    assert STAGE41_COMPLETED_ENDPOINT["workflow_run_id"] == 30727704751
    assert STAGE41_COMPLETED_ENDPOINT["workflow_job_id"] == 91442360250
    assert STAGE41_COMPLETED_ENDPOINT["artifact_id"] == 8827830725
    assert STAGE41_COMPLETED_ENDPOINT["decision"] == (
        "projected_polar_operators_pass_stage42_heated_cavity_pilot"
    )


def test_stage42_design_is_frozen() -> None:
    validate_stage42_design(
        STAGE42_GRID,
        STAGE42_RULE,
        STAGE42_KNUDSEN,
        STAGE42_RATIO,
        STAGE42_MAX_ITERATIONS,
        STAGE42_TOLERANCE,
        STAGE42_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage42_design(
            (10, 10),
            STAGE42_RULE,
            STAGE42_KNUDSEN,
            STAGE42_RATIO,
            STAGE42_MAX_ITERATIONS,
            STAGE42_TOLERANCE,
            STAGE42_SOURCE_RELAXATION,
        )


def test_projected_wall_profiles_enforce_mass_balance() -> None:
    cfg = _small_cfg()
    quadrature = mapped_polar_quadrature(8, 24)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.full_like(rho, 0.5)
    phi, psi = projected_maxwellian(rho, zero, zero, temperature, quadrature)
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    left_phi, _, right_phi, _, bottom_phi, _, top_phi, _ = incoming
    errors = [
        wall_mass_balance_error(
            phi[:, 0], left_phi, quadrature.vx, quadrature.vx > 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[:, -1], right_phi, -quadrature.vx, quadrature.vx < 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[0], bottom_phi, quadrature.vy, quadrature.vy > 0.0, quadrature
        ),
        wall_mass_balance_error(
            phi[-1], top_phi, -quadrature.vy, quadrature.vy < 0.0, quadrature
        ),
    ]
    assert max(errors) < 1.0e-12


def test_one_source_iteration_is_positive_and_finite() -> None:
    cfg = _small_cfg()
    quadrature = mapped_polar_quadrature(8, 24)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.repeat(
        np.linspace(0.9, 0.2, cfg.ny)[:, None], cfg.nx, axis=1
    )
    phi, psi = projected_maxwellian(rho, zero, zero, temperature, quadrature)
    next_phi, next_psi, clipping = steady_source_iteration_step(
        phi, psi, cfg, quadrature
    )
    assert np.isfinite(next_phi).all()
    assert np.isfinite(next_psi).all()
    assert np.min(next_phi) > 0.0
    assert np.min(next_psi) > 0.0
    fields = projected_macroscopic(next_phi, next_psi, quadrature)
    assert np.isfinite(fields["T"]).all()
    assert np.isfinite(clipping["phi_clipped_weight_fraction"]).all()


def test_matching_bottom_wall_has_nearly_zero_heat_flux() -> None:
    cfg = _small_cfg()
    quadrature = mapped_polar_quadrature(16, 48)
    rho = np.ones((cfg.ny, cfg.nx))
    zero = np.zeros_like(rho)
    temperature = np.ones_like(rho)
    phi, psi = projected_maxwellian(rho, zero, zero, temperature, quadrature)
    incoming = projected_wall_incoming(phi, psi, cfg, quadrature)
    bottom_phi, bottom_psi = incoming[4], incoming[5]
    flux = bottom_wall_heat_flux(
        phi, psi, bottom_phi, bottom_psi, quadrature
    )
    assert np.max(np.abs(flux)) < 1.0e-10


def test_stage42_decision_preserves_negative_endpoint() -> None:
    base = {
        "finite": True,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "wall_mass_balance_relative_error": 1.0e-16,
        "converged": False,
    }
    assert stage42_decision(base) == (
        "projected_polar_pilot_stable_nonconverged_stage43_iteration_acceleration"
    )
    converged = dict(base, converged=True)
    assert stage42_decision(converged) == (
        "projected_polar_pilot_converged_stage43_resolution_sequence"
    )
    blocked = dict(base, finite=False)
    assert stage42_decision(blocked) == "projected_polar_heated_cavity_blocker"
