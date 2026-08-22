from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage43_projected_polar_resolution_sequence import (
    STAGE42_COMPLETED_ENDPOINT,
    STAGE43_GRIDS,
    STAGE43_KNUDSEN,
    STAGE43_MAX_ITERATIONS,
    STAGE43_RATIO,
    STAGE43_RULE,
    STAGE43_SOURCE_RELAXATION,
    STAGE43_TOLERANCE,
    build_stage43_config,
    linear_h_extrapolation,
    reproduce_stage42_endpoint,
    stage43_decision,
    validate_stage43_design,
)


def _case(
    q_error: float,
    velocity_error: float,
    qav: float,
    profile: list[float],
    *,
    converged: bool = True,
    finite: bool = True,
) -> dict[str, object]:
    return {
        "finite": finite,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "wall_mass_balance_relative_error": 1.0e-16,
        "converged": converged,
        "qav_relative_error": q_error,
        "predicted_qav": qav,
        "velocity_metrics": {
            "relative_rms": velocity_error,
            "relative_l1": velocity_error,
            "sign_agreement": 1.0,
        },
        "table_velocity": np.asarray(profile, dtype=np.float64),
    }


def _stage42_case() -> dict[str, object]:
    return {
        "iterations": STAGE42_COMPLETED_ENDPOINT["iterations"],
        "converged": STAGE42_COMPLETED_ENDPOINT["converged"],
        "predicted_qav": STAGE42_COMPLETED_ENDPOINT["predicted_qav"],
        "qav_relative_error": STAGE42_COMPLETED_ENDPOINT["qav_relative_error"],
        "velocity_metrics": {
            "relative_rms": STAGE42_COMPLETED_ENDPOINT["velocity_relative_rms"],
            "relative_l1": STAGE42_COMPLETED_ENDPOINT["velocity_relative_l1"],
            "sign_agreement": STAGE42_COMPLETED_ENDPOINT["velocity_sign_agreement"],
        },
        "wall_mass_balance_relative_error": STAGE42_COMPLETED_ENDPOINT[
            "wall_mass_balance_relative_error"
        ],
    }


def test_stage42_completed_endpoint_is_exactly_retained() -> None:
    assert STAGE42_COMPLETED_ENDPOINT["workflow_run_id"] == 30731231176
    assert STAGE42_COMPLETED_ENDPOINT["workflow_job_id"] == 91451917655
    assert STAGE42_COMPLETED_ENDPOINT["artifact_id"] == 8828768222
    assert STAGE42_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "4d9181f5e01407ebb15da0b3a52539d3e536a62e6b38271b905cc5d1be646552"
    )
    assert STAGE42_COMPLETED_ENDPOINT["regression_tests_passed"] == 44
    assert STAGE42_COMPLETED_ENDPOINT["regression_tests_failed"] == 0


def test_stage43_design_is_frozen() -> None:
    validate_stage43_design(
        STAGE43_GRIDS,
        STAGE43_RULE,
        STAGE43_KNUDSEN,
        STAGE43_RATIO,
        STAGE43_MAX_ITERATIONS,
        STAGE43_TOLERANCE,
        STAGE43_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage43_design(
            ((8, 8), (12, 12), (20, 20)),
            STAGE43_RULE,
            STAGE43_KNUDSEN,
            STAGE43_RATIO,
            STAGE43_MAX_ITERATIONS,
            STAGE43_TOLERANCE,
            STAGE43_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage43_design(
            STAGE43_GRIDS,
            (24, 72),
            STAGE43_KNUDSEN,
            STAGE43_RATIO,
            STAGE43_MAX_ITERATIONS,
            STAGE43_TOLERANCE,
            STAGE43_SOURCE_RELAXATION,
        )


def test_stage43_config_freezes_physics_and_numerics() -> None:
    cfg = build_stage43_config((12, 12))
    assert (cfg.nx, cfg.ny) == (12, 12)
    assert cfg.kn0 == 0.1
    assert cfg.cold_hot_ratio == 0.1
    assert cfg.viscosity_exponent == 0.5
    assert cfg.prandtl == 2.0 / 3.0
    assert cfg.max_steps == 3000
    assert cfg.tolerance == 2.0e-5
    assert cfg.check_interval == 25
    assert cfg.minimum_steps == 500
    assert cfg.positivity_floor == 1.0e-30
    with pytest.raises(ValueError):
        build_stage43_config((20, 20))


def test_stage42_reproduction_detects_exact_match_and_drift() -> None:
    exact = reproduce_stage42_endpoint(_stage42_case())
    assert exact["passed"] is True
    drifted = _stage42_case()
    drifted["predicted_qav"] = float(drifted["predicted_qav"]) + 1.0e-6
    assert reproduce_stage42_endpoint(drifted)["passed"] is False


def test_linear_h_extrapolation_recovers_exact_linear_limit() -> None:
    sizes = np.asarray([8, 12, 16])
    h = 1.0 / sizes
    qav = 0.072 + 0.16 * h
    literature_velocity = np.asarray([0.01, -0.02, 0.03])
    profile = literature_velocity[None, :] + h[:, None] * np.asarray([0.4, -0.2, 0.1])
    result = linear_h_extrapolation(
        sizes, qav, profile, 0.072, literature_velocity
    )
    assert result["extrapolated_qav"] == pytest.approx(0.072, abs=1.0e-14)
    assert result["qav_fit_r2"] == pytest.approx(1.0, abs=1.0e-14)
    assert result["velocity_fit_r2"] == pytest.approx(1.0, abs=1.0e-14)
    assert result["extrapolated_velocity_relative_rms"] < 1.0e-12


def test_stage43_decision_allows_cross_kn_only_after_joint_convergence() -> None:
    cases = [
        _case(0.30, 3.0, 0.090, [1.00, -1.00]),
        _case(0.20, 2.0, 0.080, [1.05, -1.05]),
        _case(0.10, 1.0, 0.079, [1.10, -1.10]),
    ]
    assert stage43_decision(cases, True) == (
        "projected_polar_spatial_sequence_converging_stage44_cross_kn_extension"
    )


def test_stage43_decision_retains_improvement_without_false_convergence() -> None:
    cases = [
        _case(0.30, 3.0, 0.100, [0.2, -0.2]),
        _case(0.20, 2.0, 0.090, [0.5, -0.5]),
        _case(0.10, 1.0, 0.080, [1.0, -1.0]),
    ]
    assert stage43_decision(cases, True) == (
        "projected_polar_spatial_improvement_not_converged_stage44_finer_grid"
    )


def test_stage43_decision_preserves_mixed_scientific_outcomes() -> None:
    q_only = [
        _case(0.30, 1.0, 0.090, [1.0]),
        _case(0.20, 1.2, 0.080, [1.0]),
        _case(0.10, 1.4, 0.079, [1.0]),
    ]
    assert stage43_decision(q_only, True) == (
        "projected_polar_heat_flux_improves_velocity_discrepancy_stage44_wall_observable_audit"
    )
    velocity_only = [
        _case(0.10, 3.0, 0.079, [1.0]),
        _case(0.20, 2.0, 0.080, [1.0]),
        _case(0.30, 1.0, 0.090, [1.0]),
    ]
    assert stage43_decision(velocity_only, True) == (
        "projected_polar_velocity_improves_heat_flux_discrepancy_stage44_heat_flux_definition_audit"
    )


def test_stage43_decision_preserves_blockers_and_nonmonotonicity() -> None:
    base = [
        _case(0.30, 3.0, 0.090, [1.0]),
        _case(0.10, 1.0, 0.080, [1.0]),
        _case(0.20, 2.0, 0.085, [1.0]),
    ]
    assert stage43_decision(base, True) == (
        "projected_polar_nonmonotonic_stage44_space_velocity_coupling_audit"
    )
    assert stage43_decision(base, False) == "stage42_reproduction_blocker"
    blocked = list(base)
    blocked[2] = _case(0.20, 2.0, 0.085, [1.0], converged=False)
    assert stage43_decision(blocked, True) == "stage43_resolution_blocker"
