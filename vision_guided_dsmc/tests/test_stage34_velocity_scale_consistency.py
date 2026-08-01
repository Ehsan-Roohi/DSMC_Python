import math
import numpy as np
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig
from vgdsmc.stage34_velocity_scale_consistency import (
    LEGACY_SPHERICAL_BASELINES,
    STAGE34_GRID,
    STAGE34_KNUDSEN,
    corrected_case_metrics,
    legacy_c0_tau_prefactor,
    local_relaxation_time,
    paper_consistent_c0_tau_prefactor,
    paper_zeta_tau_prefactor,
    solve_reduced_case_with_mapping,
    stage34_decision,
    validate_stage34_design,
)
from vgdsmc.velocity_quadrature_audit import spherical_product


def test_paper_zeta_prefactor_matches_equation_six():
    assert paper_zeta_tau_prefactor(1.0) == pytest.approx(2.0 / math.sqrt(math.pi))


def test_c0_conversion_reduces_tau_by_sqrt_two():
    legacy = legacy_c0_tau_prefactor(0.1)
    corrected = paper_consistent_c0_tau_prefactor(0.1)
    assert legacy / corrected == pytest.approx(math.sqrt(2.0))
    assert corrected == pytest.approx(math.sqrt(2.0) * 0.1 / math.sqrt(math.pi))


def test_local_relaxation_time_modes_are_positive_and_distinct():
    cfg = LinearSidewallConfig(kn0=1.0)
    density = np.array([[1.0, 2.0]])
    temperature = np.array([[0.5, 1.0]])
    legacy = local_relaxation_time(density, temperature, cfg, "legacy_c0")
    corrected = local_relaxation_time(
        density, temperature, cfg, "paper_consistent_c0"
    )
    assert np.all(corrected > 0.0)
    assert np.allclose(legacy / corrected, math.sqrt(2.0))


def test_unknown_mapping_is_rejected():
    cfg = LinearSidewallConfig(kn0=1.0)
    with pytest.raises(ValueError, match="unknown relaxation mapping"):
        local_relaxation_time(np.ones((1, 1)), np.ones((1, 1)), cfg, "tuned")


def test_stage34_design_is_fixed_and_rejects_retuning():
    validate_stage34_design(STAGE34_KNUDSEN, STAGE34_GRID, 100, 1e-5)
    with pytest.raises(ValueError):
        validate_stage34_design((0.2, 1.0, 10.0), STAGE34_GRID, 100, 1e-5)
    with pytest.raises(ValueError):
        validate_stage34_design(STAGE34_KNUDSEN, (16, 16), 100, 1e-5)


def test_small_corrected_solver_produces_finite_positive_state():
    quadrature = spherical_product(4, 4, 8, 4.0, "stage34_test_rule")
    cfg = LinearSidewallConfig(
        nx=3,
        ny=3,
        kn0=0.1,
        cold_hot_ratio=0.1,
        max_steps=3,
        check_interval=1,
        minimum_steps=1,
        tolerance=1e-30,
        cfl=0.1,
    )
    result = solve_reduced_case_with_mapping(cfg, quadrature)
    assert np.isfinite(result["T"]).all()
    assert np.isfinite(result["bottom_heat_flux"]).all()
    assert np.isfinite(result["left_wall_velocity"]).all()
    assert float(result["minimum_distribution"]) > 0.0
    assert float(result["wall_mass_balance_relative_error"]) < 1e-10


def test_corrected_metrics_include_all_fixed_wall_observables():
    quadrature = spherical_product(4, 4, 8, 4.0, "stage34_metric_rule")
    cfg = LinearSidewallConfig(
        nx=4,
        ny=4,
        kn0=0.1,
        cold_hot_ratio=0.1,
        max_steps=2,
        check_interval=1,
        minimum_steps=1,
        tolerance=1e-30,
        cfl=0.1,
    )
    result = solve_reduced_case_with_mapping(cfg, quadrature)
    metrics = corrected_case_metrics(result, cfg, quadrature)
    assert set(metrics["observable_metrics"]) == {
        "boundary_mixture",
        "adjacent_cell_center",
        "linear_extrapolated_wall",
    }
    assert math.isfinite(metrics["qav_relative_error"])


def test_decision_adopts_only_consistent_cross_kn_improvement():
    improving = [
        {
            "qav_error_ratio_corrected_to_legacy": 0.7,
            "boundary_velocity_error_ratio_corrected_to_legacy": 0.8,
            "boundary_sign_agreement_change": 0.0,
        },
        {
            "qav_error_ratio_corrected_to_legacy": 0.8,
            "boundary_velocity_error_ratio_corrected_to_legacy": 0.75,
            "boundary_sign_agreement_change": 0.1,
        },
        {
            "qav_error_ratio_corrected_to_legacy": 0.95,
            "boundary_velocity_error_ratio_corrected_to_legacy": 0.95,
            "boundary_sign_agreement_change": 0.0,
        },
    ]
    assert stage34_decision(improving).startswith("adopt_paper_consistent")


def test_decision_preserves_mixed_or_negative_outcomes():
    mixed = [
        {
            "qav_error_ratio_corrected_to_legacy": 0.7,
            "boundary_velocity_error_ratio_corrected_to_legacy": 1.1,
            "boundary_sign_agreement_change": 0.0,
        },
        {
            "qav_error_ratio_corrected_to_legacy": 1.1,
            "boundary_velocity_error_ratio_corrected_to_legacy": 0.8,
            "boundary_sign_agreement_change": 0.0,
        },
        {
            "qav_error_ratio_corrected_to_legacy": 1.0,
            "boundary_velocity_error_ratio_corrected_to_legacy": 1.0,
            "boundary_sign_agreement_change": 0.0,
        },
    ]
    assert stage34_decision(mixed).startswith("mixed_cross_kn")
    negative = [
        {
            "qav_error_ratio_corrected_to_legacy": 1.3,
            "boundary_velocity_error_ratio_corrected_to_legacy": 1.0,
            "boundary_sign_agreement_change": 0.0,
        },
        {
            "qav_error_ratio_corrected_to_legacy": 1.0,
            "boundary_velocity_error_ratio_corrected_to_legacy": 1.3,
            "boundary_sign_agreement_change": 0.0,
        },
        {
            "qav_error_ratio_corrected_to_legacy": 1.0,
            "boundary_velocity_error_ratio_corrected_to_legacy": 1.0,
            "boundary_sign_agreement_change": 0.0,
        },
    ]
    assert "additional_implementation_discrepancy" in stage34_decision(negative)


def test_legacy_baselines_are_explicit_for_all_three_knudsen_numbers():
    assert set(LEGACY_SPHERICAL_BASELINES) == {0.1, 1.0, 10.0}
    assert LEGACY_SPHERICAL_BASELINES[1.0]["source_stage"] == 30
    assert LEGACY_SPHERICAL_BASELINES[0.1]["source_stage"] == 31
