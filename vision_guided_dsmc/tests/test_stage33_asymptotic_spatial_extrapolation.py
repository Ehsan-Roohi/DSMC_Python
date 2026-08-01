from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage33_asymptotic_spatial_extrapolation import (
    STAGE33_GRIDS,
    linear_h_extrapolation,
    stage33_decision,
    strictly_decreasing,
    validate_stage33_design,
)


def test_stage33_fixed_design_is_valid() -> None:
    validate_stage33_design(STAGE33_GRIDS, 12000, 2.0e-5)


def test_stage33_requires_exactly_three_monotone_square_grids() -> None:
    with pytest.raises(ValueError):
        validate_stage33_design(((24, 24), (30, 30)), 100, 1e-5)
    with pytest.raises(ValueError):
        validate_stage33_design(((24, 24), (24, 30), (36, 36)), 100, 1e-5)
    with pytest.raises(ValueError):
        validate_stage33_design(((24, 24), (20, 20), (36, 36)), 100, 1e-5)


def test_linear_h_extrapolation_recovers_scalar_limit() -> None:
    sizes = np.array([24.0, 30.0, 36.0])
    values = 0.072 + 0.6 / sizes
    limit, slope, r2 = linear_h_extrapolation(sizes, values)
    assert float(limit) == pytest.approx(0.072, abs=1e-13)
    assert float(slope) == pytest.approx(0.6, abs=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-13)


def test_linear_h_extrapolation_recovers_profile_limit() -> None:
    sizes = np.array([24.0, 30.0, 36.0])
    limit_true = np.array([0.1, 0.2, 0.3])
    slope_true = np.array([1.0, -0.5, 0.25])
    values = np.stack([limit_true + slope_true / n for n in sizes])
    limit, slope, r2 = linear_h_extrapolation(sizes, values)
    np.testing.assert_allclose(limit, limit_true, atol=1e-13)
    np.testing.assert_allclose(slope, slope_true, atol=1e-12)
    assert r2 == pytest.approx(1.0, abs=1e-13)


def test_linear_h_extrapolation_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        linear_h_extrapolation(np.array([24.0, 30.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        linear_h_extrapolation(np.array([24.0, 0.0]), np.array([1.0, 2.0]))


def _row(q_error: float, velocity_error: float, converged: bool = True) -> dict[str, object]:
    return {
        "converged": converged,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": velocity_error,
            "sign_agreement": 1.0,
            "relative_l1": velocity_error,
        },
    }


def test_decision_advances_when_extrapolated_errors_are_small() -> None:
    rows = [_row(0.20, 0.40), _row(0.15, 0.30), _row(0.10, 0.20)]
    decision = stage33_decision(
        rows,
        extrapolated_q_error=0.05,
        extrapolated_velocity_metrics={
            "relative_rms": 0.15,
            "sign_agreement": 1.0,
            "relative_l1": 0.12,
        },
        finest_profile_change=0.10,
    )
    assert decision == (
        "spatial_truncation_explains_most_error_advance_second_order_spherical_kn0p1"
    )


def test_decision_reports_persistent_asymptotic_discrepancy() -> None:
    rows = [_row(0.20, 0.50), _row(0.18, 0.40), _row(0.17, 0.35)]
    decision = stage33_decision(
        rows,
        extrapolated_q_error=0.16,
        extrapolated_velocity_metrics={
            "relative_rms": 0.32,
            "sign_agreement": 0.9,
            "relative_l1": 0.30,
        },
        finest_profile_change=0.08,
    )
    assert decision == "asymptotic_limit_retains_model_or_benchmark_discrepancy"


def test_decision_extends_sequence_when_not_asymptotic() -> None:
    rows = [_row(0.20, 0.50), _row(0.18, 0.40), _row(0.17, 0.35)]
    decision = stage33_decision(
        rows,
        extrapolated_q_error=0.16,
        extrapolated_velocity_metrics={
            "relative_rms": 0.32,
            "sign_agreement": 0.9,
            "relative_l1": 0.30,
        },
        finest_profile_change=0.25,
    )
    assert decision == "not_yet_asymptotic_extend_spatial_sequence_without_retuning"


def test_strictly_decreasing_allows_small_numerical_tolerance() -> None:
    assert strictly_decreasing([1.0, 0.9, 0.8])
    assert strictly_decreasing([1.0, 1.005, 0.9])
    assert not strictly_decreasing([1.0, 1.05, 0.9])
