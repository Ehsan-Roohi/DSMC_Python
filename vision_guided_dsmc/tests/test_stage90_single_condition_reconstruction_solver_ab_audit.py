import numpy as np
import pytest

from vgdsmc import stage90_single_condition_reconstruction_solver_ab_audit as stage90


def test_frozen_design_accepts_defaults():
    stage90.validate_stage90_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 0.1},
        {"grid": (32, 32)},
        {"radial_scale": 1.0},
        {"source_relaxation": 0.5},
        {"material_benchmark_improvement": 0.05},
    ],
)
def test_frozen_design_rejects_retuning(override):
    with pytest.raises(ValueError):
        stage90.validate_stage90_design(**override)


def test_minmod_same_sign_uses_smaller_magnitude():
    a = np.array([2.0, -4.0])
    b = np.array([1.0, -3.0])
    np.testing.assert_allclose(stage90.minmod(a, b), [1.0, -3.0])


def test_minmod_opposite_sign_is_zero():
    np.testing.assert_array_equal(stage90.minmod(np.array([1.0, -1.0]), np.array([-2.0, 3.0])), 0.0)


def test_zero_boundary_slopes_are_retained():
    field = np.arange(4 * 5 * 2, dtype=float).reshape(4, 5, 2)
    sx = stage90.limited_slopes(field, axis=1, one_sided_boundary=False)
    assert np.all(sx[:, 0] == 0.0)
    assert np.all(sx[:, -1] == 0.0)


def test_one_sided_x_boundary_slopes_are_exact_first_differences():
    field = np.arange(4 * 5 * 2, dtype=float).reshape(4, 5, 2)
    sx = stage90.limited_slopes(field, axis=1, one_sided_boundary=True)
    np.testing.assert_allclose(sx[:, 0], field[:, 1] - field[:, 0])
    np.testing.assert_allclose(sx[:, -1], field[:, -1] - field[:, -2])


def test_y_boundary_slopes_remain_zero_in_both_stage90_arms():
    field = np.arange(4 * 5 * 2, dtype=float).reshape(4, 5, 2)
    sy = stage90.limited_slopes(field, axis=0, one_sided_boundary=False)
    assert np.all(sy[0] == 0.0)
    assert np.all(sy[-1] == 0.0)


def test_muscl_correction_is_globally_conservative_per_velocity_node():
    rng = np.random.default_rng(20260808)
    field = rng.random((5, 6, 4))
    vx = np.array([-2.0, -0.5, 0.75, 1.5])
    vy = np.array([0.5, -1.0, 1.25, -0.25])
    correction = stage90.muscl_correction_divergence(field, vx, vy, 1 / 6, 1 / 5, False)
    np.testing.assert_allclose(np.sum(correction, axis=(0, 1)), 0.0, atol=1e-12)


def test_ab_difference_is_confined_to_x_wall_adjacent_cell_pairs():
    rng = np.random.default_rng(89)
    field = rng.random((5, 7, 4))
    vx = np.array([-2.0, -0.5, 0.75, 1.5])
    vy = np.array([0.5, -1.0, 1.25, -0.25])
    baseline = stage90.muscl_correction_divergence(field, vx, vy, 1 / 7, 1 / 5, False)
    one_sided = stage90.muscl_correction_divergence(field, vx, vy, 1 / 7, 1 / 5, True)
    delta = one_sided - baseline
    np.testing.assert_allclose(delta[:, 2:-2], 0.0, atol=0.0)
    assert np.max(np.abs(delta[:, :2])) > 0.0
    assert np.max(np.abs(delta[:, -2:])) > 0.0


def test_compare_endpoints_uses_preregistered_ten_percent_materiality_guard():
    baseline = {
        "qav_relative_error": 0.30,
        "predicted_qav": 0.23,
        "velocity_metrics": {"relative_rms": 0.40},
    }
    one_sided = {
        "qav_relative_error": 0.24,
        "predicted_qav": 0.20,
        "velocity_metrics": {"relative_rms": 0.32},
    }
    comparison = stage90.compare_endpoints(baseline, one_sided)
    assert comparison["qav_material_improvement"] is True
    assert comparison["table3_material_improvement"] is True
    assert comparison["qav_not_degraded"] is True
    assert comparison["table3_not_degraded"] is True


def test_stage90_decision_preserves_all_negative_and_positive_routes():
    base = {
        "finite": True,
        "converged": True,
        "qav_relative_error": 0.30,
        "predicted_qav": 0.23,
        "velocity_metrics": {"relative_rms": 0.40},
    }
    improved = {
        "finite": True,
        "converged": True,
        "qav_relative_error": 0.24,
        "predicted_qav": 0.20,
        "velocity_metrics": {"relative_rms": 0.32},
    }
    c = stage90.compare_endpoints(base, improved)
    assert stage90.stage90_decision(base, improved, c).startswith("stage90_material_table3_table6_improvement")
    nonfinite = dict(improved, finite=False)
    assert stage90.stage90_decision(base, nonfinite, c) == "stage90_nonfinite_solver_blocker_without_retuning"
    nonconverged = dict(improved, converged=False)
    assert stage90.stage90_decision(base, nonconverged, c) == "stage90_nonconverged_solver_blocker_without_retuning"
    neutral = dict(base)
    c0 = stage90.compare_endpoints(base, neutral)
    assert stage90.stage90_decision(base, neutral, c0).startswith("stage90_no_material_benchmark_improvement")
