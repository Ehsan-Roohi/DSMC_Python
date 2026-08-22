import numpy as np
import pytest

from vgdsmc import stage110_same_sign_slope_asymmetry_audit as s


def test_stage110_design_is_frozen():
    s.validate_stage110_design()
    with pytest.raises(ValueError):
        s.validate_stage110_design(limiter="vanleer")
    with pytest.raises(ValueError):
        s.validate_stage110_design(asymmetry_rank_coupling_guard=0.2)
    with pytest.raises(ValueError):
        s.validate_stage110_design(stage109_run_id=-1)


def test_radial_shell_partition_has_fixed_960_points_each():
    theta = np.tile(np.linspace(0.0, 2.0 * np.pi, s.RULE[1], endpoint=False), s.RULE[0])
    radius = np.repeat(np.linspace(0.1, 4.0, s.RULE[0]), s.RULE[1])
    vx = radius * np.cos(theta)
    vy = radius * np.sin(theta)
    labels = s._radial_shell_indices(vx, vy)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert [int(np.count_nonzero(labels == i)) for i in range(4)] == [960, 960, 960, 960]


def test_equal_same_sign_slopes_have_zero_relative_asymmetry():
    x = np.arange(64, dtype=float)
    field = x[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._same_sign_asymmetry_maps(field, np.array([1.0]))
    assert np.allclose(maps["same_sign_change_weighted_abs"], 0.0)
    assert np.allclose(maps["same_sign_relative_asymmetry"], 0.0)


def test_quadratic_same_sign_slopes_have_nonzero_asymmetry():
    x = np.arange(64, dtype=float)
    field = (x * x)[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._same_sign_asymmetry_maps(field, np.array([1.0]))
    assert np.sum(maps["same_sign_change_weighted_abs"]) > 0.0
    assert np.max(maps["same_sign_relative_asymmetry"]) > 0.0
    assert np.min(maps["same_sign_relative_asymmetry"]) >= 0.0
    assert np.max(maps["same_sign_relative_asymmetry"]) <= 1.0 + 1.0e-15


def test_same_sign_change_matches_half_magnitude_imbalance_for_simple_profile():
    x = np.arange(64, dtype=float)
    profile = x * x
    field = profile[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._same_sign_asymmetry_maps(field, np.array([1.0]))
    j = 20
    expected_local_x_change = 0.5 * abs((2.0 * j - 1.0) - (2.0 * j + 1.0))
    assert np.isclose(maps["same_sign_change_weighted_abs"][10, j - s.WALL_BAND_CELLS], expected_local_x_change)


def test_coupling_metrics_detects_strong_monotone_stratification():
    severity = np.arange(56.0 * 56.0, dtype=float).reshape(56, 56)
    amplitude = 1.0 + 2.0 * severity
    metrics = s._coupling_metrics(severity, amplitude)
    assert metrics["spearman"] > 0.99
    assert metrics["upper_to_lower_mean_amplitude_ratio"] > s.QUARTILE_AMPLITUDE_RATIO_GUARD


def test_coupling_metrics_rejects_negative_values():
    severity = np.ones((56, 56))
    amplitude = np.ones((56, 56))
    amplitude[0, 0] = -1.0
    with pytest.raises(ValueError):
        s._coupling_metrics(severity, amplitude)


def _factor_metrics(asym_spearman=0.7, asym_ratio=2.0, grad_spearman=0.3, grad_ratio=1.1):
    block = {
        "relative_asymmetry_coupling": {
            "spearman": asym_spearman,
            "upper_to_lower_mean_amplitude_ratio": asym_ratio,
        },
        "same_sign_gradient_strength_coupling": {
            "spearman": grad_spearman,
            "upper_to_lower_mean_amplitude_ratio": grad_ratio,
        },
    }
    return {"phi": dict(block), "psi": dict(block)}


def test_stage110_decision_relative_asymmetry_route():
    assert s.stage110_decision(_factor_metrics(), True, 1.0e-15) == (
        "stage110_relative_same_sign_asymmetry_coupled_stage111_axis_conditioned_asymmetry_audit"
    )


def test_stage110_decision_gradient_strength_route():
    metrics = _factor_metrics(asym_spearman=0.2, asym_ratio=1.1, grad_spearman=0.7, grad_ratio=2.0)
    assert s.stage110_decision(metrics, True, 1.0e-15) == (
        "stage110_same_sign_gradient_strength_coupled_stage111_gradient_strength_conditioning_audit"
    )


def test_stage110_decision_joint_factor_route():
    metrics = _factor_metrics(asym_spearman=0.2, asym_ratio=1.1, grad_spearman=0.3, grad_ratio=1.2)
    assert s.stage110_decision(metrics, True, 1.0e-15) == (
        "stage110_same_sign_mode_not_explained_by_single_factor_stage111_joint_factor_spatial_audit"
    )


def test_stage110_decision_preserves_blockers():
    assert s.stage110_decision({}, False, 0.0) == (
        "stage110_nonfinite_same_sign_asymmetry_blocker_without_retuning"
    )
    assert s.stage110_decision({}, True, 1.0e-8) == (
        "stage110_same_sign_mode_closure_blocker_without_retuning"
    )
