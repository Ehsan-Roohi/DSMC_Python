import numpy as np
import pytest

from vgdsmc import stage109_limiter_intervention_mode_decomposition_audit as s


def test_stage109_design_is_frozen():
    s.validate_stage109_design()
    with pytest.raises(ValueError):
        s.validate_stage109_design(limiter="vanleer")
    with pytest.raises(ValueError):
        s.validate_stage109_design(same_sign_dominance_guard=0.5)
    with pytest.raises(ValueError):
        s.validate_stage109_design(stage108_run_id=-1)


def test_radial_shell_partition_has_fixed_960_points_each():
    theta = np.tile(np.linspace(0.0, 2.0 * np.pi, s.RULE[1], endpoint=False), s.RULE[0])
    radius = np.repeat(np.linspace(0.1, 4.0, s.RULE[0]), s.RULE[1])
    vx = radius * np.cos(theta)
    vy = radius * np.sin(theta)
    labels = s._radial_shell_indices(vx, vy)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert [int(np.count_nonzero(labels == i)) for i in range(4)] == [960, 960, 960, 960]


def test_average_ranks_ties_are_average_ranked():
    x = np.array([1.0, 2.0, 2.0, 4.0])
    assert np.allclose(s._average_ranks(x), [1.0, 2.5, 2.5, 4.0])


def test_spearman_detects_monotone_relation():
    x = np.arange(12.0, dtype=float)
    assert np.isclose(s._spearman(x, x * x), 1.0)


def test_sign_reversal_pattern_is_assigned_to_zeroing_mode():
    pattern = np.resize(np.array([0.0, 2.0, 1.0]), 64)
    field = pattern[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._limiter_mode_maps(field, np.array([1.0]))
    assert np.sum(maps["zeroing_change_weighted_abs"]) > 0.0
    assert np.sum(maps["total_intervention_fraction"]) >= np.sum(maps["zeroing_intervention_fraction"])


def test_monotone_quadratic_pattern_is_assigned_to_same_sign_mode():
    x = np.arange(64, dtype=float)
    field = (x * x)[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._limiter_mode_maps(field, np.array([1.0]))
    assert np.sum(maps["same_sign_amplitude_change_weighted_abs"]) > 0.0
    assert np.sum(maps["zeroing_change_weighted_abs"]) == 0.0


def test_mode_fractions_close_to_total_by_construction():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(64, 64, 3))
    maps = s._limiter_mode_maps(field, np.array([0.2, 0.3, 0.5]))
    assert np.allclose(
        maps["total_intervention_fraction"],
        maps["zeroing_intervention_fraction"] + maps["same_sign_amplitude_intervention_fraction"],
        rtol=0.0,
        atol=0.0,
    )


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


def _mode_metrics(same_share=0.95, same_spearman=0.7, same_ratio=2.0, zero_share=0.05):
    block = {
        "same_sign_amplitude_weighted_change_share": same_share,
        "zeroing_weighted_change_share": zero_share,
        "same_sign_amplitude_coupling": {
            "spearman": same_spearman,
            "upper_to_lower_mean_amplitude_ratio": same_ratio,
        },
    }
    return {"phi": dict(block), "psi": dict(block)}


def test_stage109_decision_same_sign_route():
    assert s.stage109_decision(_mode_metrics(), True, 1.0e-15) == (
        "stage109_same_sign_amplitude_mode_dominates_stage110_same_sign_slope_asymmetry_audit"
    )


def test_stage109_decision_zeroing_route():
    metrics = _mode_metrics(same_share=0.6, same_spearman=0.3, same_ratio=1.1, zero_share=0.4)
    assert s.stage109_decision(metrics, True, 1.0e-15) == (
        "stage109_zeroing_mode_material_stage110_sign_reversal_geometry_audit"
    )


def test_stage109_decision_mixed_route():
    metrics = _mode_metrics(same_share=0.8, same_spearman=0.3, same_ratio=1.1, zero_share=0.2)
    assert s.stage109_decision(metrics, True, 1.0e-15) == (
        "stage109_mixed_or_weak_mode_coupling_stage110_mode_conditioned_spatial_audit"
    )


def test_stage109_decision_preserves_blockers():
    assert s.stage109_decision({}, False, 0.0) == (
        "stage109_nonfinite_mode_decomposition_blocker_without_retuning"
    )
    assert s.stage109_decision({}, True, 1.0e-8) == (
        "stage109_mode_decomposition_closure_blocker_without_retuning"
    )
