import numpy as np
import pytest

from vgdsmc import stage111_axis_conditioned_asymmetry_audit as s


def test_stage111_design_is_frozen():
    s.validate_stage111_design()
    with pytest.raises(ValueError):
        s.validate_stage111_design(limiter="vanleer")
    with pytest.raises(ValueError):
        s.validate_stage111_design(axis_rank_coupling_guard=0.2)
    with pytest.raises(ValueError):
        s.validate_stage111_design(stage110_run_id=-1)


def test_axis_maps_linear_x_profile_has_zero_asymmetry():
    x = np.arange(64, dtype=float)
    field = x[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._axis_same_sign_asymmetry_maps(field, np.array([1.0]))
    assert np.allclose(maps["x_same_sign_relative_asymmetry"], 0.0)
    assert np.allclose(maps["y_same_sign_relative_asymmetry"], 0.0)
    assert np.allclose(maps["y_same_sign_change_weighted_abs"], 0.0)


def test_axis_maps_quadratic_x_profile_localizes_change_to_x():
    x = np.arange(64, dtype=float)
    field = (x * x)[None, :, None]
    field = np.broadcast_to(field, (64, 64, 1)).copy()
    maps = s._axis_same_sign_asymmetry_maps(field, np.array([1.0]))
    assert np.sum(maps["x_same_sign_change_weighted_abs"]) > 0.0
    assert np.max(maps["x_same_sign_relative_asymmetry"]) > 0.0
    assert np.allclose(maps["y_same_sign_change_weighted_abs"], 0.0)


def test_axis_recombination_matches_stage110_definition():
    y, x = np.mgrid[:64, :64]
    field = (x * x + 0.5 * y * y)[..., None].astype(float)
    weight = np.array([1.0])
    axis = s._axis_same_sign_asymmetry_maps(field, weight)
    total = s.s110._same_sign_asymmetry_maps(field, weight)
    recombined_change = axis["x_same_sign_change_weighted_abs"] + axis["y_same_sign_change_weighted_abs"]
    recombined_centered = axis["x_same_sign_centered_slope_weighted_abs"] + axis["y_same_sign_centered_slope_weighted_abs"]
    recombined_relative = np.divide(
        recombined_change,
        recombined_centered,
        out=np.zeros_like(recombined_change),
        where=recombined_centered > 0.0,
    )
    assert np.allclose(recombined_change, total["same_sign_change_weighted_abs"])
    assert np.allclose(recombined_centered, total["same_sign_centered_slope_weighted_abs"])
    assert np.allclose(recombined_relative, total["same_sign_relative_asymmetry"])


def _axis_block(spearman: float, ratio: float):
    return {"coupling": {"spearman": spearman, "upper_to_lower_mean_amplitude_ratio": ratio}}


def _metrics(x_phi=(0.7, 2.0), y_phi=(0.2, 1.1), x_psi=(0.7, 2.0), y_psi=(0.2, 1.1), common_x=(0.7, 2.0), common_y=(0.2, 1.1)):
    return {
        "phi": {"x": _axis_block(*x_phi), "y": _axis_block(*y_phi)},
        "psi": {"x": _axis_block(*x_psi), "y": _axis_block(*y_psi)},
        "common_axis_coupling": {
            "x": {"spearman": common_x[0], "upper_to_lower_mean_amplitude_ratio": common_x[1]},
            "y": {"spearman": common_y[0], "upper_to_lower_mean_amplitude_ratio": common_y[1]},
        },
    }


def test_stage111_decision_x_only_route():
    assert s.stage111_decision(_metrics(), True, 1.0e-15) == (
        "stage111_x_axis_asymmetry_coupled_stage112_axis_specific_spatial_audit"
    )


def test_stage111_decision_y_only_route():
    metrics = _metrics(x_phi=(0.2, 1.1), x_psi=(0.2, 1.1), y_phi=(0.7, 2.0), y_psi=(0.7, 2.0), common_x=(0.2, 1.1), common_y=(0.7, 2.0))
    assert s.stage111_decision(metrics, True, 1.0e-15) == (
        "stage111_y_axis_asymmetry_coupled_stage112_axis_specific_spatial_audit"
    )


def test_stage111_decision_both_axes_without_dominance():
    metrics = _metrics(
        x_phi=(0.70, 2.0), y_phi=(0.65, 1.9), x_psi=(0.72, 2.1), y_psi=(0.66, 1.8),
        common_x=(0.70, 2.0), common_y=(0.64, 1.8),
    )
    assert s.stage111_decision(metrics, True, 1.0e-15) == (
        "stage111_both_axes_asymmetry_coupled_stage112_joint_axis_interaction_audit"
    )


def test_stage111_decision_x_dominance_requires_both_common_guards():
    metrics = _metrics(
        x_phi=(0.8, 2.5), y_phi=(0.6, 1.6), x_psi=(0.8, 2.5), y_psi=(0.6, 1.6),
        common_x=(0.8, 2.5), common_y=(0.6, 1.6),
    )
    assert s.stage111_decision(metrics, True, 1.0e-15) == (
        "stage111_x_axis_dominates_stage112_axis_specific_spatial_audit"
    )


def test_stage111_decision_no_axis_route():
    metrics = _metrics(x_phi=(0.2, 1.1), x_psi=(0.2, 1.1), y_phi=(0.3, 1.2), y_psi=(0.3, 1.2))
    assert s.stage111_decision(metrics, True, 1.0e-15) == (
        "stage111_axis_conditioning_not_sufficient_stage112_gradient_strength_confound_audit"
    )


def test_stage111_decision_preserves_blockers():
    assert s.stage111_decision({}, False, 0.0) == "stage111_nonfinite_axis_conditioning_blocker_without_retuning"
    assert s.stage111_decision({}, True, 1.0e-8) == "stage111_axis_decomposition_closure_blocker_without_retuning"
