import numpy as np
import pytest

from vgdsmc import stage129_transition_strength_audit as s129


def _profile(width=4.0, support_scale=1.0):
    depth = np.arange(1, s129.DEPTH_COUNT + 1, dtype=float)
    a = -0.8 * np.tanh((depth - 15.0) / max(width / 2.0, 1e-6))
    support = np.ones_like(depth) * support_scale
    return depth, a, support


def test_stage129_design_is_frozen():
    s129.validate_stage129_design()
    with pytest.raises(ValueError):
        s129.validate_stage129_design(witness_node=1)
    with pytest.raises(ValueError):
        s129.validate_stage129_design(max_transition_width_cells=7.0)


def test_crossing_depth_uses_linear_interpolation():
    a = np.ones(s129.DEPTH_COUNT)
    a[14:] = -1.0
    x, k = s129.crossing_depth(a)
    assert x == pytest.approx(14.5)
    assert k == 13


def test_transition_metrics_identifies_material_transition():
    depth, a, support = _profile(width=3.0)
    m = s129.transition_metrics(depth, a, support)
    assert m["material_strength_per_wall"] is True
    assert m["mid_band_asymmetry_median"] >= s129.ASYMMETRY_MEDIAN_MIN
    assert m["inner_band_asymmetry_median"] <= -s129.ASYMMETRY_MEDIAN_MIN
    assert m["crossing_support_ratio"] == pytest.approx(1.0)
    assert m["transition_width_25_to_75_cells"] <= s129.MAX_TRANSITION_WIDTH_CELLS


def test_transition_metrics_rejects_weak_side_contrast():
    depth = np.arange(1, s129.DEPTH_COUNT + 1, dtype=float)
    a = -0.3 * np.tanh((depth - 15.0) / 1.5)
    m = s129.transition_metrics(depth, a, np.ones_like(depth))
    assert m["material_strength_per_wall"] is False


def test_transition_metrics_rejects_low_crossing_support():
    depth, a, support = _profile(width=3.0)
    _, k = s129.crossing_depth(a)
    support[k:k + 2] = 0.01
    m = s129.transition_metrics(depth, a, support)
    assert m["crossing_support_ratio"] < s129.TRANSITION_SUPPORT_RATIO_MIN
    assert m["material_strength_per_wall"] is False


def test_stage129_decision_routes_material_and_weak():
    assert s129.stage129_decision(
        finite=True, provenance_ok=True, wall_material=[True, True], width_difference=0.5
    ) == s129.MATERIAL_TRANSITION
    assert s129.stage129_decision(
        finite=True, provenance_ok=True, wall_material=[True, False], width_difference=0.5
    ) == s129.WEAK_TRANSITION
    assert s129.stage129_decision(
        finite=True, provenance_ok=True, wall_material=[True, True], width_difference=3.0
    ) == s129.WEAK_TRANSITION


def test_stage129_decision_blocks_nonfinite_and_bad_provenance():
    assert s129.stage129_decision(
        finite=False, provenance_ok=True, wall_material=[True, True], width_difference=0.0
    ) == s129.NONFINITE
    assert s129.stage129_decision(
        finite=True, provenance_ok=False, wall_material=[True, True], width_difference=0.0
    ) == s129.PROVENANCE_BLOCKER
