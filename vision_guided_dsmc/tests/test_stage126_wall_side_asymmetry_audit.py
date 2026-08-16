import numpy as np
import pytest

from vgdsmc import stage126_wall_side_asymmetry_audit as s126


def test_stage126_design_is_frozen():
    s126.validate_stage126_design()
    with pytest.raises(ValueError):
        s126.validate_stage126_design(wall_balance_min=0.30)
    with pytest.raises(ValueError):
        s126.validate_stage126_design(stage125_run_id=1)


def test_recompute_halfspace_fractions_balanced_uniform_field():
    field = np.ones(s126.GRID)
    band = np.ones(s126.GRID, dtype=bool)
    f = s126.recompute_halfspace_fractions(field, band, 1)
    assert f["axis0_low"] == pytest.approx(0.5)
    assert f["axis0_high"] == pytest.approx(0.5)
    assert f["axis1_low"] == pytest.approx(0.5)
    assert f["axis1_high"] == pytest.approx(0.5)


def test_wall_side_metrics_detect_bilateral_axis0_high():
    ii, jj = np.indices(s126.GRID)
    band = np.ones(s126.GRID, dtype=bool)
    field = np.where(ii >= s126.GRID[0] // 2, 2.0, -0.2)
    m = s126.wall_side_metrics(field, band, 1)
    assert m["cross_wall_same_tangential_side"] is True
    assert m["combined_dominant_tangential_side"] == "axis0_high"
    assert m["minimum_wall_same_sign_l1_fraction"] == pytest.approx(0.5)
    assert m["minimum_per_wall_tangential_dominance_fraction"] == pytest.approx(1.0)


def test_wall_side_metrics_detect_bilateral_axis0_low():
    ii, jj = np.indices(s126.GRID)
    band = np.ones(s126.GRID, dtype=bool)
    field = np.where(ii < s126.GRID[0] // 2, -3.0, 0.1)
    m = s126.wall_side_metrics(field, band, -1)
    assert m["cross_wall_same_tangential_side"] is True
    assert m["combined_dominant_tangential_side"] == "axis0_low"
    assert m["minimum_per_wall_tangential_dominance_fraction"] == pytest.approx(1.0)


def test_wall_side_metrics_exposes_one_wall_dominance():
    ii, jj = np.indices(s126.GRID)
    band = np.ones(s126.GRID, dtype=bool)
    field = np.zeros(s126.GRID)
    field[(jj < s126.GRID[1] // 2) & (ii >= s126.GRID[0] // 2)] = 10.0
    field[(jj >= s126.GRID[1] // 2) & (ii >= s126.GRID[0] // 2)] = 1.0
    m = s126.wall_side_metrics(field, band, 1)
    assert m["minimum_wall_same_sign_l1_fraction"] == pytest.approx(1.0 / 11.0)
    assert m["cross_wall_same_tangential_side"] is True


def test_stage126_decision_routes_depth_side_reversal():
    assert s126.stage126_decision(
        finite=True,
        parent_metric_closure=0.0,
        minimum_wall_balance=0.45,
        minimum_per_wall_tangential_dominance=0.8,
        cross_wall_same_side=[True, True, True],
        band_side_codes=[1, 1, 0],
    ) == s126.BILATERAL_REVERSAL


def test_stage126_decision_routes_single_side_when_no_reversal():
    assert s126.stage126_decision(
        finite=True,
        parent_metric_closure=0.0,
        minimum_wall_balance=0.45,
        minimum_per_wall_tangential_dominance=0.8,
        cross_wall_same_side=[True, True, True],
        band_side_codes=[1, 1, 1],
    ) == s126.BILATERAL_SINGLE_SIDE


def test_stage126_decision_routes_one_wall_and_incoherent():
    common = dict(
        finite=True,
        parent_metric_closure=0.0,
        cross_wall_same_side=[True, True, True],
        band_side_codes=[1, 1, 1],
    )
    assert s126.stage126_decision(
        minimum_wall_balance=0.2,
        minimum_per_wall_tangential_dominance=0.9,
        **common,
    ) == s126.ONE_WALL
    assert s126.stage126_decision(
        minimum_wall_balance=0.45,
        minimum_per_wall_tangential_dominance=0.65,
        **common,
    ) == s126.INCOHERENT


def test_stage126_blockers_precede_spatial_routes():
    common = dict(
        minimum_wall_balance=0.5,
        minimum_per_wall_tangential_dominance=1.0,
        cross_wall_same_side=[True, True, True],
        band_side_codes=[1, 1, 0],
    )
    assert s126.stage126_decision(
        finite=False,
        parent_metric_closure=0.0,
        **common,
    ) == s126.NONFINITE
    assert s126.stage126_decision(
        finite=True,
        parent_metric_closure=10.0 * s126.PARENT_METRIC_CLOSURE_TOLERANCE,
        **common,
    ) == s126.CLOSURE_BLOCKER
