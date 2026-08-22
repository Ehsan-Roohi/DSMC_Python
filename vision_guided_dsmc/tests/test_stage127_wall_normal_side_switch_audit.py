import numpy as np
import pytest

from vgdsmc import stage127_wall_normal_side_switch_audit as s127


def _synthetic_profiles(switch_low=15, switch_high=16):
    field = np.zeros(s127.GRID, dtype=float)
    band = np.zeros(s127.GRID, dtype=np.int8)
    band[:, 4:14] = 1
    band[:, 14:42] = 2
    band[:, 42:52] = 1
    band[:, 52:] = 0
    signs = np.array([1, 1, 1], dtype=np.int8)
    for wall, switch in ((0, switch_low), (1, switch_high)):
        for depth in range(1, s127.DEPTH_COUNT + 1):
            j = s127._column_for_depth(wall, depth)
            if depth < switch:
                field[:28, j] = 1.0
                field[28:, j] = 3.0
            else:
                field[:28, j] = 3.0
                field[28:, j] = 1.0
    return field, band, signs


def test_stage127_design_is_frozen():
    s127.validate_stage127_design()
    with pytest.raises(ValueError):
        s127.validate_stage127_design(cross_wall_profile_cosine_min=0.90)
    with pytest.raises(ValueError):
        s127.validate_stage127_design(stage126_run_id=1)


def test_column_for_depth_maps_opposite_walls():
    assert s127._column_for_depth(0, 1) == 0
    assert s127._column_for_depth(0, 28) == 27
    assert s127._column_for_depth(1, 1) == 55
    assert s127._column_for_depth(1, 28) == 28


def test_crossing_depths_linear_interpolation():
    a = np.ones(s127.DEPTH_COUNT)
    a[1:] = -1.0
    x = s127.crossing_depths(a)
    assert x.shape == (1,)
    assert x[0] == pytest.approx(1.5)


def test_wall_depth_profile_detects_single_switch():
    field, band, signs = _synthetic_profiles()
    p0 = s127.wall_depth_profile(field, band, signs, 0)
    p1 = s127.wall_depth_profile(field, band, signs, 1)
    assert np.all(p0["dominant_side_code"][:14] == 1)
    assert np.all(p0["dominant_side_code"][14:] == 0)
    assert np.all(p1["dominant_side_code"][:15] == 1)
    assert np.all(p1["dominant_side_code"][15:] == 0)
    assert s127.crossing_depths(p0["tangential_asymmetry"])[0] == pytest.approx(14.5)
    assert s127.crossing_depths(p1["tangential_asymmetry"])[0] == pytest.approx(15.5)


def test_parent_fraction_closure_recombines_depth_lines():
    field, band, signs = _synthetic_profiles()
    profiles = [s127.wall_depth_profile(field, band, signs, w) for w in range(2)]
    parent_wall = np.zeros((3, 2))
    parent_side = np.zeros((3, 2, 2))
    for b in range(3):
        totals = []
        for w, p in enumerate(profiles):
            m = p["band_code"] == b
            total = float(np.sum(p["same_sign_l1"][m]))
            totals.append(total)
            parent_side[b, w, 0] = np.sum(p["same_sign_l1"][m] * p["axis0_low_fraction"][m]) / total
            parent_side[b, w, 1] = np.sum(p["same_sign_l1"][m] * p["axis0_high_fraction"][m]) / total
        parent_wall[b] = np.asarray(totals) / sum(totals)
    assert s127._stage126_fraction_closure(profiles, parent_wall, parent_side) < 1e-15


def test_stage127_decision_routes_bilateral_switch():
    assert s127.stage127_decision(
        finite=True,
        parent_metric_closure=0.0,
        crossing_counts=[1, 1],
        crossing_separation=1.0,
        cross_wall_profile_cosine=0.98,
        depth_side_agreement=27 / 28,
    ) == s127.BILATERAL_SWITCH


def test_stage127_decision_routes_wall_specific_switch():
    assert s127.stage127_decision(
        finite=True,
        parent_metric_closure=0.0,
        crossing_counts=[1, 1],
        crossing_separation=3.0,
        cross_wall_profile_cosine=0.98,
        depth_side_agreement=1.0,
    ) == s127.WALL_SPECIFIC_SWITCH
    assert s127.stage127_decision(
        finite=True,
        parent_metric_closure=0.0,
        crossing_counts=[2, 1],
        crossing_separation=float("inf"),
        cross_wall_profile_cosine=0.98,
        depth_side_agreement=1.0,
    ) == s127.WALL_SPECIFIC_SWITCH


def test_stage127_decision_routes_no_switch_and_blockers():
    common = dict(
        crossing_counts=[0, 0],
        crossing_separation=float("inf"),
        cross_wall_profile_cosine=1.0,
        depth_side_agreement=1.0,
    )
    assert s127.stage127_decision(finite=True, parent_metric_closure=0.0, **common) == s127.NO_SWITCH
    assert s127.stage127_decision(finite=False, parent_metric_closure=0.0, **common) == s127.NONFINITE
    assert s127.stage127_decision(
        finite=True,
        parent_metric_closure=10 * s127.PARENT_METRIC_CLOSURE_TOLERANCE,
        **common,
    ) == s127.CLOSURE_BLOCKER
