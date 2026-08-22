import numpy as np
import pytest

from vgdsmc import stage128_radial_node_continuity_audit as s128


def _synthetic_residual():
    residual = np.zeros(s128.GRID + (s128.RADIAL_NODES,), dtype=float)
    band = np.zeros(s128.GRID, dtype=np.int8)
    band[:, 4:14] = 1
    band[:, 14:42] = 2
    band[:, 42:52] = 1
    node_net = np.ones((3, s128.RADIAL_NODES), dtype=float)
    node_net[1, 1] = -1.0
    node_net[2, 1] = 1.0
    for node in (1, 9):
        for wall in (0, 1):
            switch = 15
            for depth in range(1, s128.DEPTH_COUNT + 1):
                j = s128._column_for_depth(wall, depth)
                b = int(np.unique(band[:, j])[0])
                sign = 1 if node_net[b, node] > 0 else -1
                if depth < switch:
                    residual[:28, j, node] = 1.0 * sign
                    residual[28:, j, node] = 3.0 * sign
                else:
                    residual[:28, j, node] = 3.0 * sign
                    residual[28:, j, node] = 1.0 * sign
    return residual, band, node_net


def test_stage128_design_is_frozen():
    s128.validate_stage128_design()
    with pytest.raises(ValueError):
        s128.validate_stage128_design(max_parent_crossing_offset_cells=3.0)
    with pytest.raises(ValueError):
        s128.validate_stage128_design(stage127_run_id=1)


def test_column_for_depth_maps_opposite_walls():
    assert s128._column_for_depth(0, 1) == 0
    assert s128._column_for_depth(0, 28) == 27
    assert s128._column_for_depth(1, 1) == 55
    assert s128._column_for_depth(1, 28) == 28


def test_crossing_depths_uses_linear_interpolation():
    a = np.ones(s128.DEPTH_COUNT)
    a[14:] = -1.0
    x = s128.crossing_depths(a)
    assert x.tolist() == pytest.approx([14.5])


def test_fixed_node_profile_holds_node_across_band_boundary():
    residual, band, node_net = _synthetic_residual()
    p0 = s128.fixed_node_profile(residual, band, node_net, 9, 0)
    p1 = s128.fixed_node_profile(residual, band, node_net, 9, 1)
    assert s128.crossing_depths(p0["tangential_asymmetry"])[0] == pytest.approx(14.5)
    assert s128.crossing_depths(p1["tangential_asymmetry"])[0] == pytest.approx(14.5)
    assert np.all(p0["net_sign"] == 1)
    assert np.all(p1["net_sign"] == 1)


def test_node_transition_metrics_identifies_sign_continuous_witness():
    residual, band, node_net = _synthetic_residual()
    p0 = s128.fixed_node_profile(residual, band, node_net, 9, 0)
    p1 = s128.fixed_node_profile(residual, band, node_net, 9, 1)
    m = s128.node_transition_metrics(p0, p1, np.array([14.5, 14.5]), node_net, 9)
    assert m["transition_reproduced"] is True
    assert m["mid_inner_net_sign_continuous"] is True
    assert m["maximum_parent_crossing_offset_cells"] == pytest.approx(0.0)


def test_node_transition_metrics_tracks_sign_change_separately():
    residual, band, node_net = _synthetic_residual()
    p0 = s128.fixed_node_profile(residual, band, node_net, 1, 0)
    p1 = s128.fixed_node_profile(residual, band, node_net, 1, 1)
    m = s128.node_transition_metrics(p0, p1, np.array([14.5, 14.5]), node_net, 1)
    assert m["transition_reproduced"] is True
    assert m["mid_inner_net_sign_continuous"] is False


def test_stage128_decision_routes_sign_continuous_reproduction():
    assert s128.stage128_decision(
        finite=True,
        parent_profile_closure=0.0,
        reproduced=[True, True],
        sign_continuous=[False, True],
    ) == s128.SIGN_CONTINUOUS_REPRODUCTION


def test_stage128_decision_routes_sign_changing_only_and_no_reproduction():
    assert s128.stage128_decision(
        finite=True,
        parent_profile_closure=0.0,
        reproduced=[True, False],
        sign_continuous=[False, True],
    ) == s128.SIGN_CHANGING_REPRODUCTION
    assert s128.stage128_decision(
        finite=True,
        parent_profile_closure=0.0,
        reproduced=[False, False],
        sign_continuous=[False, True],
    ) == s128.NO_FIXED_NODE_REPRODUCTION


def test_stage128_decision_blocks_nonfinite_and_parent_mismatch():
    assert s128.stage128_decision(
        finite=False,
        parent_profile_closure=0.0,
        reproduced=[True],
        sign_continuous=[True],
    ) == s128.NONFINITE
    assert s128.stage128_decision(
        finite=True,
        parent_profile_closure=10 * s128.PARENT_PROFILE_CLOSURE_TOLERANCE,
        reproduced=[True],
        sign_continuous=[True],
    ) == s128.CLOSURE_BLOCKER
