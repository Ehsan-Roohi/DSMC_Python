import numpy as np
import pytest

from vgdsmc import stage125_dominant_node_spatial_sign_audit as s125


def test_stage125_design_is_frozen():
    s125.validate_stage125_design()
    with pytest.raises(ValueError):
        s125.validate_stage125_design(net_sign_l1_fraction_min=0.7)
    with pytest.raises(ValueError):
        s125.validate_stage125_design(stage124_run_id=1)


def test_recompute_parent_arrays_closes_simple_band_fields():
    r = np.zeros((*s125.GRID, s125.RADIAL_NODES))
    band = np.zeros(s125.GRID, dtype=np.int8)
    band[20:40] = 1
    band[40:] = 2
    r[band == 0, 0] = 1.0
    r[band == 1, 0] = -2.0
    r[band == 2, 0] = 3.0
    net, absolute, unc = s125.recompute_parent_arrays(r, band)
    assert net[0, 0] == pytest.approx(np.count_nonzero(band == 0))
    assert net[1, 0] == pytest.approx(-2.0 * np.count_nonzero(band == 1))
    assert net[2, 0] == pytest.approx(3.0 * np.count_nonzero(band == 2))
    assert np.allclose(unc[:, 0], 1.0)
    assert np.all(absolute[:, 0] > 0.0)


def test_largest_component_uses_four_neighbor_connectivity_and_l1_weight():
    mask = np.zeros(s125.GRID, dtype=bool)
    weights = np.zeros(s125.GRID)
    mask[2:4, 2:4] = True
    weights[2:4, 2:4] = 1.0
    mask[10, 10] = True
    weights[10, 10] = 5.0
    share, cells, count = s125._largest_component_l1_fraction(mask, weights)
    assert count == 2
    assert cells == 1
    assert share == pytest.approx(5.0 / 9.0)


def test_spatial_metrics_detect_same_sign_halfspace_localization():
    band = np.ones(s125.GRID, dtype=bool)
    x = np.full(s125.GRID, -0.1)
    x[s125.GRID[0] // 2 :, :] = 1.0
    m = s125.dominant_node_spatial_metrics(x, band)
    assert m["net_sign"] == 1
    assert m["net_sign_l1_fraction"] > 0.8
    assert m["dominant_halfspace"] == "axis0_high"
    assert m["dominant_halfspace_l1_fraction"] == pytest.approx(1.0)


def test_spatial_metrics_exact_uncancelled_identity():
    band = np.ones(s125.GRID, dtype=bool)
    x = np.ones(s125.GRID)
    x[:14] = -1.0
    m = s125.dominant_node_spatial_metrics(x, band)
    assert m["node_uncancelled_fraction"] == pytest.approx(0.5)
    assert m["net_sign_l1_fraction"] == pytest.approx(0.75)
    assert m["opposite_sign_l1_fraction"] == pytest.approx(0.25)


def test_stage125_decision_prefers_localization_after_persistence():
    assert s125.stage125_decision(
        finite=True,
        parent_array_closure=0.0,
        net_sign_l1_fractions=[0.8, 0.76, 0.9],
        dominant_halfspace_l1_fractions=[0.8, 0.9, 0.75],
        largest_component_l1_fractions=[0.2, 0.3, 0.9],
    ) == s125.LOCALIZED


def test_stage125_decision_connected_diffuse_and_weak_routes():
    common = dict(finite=True, parent_array_closure=0.0)
    assert s125.stage125_decision(
        net_sign_l1_fractions=[0.8, 0.76, 0.9],
        dominant_halfspace_l1_fractions=[0.7, 0.7, 0.7],
        largest_component_l1_fractions=[0.5, 0.7, 0.9],
        **common,
    ) == s125.CONNECTED
    assert s125.stage125_decision(
        net_sign_l1_fractions=[0.8, 0.76, 0.9],
        dominant_halfspace_l1_fractions=[0.7, 0.7, 0.7],
        largest_component_l1_fractions=[0.49, 0.7, 0.9],
        **common,
    ) == s125.DIFFUSE
    assert s125.stage125_decision(
        net_sign_l1_fractions=[0.8, 0.74, 0.9],
        dominant_halfspace_l1_fractions=[1.0, 1.0, 1.0],
        largest_component_l1_fractions=[1.0, 1.0, 1.0],
        **common,
    ) == s125.WEAK


def test_stage125_blockers_precede_topology_routes():
    kwargs = dict(
        net_sign_l1_fractions=[1.0, 1.0, 1.0],
        dominant_halfspace_l1_fractions=[1.0, 1.0, 1.0],
        largest_component_l1_fractions=[1.0, 1.0, 1.0],
    )
    assert s125.stage125_decision(finite=False, parent_array_closure=0.0, **kwargs) == s125.NONFINITE
    assert s125.stage125_decision(
        finite=True,
        parent_array_closure=10.0 * s125.PARENT_ARRAY_CLOSURE_TOLERANCE,
        **kwargs,
    ) == s125.CLOSURE_BLOCKER
