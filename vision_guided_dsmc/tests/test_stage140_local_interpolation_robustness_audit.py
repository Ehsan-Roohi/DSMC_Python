import numpy as np
import pytest

from vgdsmc import stage140_local_interpolation_robustness_audit as s140


def test_stage140_design_is_frozen():
    s140.validate_stage140_design()
    with pytest.raises(ValueError):
        s140.validate_stage140_design(kn0=0.1)
    with pytest.raises(ValueError):
        s140.validate_stage140_design(physical_parameter_retuning=True)
    with pytest.raises(ValueError):
        s140.validate_stage140_design(cross_knudsen_extension_permitted=True)
    with pytest.raises(ValueError):
        s140.validate_stage140_design(root_spread_max_cells=0.30)


def test_lagrange3_reproduces_all_three_nodes():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, -2.0, 3.0])
    for xi, yi in zip(x, y):
        assert s140.lagrange3_value(float(xi), x, y) == pytest.approx(float(yi))


def test_quadratic_root_is_bracketed_and_reproducible():
    x = np.array([0.0, 1.0, 2.0])
    y = (x - 1.25) * (x + 2.0)
    root = s140.quadratic_root_in_bracket(x, y, (0, 1, 2), 1)
    assert root == pytest.approx(1.25, abs=1e-12)


def test_local_root_candidates_use_fixed_three_interpolants():
    x = np.arange(5.0)
    y = np.array([-2.0, -1.0, -0.2, 0.4, 1.1])
    names, roots = s140.local_root_candidates(x, y, 2)
    assert names == ["linear_secant", "left_quadratic", "right_quadratic"]
    assert roots.shape == (3,)
    assert np.isfinite(roots).all()
    assert np.all((roots >= 2.0) & (roots <= 3.0))


def test_stage140_classifies_robust_but_node_proximate_crossing():
    assert s140.classify_interpolation_robustness(
        candidate_count=3,
        all_roots_in_parent_bracket=True,
        root_span_cells=0.09,
        minimum_edge_clearance_fraction=0.08,
    ) == s140.ROBUST_NODE_PROXIMATE


def test_stage140_routes_interpolation_sensitive_crossing_without_retuning():
    assert s140.classify_interpolation_robustness(
        candidate_count=3,
        all_roots_in_parent_bracket=True,
        root_span_cells=0.30,
        minimum_edge_clearance_fraction=0.10,
    ) == s140.INTERPOLATION_SENSITIVE
    assert s140.classify_interpolation_robustness(
        candidate_count=3,
        all_roots_in_parent_bracket=False,
        root_span_cells=0.10,
        minimum_edge_clearance_fraction=0.10,
    ) == s140.INTERPOLATION_SENSITIVE


def test_stage140_provenance_and_support_blockers_are_hard():
    base = dict(
        candidate_count=3,
        all_roots_in_parent_bracket=True,
        root_span_cells=0.09,
        minimum_edge_clearance_fraction=0.08,
    )
    assert s140.classify_interpolation_robustness(**base, parent_record_ok=False) == s140.PARENT_RECORD_BLOCKER
    assert s140.classify_interpolation_robustness(**base, parent_route_ok=False) == s140.PARENT_ROUTE_BLOCKER
    sparse = dict(base)
    sparse["candidate_count"] = 2
    assert s140.classify_interpolation_robustness(**sparse) == s140.INSUFFICIENT_STENCILS
    assert s140.classify_interpolation_robustness(**base, finite=False) == s140.NONFINITE
