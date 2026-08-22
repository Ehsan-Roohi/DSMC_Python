import numpy as np
import pytest

from vgdsmc import stage79_dominant_moment_radial_angular_gradient_audit as stage79
from vgdsmc import stage87_one_sided_boundary_slope_counterfactual_audit as stage87


def test_stage87_design_is_frozen():
    stage87.validate_stage87_design()
    with pytest.raises(ValueError):
        stage87.validate_stage87_design(kn0=9.0)
    with pytest.raises(ValueError):
        stage87.validate_stage87_design(boundary_jump_recovery_primary_guard=0.4)


def test_one_sided_boundary_slopes_change_only_boundaries():
    x = np.arange(6.0)[None, :, None]
    field = np.repeat(np.repeat(x, 3, axis=0), 4, axis=2)
    retained = stage79.limited_slopes_x(field)
    cf = stage87.one_sided_boundary_slopes_x(field)
    assert np.allclose(cf[:, 1:-1], retained[:, 1:-1])
    assert np.allclose(cf[:, 0], field[:, 1] - field[:, 0])
    assert np.allclose(cf[:, -1], field[:, -1] - field[:, -2])
    assert np.all(retained[:, 0] == 0.0)
    assert np.all(retained[:, -1] == 0.0)


def test_counterfactual_flux_change_is_confined_to_wall_adjacent_faces_and_upwind_signs():
    rng = np.random.default_rng(7)
    field = rng.normal(size=(3, 7, 4))
    vx = np.array([2.0, -3.0, 0.5, -0.25])
    retained = stage79.interior_x_face_flux_difference_chunk(field, vx)
    cf = stage87.counterfactual_interior_x_face_flux_difference_chunk(field, vx)
    change = cf - retained
    assert np.all(change[:, 1:-1] == 0.0)
    assert np.all(change[:, 0, vx < 0.0] == 0.0)
    assert np.all(change[:, -1, vx > 0.0] == 0.0)
    assert np.any(change[:, 0, vx > 0.0] != 0.0)
    assert np.any(change[:, -1, vx < 0.0] != 0.0)


def test_constant_distribution_has_zero_counterfactual_change():
    field = np.ones((2, 5, 4))
    vx = np.array([1.0, -1.0, 2.0, -2.0])
    retained = stage79.interior_x_face_flux_difference_chunk(field, vx)
    cf = stage87.counterfactual_interior_x_face_flux_difference_chunk(field, vx)
    assert np.array_equal(retained, cf)


def test_stage87_decision_guards():
    base = {
        "finite": True,
        "baseline_stage86_relative_l2_closure_error": 0.0,
        "mean_boundary_jump_recovery_fraction": 0.6,
    }
    assert stage87.stage87_decision(base).endswith("full_moment_boundary_counterfactual_audit")
    partial = dict(base, mean_boundary_jump_recovery_fraction=0.2)
    assert stage87.stage87_decision(partial).endswith("remaining_direction_counterfactual_decomposition")
    weak = dict(base, mean_boundary_jump_recovery_fraction=0.01)
    assert stage87.stage87_decision(weak).endswith("boundary_distribution_curvature_audit")
    bad = dict(base, baseline_stage86_relative_l2_closure_error=1.0e-4)
    assert stage87.stage87_decision(bad) == "stage87_baseline_reconstruction_closure_blocker"
    nonfinite = dict(base, finite=False)
    assert stage87.stage87_decision(nonfinite) == "stage87_nonfinite_counterfactual_blocker"
