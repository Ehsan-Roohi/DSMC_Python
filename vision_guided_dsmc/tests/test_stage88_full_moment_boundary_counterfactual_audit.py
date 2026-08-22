import numpy as np
import pytest

from vgdsmc import stage76_local_velocity_frame_jump_audit as stage76
from vgdsmc import stage88_full_moment_boundary_counterfactual_audit as stage88


def test_stage88_design_is_frozen():
    stage88.validate_stage88_design()
    with pytest.raises(ValueError):
        stage88.validate_stage88_design(kn0=9.0)
    with pytest.raises(ValueError):
        stage88.validate_stage88_design(boundary_jump_recovery_primary_guard=0.4)
    with pytest.raises(ValueError):
        stage88.validate_stage88_design(moment_names=("transverse_kinetic",))


def test_stage88_one_sided_slope_changes_only_boundary_cells():
    x = np.arange(7.0)[None, :, None]
    field = np.repeat(np.repeat(x, 3, axis=0), 4, axis=2)
    retained = stage76.limited_slopes_x(field)
    counterfactual = stage88.one_sided_boundary_slopes_x(field)
    assert np.array_equal(counterfactual[:, 1:-1], retained[:, 1:-1])
    assert np.allclose(counterfactual[:, 0], field[:, 1] - field[:, 0])
    assert np.allclose(counterfactual[:, -1], field[:, -1] - field[:, -2])
    assert np.all(retained[:, 0] == 0.0)
    assert np.all(retained[:, -1] == 0.0)


def test_stage88_counterfactual_flux_change_is_wall_adjacent_only():
    rng = np.random.default_rng(88)
    field = rng.normal(size=(4, 8, 5))
    vx = np.array([2.0, -3.0, 0.5, -0.25, 0.0])
    retained = stage76.interior_x_face_flux_difference_chunk(field, vx)
    counterfactual = stage88.counterfactual_interior_x_face_flux_difference_chunk(field, vx)
    change = counterfactual - retained
    assert np.all(change[:, 1:-1] == 0.0)
    assert np.all(change[:, 0, vx <= 0.0] == 0.0)
    assert np.all(change[:, -1, vx >= 0.0] == 0.0)
    assert np.any(change[:, 0, vx > 0.0] != 0.0)
    assert np.any(change[:, -1, vx < 0.0] != 0.0)


def test_stage88_component_divergence_is_conservative():
    rng = np.random.default_rng(4)
    faces = rng.normal(size=(3, 64, 63))
    cells = stage88.component_divergence(faces)
    assert cells.shape == (3, 64, 64)
    for index in range(3):
        assert abs(float(np.sum(cells[index]))) < 1.0e-12
    total_from_components = np.sum(cells, axis=0)
    total_direct = stage88.divergence_from_interior_faces(np.sum(faces, axis=0))
    assert np.allclose(total_from_components, total_direct)


def test_stage88_decision_guards():
    base = {
        "finite": True,
        "baseline_stage76_component_relative_l2_closure_error": 0.0,
        "maximum_counterfactual_change_away_from_wall_adjacent_faces": 0.0,
        "maximum_global_conservation_ratio": 0.0,
        "baseline_moment_sum_cell_relative_l2_error": 0.0,
        "counterfactual_moment_sum_cell_relative_l2_error": 0.0,
        "minimum_total_boundary_jump_recovery_fraction": 0.6,
    }
    assert stage88.stage88_decision(base).endswith("boundary_reconstruction_admissibility_audit")
    partial = dict(base, minimum_total_boundary_jump_recovery_fraction=0.2)
    assert stage88.stage88_decision(partial).endswith("momentwise_cancellation_audit")
    weak = dict(base, minimum_total_boundary_jump_recovery_fraction=0.01)
    assert stage88.stage88_decision(weak).endswith("dominant_subspace_reconciliation_audit")
    nonfinite = dict(base, finite=False)
    assert stage88.stage88_decision(nonfinite) == "stage88_nonfinite_full_moment_counterfactual_blocker"
    bad_closure = dict(base, baseline_stage76_component_relative_l2_closure_error=1.0e-3)
    assert stage88.stage88_decision(bad_closure) == "stage88_baseline_common_frame_component_closure_blocker"
    leaked = dict(base, maximum_counterfactual_change_away_from_wall_adjacent_faces=1.0e-12)
    assert stage88.stage88_decision(leaked) == "stage88_counterfactual_scope_leakage_blocker"
    bad_conservation = dict(base, maximum_global_conservation_ratio=1.0e-5)
    assert stage88.stage88_decision(bad_conservation) == "stage88_counterfactual_conservation_blocker"
