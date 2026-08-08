import numpy as np
import pytest

from vgdsmc import stage89_boundary_reconstruction_admissibility_audit as stage89


def test_stage89_design_is_frozen():
    stage89.validate_stage89_design()
    with pytest.raises(ValueError):
        stage89.validate_stage89_design(kn0=9.0)
    with pytest.raises(ValueError):
        stage89.validate_stage89_design(negativity_rel_guard=1.0e-8)
    with pytest.raises(ValueError):
        stage89.validate_stage89_design(counterfactual_boundary_slope="zero")


def test_stage89_modified_boundary_upwind_state_is_neighbor_midpoint():
    rng = np.random.default_rng(89)
    field = 0.2 + rng.random((5, 7, 6))
    vx = np.array([2.0, 1.0, -0.5, -3.0, 0.25, -0.75])

    left = stage89.wall_adjacent_face_states(field, vx, "left")
    left_mask = vx > 0.0
    assert np.allclose(
        left["counterfactual"][:, left_mask],
        0.5 * (field[:, 0, left_mask] + field[:, 1, left_mask]),
    )
    assert np.array_equal(
        left["counterfactual"][:, vx < 0.0],
        left["baseline"][:, vx < 0.0],
    )

    right = stage89.wall_adjacent_face_states(field, vx, "right")
    right_mask = vx < 0.0
    assert np.allclose(
        right["counterfactual"][:, right_mask],
        0.5 * (field[:, -1, right_mask] + field[:, -2, right_mask]),
    )
    assert np.array_equal(
        right["counterfactual"][:, vx > 0.0],
        right["baseline"][:, vx > 0.0],
    )


def test_stage89_minmod_counterfactual_face_state_is_neighbor_bounded_for_positive_data():
    rng = np.random.default_rng(890)
    field = 1.0e-3 + rng.random((8, 9, 10))
    vx = np.linspace(-2.5, 2.5, 10)
    for side in ("left", "right"):
        state = stage89.wall_adjacent_face_states(field, vx, side)
        used = state["used_mask"]
        lower = np.minimum(state["boundary"], state["neighbor"])
        upper = np.maximum(state["boundary"], state["neighbor"])
        candidate = state["counterfactual"]
        assert np.all(candidate[:, used] >= lower[:, used] - 1.0e-15)
        assert np.all(candidate[:, used] <= upper[:, used] + 1.0e-15)
        assert np.all(candidate[:, used] > 0.0)
        row = stage89._state_admissibility_row("phi", side, state)
        assert row["new_negative_count"] == 0
        assert row["neighbor_bound_overshoot_normalized"] <= 1.0e-15
        assert row["maximum_change_on_unmodified_halfspace"] == 0.0


def test_stage89_reduced_moment_linearity_for_modified_halfspace():
    rng = np.random.default_rng(891)
    phi = 0.1 + rng.random((6, 8, 12))
    psi = 0.1 + rng.random((6, 8, 12))
    theta = np.linspace(0.1, 2.9, 12)
    vx = np.cos(theta)
    vy = np.sin(theta)
    weight = 0.2 + rng.random(12)
    for side in ("left", "right"):
        phi_state = stage89.wall_adjacent_face_states(phi, vx, side)
        psi_state = stage89.wall_adjacent_face_states(psi, vx, side)
        metrics, profiles = stage89.side_moment_metrics(side, phi_state, psi_state, vx, vy, weight)
        assert metrics["modified_halfspace_moment_linearity_relative_l2_error"] <= 1.0e-14
        assert np.allclose(profiles["candidate_active"], profiles["reference_active"])


def test_stage89_decision_guards():
    base = {
        "finite": True,
        "maximum_change_on_unmodified_halfspace": 0.0,
        "maximum_candidate_negativity_normalized": 0.0,
        "maximum_neighbor_bound_overshoot_normalized": 0.0,
        "maximum_modified_halfspace_midpoint_relative_l2_error": 0.0,
        "maximum_modified_halfspace_moment_linearity_relative_l2_error": 0.0,
    }
    assert stage89.stage89_decision(base).endswith("single_condition_reconstruction_solver_ab_audit")
    assert stage89.stage89_decision(dict(base, finite=False)) == "stage89_nonfinite_reconstruction_admissibility_blocker"
    assert stage89.stage89_decision(dict(base, maximum_change_on_unmodified_halfspace=1.0e-15)) == "stage89_counterfactual_halfspace_scope_leakage_blocker"
    assert "nonnegativity_blocker" in stage89.stage89_decision(
        dict(base, maximum_candidate_negativity_normalized=1.0e-8)
    )
    assert "overshoot_blocker" in stage89.stage89_decision(
        dict(base, maximum_neighbor_bound_overshoot_normalized=1.0e-8)
    )
    assert stage89.stage89_decision(
        dict(base, maximum_modified_halfspace_midpoint_relative_l2_error=1.0e-8)
    ) == "stage89_midpoint_reconstruction_closure_blocker"
    assert stage89.stage89_decision(
        dict(base, maximum_modified_halfspace_moment_linearity_relative_l2_error=1.0e-8)
    ) == "stage89_reduced_moment_linearity_closure_blocker"
