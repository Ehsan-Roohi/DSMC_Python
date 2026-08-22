import numpy as np
import pytest

from vgdsmc.stage67_frozen_full_distribution_residual_decomposition import (
    STAGE67_RESIDUAL_BALANCE_GUARD,
    split_upwind_transport_chunk,
    stage67_decision,
    validate_stage67_design,
)


def _metrics(**updates):
    values = {
        "finite": True,
        "converged": True,
        "replay_within_tolerance": True,
        "normal_heat_flux_residual_ratio": STAGE67_RESIDUAL_BALANCE_GUARD / 10.0,
        "conserved_residual_ratio": STAGE67_RESIDUAL_BALANCE_GUARD / 10.0,
    }
    values.update(updates)
    return values


def test_stage67_design_accepts_frozen_stage58_values():
    validate_stage67_design()


def test_stage67_design_rejects_parameter_retuning():
    with pytest.raises(ValueError, match="Stage 58 is frozen"):
        validate_stage67_design(radial_scale=1.5)


def test_split_transport_is_zero_for_uniform_matching_state():
    distribution = np.ones((3, 4, 4))
    wall_profile_y = np.ones((3, 4))
    wall_profile_x = np.ones((4, 4))
    vx = np.asarray([1.0, -1.0, 0.0, 0.0])
    vy = np.asarray([0.0, 0.0, 1.0, -1.0])
    interior, wall = split_upwind_transport_chunk(
        distribution,
        wall_profile_y,
        wall_profile_y,
        wall_profile_x,
        wall_profile_x,
        vx,
        vy,
        0.25,
        1.0 / 3.0,
    )
    assert np.all(interior == 0.0)
    assert np.all(wall == 0.0)


def test_split_transport_places_boundary_inflow_only_in_wall_term():
    distribution = np.zeros((2, 2, 2))
    left = np.ones((2, 2))
    right = np.full((2, 2), 2.0)
    bottom = np.full((2, 2), 3.0)
    top = np.full((2, 2), 4.0)
    vx = np.asarray([1.0, -1.0])
    vy = np.asarray([1.0, -1.0])
    interior, wall = split_upwind_transport_chunk(
        distribution, left, right, bottom, top, vx, vy, 0.5, 0.5
    )
    assert np.all(interior == 0.0)
    assert wall[0, 0, 0] == pytest.approx(2.0 + 6.0)
    assert wall[-1, -1, 1] == pytest.approx(4.0 + 8.0)
    assert wall[-1, -1, 0] == 0.0
    assert wall[0, 0, 1] == 0.0


def test_decision_routes_nonfinite_blocker():
    assert stage67_decision(_metrics(finite=False)) == (
        "stage67_nonfinite_full_distribution_replay_blocker"
    )


def test_decision_routes_nonconverged_blocker():
    assert stage67_decision(_metrics(converged=False)) == (
        "stage67_frozen_replay_nonconverged_blocker_without_retuning"
    )


def test_decision_routes_replay_mismatch_blocker():
    assert stage67_decision(_metrics(replay_within_tolerance=False)) == (
        "stage67_stage58_replay_mismatch_blocker"
    )


def test_decision_routes_distribution_residual_blocker():
    assert stage67_decision(
        _metrics(normal_heat_flux_residual_ratio=STAGE67_RESIDUAL_BALANCE_GUARD * 1.01)
    ) == "stage67_distribution_fixed_point_residual_blocker"


def test_decision_routes_conserved_residual_blocker():
    assert stage67_decision(
        _metrics(conserved_residual_ratio=STAGE67_RESIDUAL_BALANCE_GUARD * 1.01)
    ) == "stage67_distribution_fixed_point_residual_blocker"


def test_decision_routes_independent_transport_audit():
    decision = stage67_decision(_metrics())
    assert "stage68" in decision
    assert "transport_operator_residual_audit" in decision
