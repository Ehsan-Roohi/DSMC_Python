from __future__ import annotations

import math

import numpy as np
import pytest

from vgdsmc import stage73_velocity_sign_angular_bin_interior_face_flux_attribution_audit as stage73
from vgdsmc import stage74_radial_speed_shell_opposite_sector_cancellation_audit as stage74


def synthetic_case(seed: int = 74):
    rng = np.random.default_rng(seed)
    ny, nx = 4, 5
    speeds = np.repeat(np.array([0.4, 0.8, 1.2, 1.6]), 8)
    angles = np.tile((np.arange(8) + 0.5) * 2.0 * math.pi / 8, 4)
    vx = speeds * np.cos(angles)
    vy = speeds * np.sin(angles)
    nq = vx.size
    phi = 0.2 + rng.random((ny, nx, nq))
    psi = 0.1 + 0.4 * rng.random((ny, nx, nq))
    weight = np.linspace(0.05, 0.15, nq)
    return phi, psi, vx, vy, weight


def test_stage74_frozen_design_accepts_only_preregistered_values():
    stage74.validate_stage74_design()
    with pytest.raises(ValueError, match="no retuning"):
        stage74.validate_stage74_design(radial_shell_count=5)
    with pytest.raises(ValueError, match="no retuning"):
        stage74.validate_stage74_design(opposite_pair_cancellation_guard=0.55)


def test_radial_shell_indices_make_equal_ordered_speed_groups():
    _, _, vx, vy, _ = synthetic_case()
    shells = stage74.radial_shell_indices(vx, vy)
    assert [int(np.sum(shells == shell)) for shell in range(4)] == [8, 8, 8, 8]
    speed = np.hypot(vx, vy)
    for shell in range(3):
        assert np.max(speed[shells == shell]) <= np.min(speed[shells == shell + 1])


def test_radial_shell_metadata_records_complete_partition():
    _, _, vx, vy, _ = synthetic_case()
    shells = stage74.radial_shell_indices(vx, vy)
    metadata = stage74.radial_shell_metadata(vx, vy, shells)
    assert [row["velocity_point_count"] for row in metadata] == [8, 8, 8, 8]
    assert [row["shell"] for row in metadata] == [0, 1, 2, 3]
    assert metadata[0]["maximum_speed"] < metadata[-1]["minimum_speed"]


def test_radial_shell_groups_sum_to_exact_stage73_x_component():
    phi, psi, vx, vy, weight = synthetic_case()
    shell_groups = stage74.evaluate_radial_shell_angular_qy_maps(
        phi, psi, vx, vy, weight, chunk_size=7
    )
    stage73_groups = stage73.evaluate_velocity_group_x_qy_maps(
        phi, psi, vx, vy, weight, chunk_size=7
    )
    np.testing.assert_allclose(
        np.sum(shell_groups, axis=(0, 1)),
        np.sum(stage73_groups, axis=(0, 1)),
        rtol=0.0,
        atol=6.0e-14,
    )


def test_grouped_closure_reproduces_each_angular_bin():
    rng = np.random.default_rng(3)
    groups = rng.normal(size=(4, 8, 3, 5))
    reference = np.sum(groups, axis=0)
    exact = stage74.grouped_closure(groups, reference)
    assert exact["within_guard"] is True
    assert exact["maximum_absolute_error"] < 1.0e-14
    failed = stage74.grouped_closure(groups, reference + 1.0e-4)
    assert failed["within_guard"] is False


def test_opposite_pair_metrics_detect_exact_cancellation():
    bins = np.zeros((8, 3, 4))
    bins[1] = 2.0
    bins[5] = -2.0
    metrics = stage74.opposite_pair_metrics(bins)
    assert metrics["pairs"]["1_5"]["cancellation_ratio"] == 0.0
    assert metrics["vertical_oblique_cancellation_ratio"] == 0.0
    assert metrics["vertical_oblique_absolute_share"] == 1.0


def test_shell_angular_attribution_forms_complete_shell_shares():
    groups = np.zeros((4, 8, 4, 4))
    groups[0, 1] = 1.0
    groups[1, 5] = -1.0
    groups[2, 2] = 0.5
    result = stage74.shell_angular_attribution(groups)
    assert np.isclose(sum(result["shell_absolute_shares"]), 1.0)
    assert result["dominant_shell"] in (0, 1)
    assert 0.0 <= result["opposite_pair_metrics"]["vertical_oblique_cancellation_ratio"] <= 1.0


def test_stage74_decision_blocks_invalid_endpoints():
    args = dict(
        finite=True,
        provenance_consistent=True,
        grouped_closure_closed=True,
        radial_concentrated=True,
        vertical_pair_cancellation_strong=True,
    )
    assert "nonfinite" in stage74.stage74_decision(**{**args, "finite": False})
    assert "endpoint" in stage74.stage74_decision(
        **{**args, "provenance_consistent": False}
    )
    assert "closure" in stage74.stage74_decision(
        **{**args, "grouped_closure_closed": False}
    )


def test_stage74_decision_routes_concentrated_cancelling_result():
    decision = stage74.stage74_decision(True, True, True, True, True)
    assert "radial_shell_concentration" in decision
    assert "stage75_facewise_shell_pair" in decision


def test_stage74_decision_preserves_diffuse_or_weak_cancellation_result():
    decision = stage74.stage74_decision(True, True, True, False, True)
    assert "diffuse_speed_or_weak" in decision
    assert "stage75_signed_face_location" in decision


def test_stage73_completed_endpoint_is_frozen_exactly():
    endpoint = stage74.STAGE73_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31088628167
    assert endpoint["workflow_job_id"] == 92573989499
    assert endpoint["tests_passed"] == 166
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8966181833
    assert endpoint["decision"].startswith("stage73_balanced_vx_sign")


def test_radial_and_opposite_pair_contracts_are_fixed():
    assert stage74.RADIAL_SHELL_COUNT == 4
    assert stage74.RADIAL_NODES_PER_SHELL == 10
    assert stage74.OPPOSITE_BIN_PAIRS == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert stage74.VERTICAL_OBLIQUE_OPPOSITE_PAIRS == ((1, 5), (2, 6))
    assert stage74.TOP_TWO_SHELL_CONCENTRATION_GUARD == 0.65
    assert stage74.OPPOSITE_PAIR_CANCELLATION_GUARD == 0.50
