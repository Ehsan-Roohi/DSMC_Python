from __future__ import annotations

import math

import numpy as np
import pytest

from vgdsmc import stage72_directional_transport_component_audit as stage72
from vgdsmc import stage73_velocity_sign_angular_bin_interior_face_flux_attribution_audit as stage73


def synthetic_case(seed: int = 73):
    rng = np.random.default_rng(seed)
    ny, nx, nq = 5, 6, 8
    phi = 0.2 + rng.random((ny, nx, nq))
    psi = 0.1 + 0.4 * rng.random((ny, nx, nq))
    angles = (np.arange(nq) + 0.5) * 2.0 * math.pi / nq
    speed = np.linspace(0.5, 1.2, nq)
    vx = speed * np.cos(angles)
    vy = speed * np.sin(angles)
    weight = np.linspace(0.1, 0.2, nq)
    incoming = (
        0.2 + rng.random((ny, nq)),
        0.1 + rng.random((ny, nq)),
        0.2 + rng.random((ny, nq)),
        0.1 + rng.random((ny, nq)),
        0.2 + rng.random((nx, nq)),
        0.1 + rng.random((nx, nq)),
        0.2 + rng.random((nx, nq)),
        0.1 + rng.random((nx, nq)),
    )
    return phi, psi, vx, vy, weight, incoming


def test_stage73_frozen_design_accepts_only_preregistered_values():
    stage73.validate_stage73_design()
    with pytest.raises(ValueError, match="no retuning"):
        stage73.validate_stage73_design(angular_bin_count=12)
    with pytest.raises(ValueError, match="no retuning"):
        stage73.validate_stage73_design(sign_balance_tolerance=0.06)


def test_angular_bins_cover_fixed_eight_midpoint_sectors():
    angles = (np.arange(8) + 0.5) * math.pi / 4.0
    bins = stage73.angular_bin_indices(np.cos(angles), np.sin(angles))
    np.testing.assert_array_equal(bins, np.arange(8))


def test_velocity_sign_indices_match_transport_upwind_sign():
    signs = stage73.velocity_sign_indices(np.array([-2.0, -0.1, 0.2, 3.0]))
    np.testing.assert_array_equal(signs, np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="nonzero-vx"):
        stage73.velocity_sign_indices(np.array([-1.0, 0.0, 1.0]))


def test_interior_face_difference_uses_upwind_limited_slope():
    distribution = np.array([[[0.0], [1.0], [2.0], [3.0]]])
    positive = stage73.interior_x_face_flux_difference_chunk(
        distribution, np.array([2.0])
    )
    negative = stage73.interior_x_face_flux_difference_chunk(
        distribution, np.array([-2.0])
    )
    np.testing.assert_allclose(positive[0, :, 0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(negative[0, :, 0], [1.0, 1.0, 0.0])


def test_velocity_groups_sum_to_exact_stage72_x_component():
    phi, psi, vx, vy, weight, incoming = synthetic_case()
    groups = stage73.evaluate_velocity_group_x_qy_maps(
        phi, psi, vx, vy, weight, chunk_size=3
    )
    directional = stage72.evaluate_directional_qy_components(
        phi, psi, vx, vy, weight, incoming, chunk_size=3
    )
    np.testing.assert_allclose(
        np.sum(groups, axis=(0, 1)),
        directional["difference_x_qy"],
        rtol=0.0,
        atol=3.0e-14,
    )


def test_grouped_closure_reports_exact_and_failed_maps():
    groups = np.zeros((8, 2, 3, 4))
    groups[1, 0] = 2.0
    groups[6, 1] = -0.5
    reference = np.sum(groups, axis=(0, 1))
    exact = stage73.grouped_closure(groups, reference)
    assert exact["within_guard"] is True
    assert exact["maximum_absolute_error"] == 0.0
    failed = stage73.grouped_closure(groups, reference + 1.0e-4)
    assert failed["within_guard"] is False


def test_velocity_group_attribution_forms_complete_shares():
    groups = np.zeros((8, 2, 16, 16))
    groups[1, 0, :, 0] = 3.0
    groups[2, 1, :, -1] = -2.0
    groups[5, 0, 4:12, 4:12] = 1.0
    result = stage73.velocity_group_attribution(groups, 16)
    assert np.isclose(sum(result["angular_bin_absolute_shares"]), 1.0)
    assert np.isclose(sum(result["sign_absolute_shares"].values()), 1.0)
    assert result["vertical_oblique_absolute_share"] == 1.0
    assert result["ranked_angular_bins"][0] == 5


def test_stage73_decision_blocks_invalid_endpoints():
    args = dict(
        finite=True,
        provenance_consistent=True,
        grouped_closure_closed=True,
        sign_balanced=True,
        fine_vertical_oblique_concentrated=True,
        coarse_vertical_oblique_supported=True,
    )
    assert "nonfinite" in stage73.stage73_decision(**{**args, "finite": False})
    assert "endpoint" in stage73.stage73_decision(
        **{**args, "provenance_consistent": False}
    )
    assert "closure" in stage73.stage73_decision(
        **{**args, "grouped_closure_closed": False}
    )


def test_stage73_decision_routes_concentrated_balanced_endpoint():
    decision = stage73.stage73_decision(True, True, True, True, True, True)
    assert "balanced_vx_sign_vertical_oblique_sector_concentration" in decision
    assert "stage74_radial_speed_shell" in decision


def test_stage73_decision_preserves_diffuse_or_grid_sensitive_result():
    decision = stage73.stage73_decision(True, True, True, True, False, True)
    assert "diffuse_or_grid_sensitive" in decision
    assert "face_location_and_group_cancellation" in decision


def test_stage72_completed_endpoint_is_frozen_exactly():
    endpoint = stage73.STAGE72_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31074807690
    assert endpoint["workflow_job_id"] == 92530390942
    assert endpoint["tests_passed"] == 154
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8958793397
    assert endpoint["decision"].startswith("stage72_x_direction")


def test_vertical_oblique_bins_are_fixed_physical_sectors():
    assert stage73.VERTICAL_OBLIQUE_BINS == (1, 2, 5, 6)
    assert stage73.FINE_VERTICAL_OBLIQUE_CONCENTRATION == 0.70
    assert stage73.COARSE_VERTICAL_OBLIQUE_FLOOR == 0.65
    assert stage73.SIGN_BALANCE_TOLERANCE == 0.05
