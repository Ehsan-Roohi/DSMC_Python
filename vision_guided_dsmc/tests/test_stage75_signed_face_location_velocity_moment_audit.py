import numpy as np
import pytest

from vgdsmc import stage75_signed_face_location_velocity_moment_audit as stage75


def test_stage75_frozen_design_accepts_exact_contract():
    stage75.validate_stage75_design()


def test_stage75_frozen_design_rejects_retuning():
    with pytest.raises(ValueError):
        stage75.validate_stage75_design(kn0=1.0)


def test_minmod_preserves_only_common_sign():
    left = np.array([2.0, -3.0, 2.0, 0.0])
    right = np.array([1.0, -4.0, -1.0, 5.0])
    assert np.allclose(stage75.minmod(left, right), [1.0, -3.0, 0.0, 0.0])


def test_limited_slopes_have_zero_boundary_values():
    values = np.arange(15.0).reshape(1, 5, 3)
    slopes = stage75.limited_slopes_x(values)
    assert np.all(slopes[:, 0] == 0.0)
    assert np.all(slopes[:, -1] == 0.0)
    assert np.all(slopes[:, 1:-1] == 3.0)


def test_moment_components_sum_to_direct_heat_flux_moment():
    rng = np.random.default_rng(4)
    dp = rng.normal(size=(2, 3, 5))
    dq = rng.normal(size=(2, 3, 5))
    vx = np.linspace(-2.0, 2.0, 5)
    vy = np.linspace(1.5, -1.5, 5)
    w = np.linspace(0.1, 0.5, 5)
    u = rng.normal(size=(2, 3))
    v = rng.normal(size=(2, 3))
    pieces = stage75.moment_components(dp, dq, vx, vy, w, u, v, -1.0, 0.25)
    cx = vx[None, None, :] - u[..., None]
    cy = vy[None, None, :] - v[..., None]
    direct = 0.5 * np.sum(cy * ((cx * cx + cy * cy) * (-dp / 0.25) + (-dq / 0.25)) * w, axis=-1)
    assert np.allclose(np.sum(pieces, axis=0), direct)


def test_fixed_spatial_masks_partition_faces_and_rows():
    x_masks, y_masks = stage75.fixed_spatial_masks()
    assert np.all(np.sum(np.stack(x_masks), axis=0) == 1)
    assert np.all(np.sum(np.stack(y_masks), axis=0) == 1)


def test_face_location_metrics_report_exact_symmetric_side_balance():
    left = np.ones((3, 4, 3))
    right = np.ones_like(left)
    cell = np.zeros((3, 4, 4))
    cell[:, :, :-1] += left
    cell[:, :, 1:] += right
    metrics = stage75.face_location_moment_metrics(cell, left, right)
    assert metrics["left_right_signed_balance_error"] == 0.0
    assert abs(sum(metrics["moment_absolute_shares"]) - 1.0) < 1e-15


def test_stage75_decision_nonfinite_blocker():
    assert stage75.stage75_decision(False, True, True, 0.0, 0.0).endswith("blocker")


def test_stage75_decision_provenance_blocker():
    assert stage75.stage75_decision(True, False, True, 0.0, 0.0).endswith("blocker")


def test_stage75_decision_closure_blocker():
    assert stage75.stage75_decision(True, True, False, 0.0, 0.0).endswith("blocker")


def test_stage75_decision_routes_conservative_face_pair_result():
    decision = stage75.stage75_decision(True, True, True, 0.01, 1e-14)
    assert "local_velocity_frame_jump_audit" in decision


def test_stage75_decision_routes_asymmetric_face_pair_result():
    decision = stage75.stage75_decision(True, True, True, 0.20, 0.1)
    assert "signed_boundary_cell_residual_audit" in decision
