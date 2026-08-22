import numpy as np
import pytest

from vgdsmc import stage76_local_velocity_frame_jump_audit as stage76


def test_stage76_frozen_design_accepts_exact_contract():
    stage76.validate_stage76_design()


def test_stage76_frozen_design_rejects_kn_retuning():
    with pytest.raises(ValueError):
        stage76.validate_stage76_design(kn0=1.0)


def test_stage76_frozen_design_rejects_materiality_retuning():
    with pytest.raises(ValueError):
        stage76.validate_stage76_design(material_frame_ratio_guard=0.1)


def test_minmod_preserves_only_common_sign():
    a = np.array([2.0, -3.0, 2.0, 0.0])
    b = np.array([1.0, -4.0, -1.0, 5.0])
    assert np.allclose(stage76.minmod(a, b), [1.0, -3.0, 0.0, 0.0])


def test_limited_slopes_zero_boundaries():
    values = np.arange(15.0).reshape(1, 5, 3)
    slopes = stage76.limited_slopes_x(values)
    assert np.all(slopes[:, 0] == 0.0)
    assert np.all(slopes[:, -1] == 0.0)
    assert np.all(slopes[:, 1:-1] == 3.0)


def test_frame_kernel_jump_expansion_matches_direct_polynomials():
    vx = np.array([-1.2, 0.7])
    vy = np.array([0.3, 1.1])
    ul = np.array([[0.05]])
    ur = np.array([[0.09]])
    vl = np.array([[-0.02]])
    vr = np.array([[0.03]])
    su, sv, snl, tv, tnl, iv = stage76.frame_kernel_jump_terms(vx, vy, ul, ur, vl, vr)
    cx_l = vx[None, None, :] - ul[..., None]
    cx_r = vx[None, None, :] - ur[..., None]
    cy_l = vy[None, None, :] - vl[..., None]
    cy_r = vy[None, None, :] - vr[..., None]
    assert np.allclose(su + sv + snl, 0.5 * (cy_r * cx_r**2 - cy_l * cx_l**2))
    assert np.allclose(tv + tnl, 0.5 * (cy_r**3 - cy_l**3))
    assert np.allclose(iv, 0.5 * (cy_r - cy_l))


def test_frame_kernel_jump_zero_when_frames_equal():
    vx = np.array([-1.0, 1.0])
    vy = np.array([0.5, -0.5])
    u = np.array([[0.2]])
    v = np.array([[0.1]])
    terms = stage76.frame_kernel_jump_terms(vx, vy, u, u, v, v)
    assert all(np.allclose(term, 0.0) for term in terms)


def test_abs_shares_sum_to_one():
    arrays = np.ones((3, 2, 4))
    shares = stage76._abs_shares(arrays)
    assert np.allclose(shares, [1 / 3, 1 / 3, 1 / 3])


def test_stage76_decision_nonfinite_blocker():
    assert stage76.stage76_decision(False, True, True, True, 0.0, 0.0).endswith("blocker")


def test_stage76_decision_provenance_blocker():
    assert stage76.stage76_decision(True, False, True, True, 0.0, 0.0).endswith("blocker")


def test_stage76_decision_frame_closure_blocker():
    assert stage76.stage76_decision(True, True, False, True, 0.0, 0.0).endswith("blocker")


def test_stage76_decision_group_closure_blocker():
    assert stage76.stage76_decision(True, True, True, False, 0.0, 0.0).endswith("blocker")


def test_stage76_decision_routes_negligible_frame_jump_to_common_frame_divergence():
    decision = stage76.stage76_decision(True, True, True, True, 1e-4, 1e-4)
    assert "common_frame_face_flux_divergence_audit" in decision


def test_stage76_decision_routes_material_frame_jump_to_gradient_localization():
    decision = stage76.stage76_decision(True, True, True, True, 0.02, 0.02)
    assert "frame_gradient_localization_audit" in decision
