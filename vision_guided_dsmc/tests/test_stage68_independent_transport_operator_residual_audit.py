import numpy as np
import pytest

from vgdsmc import stage68_independent_transport_operator_residual_audit as s


def test_completed_endpoint_is_exact():
    endpoint = s.STAGE67_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 30991124477
    assert endpoint["workflow_job_id"] == 92257254811
    assert endpoint["tests_passed"] == 71
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8931272132
    assert endpoint["artifact_sha256"] == (
        "5b871e94bdca02bce33846657f8f4dc263d68af9e671d7eb137ca7c83da198d4"
    )


def test_frozen_design_accepts_exact_configuration():
    s.validate_stage68_design()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": (32, 32)},
        {"kn0": 1.0},
        {"cold_hot_ratio": 0.2},
        {"rule": (32, 96)},
        {"radial_scale": 1.0},
        {"limiter": "vanleer"},
        {"material_heat_flux_ratio": 0.05},
    ],
)
def test_frozen_design_rejects_retuning(kwargs):
    with pytest.raises(ValueError):
        s.validate_stage68_design(**kwargs)


def test_minmod_preserves_only_common_sign_small_magnitude():
    left = np.array([-3.0, -1.0, 0.0, 2.0, 4.0])
    right = np.array([-2.0, 2.0, 1.0, 3.0, -5.0])
    np.testing.assert_allclose(
        s.minmod(left, right), [-2.0, 0.0, 0.0, 2.0, 0.0]
    )


def test_limited_slopes_are_zero_at_boundaries_and_exact_for_linear_interior():
    x = np.arange(5.0)[None, :, None]
    distribution = np.repeat(x, 3, axis=0)
    slope = s.limited_slopes_x(distribution)
    assert np.all(slope[:, 0] == 0.0)
    assert np.all(slope[:, -1] == 0.0)
    np.testing.assert_allclose(slope[:, 1:-1], 1.0)


def test_first_and_second_order_constant_state_are_zero_with_matching_walls():
    distribution = np.ones((4, 5, 2))
    left = np.ones((4, 2))
    right = np.ones((4, 2))
    bottom = np.ones((5, 2))
    top = np.ones((5, 2))
    vx = np.array([1.0, -0.5])
    vy = np.array([-0.25, 0.75])
    args = (distribution, left, right, bottom, top, vx, vy, 0.2, 0.25)
    np.testing.assert_allclose(s.first_order_transport_chunk(*args), 0.0)
    np.testing.assert_allclose(s.second_order_transport_chunk(*args), 0.0)


def test_second_order_flux_is_globally_conservative_for_positive_velocity():
    distribution = np.arange(12.0).reshape(3, 4, 1) + 1.0
    left = np.full((3, 1), 2.5)
    right = np.zeros((3, 1))
    bottom = np.zeros((4, 1))
    top = np.zeros((4, 1))
    vx = np.array([2.0])
    vy = np.array([0.0])
    dx = 0.25
    residual = s.second_order_transport_chunk(
        distribution, left, right, bottom, top, vx, vy, dx, 1.0 / 3.0
    )
    slope = s.limited_slopes_x(distribution)
    left_flux = vx[0] * left[:, 0]
    right_flux = vx[0] * (
        distribution[:, -1, 0] + 0.5 * slope[:, -1, 0]
    )
    expected = -np.sum(right_flux - left_flux) / dx
    assert np.isclose(np.sum(residual), expected)


def test_zero_velocity_gives_zero_transport():
    rng = np.random.default_rng(4)
    distribution = rng.random((3, 4, 1))
    side_walls = np.zeros((3, 1))
    horizontal_walls = np.zeros((4, 1))
    args = (
        distribution,
        side_walls,
        side_walls,
        horizontal_walls,
        horizontal_walls,
        np.array([0.0]),
        np.array([0.0]),
        0.25,
        1.0 / 3.0,
    )
    np.testing.assert_allclose(s.first_order_transport_chunk(*args), 0.0)
    np.testing.assert_allclose(s.second_order_transport_chunk(*args), 0.0)


def test_projected_wall_maxwellian_has_unit_mass_and_psi_relation():
    vx = np.array([-1.0, 0.0, 1.0, 0.0])
    vy = np.array([0.0, 1.0, 0.0, -1.0])
    weight = np.ones(4)
    temperature = np.array([0.1, 1.0])
    phi, psi = s.projected_unit_wall_maxwellian(
        temperature, vx, vy, weight
    )
    np.testing.assert_allclose(np.sum(phi * weight, axis=-1), 1.0)
    np.testing.assert_allclose(psi, temperature[:, None] * phi)


def test_decision_paths_are_guarded_and_do_not_rehabilitate_muscl():
    assert s.stage68_decision(False, True, 1.0) == (
        "stage68_nonfinite_transport_operator_blocker"
    )
    assert s.stage68_decision(True, False, 1.0) == (
        "stage68_retained_transport_reconstruction_blocker"
    )
    assert "material_higher_order_transport_residual" in s.stage68_decision(
        True, True, 0.10
    )
    assert "not_material" in s.stage68_decision(True, True, 0.099)
    assert "muscl" not in s.stage68_decision(True, True, 0.5).lower()
