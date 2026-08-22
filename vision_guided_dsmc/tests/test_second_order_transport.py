import numpy as np

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig
from vgdsmc.second_order_transport import (
    limited_slopes,
    minmod3,
    muscl_flux_divergence,
    positivity_blend,
    solve_second_order_case,
)


def test_minmod3_keeps_common_sign_and_rejects_extrema():
    a = np.array([2.0, -2.0, 1.0, 0.0])
    b = np.array([1.0, -1.0, -1.0, 2.0])
    c = np.array([3.0, -3.0, 2.0, 1.0])
    np.testing.assert_allclose(minmod3(a, b, c), [1.0, -1.0, 0.0, 0.0])


def test_limited_slopes_reproduce_linear_interior():
    x = np.arange(7.0)
    field = x[None, :, None, None, None]
    slopes = limited_slopes(field, axis=1, theta=1.5)
    np.testing.assert_allclose(slopes[0, 1:-1, 0, 0, 0], 1.0)
    assert slopes[0, 0, 0, 0, 0] == 0.0
    assert slopes[0, -1, 0, 0, 0] == 0.0


def test_positivity_blend_preserves_cell_and_floor():
    old = np.ones((2, 2, 2, 2, 2))
    candidate = old.copy()
    candidate[0, 0, 0, 0, 0] = -1.0
    candidate[0, 0, 1, 1, 1] = 2.0
    limited = positivity_blend(old, candidate, 0.1)
    assert float(limited.min()) >= 0.1
    assert limited[1, 1, 0, 0, 0] == 1.0
    assert limited[0, 0, 1, 1, 1] < 2.0


def test_constant_state_has_zero_muscl_divergence():
    f = np.ones((3, 4, 3, 3, 3))
    values = np.array([-1.0, 0.0, 1.0])
    vx, vy, _ = np.meshgrid(values, values, values, indexing="ij")
    left = np.ones((3, 3, 3, 3))
    right = np.ones((3, 3, 3, 3))
    bottom = np.ones((4, 3, 3, 3))
    top = np.ones((4, 3, 3, 3))
    divergence = muscl_flux_divergence(
        f, left, right, bottom, top, vx, vy, 0.25, 1.0 / 3.0
    )
    np.testing.assert_allclose(divergence, 0.0, atol=1.0e-14)


def test_tiny_second_order_case_is_finite_and_positive():
    cfg = LinearSidewallConfig(
        nx=4,
        ny=4,
        nv=7,
        velocity_extent=4.0,
        kn0=1.0,
        cold_hot_ratio=0.5,
        max_steps=8,
        check_interval=2,
        minimum_steps=8,
        tolerance=1.0e-12,
        cfl=0.15,
    )
    result = solve_second_order_case(cfg)
    for name in ("T", "rho", "u", "v", "qx", "qy", "bottom_heat_flux"):
        assert np.isfinite(result[name]).all()
    assert float(np.min(result["T"])) > 0.0
    assert np.asarray(result["residual_history"]).size == 4
