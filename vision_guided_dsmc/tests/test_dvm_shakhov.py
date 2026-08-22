import numpy as np

from vgdsmc.dvm_shakhov import (
    ShakhovReferenceConfig,
    _discrete_maxwellian,
    _macroscopic,
    _shakhov_equilibrium,
    _velocity_grid,
)
from vgdsmc.dvm_shakhov_corrected import (
    reconstruct_temperature,
    save_shakhov_reference,
    solve_shakhov_reference,
)


def test_shakhov_equilibrium_preserves_primary_moments():
    cfg = ShakhovReferenceConfig(nx=3, ny=3, nv=8)
    vx, vy, vz, dv = _velocity_grid(cfg)
    shape = (3, 3)
    rho = np.full(shape, 1.1)
    u = np.full(shape, 0.08)
    v = np.full(shape, -0.04)
    w = np.full(shape, 0.02)
    temperature = np.full(shape, 1.05)
    distribution = _discrete_maxwellian(rho, u, v, w, temperature, vx, vy, vz, dv)
    fields = _macroscopic(distribution, vx, vy, vz, dv)
    fields["qx"][:] = 0.015
    fields["qy"][:] = -0.010
    fields["qz"][:] = 0.005
    equilibrium = _shakhov_equilibrium(fields, vx, vy, vz, dv, cfg.prandtl)
    recovered = _macroscopic(equilibrium, vx, vy, vz, dv)
    assert np.allclose(recovered["rho"], fields["rho"], rtol=2.0e-3)
    assert np.allclose(recovered["u"], fields["u"], atol=5.0e-3)
    assert np.allclose(recovered["v"], fields["v"], atol=5.0e-3)
    assert np.allclose(recovered["T"], fields["T"], rtol=1.0e-2)
    assert np.all(equilibrium > 0.0)


def test_temperature_reconstruction_inverts_discrete_maxwellian_response():
    cfg = ShakhovReferenceConfig(nx=3, ny=3, nv=6)
    vx, vy, vz, dv = _velocity_grid(cfg)
    one = np.ones((1, 1))
    zero = np.zeros((1, 1))
    target_kelvin = 300.0
    parameter = np.full((1, 1), target_kelvin / cfg.reference_temperature)
    distribution = _discrete_maxwellian(
        one,
        zero,
        zero,
        zero,
        parameter,
        vx,
        vy,
        vz,
        dv,
    )
    measured_kelvin = (
        _macroscopic(distribution, vx, vy, vz, dv)["T"]
        * cfg.reference_temperature
    )
    reconstructed = reconstruct_temperature(measured_kelvin, cfg)
    assert np.allclose(reconstructed, target_kelvin, atol=0.5)


def test_isothermal_shakhov_reference_remains_near_uniform():
    cfg = ShakhovReferenceConfig(
        nx=6,
        ny=6,
        nv=6,
        t_left=300.0,
        t_right=300.0,
        t_top=300.0,
        t_bottom=300.0,
        max_steps=700,
        tolerance=8.0e-6,
    )
    result = solve_shakhov_reference(cfg)
    assert abs(float(np.mean(result["T"])) - 300.0) < 3.0
    assert np.allclose(result["T_raw_quadrature"], result["T"])
    assert float(np.max(np.hypot(result["u"], result["v"]))) < 0.5
    assert float(np.max(np.abs(result["rho"] - 1.0))) < 1.0e-2


def test_hot_left_shakhov_reference_and_contract(tmp_path):
    cfg = ShakhovReferenceConfig(
        nx=6,
        ny=6,
        nv=6,
        t_left=330.0,
        t_right=270.0,
        max_steps=900,
        tolerance=8.0e-6,
    )
    path = save_shakhov_reference(tmp_path / "shakhov.npz", cfg)
    with np.load(path) as data:
        assert {
            "T",
            "T_raw_quadrature",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
        }.issubset(data.files)
        assert data["T"].shape == (6, 6)
        assert (
            float(np.mean(data["T"][:, 0]))
            > float(np.mean(data["T"][:, -1])) + 8.0
        )
        assert np.isfinite(data["qx"]).all()
        assert float(data["residual_history"][-1]) < 8.0e-6
