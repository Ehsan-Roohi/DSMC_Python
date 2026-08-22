from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from vgdsmc.bottom_heated_corrected import (
    CorrectedBottomHeatedConfig,
    discrete_wall_temperature,
    initial_temperature_field,
    run_corrected_feasibility_matrix,
    solve_corrected_bottom_heated_case,
    validate_corrected_config,
)
from vgdsmc.dvm_shakhov import ShakhovReferenceConfig, _velocity_grid


def test_odd_velocity_grid_contains_zero_and_even_grid_does_not():
    odd = ShakhovReferenceConfig(nv=17, velocity_extent=5.0)
    vx, _, _, _ = _velocity_grid(odd)
    assert np.any(vx == 0.0)
    even = ShakhovReferenceConfig(nv=12, velocity_extent=6.0)
    vx_even, _, _, _ = _velocity_grid(even)
    assert not np.any(vx_even == 0.0)
    assert np.min(np.abs(vx_even)) == pytest.approx(0.5)


def test_corrected_reference_convention_and_low_temperature_representation():
    cfg = CorrectedBottomHeatedConfig(nv=17, velocity_extent=5.0, cold_hot_ratio=0.1)
    assert cfg.hot_temperature == 1.0
    assert cfg.cold_temperature == 0.1
    cold_actual = discrete_wall_temperature(cfg.cold_temperature, cfg)
    hot_actual = discrete_wall_temperature(cfg.hot_temperature, cfg)
    assert abs(cold_actual - 0.1) / 0.1 < 0.02
    assert abs(hot_actual - 1.0) < 0.02


def test_initial_temperature_is_vertical_hot_to_cold_warm_start():
    cfg = CorrectedBottomHeatedConfig(nx=5, ny=6, nv=7, cold_hot_ratio=0.2)
    temperature = initial_temperature_field(cfg)
    assert temperature.shape == (6, 5)
    np.testing.assert_allclose(temperature[:, 0], temperature[:, -1])
    assert np.all(np.diff(temperature[:, 0]) < 0.0)
    assert cfg.cold_temperature < temperature[-1, 0] < temperature[0, 0] < 1.0


def test_even_velocity_count_is_rejected():
    with pytest.raises(ValueError):
        validate_corrected_config(CorrectedBottomHeatedConfig(nv=12))


def test_small_corrected_solver_is_finite():
    cfg = CorrectedBottomHeatedConfig(
        nx=4,
        ny=4,
        nv=7,
        velocity_extent=4.0,
        kn0=1.0,
        cold_hot_ratio=0.5,
        max_steps=20,
        minimum_steps=20,
        check_interval=10,
        tolerance=1.0e-2,
    )
    result = solve_corrected_bottom_heated_case(cfg)
    for name in (
        "T", "rho", "u", "v", "qx", "qy", "bottom_heat_flux",
        "residual_history", "component_change_history",
    ):
        assert np.all(np.isfinite(result[name]))
    assert np.all(np.asarray(result["T"]) > 0.0)
    assert float(np.mean(result["rho"])) == pytest.approx(1.0)
    assert np.asarray(result["component_change_history"]).shape[1] == 3


def _fake_corrected_solution(cfg: CorrectedBottomHeatedConfig):
    shape = (cfg.ny, cfg.nx)
    q = (1.0 - cfg.cold_hot_ratio) * cfg.kn0 / (1.0 + cfg.kn0)
    return {
        "T": np.full(shape, 0.5 * (1.0 + cfg.cold_hot_ratio)),
        "rho": np.ones(shape),
        "u": np.zeros(shape),
        "v": np.full(shape, 0.01),
        "qx": np.zeros(shape),
        "qy": np.full(shape, q),
        "bottom_wall_temperature": np.full(cfg.nx, 1.0),
        "bottom_heat_flux": np.full(cfg.nx, q),
        "residual_history": np.array([1.0e-6]),
        "component_change_history": np.array([[1.0e-6, 1.0e-6, 1.0e-6]]),
        "iterations": 1200,
        "dt": 0.01,
        "converged": True,
    }


def test_corrected_matrix_records_five_fixed_conditions_and_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vgdsmc.bottom_heated_corrected.solve_corrected_bottom_heated_case",
        _fake_corrected_solution,
    )
    monkeypatch.setattr(
        "vgdsmc.bottom_heated_corrected.discrete_wall_temperature",
        lambda target, cfg: target,
    )
    summary = run_corrected_feasibility_matrix(
        tmp_path,
        CorrectedBottomHeatedConfig(nx=4, ny=4, nv=7),
    )
    assert summary["stage"] == 25
    assert len(summary["rows"]) == 5
    assert summary["configuration"]["reference_temperature_convention"] == "T0=TH"
    assert summary["success_count"] == summary["check_count"] == 4
    assert all(summary["preregistered_checks"].values())
    assert Path(tmp_path / "summary.json").exists()
    arrays = np.load(tmp_path / "fields_and_profiles.npz")
    assert "T_ratio0p1_kn0p1" in arrays.files
    assert "bottom_heat_flux_ratio0p5_kn10" in arrays.files
