from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from vgdsmc.bottom_heated_benchmark import (
    BottomHeatedBenchmarkConfig,
    bottom_wall_temperature_profile,
    local_relaxation_time,
    paper_kn0_to_solver_relaxation_scale,
    run_literature_validation_matrix,
    solve_bottom_heated_case,
    summarize_literature_case,
)


def test_bottom_temperature_has_five_percent_linear_corner_ramps():
    profile = bottom_wall_temperature_profile(100, 0.5, 1.5, 0.05)
    x = (np.arange(100) + 0.5) / 100
    expected = 0.5 + np.clip(np.minimum(x, 1.0 - x) / 0.05, 0.0, 1.0)
    np.testing.assert_allclose(profile, expected)
    assert profile[0] == pytest.approx(0.6)
    assert profile[4] == pytest.approx(1.4)
    assert profile[5] == pytest.approx(1.5)
    np.testing.assert_allclose(profile, profile[::-1])


def test_paper_kn_mapping_and_hard_sphere_local_relaxation():
    assert paper_kn0_to_solver_relaxation_scale(1.0) == pytest.approx(
        math.sqrt(2.0 / math.pi)
    )
    cfg = BottomHeatedBenchmarkConfig(kn0=1.0, cold_hot_ratio=0.5)
    density = np.array([[1.0, 2.0]])
    temperature = np.array([[1.0, 4.0]])
    tau = local_relaxation_time(density, temperature, cfg)
    expected = math.sqrt(2.0 / math.pi) * temperature ** (-0.5) / density
    np.testing.assert_allclose(tau, expected)


def test_reference_temperature_convention_preserves_requested_ratio():
    cfg = BottomHeatedBenchmarkConfig(cold_hot_ratio=0.1)
    assert cfg.cold_temperature / cfg.hot_temperature == pytest.approx(0.1)
    assert 0.5 * (cfg.cold_temperature + cfg.hot_temperature) == pytest.approx(1.0)


def test_small_bottom_heated_solver_is_finite_and_mass_flux_wall_is_stable():
    cfg = BottomHeatedBenchmarkConfig(
        nx=4,
        ny=4,
        nv=6,
        velocity_extent=5.0,
        kn0=1.0,
        cold_hot_ratio=0.5,
        max_steps=20,
        minimum_steps=20,
        check_interval=10,
        tolerance=1.0e-2,
    )
    result = solve_bottom_heated_case(cfg)
    for name in ("T", "rho", "u", "v", "qx", "qy", "bottom_heat_flux"):
        assert np.all(np.isfinite(result[name]))
    assert np.all(np.asarray(result["T"]) > 0.0)
    assert np.mean(result["rho"]) == pytest.approx(1.0)
    assert np.asarray(result["bottom_heat_flux"]).shape == (4,)


def _fake_solution(cfg: BottomHeatedBenchmarkConfig):
    shape = (cfg.ny, cfg.nx)
    factor = {0.1: 1.0, 0.5: 1.2, 0.9: 0.6}[round(cfg.cold_hot_ratio, 1)]
    q = factor * cfg.kn0 / (1.0 + cfg.kn0)
    return {
        "T": np.ones(shape),
        "rho": np.ones(shape),
        "u": np.zeros(shape),
        "v": np.ones(shape) * 0.01,
        "qx": np.zeros(shape),
        "qy": np.ones(shape) * q,
        "bottom_wall_temperature": np.ones(cfg.nx),
        "bottom_heat_flux": np.ones(cfg.nx) * q,
        "residual_history": np.array([1.0e-5]),
        "iterations": 10,
        "dt": 0.01,
    }


def test_matrix_records_published_structural_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vgdsmc.bottom_heated_benchmark.solve_bottom_heated_case",
        _fake_solution,
    )
    summary = run_literature_validation_matrix(
        tmp_path,
        BottomHeatedBenchmarkConfig(nx=4, ny=4, nv=6),
    )
    assert summary["stage"] == 24
    assert len(summary["rows"]) == 9
    assert summary["structural_success_count"] == 3
    assert all(summary["structural_checks"].values())
    assert summary["ratio_ordering_checks"] == {"kn0_1": True, "kn0_10": True}
    assert Path(tmp_path / "summary.json").exists()
    fields = np.load(tmp_path / "fields_and_profiles.npz")
    assert "bottom_heat_flux_ratio0p5_kn1" in fields.files
    assert "lateral_velocity_ratio0p9_kn10" in fields.files


def test_case_summary_uses_both_lateral_walls():
    cfg = BottomHeatedBenchmarkConfig(nx=4, ny=4, nv=6)
    result = _fake_solution(cfg)
    result["v"][:, 0] = 1.0
    result["v"][:, -1] = -1.0
    row = summarize_literature_case(result, cfg)
    assert row["positive_lateral_velocity_fraction"] == pytest.approx(0.5)
    assert row["lateral_hot_to_cold_majority"] is False
