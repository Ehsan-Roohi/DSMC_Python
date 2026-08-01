from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from vgdsmc import stage32_near_continuum_observable_audit as stage32
from vgdsmc.linear_sidewall_validation import LinearSidewallConfig


def test_stage32_fixed_design_constants() -> None:
    assert stage32.STAGE32_GRIDS == ((12, 12), (18, 18), (24, 24))
    assert stage32.STAGE32_KNUDSEN == 0.1
    assert stage32.STAGE32_RATIO == 0.1
    assert stage32.STAGE32_OBSERVABLES == (
        "boundary_mixture",
        "adjacent_cell_center",
        "linear_extrapolated_wall",
    )


def test_validate_stage32_design_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        stage32.validate_stage32_design(((12, 12),), 100, 1e-5)
    with pytest.raises(ValueError):
        stage32.validate_stage32_design(((12, 12), (10, 10)), 100, 1e-5)
    with pytest.raises(ValueError):
        stage32.validate_stage32_design(((2, 12), (3, 12)), 100, 1e-5)
    with pytest.raises(ValueError):
        stage32.validate_stage32_design(((3, 3), (4, 4)), 0, 1e-5)
    with pytest.raises(ValueError):
        stage32.validate_stage32_design(((3, 3), (4, 4)), 100, 0.0)


def test_wall_observable_profiles_use_fixed_formulas() -> None:
    cfg = LinearSidewallConfig(nx=3, ny=3, kn0=0.1, cold_hot_ratio=0.1)
    velocity = np.array(
        [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0], [3.0, 5.0, 7.0]],
        dtype=np.float64,
    )
    boundary = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    profiles = stage32.wall_observable_profiles(
        {"v": velocity, "left_wall_velocity": boundary}, cfg
    )
    y = (np.arange(3) + 0.5) / 3.0
    assert np.allclose(
        profiles["boundary_mixture"], np.interp(stage32.TABLE3_Y, y, boundary)
    )
    assert np.allclose(
        profiles["adjacent_cell_center"], np.interp(stage32.TABLE3_Y, y, velocity[:, 0])
    )
    extrapolated = 1.5 * velocity[:, 0] - 0.5 * velocity[:, 1]
    assert np.allclose(
        profiles["linear_extrapolated_wall"],
        np.interp(stage32.TABLE3_Y, y, extrapolated),
    )


def test_observable_metrics_and_profile_change_are_finite() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, -2.0, 2.0])
    metrics = stage32.observable_metrics(prediction, reference)
    assert metrics["relative_rms"] > 0.0
    assert metrics["relative_l1"] > 0.0
    assert metrics["sign_agreement"] == pytest.approx(2.0 / 3.0)
    assert stage32.relative_profile_change(reference, reference) == 0.0


def _row(q_error: float, boundary: float, adjacent: float, extrapolated: float):
    return {
        "qav_relative_error": q_error,
        "observables": {
            "boundary_mixture": {"relative_rms": boundary, "sign_agreement": 0.2},
            "adjacent_cell_center": {"relative_rms": adjacent, "sign_agreement": 0.8},
            "linear_extrapolated_wall": {
                "relative_rms": extrapolated,
                "sign_agreement": 1.0,
            },
        },
    }


def test_stage32_decision_can_adopt_extrapolated_observable() -> None:
    rows = [_row(0.3, 3.0, 2.5, 2.0), _row(0.3, 3.0, 2.0, 1.0)]
    changes = {
        "boundary_mixture": 0.2,
        "adjacent_cell_center": 0.08,
        "linear_extrapolated_wall": 0.05,
    }
    assert stage32.stage32_decision(rows, changes) == (
        "adopt_interior_or_extrapolated_wall_observable_and_cross_validate"
    )


def test_stage32_decision_records_unexplained_error() -> None:
    rows = [_row(0.3, 3.0, 2.5, 2.0), _row(0.29, 3.1, 2.6, 2.1)]
    changes = {name: 0.2 for name in stage32.STAGE32_OBSERVABLES}
    assert stage32.stage32_decision(rows, changes) == (
        "observable_and_spatial_resolution_do_not_explain_kn0p1_error_audit_model_limit"
    )


def _fake_result(cfg: LinearSidewallConfig) -> dict[str, object]:
    field = np.ones((cfg.ny, cfg.nx), dtype=np.float64)
    return {
        "T": field,
        "rho": field,
        "u": field,
        "v": field * 0.002,
        "qx": field,
        "qy": field,
        "left_wall_velocity": np.full(cfg.ny, 0.001),
        "table_velocity": np.full(10, 0.001),
        "bottom_heat_flux": np.full(cfg.nx, 0.08),
        "residual_history": np.array([1e-5]),
        "iterations": 100,
        "converged": True,
        "wall_mass_balance_relative_error": 1e-16,
        "minimum_distribution": 1e-30,
    }


def test_run_stage32_writes_reproducible_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        stage32,
        "solve_reduced_case",
        lambda cfg, quadrature: _fake_result(cfg),
    )
    grids = ((3, 3), (4, 4))
    summary = stage32.run_stage32(
        tmp_path, grids=grids, max_steps=100, tolerance=1e-4
    )
    assert summary["stage"] == 32
    assert summary["configuration"]["physical_parameter_retuning"] is False
    assert summary["configuration"]["grids"] == [[3, 3], [4, 4]]
    assert len(summary["rows"]) == 2
    assert summary["all_cases_converged"] is True
    assert summary["decision"] in {
        "adopt_interior_or_extrapolated_wall_observable_and_cross_validate",
        "spatial_refinement_materially_reduces_near_continuum_error",
        "observable_and_spatial_resolution_do_not_explain_kn0p1_error_audit_model_limit",
    }
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "fields_and_profiles.npz").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["configuration"]["kn0"] == 0.1
