from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from vgdsmc.combined_dvm_reference import (
    CombinedDVMReferenceConfig,
    extract_comparison_profiles,
    run_combined_dvm_reference_study,
)


def _fake_result(cfg):
    y, x = np.meshgrid(
        (np.arange(cfg.ny) + 0.5) / cfg.ny,
        (np.arange(cfg.nx) + 0.5) / cfg.nx,
        indexing="ij",
    )
    correction = 1.0 / cfg.nx**2 + 1.0 / cfg.nv**3
    return {
        "T": 300.0 + 20.0 * (0.5 - x) + correction,
        "rho": 1.0 + 0.02 * (x - 0.5) + correction * 0.01,
        "u": 2.0 * np.sin(np.pi * x) * np.sin(np.pi * y) + correction,
        "v": -1.5 * np.sin(np.pi * x) * np.sin(np.pi * y) - correction,
        "qx": np.full_like(x, 7.0e5 * (1.0 + correction)),
        "qy": 1.0e4 * (y - 0.5) * (1.0 + correction),
        "residual_history": np.array([1.0e-4, 1.0e-6]),
        "iterations": 100,
        "dt": 1.0e-3,
    }


def test_extract_profiles_uses_even_grid_midline_average_and_outward_wall_signs():
    base = np.arange(16, dtype=float).reshape(4, 4)
    fields = {
        "T": base,
        "rho": base + 1.0,
        "u": base + 2.0,
        "v": base + 3.0,
        "qx": np.full((4, 4), 5.0),
        "qy": np.full((4, 4), 7.0),
    }
    profiles = extract_comparison_profiles(fields)
    np.testing.assert_allclose(profiles["T_horizontal_centerline"], 0.5 * (base[1] + base[2]))
    np.testing.assert_allclose(profiles["T_vertical_centerline"], 0.5 * (base[:, 1] + base[:, 2]))
    np.testing.assert_allclose(profiles["normal_heat_flux_left"], 5.0)
    np.testing.assert_allclose(profiles["normal_heat_flux_right"], -5.0)
    np.testing.assert_allclose(profiles["normal_heat_flux_bottom"], 7.0)
    np.testing.assert_allclose(profiles["normal_heat_flux_top"], -7.0)


def test_profile_contract_rejects_missing_or_misaligned_fields():
    complete = {name: np.ones((4, 4)) for name in ("T", "rho", "u", "v", "qx", "qy")}
    incomplete = dict(complete)
    incomplete.pop("qy")
    with pytest.raises(ValueError):
        extract_comparison_profiles(incomplete)
    bad = dict(complete)
    bad["u"] = np.ones((3, 4))
    with pytest.raises(ValueError):
        extract_comparison_profiles(bad)


def test_combined_study_builds_full_matrix_and_external_comparison_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vgdsmc.combined_dvm_reference.solve_shakhov_reference",
        _fake_result,
    )
    cfg = CombinedDVMReferenceConfig(
        spatial_levels=(4, 6, 8),
        velocity_levels=(6, 8),
        max_steps=100,
    )
    summary = run_combined_dvm_reference_study(tmp_path, cfg)
    assert summary["stage"] == 23
    assert summary["run_count"] == 6
    assert summary["canonical_reference"] == {
        "grid": [8, 8],
        "nv": 8,
        "status": "finest_internal_reference_not_external_validation",
    }
    canonical_key = "nx8_ny8_nv8"
    assert summary["errors_relative_to_canonical"][canonical_key]["composite_error"] == pytest.approx(0.0)
    assert set(summary["spatial_sequences"]) == {"Nv_6", "Nv_8"}
    assert set(summary["velocity_increments"]) == {"grid_4", "grid_6", "grid_8"}
    assert Path(tmp_path / "summary.json").exists()
    assert Path(tmp_path / "fields.npz").exists()
    assert Path(tmp_path / "profiles.npz").exists()
    profiles = np.load(tmp_path / "profiles.npz")
    assert "T_horizontal_centerline_nx8_ny8_nv8" in profiles.files
    assert "normal_heat_flux_left_nx8_ny8_nv8" in profiles.files
    persisted = json.loads((tmp_path / "summary.json").read_text())
    assert persisted["external_validation_contract"]["artifact"] == "profiles.npz"


def test_combined_study_requires_ordered_unique_levels(tmp_path):
    with pytest.raises(ValueError):
        run_combined_dvm_reference_study(
            tmp_path,
            CombinedDVMReferenceConfig(spatial_levels=(10, 8, 14)),
        )
    with pytest.raises(ValueError):
        run_combined_dvm_reference_study(
            tmp_path,
            CombinedDVMReferenceConfig(velocity_levels=(10, 10)),
        )
