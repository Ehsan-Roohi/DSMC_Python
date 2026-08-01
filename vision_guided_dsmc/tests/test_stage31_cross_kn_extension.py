from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from vgdsmc import stage31_cross_kn_extension as stage31


def test_stage31_fixed_sequences() -> None:
    assert stage31.STAGE31_KNUDSEN_SEQUENCE == (0.1, 10.0)
    assert stage31.STAGE31_QUADRATURE_NAMES == (
        "cartesian_midpoint_nv19",
        "spherical_matched_r16_mu12_phi24",
    )


def test_validate_stage31_design_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        stage31.validate_stage31_design(2, 12, 100, 1e-5)
    with pytest.raises(ValueError):
        stage31.validate_stage31_design(12, 12, 0, 1e-5)
    with pytest.raises(ValueError):
        stage31.validate_stage31_design(12, 12, 100, 0.0)


def test_material_improvement_requires_both_errors_and_nonworse_sign() -> None:
    cartesian = {
        "qav_relative_error": 0.4,
        "wall_velocity_relative_rms": 4.0,
        "wall_velocity_sign_agreement": 0.2,
    }
    spherical = {
        "qav_relative_error": 0.3,
        "wall_velocity_relative_rms": 3.0,
        "wall_velocity_sign_agreement": 0.3,
    }
    assert stage31._materially_better(spherical, cartesian)
    spherical["wall_velocity_relative_rms"] = 3.8
    assert not stage31._materially_better(spherical, cartesian)


def _fake_result() -> dict[str, object]:
    field = np.ones((3, 3), dtype=np.float64)
    profile = np.ones(10, dtype=np.float64)
    return {
        "T": field,
        "rho": field,
        "u": field,
        "v": field,
        "qx": field,
        "qy": field,
        "left_wall_velocity": np.ones(3),
        "table_velocity": profile,
        "bottom_heat_flux": np.ones(3),
        "residual_history": np.array([1e-5]),
        "iterations": 100,
        "converged": True,
        "wall_mass_balance_relative_error": 1e-16,
        "minimum_distribution": 1e-30,
    }


def test_run_stage31_writes_reproducible_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stage31, "solve_reduced_case", lambda cfg, quadrature: _fake_result())

    def fake_metrics(result, cfg, quadrature):
        spherical = quadrature.family == "spherical_product"
        return {
            "scheme": quadrature.name,
            "family": quadrature.family,
            "point_count": quadrature.point_count,
            "iterations": 100,
            "converged": True,
            "final_change": 1e-5,
            "predicted_qav": 0.1,
            "literature_qav": 0.1,
            "qav_relative_error": 0.1 if spherical else 0.4,
            "wall_velocity_relative_rms": 0.2 if spherical else 4.0,
            "wall_velocity_sign_agreement": 1.0 if spherical else 0.0,
            "wall_mass_balance_relative_error": 1e-16,
            "minimum_distribution": 1e-30,
            "minimum_temperature": 0.1,
            "maximum_temperature": 1.0,
            "work_proxy": 100,
        }

    monkeypatch.setattr(stage31, "_case_metrics", fake_metrics)
    summary = stage31.run_stage31(tmp_path, nx=3, ny=3, max_steps=100, tolerance=1e-4)
    assert summary["stage"] == 31
    assert len(summary["rows"]) == 4
    assert len(summary["comparisons"]) == 2
    assert summary["all_cases_converged"] is True
    assert summary["spherical_materially_better_at_both_new_knudsen_numbers"] is True
    assert summary["decision"] == "spherical_quadrature_supported_across_kn0_0p1_1_10"
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "fields_and_profiles.npz").exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["configuration"]["physical_parameter_retuning"] is False


def test_run_stage31_records_negative_extension(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(stage31, "solve_reduced_case", lambda cfg, quadrature: _fake_result())

    def fake_metrics(result, cfg, quadrature):
        return {
            "scheme": quadrature.name,
            "family": quadrature.family,
            "point_count": quadrature.point_count,
            "iterations": 100,
            "converged": cfg.kn0 != 10.0,
            "final_change": 1e-4,
            "predicted_qav": 0.2,
            "literature_qav": 0.1,
            "qav_relative_error": 0.5,
            "wall_velocity_relative_rms": 2.0,
            "wall_velocity_sign_agreement": 0.0,
            "wall_mass_balance_relative_error": 1e-16,
            "minimum_distribution": 1e-30,
            "minimum_temperature": 0.1,
            "maximum_temperature": 1.0,
            "work_proxy": 100,
        }

    monkeypatch.setattr(stage31, "_case_metrics", fake_metrics)
    summary = stage31.run_stage31(tmp_path, nx=3, ny=3, max_steps=100, tolerance=1e-4)
    assert summary["all_cases_converged"] is False
    assert summary["spherical_materially_better_at_both_new_knudsen_numbers"] is False
    assert summary["decision"] == (
        "record_partial_or_negative_cross_kn_extension_and_audit_remaining_model_error"
    )
