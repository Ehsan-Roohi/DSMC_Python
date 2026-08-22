from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.stage51_heat_flux_definition_audit import (
    STAGE50_COMPLETED_ENDPOINT,
    STAGE51_GRID,
    STAGE51_SOURCE_NORMALIZATION,
    normalization_diagnostics,
    polynomial_wall_extrapolation,
    run_stage51,
    source_consistent_estimators,
    stage51_decision,
    validate_stage50_summary,
)


def _summary() -> dict[str, object]:
    cases = []
    for kn0, qav in (
        (1.0, 0.17195346890049845),
        (10.0, 0.22492827563466516),
    ):
        cases.append(
            {
                "kn0": kn0,
                "predicted_qav": qav,
                "converged": True,
                "finite": True,
            }
        )
    return {
        "stage": 50,
        "retained_stage49_endpoint": {"workflow_run_id": 30767671512},
        "configuration": {
            "knudsen_numbers": [1.0, 10.0],
            "grid": [64, 64],
            "polar_rule": {
                "radial_nodes": 32,
                "angular_nodes": 96,
                "point_count": 3072,
            },
            "physical_parameter_retuning": False,
            "velocity_quadrature_retuning": False,
        },
        "new_cases": cases,
        "decision": STAGE50_COMPLETED_ENDPOINT["decision"],
    }


def _write_artifact(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(_summary()), encoding="utf-8"
    )
    for qav, filename in (
        (0.17195346890049845, "kn1_fields_and_profiles.npz"),
        (0.22492827563466516, "kn10_fields_and_profiles.npz"),
    ):
        qy = np.repeat(
            np.linspace(qav * 0.99, qav * 1.15, 64)[:, None], 64, axis=1
        )
        np.savez_compressed(
            root / filename,
            bottom_heat_flux=np.full(64, qav),
            qy=qy,
        )


def test_stage50_provenance_and_source_scaling_are_exact() -> None:
    assert STAGE50_COMPLETED_ENDPOINT["workflow_run_id"] == 30776673982
    assert STAGE50_COMPLETED_ENDPOINT["workflow_job_id"] == 91573366717
    assert STAGE50_COMPLETED_ENDPOINT["tests_passed"] == 63
    assert STAGE50_COMPLETED_ENDPOINT["artifact_id"] == 8843553740
    assert STAGE50_COMPLETED_ENDPOINT["source_head_sha"] == (
        "df37f0a4e464130bc12ca4ae815b643602fc9521"
    )
    assert STAGE51_SOURCE_NORMALIZATION[
        "solver_to_source_heat_flux_multiplier"
    ] == 1.0 / math.sqrt(2.0)


def test_design_validation_rejects_retuning_or_wrong_quadrature() -> None:
    good = _summary()
    validate_stage50_summary(good)
    wrong = json.loads(json.dumps(good))
    wrong["configuration"]["polar_rule"]["radial_nodes"] = 48
    with pytest.raises(ValueError, match="32x96"):
        validate_stage50_summary(wrong)
    retuned = json.loads(json.dumps(good))
    retuned["configuration"]["physical_parameter_retuning"] = True
    with pytest.raises(ValueError, match="no physical"):
        validate_stage50_summary(retuned)


def test_linear_and_quadratic_extrapolation_recover_wall_value() -> None:
    n = 64
    y = (np.arange(n) + 0.5) / n
    linear = 0.2 + 0.3 * y
    quadratic = 0.4 - 0.2 * y + 0.7 * y**2
    assert polynomial_wall_extrapolation(linear, 1) == pytest.approx(
        0.2, abs=1e-14
    )
    assert polynomial_wall_extrapolation(quadratic, 2) == pytest.approx(
        0.4, abs=1e-14
    )


def test_source_consistent_estimators_preserve_constant_flux() -> None:
    qav = 0.173
    estimates = source_consistent_estimators(
        np.full(STAGE51_GRID[0], qav), np.full(STAGE51_GRID, qav)
    )
    assert set(estimates) == {
        "wall_face_total_energy_flux",
        "first_cell_center_peculiar_heat_flux",
        "two_row_linear_wall_extrapolation",
        "three_row_quadratic_wall_extrapolation",
    }
    assert all(value == pytest.approx(qav) for value in estimates.values())


def test_inconsistent_normalization_variants_are_excluded() -> None:
    diagnostics = normalization_diagnostics(0.2)
    assert diagnostics["stored_source_scaled"]["eligible_for_decision"] is True
    assert diagnostics["omitted_sqrt2_conversion"]["eligible_for_decision"] is False
    assert diagnostics["double_applied_sqrt2_conversion"][
        "eligible_for_decision"
    ] is False
    assert diagnostics["omitted_sqrt2_conversion"]["value"] == pytest.approx(
        0.2 * math.sqrt(2.0)
    )


def test_known_stage50_pattern_routes_to_velocity_resolution() -> None:
    audits = {
        "1.0": {
            "source_consistent_estimators": {
                "wall": 0.17195,
                "linear": 0.16959,
            },
            "relative_errors": {"wall": 0.1618, "linear": 0.1459},
            "source_consistent_spread_relative_to_literature": 0.016,
        },
        "10.0": {
            "source_consistent_estimators": {
                "wall": 0.22493,
                "linear": 0.22315,
            },
            "relative_errors": {"wall": 0.2636, "linear": 0.2537},
            "source_consistent_spread_relative_to_literature": 0.010,
        },
    }
    assert stage51_decision(audits) == (
        "heat_flux_definition_and_location_do_not_explain_cross_kn_error_"
        "stage52_velocity_resolution_audit"
    )


def test_decision_requires_same_successful_estimator_across_kn() -> None:
    audits = {
        "1.0": {
            "source_consistent_estimators": {"wall": 0.15, "linear": 0.18},
            "relative_errors": {"wall": 0.01, "linear": 0.20},
            "source_consistent_spread_relative_to_literature": 0.02,
        },
        "10.0": {
            "source_consistent_estimators": {"wall": 0.22, "linear": 0.18},
            "relative_errors": {"wall": 0.20, "linear": 0.01},
            "source_consistent_spread_relative_to_literature": 0.02,
        },
    }
    assert stage51_decision(audits).endswith(
        "stage52_velocity_resolution_audit"
    )


def test_run_stage51_records_negative_endpoint_and_rejects_nonfinite(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    output = tmp_path / "output"
    _write_artifact(artifact)
    result = run_stage51(artifact, output)
    assert result["decision"].endswith("stage52_velocity_resolution_audit")
    assert (output / "summary.json").is_file()
    assert (output / "heat_flux_profiles.npz").is_file()
    with np.load(artifact / "kn1_fields_and_profiles.npz") as data:
        qy = data["qy"].copy()
        wall = data["bottom_heat_flux"].copy()
    qy[0, 0] = np.nan
    np.savez_compressed(
        artifact / "kn1_fields_and_profiles.npz",
        bottom_heat_flux=wall,
        qy=qy,
    )
    with pytest.raises(ValueError, match="finite"):
        run_stage51(artifact, tmp_path / "bad")
