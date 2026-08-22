from __future__ import annotations

import math

import pytest

from vgdsmc.stage59_independent_reference_audit import (
    MAXIMUM_ACCEPTABLE_PAIRED_OBSERVABLE_DEGRADATION,
    PUBLISHED_REFERENCE,
    STAGE58_COMPLETED_ENDPOINT,
    _reference_metrics,
    evaluate_stage58_summary,
)


def _stage58_fixture() -> dict[str, object]:
    return {
        "stage": 58,
        "decision": STAGE58_COMPLETED_ENDPOINT["decision"],
        "configuration": {
            "kn0": 10.0,
            "cold_hot_ratio": 0.1,
            "grid": [64, 64],
            "radial_nodes": 40,
            "angular_nodes": 96,
        },
        "baseline_clipped": {
            "predicted_qav": 0.2249270610451477,
            "velocity_metrics": {"relative_rms": 0.42475963866887706, "sign_agreement": 1.0},
        },
        "conservative_projection": {
            "predicted_qav": 0.2259322267650623,
            "velocity_metrics": {"relative_rms": 0.4894979086206495, "sign_agreement": 1.0},
        },
        "paired_comparison": {
            "velocity_rms_error_change_fraction": 0.15241153833412927,
        },
    }


def test_published_reference_preserves_both_shakhov_and_dsmc_values() -> None:
    refs = PUBLISHED_REFERENCE["table6_qav"]
    assert refs == {"shakhov": 0.178, "dsmc": 0.179}
    assert PUBLISHED_REFERENCE["doi"] == "10.1063/1.4875235"
    assert PUBLISHED_REFERENCE["table3_independent_dsmc_profile_available"] is False


def test_reference_metrics_use_the_closer_reference_without_hiding_either_error() -> None:
    row = _reference_metrics(0.2249270610451477)
    assert row["errors"]["shakhov"]["relative"] == pytest.approx(0.2636351744109422)
    assert row["errors"]["dsmc"]["relative"] == pytest.approx(
        abs(0.2249270610451477 - 0.179) / 0.179
    )
    assert row["minimum_relative_error"] == row["errors"]["dsmc"]["relative"]
    assert row["reference_spread"] == pytest.approx(0.001)
    assert row["gap_to_reference_spread"] > 45.0


def test_stage59_preserves_positive_and_negative_findings_and_rejects_adoption() -> None:
    summary = evaluate_stage58_summary(_stage58_fixture())
    assert summary["stage"] == 59
    assert summary["configuration"]["solver_rerun"] is False
    for key, value in summary["configuration"].items():
        if key.endswith("_retuning"):
            assert value is False
    assert summary["configuration"]["cross_knudsen_extension_permitted"] is False
    assert summary["paired_review"]["independent_dsmc_heat_flux_confirms_large_discrepancy"] is True
    assert summary["paired_review"]["velocity_guard_failed"] is True
    assert summary["paired_review"]["projection_not_adopted"] is True
    assert summary["decision"] == (
        "stage59_independent_dsmc_heat_flux_confirms_discrepancy_"
        "projection_not_adopted_transport_wall_audit_next"
    )
    assert len(summary["positive_findings"]) >= 3
    assert len(summary["negative_findings"]) >= 3
    assert "does not validate" in summary["interpretation_guard"]


def test_velocity_guard_is_the_preregistered_ten_percent_limit() -> None:
    assert math.isclose(MAXIMUM_ACCEPTABLE_PAIRED_OBSERVABLE_DEGRADATION, 0.10)
    fixture = _stage58_fixture()
    fixture["paired_comparison"]["velocity_rms_error_change_fraction"] = 0.099
    summary = evaluate_stage58_summary(fixture)
    assert summary["paired_review"]["velocity_guard_failed"] is False
    assert summary["paired_review"]["projection_not_adopted"] is True


def test_stage59_rejects_nonfrozen_resolution() -> None:
    fixture = _stage58_fixture()
    fixture["configuration"]["grid"] = [32, 32]
    with pytest.raises(ValueError, match="exact completed Stage 58 resolution"):
        evaluate_stage58_summary(fixture)
