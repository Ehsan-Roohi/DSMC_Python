from __future__ import annotations

import json
import math

import pytest

from vgdsmc.stage40_external_benchmark_fidelity_audit import (
    PUBLISHED_DETERMINISTIC_METHOD,
    SOURCE,
    SOURCE_TABLE3_SHAKHOV_RATIO_0P1,
    SOURCE_TABLE3_Y,
    SOURCE_TABLE6_DSMC_RATIO_0P1,
    SOURCE_TABLE6_SHAKHOV_RATIO_0P1,
    STAGE39_COMPLETED_ENDPOINT,
    method_fidelity_gap,
    relative_error,
    run_stage40,
    source_table6_model_spread,
    source_transcription_audit,
    stage40_decision,
)


def test_source_identity_is_fixed() -> None:
    assert SOURCE["title"] == (
        "Nonequilibrium Gas Flow and Heat Transfer in a Heated Square Microcavity"
    )
    assert SOURCE["doi"] == "10.1080/01457632.2015.1111079"
    assert SOURCE["year"] == 2016


def test_source_table3_values_are_frozen() -> None:
    assert SOURCE_TABLE3_Y == [
        0.05, 0.15, 0.25, 0.35, 0.45,
        0.55, 0.65, 0.75, 0.85, 0.95,
    ]
    assert SOURCE_TABLE3_SHAKHOV_RATIO_0P1[0.1] == [
        1.7e-3, 8.8e-4, -1.5e-4, -1.1e-3, -1.8e-3,
        -2.5e-3, -2.8e-3, -2.8e-3, -2.0e-3, -5.9e-5,
    ]
    assert set(SOURCE_TABLE3_SHAKHOV_RATIO_0P1) == {0.1, 1.0, 10.0}


def test_source_table6_values_are_frozen() -> None:
    assert SOURCE_TABLE6_SHAKHOV_RATIO_0P1 == {
        0.01: 1.33e-2,
        0.1: 7.20e-2,
        1.0: 1.48e-1,
        10.0: 1.78e-1,
    }
    assert SOURCE_TABLE6_DSMC_RATIO_0P1 == {
        0.01: 1.38e-2,
        0.1: 7.16e-2,
        1.0: 1.49e-1,
        10.0: 1.79e-1,
    }


def test_repository_source_transcription_is_exact() -> None:
    audit = source_transcription_audit()
    assert audit["table3_y_exact"] is True
    assert all(audit["table3_shakhov_rows_exact"].values())
    assert all(audit["table6_shakhov_rows_exact"].values())
    assert audit["all_exact"] is True


def test_stage39_provenance_is_exact() -> None:
    assert STAGE39_COMPLETED_ENDPOINT["workflow_run_id"] == 30718896816
    assert STAGE39_COMPLETED_ENDPOINT["workflow_job_id"] == 91419210631
    assert STAGE39_COMPLETED_ENDPOINT["tests_passed"] == 64
    assert STAGE39_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE39_COMPLETED_ENDPOINT["artifact_id"] == 8825367919
    assert STAGE39_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "c2d44bb727009bb6f7d15255606fad2270e0d80bec5f1ce60643a95a01f43656"
    )
    assert STAGE39_COMPLETED_ENDPOINT["configuration"][
        "physical_parameter_retuning"
    ] is False


def test_relative_error_contract() -> None:
    assert relative_error(0.0726173364328985, 0.072) == pytest.approx(
        0.008574117123590345
    )
    with pytest.raises(ValueError):
        relative_error(1.0, 0.0)
    with pytest.raises(ValueError):
        relative_error(math.inf, 1.0)
    with pytest.raises(ValueError):
        relative_error(1.0, math.nan)


def test_source_table6_model_spread_is_reported() -> None:
    spread = source_table6_model_spread()
    assert tuple(spread) == ("0.01", "0.1", "1.0", "10.0")
    row = spread["0.1"]
    assert row["shakhov"] == 0.072
    assert row["dsmc"] == 0.0716
    assert row["absolute_difference"] == pytest.approx(0.0004)
    assert row["relative_to_dsmc"] == pytest.approx(0.005586592178770916)


def test_method_fidelity_gap_is_explicit() -> None:
    gap = method_fidelity_gap()
    assert PUBLISHED_DETERMINISTIC_METHOD["velocity_vector_count"] == 32000
    assert PUBLISHED_DETERMINISTIC_METHOD["physical_cells_I"] == 400
    assert PUBLISHED_DETERMINISTIC_METHOD["transport"] == (
        "second_order_control_volume"
    )
    assert gap["physical_cells_per_direction_ratio_source_to_stage39"] == pytest.approx(
        16.666666666666668
    )
    assert gap["physical_cell_count_ratio_source_to_stage39"] == pytest.approx(
        277.77777777777777
    )
    assert gap["velocity_vector_count_ratio_source_to_stage39"] == pytest.approx(
        6.944444444444445
    )
    assert gap["convergence_tolerance_ratio_stage39_to_source"] == pytest.approx(
        200000.0
    )
    assert gap["transport_order_matches"] is False
    assert gap["distribution_representation_matches"] is False
    assert gap["velocity_quadrature_architecture_matches"] is False


def test_stage40_decision_paths_are_preregistered() -> None:
    assert stage40_decision(False, True, True, False) == (
        "stage40_source_transcription_blocker"
    )
    assert stage40_decision(True, True, True, False) == (
        "heat_flux_independently_supported_stage41_projected_polar_dvm"
    )
    assert stage40_decision(True, False, True, False) == (
        "heat_flux_external_mismatch_stage41_normalization_and_flux_audit"
    )
    assert stage40_decision(True, True, False, False) == (
        "heat_flux_external_mismatch_stage41_normalization_and_flux_audit"
    )
    assert stage40_decision(True, True, True, True) == (
        "external_benchmarks_resolved_stage41_source_faithful_confirmation"
    )


def test_run_stage40_writes_reproducible_summary(tmp_path) -> None:
    summary = run_stage40(tmp_path)
    assert summary["stage"] == 40
    assert summary["repository_source_transcription"]["all_exact"] is True
    comparison = summary["stage39_external_heat_flux_comparison"]
    assert comparison["predicted_qav"] == pytest.approx(0.0726173364328985)
    assert comparison["relative_error_to_source_shakhov"] == pytest.approx(
        0.008574117123590345
    )
    assert comparison["relative_error_to_source_dsmc"] == pytest.approx(
        0.014208609398023778
    )
    assert comparison["agrees_with_source_shakhov"] is True
    assert comparison["agrees_with_independent_dsmc"] is True
    assert summary["table3_reference_status"][
        "independent_table3_reference_available"
    ] is False
    assert summary["physical_parameter_retuning"] is False
    assert summary["decision"] == (
        "heat_flux_independently_supported_stage41_projected_polar_dvm"
    )

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved == summary
