import numpy as np
import pytest

from vgdsmc import stage82_within_sector_angular_coherence_audit as stage82


def _synthetic_groups():
    face = np.zeros((96, 64, 63), dtype=float)
    cell = np.zeros((96, 64, 64), dtype=float)
    mapping = np.repeat(np.arange(8), 12).astype(np.int16)
    angles = np.arange(96, dtype=float) * 3.75
    base_face = np.linspace(-1.0, 1.0, 64 * 63).reshape(64, 63)
    base_cell = np.linspace(-1.0, 1.0, 64 * 64).reshape(64, 64)
    for i in range(96):
        amp = 1.0 + 0.01 * i
        face[i] = amp * base_face
        cell[i] = amp * base_cell
    return face, cell, mapping, angles


def test_stage82_frozen_design_accepts_only_registered_values():
    stage82.validate_stage82_design()
    with pytest.raises(ValueError):
        stage82.validate_stage82_design(kn0=0.1)
    with pytest.raises(ValueError):
        stage82.validate_stage82_design(vertical_oblique_bins=(0, 1, 4, 5))
    with pytest.raises(ValueError):
        stage82.validate_stage82_design(weighted_adjacent_coherence_guard=0.91)


def test_stage82_exact_completed_stage81_endpoint_is_frozen():
    endpoint = stage82.STAGE81_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31189580849
    assert endpoint["workflow_job_id"] == 92902686472
    assert endpoint["artifact_id"] == 9003488020
    assert endpoint["tests_passed"] == 247
    assert endpoint["tests_failed"] == 0
    assert endpoint["summary_sha256"] == "0f5c1772622303519f31c4faf447592150de97849667aee2f4b2eb5b1f15a5d4"
    assert endpoint["maps_sha256"] == "b9d60daed1e4df0d011e9885b332654939911182cdc732f984c2082b56122adb"


def test_stage82_vertical_bins_and_opposite_pairs_are_exactly_inherited():
    assert stage82.VERTICAL_OBLIQUE_BINS == (1, 2, 5, 6)
    assert stage82.OPPOSITE_SECTOR_PAIRS == ((1, 5), (2, 6))
    assert stage82.DOMINANT_GLOBAL_RADIAL_NODE == 21
    assert stage82.ANGULAR_NODES == 96


def test_correlation_and_weighted_adjacent_coherence_are_one_for_scaled_identical_maps():
    face, _, mapping, _ = _synthetic_groups()
    indices = np.flatnonzero(mapping == 1)
    weighted, correlations, weights = stage82._weighted_adjacent_coherence(face, indices)
    assert weighted == pytest.approx(1.0)
    assert np.allclose(correlations, 1.0)
    assert np.all(np.asarray(weights) > 0.0)
    assert stage82._corr(face[indices[0]], face[indices[-1]]) == pytest.approx(1.0)


def test_internal_retention_distinguishes_reinforcement_from_cancellation():
    base = np.ones((64, 64), dtype=float)
    reinforcing = np.stack([base, 2.0 * base])
    canceling = np.stack([base, -base])
    assert stage82._retention_ratio(reinforcing) == pytest.approx(1.0)
    assert stage82._retention_ratio(canceling) == pytest.approx(0.0)


def test_central_half_preserves_order_without_rebucketing():
    indices = np.arange(12, 24)
    core = stage82._central_half(indices)
    assert np.array_equal(core, np.arange(15, 21))
    odd = np.arange(11)
    assert np.array_equal(stage82._central_half(odd), np.arange(3, 8))


def test_within_sector_metrics_use_only_exact_vertical_bins_and_preserve_ordinates():
    face, cell, mapping, angles = _synthetic_groups()
    metrics = stage82.within_sector_metrics(face, cell, mapping, angles)
    assert [row["angular_bin"] for row in metrics["sectors"]] == [1, 2, 5, 6]
    assert [row["ordinate_count"] for row in metrics["sectors"]] == [12, 12, 12, 12]
    assert metrics["sector_face_groups"].shape == (4, 64, 63)
    assert metrics["sector_cell_groups"].shape == (4, 64, 64)
    assert metrics["vertical_oblique_cell_divergence_share_within_node"] > 0.0
    assert metrics["minimum_sector_cell_weighted_adjacent_coherence"] == pytest.approx(1.0)
    assert metrics["minimum_sector_cell_internal_retention_ratio"] == pytest.approx(1.0)


def test_stage82_decision_preregisters_smooth_retained_and_smooth_canceling_routes():
    metrics = {
        "minimum_sector_cell_weighted_adjacent_coherence": 0.95,
        "minimum_sector_cell_internal_retention_ratio": 0.80,
    }
    assert stage82.stage82_decision(True, True, metrics).startswith("stage82_smooth_retained")
    metrics["minimum_sector_cell_internal_retention_ratio"] = 0.70
    assert stage82.stage82_decision(True, True, metrics).startswith("stage82_smooth_but_internally_canceling")


def test_stage82_decision_preserves_mixed_route_and_blockers():
    metrics = {
        "minimum_sector_cell_weighted_adjacent_coherence": 0.85,
        "minimum_sector_cell_internal_retention_ratio": 0.90,
    }
    assert stage82.stage82_decision(True, True, metrics).startswith("stage82_mixed_within_sector")
    assert stage82.stage82_decision(False, True, metrics).endswith("blocker")
    assert stage82.stage82_decision(True, False, metrics).endswith("blocker")


def test_stage82_guards_are_diagnostic_only_and_no_solver_retuning_is_encoded():
    assert stage82.WEIGHTED_ADJACENT_COHERENCE_GUARD == 0.90
    assert stage82.SECTOR_INTERNAL_RETENTION_GUARD == 0.75
    assert stage82.KNUDSEN == 10.0
    assert stage82.LIMITER == "minmod"
