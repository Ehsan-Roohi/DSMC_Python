from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc import stage83_opposite_sector_spatial_cancellation_localization_audit as s83


def _synthetic_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face = np.zeros((4, 64, 63), dtype=np.float64)
    cell = np.zeros((4, 64, 64), dtype=np.float64)
    bins = np.asarray([1, 2, 5, 6], dtype=np.int16)
    for left, right, sign in [(0, 2, 1.0), (1, 3, -1.0)]:
        face[left] = 0.5
        face[right] = 0.5
        pair = np.zeros((64, 64), dtype=np.float64)
        pair[:, 0] = sign
        pair[:, -1] = -sign
        cell[left] = 0.5 * pair
        cell[right] = 0.5 * pair
    return face, cell, bins


def test_stage83_design_is_frozen() -> None:
    s83.validate_stage83_design()


def test_stage83_rejects_knudsen_retuning() -> None:
    with pytest.raises(ValueError, match="frozen"):
        s83.validate_stage83_design(kn0=1.0)


def test_stage83_rejects_spatial_partition_retuning() -> None:
    with pytest.raises(ValueError, match="frozen"):
        s83.validate_stage83_design(outer_x_quarter_width=8)


def test_signed_retention_extremes() -> None:
    assert s83._retention(np.ones((2, 2))) == pytest.approx(1.0)
    assert s83._retention(np.asarray([[1.0, -1.0]])) == pytest.approx(0.0)


def test_rowwise_and_columnwise_retention_distinguish_cancellation_direction() -> None:
    arr = np.tile(np.asarray([1.0, -1.0]), (4, 1))
    assert s83._profile_retention(arr, 1) == pytest.approx(0.0)
    assert s83._profile_retention(arr, 0) == pytest.approx(1.0)


def test_spatial_metrics_reconstruct_fixed_pairs_exactly() -> None:
    face, cell, bins = _synthetic_maps()
    metrics = s83.spatial_cancellation_metrics(face, cell, bins)
    closure = metrics["pair_reconstruction_closure"]
    assert closure["within_guard"] is True
    assert closure["maximum_absolute_error"] == pytest.approx(0.0)
    assert closure["face_relative_l2_error"] == pytest.approx(0.0)
    assert closure["cell_relative_l2_error"] == pytest.approx(0.0)


def test_synthetic_sidewall_rowwise_case_selects_diagnostic_branch() -> None:
    face, cell, bins = _synthetic_maps()
    metrics = s83.spatial_cancellation_metrics(face, cell, bins)
    assert metrics["decision"] == "stage83_rowwise_sidewall_localized_conservative_cancellation_stage84_wall_normal_sign_lobe_audit"
    for row in metrics["pairs"]:
        assert row["cell_retention_ratio"] == pytest.approx(1.0)
        assert row["rowwise_signed_retention_ratio"] == pytest.approx(0.0)
        assert row["outer_x_quarters_absolute_share"] == pytest.approx(1.0)
        assert row["face_to_cell_cancellation_ratio"] < s83.FACE_TO_CELL_CANCELLATION_GUARD


def test_spatial_metrics_reject_rebucketed_labels() -> None:
    face, cell, _ = _synthetic_maps()
    with pytest.raises(ValueError, match="exact inherited"):
        s83.spatial_cancellation_metrics(face, cell, np.asarray([1, 2, 4, 6]))


def test_spatial_metrics_reject_nonfinite_input() -> None:
    face, cell, bins = _synthetic_maps()
    cell[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s83.spatial_cancellation_metrics(face, cell, bins)


def test_stage82_endpoint_is_pinned_to_completed_success() -> None:
    endpoint = s83.STAGE82_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31203808127
    assert endpoint["workflow_job_id"] == 92949872247
    assert endpoint["workflow_conclusion"] == "success"
    assert endpoint["tests_passed"] == 257
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 9011460708
    assert endpoint["decision"].startswith("stage82_smooth_retained_vertical_oblique_sectors")


def test_stage82_artifact_checksum_guard_rejects_unregistered_files(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"stage": 82}), encoding="utf-8")
    np.savez_compressed(tmp_path / "within_sector_angular_coherence_maps.npz", x=np.asarray([1.0]))
    with pytest.raises(ValueError, match="checksum"):
        s83._validate_stage82_artifact(tmp_path)
