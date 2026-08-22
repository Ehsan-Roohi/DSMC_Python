from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc import stage84_wall_normal_sign_lobe_geometry_audit as s84


def _synthetic_pair_maps() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face = np.zeros((2, 64, 63), dtype=np.float64)
    profile = np.zeros(63, dtype=np.float64)
    profile[1:62] = np.linspace(1.0, -1.0, 61)
    face[:] = profile
    cell = s84.conservative_cell_from_face(face)
    pairs = np.asarray([[1, 5], [2, 6]], dtype=np.int16)
    return face, cell, pairs


def test_stage84_design_is_frozen() -> None:
    s84.validate_stage84_design()


def test_stage84_rejects_knudsen_retuning() -> None:
    with pytest.raises(ValueError, match="frozen"):
        s84.validate_stage84_design(kn0=1.0)


def test_stage84_rejects_pair_redefinition() -> None:
    with pytest.raises(ValueError, match="frozen"):
        s84.validate_stage84_design(opposite_sector_pairs=((0, 4), (2, 6)))


def test_conservative_face_divergence_telescopes_rowwise() -> None:
    face, _, _ = _synthetic_pair_maps()
    cell = s84.conservative_cell_from_face(face)
    assert np.max(np.abs(np.sum(cell, axis=2))) <= 1e-15


def test_sign_lobe_geometry_reconstructs_exact_cell_maps() -> None:
    face, cell, pairs = _synthetic_pair_maps()
    metrics = s84.sign_lobe_geometry_metrics(face, cell, pairs)
    closure = metrics["face_to_cell_divergence_closure"]
    assert closure["within_guard"] is True
    assert closure["maximum_absolute_error"] == pytest.approx(0.0)
    assert closure["cell_relative_l2_error"] == pytest.approx(0.0)


def test_synthetic_first_interior_lobes_select_face_amplitude_audit() -> None:
    face, cell, pairs = _synthetic_pair_maps()
    metrics = s84.sign_lobe_geometry_metrics(face, cell, pairs)
    assert metrics["decision"] == "stage84_first_interior_negative_lobes_broad_compensation_stage85_near_wall_face_amplitude_suppression_audit"
    for row in metrics["pairs"]:
        assert row["rowwise_signed_retention_ratio"] <= 1e-15
        assert row["first_interior_cells_absolute_share"] >= s84.FIRST_INTERIOR_TOTAL_SHARE_GUARD
        assert row["negative_mass_first_interior_share"] >= s84.NEGATIVE_FIRST_INTERIOR_SHARE_GUARD
        assert row["positive_mass_first_interior_share"] <= s84.POSITIVE_FIRST_INTERIOR_SHARE_MAX
        assert row["outermost_cells_absolute_share"] <= s84.OUTERMOST_CELL_SHARE_MAX


def test_sign_specific_effective_support_distinguishes_compact_and_broad_maps() -> None:
    compact = np.zeros(64)
    compact[[1, 62]] = 1.0
    broad = np.zeros(64)
    broad[8:56] = 1.0
    assert s84._effective_support(compact) == pytest.approx(2.0)
    assert s84._effective_support(broad) == pytest.approx(48.0)


def test_sign_lobe_geometry_rejects_redefined_pairs() -> None:
    face, cell, _ = _synthetic_pair_maps()
    with pytest.raises(ValueError, match="exact inherited"):
        s84.sign_lobe_geometry_metrics(face, cell, np.asarray([[0, 4], [2, 6]], dtype=np.int16))


def test_sign_lobe_geometry_rejects_nonfinite_input() -> None:
    face, cell, pairs = _synthetic_pair_maps()
    cell[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s84.sign_lobe_geometry_metrics(face, cell, pairs)


def test_stage83_endpoint_is_pinned_to_completed_success() -> None:
    endpoint = s84.STAGE83_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31225050170
    assert endpoint["workflow_job_id"] == 93017544665
    assert endpoint["workflow_conclusion"] == "success"
    assert endpoint["tests_passed"] == 268
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 9014969851
    assert endpoint["decision"].startswith("stage83_rowwise_sidewall_localized")


def test_stage83_artifact_checksum_guard_rejects_unregistered_files(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"stage": 83}), encoding="utf-8")
    np.savez_compressed(tmp_path / "opposite_sector_spatial_cancellation_localization_maps.npz", x=np.asarray([1.0]))
    with pytest.raises(ValueError, match="checksum"):
        s84._validate_stage83_artifact(tmp_path)
