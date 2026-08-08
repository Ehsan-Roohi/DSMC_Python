from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc import stage85_near_wall_face_amplitude_suppression_audit as s85


def _synthetic_faces(wall_scale: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, 64)
    face = np.zeros((2, 64, 63), dtype=np.float64)
    for p in range(2):
        amp = 1.0 + 0.2 * p
        first = amp * y
        second = 0.95 * amp * y
        face[p, :, 0] = wall_scale * first
        face[p, :, 1] = first
        face[p, :, 2] = second
        face[p, :, -1] = wall_scale * first
        face[p, :, -2] = first
        face[p, :, -3] = second
    pairs = np.asarray([[1, 5], [2, 6]], dtype=np.int16)
    return face, pairs


def test_stage85_design_is_frozen() -> None:
    s85.validate_stage85_design()


def test_stage85_rejects_knudsen_retuning() -> None:
    with pytest.raises(ValueError, match="frozen"):
        s85.validate_stage85_design(kn0=1.0)


def test_stage85_rejects_diagnostic_guard_retuning() -> None:
    with pytest.raises(ValueError, match="frozen"):
        s85.validate_stage85_design(wall_to_first_l2_ratio_max=0.2)


def test_abrupt_synthetic_suppression_selects_boundary_stencil_audit() -> None:
    face, pairs = _synthetic_faces(wall_scale=0.01)
    metrics = s85.near_wall_face_metrics(face, pairs)
    assert metrics["decision"] == "stage85_abrupt_near_wall_face_suppression_stage86_boundary_reconstruction_stencil_activation_audit"
    assert metrics["maximum_wall_to_first_l2_ratio"] == pytest.approx(0.01)
    assert metrics["maximum_wall_to_first_l1_ratio"] == pytest.approx(0.01)
    assert metrics["maximum_scaled_shape_residual_rel_to_first"] <= 1e-14
    assert metrics["minimum_near_wall_jump_share_of_first_two_transitions"] >= s85.NEAR_WALL_JUMP_SHARE_GUARD


def test_distributed_synthetic_variation_does_not_select_abrupt_route() -> None:
    face, pairs = _synthetic_faces(wall_scale=0.8)
    metrics = s85.near_wall_face_metrics(face, pairs)
    assert metrics["decision"] == "stage85_distributed_or_incoherent_face_variation_stage86_frozen_spatial_profile_audit"
    assert metrics["maximum_wall_to_first_l2_ratio"] == pytest.approx(0.8)


def test_side_vector_indexing_is_exact() -> None:
    face = np.arange(2 * 64 * 63, dtype=np.float64).reshape(2, 64, 63)
    left = s85._side_vectors(face[0], "left")
    right = s85._side_vectors(face[0], "right")
    assert np.array_equal(left[0], face[0, :, 0])
    assert np.array_equal(left[1], face[0, :, 1])
    assert np.array_equal(left[2], face[0, :, 2])
    assert np.array_equal(right[0], face[0, :, -1])
    assert np.array_equal(right[1], face[0, :, -2])
    assert np.array_equal(right[2], face[0, :, -3])


def test_scaled_shape_residual_handles_nearly_extinguished_wall_signal() -> None:
    face, pairs = _synthetic_faces(wall_scale=0.0)
    metrics = s85.near_wall_face_metrics(face, pairs)
    for row in metrics["rows"]:
        assert row["wall_to_first_l2_ratio"] == pytest.approx(0.0)
        assert row["wall_as_scaled_first_factor"] == pytest.approx(0.0)
        assert row["scaled_shape_residual_rel_to_first"] == pytest.approx(0.0)
        assert row["wall_first_raw_y_correlation"] == pytest.approx(0.0)


def test_near_wall_jump_share_detects_first_transition_concentration() -> None:
    face, pairs = _synthetic_faces(wall_scale=0.01)
    metrics = s85.near_wall_face_metrics(face, pairs)
    for row in metrics["rows"]:
        assert row["near_wall_jump_share_of_first_two_transitions"] > 0.9
        assert row["first_to_second_l2_ratio"] == pytest.approx(1.0 / 0.95)


def test_stage85_rejects_wrong_face_shape() -> None:
    _, pairs = _synthetic_faces()
    with pytest.raises(ValueError, match="shape"):
        s85.near_wall_face_metrics(np.zeros((2, 64, 62)), pairs)


def test_stage85_rejects_redefined_pairs() -> None:
    face, _ = _synthetic_faces()
    with pytest.raises(ValueError, match="exact inherited"):
        s85.near_wall_face_metrics(face, np.asarray([[0, 4], [2, 6]], dtype=np.int16))


def test_stage85_rejects_nonfinite_input() -> None:
    face, pairs = _synthetic_faces()
    face[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        s85.near_wall_face_metrics(face, pairs)


def test_stage84_endpoint_is_pinned_to_completed_success() -> None:
    endpoint = s85.STAGE84_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31235732726
    assert endpoint["workflow_job_id"] == 93047750025
    assert endpoint["workflow_conclusion"] == "success"
    assert endpoint["tests_passed"] == 279
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 9017015976
    assert endpoint["summary_sha256"] == "18281901612355cc62e9395e78f7f5b4040d17f0fac6b56d483a27ab4726fe4f"
    assert endpoint["maps_sha256"] == "4b9775d20b964bab0deb0fe6c09e4a085388d5e49977b1cc9df33884969490d2"


def test_stage84_artifact_checksum_guard_rejects_unregistered_files(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"stage": 84}), encoding="utf-8")
    np.savez_compressed(tmp_path / "wall_normal_sign_lobe_geometry_maps.npz", x=np.asarray([1.0]))
    with pytest.raises(ValueError, match="checksum"):
        s85._validate_stage84_artifact(tmp_path)
