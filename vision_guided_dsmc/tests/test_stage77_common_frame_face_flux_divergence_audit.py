import numpy as np
import pytest

from vgdsmc import stage77_common_frame_face_flux_divergence_audit as stage77


def test_stage77_design_is_frozen():
    stage77.validate_stage77_design()
    with pytest.raises(ValueError):
        stage77.validate_stage77_design(kn0=9.0)
    with pytest.raises(ValueError):
        stage77.validate_stage77_design(face_to_cell_cancellation_guard=0.2)


def test_divergence_from_interior_faces_is_conservative():
    face = np.zeros((64, 63), dtype=float)
    face[20, 30] = 2.0
    cell = stage77.divergence_from_interior_faces(face)
    assert cell[20, 30] == -2.0
    assert cell[20, 31] == 2.0
    assert np.isclose(np.sum(cell), 0.0)
    assert np.isclose(np.sum(np.abs(cell)), 4.0)


def test_face_divergence_metrics_identify_coherent_cancellation():
    x = np.linspace(-1.0, 1.0, 63)
    face = np.tile(x, (64, 1))
    cell = stage77.divergence_from_interior_faces(face)
    metrics = stage77.face_divergence_metrics(face, cell)
    assert 0.0 <= metrics["face_to_cell_cancellation_ratio"] <= 1.0
    assert metrics["adjacent_x_face_correlation"] > 0.99
    assert abs(metrics["cell_signed_to_absolute_ratio"]) < 1e-12


def test_spatial_shares_close():
    cell = np.ones((64, 64), dtype=float)
    shares = stage77.spatial_shares(cell)
    assert 0.0 < shares["outer_one_cell_wall_share"] < 1.0
    assert 0.0 < shares["outer_two_cell_wall_share"] < 1.0
    assert np.isclose(
        shares["outer_two_cell_wall_share"] + shares["interior_two_cell_complement_share"],
        1.0,
    )


def test_stage77_decision_guards():
    assert stage77.stage77_decision(False, True, 0.01, 0.99).endswith("blocker")
    assert stage77.stage77_decision(True, False, 0.01, 0.99).endswith("blocker")
    assert "coherent_face_cancellation" in stage77.stage77_decision(True, True, 0.05, 0.95)
    assert "noncoherent_or_material_divergence" in stage77.stage77_decision(True, True, 0.20, 0.95)
    assert "noncoherent_or_material_divergence" in stage77.stage77_decision(True, True, 0.05, 0.50)
