import numpy as np
import pytest

from vgdsmc import stage78_face_gradient_moment_attribution_audit as stage78


def test_stage78_design_is_frozen():
    stage78.validate_stage78_design()
    with pytest.raises(ValueError):
        stage78.validate_stage78_design(kn0=9.0)
    with pytest.raises(ValueError):
        stage78.validate_stage78_design(dominant_cell_divergence_share_guard=0.49)


def test_divergence_from_interior_faces_is_conservative():
    face = np.zeros((64, 63), dtype=float)
    face[12, 18] = 3.0
    cell = stage78.divergence_from_interior_faces(face)
    assert cell[12, 18] == -3.0
    assert cell[12, 19] == 3.0
    assert np.isclose(np.sum(cell), 0.0)
    assert np.isclose(np.sum(np.abs(cell)), 6.0)


def test_component_gradient_maps_close_to_summed_gradient():
    face = np.zeros((3, 64, 63), dtype=float)
    face[0, 20, 10] = 1.0
    face[1, 20, 10] = -0.25
    face[2, 30, 40] = 2.0
    cell = stage78.component_gradient_maps(face)
    summed = np.sum(cell, axis=0)
    direct = stage78.divergence_from_interior_faces(np.sum(face, axis=0))
    assert cell.shape == (3, 64, 64)
    assert np.allclose(summed, direct)
    assert np.isclose(np.sum(summed), 0.0)


def test_component_metrics_shares_close_and_bounds_hold():
    x = np.linspace(-1.0, 1.0, 63)
    face = np.stack([
        np.tile(x, (64, 1)),
        2.0 * np.tile(x, (64, 1)),
        0.5 * np.tile(x, (64, 1)),
    ])
    cell = stage78.component_gradient_maps(face)
    metrics = stage78.component_metrics(face, cell)
    shares = [
        metrics["per_component"][name]["cell_divergence_absolute_share"]
        for name in stage78.MOMENT_NAMES
    ]
    assert np.isclose(sum(shares), 1.0)
    assert metrics["dominant_component"] == "transverse_kinetic"
    assert 0.0 <= metrics["intercomponent_face_cancellation_ratio"] <= 1.0 + 1e-12
    assert 0.0 <= metrics["intercomponent_cell_divergence_cancellation_ratio"] <= 1.0 + 1e-12


def test_component_metrics_conservative_signed_sums():
    rng = np.random.default_rng(4)
    face = rng.normal(size=(3, 64, 63))
    cell = stage78.component_gradient_maps(face)
    metrics = stage78.component_metrics(face, cell)
    for name in stage78.MOMENT_NAMES:
        assert abs(metrics["per_component"][name]["cell_signed_sum"]) < 1e-10


def test_stage78_decision_blockers_and_dominant_routes():
    assert stage78.stage78_decision(False, True, 0.8, 0.95, 1.0).endswith("blocker")
    assert stage78.stage78_decision(True, False, 0.8, 0.95, 1.0).endswith("blocker")
    assert "dominant_coherent_moment" in stage78.stage78_decision(True, True, 0.60, 0.95, 1.0)
    assert "dominant_noncoherent_moment" in stage78.stage78_decision(True, True, 0.60, 0.50, 1.0)


def test_stage78_decision_mixed_routes():
    assert "mixed_moment_cancellation" in stage78.stage78_decision(True, True, 0.40, 0.95, 0.70)
    assert "mixed_moment_divergence" in stage78.stage78_decision(True, True, 0.40, 0.95, 0.90)
