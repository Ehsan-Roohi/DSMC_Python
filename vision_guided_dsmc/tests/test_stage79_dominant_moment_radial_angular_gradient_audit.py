import math

import numpy as np
import pytest

from vgdsmc import stage79_dominant_moment_radial_angular_gradient_audit as stage79


def _polar_rule():
    radii = np.arange(1.0, stage79.RULE[0] + 1.0)
    angles = np.arange(stage79.RULE[1]) * (2.0 * math.pi / stage79.RULE[1])
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    return (rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()


def test_stage79_frozen_design_accepts_only_registered_values():
    stage79.validate_stage79_design()
    with pytest.raises(ValueError):
        stage79.validate_stage79_design(kn0=9.0)
    with pytest.raises(ValueError):
        stage79.validate_stage79_design(radial_shell_count=5)
    with pytest.raises(ValueError):
        stage79.validate_stage79_design(angular_bin_count=12)


def test_stage79_fixed_endpoint_and_dominant_moment_are_exact():
    assert stage79.STAGE78_COMPLETED_ENDPOINT["workflow_run_id"] == 31144240478
    assert stage79.STAGE78_COMPLETED_ENDPOINT["artifact_id"] == 8986364040
    assert stage79.DOMINANT_MOMENT == "transverse_kinetic"
    assert stage79.DOMINANT_MOMENT_INDEX == 1
    assert stage79.RADIAL_SHELL_COUNT == 4
    assert stage79.RADIAL_NODES_PER_SHELL == 10
    assert stage79.ANGULAR_BIN_COUNT == 8


def test_radial_shells_are_four_equal_frozen_node_groups():
    vx, vy = _polar_rule()
    labels = stage79.radial_shell_indices(vx, vy)
    counts = np.bincount(labels, minlength=stage79.RADIAL_SHELL_COUNT)
    assert counts.tolist() == [960, 960, 960, 960]
    speed = np.hypot(vx, vy)
    maxima = [np.max(speed[labels == shell]) for shell in range(4)]
    minima = [np.min(speed[labels == shell]) for shell in range(4)]
    assert maxima[0] < minima[1] <= maxima[1] < minima[2] <= maxima[2] < minima[3]


def test_angular_bins_are_eight_zero_offset_sectors():
    vx, vy = _polar_rule()
    labels = stage79.angular_bin_indices(vx, vy)
    counts = np.bincount(labels, minlength=stage79.ANGULAR_BIN_COUNT)
    assert counts.sum() == stage79.POINT_COUNT
    assert np.all(counts > 0)
    cardinal_vx = np.array([1.0, 0.0, -1.0, 0.0])
    cardinal_vy = np.array([0.0, 1.0, 0.0, -1.0])
    cardinal = stage79.angular_bin_indices(cardinal_vx, cardinal_vy)
    assert cardinal.tolist() == [0, 2, 4, 6]


def test_divergence_is_conservative_for_interior_faces():
    rng = np.random.default_rng(79)
    face = rng.normal(size=(64, 63))
    cell = stage79.divergence_from_interior_faces(face)
    assert cell.shape == (64, 64)
    assert abs(float(np.sum(cell))) < 1.0e-12


def test_stage79_minmod_is_not_a_retuned_limiter():
    left = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    right = np.array([-1.0, 2.0, 3.0, 0.5, -2.0])
    got = stage79.minmod(left, right)
    assert np.allclose(got, [-1.0, 0.0, 0.0, 0.5, 0.0])
    assert stage79.LIMITER == "minmod"


def test_stage79_decision_follows_preregistered_concentration_branches():
    metrics = {
        "dominant_radial_shell_cell_divergence_share": 0.60,
        "top_two_radial_shell_cell_divergence_share": 0.80,
        "vertical_oblique_cell_divergence_share": 0.75,
    }
    assert stage79.stage79_decision(True, True, metrics).startswith(
        "stage79_dominant_radial_shell_vertical_oblique"
    )
    metrics["dominant_radial_shell_cell_divergence_share"] = 0.40
    metrics["vertical_oblique_cell_divergence_share"] = 0.60
    assert stage79.stage79_decision(True, True, metrics).startswith(
        "stage79_radially_concentrated_angularly_mixed"
    )
    metrics["top_two_radial_shell_cell_divergence_share"] = 0.50
    metrics["vertical_oblique_cell_divergence_share"] = 0.80
    assert stage79.stage79_decision(True, True, metrics).startswith(
        "stage79_angularly_concentrated_radially_mixed"
    )


def test_stage79_blockers_precede_scientific_branching():
    metrics = {
        "dominant_radial_shell_cell_divergence_share": 1.0,
        "top_two_radial_shell_cell_divergence_share": 1.0,
        "vertical_oblique_cell_divergence_share": 1.0,
    }
    assert stage79.stage79_decision(False, True, metrics).endswith("blocker")
    assert stage79.stage79_decision(True, False, metrics).endswith("blocker")
