import math

import numpy as np
import pytest

from vgdsmc import stage81_dominant_node_individual_ordinate_audit as stage81


def _polar_rule():
    radii = np.arange(1.0, stage81.RULE[0] + 1.0)
    angles = np.arange(stage81.RULE[1]) * (2.0 * math.pi / stage81.RULE[1])
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    return (rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()


def test_stage81_frozen_design_accepts_only_registered_values():
    stage81.validate_stage81_design()
    with pytest.raises(ValueError):
        stage81.validate_stage81_design(kn0=0.1)
    with pytest.raises(ValueError):
        stage81.validate_stage81_design(dominant_global_radial_node=20)
    with pytest.raises(ValueError):
        stage81.validate_stage81_design(dominant_ordinate_share_guard=0.04)


def test_stage81_exact_completed_stage80_endpoint_is_frozen():
    endpoint = stage81.STAGE80_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31174110559
    assert endpoint["workflow_job_id"] == 92852196029
    assert endpoint["artifact_id"] == 8997800347
    assert endpoint["tests_passed"] == 235
    assert endpoint["tests_failed"] == 0
    assert endpoint["summary_sha256"] == "7bd0d09e7961c06fec5e797ff6010e0c08e4db50019ca4fb3fc2e208587d2baa"
    assert endpoint["maps_sha256"] == "3dbc3cdf0098c219ac20d7176d37bf0c6a6902d8b7fadcbf1e27af3a99ab62fb"


def test_dominant_node_is_resolved_into_exactly_96_unique_ordinates():
    vx, vy = _polar_rule()
    nodes = stage81.stage80.radial_node_indices(vx, vy)
    ordinates = stage81.dominant_node_ordinate_indices(vx, vy, nodes)
    selected = nodes == stage81.DOMINANT_GLOBAL_RADIAL_NODE
    assert np.sum(selected) == stage81.ORDINATE_COUNT
    assert np.array_equal(np.sort(ordinates[selected]), np.arange(stage81.ORDINATE_COUNT))
    assert np.all(ordinates[~selected] == -1)


def test_ordinate_labels_follow_increasing_polar_angle():
    vx, vy = _polar_rule()
    nodes = stage81.stage80.radial_node_indices(vx, vy)
    ordinates = stage81.dominant_node_ordinate_indices(vx, vy, nodes)
    angle_by_ordinate = []
    for ordinate in range(stage81.ORDINATE_COUNT):
        point = np.flatnonzero(ordinates == ordinate)[0]
        angle_by_ordinate.append(np.mod(np.arctan2(vy[point], vx[point]), 2.0 * math.pi))
    assert np.all(np.diff(angle_by_ordinate) > 0.0)
    assert angle_by_ordinate[0] == pytest.approx(0.0)


def test_stage81_preserves_inherited_stage80_bin_membership_without_rebucketing():
    vx, vy = _polar_rule()
    nodes = stage81.stage80.radial_node_indices(vx, vy)
    angular = stage81.stage80.stage79.angular_bin_indices(vx, vy)
    ordinates = stage81.dominant_node_ordinate_indices(vx, vy, nodes)
    mapping = stage81.ordinate_to_angular_bin(ordinates, angular)
    assert mapping.shape == (stage81.ORDINATE_COUNT,)
    assert int(np.sum(np.bincount(mapping, minlength=stage81.ANGULAR_BIN_COUNT))) == stage81.ORDINATE_COUNT
    for ordinate in range(stage81.ORDINATE_COUNT):
        point = np.flatnonzero(ordinates == ordinate)[0]
        assert mapping[ordinate] == angular[point]


def test_stage81_nominal_sector_size_is_documented_but_not_used_to_rebucket():
    assert stage81.NOMINAL_ORDINATES_PER_ANGULAR_BIN == 12
    assert stage81.ORDINATE_COUNT == 96
    assert stage81.ANGULAR_BIN_COUNT == 8


def test_retained_divergence_from_interior_faces_is_conservative():
    rng = np.random.default_rng(81)
    face = rng.normal(size=(64, 63))
    cell = stage81.stage80.stage79.divergence_from_interior_faces(face)
    assert cell.shape == (64, 64)
    assert abs(float(np.sum(cell))) < 1.0e-12


def test_stage81_retains_stage80_minmod_without_retuning():
    left = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    right = np.array([-1.0, 2.0, 3.0, 0.5, -2.0])
    assert np.allclose(stage81.stage80.stage79.minmod(left, right), [-1.0, 0.0, 0.0, 0.5, 0.0])
    assert stage81.LIMITER == "minmod"


def test_stage81_decision_preregisters_single_ordinate_and_cluster_routes():
    metrics = {
        "dominant_ordinate_cell_divergence_share": 0.06,
        "top_twelve_ordinate_cell_divergence_share": 0.60,
        "vertical_oblique_cell_divergence_share_within_node": 0.80,
    }
    assert stage81.stage81_decision(True, True, metrics).startswith("stage81_single_ordinate_concentration")
    metrics["dominant_ordinate_cell_divergence_share"] = 0.04
    assert stage81.stage81_decision(True, True, metrics).startswith("stage81_ordinate_cluster")


def test_stage81_decision_keeps_vertical_only_and_diffuse_routes_distinct():
    metrics = {
        "dominant_ordinate_cell_divergence_share": 0.04,
        "top_twelve_ordinate_cell_divergence_share": 0.40,
        "vertical_oblique_cell_divergence_share_within_node": 0.80,
    }
    assert stage81.stage81_decision(True, True, metrics).startswith("stage81_vertical_oblique_sector_distributed")
    metrics["vertical_oblique_cell_divergence_share_within_node"] = 0.60
    assert stage81.stage81_decision(True, True, metrics).startswith("stage81_dominant_node_angular_attribution_diffuse")


def test_stage81_blockers_precede_scientific_branching():
    metrics = {
        "dominant_ordinate_cell_divergence_share": 1.0,
        "top_twelve_ordinate_cell_divergence_share": 1.0,
        "vertical_oblique_cell_divergence_share_within_node": 1.0,
    }
    assert stage81.stage81_decision(False, True, metrics).endswith("blocker")
    assert stage81.stage81_decision(True, False, metrics).endswith("blocker")


def test_stage81_guards_are_fixed_before_execution_and_do_not_enable_retuning():
    assert stage81.DOMINANT_ORDINATE_SHARE_GUARD == 0.05
    assert stage81.TOP_TWELVE_ORDINATE_CONCENTRATION_GUARD == 0.50
    assert stage81.VERTICAL_OBLIQUE_CONCENTRATION_GUARD == 0.70
    assert stage81.KNUDSEN == 10.0
    assert stage81.DOMINANT_RADIAL_SHELL == 2
    assert stage81.DOMINANT_GLOBAL_RADIAL_NODE == 21
    assert stage81.DOMINANT_LOCAL_RADIAL_NODE == 1
