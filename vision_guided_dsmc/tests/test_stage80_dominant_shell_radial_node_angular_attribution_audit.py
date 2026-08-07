import math

import numpy as np
import pytest

from vgdsmc import stage80_dominant_shell_radial_node_angular_attribution_audit as stage80


def _polar_rule():
    radii = np.arange(1.0, stage80.RULE[0] + 1.0)
    angles = np.arange(stage80.RULE[1]) * (2.0 * math.pi / stage80.RULE[1])
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    return (rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()


def test_stage80_frozen_design_accepts_only_registered_values():
    stage80.validate_stage80_design()
    with pytest.raises(ValueError):
        stage80.validate_stage80_design(kn0=0.1)
    with pytest.raises(ValueError):
        stage80.validate_stage80_design(dominant_radial_shell=1)
    with pytest.raises(ValueError):
        stage80.validate_stage80_design(dominant_radial_node_share_guard=0.21)


def test_stage80_exact_completed_stage79_endpoint_is_frozen():
    endpoint = stage80.STAGE79_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31159895420
    assert endpoint["workflow_job_id"] == 92807602344
    assert endpoint["artifact_id"] == 8991295336
    assert endpoint["tests_passed"] == 224
    assert endpoint["tests_failed"] == 0
    assert endpoint["summary_sha256"] == "9896aef5ec092e8deb6922fda36065f3dd1f54f452a2d41844d96d0a523930bf"
    assert endpoint["maps_sha256"] == "b05cb4d125434c7f1c4b6e5bbc2805915d86467d53b310910286e535ae566d5a"


def test_radial_nodes_are_forty_equal_96_point_groups():
    vx, vy = _polar_rule()
    labels = stage80.radial_node_indices(vx, vy)
    counts = np.bincount(labels, minlength=stage80.RULE[0])
    assert counts.tolist() == [stage80.RULE[1]] * stage80.RULE[0]
    assert labels.min() == 0
    assert labels.max() == 39


def test_node_to_shell_mapping_reproduces_four_equal_stage79_shells():
    vx, vy = _polar_rule()
    nodes = stage80.radial_node_indices(vx, vy)
    shells = stage80.radial_shell_indices_from_nodes(nodes)
    assert np.bincount(shells, minlength=4).tolist() == [960, 960, 960, 960]
    assert set(np.unique(nodes[shells == stage80.DOMINANT_RADIAL_SHELL]).tolist()) == set(range(20, 30))


def test_angular_bins_retain_eight_zero_offset_sectors():
    vx, vy = _polar_rule()
    labels = stage80.stage79.angular_bin_indices(vx, vy)
    assert labels.shape == (stage80.POINT_COUNT,)
    assert np.all(np.bincount(labels, minlength=8) > 0)
    cardinal = stage80.stage79.angular_bin_indices(
        np.array([1.0, 0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, -1.0]),
    )
    assert cardinal.tolist() == [0, 2, 4, 6]


def test_retained_divergence_from_interior_faces_is_conservative():
    rng = np.random.default_rng(80)
    face = rng.normal(size=(64, 63))
    cell = stage80.stage79.divergence_from_interior_faces(face)
    assert cell.shape == (64, 64)
    assert abs(float(np.sum(cell))) < 1.0e-12


def test_stage80_retains_stage79_minmod_without_retuning():
    left = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    right = np.array([-1.0, 2.0, 3.0, 0.5, -2.0])
    assert np.allclose(stage80.stage79.minmod(left, right), [-1.0, 0.0, 0.0, 0.5, 0.0])
    assert stage80.LIMITER == "minmod"


def test_stage80_decision_preregisters_single_node_and_cluster_routes():
    metrics = {
        "dominant_radial_node_cell_divergence_share": 0.25,
        "top_three_radial_node_cell_divergence_share": 0.70,
        "vertical_oblique_cell_divergence_share_within_shell": 0.80,
    }
    assert stage80.stage80_decision(True, True, metrics).startswith("stage80_single_radial_node_vertical_oblique")
    metrics["dominant_radial_node_cell_divergence_share"] = 0.15
    metrics["vertical_oblique_cell_divergence_share_within_shell"] = 0.60
    assert stage80.stage80_decision(True, True, metrics).startswith("stage80_radial_node_cluster")


def test_stage80_decision_keeps_vertical_only_and_diffuse_routes_distinct():
    metrics = {
        "dominant_radial_node_cell_divergence_share": 0.15,
        "top_three_radial_node_cell_divergence_share": 0.50,
        "vertical_oblique_cell_divergence_share_within_shell": 0.80,
    }
    assert stage80.stage80_decision(True, True, metrics).startswith("stage80_vertical_oblique_radially_distributed")
    metrics["vertical_oblique_cell_divergence_share_within_shell"] = 0.60
    assert stage80.stage80_decision(True, True, metrics).startswith("stage80_within_shell_attribution_diffuse")


def test_stage80_blockers_precede_scientific_branching():
    metrics = {
        "dominant_radial_node_cell_divergence_share": 1.0,
        "top_three_radial_node_cell_divergence_share": 1.0,
        "vertical_oblique_cell_divergence_share_within_shell": 1.0,
    }
    assert stage80.stage80_decision(False, True, metrics).endswith("blocker")
    assert stage80.stage80_decision(True, False, metrics).endswith("blocker")


def test_stage80_guards_are_fixed_before_execution_and_do_not_enable_retuning():
    assert stage80.DOMINANT_RADIAL_NODE_SHARE_GUARD == 0.20
    assert stage80.TOP_THREE_RADIAL_NODE_CONCENTRATION_GUARD == 0.60
    assert stage80.VERTICAL_OBLIQUE_CONCENTRATION_GUARD == 0.70
    assert stage80.KNUDSEN == 10.0
    assert stage80.DOMINANT_SHELL_GLOBAL_NODE_START == 20
    assert stage80.DOMINANT_SHELL_GLOBAL_NODE_STOP == 30
