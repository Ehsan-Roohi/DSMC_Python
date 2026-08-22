from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage147_dual_channel_neighborhood_audit import (
    BILATERAL_REVERSAL,
    IDENTITY_BLOCKER,
    MATERIAL_NEIGHBOR_RATIO_MIN,
    MIXED,
    PERSISTENT,
    PROVENANCE_BLOCKER,
    classify_dual_channel_neighborhood,
    dual_channel_neighborhood_metrics,
    validate_stage147_design,
)


def _actual_profiles():
    depth = np.array([5.961196168207854, 6.961196168207854, 7.961196168207854, 8.961196168207854, 9.961196168207854, 10.961196168207854, 11.961196168207854])
    dominant = np.array([-0.08917300, -0.06528149620780105, -0.06942764887925673, -0.06230659651118298, -0.04816489506330546, -0.04655430429054869, -0.03562422])
    parent = np.array([-0.10420902, -0.08998564224831607, -0.0657307037504089, -0.04500569892676798, -0.03899527101381384, -0.02863689878165021, -0.01905201])
    complement = parent - dominant
    return depth, dominant, parent, complement


def test_stage147_frozen_design_accepts_defaults() -> None:
    validate_stage147_design()


def test_stage147_rejects_retuning_or_materiality_change() -> None:
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage147_design(kn0=9.0)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage147_design(material_neighbor_ratio_min=0.20)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage147_design(limiter_retuning=True)


def test_stage147_actual_parent_profile_closes_and_reverses_bilaterally() -> None:
    depth, dominant, parent, complement = _actual_profiles()
    m = dual_channel_neighborhood_metrics(depth, dominant, parent, complement, 9.961196168207854)
    assert m["maximum_channel_identity_or_decomposition_closure"] < 1e-14
    assert m["curvature_sign_sequence"] == [-1, 1, -1]
    assert m["neighbor_material_bilateral_reversal_count"] == 2
    assert m["bilateral_material_channel_sign_reversal"] is True


def test_stage147_actual_center_reproduces_stage146_contributions() -> None:
    depth, dominant, parent, complement = _actual_profiles()
    m = dual_channel_neighborhood_metrics(depth, dominant, parent, complement, 9.961196168207854)
    assert m["dominant_projected_curvature"][1] == pytest.approx(0.0062655553375603745)
    assert m["parent_projected_curvature"][1] == pytest.approx(0.0021739721596047423)
    assert m["complement_secant_deficit"][1] == pytest.approx(0.008439527497165117)


def test_stage147_actual_neighbor_materiality_is_not_tiny() -> None:
    depth, dominant, parent, complement = _actual_profiles()
    m = dual_channel_neighborhood_metrics(depth, dominant, parent, complement, 9.961196168207854)
    assert m["minimum_neighbor_absolute_ratio_to_center"] == pytest.approx(0.5979585193305169)
    assert m["minimum_neighbor_absolute_ratio_to_center"] > MATERIAL_NEIGHBOR_RATIO_MIN


def test_stage147_classifies_bilateral_material_reversal() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "minimum_neighbor_absolute_ratio_to_center": 0.6,
        "maximum_channel_identity_or_decomposition_closure": 0.0,
        "center_channels_reinforce": True,
        "neighbor_reinforcing_count": 0,
        "bilateral_material_channel_sign_reversal": True,
    }
    assert classify_dual_channel_neighborhood(metrics=metrics) == BILATERAL_REVERSAL


def test_stage147_classifies_persistent_when_both_neighbors_reinforce() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "minimum_neighbor_absolute_ratio_to_center": 0.5,
        "maximum_channel_identity_or_decomposition_closure": 0.0,
        "center_channels_reinforce": True,
        "neighbor_reinforcing_count": 2,
        "bilateral_material_channel_sign_reversal": False,
    }
    assert classify_dual_channel_neighborhood(metrics=metrics) == PERSISTENT


def test_stage147_classifies_mixed_when_reversal_not_bilateral() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "minimum_neighbor_absolute_ratio_to_center": 0.1,
        "maximum_channel_identity_or_decomposition_closure": 0.0,
        "center_channels_reinforce": True,
        "neighbor_reinforcing_count": 1,
        "bilateral_material_channel_sign_reversal": False,
    }
    assert classify_dual_channel_neighborhood(metrics=metrics) == MIXED


def test_stage147_blocks_center_provenance_mismatch() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "minimum_neighbor_absolute_ratio_to_center": 0.5,
        "maximum_channel_identity_or_decomposition_closure": 0.0,
        "center_channels_reinforce": True,
        "neighbor_reinforcing_count": 0,
        "bilateral_material_channel_sign_reversal": True,
    }
    assert classify_dual_channel_neighborhood(metrics=metrics, center_metric_match_error=1e-6) == PROVENANCE_BLOCKER


def test_stage147_blocks_identity_failure() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "minimum_neighbor_absolute_ratio_to_center": 0.5,
        "maximum_channel_identity_or_decomposition_closure": 1e-6,
        "center_channels_reinforce": True,
        "neighbor_reinforcing_count": 0,
        "bilateral_material_channel_sign_reversal": True,
    }
    assert classify_dual_channel_neighborhood(metrics=metrics) == IDENTITY_BLOCKER


def test_stage147_rejects_nonmonotone_depth() -> None:
    depth, dominant, parent, complement = _actual_profiles()
    depth[3] = depth[2]
    with pytest.raises(ValueError, match="strictly increasing"):
        dual_channel_neighborhood_metrics(depth, dominant, parent, complement, 9.961196168207854)
