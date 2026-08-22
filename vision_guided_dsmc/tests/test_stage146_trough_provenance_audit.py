from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage146_trough_provenance_audit import (
    IDENTITY_BLOCKER,
    MIXED_REINFORCING,
    OPPOSED,
    PROVENANCE_BLOCKER,
    SINGLE_CHANNEL,
    classify_trough_provenance,
    trough_provenance_metrics,
    validate_stage146_design,
)


def _profiles():
    depth = np.arange(5.0)
    dominant = np.array([-5.0, -4.0, -3.0, -4.0, -5.0])
    parent = np.array([-6.0, -3.0, -2.5, -3.0, -6.0])
    complement = parent - dominant
    return depth, dominant, parent, complement


def test_stage146_frozen_design_accepts_defaults() -> None:
    validate_stage146_design()


def test_stage146_rejects_retuning_or_threshold_change() -> None:
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage146_design(kn0=9.0)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage146_design(limiter="vanleer")
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage146_design(single_channel_dominance_min=0.70)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage146_design(physical_parameter_retuning=True)


def test_stage146_exact_channel_identity_and_deficit_decomposition() -> None:
    depth, dominant, parent, complement = _profiles()
    metrics = trough_provenance_metrics(depth, dominant, parent, complement, 2.0)
    assert metrics["channel_identity_closure"] == pytest.approx(0.0)
    assert metrics["trough_deficit_decomposition_closure"] == pytest.approx(0.0)


def test_stage146_metrics_preserve_inherited_sample() -> None:
    depth, dominant, parent, complement = _profiles()
    metrics = trough_provenance_metrics(depth, dominant, parent, complement, 2.0)
    assert metrics["trough_profile_index"] == 2
    assert metrics["trough_depth"] == pytest.approx(2.0)
    assert metrics["complement_value"] == pytest.approx(complement[2])


def test_stage146_classifies_mixed_reinforcing_below_supermajority() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "channel_identity_closure": 0.0,
        "trough_deficit_decomposition_closure": 0.0,
        "complement_trough_deficit": 1.0,
        "maximum_single_channel_absolute_share": 0.74,
        "projected_contributions_reinforce": True,
    }
    assert classify_trough_provenance(metrics=metrics) == MIXED_REINFORCING


def test_stage146_classifies_single_channel_at_fixed_supermajority() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "channel_identity_closure": 0.0,
        "trough_deficit_decomposition_closure": 0.0,
        "complement_trough_deficit": 1.0,
        "maximum_single_channel_absolute_share": 0.75,
        "projected_contributions_reinforce": True,
    }
    assert classify_trough_provenance(metrics=metrics) == SINGLE_CHANNEL


def test_stage146_classifies_opposed_curvature() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "channel_identity_closure": 0.0,
        "trough_deficit_decomposition_closure": 0.0,
        "complement_trough_deficit": 1.0,
        "maximum_single_channel_absolute_share": 0.80,
        "projected_contributions_reinforce": False,
    }
    assert classify_trough_provenance(metrics=metrics) == OPPOSED


def test_stage146_blocks_inherited_sample_mismatch() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "channel_identity_closure": 0.0,
        "trough_deficit_decomposition_closure": 0.0,
        "complement_trough_deficit": 1.0,
        "maximum_single_channel_absolute_share": 0.5,
        "projected_contributions_reinforce": True,
    }
    assert classify_trough_provenance(metrics=metrics, inherited_value_match_error=1e-6) == PROVENANCE_BLOCKER


def test_stage146_blocks_channel_identity_failure() -> None:
    metrics = {
        "depth_match_error": 0.0,
        "channel_identity_closure": 1e-6,
        "trough_deficit_decomposition_closure": 0.0,
        "complement_trough_deficit": 1.0,
        "maximum_single_channel_absolute_share": 0.5,
        "projected_contributions_reinforce": True,
    }
    assert classify_trough_provenance(metrics=metrics) == IDENTITY_BLOCKER


def test_stage146_rejects_nonmonotone_depth() -> None:
    depth, dominant, parent, complement = _profiles()
    depth[2] = depth[1]
    with pytest.raises(ValueError, match="strictly increasing"):
        trough_provenance_metrics(depth, dominant, parent, complement, 2.0)
