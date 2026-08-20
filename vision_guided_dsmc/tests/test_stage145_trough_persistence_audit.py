from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage145_trough_persistence_audit import (
    ISOLATED,
    PERSISTENT,
    classify_trough_persistence,
    trough_persistence_metrics,
    validate_stage145_design,
)


def test_stage145_frozen_design_accepts_defaults() -> None:
    validate_stage145_design()


def test_stage145_rejects_retuning() -> None:
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage145_design(kn0=9.0)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage145_design(limiter="vanleer")
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage145_design(material_depression_min=0.20)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage145_design(physical_parameter_retuning=True)


def test_stage145_metrics_identify_isolated_trough() -> None:
    depth = np.arange(4.0)
    values = np.array([1.0, 0.5, 1.0, 1.0])
    metrics, mask = trough_persistence_metrics(depth, values, 1)
    assert metrics["trough_relative_depression_to_nontrough_median"] == pytest.approx(0.5)
    assert metrics["neighbor_material_depression_count"] == 0
    assert metrics["contiguous_material_support_count"] == 1
    assert metrics["trough_relative_deficit_to_local_neighbor_secant"] == pytest.approx(0.5)
    assert mask.tolist() == [0, 1, 0, 0]


def test_stage145_classifies_isolated_trough() -> None:
    assert classify_trough_persistence(
        trough_relative_depression=0.50,
        contiguous_support_count=1,
    ) == ISOLATED


def test_stage145_classifies_persistent_trough() -> None:
    assert classify_trough_persistence(
        trough_relative_depression=0.50,
        contiguous_support_count=2,
    ) == PERSISTENT


def test_stage145_blocks_parent_metric_mismatch() -> None:
    decision = classify_trough_persistence(
        trough_relative_depression=0.50,
        contiguous_support_count=1,
        parent_metric_closure=1.0e-6,
    )
    assert "parent_metric_closure_blocker" in decision


def test_stage145_blocks_missing_neighbor_support() -> None:
    decision = classify_trough_persistence(
        trough_relative_depression=0.50,
        contiguous_support_count=1,
        has_neighbor_support=False,
    )
    assert "neighbor_support_blocker" in decision
