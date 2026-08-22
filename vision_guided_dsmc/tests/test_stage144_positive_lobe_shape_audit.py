from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage144_positive_lobe_shape_audit import (
    DISTRIBUTED,
    SINGLE_PEAK,
    SINGLE_TROUGH,
    classify_positive_lobe_shape,
    positive_lobe_shape_metrics,
    validate_stage144_design,
)


def test_stage144_frozen_design_accepts_defaults() -> None:
    validate_stage144_design()


def test_stage144_rejects_retuning() -> None:
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage144_design(kn0=9.0)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage144_design(limiter="vanleer")
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage144_design(source_relaxation=0.5)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage144_design(physical_parameter_retuning=True)


def test_stage144_metrics_identify_single_low_sample() -> None:
    metrics, normalized, fractions, loo_cv = positive_lobe_shape_metrics(np.array([1.0, 0.5, 1.0, 1.0]))
    assert metrics["dominant_sample_index"] == 1
    assert metrics["dominant_relative_deviation"] == pytest.approx(-0.5)
    assert metrics["dominant_deviation_energy_share"] == pytest.approx(1.0)
    assert metrics["leave_dominant_out_coefficient_of_variation"] == pytest.approx(0.0)
    assert normalized.shape == (4,)
    assert fractions.shape == (4,)
    assert loo_cv.shape == (4,)


def test_stage144_classifies_material_single_trough() -> None:
    assert classify_positive_lobe_shape(
        dominant_energy_share=0.90,
        leave_one_out_cv_reduction=0.80,
        dominant_relative_deviation=-0.40,
        sample_count=4,
    ) == SINGLE_TROUGH


def test_stage144_classifies_material_single_peak() -> None:
    assert classify_positive_lobe_shape(
        dominant_energy_share=0.90,
        leave_one_out_cv_reduction=0.80,
        dominant_relative_deviation=0.40,
        sample_count=4,
    ) == SINGLE_PEAK


def test_stage144_requires_all_materiality_guards() -> None:
    assert classify_positive_lobe_shape(
        dominant_energy_share=0.90,
        leave_one_out_cv_reduction=0.80,
        dominant_relative_deviation=-0.10,
        sample_count=4,
    ) == DISTRIBUTED
    assert classify_positive_lobe_shape(
        dominant_energy_share=0.60,
        leave_one_out_cv_reduction=0.80,
        dominant_relative_deviation=-0.40,
        sample_count=4,
    ) == DISTRIBUTED
    assert classify_positive_lobe_shape(
        dominant_energy_share=0.90,
        leave_one_out_cv_reduction=0.20,
        dominant_relative_deviation=-0.40,
        sample_count=4,
    ) == DISTRIBUTED


def test_stage144_blocks_parent_metric_mismatch() -> None:
    decision = classify_positive_lobe_shape(
        dominant_energy_share=0.90,
        leave_one_out_cv_reduction=0.80,
        dominant_relative_deviation=-0.40,
        sample_count=4,
        parent_metric_closure=1.0e-6,
    )
    assert "parent_metric_closure_blocker" in decision


def test_stage144_blocks_insufficient_support() -> None:
    decision = classify_positive_lobe_shape(
        dominant_energy_share=0.90,
        leave_one_out_cv_reduction=0.80,
        dominant_relative_deviation=-0.40,
        sample_count=3,
    )
    assert "insufficient_positive_lobe_support" in decision
