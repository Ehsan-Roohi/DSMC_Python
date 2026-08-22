from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage143_positive_lobe_scale_audit import (
    BROAD_COHERENT,
    CONCENTRATED,
    VARIABLE,
    classify_positive_lobe_scale,
    effective_support_count,
    validate_stage143_design,
)


def test_stage143_frozen_design_accepts_defaults() -> None:
    validate_stage143_design()


def test_stage143_rejects_physical_retuning() -> None:
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage143_design(kn0=9.0)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage143_design(limiter="vanleer")
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage143_design(source_relaxation=0.5)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage143_design(physical_parameter_retuning=True)


def test_effective_support_count_is_broad_for_equal_samples() -> None:
    assert effective_support_count(np.ones(4)) == pytest.approx(4.0)


def test_stage143_classifies_broad_coherent_scale() -> None:
    assert classify_positive_lobe_scale(
        later_count=4,
        sign_coherence=1.0,
        coefficient_of_variation=0.25,
        effective_count=3.5,
        peak_to_median=1.5,
    ) == BROAD_COHERENT


def test_stage143_classifies_concentrated_scale() -> None:
    assert classify_positive_lobe_scale(
        later_count=4,
        sign_coherence=1.0,
        coefficient_of_variation=0.8,
        effective_count=2.2,
        peak_to_median=2.5,
    ) == CONCENTRATED


def test_stage143_classifies_variable_scale_without_concentration() -> None:
    assert classify_positive_lobe_scale(
        later_count=4,
        sign_coherence=1.0,
        coefficient_of_variation=0.7,
        effective_count=3.2,
        peak_to_median=2.5,
    ) == VARIABLE


def test_stage143_blocks_insufficient_support() -> None:
    decision = classify_positive_lobe_scale(
        later_count=3,
        sign_coherence=1.0,
        coefficient_of_variation=0.1,
        effective_count=3.0,
        peak_to_median=1.1,
    )
    assert "insufficient_positive_lobe_support" in decision
