import numpy as np
import pytest

from vgdsmc.stage133_crossing_amplitude_scale_audit import (
    CLOSURE_BLOCKER,
    INSUFFICIENT,
    MULTIPLICATIVE,
    NONFINITE,
    OFFSET_DOMINATED,
    classify_amplitude_scale,
    positive_affine_fit,
    positive_scale_fit,
    validate_stage133_design,
)


def test_frozen_design_accepts_exact_defaults():
    validate_stage133_design()


def test_frozen_design_rejects_retuning_and_guard_changes():
    with pytest.raises(ValueError):
        validate_stage133_design(kn0=0.1)
    with pytest.raises(ValueError):
        validate_stage133_design(core_half_width_cells=5.0)


def test_positive_scale_fit_recovers_exact_scale():
    source = np.array([1.0, 2.0, 3.0, 4.0])
    target = 1.7 * source
    fit = positive_scale_fit(target, source)
    assert fit["scale"] == pytest.approx(1.7)
    assert fit["residual_relative_l2"] < 1e-14
    assert fit["gain_fraction"] == pytest.approx(1.0)


def test_positive_affine_fit_recovers_exact_scale_and_offset():
    source = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    target = 1.25 * source - 0.3
    fit = positive_affine_fit(target, source)
    assert fit["scale"] == pytest.approx(1.25)
    assert fit["offset"] == pytest.approx(-0.3)
    assert fit["residual_relative_l2"] < 1e-14


def test_positive_fit_rejects_nonpositive_scale():
    source = np.array([1.0, 2.0, 3.0])
    target = -source
    with pytest.raises(ValueError):
        positive_scale_fit(target, source)


def test_classify_nonfinite_blocker():
    assert classify_amplitude_scale(
        dominant_scale_gain=np.nan,
        parent_scale_gain=0.5,
        dominant_affine_gain=0.6,
        parent_affine_gain=0.6,
        dominant_affine_residual=0.1,
        parent_affine_residual=0.1,
        finite=False,
        closure=0.0,
    ) == NONFINITE


def test_classify_parent_closure_blocker():
    assert classify_amplitude_scale(
        dominant_scale_gain=0.5,
        parent_scale_gain=0.5,
        dominant_affine_gain=0.6,
        parent_affine_gain=0.6,
        dominant_affine_residual=0.1,
        parent_affine_residual=0.1,
        closure=2e-12,
    ) == CLOSURE_BLOCKER


def test_classify_multiplicative_amplitude_case():
    assert classify_amplitude_scale(
        dominant_scale_gain=0.30,
        parent_scale_gain=0.25,
        dominant_affine_gain=0.50,
        parent_affine_gain=0.50,
        dominant_affine_residual=0.10,
        parent_affine_residual=0.10,
    ) == MULTIPLICATIVE


def test_classify_offset_dominated_case():
    assert classify_amplitude_scale(
        dominant_scale_gain=0.05,
        parent_scale_gain=0.10,
        dominant_affine_gain=0.50,
        parent_affine_gain=0.45,
        dominant_affine_residual=0.10,
        parent_affine_residual=0.11,
    ) == OFFSET_DOMINATED


def test_classify_amplitude_scaling_insufficient():
    assert classify_amplitude_scale(
        dominant_scale_gain=0.02,
        parent_scale_gain=0.03,
        dominant_affine_gain=0.20,
        parent_affine_gain=0.25,
        dominant_affine_residual=0.22,
        parent_affine_residual=0.20,
    ) == INSUFFICIENT
