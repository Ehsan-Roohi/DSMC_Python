import numpy as np
import pytest

from vgdsmc.stage134_additive_baseline_origin_audit import (
    CLOSURE_BLOCKER,
    GLOBAL_BASELINE,
    LOBE_IMBALANCE,
    MIXED,
    NONFINITE,
    classify_baseline_origin,
    decompose_plateau_shift,
    validate_stage134_design,
)


def test_frozen_design_accepts_exact_defaults():
    validate_stage134_design()


def test_frozen_design_rejects_retuning_and_guard_changes():
    with pytest.raises(ValueError):
        validate_stage134_design(kn0=0.1)
    with pytest.raises(ValueError):
        validate_stage134_design(plateau_depth_cells=4.0)
    with pytest.raises(ValueError):
        validate_stage134_design(solver_rerun=True)


def test_even_odd_decomposition_closes_exactly():
    out = decompose_plateau_shift(0.01, -0.06)
    assert out["even_global_baseline_component"] == pytest.approx(-0.025)
    assert out["odd_lobe_imbalance_component"] == pytest.approx(0.035)
    assert out["even_global_baseline_component"] + out["odd_lobe_imbalance_component"] == pytest.approx(0.01)
    assert out["even_global_baseline_component"] - out["odd_lobe_imbalance_component"] == pytest.approx(-0.06)
    assert out["even_fraction"] + out["odd_fraction"] == pytest.approx(1.0)
    assert out["opposite_side_signs"] is True


def test_classify_opposite_side_lobe_imbalance():
    assert classify_baseline_origin(
        dominant_left_shift=0.01, dominant_right_shift=-0.06,
        parent_left_shift=0.006, parent_right_shift=-0.056,
        dominant_even_fraction=0.42, parent_even_fraction=0.44,
        dominant_odd_fraction=0.58, parent_odd_fraction=0.56,
        dominant_offset_even_relative_mismatch=0.5,
        parent_offset_even_relative_mismatch=0.5,
    ) == LOBE_IMBALANCE


def test_classify_global_baseline():
    assert classify_baseline_origin(
        dominant_left_shift=-0.05, dominant_right_shift=-0.04,
        parent_left_shift=-0.06, parent_right_shift=-0.05,
        dominant_even_fraction=0.90, parent_even_fraction=0.91,
        dominant_odd_fraction=0.10, parent_odd_fraction=0.09,
        dominant_offset_even_relative_mismatch=0.1,
        parent_offset_even_relative_mismatch=0.1,
    ) == GLOBAL_BASELINE


def test_classify_mixed_case():
    assert classify_baseline_origin(
        dominant_left_shift=0.02, dominant_right_shift=-0.01,
        parent_left_shift=-0.03, parent_right_shift=-0.02,
        dominant_even_fraction=0.3, parent_even_fraction=0.8,
        dominant_odd_fraction=0.7, parent_odd_fraction=0.2,
        dominant_offset_even_relative_mismatch=0.7,
        parent_offset_even_relative_mismatch=0.2,
    ) == MIXED


def test_classify_nonfinite_blocker():
    assert classify_baseline_origin(
        dominant_left_shift=np.nan, dominant_right_shift=-0.01,
        parent_left_shift=0.01, parent_right_shift=-0.01,
        dominant_even_fraction=0.4, parent_even_fraction=0.4,
        dominant_odd_fraction=0.6, parent_odd_fraction=0.6,
        dominant_offset_even_relative_mismatch=0.5,
        parent_offset_even_relative_mismatch=0.5,
        finite=False,
    ) == NONFINITE


def test_classify_parent_closure_blocker():
    assert classify_baseline_origin(
        dominant_left_shift=0.01, dominant_right_shift=-0.06,
        parent_left_shift=0.006, parent_right_shift=-0.056,
        dominant_even_fraction=0.42, parent_even_fraction=0.44,
        dominant_odd_fraction=0.58, parent_odd_fraction=0.56,
        dominant_offset_even_relative_mismatch=0.5,
        parent_offset_even_relative_mismatch=0.5,
        closure=2.0e-12,
    ) == CLOSURE_BLOCKER
