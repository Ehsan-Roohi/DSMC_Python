import numpy as np
import pytest

from vgdsmc.stage132_crossing_phase_transition_width_audit import (
    CLOSURE_BLOCKER,
    COMMON,
    NONFINITE,
    PHASE,
    WIDTH,
    crossing_nearest_zero,
    classify_phase_width,
    transition_signature,
    validate_stage132_design,
)


def _common_kwargs():
    return dict(
        dominant_zero_phase_offset_cells=0.2,
        dominant_midpoint_phase_offset_cells=0.3,
        dominant_width_25_75_ratio=1.05,
        dominant_width_10_90_ratio=1.10,
        parent_zero_phase_offset_cells=0.1,
        parent_midpoint_phase_offset_cells=0.2,
        parent_width_25_75_ratio=1.08,
        parent_width_10_90_ratio=1.12,
        finite=True,
        closure=1e-15,
    )


def test_frozen_design_accepts_exact_defaults():
    validate_stage132_design()


def test_frozen_design_rejects_retuning():
    with pytest.raises(ValueError):
        validate_stage132_design(kn0=0.1)
    with pytest.raises(ValueError):
        validate_stage132_design(width_ratio_max=1.5)


def test_crossing_nearest_zero_linear_interpolation():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([2.0, 0.5, -1.0])
    assert crossing_nearest_zero(x, y) == pytest.approx(1.0 / 3.0)


def test_transition_signature_recovers_ordered_widths():
    x = np.arange(-10.0, 11.0)
    y = -np.tanh(x / 3.0)
    sig = transition_signature(x, y)
    assert abs(sig["zero_crossing_cells"]) < 1e-12
    assert sig["width_25_75_cells"] > 0.0
    assert sig["width_10_90_cells"] > sig["width_25_75_cells"]
    assert np.all(np.diff(sig["quantile_depths"]) > 0.0)


def test_classify_common_phase_width():
    assert classify_phase_width(**_common_kwargs()) == COMMON


def test_classify_phase_mismatch():
    kw = _common_kwargs()
    kw["dominant_zero_phase_offset_cells"] = 0.51
    assert classify_phase_width(**kw) == PHASE


def test_classify_width_mismatch():
    kw = _common_kwargs()
    kw["parent_width_10_90_ratio"] = 1.16
    assert classify_phase_width(**kw) == WIDTH


def test_classify_nonfinite_blocker():
    kw = _common_kwargs()
    kw["finite"] = False
    assert classify_phase_width(**kw) == NONFINITE


def test_classify_closure_blocker():
    kw = _common_kwargs()
    kw["closure"] = 2e-12
    assert classify_phase_width(**kw) == CLOSURE_BLOCKER
