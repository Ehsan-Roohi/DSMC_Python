from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage135_fixed_lobe_amplitude_origin_audit import (
    LEFT_LOBE,
    MIXED_LOBES,
    NONFINITE,
    PARENT_CLOSURE_BLOCKER,
    RIGHT_LOBE,
    SHIFT_CLOSURE_BLOCKER,
    classify_lobe_origin,
    lobe_mismatch_metrics,
    validate_stage135_design,
)


def _block(*, left_share=0.2, right_share=0.8, left_coh=0.5, right_coh=1.0, left_mean=0.01, right_mean=-0.06):
    return {
        "left_sample_count": 4,
        "right_sample_count": 4,
        "left_mean_shift_wall0_minus_wall1": left_mean,
        "right_mean_shift_wall0_minus_wall1": right_mean,
        "left_l1_mismatch": left_share,
        "right_l1_mismatch": right_share,
        "left_l1_share": left_share,
        "right_l1_share": right_share,
        "left_sign_coherence": left_coh,
        "right_sign_coherence": right_coh,
        "left_constant_shift_relative_l2_residual": 0.4,
        "right_constant_shift_relative_l2_residual": 0.3,
        "left_uniform_sign": False,
        "right_uniform_sign": True,
    }


def test_validate_frozen_design_accepts_exact_values():
    validate_stage135_design()


def test_validate_rejects_grid_retuning():
    with pytest.raises(ValueError):
        validate_stage135_design(grid=(65, 64))


def test_validate_rejects_knudsen_extension():
    with pytest.raises(ValueError):
        validate_stage135_design(kn0=0.1)


def test_lobe_metrics_detect_right_tail_dominance():
    x = np.array([-7.0, -6.0, -5.0, 0.0, 5.0, 6.0, 7.0])
    wall1 = np.zeros_like(x)
    wall0 = np.array([0.01, -0.01, 0.01, 0.0, -0.05, -0.06, -0.07])
    m = lobe_mismatch_metrics(x, wall0, wall1)
    assert m["right_l1_share"] > 0.80
    assert m["right_sign_coherence"] == pytest.approx(1.0)
    assert m["right_mean_shift_wall0_minus_wall1"] < 0.0


def test_lobe_metrics_sign_coherence_penalizes_cancellation():
    x = np.array([-6.0, -5.0, 5.0, 6.0])
    wall1 = np.zeros_like(x)
    wall0 = np.array([1.0, -1.0, -2.0, 2.0])
    m = lobe_mismatch_metrics(x, wall0, wall1)
    assert m["left_sign_coherence"] == pytest.approx(0.0)
    assert m["right_sign_coherence"] == pytest.approx(0.0)


def test_lobe_metrics_reject_empty_fixed_support():
    x = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        lobe_mismatch_metrics(x, np.zeros(3), np.zeros(3))


def test_classify_right_negative_lobe():
    d = _block()
    p = _block(right_share=0.75, left_share=0.25, right_coh=0.98)
    assert classify_lobe_origin(dominant=d, parent=p) == RIGHT_LOBE


def test_classify_left_positive_lobe():
    d = _block(left_share=0.8, right_share=0.2, left_coh=1.0, right_coh=0.5, left_mean=0.05, right_mean=-0.01)
    p = _block(left_share=0.74, right_share=0.26, left_coh=0.99, right_coh=0.5, left_mean=0.04, right_mean=-0.01)
    assert classify_lobe_origin(dominant=d, parent=p) == LEFT_LOBE


def test_classify_mixed_when_share_guard_not_met():
    d = _block(right_share=0.69, left_share=0.31)
    p = _block(right_share=0.8, left_share=0.2)
    assert classify_lobe_origin(dominant=d, parent=p) == MIXED_LOBES


def test_classify_nonfinite_blocker():
    d = _block()
    p = _block()
    assert classify_lobe_origin(dominant=d, parent=p, finite=False) == NONFINITE


def test_classify_parent_closure_blocker():
    d = _block()
    p = _block()
    assert classify_lobe_origin(dominant=d, parent=p, parent_closure=2e-12) == PARENT_CLOSURE_BLOCKER


def test_classify_shift_closure_blocker():
    d = _block()
    p = _block()
    assert classify_lobe_origin(dominant=d, parent=p, shift_closure=2e-12) == SHIFT_CLOSURE_BLOCKER

# Explicit launch-touch: no scientific guard or numerical parameter is changed.
