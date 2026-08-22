from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage136_right_lobe_depth_support_audit import (
    DISTRIBUTED_COMMON,
    DISTRIBUTED_MIXED,
    NEAR_LOCALIZED,
    NONFINITE,
    PARENT_CLOSURE_BLOCKER,
    PARENT_RECORD_BLOCKER,
    classify_depth_support,
    depth_support_metrics,
    profile_cosine,
    validate_stage136_design,
)


def _block(*, near=0.60, effective=5.5, coherence=1.0):
    return {
        "sample_count": 7,
        "l1_mismatch": 1.0,
        "sign_coherence": coherence,
        "uniform_negative_sign": True,
        "near_l1_share": near,
        "far_l1_share": 0.25,
        "max_single_sample_share": 0.25,
        "effective_sample_count": effective,
        "weighted_centroid_depth_cells": 8.0,
        "weighted_spread_cells": 1.8,
        "half_l1_depth_cells": 8.0,
        "endpoint_to_nearest_magnitude_ratio": 0.2,
        "nonincreasing_step_fraction": 1.0,
    }


def test_validate_frozen_design_accepts_exact_values():
    validate_stage136_design()


def test_validate_rejects_knudsen_extension():
    with pytest.raises(ValueError):
        validate_stage136_design(kn0=0.1)


def test_validate_rejects_support_redefinition():
    with pytest.raises(ValueError):
        validate_stage136_design(right_support_min_depth=4.0)


def test_depth_support_metrics_detect_broad_negative_tail():
    x = np.arange(4.0, 12.0)
    diff = -np.array([0.01, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04])
    mask = x >= 5.0
    m = depth_support_metrics(x, diff, mask)
    assert m["sample_count"] == 7
    assert m["uniform_negative_sign"] is True
    assert m["sign_coherence"] == pytest.approx(1.0)
    assert m["effective_sample_count"] > 4.5
    assert m["near_l1_share"] < 0.75


def test_depth_support_metrics_rejects_short_support():
    x = np.arange(5.0, 10.0)
    with pytest.raises(ValueError):
        depth_support_metrics(x, -np.ones(5), np.ones(5, dtype=bool))


def test_profile_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert profile_cosine(a, a) == pytest.approx(1.0)


def test_classify_distributed_common_support():
    assert classify_depth_support(
        dominant=_block(near=0.54, effective=6.5),
        parent=_block(near=0.66, effective=5.5),
        common_profile_cosine=0.97,
    ) == DISTRIBUTED_COMMON


def test_classify_near_localized_support():
    assert classify_depth_support(
        dominant=_block(near=0.80),
        parent=_block(near=0.76),
        common_profile_cosine=0.90,
    ) == NEAR_LOCALIZED


def test_classify_distributed_mixed_when_profile_not_common():
    assert classify_depth_support(
        dominant=_block(near=0.55, effective=6.0),
        parent=_block(near=0.65, effective=5.0),
        common_profile_cosine=0.90,
    ) == DISTRIBUTED_MIXED


def test_classify_distributed_mixed_when_support_not_broad():
    assert classify_depth_support(
        dominant=_block(near=0.55, effective=4.0),
        parent=_block(near=0.65, effective=5.0),
        common_profile_cosine=0.99,
    ) == DISTRIBUTED_MIXED


def test_classify_parent_record_blocker():
    assert classify_depth_support(
        dominant=_block(), parent=_block(), common_profile_cosine=0.99, parent_record_ok=False
    ) == PARENT_RECORD_BLOCKER


def test_classify_parent_closure_and_nonfinite_blockers():
    assert classify_depth_support(
        dominant=_block(), parent=_block(), common_profile_cosine=0.99, parent_closure=2e-12
    ) == PARENT_CLOSURE_BLOCKER
    assert classify_depth_support(
        dominant=_block(), parent=_block(), common_profile_cosine=0.99, finite=False
    ) == NONFINITE
