from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage141_node_leverage_audit import (
    LOW_LEVERAGE,
    MATERIAL_LEVERAGE,
    MIXED_LEVERAGE,
    classify_node_leverage,
    linear_root_and_sensitivities,
    quadratic_root_if_bracketed,
    validate_stage141_design,
)


def test_stage141_frozen_design_accepts_only_registered_values():
    validate_stage141_design()
    with pytest.raises(ValueError):
        validate_stage141_design(kn0=9.0)
    with pytest.raises(ValueError):
        validate_stage141_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage141_design(physical_parameter_retuning=True)


def test_linear_root_and_sensitivity_match_closed_form():
    depth = np.array([0.0, 1.0, 2.0])
    values = np.array([-1.0, -4.0, 1.0])
    root, lower_s, upper_s = linear_root_and_sensitivities(depth, values, 1)
    assert root == pytest.approx(1.8)
    assert lower_s == pytest.approx(1.0 / 25.0)
    assert upper_s == pytest.approx(4.0 / 25.0)
    assert upper_s / lower_s == pytest.approx(4.0)


def test_quadratic_root_if_bracketed_can_report_missing_root():
    depth = np.arange(7, dtype=float)
    values = np.array([-2.0, -1.0, 0.1, 1.0, 2.0, 3.0, 4.0])
    synthetic = np.array([-2.0, -1.0, 0.1, -0.8, 1.0, 2.0, 3.0])
    missing = quadratic_root_if_bracketed(depth, synthetic, (0, 1, 3), 1)
    assert missing is None
    retained = quadratic_root_if_bracketed(depth, values, (1, 3, 4), 1)
    assert retained is not None
    assert 1.0 <= retained <= 2.0


def test_material_leverage_route_requires_small_sensitive_fragile_endpoint():
    decision = classify_node_leverage(
        upper_endpoint_abs_ratio_to_lower=0.15,
        upper_to_lower_raw_sensitivity_ratio=6.0,
        upper_deletion_root_retention_fraction=0.5,
        maximum_retained_upper_deletion_root_shift_cells=0.12,
    )
    assert decision == MATERIAL_LEVERAGE


def test_low_leverage_route_requires_both_deletion_bridges_to_survive():
    decision = classify_node_leverage(
        upper_endpoint_abs_ratio_to_lower=0.20,
        upper_to_lower_raw_sensitivity_ratio=1.5,
        upper_deletion_root_retention_fraction=1.0,
        maximum_retained_upper_deletion_root_shift_cells=0.10,
    )
    assert decision == LOW_LEVERAGE


def test_mixed_route_for_incomplete_nonmaterial_case():
    decision = classify_node_leverage(
        upper_endpoint_abs_ratio_to_lower=0.40,
        upper_to_lower_raw_sensitivity_ratio=1.2,
        upper_deletion_root_retention_fraction=0.5,
        maximum_retained_upper_deletion_root_shift_cells=0.10,
    )
    assert decision == MIXED_LEVERAGE
