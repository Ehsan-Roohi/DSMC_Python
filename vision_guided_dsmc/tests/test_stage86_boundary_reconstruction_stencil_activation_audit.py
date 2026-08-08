from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage86_boundary_reconstruction_stencil_activation_audit import (
    BOUNDARY_UPWIND_FIRST_SHARE_GUARD,
    BOUNDARY_UPWIND_WALL_ABS_GUARD,
    PAIR_CLOSURE_GUARD,
    REMAINING_WALL_FIRST_RATIO_MIN,
    boundary_stencil_metrics,
    reconstruction_implementation_invariants,
    stage86_decision,
    validate_stage86_design,
)


def test_stage86_design_is_frozen() -> None:
    validate_stage86_design()


def test_stage86_rejects_design_changes() -> None:
    with pytest.raises(ValueError):
        validate_stage86_design(kn0=1.0)
    with pytest.raises(ValueError):
        validate_stage86_design(limiter="none")


def test_frozen_reconstruction_has_zero_boundary_slope_invariants() -> None:
    invariants = reconstruction_implementation_invariants()
    assert all(invariants.values())


def test_stage86_primary_route_requires_all_guards() -> None:
    decision = stage86_decision(
        True,
        True,
        PAIR_CLOSURE_GUARD / 10.0,
        BOUNDARY_UPWIND_WALL_ABS_GUARD / 10.0,
        BOUNDARY_UPWIND_FIRST_SHARE_GUARD + 0.01,
        REMAINING_WALL_FIRST_RATIO_MIN + 0.1,
    )
    assert decision.endswith("stage87_one_sided_boundary_slope_counterfactual_audit")


def test_stage86_keeps_partial_boundary_stencil_route_distinct() -> None:
    decision = stage86_decision(
        True,
        True,
        PAIR_CLOSURE_GUARD / 10.0,
        0.0,
        BOUNDARY_UPWIND_FIRST_SHARE_GUARD - 0.1,
        REMAINING_WALL_FIRST_RATIO_MIN + 0.1,
    )
    assert "contributes_but_not_dominant" in decision


def test_stage86_reports_integrity_blocker() -> None:
    assert "blocker" in stage86_decision(False, True, 0.0, 0.0, 1.0, 1.0)
    assert "blocker" in stage86_decision(True, False, 0.0, 0.0, 1.0, 1.0)
    assert "blocker" in stage86_decision(
        True, True, PAIR_CLOSURE_GUARD * 2.0, 0.0, 1.0, 1.0
    )


def test_boundary_stencil_metrics_reject_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        boundary_stencil_metrics(
            np.zeros((95, 64, 63)),
            np.zeros(96, dtype=np.int16),
            np.arange(96) * 3.75,
            np.zeros((2, 64, 63)),
            np.asarray([[1, 5], [2, 6]], dtype=np.int16),
        )


def test_boundary_stencil_metrics_closes_synthetic_pair_maps() -> None:
    angles = np.arange(96, dtype=np.float64) * 3.75
    bins = np.floor(np.mod(angles, 360.0) / 45.0).astype(np.int16)
    face = np.zeros((96, 64, 63), dtype=np.float64)
    cos_angle = np.cos(np.deg2rad(angles))
    positive = cos_angle > 1.0e-12
    negative = cos_angle < -1.0e-12
    selected = np.isin(bins, [1, 2, 5, 6])

    # Mimic only the structural property under audit: boundary-upwind contributions vanish
    # at the wall-adjacent face but are active one face inward; remaining directions stay active.
    for ordinate in np.flatnonzero(selected):
        if positive[ordinate]:
            face[ordinate, :, 1] = 2.0
            face[ordinate, :, -1] = 1.0
            face[ordinate, :, -2] = 1.0
        elif negative[ordinate]:
            face[ordinate, :, 0] = 1.0
            face[ordinate, :, 1] = 1.0
            face[ordinate, :, -2] = 2.0

    pairs = np.asarray([[1, 5], [2, 6]], dtype=np.int16)
    retained = np.stack([np.sum(face[np.isin(bins, pair)], axis=0) for pair in pairs], axis=0)
    metrics = boundary_stencil_metrics(face, bins, angles, retained, pairs)
    assert metrics["maximum_pair_reconstruction_relative_l2_error"] == 0.0
    assert metrics["maximum_boundary_upwind_wall_individual_absolute_value"] == 0.0
    assert metrics["finite"] is True
