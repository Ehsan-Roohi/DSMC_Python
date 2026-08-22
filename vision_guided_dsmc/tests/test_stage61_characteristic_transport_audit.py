from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig
from vgdsmc.stage41_projected_polar_operator_audit import mapped_polar_quadrature
from vgdsmc.stage61_characteristic_transport_audit import (
    STAGE60_COMPLETED_ENDPOINT,
    STAGE61_COLD_HOT_RATIO,
    STAGE61_DISCRETE_RESIDUAL_TOLERANCE,
    STAGE61_GRIDS,
    STAGE61_KNUDSEN_SCOPE,
    STAGE61_MATERIAL_ERROR_THRESHOLD,
    STAGE61_RADIAL_SCALE,
    STAGE61_RULE,
    _inflow_profiles,
    build_characteristic_wall_operator,
    first_order_residual_relative_error,
    solve_first_order_upwind_with_fixed_inflow,
    trace_back_to_wall_faces,
    validate_stage60_artifact,
    validate_stage61_design,
)


def test_stage61_design_is_frozen() -> None:
    validate_stage61_design(
        STAGE61_GRIDS,
        STAGE61_RULE,
        STAGE61_RADIAL_SCALE,
        STAGE61_KNUDSEN_SCOPE,
        STAGE61_COLD_HOT_RATIO,
        STAGE61_MATERIAL_ERROR_THRESHOLD,
    )
    with pytest.raises(ValueError):
        validate_stage61_design(
            ((8, 8), (16, 16), (64, 64)),
            STAGE61_RULE,
            STAGE61_RADIAL_SCALE,
            STAGE61_KNUDSEN_SCOPE,
            STAGE61_COLD_HOT_RATIO,
            STAGE61_MATERIAL_ERROR_THRESHOLD,
        )
    with pytest.raises(ValueError):
        validate_stage61_design(
            STAGE61_GRIDS,
            (32, 96),
            STAGE61_RADIAL_SCALE,
            STAGE61_KNUDSEN_SCOPE,
            STAGE61_COLD_HOT_RATIO,
            STAGE61_MATERIAL_ERROR_THRESHOLD,
        )
    with pytest.raises(ValueError):
        validate_stage61_design(
            STAGE61_GRIDS,
            STAGE61_RULE,
            STAGE61_RADIAL_SCALE,
            STAGE61_KNUDSEN_SCOPE,
            STAGE61_COLD_HOT_RATIO,
            0.05,
        )


def test_characteristics_from_center_reach_expected_wall_faces() -> None:
    vx = np.asarray([1.0, -1.0, 0.0, 0.0])
    vy = np.asarray([0.0, 0.0, 1.0, -1.0])
    source_a, source_b, blend = trace_back_to_wall_faces(
        0.5, 0.5, vx, vy, nx=4, ny=4
    )
    assert source_a.tolist() == [2, 6, 10, 14]
    assert np.array_equal(source_a, source_b)
    assert np.all(blend == 0.0)


def test_characteristic_wall_operator_has_unit_positive_fixed_point() -> None:
    quadrature = mapped_polar_quadrature(8, 16, 2.0)
    cfg = LinearSidewallConfig(nx=4, ny=4, kn0=10.0, cold_hot_ratio=0.1)
    operator = build_characteristic_wall_operator(cfg, quadrature)
    assert operator["dominant_eigenvalue_defect"] <= 1.0e-10
    assert operator["eigen_residual"] <= 1.0e-10
    assert np.min(operator["alpha"]) > 0.0
    assert np.all(np.isfinite(operator["transfer"]))


def test_causal_upwind_sweep_solves_algebraic_equation() -> None:
    quadrature = mapped_polar_quadrature(8, 16, 2.0)
    cfg = LinearSidewallConfig(nx=4, ny=4, kn0=10.0, cold_hot_ratio=0.1)
    operator = build_characteristic_wall_operator(cfg, quadrature)
    inflow = _inflow_profiles(cfg, operator)
    phi, psi = solve_first_order_upwind_with_fixed_inflow(cfg, quadrature, inflow)
    (
        left_phi, left_psi, right_phi, right_psi,
        bottom_phi, bottom_psi, top_phi, top_psi,
    ) = inflow
    assert first_order_residual_relative_error(
        phi, cfg, quadrature, left_phi, right_phi, bottom_phi, top_phi
    ) <= STAGE61_DISCRETE_RESIDUAL_TOLERANCE
    assert first_order_residual_relative_error(
        psi, cfg, quadrature, left_psi, right_psi, bottom_psi, top_psi
    ) <= STAGE61_DISCRETE_RESIDUAL_TOLERANCE
    assert np.all(phi >= 0.0)
    assert np.all(psi >= 0.0)


def test_stage60_provenance_is_pinned() -> None:
    assert STAGE60_COMPLETED_ENDPOINT["workflow_run_id"] == 30897895198
    assert STAGE60_COMPLETED_ENDPOINT["workflow_job_id"] == 91955054751
    assert STAGE60_COMPLETED_ENDPOINT["workflow_conclusion"] == "success"
    assert STAGE60_COMPLETED_ENDPOINT["artifact_id"] == 8891954081
    assert STAGE60_COMPLETED_ENDPOINT["summary_sha256"] == (
        "38201cc72f824c27b588bdc3c2b7a82973d2a0de886e7b07ebcd02f4af1790a3"
    )


def test_stage60_artifact_validation_rejects_missing_or_modified_summary(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        validate_stage60_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps({"stage": 60, "decision": STAGE60_COMPLETED_ENDPOINT["decision"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_stage60_artifact(tmp_path)
