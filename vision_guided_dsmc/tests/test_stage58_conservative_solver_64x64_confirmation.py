from __future__ import annotations

import pytest

from vgdsmc.stage57_conservative_solver_pilot import (
    STAGE57_KNUDSEN,
    STAGE57_RADIAL_SCALE,
    STAGE57_RULE,
)
from vgdsmc.stage58_conservative_solver_64x64_confirmation import (
    STAGE57_COMPLETED_ENDPOINT,
    STAGE58_GRID,
    STAGE58_KNUDSEN,
    STAGE58_RADIAL_SCALE,
    STAGE58_RULE,
    stage58_decision,
    validate_stage58_design,
)


def _arm(q_error: float, v_error: float, *, converged: bool = True) -> dict:
    return {
        "iterations": 500,
        "converged": converged,
        "final_change": 1.0e-6,
        "predicted_qav": 0.2,
        "literature_qav": 0.178,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": v_error,
            "relative_l1": v_error,
            "sign_agreement": 1.0,
        },
        "wall_mass_balance_relative_error": 1.0e-16,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "finite": True,
        "work_proxy": 1,
        "table_velocity": [1.0e-3] * 10,
    }


def _projection() -> dict[str, float]:
    return {
        "projection_success_fraction": 1.0,
        "maximum_conserved_moment_defect": 1.0e-13,
        "heat_flux_closure_relative_l2": 1.0e-13,
        "maximum_floor_violation": 0.0,
        "maximum_active_fraction": 0.01,
        "maximum_weighted_relative_modification": 0.01,
        "maximum_projection_iterations": 2.0,
        "rank_loss_count": 0.0,
        "roundoff_floor_clamp_count": 0.0,
        "maximum_phi_clipped_weight_fraction": 0.01,
        "maximum_psi_clipped_weight_fraction": 0.01,
    }


def test_stage58_design_is_frozen_to_preregistered_confirmation() -> None:
    validate_stage58_design()
    assert STAGE58_GRID == (64, 64)
    assert STAGE58_KNUDSEN == STAGE57_KNUDSEN == 10.0
    assert STAGE58_RULE == STAGE57_RULE == (40, 96)
    assert STAGE58_RADIAL_SCALE == STAGE57_RADIAL_SCALE == 2.0
    assert STAGE57_COMPLETED_ENDPOINT["workflow_run_id"] == 30859493733
    assert STAGE57_COMPLETED_ENDPOINT["artifact_id"] == 8875278804


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": (32, 32)},
        {"kn0": 1.0},
        {"rule": (48, 96)},
        {"radial_scale": 1.0},
        {"source_relaxation": 0.8},
        {"correction_floor": 0.01},
    ],
)
def test_stage58_rejects_retuning(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        validate_stage58_design(**kwargs)


def test_stage58_decision_passes_bounded_degradation() -> None:
    baseline = _arm(0.28, 0.62)
    conservative = _arm(0.29, 0.65)
    comparison = {
        "qav_error_change_fraction": (0.29 - 0.28) / 0.28,
        "velocity_rms_error_change_fraction": (0.65 - 0.62) / 0.62,
        "sign_agreement_change": 0.0,
        "qav_relative_change": 0.01,
        "velocity_profile_change": 0.02,
        "heat_flux_error_improves": False,
        "velocity_rms_improves": False,
    }
    assert stage58_decision(
        baseline, conservative, comparison, _projection()
    ) == (
        "stage58_frozen_64x64_confirmation_passes_"
        "independent_reference_review_before_any_extension"
    )


def test_stage58_decision_retains_observable_degradation() -> None:
    baseline = _arm(0.28, 0.62)
    conservative = _arm(0.34, 0.65)
    comparison = {
        "qav_error_change_fraction": (0.34 - 0.28) / 0.28,
        "velocity_rms_error_change_fraction": (0.65 - 0.62) / 0.62,
        "sign_agreement_change": 0.0,
        "qav_relative_change": 0.01,
        "velocity_profile_change": 0.02,
        "heat_flux_error_improves": False,
        "velocity_rms_improves": False,
    }
    assert "observables_degrade" in stage58_decision(
        baseline, conservative, comparison, _projection()
    )
