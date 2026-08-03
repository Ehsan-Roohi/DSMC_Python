from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_maxwellian,
)
from vgdsmc.stage56_conservative_projection_pilot import (
    STAGE56_BOUND_TOLERANCE,
    _retained_clipped_lower_bounds,
)
from vgdsmc.stage57_conservative_solver_pilot import (
    STAGE56_COMPLETED_ENDPOINT,
    STAGE57_GRID,
    STAGE57_KNUDSEN,
    STAGE57_RADIAL_SCALE,
    STAGE57_RULE,
    build_stage57_config,
    compare_arms,
    conservative_projected_shakhov_equilibrium,
    stage57_decision,
    validate_stage57_design,
)


def _result(
    *,
    q_error: float = 0.20,
    velocity_rms: float = 0.40,
    sign: float = 1.0,
    converged: bool = True,
    finite: bool = True,
) -> dict[str, object]:
    return {
        "iterations": 500,
        "converged": converged,
        "final_change": 1.0e-6,
        "predicted_qav": 0.15,
        "literature_qav": 0.178,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": velocity_rms,
            "relative_l1": velocity_rms,
            "sign_agreement": sign,
        },
        "wall_mass_balance_relative_error": 1.0e-15,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "finite": finite,
        "work_proxy": 1,
        "table_velocity": [1.0e-3] * 10,
    }


def _projection(**changes) -> dict[str, float]:
    row = {
        "projection_success_fraction": 1.0,
        "maximum_conserved_moment_defect": 1.0e-12,
        "heat_flux_closure_relative_l2": 1.0e-12,
        "maximum_floor_violation": 0.0,
        "maximum_active_fraction": 0.05,
        "maximum_weighted_relative_modification": 0.20,
        "maximum_projection_iterations": 3.0,
        "rank_loss_count": 0.0,
        "roundoff_floor_clamp_count": 0.0,
        "maximum_phi_clipped_weight_fraction": 0.01,
        "maximum_psi_clipped_weight_fraction": 0.02,
    }
    row.update(changes)
    return row


def test_stage56_completed_endpoint_is_exact() -> None:
    assert STAGE56_COMPLETED_ENDPOINT["workflow_run_id"] == 30838743466
    assert STAGE56_COMPLETED_ENDPOINT["workflow_job_id"] == 91770307142
    assert STAGE56_COMPLETED_ENDPOINT["tests_passed"] == 113
    assert STAGE56_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE56_COMPLETED_ENDPOINT["artifact_id"] == 8870328780
    assert STAGE56_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "eb16133a3288e652b15986fdf9ae6c1738499a030509c06ac9cd785a3ca45946"
    )
    assert STAGE56_COMPLETED_ENDPOINT["decision"] == (
        "conservative_positive_projection_closes_frozen_fields_"
        "stage57_single_case_solver_pilot"
    )


def test_stage57_design_is_frozen_without_failed_parameter_retuning() -> None:
    validate_stage57_design()
    cfg = build_stage57_config()
    assert STAGE57_GRID == (16, 16)
    assert STAGE57_KNUDSEN == 10.0
    assert STAGE57_RULE == (40, 96)
    assert STAGE57_RADIAL_SCALE == 2.0
    assert cfg.positivity_floor == 1.0e-30
    assert cfg.prandtl == 2.0 / 3.0


def test_stage57_design_rejects_correction_floor_retuning() -> None:
    with pytest.raises(ValueError, match="Stage 57 is frozen"):
        validate_stage57_design(correction_floor=0.01)


def test_conservative_equilibrium_closes_and_preserves_retained_lower_bound() -> None:
    quadrature = mapped_polar_quadrature(12, 32, radial_scale=2.0)
    fields = {
        "rho": np.array([[1.0, 0.9], [1.1, 1.0]]),
        "u": np.array([[0.02, -0.01], [0.01, 0.0]]),
        "v": np.array([[-0.01, 0.02], [0.0, 0.01]]),
        "T": np.array([[0.5, 0.4], [0.6, 0.3]]),
        "qx": np.array([[1.0, -0.8], [0.6, -0.5]]),
        "qy": np.array([[0.5, 0.3], [-0.4, 0.2]]),
    }
    phi, psi, diagnostics = conservative_projected_shakhov_equilibrium(
        fields, quadrature
    )
    assert phi.shape == (2, 2, quadrature.point_count)
    assert psi.shape == phi.shape
    assert np.isfinite(phi).all()
    assert np.isfinite(psi).all()
    assert np.min(phi) >= 0.0
    assert np.min(psi) >= 0.0

    phi_maxwellian, psi_maxwellian = projected_maxwellian(
        fields["rho"], fields["u"], fields["v"], fields["T"], quadrature
    )
    for i in range(2):
        for j in range(2):
            phi_lower, psi_lower = _retained_clipped_lower_bounds(
                float(fields["rho"][i, j]),
                float(fields["u"][i, j]),
                float(fields["v"][i, j]),
                float(fields["T"][i, j]),
                float(fields["qx"][i, j]),
                float(fields["qy"][i, j]),
                phi_maxwellian[i, j],
                psi_maxwellian[i, j],
                quadrature,
            )
            phi_tolerance = STAGE56_BOUND_TOLERANCE * max(
                float(np.max(phi_lower)), 1.0e-300
            )
            psi_tolerance = STAGE56_BOUND_TOLERANCE * max(
                float(np.max(psi_lower)), 1.0e-300
            )
            assert np.all(phi[i, j] >= phi_lower - phi_tolerance)
            assert np.all(psi[i, j] >= psi_lower - psi_tolerance)

    assert diagnostics["projection_success_fraction"] == 1.0
    assert diagnostics["rank_loss_count"] == 0.0
    assert diagnostics["maximum_floor_violation"] == 0.0
    assert diagnostics["maximum_conserved_moment_defect"] < 1.0e-9
    assert diagnostics["heat_flux_closure_relative_l2"] < 1.0e-9
    assert STAGE41_CORRECTION_FLOOR == 0.05


def test_compare_arms_retains_positive_and_negative_changes() -> None:
    baseline = _result(q_error=0.20, velocity_rms=0.40)
    conservative = _result(q_error=0.18, velocity_rms=0.44)
    conservative["predicted_qav"] = 0.16
    comparison = compare_arms(baseline, conservative)
    assert comparison["heat_flux_error_improves"] is True
    assert comparison["velocity_rms_improves"] is False
    assert comparison["qav_relative_change"] > 0.0
    assert comparison["velocity_rms_error_change_fraction"] > 0.0


def test_stage57_decision_advances_only_stable_converged_guarded_pilot() -> None:
    baseline = _result()
    conservative = _result(q_error=0.19, velocity_rms=0.39)
    comparison = compare_arms(baseline, conservative)
    assert stage57_decision(
        baseline, conservative, comparison, _projection()
    ) == (
        "stage57_conservative_solver_pilot_passes_"
        "stage58_frozen_64x64_confirmation"
    )


def test_stage57_decision_preserves_projection_failure_as_blocker() -> None:
    baseline = _result()
    conservative = _result()
    comparison = compare_arms(baseline, conservative)
    assert stage57_decision(
        baseline,
        conservative,
        comparison,
        _projection(rank_loss_count=1.0),
    ) == "stage57_conservative_projection_in_solver_blocker_requires_review"


def test_stage57_decision_does_not_retune_stable_nonconvergence() -> None:
    baseline = _result()
    conservative = _result(converged=False)
    comparison = compare_arms(baseline, conservative)
    assert stage57_decision(
        baseline, conservative, comparison, _projection()
    ) == "stage57_stable_nonconverged_blocker_without_parameter_retuning"


def test_stage57_decision_preserves_observable_degradation() -> None:
    baseline = _result(q_error=0.20, velocity_rms=0.40)
    conservative = _result(q_error=0.24, velocity_rms=0.40)
    comparison = compare_arms(baseline, conservative)
    assert stage57_decision(
        baseline, conservative, comparison, _projection()
    ) == (
        "stage57_conservative_solver_stable_but_observables_degrade_"
        "requires_review_before_full_resolution"
    )
