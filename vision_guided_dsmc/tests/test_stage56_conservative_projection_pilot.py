from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    mapped_polar_quadrature,
    projected_maxwellian,
    projected_shakhov_equilibrium,
)
from vgdsmc.stage56_conservative_projection_pilot import (
    STAGE55_COMPLETED_ENDPOINT,
    STAGE56_RULE,
    _linear_moments,
    _moment_basis,
    _retained_clipped_lower_bounds,
    _target_moments,
    bounded_conservative_projection,
    stage56_decision,
    validate_stage56_design,
)


def _row(
    success: float = 1.0,
    conserved: float = 1.0e-12,
    heat: float = 1.0e-12,
    floor: float = 0.0,
    active: float = 0.05,
    modification: float = 0.20,
) -> dict[str, float]:
    return {
        "projection_success_fraction": success,
        "maximum_conserved_moment_defect": conserved,
        "heat_flux_closure_relative_l2": heat,
        "maximum_floor_violation": floor,
        "maximum_active_fraction": active,
        "maximum_weighted_relative_modification": modification,
    }


def _audits(**changes) -> dict[str, dict[str, float]]:
    base = _row(**changes)
    return {
        "compressed_tail": base.copy(),
        "expanded_tail": base.copy(),
    }


def test_stage55_completed_endpoint_is_exact() -> None:
    assert STAGE55_COMPLETED_ENDPOINT["workflow_run_id"] == 30822272403
    assert STAGE55_COMPLETED_ENDPOINT["workflow_job_id"] == 91714898163
    assert STAGE55_COMPLETED_ENDPOINT["tests_passed"] == 105
    assert STAGE55_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE55_COMPLETED_ENDPOINT["artifact_id"] == 8865015655
    assert STAGE55_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "4a491f39b2c00bf96b01b950bbbea04fb9a792c054717369cb92c0ca3c1dba19"
    )
    assert STAGE55_COMPLETED_ENDPOINT["decision"] == (
        "radial_quadrature_closes_unclipped_formula_"
        "positivity_clipping_breaks_invariants_"
        "stage56_conservative_projection_pilot"
    )


def test_stage56_frozen_design_accepts_only_preregistered_values() -> None:
    validate_stage56_design()
    assert STAGE56_RULE == (40, 96)


def test_stage56_frozen_design_rejects_floor_retuning() -> None:
    with pytest.raises(ValueError, match="Stage 56 is frozen"):
        validate_stage56_design(correction_floor=0.01)


def test_bounded_projection_preserves_retained_floor_and_closes_moments() -> None:
    quadrature = mapped_polar_quadrature(12, 32, radial_scale=1.0)
    rho = 1.0
    u = 0.02
    v = -0.01
    temperature = 0.5
    qx = 1.0
    qy = 0.5
    fields = {
        "rho": np.asarray(rho),
        "u": np.asarray(u),
        "v": np.asarray(v),
        "T": np.asarray(temperature),
        "qx": np.asarray(qx),
        "qy": np.asarray(qy),
    }
    phi_reference, psi_reference, _ = projected_shakhov_equilibrium(
        fields,
        quadrature,
        correction_floor=STAGE41_CORRECTION_FLOOR,
    )
    phi_maxwellian, psi_maxwellian = projected_maxwellian(
        fields["rho"], fields["u"], fields["v"], fields["T"], quadrature
    )
    phi_lower, psi_lower = _retained_clipped_lower_bounds(
        rho,
        u,
        v,
        temperature,
        qx,
        qy,
        phi_maxwellian,
        psi_maxwellian,
        quadrature,
    )
    phi_basis, psi_basis = _moment_basis(quadrature.vx, quadrature.vy, u, v)
    target = _target_moments(rho, u, v, temperature, qx, qy)
    phi, psi, diagnostics = bounded_conservative_projection(
        phi_reference,
        psi_reference,
        phi_lower,
        psi_lower,
        phi_basis,
        psi_basis,
        quadrature.weight,
        target,
    )
    defect = _linear_moments(
        phi, psi, phi_basis, psi_basis, quadrature.weight
    ) - target

    assert diagnostics["converged"] is True
    assert diagnostics["linear_system_rank"] == 6
    assert diagnostics["active_fraction"] > 0.0
    assert np.all(phi >= phi_lower)
    assert np.all(psi >= psi_lower)
    assert np.max(np.abs(defect)) < 1.0e-9


def test_stage56_decision_advances_only_feasible_bounded_projection() -> None:
    assert stage56_decision(_audits()) == (
        "conservative_positive_projection_closes_frozen_fields_"
        "stage57_single_case_solver_pilot"
    )


def test_stage56_decision_retains_infeasible_projection_as_blocker() -> None:
    assert stage56_decision(_audits(success=0.99)) == (
        "conservative_projection_infeasible_blocker_requires_review"
    )


def test_stage56_decision_retains_large_deformation_for_review() -> None:
    assert stage56_decision(_audits(modification=0.31)) == (
        "conservative_projection_closes_but_large_deformation_"
        "requires_review_before_solver_rerun"
    )


def test_stage56_decision_retains_nonfinite_result_as_blocker() -> None:
    audits = _audits()
    audits["compressed_tail"]["maximum_conserved_moment_defect"] = float("nan")
    assert stage56_decision(audits) == (
        "conservative_projection_nonfinite_blocker_requires_review"
    )
