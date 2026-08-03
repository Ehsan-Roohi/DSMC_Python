from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from vgdsmc.stage41_projected_polar_operator_audit import (
    mapped_polar_quadrature,
    projected_macroscopic,
)
from vgdsmc.stage54_projected_collision_moment_audit import (
    STAGE53_COMPLETED_ENDPOINT,
    _local_and_global_defects,
    audit_case,
    restore_internal_fields,
    sha256_file,
    stage54_decision,
    unclipped_projected_shakhov_equilibrium,
    validate_stage54_design,
)


def test_completed_stage53_endpoint_is_exact() -> None:
    assert STAGE53_COMPLETED_ENDPOINT["workflow_run_id"] == 30803098842
    assert STAGE53_COMPLETED_ENDPOINT["workflow_job_id"] == 91652089093
    assert STAGE53_COMPLETED_ENDPOINT["tests_passed"] == 87
    assert STAGE53_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE53_COMPLETED_ENDPOINT["artifact_id"] == 8854974012


def test_stage54_design_is_frozen() -> None:
    validate_stage54_design()
    with pytest.raises(ValueError):
        validate_stage54_design(kn0=1.0)


def test_restore_internal_fields_applies_exact_sqrt2_conversion() -> None:
    stored = {
        key: np.ones((1, 1), dtype=np.float64)
        for key in ("rho", "T", "u", "v", "qx", "qy")
    }
    restored = restore_internal_fields(stored)
    assert restored["rho"][0, 0] == 1.0
    assert restored["T"][0, 0] == 1.0
    assert restored["u"][0, 0] == math.sqrt(2.0)
    assert restored["v"][0, 0] == math.sqrt(2.0)
    assert restored["qx"][0, 0] == math.sqrt(2.0)
    assert restored["qy"][0, 0] == math.sqrt(2.0)


def test_unclipped_projected_equilibrium_closes_representative_state() -> None:
    quadrature = mapped_polar_quadrature(32, 96)
    fields = {
        "rho": np.array([[1.0]]),
        "u": np.array([[0.03]]),
        "v": np.array([[-0.02]]),
        "T": np.array([[0.5]]),
        "qx": np.array([[0.005]]),
        "qy": np.array([[-0.003]]),
    }
    phi, psi = unclipped_projected_shakhov_equilibrium(fields, quadrature)
    metrics, _, _ = _local_and_global_defects(
        fields, projected_macroscopic(phi, psi, quadrature)
    )
    assert metrics["maximum_conserved_moment_defect"] < 1.0e-5
    assert metrics["heat_flux_closure_relative_l2"] < 1.0e-3


def test_retained_clipping_can_expose_larger_actual_moment_defect() -> None:
    fields = {
        "rho": np.ones((2, 2)),
        "u": np.zeros((2, 2)),
        "v": np.zeros((2, 2)),
        "T": np.full((2, 2), 0.1),
        "qx": np.full((2, 2), 0.04),
        "qy": np.full((2, 2), -0.03),
    }
    row, arrays = audit_case(fields, 1.0)
    assert row["current_clipped_operator"][
        "maximum_conserved_moment_defect"
    ] > row["unclipped_algebraic_diagnostic"][
        "maximum_conserved_moment_defect"
    ]
    assert np.isfinite(arrays["clipped_local_conserved_defect"]).all()


def _fake_case(current_defect: float, unclipped_defect: float) -> dict:
    return {
        "current_clipped_operator": {
            "maximum_conserved_moment_defect": current_defect,
            "heat_flux_closure_relative_l2": current_defect,
        },
        "unclipped_algebraic_diagnostic": {
            "maximum_conserved_moment_defect": unclipped_defect,
            "heat_flux_closure_relative_l2": unclipped_defect,
        },
    }


def test_decision_routes_clipping_defect_to_conservative_projection() -> None:
    assert stage54_decision({"case": _fake_case(0.02, 1.0e-6)}) == (
        "positivity_clipping_breaks_collision_invariants_"
        "stage55_conservative_positive_projection_pilot"
    )


def test_decision_preserves_formula_or_quadrature_blocker() -> None:
    assert stage54_decision({"case": _fake_case(0.02, 0.02)}) == (
        "projected_collision_formula_or_quadrature_blocker"
    )


def test_decision_routes_negligible_collision_defect_to_knudsen_audit() -> None:
    assert stage54_decision({"case": _fake_case(1.0e-6, 1.0e-6)}) == (
        "projected_collision_moments_do_not_explain_cross_kn_heat_flux_"
        "stage55_knudsen_convention_audit"
    )


def test_sha256_helper_is_exact(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
