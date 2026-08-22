from __future__ import annotations

import math

import pytest

from vgdsmc.stage55_unclipped_quadrature_closure_audit import (
    STAGE54_COMPLETED_ENDPOINT,
    STAGE55_RULES,
    stage55_decision,
    validate_stage55_design,
)


def _row(conserved: float, heat: float) -> dict[str, float]:
    return {
        "maximum_conserved_moment_defect": conserved,
        "heat_flux_closure_relative_l2": heat,
    }


def _audits(
    baseline=(2.0e-5, 2.0e-3),
    radial=(5.0e-6, 5.0e-4),
    angular=(2.0e-5, 2.0e-3),
    coupled=(5.0e-6, 5.0e-4),
):
    rows = {
        "retained_32x96": _row(*baseline),
        "radial_40x96": _row(*radial),
        "angular_32x120": _row(*angular),
        "coupled_40x120": _row(*coupled),
    }
    return {
        "compressed_tail": {key: value.copy() for key, value in rows.items()},
        "expanded_tail": {key: value.copy() for key, value in rows.items()},
    }


def test_stage54_completed_endpoint_is_exact() -> None:
    assert STAGE54_COMPLETED_ENDPOINT["workflow_run_id"] == 30814308875
    assert STAGE54_COMPLETED_ENDPOINT["workflow_job_id"] == 91688181380
    assert STAGE54_COMPLETED_ENDPOINT["tests_passed"] == 96
    assert STAGE54_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE54_COMPLETED_ENDPOINT["artifact_id"] == 8858775690
    assert STAGE54_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "0b8ee8fae6ef5d74eafbb17802ce6f76846f0db95f007ab299e19f4db1726afb"
    )
    assert STAGE54_COMPLETED_ENDPOINT["decision"] == (
        "projected_collision_formula_or_quadrature_blocker"
    )


def test_stage55_frozen_design_accepts_only_preregistered_values() -> None:
    validate_stage55_design()


def test_stage55_frozen_design_rejects_quadrature_substitution() -> None:
    changed = STAGE55_RULES[:-1] + (("coupled_48x144", (48, 144)),)
    with pytest.raises(ValueError, match="Stage 55 is frozen"):
        validate_stage55_design(rules=changed)


def test_stage55_rules_are_orthogonal_and_have_exact_point_counts() -> None:
    rules = dict(STAGE55_RULES)
    assert rules["retained_32x96"] == (32, 96)
    assert rules["radial_40x96"] == (40, 96)
    assert rules["angular_32x120"] == (32, 120)
    assert rules["coupled_40x120"] == (40, 120)
    assert math.prod(rules["retained_32x96"]) == 3072
    assert math.prod(rules["radial_40x96"]) == 3840
    assert math.prod(rules["angular_32x120"]) == 3840
    assert math.prod(rules["coupled_40x120"]) == 4800


def test_stage55_decision_routes_closed_formula_to_conservative_projection() -> None:
    assert stage55_decision(_audits()) == (
        "radial_quadrature_closes_unclipped_formula_"
        "positivity_clipping_breaks_invariants_"
        "stage56_conservative_projection_pilot"
    )


def test_stage55_decision_retains_converging_but_not_closed_result() -> None:
    audits = _audits(
        baseline=(4.0e-5, 4.0e-3),
        radial=(1.5e-5, 1.5e-3),
        angular=(4.0e-5, 4.0e-3),
        coupled=(1.5e-5, 1.5e-3),
    )
    assert stage55_decision(audits) == (
        "unclipped_formula_quadrature_converging_"
        "stage56_higher_radial_resolution_confirmation"
    )


def test_stage55_decision_retains_small_unresolved_change() -> None:
    audits = _audits(
        baseline=(2.0e-5, 2.0e-3),
        radial=(1.6e-5, 1.6e-3),
        angular=(1.9e-5, 1.9e-3),
        coupled=(1.5e-5, 1.5e-3),
    )
    assert stage55_decision(audits) == (
        "unclipped_formula_quadrature_unresolved_"
        "stage56_higher_radial_resolution_confirmation"
    )


def test_stage55_decision_retains_material_worsening_as_blocker() -> None:
    audits = _audits(coupled=(2.3e-5, 2.3e-3))
    assert stage55_decision(audits) == (
        "projected_collision_formula_blocker_requires_review"
    )


def test_stage55_decision_retains_nonfinite_result_as_blocker() -> None:
    audits = _audits()
    audits["compressed_tail"]["coupled_40x120"][
        "maximum_conserved_moment_defect"
    ] = float("nan")
    assert stage55_decision(audits) == (
        "projected_collision_formula_blocker_requires_review"
    )
