import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.stage63_collision_target_equivalence_audit import (
    FROZEN_PAPER_STATES,
    STAGE62_COMPLETED_ENDPOINT,
    STAGE63_FORMULA_TOLERANCE,
    STAGE63_IMPLEMENTATION_TOLERANCE,
    STAGE63_MATERIAL_DEFECT_THRESHOLD,
    STAGE63_MINIMUM_CLIPPED_WEIGHT_FRACTION,
    STAGE63_RADIAL_SCALE,
    STAGE63_RULE,
    STAGE63_UNCLIPPED_CONSERVED_TOLERANCE,
    STAGE63_UNCLIPPED_HEAT_FLUX_TOLERANCE,
    aggregate_rows,
    audit_frozen_state,
    mapped_polar_quadrature,
    run_stage63,
    stage63_decision,
    validate_stage62_artifact,
    validate_stage63_design,
)


def _rows():
    quadrature = mapped_polar_quadrature(
        *STAGE63_RULE, radial_scale=STAGE63_RADIAL_SCALE
    )
    return [audit_frozen_state(state, quadrature) for state in FROZEN_PAPER_STATES]


def test_stage63_frozen_design_accepts_exact_contract():
    validate_stage63_design()


def test_stage63_frozen_design_rejects_retuning():
    with pytest.raises(ValueError, match="not a retuning stage"):
        validate_stage63_design(correction_floor=0.01)


def test_stage63_rejects_wrong_stage62_checksum(tmp_path: Path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"stage": 62, "decision": STAGE62_COMPLETED_ENDPOINT["decision"]})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_stage62_artifact(tmp_path)


def test_frozen_state_suite_covers_equilibrium_and_material_clipping():
    rows = _rows()
    assert rows[0]["clipping"]["phi_weighted_fraction"] == 0.0
    assert rows[0]["clipping"]["psi_weighted_fraction"] == 0.0
    clipped = [
        row
        for row in rows
        if max(
            row["clipping"]["phi_weighted_fraction"],
            row["clipping"]["psi_weighted_fraction"],
        )
        >= STAGE63_MINIMUM_CLIPPED_WEIGHT_FRACTION
    ]
    assert len(clipped) >= 3


def test_independent_zeta_and_c_paper_formulas_are_equivalent():
    aggregate = aggregate_rows(_rows())
    assert aggregate["maximum_formula_transform_error"] < STAGE63_FORMULA_TOLERANCE


def test_retained_implementation_matches_independent_clipped_path():
    aggregate = aggregate_rows(_rows())
    assert (
        aggregate["maximum_implementation_match_error"]
        < STAGE63_IMPLEMENTATION_TOLERANCE
    )


def test_unclipped_paper_target_closes_required_moments():
    aggregate = aggregate_rows(_rows())
    assert (
        aggregate["maximum_unclipped_conserved_moment_defect"]
        < STAGE63_UNCLIPPED_CONSERVED_TOLERANCE
    )
    assert (
        aggregate["maximum_unclipped_heat_flux_closure_error"]
        < STAGE63_UNCLIPPED_HEAT_FLUX_TOLERANCE
    )


def test_retained_clipping_produces_material_defects_without_hiding_them():
    aggregate = aggregate_rows(_rows())
    assert max(
        aggregate["maximum_clipped_conserved_moment_defect"],
        aggregate["maximum_clipped_heat_flux_closure_error"],
    ) >= STAGE63_MATERIAL_DEFECT_THRESHOLD


def test_stage63_decision_routes_material_result_and_blocker():
    aggregate = aggregate_rows(_rows())
    assert stage63_decision(aggregate) == (
        "stage63_collision_target_matches_independent_paper_equations_"
        "clipping_defects_material_stage64_source_step_consistency_audit"
    )
    aggregate["maximum_formula_transform_error"] = 1.0
    assert stage63_decision(aggregate) == (
        "stage63_independent_paper_formula_transform_blocker"
    )


def test_stage63_decision_preserves_nonfinite_blocker():
    aggregate = aggregate_rows(_rows())
    aggregate["maximum_clipped_heat_flux_closure_error"] = np.nan
    assert stage63_decision(aggregate) == (
        "stage63_nonfinite_collision_target_audit_blocker"
    )


def test_run_stage63_writes_guarded_summary(tmp_path: Path, monkeypatch):
    import vgdsmc.stage63_collision_target_equivalence_audit as stage63

    retained = {
        "stage": 62,
        "decision": STAGE62_COMPLETED_ENDPOINT["decision"],
    }
    monkeypatch.setattr(stage63, "validate_stage62_artifact", lambda _: retained)
    summary = run_stage63(tmp_path / "stage62", tmp_path / "out")
    assert summary["stage"] == 63
    assert summary["configuration"]["physical_parameter_retuning"] is False
    assert summary["configuration"]["correction_floor_retuning"] is False
    assert summary["configuration"]["cross_knudsen_extension_permitted"] is False
    assert summary["configuration"]["conservative_projection_adopted"] is False
    assert "not physical correctness" in summary["negative_findings"][0]
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert written["decision"] == summary["decision"]
    assert written["aggregate"]["states_with_material_clipping"] >= 3
