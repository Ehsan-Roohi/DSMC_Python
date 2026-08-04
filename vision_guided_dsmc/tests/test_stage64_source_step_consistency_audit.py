import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.stage64_source_step_consistency_audit import (
    STAGE63_COMPLETED_ENDPOINT,
    STAGE64_IDENTITY_TOLERANCE,
    STAGE64_INITIAL_MOMENT_TOLERANCE,
    aggregate_rows,
    audit_frozen_state,
    mapped_polar_quadrature,
    run_stage64,
    stage64_decision,
    validate_stage63_artifact,
    validate_stage64_design,
)
from vgdsmc.stage63_collision_target_equivalence_audit import (
    FROZEN_PAPER_STATES,
    STAGE63_RADIAL_SCALE,
    STAGE63_RULE,
)


def _rows():
    quadrature = mapped_polar_quadrature(
        *STAGE63_RULE, radial_scale=STAGE63_RADIAL_SCALE
    )
    return [audit_frozen_state(state, quadrature) for state in FROZEN_PAPER_STATES]


def test_stage64_frozen_design_accepts_exact_contract():
    validate_stage64_design()


def test_stage64_frozen_design_rejects_retuning():
    with pytest.raises(ValueError, match="not a retuning stage"):
        validate_stage64_design(correction_floor=0.01)


def test_stage64_rejects_wrong_stage63_checksum(tmp_path: Path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"stage": 63, "decision": STAGE63_COMPLETED_ENDPOINT["decision"]})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_stage63_artifact(tmp_path)


def test_manufactured_initial_states_recover_frozen_moments():
    aggregate = aggregate_rows(_rows())
    assert aggregate["maximum_initial_state_error"] < STAGE64_INITIAL_MOMENT_TOLERANCE
    assert aggregate["maximum_initial_heat_flux_error"] < STAGE64_INITIAL_MOMENT_TOLERANCE


def test_all_source_arms_obey_raw_moment_linear_identity():
    aggregate = aggregate_rows(_rows())
    for arm in aggregate["arms"].values():
        assert arm["maximum_raw_moment_identity_error"] < STAGE64_IDENTITY_TOLERANCE


def test_stage64_preserves_active_and_diagnostic_labels():
    rows = _rows()
    arms = {item["arm"] for item in rows[0]["source_rows"]}
    assert arms == {
        "retained_clipped_active",
        "unclipped_paper_diagnostic",
        "bounded_conservative_diagnostic",
    }
    assert rows[0]["manufactured_initial"]["description"].startswith("Signed")


def test_stage64_decision_preserves_nonfinite_blocker():
    aggregate = aggregate_rows(_rows())
    aggregate["maximum_initial_state_error"] = np.nan
    assert stage64_decision(aggregate) == "stage64_nonfinite_source_step_audit_blocker"


def test_run_stage64_writes_guarded_summary(tmp_path: Path, monkeypatch):
    import vgdsmc.stage64_source_step_consistency_audit as stage64

    retained = {
        "stage": 63,
        "decision": STAGE63_COMPLETED_ENDPOINT["decision"],
    }
    monkeypatch.setattr(stage64, "validate_stage63_artifact", lambda _: retained)
    summary = run_stage64(tmp_path / "stage63", tmp_path / "out")
    assert summary["stage"] == 64
    assert summary["configuration"]["physical_parameter_retuning"] is False
    assert summary["configuration"]["correction_floor_retuning"] is False
    assert summary["configuration"]["cross_knudsen_extension_permitted"] is False
    assert summary["configuration"]["conservative_projection_adopted"] is False
    assert summary["configuration"]["active_arm"] == "retained_clipped_active"
    assert "not a positivity" in summary["negative_findings"][0]
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert written["decision"] == summary["decision"]
    assert written["aggregate"]["state_count"] == len(FROZEN_PAPER_STATES)
