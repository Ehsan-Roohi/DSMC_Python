import json
from pathlib import Path

import numpy as np
import pytest

from vgdsmc.stage62_finite_kn_normalization_audit import (
    STAGE61_COMPLETED_ENDPOINT,
    STAGE62_COLLISION_TOLERANCE,
    STAGE62_COORDINATE_TOLERANCE,
    STAGE62_OBSERVABLE_TOLERANCE,
    STAGE62_RATIO_TOLERANCE,
    STAGE62_SHAKHOV_TOLERANCE,
    collision_frequency_audit,
    coordinate_scale_audit,
    observable_normalization_audit,
    run_stage62,
    shakhov_target_equivalence_audit,
    stage62_decision,
    transport_collision_ratio_audit,
    validate_stage61_artifact,
    validate_stage62_design,
)


def test_stage62_frozen_design_accepts_exact_contract():
    validate_stage62_design()


def test_stage62_frozen_design_rejects_retuning():
    with pytest.raises(ValueError, match="not a retuning stage"):
        validate_stage62_design(kn0=9.5)


def test_stage62_rejects_wrong_stage61_checksum(tmp_path: Path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"stage": 61, "decision": STAGE61_COMPLETED_ENDPOINT["decision"]})
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_stage61_artifact(tmp_path)


def test_coordinate_scale_identifies_sqrt2_mapping():
    row = coordinate_scale_audit()
    assert row["mass_error"] < STAGE62_COORDINATE_TOLERANCE
    assert row["temperature_error"] < STAGE62_COORDINATE_TOLERANCE
    assert row["maximum_scale_error_from_sqrt2"] < STAGE62_COORDINATE_TOLERANCE


def test_collision_frequency_matches_transformed_paper_equation():
    row = collision_frequency_audit()
    assert row["maximum_relative_collision_frequency_error"] < STAGE62_COLLISION_TOLERANCE
    assert row["tau_prefactor_relative_error"] < STAGE62_COLLISION_TOLERANCE


def test_transport_collision_ratio_is_coordinate_invariant():
    row = transport_collision_ratio_audit()
    assert row["maximum_relative_ratio_error"] < STAGE62_RATIO_TOLERANCE


def test_observable_normalization_recovers_paper_units():
    row = observable_normalization_audit()
    assert row["maximum_relative_velocity_conversion_error"] < STAGE62_OBSERVABLE_TOLERANCE
    assert row["maximum_relative_heat_flux_conversion_error"] < STAGE62_OBSERVABLE_TOLERANCE


def test_projected_shakhov_target_matches_paper_transform_without_clipping():
    row = shakhov_target_equivalence_audit()
    assert row["maximum_relative_phi_target_error"] < STAGE62_SHAKHOV_TOLERANCE
    assert row["maximum_relative_psi_target_error"] < STAGE62_SHAKHOV_TOLERANCE
    assert row["maximum_phi_clipped_weight_fraction"] == 0.0
    assert row["maximum_psi_clipped_weight_fraction"] == 0.0


def test_stage62_decision_routes_success_and_blocker():
    audits = {
        "coordinate_scale": coordinate_scale_audit(),
        "collision_frequency": collision_frequency_audit(),
        "transport_collision_ratio": transport_collision_ratio_audit(),
        "observable_normalization": observable_normalization_audit(),
        "shakhov_target_equivalence": shakhov_target_equivalence_audit(),
    }
    assert stage62_decision(audits) == (
        "stage62_finite_kn_relaxation_and_normalization_close_"
        "stage63_collision_target_equivalence_audit"
    )
    audits["collision_frequency"]["maximum_relative_collision_frequency_error"] = 1.0
    assert stage62_decision(audits) == "stage62_relaxation_frequency_mapping_blocker"


def test_run_stage62_writes_guarded_summary(tmp_path: Path, monkeypatch):
    import vgdsmc.stage62_finite_kn_normalization_audit as stage62

    retained = {
        "stage": 61,
        "decision": STAGE61_COMPLETED_ENDPOINT["decision"],
    }
    monkeypatch.setattr(stage62, "validate_stage61_artifact", lambda _: retained)
    summary = run_stage62(tmp_path / "stage61", tmp_path / "out")
    assert summary["stage"] == 62
    assert summary["configuration"]["physical_parameter_retuning"] is False
    assert summary["configuration"]["normalization_retuning"] is False
    assert summary["configuration"]["cross_knudsen_extension_permitted"] is False
    assert "does not validate" in summary["negative_findings"][0]
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert written["decision"] == summary["decision"]
    assert np.isfinite(
        written["audits"]["collision_frequency"][
            "maximum_relative_collision_frequency_error"
        ]
    )
