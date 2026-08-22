from __future__ import annotations

from pathlib import Path
import json
import pytest

from vgdsmc.linear_sidewall_validation import LinearSidewallConfig
from vgdsmc.stage41_projected_polar_operator_audit import mapped_polar_quadrature
from vgdsmc.stage60_transport_wall_equation_audit import (
    STAGE59_COMPLETED_ENDPOINT,
    STAGE60_COLD_HOT_RATIO,
    STAGE60_GRID,
    STAGE60_HALF_SPACE_ENERGY_TOLERANCE,
    STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE,
    STAGE60_KNUDSEN,
    STAGE60_OBSERVABLE_IDENTITY_TOLERANCE,
    STAGE60_RADIAL_SCALE,
    STAGE60_RULE,
    STAGE60_TRANSPORT_BALANCE_TOLERANCE,
    STAGE60_WALL_MASS_TOLERANCE,
    diffuse_wall_half_space_audit,
    evaluate_stage60,
    isothermal_collision_off_audit,
    transport_conservation_audit,
    validate_stage59_artifact,
    validate_stage60_design,
)


def _quadrature():
    return mapped_polar_quadrature(
        STAGE60_RULE[0], STAGE60_RULE[1], STAGE60_RADIAL_SCALE
    )


def test_stage60_design_is_frozen() -> None:
    validate_stage60_design(
        STAGE60_GRID,
        STAGE60_RULE,
        STAGE60_RADIAL_SCALE,
        STAGE60_KNUDSEN,
        STAGE60_COLD_HOT_RATIO,
    )
    with pytest.raises(ValueError):
        validate_stage60_design(
            (8, 5),
            STAGE60_RULE,
            STAGE60_RADIAL_SCALE,
            STAGE60_KNUDSEN,
            STAGE60_COLD_HOT_RATIO,
        )
    with pytest.raises(ValueError):
        validate_stage60_design(
            STAGE60_GRID,
            (32, 96),
            STAGE60_RADIAL_SCALE,
            STAGE60_KNUDSEN,
            STAGE60_COLD_HOT_RATIO,
        )
    with pytest.raises(ValueError):
        validate_stage60_design(
            STAGE60_GRID,
            STAGE60_RULE,
            1.0,
            STAGE60_KNUDSEN,
            STAGE60_COLD_HOT_RATIO,
        )


def test_transport_operator_telescopes_to_boundary_flux() -> None:
    cfg = LinearSidewallConfig(
        nx=STAGE60_GRID[0],
        ny=STAGE60_GRID[1],
        kn0=STAGE60_KNUDSEN,
        cold_hot_ratio=STAGE60_COLD_HOT_RATIO,
    )
    result = transport_conservation_audit(cfg, _quadrature())
    assert result["phi_telescoping_relative_error"] <= STAGE60_TRANSPORT_BALANCE_TOLERANCE
    assert result["psi_telescoping_relative_error"] <= STAGE60_TRANSPORT_BALANCE_TOLERANCE
    assert result["mass_balance_identity_relative_error"] <= STAGE60_TRANSPORT_BALANCE_TOLERANCE
    assert result["energy_balance_identity_relative_error"] <= STAGE60_TRANSPORT_BALANCE_TOLERANCE


def test_collision_off_isothermal_maxwellian_is_fixed_point() -> None:
    result = isothermal_collision_off_audit(_quadrature())
    assert result["phi_fixed_point_relative_error"] <= STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE
    assert result["psi_fixed_point_relative_error"] <= STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE
    assert result["maximum_wall_mass_balance_error"] <= STAGE60_WALL_MASS_TOLERANCE
    assert result["maximum_absolute_bottom_wall_heat_flux"] <= STAGE60_ISOTHERMAL_FIXED_POINT_TOLERANCE


def test_diffuse_wall_half_space_mass_and_energy_identities() -> None:
    result = diffuse_wall_half_space_audit(_quadrature())
    assert len(result["rows"]) == 12
    assert result["maximum_relative_mass_balance_error"] <= STAGE60_WALL_MASS_TOLERANCE
    assert (
        result["maximum_incoming_energy_per_mass_relative_error"]
        <= STAGE60_HALF_SPACE_ENERGY_TOLERANCE
    )
    assert (
        result["maximum_net_energy_exchange_relative_error"]
        <= STAGE60_HALF_SPACE_ENERGY_TOLERANCE
    )
    assert (
        result["maximum_bottom_observable_identity_relative_error"]
        <= STAGE60_OBSERVABLE_IDENTITY_TOLERANCE
    )


def test_stage60_decision_advances_only_to_characteristic_audit() -> None:
    summary = evaluate_stage60()
    assert summary["stage"] == 60
    assert summary["decision"] == (
        "stage60_transport_and_diffuse_wall_equations_close_"
        "discrepancy_not_explained_characteristic_audit_next"
    )
    assert all(summary["checks"].values())
    assert summary["configuration"]["solver_rerun"] is False
    assert summary["configuration"]["cross_knudsen_extension_permitted"] is False
    for key, value in summary["configuration"].items():
        if key.endswith("_retuning"):
            assert value is False
    assert "does not validate" in summary["interpretation_guard"]
    assert "characteristic-based" in summary["scientifically_justified_next_scope"]
    assert len(summary["positive_findings"]) >= 4
    assert len(summary["negative_findings"]) >= 3


def test_stage59_provenance_is_pinned() -> None:
    assert STAGE59_COMPLETED_ENDPOINT["workflow_run_id"] == 30884941157
    assert STAGE59_COMPLETED_ENDPOINT["workflow_job_id"] == 91913958890
    assert STAGE59_COMPLETED_ENDPOINT["workflow_conclusion"] == "success"
    assert STAGE59_COMPLETED_ENDPOINT["artifact_id"] == 8885375771
    assert STAGE59_COMPLETED_ENDPOINT["summary_sha256"] == (
        "d562563b58c5cb08aea638278cc2486b03e2ea526cd89e369956c6bb39b0929b"
    )


def test_stage59_artifact_validation_rejects_missing_or_modified_summary(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        validate_stage59_artifact(tmp_path)
    (tmp_path / "summary.json").write_text(
        json.dumps({"stage": 59, "decision": STAGE59_COMPLETED_ENDPOINT["decision"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_stage59_artifact(tmp_path)
