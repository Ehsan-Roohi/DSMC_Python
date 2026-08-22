from __future__ import annotations

import copy
import pytest

from vgdsmc.stage48_projected_polar_56x56_confirmation import (
    STAGE47_COMPLETED_ENDPOINT,
    STAGE47_RETAINED_48X48_CASE,
    STAGE48_GRID,
    STAGE48_KNUDSEN,
    STAGE48_MAX_ITERATIONS,
    STAGE48_RATIO,
    STAGE48_RULE,
    STAGE48_SOURCE_RELAXATION,
    STAGE48_TOLERANCE,
    build_stage48_config,
    stage48_decision,
    validate_stage48_design,
)


def test_stage48_design_is_frozen() -> None:
    validate_stage48_design(
        STAGE48_GRID,
        STAGE48_RULE,
        STAGE48_KNUDSEN,
        STAGE48_RATIO,
        STAGE48_MAX_ITERATIONS,
        STAGE48_TOLERANCE,
        STAGE48_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage48_design(
            (48, 48),
            STAGE48_RULE,
            STAGE48_KNUDSEN,
            STAGE48_RATIO,
            STAGE48_MAX_ITERATIONS,
            STAGE48_TOLERANCE,
            STAGE48_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage48_design(
            STAGE48_GRID,
            (40, 96),
            STAGE48_KNUDSEN,
            STAGE48_RATIO,
            STAGE48_MAX_ITERATIONS,
            STAGE48_TOLERANCE,
            STAGE48_SOURCE_RELAXATION,
        )


def test_stage48_config_retains_physics_and_numerics() -> None:
    cfg = build_stage48_config()
    assert (cfg.nx, cfg.ny) == (56, 56)
    assert cfg.kn0 == 0.1
    assert cfg.cold_hot_ratio == 0.1
    assert cfg.viscosity_exponent == 0.5
    assert cfg.prandtl == 2.0 / 3.0
    assert cfg.max_steps == 3000
    assert cfg.tolerance == 2.0e-5
    assert cfg.check_interval == 25
    assert cfg.minimum_steps == 500
    assert cfg.positivity_floor == 1.0e-30


def _case(
    qav: float,
    qerr: float,
    vrms: float,
    profile_factor: float,
    *,
    converged: bool = True,
) -> dict[str, object]:
    retained_profile = [
        float(value) for value in STAGE47_RETAINED_48X48_CASE["table_velocity"]
    ]
    return {
        "predicted_qav": qav,
        "qav_relative_error": qerr,
        "velocity_metrics": {
            "relative_rms": vrms,
            "relative_l1": vrms,
            "sign_agreement": 0.8,
        },
        "table_velocity": [profile_factor * value for value in retained_profile],
        "finite": True,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "wall_mass_balance_relative_error": 1.0e-16,
        "converged": converged,
    }


def test_stage48_decision_retains_all_endpoints() -> None:
    retained = copy.deepcopy(STAGE47_RETAINED_48X48_CASE)

    converging = _case(0.0761, 0.0569, 0.49, 0.95)
    assert stage48_decision(retained, converging) == (
        "projected_polar_56x56_converging_stage49_cross_kn_extension"
    )

    unresolved = _case(0.0759, 0.0542, 0.45, 0.70)
    assert stage48_decision(retained, unresolved) == (
        "projected_polar_56x56_improving_not_converged_"
        "stage49_64x64_confirmation"
    )

    heat_only = _case(0.0761, 0.0569, 0.60, 0.95)
    assert stage48_decision(retained, heat_only) == (
        "projected_polar_56x56_heat_flux_only_improves_"
        "stage49_wall_observable_audit"
    )

    velocity_only = _case(0.0767, 0.0653, 0.49, 0.95)
    assert stage48_decision(retained, velocity_only) == (
        "projected_polar_56x56_velocity_only_improves_"
        "stage49_heat_flux_definition_audit"
    )

    nonmonotonic = _case(0.0767, 0.0653, 0.60, 0.95)
    assert stage48_decision(retained, nonmonotonic) == (
        "projected_polar_56x56_nonmonotonic_"
        "stage49_space_velocity_coupling_audit"
    )

    blocker = _case(0.0761, 0.0569, 0.49, 0.95, converged=False)
    assert stage48_decision(retained, blocker) == "stage48_56x56_blocker"


def test_stage47_provenance_is_exactly_retained() -> None:
    endpoint = STAGE47_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 30755677448
    assert endpoint["workflow_job_id"] == 91517302666
    assert endpoint["tests_passed"] == 49
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8838257601
    assert endpoint["artifact_sha256"] == (
        "58ef46902d6e80f3e2c198fa6d33f0f9209331da08e14664b9c4e22b1ddb0227"
    )
    assert endpoint["source_head_sha"] == (
        "10eba2a345c8d6b0942bce6bbed2c0d030dc005a"
    )
    assert endpoint["decision"] == (
        "projected_polar_48x48_improving_not_converged_"
        "stage48_56x56_confirmation"
    )

    retained = STAGE47_RETAINED_48X48_CASE
    assert retained["grid"] == [48, 48]
    assert retained["predicted_qav"] == 0.07643318669591011
    assert retained["qav_relative_error"] == 0.06157203744319602
    assert retained["velocity_metrics"]["relative_rms"] == 0.5272365313931947
    assert retained["velocity_metrics"]["sign_agreement"] == 0.8
    assert retained["maximum_phi_clipped_weight_fraction"] == 0.002952525420025651
    assert retained["maximum_psi_clipped_weight_fraction"] == 0.004908712341779989
