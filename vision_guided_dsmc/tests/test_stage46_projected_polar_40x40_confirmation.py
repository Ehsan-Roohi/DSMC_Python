from __future__ import annotations

import copy
import pytest

from vgdsmc.stage46_projected_polar_40x40_confirmation import (
    STAGE45_COMPLETED_ENDPOINT,
    STAGE45_RETAINED_32X32_CASE,
    STAGE46_GRID,
    STAGE46_KNUDSEN,
    STAGE46_MAX_ITERATIONS,
    STAGE46_RATIO,
    STAGE46_RULE,
    STAGE46_SOURCE_RELAXATION,
    STAGE46_TOLERANCE,
    build_stage46_config,
    stage46_decision,
    validate_stage46_design,
)


def test_stage46_design_is_frozen() -> None:
    validate_stage46_design(
        STAGE46_GRID,
        STAGE46_RULE,
        STAGE46_KNUDSEN,
        STAGE46_RATIO,
        STAGE46_MAX_ITERATIONS,
        STAGE46_TOLERANCE,
        STAGE46_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage46_design(
            (36, 36),
            STAGE46_RULE,
            STAGE46_KNUDSEN,
            STAGE46_RATIO,
            STAGE46_MAX_ITERATIONS,
            STAGE46_TOLERANCE,
            STAGE46_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage46_design(
            STAGE46_GRID,
            (40, 96),
            STAGE46_KNUDSEN,
            STAGE46_RATIO,
            STAGE46_MAX_ITERATIONS,
            STAGE46_TOLERANCE,
            STAGE46_SOURCE_RELAXATION,
        )


def test_stage46_config_retains_physics_and_numerics() -> None:
    cfg = build_stage46_config()
    assert (cfg.nx, cfg.ny) == (40, 40)
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
        float(value) for value in STAGE45_RETAINED_32X32_CASE["table_velocity"]
    ]
    return {
        "predicted_qav": qav,
        "qav_relative_error": qerr,
        "velocity_metrics": {
            "relative_rms": vrms,
            "relative_l1": vrms,
            "sign_agreement": 0.6,
        },
        "table_velocity": [
            profile_factor * value for value in retained_profile
        ],
        "finite": True,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "wall_mass_balance_relative_error": 1.0e-16,
        "converged": converged,
    }


def test_stage46_decision_retains_all_endpoints() -> None:
    retained = copy.deepcopy(STAGE45_RETAINED_32X32_CASE)

    converging = _case(0.0770, 0.070, 0.80, 0.95)
    assert stage46_decision(retained, converging) == (
        "projected_polar_40x40_converging_stage47_cross_kn_extension"
    )

    unresolved = _case(0.0765, 0.063, 0.75, 0.70)
    assert stage46_decision(retained, unresolved) == (
        "projected_polar_40x40_improving_not_converged_"
        "stage47_48x48_confirmation"
    )

    heat_only = _case(0.0770, 0.070, 1.00, 0.95)
    assert stage46_decision(retained, heat_only) == (
        "projected_polar_40x40_heat_flux_only_improves_"
        "stage47_wall_observable_audit"
    )

    velocity_only = _case(0.0780, 0.083, 0.80, 0.95)
    assert stage46_decision(retained, velocity_only) == (
        "projected_polar_40x40_velocity_only_improves_"
        "stage47_heat_flux_definition_audit"
    )

    blocker = _case(0.0770, 0.070, 0.80, 0.95, converged=False)
    assert stage46_decision(retained, blocker) == "stage46_40x40_blocker"


def test_stage45_provenance_is_exactly_retained() -> None:
    assert STAGE45_COMPLETED_ENDPOINT["workflow_run_id"] == 30751714392
    assert STAGE45_COMPLETED_ENDPOINT["workflow_job_id"] == 91506805495
    assert STAGE45_COMPLETED_ENDPOINT["tests_passed"] == 41
    assert STAGE45_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE45_COMPLETED_ENDPOINT["artifact_id"] == 8834940840
    assert STAGE45_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "3d0e7eacd03986c90646a08c9175a6de435e406ac94d8a6c4db76d140ccfd8c7"
    )
    assert STAGE45_COMPLETED_ENDPOINT["source_head_sha"] == (
        "6b96df929af9389542314625ca5ba46f5ed3c612"
    )
    retained = STAGE45_RETAINED_32X32_CASE
    assert retained["grid"] == [32, 32]
    assert retained["predicted_qav"] == 0.07765042067403488
    assert retained["qav_relative_error"] == 0.07847806491715122
    assert retained["velocity_metrics"]["relative_rms"] == 0.9048587311194247
