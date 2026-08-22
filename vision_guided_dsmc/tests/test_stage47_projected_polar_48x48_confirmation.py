from __future__ import annotations

import copy
import pytest

from vgdsmc.stage47_projected_polar_48x48_confirmation import (
    STAGE46_COMPLETED_ENDPOINT,
    STAGE46_RETAINED_40X40_CASE,
    STAGE47_GRID,
    STAGE47_KNUDSEN,
    STAGE47_MAX_ITERATIONS,
    STAGE47_RATIO,
    STAGE47_RULE,
    STAGE47_SOURCE_RELAXATION,
    STAGE47_TOLERANCE,
    build_stage47_config,
    stage47_decision,
    validate_stage47_design,
)


def test_stage47_design_is_frozen() -> None:
    validate_stage47_design(
        STAGE47_GRID,
        STAGE47_RULE,
        STAGE47_KNUDSEN,
        STAGE47_RATIO,
        STAGE47_MAX_ITERATIONS,
        STAGE47_TOLERANCE,
        STAGE47_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage47_design(
            (40, 40),
            STAGE47_RULE,
            STAGE47_KNUDSEN,
            STAGE47_RATIO,
            STAGE47_MAX_ITERATIONS,
            STAGE47_TOLERANCE,
            STAGE47_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage47_design(
            STAGE47_GRID,
            (40, 96),
            STAGE47_KNUDSEN,
            STAGE47_RATIO,
            STAGE47_MAX_ITERATIONS,
            STAGE47_TOLERANCE,
            STAGE47_SOURCE_RELAXATION,
        )


def test_stage47_config_retains_physics_and_numerics() -> None:
    cfg = build_stage47_config()
    assert (cfg.nx, cfg.ny) == (48, 48)
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
        float(value) for value in STAGE46_RETAINED_40X40_CASE["table_velocity"]
    ]
    return {
        "predicted_qav": qav,
        "qav_relative_error": qerr,
        "velocity_metrics": {
            "relative_rms": vrms,
            "relative_l1": vrms,
            "sign_agreement": 0.8,
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


def test_stage47_decision_retains_all_endpoints() -> None:
    retained = copy.deepcopy(STAGE46_RETAINED_40X40_CASE)

    converging = _case(0.0765, 0.0625, 0.60, 0.95)
    assert stage47_decision(retained, converging) == (
        "projected_polar_48x48_converging_stage48_cross_kn_extension"
    )

    unresolved = _case(0.0758, 0.0528, 0.55, 0.70)
    assert stage47_decision(retained, unresolved) == (
        "projected_polar_48x48_improving_not_converged_"
        "stage48_56x56_confirmation"
    )

    heat_only = _case(0.0765, 0.0625, 0.75, 0.95)
    assert stage47_decision(retained, heat_only) == (
        "projected_polar_48x48_heat_flux_only_improves_"
        "stage48_wall_observable_audit"
    )

    velocity_only = _case(0.0775, 0.0764, 0.60, 0.95)
    assert stage47_decision(retained, velocity_only) == (
        "projected_polar_48x48_velocity_only_improves_"
        "stage48_heat_flux_definition_audit"
    )

    blocker = _case(0.0765, 0.0625, 0.60, 0.95, converged=False)
    assert stage47_decision(retained, blocker) == "stage47_48x48_blocker"


def test_stage46_provenance_is_exactly_retained() -> None:
    assert STAGE46_COMPLETED_ENDPOINT["workflow_run_id"] == 30753260337
    assert STAGE46_COMPLETED_ENDPOINT["workflow_job_id"] == 91510943443
    assert STAGE46_COMPLETED_ENDPOINT["tests_passed"] == 45
    assert STAGE46_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE46_COMPLETED_ENDPOINT["artifact_id"] == 8835645496
    assert STAGE46_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "384af3378996f74f7f17905f064c86c9aa702cbb2f86549cacef3998f24886e8"
    )
    assert STAGE46_COMPLETED_ENDPOINT["source_head_sha"] == (
        "9a7d48b7925b404513f1e2b40cff3c9588c60ce0"
    )
    retained = STAGE46_RETAINED_40X40_CASE
    assert retained["grid"] == [40, 40]
    assert retained["predicted_qav"] == 0.07691741440108028
    assert retained["qav_relative_error"] == 0.06829742223722624
    assert retained["velocity_metrics"]["relative_rms"] == 0.6767910267583875
