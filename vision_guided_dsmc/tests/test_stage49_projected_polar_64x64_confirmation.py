from __future__ import annotations

import copy
import pytest

from vgdsmc.stage49_projected_polar_64x64_confirmation import (
    STAGE48_COMPLETED_ENDPOINT,
    STAGE48_RETAINED_56X56_CASE,
    STAGE49_GRID,
    STAGE49_KNUDSEN,
    STAGE49_MAX_ITERATIONS,
    STAGE49_RATIO,
    STAGE49_RULE,
    STAGE49_SOURCE_RELAXATION,
    STAGE49_TOLERANCE,
    build_stage49_config,
    stage49_decision,
    validate_stage49_design,
)


def test_stage49_design_is_frozen() -> None:
    validate_stage49_design(
        STAGE49_GRID,
        STAGE49_RULE,
        STAGE49_KNUDSEN,
        STAGE49_RATIO,
        STAGE49_MAX_ITERATIONS,
        STAGE49_TOLERANCE,
        STAGE49_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage49_design(
            (56, 56),
            STAGE49_RULE,
            STAGE49_KNUDSEN,
            STAGE49_RATIO,
            STAGE49_MAX_ITERATIONS,
            STAGE49_TOLERANCE,
            STAGE49_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage49_design(
            STAGE49_GRID,
            (40, 96),
            STAGE49_KNUDSEN,
            STAGE49_RATIO,
            STAGE49_MAX_ITERATIONS,
            STAGE49_TOLERANCE,
            STAGE49_SOURCE_RELAXATION,
        )


def test_stage49_config_retains_physics_and_numerics() -> None:
    cfg = build_stage49_config()
    assert (cfg.nx, cfg.ny) == (64, 64)
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
        float(value) for value in STAGE48_RETAINED_56X56_CASE["table_velocity"]
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


def test_stage49_decision_retains_all_endpoints() -> None:
    retained = copy.deepcopy(STAGE48_RETAINED_56X56_CASE)

    converging = _case(0.0758, 0.0528, 0.39, 0.95)
    assert stage49_decision(retained, converging) == (
        "projected_polar_64x64_converging_stage50_cross_kn_extension"
    )

    unresolved = _case(0.0757, 0.0514, 0.36, 0.70)
    assert stage49_decision(retained, unresolved) == (
        "projected_polar_64x64_improving_not_converged_"
        "stage50_72x72_confirmation"
    )

    heat_only = _case(0.0758, 0.0528, 0.50, 0.95)
    assert stage49_decision(retained, heat_only) == (
        "projected_polar_64x64_heat_flux_only_improves_"
        "stage50_wall_observable_audit"
    )

    velocity_only = _case(0.0765, 0.0625, 0.39, 0.95)
    assert stage49_decision(retained, velocity_only) == (
        "projected_polar_64x64_velocity_only_improves_"
        "stage50_heat_flux_definition_audit"
    )

    nonmonotonic = _case(0.0765, 0.0625, 0.50, 0.95)
    assert stage49_decision(retained, nonmonotonic) == (
        "projected_polar_64x64_nonmonotonic_"
        "stage50_space_velocity_coupling_audit"
    )

    blocker = _case(0.0758, 0.0528, 0.39, 0.95, converged=False)
    assert stage49_decision(retained, blocker) == "stage49_64x64_blocker"


def test_stage48_provenance_is_exactly_retained() -> None:
    endpoint = STAGE48_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 30763997986
    assert endpoint["workflow_job_id"] == 91539406116
    assert endpoint["tests_passed"] == 53
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8839298078
    assert endpoint["artifact_size_bytes"] == 106132
    assert endpoint["artifact_sha256"] == (
        "c13aa4a0e8fe4f3a7d8d845e65b29c2ea73069bd286ccc67d4495b39b575e2d9"
    )
    assert endpoint["source_head_sha"] == (
        "f6935fd8975dafb85a104662f9e607108ccd72f5"
    )
    assert endpoint["decision"] == (
        "projected_polar_56x56_improving_not_converged_"
        "stage49_64x64_confirmation"
    )

    retained = STAGE48_RETAINED_56X56_CASE
    assert retained["grid"] == [56, 56]
    assert retained["predicted_qav"] == 0.07608945114716073
    assert retained["qav_relative_error"] == 0.05679793259945467
    assert retained["velocity_metrics"]["relative_rms"] == 0.42365149047170675
    assert retained["velocity_metrics"]["sign_agreement"] == 0.8
    assert retained["maximum_phi_clipped_weight_fraction"] == 0.003059679149669645
    assert retained["maximum_psi_clipped_weight_fraction"] == 0.005069590310985741
