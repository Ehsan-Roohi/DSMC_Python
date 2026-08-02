from __future__ import annotations

import copy
import pytest

from vgdsmc.stage45_projected_polar_32x32_confirmation import (
    STAGE44_COMPLETED_ENDPOINT,
    STAGE44_RETAINED_CASES,
    STAGE45_GRID,
    STAGE45_KNUDSEN,
    STAGE45_MAX_ITERATIONS,
    STAGE45_RATIO,
    STAGE45_RULE,
    STAGE45_SOURCE_RELAXATION,
    STAGE45_TOLERANCE,
    build_stage45_config,
    stage45_decision,
    validate_stage45_design,
)


def test_stage45_design_is_frozen() -> None:
    validate_stage45_design(
        STAGE45_GRID,
        STAGE45_RULE,
        STAGE45_KNUDSEN,
        STAGE45_RATIO,
        STAGE45_MAX_ITERATIONS,
        STAGE45_TOLERANCE,
        STAGE45_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage45_design(
            (28, 28),
            STAGE45_RULE,
            STAGE45_KNUDSEN,
            STAGE45_RATIO,
            STAGE45_MAX_ITERATIONS,
            STAGE45_TOLERANCE,
            STAGE45_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage45_design(
            STAGE45_GRID,
            (40, 96),
            STAGE45_KNUDSEN,
            STAGE45_RATIO,
            STAGE45_MAX_ITERATIONS,
            STAGE45_TOLERANCE,
            STAGE45_SOURCE_RELAXATION,
        )


def test_stage45_config_retains_physics_and_numerics() -> None:
    cfg = build_stage45_config()
    assert (cfg.nx, cfg.ny) == (32, 32)
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
    scale: float,
    *,
    converged: bool = True,
) -> dict[str, object]:
    return {
        "predicted_qav": qav,
        "qav_relative_error": qerr,
        "velocity_metrics": {
            "relative_rms": vrms,
            "relative_l1": vrms,
            "sign_agreement": 0.4,
        },
        "table_velocity": [scale * (index + 1) for index in range(10)],
        "finite": True,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "wall_mass_balance_relative_error": 1.0e-16,
        "converged": converged,
    }


def test_stage45_decision_retains_all_endpoints() -> None:
    retained = copy.deepcopy(STAGE44_RETAINED_CASES[-1])

    converging = _case(0.0780, 0.083, 1.10, 0.98)
    assert stage45_decision(retained, converging) == (
        "projected_polar_32x32_converging_stage46_cross_kn_extension"
    )

    unresolved = _case(0.0760, 0.056, 1.10, 0.70)
    assert stage45_decision(retained, unresolved) == (
        "projected_polar_32x32_improving_not_converged_"
        "stage46_40x40_confirmation"
    )

    heat_only = _case(0.0780, 0.083, 1.40, 0.98)
    assert stage45_decision(retained, heat_only) == (
        "projected_polar_32x32_heat_flux_only_improves_"
        "stage46_wall_observable_audit"
    )

    velocity_only = _case(0.0800, 0.111, 1.10, 0.98)
    assert stage45_decision(retained, velocity_only) == (
        "projected_polar_32x32_velocity_only_improves_"
        "stage46_heat_flux_definition_audit"
    )

    blocker = _case(0.0780, 0.083, 1.10, 0.98, converged=False)
    assert stage45_decision(retained, blocker) == "stage45_32x32_blocker"


def test_stage44_provenance_is_exactly_retained() -> None:
    assert STAGE44_COMPLETED_ENDPOINT["workflow_run_id"] == 30743552934
    assert STAGE44_COMPLETED_ENDPOINT["workflow_job_id"] == 91485129929
    assert STAGE44_COMPLETED_ENDPOINT["tests_passed"] == 37
    assert STAGE44_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE44_COMPLETED_ENDPOINT["artifact_id"] == 8832382991
    assert STAGE44_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "3eabd271c4b05ccdd6c04d462398ecf08cb70411ef1b977424171a619df6de34"
    )
    retained_24 = STAGE44_RETAINED_CASES[-1]
    assert retained_24["grid"] == [24, 24]
    assert retained_24["predicted_qav"] == 0.07888585433050331
    assert retained_24["qav_relative_error"] == 0.09563686570143495
    assert retained_24["velocity_metrics"]["relative_rms"] == (
        1.2973287076686175
    )
