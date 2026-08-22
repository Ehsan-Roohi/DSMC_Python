from __future__ import annotations

import copy
import pytest

from vgdsmc.stage44_projected_polar_finer_grid_sequence import (
    STAGE43_COMPLETED_ENDPOINT,
    STAGE43_FINEST_CASE,
    STAGE44_GRIDS,
    STAGE44_KNUDSEN,
    STAGE44_MAX_ITERATIONS,
    STAGE44_RATIO,
    STAGE44_RULE,
    STAGE44_SOURCE_RELAXATION,
    STAGE44_TOLERANCE,
    build_stage44_config,
    stage44_decision,
    validate_stage44_design,
)


def test_stage44_design_is_frozen() -> None:
    validate_stage44_design(
        STAGE44_GRIDS,
        STAGE44_RULE,
        STAGE44_KNUDSEN,
        STAGE44_RATIO,
        STAGE44_MAX_ITERATIONS,
        STAGE44_TOLERANCE,
        STAGE44_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage44_design(
            ((20, 20), (28, 28)),
            STAGE44_RULE,
            STAGE44_KNUDSEN,
            STAGE44_RATIO,
            STAGE44_MAX_ITERATIONS,
            STAGE44_TOLERANCE,
            STAGE44_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage44_design(
            STAGE44_GRIDS,
            (40, 96),
            STAGE44_KNUDSEN,
            STAGE44_RATIO,
            STAGE44_MAX_ITERATIONS,
            STAGE44_TOLERANCE,
            STAGE44_SOURCE_RELAXATION,
        )


def test_stage44_config_retains_physics_and_numerics() -> None:
    cfg = build_stage44_config((20, 20))
    assert (cfg.nx, cfg.ny) == (20, 20)
    assert cfg.kn0 == 0.1
    assert cfg.cold_hot_ratio == 0.1
    assert cfg.viscosity_exponent == 0.5
    assert cfg.prandtl == 2.0 / 3.0
    assert cfg.max_steps == 3000
    assert cfg.tolerance == 2.0e-5
    assert cfg.check_interval == 25
    assert cfg.minimum_steps == 500
    assert cfg.positivity_floor == 1.0e-30
    with pytest.raises(ValueError):
        build_stage44_config((28, 28))


def _case(qav: float, qerr: float, vrms: float, scale: float) -> dict[str, object]:
    return {
        "predicted_qav": qav,
        "qav_relative_error": qerr,
        "velocity_metrics": {
            "relative_rms": vrms,
            "relative_l1": vrms,
            "sign_agreement": 0.2,
        },
        "table_velocity": [scale * (index + 1) for index in range(10)],
        "finite": True,
        "minimum_phi": 1.0e-30,
        "minimum_psi": 1.0e-30,
        "wall_mass_balance_relative_error": 1.0e-16,
        "converged": True,
    }


def test_stage44_decision_retains_positive_and_negative_endpoints() -> None:
    retained = copy.deepcopy(STAGE43_FINEST_CASE)
    converging = [
        _case(0.0795, 0.104, 1.70, 1.00),
        _case(0.0780, 0.083, 1.45, 0.96),
    ]
    assert stage44_decision(retained, converging) == (
        "projected_polar_finer_grid_converging_stage45_cross_kn_extension"
    )

    improving_not_converged = [
        _case(0.0795, 0.104, 1.70, 1.00),
        _case(0.0760, 0.056, 1.45, 0.75),
    ]
    assert stage44_decision(retained, improving_not_converged) == (
        "projected_polar_finer_grid_improving_not_converged_stage45_32x32_confirmation"
    )

    velocity_worsens = [
        _case(0.0795, 0.104, 2.20, 1.00),
        _case(0.0780, 0.083, 2.30, 0.96),
    ]
    assert stage44_decision(retained, velocity_worsens) == (
        "projected_polar_heat_flux_only_improves_stage45_wall_observable_audit"
    )


def test_stage43_provenance_is_exactly_retained() -> None:
    assert STAGE43_COMPLETED_ENDPOINT["workflow_run_id"] == 30733957628
    assert STAGE43_COMPLETED_ENDPOINT["workflow_job_id"] == 91459088187
    assert STAGE43_COMPLETED_ENDPOINT["tests_passed"] == 53
    assert STAGE43_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE43_COMPLETED_ENDPOINT["artifact_id"] == 8830544995
    assert STAGE43_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "70bb2a08a1e49cfe31091ce91cf26c286c7652ae7c341855a4ec1dc81de528dd"
    )
    assert STAGE43_FINEST_CASE["predicted_qav"] == 0.0814053328666412
    assert STAGE43_FINEST_CASE["qav_relative_error"] == 0.13062962314779453
    assert STAGE43_FINEST_CASE["velocity_metrics"]["relative_rms"] == (
        2.0834768337668104
    )
