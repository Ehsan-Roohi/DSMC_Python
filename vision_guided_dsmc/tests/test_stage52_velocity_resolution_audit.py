from __future__ import annotations

import copy
import pytest

from vgdsmc.stage52_velocity_resolution_audit import (
    STAGE50_ARTIFACT,
    STAGE51_COMPLETED_ENDPOINT,
    STAGE52_BASELINE_CASE,
    STAGE52_BASELINE_RULE,
    STAGE52_GRID,
    STAGE52_KNUDSEN,
    STAGE52_MAX_ITERATIONS,
    STAGE52_MIN_ERROR_REDUCTION,
    STAGE52_MAX_VELOCITY_RMS_DEGRADATION,
    STAGE52_RATIO,
    STAGE52_RULES,
    STAGE52_SOURCE_RELAXATION,
    STAGE52_TOLERANCE,
    build_stage52_config,
    compare_to_baseline,
    stage52_decision,
    validate_stage52_design,
)


def test_stage52_design_is_frozen() -> None:
    validate_stage52_design(
        STAGE52_GRID,
        STAGE52_KNUDSEN,
        STAGE52_BASELINE_RULE,
        STAGE52_RULES,
        STAGE52_RATIO,
        STAGE52_MAX_ITERATIONS,
        STAGE52_TOLERANCE,
        STAGE52_SOURCE_RELAXATION,
    )
    with pytest.raises(ValueError):
        validate_stage52_design(
            (56, 56),
            STAGE52_KNUDSEN,
            STAGE52_BASELINE_RULE,
            STAGE52_RULES,
            STAGE52_RATIO,
            STAGE52_MAX_ITERATIONS,
            STAGE52_TOLERANCE,
            STAGE52_SOURCE_RELAXATION,
        )
    with pytest.raises(ValueError):
        validate_stage52_design(
            STAGE52_GRID,
            1.0,
            STAGE52_BASELINE_RULE,
            STAGE52_RULES,
            STAGE52_RATIO,
            STAGE52_MAX_ITERATIONS,
            STAGE52_TOLERANCE,
            STAGE52_SOURCE_RELAXATION,
        )


def test_stage52_rules_are_preregistered_orthogonal_refinements() -> None:
    assert STAGE52_BASELINE_RULE == (32, 96)
    assert STAGE52_RULES == (
        ("radial_refined", (40, 96)),
        ("angular_refined", (32, 120)),
        ("coupled_refined", (40, 120)),
    )
    assert STAGE52_MIN_ERROR_REDUCTION == 0.10
    assert STAGE52_MAX_VELOCITY_RMS_DEGRADATION == 0.10


def test_stage52_config_retains_stage50_physics_and_numerics() -> None:
    cfg = build_stage52_config()
    assert (cfg.nx, cfg.ny) == (64, 64)
    assert cfg.kn0 == 10.0
    assert cfg.cold_hot_ratio == 0.1
    assert cfg.viscosity_exponent == 0.5
    assert cfg.prandtl == 2.0 / 3.0
    assert cfg.max_steps == 3000
    assert cfg.cfl == 0.2
    assert cfg.tolerance == 2.0e-5
    assert cfg.check_interval == 25
    assert cfg.minimum_steps == 500
    assert cfg.positivity_floor == 1.0e-30


def test_stage51_and_stage50_provenance_is_exact() -> None:
    assert STAGE51_COMPLETED_ENDPOINT["workflow_run_id"] == 30781991799
    assert STAGE51_COMPLETED_ENDPOINT["workflow_job_id"] == 91588361434
    assert STAGE51_COMPLETED_ENDPOINT["tests_passed"] == 71
    assert STAGE51_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE51_COMPLETED_ENDPOINT["artifact_id"] == 8845163403
    assert STAGE51_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "95cbcb84bb57386a8d88d191566af4a2138ca62f49c3304737fd45aa7b89f974"
    )
    assert STAGE50_ARTIFACT["artifact_id"] == 8843553740
    assert STAGE50_ARTIFACT["kn10_fields_sha256"] == (
        "0d922ddfae58b26cd7e088e1ceadeec3109cd3e60d3e7f50057e4103d30359a0"
    )


def _case(
    name: str,
    qerr: float,
    vrms: float,
    sign: float = 1.0,
    *,
    converged: bool = True,
    stable: bool = True,
) -> dict[str, object]:
    case = copy.deepcopy(STAGE52_BASELINE_CASE)
    case["name"] = name
    case["qav_relative_error"] = qerr
    case["velocity_metrics"]["relative_rms"] = vrms
    case["velocity_metrics"]["sign_agreement"] = sign
    case["converged"] = converged
    if not stable:
        case["finite"] = False
    return case


def _comparisons(
    cases: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        case["name"]: compare_to_baseline(STAGE52_BASELINE_CASE, case)
        for case in cases
    }


def test_stage52_decision_accepts_material_radial_improvement() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    cases = [
        _case("radial_refined", 0.85 * base, vrms),
        _case("angular_refined", 1.01 * base, vrms),
        _case("coupled_refined", 0.95 * base, vrms),
    ]
    assert stage52_decision(cases, _comparisons(cases)) == (
        "radial_refined_materially_improves_stage53_cross_kn_confirmation"
    )


def test_stage52_decision_retains_small_or_mixed_effect() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    cases = [
        _case("radial_refined", 0.95 * base, vrms),
        _case("angular_refined", 0.99 * base, 1.2 * vrms),
        _case("coupled_refined", 1.01 * base, vrms),
    ]
    assert stage52_decision(cases, _comparisons(cases)) == (
        "velocity_resolution_small_or_mixed_effect_"
        "stage53_radial_mapping_tail_audit"
    )


def test_stage52_decision_retains_negative_endpoint() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    cases = [
        _case("radial_refined", 1.01 * base, vrms),
        _case("angular_refined", 1.02 * base, vrms),
        _case("coupled_refined", 1.03 * base, vrms),
    ]
    assert stage52_decision(cases, _comparisons(cases)) == (
        "velocity_point_count_does_not_explain_cross_kn_heat_flux_"
        "stage53_projected_collision_moment_audit"
    )


def test_stage52_decision_retains_nonconvergence_and_blocker() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    nonconverged = [
        _case("radial_refined", base, vrms, converged=False),
        _case("angular_refined", base, vrms),
        _case("coupled_refined", base, vrms),
    ]
    assert stage52_decision(nonconverged, _comparisons(nonconverged)) == (
        "stage52_velocity_resolution_stable_nonconverged_"
        "stage53_fixed_point_audit"
    )
    blocked = [
        _case("radial_refined", base, vrms, stable=False),
        _case("angular_refined", base, vrms, stable=False),
        _case("coupled_refined", base, vrms, stable=False),
    ]
    assert stage52_decision(blocked, _comparisons(blocked)) == (
        "stage52_velocity_resolution_numerical_blocker"
    )
