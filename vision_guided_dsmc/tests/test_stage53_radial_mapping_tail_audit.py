from __future__ import annotations

import copy
import pytest

from vgdsmc.stage41_projected_polar_operator_audit import mapped_polar_quadrature
from vgdsmc.stage53_radial_mapping_tail_audit import (
    STAGE52_COMPLETED_ENDPOINT,
    STAGE53_AUDIT_SCALES,
    STAGE53_BASELINE_SCALE,
    STAGE53_GRID,
    STAGE53_KNUDSEN,
    STAGE53_MAX_TAIL_MOMENT_ERROR,
    STAGE53_MAX_VELOCITY_RMS_DEGRADATION,
    STAGE53_MIN_ERROR_REDUCTION,
    STAGE53_RATIO,
    STAGE53_RULE,
    STAGE53_SOURCE_RELAXATION,
    STAGE53_TAIL_TEMPERATURES,
    build_stage53_config,
    compare_to_baseline,
    maxwellian_tail_moment_audit,
    stage53_decision,
    validate_stage53_design,
)
from vgdsmc.stage52_velocity_resolution_audit import STAGE52_BASELINE_CASE


def test_stage53_design_is_frozen() -> None:
    validate_stage53_design(
        STAGE53_GRID,
        STAGE53_KNUDSEN,
        STAGE53_RULE,
        STAGE53_BASELINE_SCALE,
        STAGE53_AUDIT_SCALES,
        STAGE53_RATIO,
        STAGE53_SOURCE_RELAXATION,
        STAGE53_TAIL_TEMPERATURES,
    )
    with pytest.raises(ValueError):
        validate_stage53_design(
            STAGE53_GRID,
            STAGE53_KNUDSEN,
            (40, 96),
            STAGE53_BASELINE_SCALE,
            STAGE53_AUDIT_SCALES,
            STAGE53_RATIO,
            STAGE53_SOURCE_RELAXATION,
            STAGE53_TAIL_TEMPERATURES,
        )
    with pytest.raises(ValueError):
        validate_stage53_design(
            STAGE53_GRID,
            STAGE53_KNUDSEN,
            STAGE53_RULE,
            STAGE53_BASELINE_SCALE,
            (("compressed_tail", 0.75), ("expanded_tail", 1.5)),
            STAGE53_RATIO,
            STAGE53_SOURCE_RELAXATION,
            STAGE53_TAIL_TEMPERATURES,
        )


def test_stage53_scales_are_symmetric_factor_of_two_audit() -> None:
    assert STAGE53_RULE == (32, 96)
    assert STAGE53_BASELINE_SCALE == 1.0
    assert STAGE53_AUDIT_SCALES == (
        ("compressed_tail", 0.5),
        ("expanded_tail", 2.0),
    )
    assert STAGE53_MIN_ERROR_REDUCTION == 0.10
    assert STAGE53_MAX_VELOCITY_RMS_DEGRADATION == 0.10
    assert STAGE53_MAX_TAIL_MOMENT_ERROR == 1.0e-3


def test_stage53_config_retains_stage52_physics_and_numerics() -> None:
    cfg = build_stage53_config()
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


def test_stage52_provenance_is_exact() -> None:
    assert STAGE52_COMPLETED_ENDPOINT["workflow_run_id"] == 30787293128
    assert STAGE52_COMPLETED_ENDPOINT["workflow_job_id"] == 91603227075
    assert STAGE52_COMPLETED_ENDPOINT["tests_passed"] == 79
    assert STAGE52_COMPLETED_ENDPOINT["tests_failed"] == 0
    assert STAGE52_COMPLETED_ENDPOINT["artifact_id"] == 8851319553
    assert STAGE52_COMPLETED_ENDPOINT["artifact_sha256"] == (
        "3c496189886de4b764ff13f7bfd737415de874329b4ecd191c270d90c97fca4b"
    )
    assert STAGE52_COMPLETED_ENDPOINT["summary_sha256"] == (
        "2e707e3220ad4a54c16e861895ad042cc517c1439a928148046f4be5d968ff10"
    )


def test_stage53_maxwellian_tail_audit_covers_cold_and_hot_walls() -> None:
    results = []
    for scale in (0.5, 1.0, 2.0):
        audit = maxwellian_tail_moment_audit(
            mapped_polar_quadrature(*STAGE53_RULE, radial_scale=scale)
        )
        assert [entry["temperature"] for entry in audit["temperatures"]] == [
            0.1,
            1.0,
        ]
        assert audit["maximum_relative_error"] <= STAGE53_MAX_TAIL_MOMENT_ERROR
        assert audit["passes_preregistered_tail_closure"] is True
        results.append(audit)
    assert results[0]["maximum_radius"] < results[1]["maximum_radius"]
    assert results[1]["maximum_radius"] < results[2]["maximum_radius"]


def _case(
    name: str,
    qerr: float,
    vrms: float,
    sign: float = 1.0,
    *,
    converged: bool = True,
    stable: bool = True,
    tail_passes: bool = True,
) -> dict[str, object]:
    case = copy.deepcopy(STAGE52_BASELINE_CASE)
    case["name"] = name
    case["qav_relative_error"] = qerr
    case["velocity_metrics"]["relative_rms"] = vrms
    case["velocity_metrics"]["sign_agreement"] = sign
    case["converged"] = converged
    case["tail_moment_audit"] = {
        "maximum_relative_error": 1.0e-6 if tail_passes else 1.0e-2,
        "passes_preregistered_tail_closure": tail_passes,
    }
    if not stable:
        case["finite"] = False
    return case


def _comparisons(cases: list[dict[str, object]]):
    return {
        case["name"]: compare_to_baseline(STAGE52_BASELINE_CASE, case)
        for case in cases
    }


def test_stage53_decision_accepts_only_material_supported_improvement() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    cases = [
        _case("compressed_tail", 0.85 * base, vrms),
        _case("expanded_tail", 0.95 * base, vrms),
    ]
    assert stage53_decision(cases, _comparisons(cases)) == (
        "compressed_tail_materially_improves_stage54_cross_kn_confirmation"
    )

    unsupported = [
        _case("compressed_tail", 0.85 * base, vrms, tail_passes=False),
        _case("expanded_tail", base, vrms),
    ]
    assert stage53_decision(unsupported, _comparisons(unsupported)) == (
        "radial_mapping_tail_does_not_explain_cross_kn_heat_flux_"
        "stage54_projected_collision_moment_audit"
    )


def test_stage53_decision_preserves_mixed_and_negative_results() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    changed = [
        _case("compressed_tail", 0.96 * base, vrms),
        _case("expanded_tail", 1.05 * base, vrms),
    ]
    changed[0]["predicted_qav"] = 0.95 * float(
        STAGE52_BASELINE_CASE["predicted_qav"]
    )
    assert stage53_decision(changed, _comparisons(changed)) == (
        "radial_mapping_changes_solution_without_material_benchmark_"
        "improvement_stage54_projected_collision_moment_audit"
    )

    negligible = [
        _case("compressed_tail", 0.999 * base, vrms),
        _case("expanded_tail", 1.001 * base, vrms),
    ]
    assert stage53_decision(negligible, _comparisons(negligible)) == (
        "radial_mapping_tail_does_not_explain_cross_kn_heat_flux_"
        "stage54_projected_collision_moment_audit"
    )


def test_stage53_decision_retains_nonconvergence_and_blocker() -> None:
    base = float(STAGE52_BASELINE_CASE["qav_relative_error"])
    vrms = float(STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"])
    nonconverged = [
        _case("compressed_tail", base, vrms, converged=False),
        _case("expanded_tail", base, vrms),
    ]
    assert stage53_decision(nonconverged, _comparisons(nonconverged)) == (
        "stage53_radial_mapping_tail_stable_nonconverged_"
        "stage54_fixed_point_audit"
    )
    blocked = [
        _case("compressed_tail", base, vrms, stable=False),
        _case("expanded_tail", base, vrms),
    ]
    assert stage53_decision(blocked, _comparisons(blocked)) == (
        "stage53_radial_mapping_tail_numerical_blocker"
    )
