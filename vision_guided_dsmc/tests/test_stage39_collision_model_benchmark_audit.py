from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage39_collision_model_benchmark_audit import (
    STAGE38_COMPLETED_ENDPOINT,
    STAGE39_CFL,
    STAGE39_COORDINATE_VARIANTS,
    STAGE39_GRID,
    STAGE39_MODELS,
    coordinate_variant_metrics,
    profile_coordinate_variants,
    stage39_decision,
    validate_stage39_design,
)


def _variant_metrics(direct_rms: float, direct_sign: float = 0.9):
    return {
        "direct": {
            "relative_rms": direct_rms,
            "sign_agreement": direct_sign,
            "relative_l1": direct_rms,
        },
        "reverse_y": {
            "relative_rms": direct_rms,
            "sign_agreement": direct_sign,
            "relative_l1": direct_rms,
        },
        "flip_tangential_sign": {
            "relative_rms": direct_rms,
            "sign_agreement": direct_sign,
            "relative_l1": direct_rms,
        },
        "reverse_y_and_flip_sign": {
            "relative_rms": direct_rms,
            "sign_agreement": direct_sign,
            "relative_l1": direct_rms,
        },
    }


def _row(
    q_error: float,
    velocity_rms: float,
    *,
    sign: float = 0.9,
    converged: bool = True,
):
    return {
        "converged": converged,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": velocity_rms,
            "sign_agreement": sign,
            "relative_l1": velocity_rms,
        },
        "coordinate_variant_metrics": _variant_metrics(velocity_rms, sign),
    }


def _reproduction(ok: bool = True):
    return {"within_tolerance": ok}


def test_stage39_frozen_design_accepts_only_preregistered_case():
    validate_stage39_design(STAGE39_GRID, STAGE39_CFL, STAGE39_MODELS, 16000, 2e-5)
    assert STAGE38_COMPLETED_ENDPOINT["workflow_run_id"] == 30714512141
    assert STAGE38_COMPLETED_ENDPOINT["tests_passed"] == 54
    assert STAGE38_COMPLETED_ENDPOINT["artifact_id"] == 8823968646


def test_stage39_rejects_grid_retuning():
    with pytest.raises(ValueError, match="24x24"):
        validate_stage39_design((36, 36), STAGE39_CFL, STAGE39_MODELS, 16000, 2e-5)


def test_stage39_rejects_cfl_retuning():
    with pytest.raises(ValueError, match="CFL"):
        validate_stage39_design(STAGE39_GRID, 0.1, STAGE39_MODELS, 16000, 2e-5)


def test_stage39_rejects_noncanonical_model_list():
    with pytest.raises(ValueError, match="canonical"):
        validate_stage39_design(
            STAGE39_GRID,
            STAGE39_CFL,
            (("fitted_prandtl", 0.72),),
            16000,
            2e-5,
        )


def test_profile_coordinate_variants_are_exact_and_complete():
    profile = np.array([1.0, -2.0, 3.0])
    variants = profile_coordinate_variants(profile)
    assert tuple(variants) == STAGE39_COORDINATE_VARIANTS
    np.testing.assert_allclose(variants["direct"], [1.0, -2.0, 3.0])
    np.testing.assert_allclose(variants["reverse_y"], [3.0, -2.0, 1.0])
    np.testing.assert_allclose(variants["flip_tangential_sign"], [-1.0, 2.0, -3.0])
    np.testing.assert_allclose(
        variants["reverse_y_and_flip_sign"], [-3.0, 2.0, -1.0]
    )


def test_profile_coordinate_variants_reject_nonvector_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        profile_coordinate_variants(np.ones((2, 2)))


def test_coordinate_variant_metrics_are_finite_for_table_length_profile():
    metrics = coordinate_variant_metrics(np.linspace(-0.003, 0.003, 10))
    assert tuple(metrics) == STAGE39_COORDINATE_VARIANTS
    for variant in metrics.values():
        assert np.isfinite(variant["relative_rms"])
        assert np.isfinite(variant["relative_l1"])
        assert 0.0 <= variant["sign_agreement"] <= 1.0


def test_stage39_decision_blocks_on_failed_stage38_reproduction():
    decision = stage39_decision(_row(0.01, 1.0), _row(0.02, 1.1), _reproduction(False))
    assert decision == "stage39_reproduction_mismatch_blocker"


def test_stage39_decision_preserves_nonconvergence():
    decision = stage39_decision(
        _row(0.01, 1.0, converged=False),
        _row(0.02, 1.1),
        _reproduction(True),
    )
    assert decision == "stage39_nonconvergence_stage40_numerical_stability_audit"


def test_stage39_decision_covers_convention_collision_and_negative_paths():
    shakhov = _row(0.10, 1.0)
    bgk = _row(0.10, 1.0)
    shakhov["coordinate_variant_metrics"]["reverse_y"] = {
        "relative_rms": 0.4,
        "sign_agreement": 0.9,
        "relative_l1": 0.4,
    }
    bgk["coordinate_variant_metrics"]["reverse_y"] = {
        "relative_rms": 0.4,
        "sign_agreement": 0.9,
        "relative_l1": 0.4,
    }
    assert stage39_decision(shakhov, bgk, _reproduction()) == (
        "benchmark_coordinate_convention_flag_stage40_source_table_audit"
    )

    shakhov = _row(0.10, 1.0)
    bgk = _row(0.05, 0.5)
    assert stage39_decision(shakhov, bgk, _reproduction()) == (
        "canonical_bgk_sensitivity_stage40_independent_collision_model_audit"
    )

    shakhov = _row(0.10, 1.0)
    bgk = _row(0.15, 1.05)
    assert stage39_decision(shakhov, bgk, _reproduction()) == (
        "mixed_collision_model_sensitivity_stage40_external_benchmark_audit"
    )

    shakhov = _row(0.10, 1.0)
    bgk = _row(0.11, 1.05)
    assert stage39_decision(shakhov, bgk, _reproduction()) == (
        "collision_model_and_simple_conventions_do_not_explain_stage40_external_reference_audit"
    )
