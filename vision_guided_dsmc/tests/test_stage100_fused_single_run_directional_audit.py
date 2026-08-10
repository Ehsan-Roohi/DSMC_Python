import json

import numpy as np
import pytest

from vgdsmc import stage100_fused_single_run_directional_audit as stage100
from vgdsmc.stage41_projected_polar_operator_audit import mapped_polar_quadrature
from vgdsmc.stage90_single_condition_reconstruction_solver_ab_audit import muscl_correction_divergence
from vgdsmc.stage98_directional_operator_growth_audit import _directional_metrics


def _decision_summary(*, x_share=0.5, y_share=0.5, x_growth=1.0, y_growth=1.0, cancellation=0.8):
    out = {}
    for distribution in ("phi", "psi"):
        out[distribution] = {
            "x_directional_abs_share": {"final": x_share},
            "y_directional_abs_share": {"final": y_share},
            "x_weighted_abs": {"final_to_first_ratio": x_growth},
            "y_weighted_abs": {"final_to_first_ratio": y_growth},
            "weighted_abs_cancellation_ratio": {"minimum": cancellation},
        }
    return out


def test_frozen_design_defaults():
    stage100.validate_stage100_design()


def test_frozen_design_rejects_retuning_or_cross_knudsen_change():
    with pytest.raises(ValueError):
        stage100.validate_stage100_design(kn0=0.1)
    with pytest.raises(ValueError):
        stage100.validate_stage100_design(same_run_parent_map_tolerance=1e-9)
    with pytest.raises(ValueError):
        stage100.validate_stage100_design(boundary_slope="one_sided")


def test_constants_preserve_stage99_and_failed_muscl_contract():
    assert stage100.STAGE99_RUN_ID == 31378863028
    assert stage100.STAGE99_ARTIFACT_ID == 9062898563
    assert stage100.SAME_RUN_PARENT_MAP_TOLERANCE == 1e-12
    assert stage100.BOUNDARY_SLOPE == "zero"


def test_relative_l2_identity_and_perturbation():
    a = np.arange(12.0).reshape(3, 4)
    assert stage100._relative_l2(a, a) == 0.0
    assert stage100._relative_l2(a + 1e-9, a) > 0.0


def test_stage99_authorization_accepts_only_retained_cross_run_blocker(tmp_path):
    root = tmp_path / "stage99"
    root.mkdir()
    payload = {
        "stage": 99,
        "decision": stage100.STAGE99_DECISION,
        "configuration": {"artifact_only": True},
        "final_replay_max_relative_l2": 1.1e-10,
        "final_replay_to_strict_tolerance_ratio": 110.0,
    }
    (root / "summary.json").write_text(json.dumps(payload))
    got = stage100._load_and_validate_stage99(root)
    assert got["decision"] == stage100.STAGE99_DECISION


def test_stage99_authorization_rejects_clean_or_wrong_endpoint(tmp_path):
    root = tmp_path / "stage99"
    root.mkdir()
    payload = {
        "stage": 99,
        "decision": stage100.STAGE99_DECISION,
        "configuration": {"artifact_only": True},
        "final_replay_max_relative_l2": 1e-13,
    }
    (root / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        stage100._load_and_validate_stage99(root)


def test_same_state_directional_map_matches_monolithic_parent_map():
    quadrature = mapped_polar_quadrature(4, 8, radial_scale=2.0)
    rng = np.random.default_rng(20260810)
    f = 0.5 + rng.random((5, 6, quadrature.vx.size))
    denominator = 1.0 + rng.random(f.shape)
    metrics, maps = _directional_metrics(f, denominator, quadrature)
    parent = muscl_correction_divergence(
        f,
        quadrature.vx,
        quadrature.vy,
        1.0 / 6.0,
        1.0 / 5.0,
        False,
    )
    parent_map = np.sum(
        np.abs(parent / denominator) * quadrature.weight[None, None, :],
        axis=-1,
    )
    assert stage100._relative_l2(maps["net_abs_m0"], parent_map) <= 1e-12
    assert metrics["decomposition_closure_relative_l2"] <= 1e-12


def test_decision_nonfinite_blocks_without_retuning():
    assert stage100.stage100_decision(_decision_summary(), 0.0, 0.0, False).startswith(
        "stage100_nonfinite"
    )


def test_decision_decomposition_mismatch_blocks_without_retuning():
    assert stage100.stage100_decision(_decision_summary(), 2e-12, 0.0, True).startswith(
        "stage100_same_run_decomposition_or_parent_mismatch"
    )


def test_decision_parent_mismatch_blocks_without_retuning():
    assert stage100.stage100_decision(_decision_summary(), 0.0, 2e-12, True).startswith(
        "stage100_same_run_decomposition_or_parent_mismatch"
    )


def test_decision_routes_x_dominant_growth():
    s = _decision_summary(x_share=0.70, y_share=0.30, x_growth=2.5, y_growth=1.2)
    assert stage100.stage100_decision(s, 0.0, 0.0, True) == (
        "stage100_x_dominant_growing_operator_stage101_x_signed_lobe_localization_audit"
    )


def test_decision_routes_y_dominant_growth():
    s = _decision_summary(x_share=0.30, y_share=0.70, x_growth=1.2, y_growth=2.5)
    assert stage100.stage100_decision(s, 0.0, 0.0, True) == (
        "stage100_y_dominant_growing_operator_stage101_y_signed_lobe_localization_audit"
    )


def test_decision_routes_cross_axis_cancellation():
    s = _decision_summary(x_share=0.50, y_share=0.50, x_growth=2.5, y_growth=2.5, cancellation=0.40)
    assert stage100.stage100_decision(s, 0.0, 0.0, True) == (
        "stage100_material_cross_axis_cancellation_stage101_signed_cancellation_localization_audit"
    )


def test_decision_routes_mixed_directional_velocity_sector_audit():
    s = _decision_summary(x_share=0.48, y_share=0.52, x_growth=2.5, y_growth=2.5, cancellation=0.80)
    assert stage100.stage100_decision(s, 0.0, 0.0, True) == (
        "stage100_mixed_directional_growth_stage101_interior_velocity_sector_audit"
    )
