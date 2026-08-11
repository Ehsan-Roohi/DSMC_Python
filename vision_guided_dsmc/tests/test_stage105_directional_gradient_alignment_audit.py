import json

import numpy as np
import pytest

from vgdsmc import stage105_directional_gradient_alignment_audit as stage105


def test_stage105_design_is_frozen():
    stage105.validate_stage105_design()
    with pytest.raises(ValueError):
        stage105.validate_stage105_design(strong_alignment_cosine=0.9)
    with pytest.raises(ValueError):
        stage105.validate_stage105_design(source_relaxation=0.5)
    with pytest.raises(ValueError):
        stage105.validate_stage105_design(principal_axis_anisotropy_guard=0.1)


def test_principal_axis_identifies_x_gradient():
    y, x = np.mgrid[:16, :16]
    gx = np.ones_like(x, dtype=float)
    gy = np.zeros_like(y, dtype=float)
    angle, anisotropy = stage105._principal_axis(gx, gy)
    assert abs(angle) < 1.0e-14
    assert np.isclose(anisotropy, 1.0)


def test_orientation_difference_is_axis_not_vector_difference():
    assert np.isclose(stage105._orientation_difference_degrees(89.0, -89.0), 2.0)
    assert np.isclose(stage105._orientation_difference_degrees(0.0, 90.0), 90.0)


def test_alignment_metrics_identical_gradients_are_strongly_aligned():
    y, x = np.mgrid[:24, :24]
    phi = 2.0 * x + y
    psi = 4.0 * x + 2.0 * y
    metrics, arrays = stage105._alignment_metrics(phi, psi)
    assert np.isclose(metrics["global_gradient_cosine"], 1.0)
    assert np.isclose(metrics["strongly_aligned_magnitude_share"], 1.0)
    assert np.isclose(metrics["strongly_opposed_magnitude_share"], 0.0)
    assert np.isclose(metrics["positive_dot_magnitude_share"], 1.0)
    assert np.allclose(arrays["local_gradient_cosine"], 1.0)


def test_alignment_metrics_opposed_gradients_are_strongly_opposed():
    y, x = np.mgrid[:24, :24]
    phi = 2.0 * x + y
    psi = -phi
    metrics, _ = stage105._alignment_metrics(phi, psi)
    assert np.isclose(metrics["global_gradient_cosine"], -1.0)
    assert np.isclose(metrics["strongly_aligned_magnitude_share"], 0.0)
    assert np.isclose(metrics["strongly_opposed_magnitude_share"], 1.0)
    assert np.isclose(metrics["positive_dot_magnitude_share"], 0.0)


def test_same_sign_product_share_distinguishes_opposed_components():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    share = stage105._same_sign_product_share(a, b)
    assert np.isclose(share, 10.0 / 14.0)


def _metrics(cosine=0.9, aligned=0.8, opposed=0.0, anis_phi=0.3, anis_psi=0.3):
    return {
        "global_gradient_cosine": cosine,
        "strongly_aligned_magnitude_share": aligned,
        "strongly_opposed_magnitude_share": opposed,
        "principal_axis_anisotropy": {"phi": anis_phi, "psi": anis_psi},
    }


def test_stage105_decision_preserves_nonfinite_blocker():
    assert stage105.stage105_decision(_metrics(), 0.0, False) == "stage105_nonfinite_alignment_metric_blocker_without_retuning"


def test_stage105_decision_preserves_parent_closure_blocker():
    assert stage105.stage105_decision(_metrics(), 2.0e-12, True) == "stage105_stage104_parent_closure_blocker_without_retuning"


def test_stage105_decision_routes_common_axis_alignment():
    assert stage105.stage105_decision(_metrics(), 0.0, True) == "stage105_common_strong_axis_alignment_stage106_directional_limiter_activation_audit"


def test_stage105_decision_routes_common_alignment_without_axis_dominance():
    metrics = _metrics(anis_phi=0.15, anis_psi=0.18)
    assert stage105.stage105_decision(metrics, 0.0, True) == "stage105_common_gradient_alignment_without_axis_dominance_stage106_gradient_magnitude_coupling_audit"


def test_stage105_decision_routes_opposed_alignment():
    metrics = _metrics(cosine=-0.9, aligned=0.0, opposed=0.8)
    assert stage105.stage105_decision(metrics, 0.0, True) == "stage105_common_opposed_gradient_stage106_signed_cancellation_audit"


def test_stage105_decision_routes_mixed_alignment():
    metrics = _metrics(cosine=0.4, aligned=0.3, opposed=0.1)
    assert stage105.stage105_decision(metrics, 0.0, True) == "stage105_mixed_gradient_alignment_stage106_spatial_phase_relation_audit"


def test_stage104_loader_requires_exact_authorization_and_maps(tmp_path):
    cfg = {
        "grid": list(stage105.GRID),
        "kn0": stage105.KNUDSEN,
        "cold_hot_ratio": stage105.COLD_HOT_RATIO,
        "rule": list(stage105.RULE),
        "radial_scale": stage105.RADIAL_SCALE,
        "limiter": stage105.LIMITER,
        "boundary_slope": stage105.BOUNDARY_SLOPE,
        "source_relaxation": stage105.SOURCE_RELAXATION,
        "tolerance": stage105.TOLERANCE,
        "correction_floor": stage105.CORRECTION_FLOOR,
        "diagnostic_steps": stage105.DIAGNOSTIC_STEPS,
        "wall_band_cells": stage105.WALL_BAND_CELLS,
        "dominant_radial_shell": stage105.DOMINANT_RADIAL_SHELL,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    summary = {
        "stage": 104,
        "decision": stage105.STAGE104_DECISION,
        "finite": True,
        "configuration": cfg,
        "metrics": {
            "phi": {
                "characteristic_gradient_length_cells": 4.5,
                "positive_growth_magnitude_share": 1.0,
                "gradient_energy": 1.0,
                "x_gradient_energy_share": 0.5,
            },
            "psi": {
                "characteristic_gradient_length_cells": 5.0,
                "positive_growth_magnitude_share": 1.0,
                "gradient_energy": 1.0,
                "x_gradient_energy_share": 0.5,
            },
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    mask = np.zeros(stage105.GRID, dtype=bool)
    mask[4:60, 4:60] = True
    z = np.zeros((56, 56))
    np.savez_compressed(
        tmp_path / "interior_gradient_scale_maps.npz",
        phi_growth_map=z,
        psi_growth_map=z,
        interior_mask=mask,
        lags_cells=np.array([1, 2, 4, 7, 14]),
    )
    loaded, arrays = stage105._load_and_validate_stage104(tmp_path)
    assert loaded["decision"] == stage105.STAGE104_DECISION
    assert arrays["phi_growth_map"].shape == (56, 56)
    summary["decision"] = "wrong"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        stage105._load_and_validate_stage104(tmp_path)
