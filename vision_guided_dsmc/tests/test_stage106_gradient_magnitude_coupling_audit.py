import json

import numpy as np
import pytest

from vgdsmc import stage106_gradient_magnitude_coupling_audit as stage106


def test_stage106_design_is_frozen():
    stage106.validate_stage106_design()
    with pytest.raises(ValueError):
        stage106.validate_stage106_design(magnitude_cosine_guard=0.9)
    with pytest.raises(ValueError):
        stage106.validate_stage106_design(source_relaxation=0.5)
    with pytest.raises(ValueError):
        stage106.validate_stage106_design(upper_quantile=0.8)


def test_centered_pearson_identical_and_reversed():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert np.isclose(stage106._centered_pearson(a, a), 1.0)
    assert np.isclose(stage106._centered_pearson(a, -a), -1.0)


def test_magnitude_coupling_identical_gradient_strength_is_unity():
    y, x = np.mgrid[:24, :24]
    arrays = {
        "phi_gx": 2.0 + 0.1 * x,
        "phi_gy": 1.0 + 0.05 * y,
        "psi_gx": 4.0 + 0.2 * x,
        "psi_gy": 2.0 + 0.10 * y,
    }
    metrics, out = stage106._magnitude_coupling_metrics(arrays)
    assert np.isclose(metrics["gradient_magnitude_cosine"], 1.0)
    assert np.isclose(metrics["gradient_magnitude_pearson"], 1.0)
    assert np.isclose(metrics["upper_quartile_overlap_coefficient"], 1.0)
    assert np.isclose(metrics["upper_quartile_jaccard"], 1.0)
    assert np.all(out["common_upper_quartile_mask"] <= 1)


def _metrics(cosine=0.9, pearson=0.8, overlap=0.6):
    return {
        "gradient_magnitude_cosine": cosine,
        "gradient_magnitude_pearson": pearson,
        "upper_quartile_overlap_coefficient": overlap,
    }


def test_stage106_decision_preserves_nonfinite_blocker():
    assert stage106.stage106_decision(_metrics(), 0.0, False) == "stage106_nonfinite_magnitude_metric_blocker_without_retuning"


def test_stage106_decision_preserves_parent_closure_blocker():
    assert stage106.stage106_decision(_metrics(), 2.0e-12, True) == "stage106_stage105_parent_closure_blocker_without_retuning"


def test_stage106_decision_routes_common_magnitude_coupling():
    assert stage106.stage106_decision(_metrics(), 0.0, True) == "stage106_common_gradient_magnitude_coupling_stage107_frozen_limiter_activation_colocation_audit"


def test_stage106_decision_routes_diffuse_upper_support():
    metrics = _metrics(overlap=0.4)
    assert stage106.stage106_decision(metrics, 0.0, True) == "stage106_global_magnitude_coupling_diffuse_upper_support_stage107_high_gradient_support_topology_audit"


def test_stage106_decision_routes_support_overlap_without_linear_coupling():
    metrics = _metrics(cosine=0.7, pearson=0.5, overlap=0.6)
    assert stage106.stage106_decision(metrics, 0.0, True) == "stage106_high_gradient_support_overlap_without_linear_coupling_stage107_rank_coupling_audit"


def test_stage106_decision_routes_amplitude_decoupling():
    metrics = _metrics(cosine=0.7, pearson=0.5, overlap=0.4)
    assert stage106.stage106_decision(metrics, 0.0, True) == "stage106_gradient_amplitude_decoupling_stage107_spatial_phase_amplitude_audit"


def test_stage105_loader_requires_exact_authorization_and_closure(tmp_path):
    cfg = {
        "grid": list(stage106.GRID),
        "kn0": stage106.KNUDSEN,
        "cold_hot_ratio": stage106.COLD_HOT_RATIO,
        "rule": list(stage106.RULE),
        "radial_scale": stage106.RADIAL_SCALE,
        "limiter": stage106.LIMITER,
        "boundary_slope": stage106.BOUNDARY_SLOPE,
        "source_relaxation": stage106.SOURCE_RELAXATION,
        "tolerance": stage106.TOLERANCE,
        "correction_floor": stage106.CORRECTION_FLOOR,
        "diagnostic_steps": stage106.DIAGNOSTIC_STEPS,
        "wall_band_cells": stage106.WALL_BAND_CELLS,
        "dominant_radial_shell": stage106.DOMINANT_RADIAL_SHELL,
        "failed_muscl_endpoint_rehabilitated": False,
        "one_sided_boundary_slope_promoted": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    summary = {
        "stage": 105,
        "decision": stage106.STAGE105_DECISION,
        "finite": True,
        "parent_closure_relative": 0.0,
        "configuration": cfg,
        "metrics": {
            "global_gradient_cosine": 1.0,
            "strongly_aligned_magnitude_share": 1.0,
            "principal_axis_anisotropy": {"phi": 0.1, "psi": 0.2},
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    shape = (stage106.INTERIOR_EXTENT, stage106.INTERIOR_EXTENT)
    gx = np.ones(shape)
    gy = np.zeros(shape)
    pair = np.ones(shape)
    local = np.ones(shape)
    np.savez_compressed(
        tmp_path / "directional_gradient_alignment_maps.npz",
        phi_gx=gx,
        phi_gy=gy,
        psi_gx=gx,
        psi_gy=gy,
        local_gradient_cosine=local,
        pair_gradient_weight=pair,
    )
    loaded, arrays, closure = stage106._load_and_validate_stage105(tmp_path)
    assert loaded["decision"] == stage106.STAGE105_DECISION
    assert arrays["phi_gx"].shape == shape
    assert closure <= stage106.PARENT_CLOSURE_TOLERANCE
    summary["decision"] = "wrong"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        stage106._load_and_validate_stage105(tmp_path)
