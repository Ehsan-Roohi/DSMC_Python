import json

import numpy as np
import pytest

from vgdsmc import stage104_interior_gradient_scale_audit as stage104


def test_stage104_design_is_frozen():
    stage104.validate_stage104_design()
    with pytest.raises(ValueError):
        stage104.validate_stage104_design(grid_scale_guard_cells=3.0)
    with pytest.raises(ValueError):
        stage104.validate_stage104_design(source_relaxation=0.5)
    with pytest.raises(ValueError):
        stage104.validate_stage104_design(lags_cells=(1, 2, 8))


def test_exact_interior_mask_has_frozen_56x56_support():
    mask = stage104._exact_interior_mask()
    assert mask.shape == stage104.GRID
    assert int(np.sum(mask)) == 56 * 56
    assert np.all(mask[4:60, 4:60])
    assert not np.any(mask[:4])
    assert not np.any(mask[60:])
    assert not np.any(mask[:, :4])
    assert not np.any(mask[:, 60:])


def test_lag_increment_ratio_zero_for_constant_field():
    field = np.ones((56, 56))
    assert stage104._lag_increment_ratio(field, 1) == 0.0
    assert stage104._lag_increment_ratio(field, 14) == 0.0


def test_lag_increment_ratio_rejects_invalid_lag():
    field = np.ones((56, 56))
    with pytest.raises(ValueError):
        stage104._lag_increment_ratio(field, 0)
    with pytest.raises(ValueError):
        stage104._lag_increment_ratio(field, 56)


def _embed(interior):
    out = np.zeros(stage104.GRID, dtype=float)
    out[4:60, 4:60] = interior
    return out


def test_growth_metrics_identify_streamwise_gradient_energy():
    x = np.arange(56, dtype=float)[None, :]
    growth = np.repeat(x, 56, axis=0)
    first = _embed(np.ones((56, 56)))
    final = first + _embed(growth)
    metrics, returned = stage104._growth_metrics(first, final)
    assert np.allclose(returned, growth)
    assert metrics["x_gradient_energy_share"] > 1.0 - 1.0e-14
    assert metrics["characteristic_gradient_length_cells"] > 1.0


def test_growth_metrics_identify_wall_normal_gradient_energy():
    y = np.arange(56, dtype=float)[:, None]
    growth = np.repeat(y, 56, axis=1)
    first = _embed(np.ones((56, 56)))
    final = first + _embed(growth)
    metrics, _ = stage104._growth_metrics(first, final)
    assert metrics["x_gradient_energy_share"] < 1.0e-14


def test_growth_metrics_positive_growth_share_is_one_for_positive_growth():
    first = _embed(np.ones((56, 56)))
    final = _embed(2.0 * np.ones((56, 56)))
    metrics, _ = stage104._growth_metrics(first, final)
    assert np.isclose(metrics["positive_growth_magnitude_share"], 1.0)
    assert metrics["minimum_growth"] == 1.0
    assert metrics["maximum_growth"] == 1.0


def _metrics_with_scale(scale):
    return {
        "phi": {"characteristic_gradient_length_cells": float(scale)},
        "psi": {"characteristic_gradient_length_cells": float(scale)},
    }


def test_stage104_decision_preserves_nonfinite_blocker():
    assert stage104.stage104_decision(_metrics_with_scale(4.0), 0.0, False) == "stage104_nonfinite_gradient_metric_blocker_without_retuning"


def test_stage104_decision_preserves_parent_closure_blocker():
    assert stage104.stage104_decision(_metrics_with_scale(4.0), 2.0e-12, True) == "stage104_stage103_parent_closure_blocker_without_retuning"


def test_stage104_decision_routes_grid_scale_gradient():
    metrics = {"phi": {"characteristic_gradient_length_cells": 1.5}, "psi": {"characteristic_gradient_length_cells": 4.0}}
    assert stage104.stage104_decision(metrics, 0.0, True) == "stage104_grid_scale_shell1_gradient_stage105_limiter_activation_audit"


def test_stage104_decision_routes_mesoscale_gradient():
    metrics = {"phi": {"characteristic_gradient_length_cells": 4.5}, "psi": {"characteristic_gradient_length_cells": 5.1}}
    assert stage104.stage104_decision(metrics, 0.0, True) == "stage104_mesoscale_shell1_gradient_stage105_directional_gradient_alignment_audit"


def test_stage104_decision_routes_broad_scale_gradient():
    metrics = {"phi": {"characteristic_gradient_length_cells": 8.0}, "psi": {"characteristic_gradient_length_cells": 9.0}}
    assert stage104.stage104_decision(metrics, 0.0, True) == "stage104_broad_scale_shell1_gradient_stage105_macroscopic_gradient_coupling_audit"


def test_stage103_loader_requires_exact_authorization_and_maps(tmp_path):
    cfg = {
        "grid": list(stage104.GRID),
        "kn0": stage104.KNUDSEN,
        "cold_hot_ratio": stage104.COLD_HOT_RATIO,
        "rule": list(stage104.RULE),
        "radial_scale": stage104.RADIAL_SCALE,
        "limiter": stage104.LIMITER,
        "boundary_slope": stage104.BOUNDARY_SLOPE,
        "source_relaxation": stage104.SOURCE_RELAXATION,
        "tolerance": stage104.TOLERANCE,
        "correction_floor": stage104.CORRECTION_FLOOR,
        "diagnostic_steps": stage104.DIAGNOSTIC_STEPS,
        "wall_band_cells": stage104.WALL_BAND_CELLS,
        "dominant_radial_shell": stage104.DOMINANT_RADIAL_SHELL,
        "failed_muscl_endpoint_rehabilitated": False,
        "cross_knudsen_extension_permitted": False,
        "physical_parameter_retuning": False,
    }
    summary = {
        "stage": 103,
        "decision": stage104.STAGE103_DECISION,
        "finite": True,
        "executed_steps": stage104.DIAGNOSTIC_STEPS,
        "maximum_stage102_shell_history_closure_relative": 1.0e-16,
        "configuration": cfg,
        "final_effective_tile_count": {"phi": 12.0, "psi": 12.0},
        "best_common_contiguous_2x2": {"score": 0.3},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    mask = stage104._exact_interior_mask()
    z = np.zeros(stage104.GRID)
    np.savez_compressed(
        tmp_path / "shell1_spatial_localization_histories.npz",
        interior_mask=mask,
        first_phi_shell1_cell_abs=z,
        final_phi_shell1_cell_abs=z,
        first_psi_shell1_cell_abs=z,
        final_psi_shell1_cell_abs=z,
    )
    loaded, arrays = stage104._load_and_validate_stage103(tmp_path)
    assert loaded["decision"] == stage104.STAGE103_DECISION
    assert arrays["interior_mask"].shape == stage104.GRID
    summary["decision"] = "wrong"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        stage104._load_and_validate_stage103(tmp_path)
