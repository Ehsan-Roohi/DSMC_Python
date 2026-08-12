import json

import numpy as np
import pytest

from vgdsmc import stage112_x_axis_spatial_localization_audit as s


def _block_from_density(density: np.ndarray) -> dict[str, object]:
    severity = np.sqrt(density)
    growth = np.sqrt(density)
    metrics, _, _ = s._spatial_metrics(severity, growth)
    return metrics


def _metrics_from_density(phi: np.ndarray, psi: np.ndarray, common: np.ndarray) -> dict[str, dict[str, object]]:
    return {
        "phi": _block_from_density(phi),
        "psi": _block_from_density(psi),
        "common": _block_from_density(common),
    }


def test_stage112_design_is_frozen():
    s.validate_stage112_design()
    with pytest.raises(ValueError):
        s.validate_stage112_design(limiter="vanleer")
    with pytest.raises(ValueError):
        s.validate_stage112_design(outer_x_pair_share_guard=0.6)
    with pytest.raises(ValueError):
        s.validate_stage112_design(stage111_run_id=-1)


def test_stage112_tile_partition_is_exact_and_equal_area():
    tiles = s._tile_index()
    assert tiles.shape == (56, 56)
    assert set(np.unique(tiles)) == set(range(16))
    for k in range(16):
        assert np.count_nonzero(tiles == k) == 14 * 14


def test_stage112_coupled_density_is_exact_product():
    severity = np.full((56, 56), 2.0)
    growth = np.full((56, 56), 3.0)
    density = s._coupled_density(severity, growth)
    assert np.all(density == 6.0)


def test_stage112_uniform_field_has_uniform_spatial_metrics():
    field = np.ones((56, 56))
    metrics, density, share = s._spatial_metrics(field, field)
    assert np.isclose(np.sum(share), 1.0)
    assert np.allclose(metrics["tile_share"], np.full(16, 1.0 / 16.0))
    assert np.allclose(metrics["x_band_share"], np.full(4, 0.25))
    assert np.isclose(metrics["outer_x_quarter_pair_share"], 0.5)
    assert np.isclose(metrics["outer_to_inner_share_ratio"], 1.0)
    assert np.isclose(metrics["effective_tile_count"], 16.0)
    assert np.all(density == 1.0)


def test_stage112_decision_symmetric_outer_x_quarter_route():
    density = np.ones((56, 56))
    density[:, :14] = 8.0
    density[:, -14:] = 8.0
    metrics = _metrics_from_density(density, density, density)
    assert s.stage112_decision(metrics, True) == (
        "stage112_symmetric_outer_x_quarter_localization_stage113_x_wall_distance_profile_audit"
    )


def test_stage112_decision_common_single_tile_route():
    density = np.full((56, 56), 1.0e-9)
    density[14:28, 14:28] = 1.0
    metrics = _metrics_from_density(density, density, density)
    assert s.stage112_decision(metrics, True) == (
        "stage112_common_single_tile_5_stage113_local_coordinate_audit"
    )


def test_stage112_decision_common_contiguous_2x2_route():
    density = np.full((56, 56), 1.0e-9)
    density[14:42, 14:42] = 1.0
    metrics = _metrics_from_density(density, density, density)
    assert s.stage112_decision(metrics, True) == (
        "stage112_common_contiguous_2x2_1_1_stage113_local_coordinate_audit"
    )


def test_stage112_decision_diffuse_route():
    density = np.ones((56, 56))
    metrics = _metrics_from_density(density, density, density)
    assert s.stage112_decision(metrics, True) == (
        "stage112_x_axis_coupling_spatially_diffuse_stage113_x_gradient_lengthscale_audit"
    )


def test_stage112_decision_preserves_nonfinite_blocker():
    assert s.stage112_decision({}, False) == (
        "stage112_nonfinite_spatial_localization_blocker_without_retuning"
    )


def test_stage112_loader_rejects_missing_exact_parent(tmp_path):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError):
        s._load_stage111(tmp_path, record)
