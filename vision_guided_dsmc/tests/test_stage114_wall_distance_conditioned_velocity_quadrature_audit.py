import numpy as np
import pytest

from vgdsmc import stage114_wall_distance_conditioned_velocity_quadrature_audit as s114


def _block(share, maximum, effective=5.0):
    values = np.full(s114.ANGULAR_SECTORS, (1.0 - maximum) / (s114.ANGULAR_SECTORS - 1))
    values[share] = maximum
    return {
        "total_conditioned_mass": 1.0,
        "total_conditioned_mass_share": 1.0 / 3.0,
        "sector_share": values.tolist(),
        "maximum_sector_index": share,
        "maximum_sector_share": maximum,
        "effective_sector_count": effective,
    }


def _metrics(near_phi=(0, 0.2, 5.0), mid_phi=(1, 0.2, 5.0), near_psi=(2, 0.2, 5.0), mid_psi=(3, 0.2, 5.0)):
    phi = {"near_1_4": _block(*near_phi), "mid_5_14": _block(*mid_phi), "inner_15_28": _block(4, 0.2, 5.0)}
    psi = {"near_1_4": _block(*near_psi), "mid_5_14": _block(*mid_psi), "inner_15_28": _block(5, 0.2, 5.0)}
    return {"phi": phi, "psi": psi}


def test_stage114_design_is_frozen():
    s114.validate_stage114_design()
    with pytest.raises(ValueError): s114.validate_stage114_design(angular_sectors=4)
    with pytest.raises(ValueError): s114.validate_stage114_design(near_wall_depth=5)
    with pytest.raises(ValueError): s114.validate_stage114_design(stage67_run_id=-1)


def test_wall_distance_bands_partition_exact_56x56_interior():
    masks = s114.wall_distance_band_masks(); total = sum(mask.astype(np.int8) for mask in masks.values())
    assert set(masks) == {"near_1_4", "mid_5_14", "inner_15_28"}; assert total.shape == (56, 56); assert np.all(total == 1)
    assert np.count_nonzero(masks["near_1_4"]) == 56 * 8; assert np.count_nonzero(masks["mid_5_14"]) == 56 * 20; assert np.count_nonzero(masks["inner_15_28"]) == 56 * 28


def test_angular_sector_indices_cover_eight_equal_45_degree_sectors():
    angle = (np.arange(96) + 0.5) * 2.0 * np.pi / 96.0; labels = s114.angular_sector_indices(np.cos(angle), np.sin(angle))
    assert set(labels.tolist()) == set(range(8)); assert [int(np.count_nonzero(labels == k)) for k in range(8)] == [12] * 8


def test_x_sector_change_maps_reconstruct_parent_change():
    rng = np.random.default_rng(114); nvel = 16; f = rng.normal(size=(64, 64, nvel)); w = np.linspace(0.1, 1.0, nvel); sector = np.repeat(np.arange(8), 2)
    maps = s114._x_sector_change_maps(f, w, sector); wb = s114.WALL_BAND_CELLS; center = f[wb:-wb, wb:-wb]
    left = center - f[wb:-wb, wb - 1 : -wb - 1]; right = f[wb:-wb, wb + 1 : -wb + 1] - center
    same = ((left > 0) & (right > 0)) | ((left < 0) & (right < 0)); parent = np.sum(np.where(same, 0.5 * np.abs(np.abs(left) - np.abs(right)), 0.0) * w[None, None, :], axis=-1)
    assert maps.shape == (8, 56, 56); assert s114._relative_l2(np.sum(maps, axis=0), parent) < 1.0e-14


def test_band_sector_metrics_are_normalized_and_finite():
    metrics = s114._band_sector_metrics(np.ones((8, 56, 56)), np.ones((56, 56)))
    for band in metrics.values():
        assert np.isclose(sum(band["sector_share"]), 1.0); assert np.isclose(band["maximum_sector_share"], 1.0 / 8.0); assert np.isclose(band["effective_sector_count"], 8.0)
    assert np.isclose(sum(metrics[name]["total_conditioned_mass_share"] for name in metrics), 1.0)


def test_stage114_decision_blocks_failed_closure_without_retuning():
    metrics = _metrics(); assert s114.stage114_decision(metrics, True, 1.1e-12) == "stage114_velocity_sector_closure_blocker_without_retuning"; assert s114.stage114_decision(metrics, False, 0.0) == "stage114_nonfinite_velocity_quadrature_conditioning_blocker_without_retuning"


def test_stage114_decision_common_sector_localization():
    metrics = _metrics(near_phi=(2, 0.34, 3.5), mid_phi=(2, 0.31, 3.8), near_psi=(2, 0.36, 3.4), mid_psi=(2, 0.33, 3.6))
    assert s114.stage114_decision(metrics, True, 1.0e-15) == "stage114_common_wall_distance_sector_localization_stage115_sector_resolved_radial_node_audit"


def test_stage114_decision_wall_distance_dependent_structure():
    metrics = _metrics(near_phi=(1, 0.34, 3.2), mid_phi=(3, 0.34, 3.2), near_psi=(1, 0.35, 3.2), mid_psi=(3, 0.35, 3.2))
    assert s114.stage114_decision(metrics, True, 1.0e-15) == "stage114_wall_distance_dependent_sector_structure_stage115_sector_transition_audit"


def test_stage114_decision_angularly_diffuse():
    metrics = _metrics(near_phi=(0, 0.20, 6.0), mid_phi=(1, 0.21, 5.5), near_psi=(2, 0.22, 5.2), mid_psi=(3, 0.23, 5.1))
    assert s114.stage114_decision(metrics, True, 1.0e-15) == "stage114_angularly_diffuse_within_broad_wall_profile_stage115_radial_node_conditioning_audit"


def test_stage114_decision_mixed_structure_preserves_negative_route():
    metrics = _metrics(near_phi=(0, 0.28, 3.5), mid_phi=(1, 0.20, 5.0), near_psi=(2, 0.27, 3.5), mid_psi=(3, 0.20, 5.0))
    assert s114.stage114_decision(metrics, True, 1.0e-15) == "stage114_mixed_velocity_quadrature_structure_stage115_distribution_specific_audit"
