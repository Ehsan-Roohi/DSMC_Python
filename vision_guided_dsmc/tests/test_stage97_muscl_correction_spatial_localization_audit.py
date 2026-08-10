import numpy as np
import pytest

from vgdsmc import stage97_muscl_correction_spatial_localization_audit as stage97


def test_stage97_frozen_design_accepts_defaults():
    stage97.validate_stage97_design()


@pytest.mark.parametrize(
    "override",
    [
        {"grid": (32, 32)},
        {"wall_band_cells": 2},
        {"top_fraction": 0.2},
        {"dominant_share": 0.5},
        {"material_share_shift": 0.2},
        {"stage96_run_id": 1},
    ],
)
def test_stage97_frozen_design_rejects_parameter_or_partition_retuning(override):
    with pytest.raises(ValueError):
        stage97.validate_stage97_design(**override)


def test_region_masks_partition_exact_grid():
    masks = stage97._region_masks(stage97.GRID)
    partition = (
        masks["corners"].astype(int)
        + masks["vertical_sidewalls"].astype(int)
        + masks["horizontal_walls"].astype(int)
        + masks["interior"].astype(int)
    )
    assert np.all(partition == 1)
    assert np.array_equal(masks["wall_band"], ~masks["interior"])


def test_map_metrics_region_shares_close_and_are_nonnegative():
    field = np.ones(stage97.GRID, dtype=float)
    metrics = stage97._map_metrics(field)
    shares = metrics["region_shares"]
    assert shares["wall_band"] + shares["interior"] == pytest.approx(1.0)
    assert shares["corners"] + shares["vertical_sidewalls"] + shares["horizontal_walls"] == pytest.approx(
        shares["wall_band"]
    )
    assert all(value >= 0.0 for value in shares.values())


def test_pair_metrics_identical_maps_have_unit_similarity_and_zero_shift():
    field = np.arange(1, 64 * 64 + 1, dtype=float).reshape(stage97.GRID)
    metrics = stage97._pair_metrics(field, field.copy())
    assert metrics["final_to_first_total_magnitude_ratio"] == pytest.approx(1.0)
    assert metrics["normalized_map_total_variation"] == pytest.approx(0.0)
    assert metrics["first_final_cosine_similarity"] == pytest.approx(1.0)
    assert metrics["first_final_pearson"] == pytest.approx(1.0)
    assert metrics["interior_share_change"] == pytest.approx(0.0)


def _decision_payload(*, final_interior, first_interior):
    return {
        "first": {"region_shares": {"interior": first_interior, "wall_band": 1.0 - first_interior}},
        "final": {"region_shares": {"interior": final_interior, "wall_band": 1.0 - final_interior}},
        "interior_share_change": final_interior - first_interior,
        "wall_band_share_change": -(final_interior - first_interior),
    }


def test_decision_routes_clear_interior_redistribution_to_directional_growth_audit():
    phi = _decision_payload(final_interior=0.75, first_interior=0.50)
    psi = _decision_payload(final_interior=0.72, first_interior=0.48)
    assert stage97.stage97_decision(phi, psi) == (
        "stage97_interior_dominant_redistribution_stage98_directional_operator_growth_audit"
    )


def test_decision_routes_clear_wall_dominance_to_wall_orientation_audit():
    phi = _decision_payload(final_interior=0.20, first_interior=0.25)
    psi = _decision_payload(final_interior=0.25, first_interior=0.30)
    assert stage97.stage97_decision(phi, psi) == (
        "stage97_wall_dominant_persistence_stage98_wall_orientation_operator_audit"
    )


def test_decision_routes_mixed_map_without_claiming_causality():
    phi = _decision_payload(final_interior=0.60, first_interior=0.55)
    psi = _decision_payload(final_interior=0.58, first_interior=0.54)
    assert stage97.stage97_decision(phi, psi) == (
        "stage97_mixed_spatial_persistence_stage98_signed_directional_balance_audit"
    )
