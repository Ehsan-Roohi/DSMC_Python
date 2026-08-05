import numpy as np
import pytest

from vgdsmc import stage71_wall_layer_interior_face_transport_attribution_audit as s


def test_completed_stage70_endpoint_is_exact():
    endpoint = s.STAGE70_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31037106991
    assert endpoint["workflow_job_id"] == 92411824859
    assert endpoint["tests_passed"] == 125
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8950151688
    assert endpoint["artifact_sha256"] == (
        "02fab9c3b577e2faf31d775da73d374a340daf74421fb0f899f7db968b9b5030"
    )
    assert endpoint["summary_sha256"] == (
        "5147c55c535b7b3f8902c90f2ffd7df0f360c47155c119aced26b9e6653f6dbb"
    )
    assert endpoint["profiles_sha256"] == (
        "1fec5ab239a0eed1333943ba3552d70498e07d269990af3929e1819311666afd"
    )
    assert endpoint["decision"].endswith(
        "stage71_wall_layer_interior_face_transport_attribution_audit"
    )


def test_frozen_design_accepts_exact_configuration():
    s.validate_stage71_design()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grids": (8, 16, 32, 64)},
        {"fine_grid": 32},
        {"kn0": 1.0},
        {"cold_hot_ratio": 0.2},
        {"rule": (32, 96)},
        {"radial_scale": 1.0},
        {"wall_band_physical_fraction": 0.125},
        {"material_wall_face_ratio": 0.05},
        {"wall_dominance_fraction": 0.6},
        {"outer_two_layer_concentration": 0.8},
        {"side_strip_dominance": 0.6},
    ],
)
def test_frozen_design_rejects_retuning(kwargs):
    with pytest.raises(ValueError):
        s.validate_stage71_design(**kwargs)


def test_wall_distance_and_layers_are_exact():
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(s.wall_distance(6), expected)
    assert s.wall_band_layers(16) == 1
    assert s.wall_band_layers(32) == 2
    assert s.wall_band_layers(64) == 4


def test_attribution_masks_are_disjoint_and_complete():
    masks = s.attribution_masks(8, 2)
    total = sum(mask.astype(int) for mask in masks.values())
    np.testing.assert_array_equal(total, np.ones((8, 8), dtype=int))
    assert masks["bottom"].sum() == 8
    assert masks["top"].sum() == 8
    assert masks["left"].sum() == 8
    assert masks["right"].sum() == 8
    assert masks["corners"].sum() == 16
    assert masks["interior"].sum() == 16


def test_region_code_map_matches_masks():
    masks = s.attribution_masks(8, 2)
    codes = s.region_code_map(8, masks)
    assert set(np.unique(codes)) == set(s.REGION_CODES.values())
    for name, mask in masks.items():
        assert np.all(codes[mask] == s.REGION_CODES[name])


def test_signed_statistics_preserves_signed_and_absolute_measures():
    values = np.array([[1.0, -2.0], [3.0, -4.0]])
    mask = np.array([[True, True], [False, False]])
    row = s.signed_statistics(values, mask, 10.0, 3.0, True)
    assert row["cell_count"] == 2
    assert row["absolute_sum"] == pytest.approx(3.0)
    assert row["absolute_global_share"] == pytest.approx(0.3)
    assert row["absolute_wall_band_share"] == pytest.approx(1.0)
    assert row["signed_sum"] == pytest.approx(-1.0)
    assert row["signed_to_absolute_ratio"] == pytest.approx(-1.0 / 3.0)
    assert row["positive_cell_fraction"] == pytest.approx(0.5)
    assert row["negative_cell_fraction"] == pytest.approx(0.5)


def test_decision_paths_preserve_scientific_guards():
    base = dict(
        finite=True,
        provenance_consistent=True,
        partition_closed=True,
        stage69_wall_share_reproduced=True,
        stage70_wall_face_submaterial=True,
        wall_band_dominant=True,
        outer_two_layers_concentrated=True,
        side_strips_dominant=True,
    )
    assert s.stage71_decision(**base) == (
        "stage71_near_wall_side_strip_interior_face_dominance_"
        "stage72_directional_transport_component_audit"
    )
    assert s.stage71_decision(**{**base, "side_strips_dominant": False}) == (
        "stage71_near_wall_interior_face_dominance_"
        "stage72_directional_transport_component_audit"
    )
    assert "facewise_flux_reconstruction" in s.stage71_decision(
        **{**base, "outer_two_layers_concentrated": False}
    )
    assert s.stage71_decision(**{**base, "finite": False}) == (
        "stage71_nonfinite_attribution_blocker"
    )
    assert s.stage71_decision(**{**base, "provenance_consistent": False}) == (
        "stage71_completed_endpoint_reproduction_blocker"
    )
    assert s.stage71_decision(**{**base, "partition_closed": False}) == (
        "stage71_region_partition_blocker"
    )
    assert s.stage71_decision(
        **{**base, "stage69_wall_share_reproduced": False}
    ) == "stage71_stage69_wall_share_reproduction_blocker"
    assert s.stage71_decision(
        **{**base, "stage70_wall_face_submaterial": False}
    ) == "stage71_material_physical_wall_face_difference_blocker"
    assert s.stage71_decision(**{**base, "wall_band_dominant": False}) == (
        "stage71_wall_band_not_dominant_stop_before_solver_response"
    )
