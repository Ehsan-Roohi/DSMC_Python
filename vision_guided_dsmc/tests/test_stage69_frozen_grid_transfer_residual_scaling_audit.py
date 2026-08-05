import numpy as np
import pytest

from vgdsmc import stage69_frozen_grid_transfer_residual_scaling_audit as s


def test_completed_stage68_endpoint_is_exact():
    endpoint = s.STAGE68_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31010112390
    assert endpoint["workflow_job_id"] == 92319799599
    assert endpoint["tests_passed"] == 87
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8938442470
    assert endpoint["artifact_sha256"] == (
        "d38fcf90e998341a8ed1ad443ac3de8a1597a91a89c454e8430491ff75751ea6"
    )
    assert endpoint["decision"].endswith(
        "stage69_frozen_grid_transfer_residual_scaling_audit"
    )


def test_frozen_design_accepts_exact_configuration():
    s.validate_stage69_design()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grids": (8, 16, 32, 64)},
        {"fine_grid": 32},
        {"kn0": 1.0},
        {"cold_hot_ratio": 0.2},
        {"rule": (32, 96)},
        {"radial_scale": 1.0},
        {"limiter": "vanleer"},
        {"restriction": "injection"},
        {"wall_band_physical_fraction": 0.125},
        {"material_heat_flux_ratio": 0.05},
    ],
)
def test_frozen_design_rejects_retuning(kwargs):
    with pytest.raises(ValueError):
        s.validate_stage69_design(**kwargs)


def test_cell_average_restriction_has_expected_blocks_and_shape():
    fine = np.arange(4 * 4 * 2, dtype=float).reshape(4, 4, 2)
    restricted = s.restrict_cell_average(fine, 2)
    assert restricted.shape == (2, 2, 2)
    np.testing.assert_allclose(restricted[0, 0], fine[:2, :2].mean(axis=(0, 1)))
    np.testing.assert_allclose(restricted[1, 1], fine[2:, 2:].mean(axis=(0, 1)))


def test_cell_average_restriction_preserves_each_velocity_bin_global_mean():
    rng = np.random.default_rng(69)
    fine = rng.random((8, 8, 7))
    for target in (1, 2, 4, 8):
        restricted = s.restrict_cell_average(fine, target)
        np.testing.assert_allclose(
            restricted.mean(axis=(0, 1)), fine.mean(axis=(0, 1)), rtol=0, atol=5e-16
        )
        assert s.restriction_conservation(fine, restricted)["within_guard"] is True


def test_restriction_rejects_non_square_and_non_divisor_targets():
    with pytest.raises(ValueError):
        s.restrict_cell_average(np.zeros((4, 3, 2)), 2)
    with pytest.raises(ValueError):
        s.restrict_cell_average(np.zeros((6, 6, 2)), 4)


def test_physical_wall_band_scales_as_one_sixteenth():
    for grid, expected_layers in ((16, 1), (32, 2), (64, 4)):
        mask, layers = s.physical_interior_mask(grid)
        assert layers == expected_layers
        assert mask.shape == (grid, grid)
        assert not np.any(mask[:layers])
        assert np.all(mask[layers:-layers, layers:-layers])


def test_observed_order_recovers_first_order_halving():
    assert s.observed_order(0.2, 0.1) == pytest.approx(1.0)
    assert s.observed_order(0.4, 0.1) == pytest.approx(2.0)
    assert np.isnan(s.observed_order(0.0, 0.1))


def test_monotonic_decrease_guard():
    assert s.monotonically_decreases_with_refinement([3.0, 2.0, 1.0]) is True
    assert s.monotonically_decreases_with_refinement([3.0, 3.0, 1.0]) is False
    assert s.monotonically_decreases_with_refinement([3.0, 4.0, 1.0]) is False


def test_decision_paths_preserve_scientific_guards():
    args = dict(
        finite=True,
        provenance_consistent=True,
        restriction_conservative=True,
        full_monotonic=True,
        interior_monotonic=True,
        fine_wall_absolute_share=0.60,
        fine_pair_full_order=0.35,
        fine_pair_interior_order=0.96,
        fine_normal_heat_flux_ratio=2.71,
    )
    assert s.stage69_decision(**args) == (
        "stage69_monotone_but_wall_dominated_slow_full_scaling_"
        "stage70_independent_wall_face_flux_discretization_audit"
    )
    assert s.stage69_decision(**{**args, "finite": False}) == (
        "stage69_nonfinite_grid_transfer_blocker"
    )
    assert s.stage69_decision(**{**args, "provenance_consistent": False}) == (
        "stage69_completed_endpoint_reproduction_blocker"
    )
    assert s.stage69_decision(**{**args, "restriction_conservative": False}) == (
        "stage69_conservative_restriction_blocker"
    )
    assert s.stage69_decision(**{**args, "full_monotonic": False}) == (
        "stage69_nonmonotone_frozen_grid_transfer_blocker"
    )


def test_nonwall_material_and_below_materiality_decisions_are_separate():
    base = dict(
        finite=True,
        provenance_consistent=True,
        restriction_conservative=True,
        full_monotonic=True,
        interior_monotonic=True,
        fine_wall_absolute_share=0.40,
        fine_pair_full_order=0.9,
        fine_pair_interior_order=0.8,
    )
    assert "linearized_response" in s.stage69_decision(
        **base, fine_normal_heat_flux_ratio=0.10
    )
    assert "below_materiality" in s.stage69_decision(
        **base, fine_normal_heat_flux_ratio=0.099
    )
