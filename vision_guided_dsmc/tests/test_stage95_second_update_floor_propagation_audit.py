import numpy as np
import pytest

from vgdsmc import stage95_second_update_floor_propagation_audit as stage95


def test_stage95_frozen_design_accepts_defaults():
    stage95.validate_stage95_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 1.0},
        {"grid": (32, 32)},
        {"source_relaxation": 0.5},
        {"radial_scale": 1.0},
        {"second_update_index": 3},
        {"correction_floor": 0.1},
        {"material_propagation_guard": 1.0e-3},
    ],
)
def test_stage95_frozen_design_rejects_retuning_or_retiming(override):
    with pytest.raises(ValueError):
        stage95.validate_stage95_design(**override)


def test_mask_jaccard_handles_empty_and_partial_overlap():
    assert stage95._mask_jaccard(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)) == pytest.approx(1.0)
    assert stage95._mask_jaccard(
        np.array([True, True, False]),
        np.array([True, False, True]),
    ) == pytest.approx(1.0 / 3.0)


def test_distribution_pair_stats_reports_weighted_difference_and_floor_topology():
    a = np.array([[[0.0, 1.0, 2.0]]], dtype=float)
    b = np.array([[[0.5, 1.0, 3.0]]], dtype=float)
    weight = np.ones(3, dtype=float)
    speed_squared = np.array([0.0, 1.0, 4.0], dtype=float)

    stats, cell_map = stage95._distribution_pair_stats(
        a,
        b,
        0.5,
        weight,
        speed_squared,
    )

    assert stats["finite"] is True
    assert stats["weighted_m0_relative_absolute_difference"] == pytest.approx(1.5 / 3.0)
    assert stats["weighted_speed_squared_relative_absolute_difference"] == pytest.approx(4.0 / 9.0)
    assert stats["unclipped_seed"]["activation_fraction_by_count"] == pytest.approx(1.0 / 3.0)
    assert stats["clipped_seed"]["activation_fraction_by_count"] == pytest.approx(0.0)
    assert stats["floor_mask_jaccard"] == pytest.approx(0.0)
    assert stats["floor_mask_symmetric_difference_fraction"] == pytest.approx(1.0 / 3.0)
    assert cell_map.tolist() == [[1.5]]


def _operator(relative):
    distribution = {
        "finite": True,
        "weighted_m0_relative_absolute_difference": relative,
        "weighted_speed_squared_relative_absolute_difference": relative,
    }
    macro = {
        name: {
            "relative_l2": relative,
            "maximum_absolute_delta": 0.0,
            "relative_to_baseline_max": relative,
            "mean_absolute_delta": 0.0,
        }
        for name in ("rho", "u", "v", "T", "qx", "qy", "total_internal_moment")
    }
    return {"phi": dict(distribution), "psi": dict(distribution), "macroscopic_difference": macro}


def test_stage95_decision_routes_negligible_propagation_to_muscl_growth_audit():
    assert stage95.stage95_decision(_operator(1.0e-8), _operator(1.0e-7)) == (
        "stage95_floor_propagation_negligible_stage96_muscl_correction_growth_audit"
    )


def test_stage95_decision_routes_material_propagation_to_localization():
    assert stage95.stage95_decision(_operator(1.0e-8), _operator(1.0e-3)) == (
        "stage95_floor_propagation_material_stage96_floor_sensitive_operator_localization_audit"
    )


def test_stage95_decision_preserves_nonfinite_blocker():
    bad = _operator(0.0)
    bad["psi"]["finite"] = False
    assert stage95.stage95_decision(_operator(0.0), bad) == (
        "stage95_nonfinite_second_update_blocker_without_retuning"
    )
