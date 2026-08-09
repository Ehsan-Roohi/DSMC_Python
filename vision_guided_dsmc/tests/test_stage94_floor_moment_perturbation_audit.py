import numpy as np
import pytest

from vgdsmc import stage94_floor_moment_perturbation_audit as stage94


def test_stage94_frozen_design_accepts_defaults():
    stage94.validate_stage94_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 1.0},
        {"grid": (32, 32)},
        {"source_relaxation": 0.5},
        {"radial_scale": 1.0},
        {"onset_step": 2},
        {"correction_floor": 0.1},
        {"material_perturbation_guard": 1.0e-3},
    ],
)
def test_stage94_frozen_design_rejects_retuning_or_retiming(override):
    with pytest.raises(ValueError):
        stage94.validate_stage94_design(**override)


def test_floor_perturbation_distinguishes_zero_and_positive_subfloor_entries():
    candidate = np.array([[[0.0, 0.25, 1.0]]], dtype=float)
    weight = np.ones(3, dtype=float)
    vx = np.array([0.0, 1.0, 2.0], dtype=float)
    vy = np.array([1.0, 0.0, 0.0], dtype=float)

    stats, delta, cell_added = stage94._floor_perturbation(
        candidate,
        0.5,
        weight,
        vx,
        vy,
    )

    assert stats["finite"] is True
    assert stats["activation_fraction_by_count"] == pytest.approx(2.0 / 3.0)
    assert stats["strict_negative_fraction"] == pytest.approx(0.0)
    assert stats["exact_zero_fraction_of_activations"] == pytest.approx(0.5)
    assert stats["positive_subfloor_fraction_of_activations"] == pytest.approx(0.5)
    assert stats["added_weighted_value"] == pytest.approx(0.75)
    assert stats["exact_zero_share_of_added_weighted_value"] == pytest.approx(2.0 / 3.0)
    assert stats["moment_perturbations"]["m0"]["relative_absolute_perturbation"] == pytest.approx(0.6)
    assert delta.tolist() == [[[0.5, 0.25, 0.0]]]
    assert cell_added.tolist() == [[0.75]]


def test_floor_perturbation_reports_negative_candidate_without_hiding_it():
    candidate = np.array([[[-0.25, 1.0]]], dtype=float)
    stats, delta, _ = stage94._floor_perturbation(
        candidate,
        0.5,
        np.ones(2),
        np.array([1.0, -1.0]),
        np.zeros(2),
    )
    assert stats["strict_negative_fraction"] == pytest.approx(0.5)
    assert delta[0, 0, 0] == pytest.approx(0.75)


def test_field_difference_reports_exact_relative_l2():
    before = np.array([1.0, 2.0])
    after = np.array([1.0, 3.0])
    result = stage94._field_difference(before, after)
    assert result["relative_l2"] == pytest.approx(1.0 / np.sqrt(5.0))
    assert result["maximum_absolute_delta"] == pytest.approx(1.0)
    assert result["relative_to_baseline_max"] == pytest.approx(0.5)


def _macroscopic_with_relative_l2(value):
    return {
        name: {
            "relative_l2": value,
            "maximum_absolute_delta": 0.0,
            "relative_to_baseline_max": value,
            "mean_absolute_delta": 0.0,
        }
        for name in ("rho", "u", "v", "T", "qx", "qy", "total_internal_moment")
    }


def test_stage94_decision_routes_negligible_perturbation_to_unclipped_second_update_audit():
    result = stage94.stage94_decision(
        {"finite": True, "strict_negative_fraction": 0.0},
        {"finite": True, "strict_negative_fraction": 0.0},
        _macroscopic_with_relative_l2(1.0e-8),
    )
    assert result == "stage94_floor_moment_perturbation_negligible_stage95_unclipped_second_update_propagation_audit"


def test_stage94_decision_routes_material_perturbation_to_paired_propagation_audit():
    result = stage94.stage94_decision(
        {"finite": True, "strict_negative_fraction": 0.0},
        {"finite": True, "strict_negative_fraction": 0.0},
        _macroscopic_with_relative_l2(1.0e-3),
    )
    assert result == "stage94_floor_moment_perturbation_material_stage95_clipped_unclipped_second_update_propagation_audit"


def test_stage94_decision_preserves_nonfinite_and_negative_blockers():
    macro = _macroscopic_with_relative_l2(0.0)
    assert stage94.stage94_decision(
        {"finite": False, "strict_negative_fraction": 0.0},
        {"finite": True, "strict_negative_fraction": 0.0},
        macro,
    ) == "stage94_nonfinite_floor_perturbation_blocker_without_retuning"
    assert stage94.stage94_decision(
        {"finite": True, "strict_negative_fraction": 0.1},
        {"finite": True, "strict_negative_fraction": 0.0},
        macro,
    ) == "stage94_unexpected_negative_candidate_blocker_without_retuning"
