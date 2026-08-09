import numpy as np
import pytest

from vgdsmc import stage92_candidate_update_localization_audit as stage92


def test_stage92_frozen_design_accepts_defaults():
    stage92.validate_stage92_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 1.0},
        {"grid": (32, 32)},
        {"source_relaxation": 0.5},
        {"radial_scale": 1.0},
        {"onset_step": 2},
        {"angular_sectors": 4},
    ],
)
def test_stage92_frozen_design_rejects_retuning_or_retiming(override):
    with pytest.raises(ValueError):
        stage92.validate_stage92_design(**override)


def test_mask_overlap_reports_exact_shared_added_removed_fractions():
    baseline = np.array([[[True, False, True, False]]], dtype=bool)
    one_sided = np.array([[[True, True, False, False]]], dtype=bool)
    result = stage92._mask_overlap(baseline, one_sided)
    assert result["jaccard_overlap"] == pytest.approx(1.0 / 3.0)
    assert result["shared_fraction_of_baseline_activations"] == pytest.approx(0.5)
    assert result["shared_fraction_of_one_sided_activations"] == pytest.approx(0.5)
    assert result["one_sided_added_fraction_of_all_entries"] == pytest.approx(0.25)
    assert result["one_sided_removed_fraction_of_all_entries"] == pytest.approx(0.25)


def test_activation_statistics_distinguishes_first_order_and_correction_induced_flooring():
    first_order = np.full((2, 4, 4), 2.0, dtype=float)
    candidate = first_order.copy()
    candidate[0, 0, 0] = -1.0
    candidate[1, 3, 1] = -0.5
    correction = first_order - candidate
    vx = np.array([1.0, -1.0, 0.2, -0.2])
    vy = np.array([0.1, -0.1, 1.0, -1.0])

    stats, mask, cell_map, velocity_map = stage92._activation_statistics(
        first_order,
        candidate,
        correction,
        0.0,
        vx,
        vy,
    )
    assert stats["floor_activation_fraction"] == pytest.approx(2.0 / candidate.size)
    assert stats["first_order_floor_fraction"] == 0.0
    assert stats["correction_induced_fraction_of_activations"] == 1.0
    assert mask.sum() == 2
    assert cell_map.shape == (2, 4)
    assert velocity_map.shape == (4,)
    assert sum(item["fraction_of_all_activations"] for item in stats["angular_sector_activation_fractions"]) == pytest.approx(1.0)


def _arm(*, finite=True, induced=1.0):
    return {
        "finite": finite,
        "correction_induced_fraction_of_activations": induced,
    }


def _result(*, overlap=0.95, finite=True, induced=1.0):
    return {
        "zero_boundary_slope": _arm(finite=finite, induced=induced),
        "one_sided_boundary_slope": _arm(finite=finite, induced=induced),
        "paired_floor_mask_comparison": {"jaccard_overlap": overlap},
    }


def test_stage92_decision_routes_shared_correction_induced_onset():
    assert stage92.stage92_decision(_result(), _result()).startswith(
        "stage92_shared_correction_induced_floor_onset"
    )


def test_stage92_decision_routes_boundary_topology_change():
    assert stage92.stage92_decision(_result(overlap=0.2), _result()).startswith(
        "stage92_boundary_counterfactual_changes_floor_topology"
    )


def test_stage92_decision_preserves_nonfinite_blocker():
    assert stage92.stage92_decision(_result(finite=False), _result()) == (
        "stage92_nonfinite_first_update_blocker_without_retuning"
    )
