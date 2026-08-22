import numpy as np
import pytest

from vgdsmc import stage93_first_order_axis_component_balance_audit as stage93


def test_stage93_frozen_design_accepts_defaults():
    stage93.validate_stage93_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 1.0},
        {"grid": (32, 32)},
        {"source_relaxation": 0.5},
        {"radial_scale": 1.0},
        {"onset_step": 2},
        {"correction_floor": 1.0e-8},
    ],
)
def test_stage93_frozen_design_rejects_retuning_or_retiming(override):
    with pytest.raises(ValueError):
        stage93.validate_stage93_design(**override)


def test_component_balance_closes_and_reports_count_weight_and_moment_footprint():
    source = np.array([[[0.1, 0.5]]], dtype=float)
    x_transport = np.array([[[0.1, 0.5]]], dtype=float)
    y_transport = np.array([[[0.1, 0.5]]], dtype=float)
    qweight = np.array([1.0, 1.0], dtype=float)
    vx = np.array([1.0, 0.0], dtype=float)
    vy = np.array([0.0, 1.0], dtype=float)

    stats, cell_map, velocity_map = stage93._component_balance(
        source,
        x_transport,
        y_transport,
        0.5,
        qweight,
        vx,
        vy,
    )

    assert stats["finite"] is True
    assert stats["floor_activation_fraction_by_count"] == pytest.approx(0.5)
    assert stats["floor_activation_fraction_by_quadrature_weight"] == pytest.approx(0.5)
    assert stats["masked_quadrature_value_fraction_of_global_first_order"] == pytest.approx(1.0 / 6.0)
    assert stats["positive_reduced_moment_fractions_from_floor_set"]["m0"] == pytest.approx(1.0 / 6.0)
    assert stats["component_sum_closure_relative_l2"] == pytest.approx(0.0)
    assert stats["component_statistics"]["source"]["masked_quadrature_value_share"] == pytest.approx(1.0 / 3.0)
    assert stats["component_statistics"]["x_transport"]["masked_quadrature_value_share"] == pytest.approx(1.0 / 3.0)
    assert stats["component_statistics"]["y_transport"]["masked_quadrature_value_share"] == pytest.approx(1.0 / 3.0)
    assert cell_map.shape == (1, 1)
    assert velocity_map.shape == (2,)
    assert velocity_map.tolist() == [1.0, 0.0]


def test_component_balance_preserves_exact_zero_floor_entries():
    shape = (1, 1, 2)
    source = np.zeros(shape)
    x_transport = np.zeros(shape)
    y_transport = np.zeros(shape)
    stats, _, _ = stage93._component_balance(
        source,
        x_transport,
        y_transport,
        1.0e-30,
        np.ones(2),
        np.array([1.0, -1.0]),
        np.array([0.0, 0.0]),
    )
    assert stats["floor_activation_fraction_by_count"] == 1.0
    assert stats["exact_zero_fraction_of_activations"] == 1.0
    assert stats["all_components_exact_zero_fraction_of_activations"] == 1.0
    assert stats["strict_negative_fraction"] == 0.0


def test_stage93_decision_routes_complete_audit_to_floor_moment_perturbation():
    assert stage93.stage93_decision({"finite": True}, {"finite": True}) == (
        "stage93_axis_component_balance_complete_stage94_floor_moment_perturbation_audit"
    )


def test_stage93_decision_preserves_nonfinite_blocker():
    assert stage93.stage93_decision({"finite": False}, {"finite": True}) == (
        "stage93_nonfinite_first_order_component_blocker_without_retuning"
    )
