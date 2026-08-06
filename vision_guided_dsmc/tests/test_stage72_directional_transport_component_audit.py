from __future__ import annotations

import numpy as np
import pytest

from vgdsmc import stage68_independent_transport_operator_residual_audit as stage68
from vgdsmc import stage72_directional_transport_component_audit as stage72


def synthetic_transport_case(seed: int = 4):
    rng = np.random.default_rng(seed)
    distribution = 0.2 + rng.random((5, 6, 7))
    left = 0.2 + rng.random((5, 7))
    right = 0.2 + rng.random((5, 7))
    bottom = 0.2 + rng.random((6, 7))
    top = 0.2 + rng.random((6, 7))
    vx = np.array([-1.7, -0.8, -0.2, 0.0, 0.3, 0.9, 1.6])
    vy = np.array([0.7, -1.2, 0.0, 1.4, -0.4, 0.2, -0.9])
    return distribution, left, right, bottom, top, vx, vy, 1 / 6, 1 / 5


def test_stage72_frozen_design_accepts_only_preregistered_values():
    stage72.validate_stage72_design()
    with pytest.raises(ValueError, match="no retuning"):
        stage72.validate_stage72_design(component_dominance_fraction=0.61)
    with pytest.raises(ValueError, match="no retuning"):
        stage72.validate_stage72_design(rule=(41, 96))


def test_first_order_directional_sum_reproduces_retained_operator():
    args = synthetic_transport_case()
    rx, ry = stage72.first_order_directional_chunk(*args)
    retained = stage68.first_order_transport_chunk(*args)
    np.testing.assert_allclose(rx + ry, retained, rtol=0.0, atol=2e-15)


def test_second_order_directional_sum_reproduces_independent_operator():
    args = synthetic_transport_case()
    rx, ry = stage72.second_order_directional_chunk(*args)
    independent = stage68.second_order_transport_chunk(*args)
    np.testing.assert_allclose(rx + ry, independent, rtol=0.0, atol=2e-15)


def test_directional_components_are_axis_specific():
    distribution, left, right, bottom, top, vx, vy, dx, dy = synthetic_transport_case()
    rx, ry = stage72.first_order_directional_chunk(
        distribution, left, right, bottom, top, vx * 0.0, vy, dx, dy
    )
    assert np.count_nonzero(rx) == 0
    assert np.count_nonzero(ry) > 0
    rx, ry = stage72.second_order_directional_chunk(
        distribution, left, right, bottom, top, vx, vy * 0.0, dx, dy
    )
    assert np.count_nonzero(rx) > 0
    assert np.count_nonzero(ry) == 0


def test_directional_closure_reports_exact_and_failed_cases():
    x = np.arange(12, dtype=float).reshape(3, 4)
    y = -0.25 * x
    exact = stage72.directional_closure(x, y, x + y)
    assert exact["within_guard"] is True
    assert exact["maximum_absolute_error"] == 0.0
    failed = stage72.directional_closure(x, y, x + y + 1e-4)
    assert failed["within_guard"] is False


def test_component_attribution_forms_complete_directional_shares():
    grid = 16
    yy, xx = np.indices((grid, grid))
    x = np.where(xx < 1, 2.0, 0.1)
    y = np.where(yy < 1, -0.5, 0.05)
    result = stage72.component_attribution(x, y, grid)
    assert result["wall_band_layers"] == 1
    assert result["x_component_absolute_share"] > result["y_component_absolute_share"]
    assert np.isclose(
        result["x_component_absolute_share"]
        + result["y_component_absolute_share"],
        1.0,
    )
    assert 0.0 <= result["component_cancellation_ratio"] <= 1.0 + 1e-15
    assert result["x_vertical_side_strip_absolute_wall_share"] > 0.0


def test_stage72_decision_blocks_invalid_endpoints():
    args = dict(
        finite=True,
        provenance_consistent=True,
        directional_closure_closed=True,
        x_component_dominant=False,
        y_component_dominant=False,
        dominant_outer_two_layers_concentrated=False,
        dominant_oriented_strips=False,
    )
    assert "nonfinite" in stage72.stage72_decision(**{**args, "finite": False})
    assert "endpoint" in stage72.stage72_decision(
        **{**args, "provenance_consistent": False}
    )
    assert "closure" in stage72.stage72_decision(
        **{**args, "directional_closure_closed": False}
    )


def test_stage72_decision_routes_x_dominant_component():
    decision = stage72.stage72_decision(
        True, True, True, True, False, True, True
    )
    assert decision.startswith("stage72_x_direction")
    assert "stage73_velocity_sign_angular_bin" in decision


def test_stage72_decision_routes_y_dominant_component():
    decision = stage72.stage72_decision(
        True, True, True, False, True, True, True
    )
    assert decision.startswith("stage72_y_direction")
    assert "stage73_velocity_sign_angular_bin" in decision


def test_stage72_decision_preserves_mixed_direction_outcome():
    decision = stage72.stage72_decision(
        True, True, True, False, False, False, False
    )
    assert "mixed_direction_or_cancellation" in decision
    assert "facewise_directional_flux_cancellation" in decision


def test_signed_statistics_preserve_positive_and_negative_findings():
    values = np.array([[-3.0, -1.0], [2.0, 4.0]])
    stats = stage72.signed_statistics(values)
    assert stats["minimum"] == -3.0
    assert stats["maximum"] == 4.0
    assert stats["positive_cell_fraction"] == 0.5
    assert stats["negative_cell_fraction"] == 0.5
    assert stats["absolute_sum"] == 10.0
