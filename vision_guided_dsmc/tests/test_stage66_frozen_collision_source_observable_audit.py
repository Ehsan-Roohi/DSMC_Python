import math
import numpy as np
import pytest

from vgdsmc.stage66_frozen_collision_source_observable_audit import (
    STAGE66_DIRECTIONAL_FRACTION_THRESHOLD,
    STAGE66_UNCLIPPED_CLOSURE_TOLERANCE,
    frozen_frame_heat_flux,
    local_collision_frequency,
    mapped_polar_quadrature,
    signed_summary,
    stage66_decision,
    stratify_signed_field,
    validate_stage66_design,
    wall_distance_layers,
    wall_layer_stratification,
)


def _metrics(**updates):
    row = {
        "finite": True,
        "maximum_unclipped_target_closure_error": 1.0e-8,
        "fraction_normal_target_bias_aligned_with_observed_excess": 0.0,
        "fraction_normal_target_bias_opposed_to_observed_excess": 1.0,
    }
    row.update(updates)
    return row


def test_frozen_design_accepts_exact_values():
    validate_stage66_design()


def test_frozen_design_rejects_retuning():
    with pytest.raises(ValueError, match="not parameter retuning"):
        validate_stage66_design(correction_floor=0.01)


def test_collision_frequency_matches_transformed_paper_expression():
    rho = np.asarray([0.8, 1.2])
    temperature = np.asarray([0.25, 1.0])
    actual = local_collision_frequency(rho, temperature)
    expected = math.sqrt(math.pi / 2.0) / 10.0 * rho * np.sqrt(temperature)
    assert actual == pytest.approx(expected)


def test_frozen_frame_heat_flux_is_zero_for_symmetric_maxwellian():
    q = mapped_polar_quadrature(12, 32, 2.0)
    temperature = 0.7
    raw = np.exp(-(q.vx**2 + q.vy**2) / (2.0 * temperature))
    raw /= 2.0 * math.pi * temperature
    phi = raw[None, :] / np.sum(raw * q.weight)
    psi = temperature * phi
    cx = q.vx[None, :]
    cy = q.vy[None, :]
    c2 = cx * cx + cy * cy
    qx, qy = frozen_frame_heat_flux(phi, psi, cx, cy, c2, q.weight)
    assert float(qx[0]) == pytest.approx(0.0, abs=1.0e-13)
    assert float(qy[0]) == pytest.approx(0.0, abs=1.0e-13)


def test_signed_summary_preserves_sign_fractions():
    result = signed_summary(np.asarray([-3.0, -1.0, 0.0, 2.0]))
    assert result["mean"] == pytest.approx(-0.5)
    assert result["negative_fraction"] == pytest.approx(0.5)
    assert result["positive_fraction"] == pytest.approx(0.25)
    assert result["near_zero_fraction"] == pytest.approx(0.25)


def test_stratification_partitions_absolute_share():
    field = np.asarray([-1.0, -2.0, -3.0, -4.0])
    coordinate = np.asarray([0.1, 0.3, 0.6, 0.9])
    rows = stratify_signed_field(field, coordinate, (0.0, 0.5, 1.0))
    assert sum(row["count"] for row in rows) == 4
    assert sum(row["absolute_share"] for row in rows) == pytest.approx(1.0)
    assert rows[0]["mean"] == pytest.approx(-1.5)


def test_wall_layer_stratification_partitions_absolute_bias():
    field = -np.ones((8, 8))
    distance = wall_distance_layers(field.shape)
    result = wall_layer_stratification(field, distance, wall_band_layers=1)
    assert (
        result["wall_band_absolute_share"] + result["interior_absolute_share"]
        == pytest.approx(1.0)
    )
    assert result["wall_band_cell_fraction"] == pytest.approx(28.0 / 64.0)


def test_decision_nonfinite_blocker():
    assert (
        stage66_decision(_metrics(finite=False))
        == "stage66_nonfinite_frozen_source_observable_blocker"
    )


def test_decision_closure_blocker():
    decision = stage66_decision(
        _metrics(
            maximum_unclipped_target_closure_error=(
                STAGE66_UNCLIPPED_CLOSURE_TOLERANCE * 1.01
            )
        )
    )
    assert decision == "stage66_unclipped_target_closure_blocker"


def test_decision_opposed_route():
    decision = stage66_decision(
        _metrics(
            fraction_normal_target_bias_opposed_to_observed_excess=(
                STAGE66_DIRECTIONAL_FRACTION_THRESHOLD
            )
        )
    )
    assert "opposes_heat_flux_overprediction" in decision
    assert "stage67" in decision


def test_decision_aligned_route():
    decision = stage66_decision(
        _metrics(
            fraction_normal_target_bias_aligned_with_observed_excess=(
                STAGE66_DIRECTIONAL_FRACTION_THRESHOLD
            ),
            fraction_normal_target_bias_opposed_to_observed_excess=0.0,
        )
    )
    assert "aligns_with_heat_flux_overprediction" in decision
    assert "stage67" in decision


def test_decision_mixed_route():
    decision = stage66_decision(
        _metrics(
            fraction_normal_target_bias_aligned_with_observed_excess=0.5,
            fraction_normal_target_bias_opposed_to_observed_excess=0.5,
        )
    )
    assert "mixed" in decision
    assert "stage67" in decision
