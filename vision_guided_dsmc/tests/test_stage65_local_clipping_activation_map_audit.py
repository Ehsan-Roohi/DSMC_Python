import numpy as np
import pytest

from vgdsmc.stage65_local_clipping_activation_map_audit import (
    STAGE65_BROAD_CELL_FRACTION_THRESHOLD,
    STAGE65_MINIMUM_CLIPPING_DEFECT_CORRELATION,
    STAGE65_WALL_LOCALIZATION_THRESHOLD,
    concentration_metrics,
    maximum_location,
    safe_correlation,
    stage65_decision,
    validate_stage65_design,
    wall_band_metrics,
    wall_distance_layers,
    weighted_centroid,
)


def _metrics(**updates):
    row = {
        "finite": True,
        "material_clipping": True,
        "material_source_defect": True,
        "maximum_clipping_defect_correlation": 0.9,
        "maximum_wall_band_defect_share": 0.4,
        "maximum_material_cell_fraction": 0.1,
    }
    row.update(updates)
    return row


def test_frozen_design_accepts_exact_values():
    validate_stage65_design()


def test_frozen_design_rejects_retuning():
    with pytest.raises(ValueError, match="not retuning"):
        validate_stage65_design(radial_scale=1.0)


def test_wall_distance_layers_are_exact():
    distance = wall_distance_layers((5, 6))
    assert distance[0, 3] == 0
    assert distance[2, 3] == 2
    assert distance[-1, -1] == 0


def test_safe_correlation_handles_constant_and_linear_data():
    assert safe_correlation(np.ones(4), np.arange(4)) == 0.0
    assert safe_correlation(np.arange(5), 2.0 * np.arange(5)) == pytest.approx(1.0)


def test_concentration_metrics_for_single_hot_cell():
    field = np.zeros((10, 10))
    field[2, 3] = 5.0
    result = concentration_metrics(field)
    assert result["top_1_percent_share"] == pytest.approx(1.0)
    assert result["top_5_percent_share"] == pytest.approx(1.0)
    assert result["top_10_percent_share"] == pytest.approx(1.0)


def test_wall_band_metrics_partition_total():
    field = np.ones((8, 8))
    distance = wall_distance_layers(field.shape)
    result = wall_band_metrics(field, distance, wall_band_layers=1)
    assert result["wall_band_share"] + result["interior_share"] == pytest.approx(1.0)
    assert result["wall_band_cell_fraction"] == pytest.approx(28.0 / 64.0)


def test_maximum_location_and_weighted_centroid():
    field = np.zeros((4, 4))
    field[1, 2] = 3.0
    assert maximum_location(field)["index"] == [1, 2]
    assert weighted_centroid(field) == pytest.approx([0.375, 0.625])


def test_decision_nonfinite_blocker():
    assert stage65_decision(_metrics(finite=False)) == "stage65_nonfinite_local_activation_map_blocker"


def test_decision_no_material_stop():
    decision = stage65_decision(_metrics(material_clipping=False))
    assert decision == "stage65_no_material_full_cavity_clipping_source_defect_stop"


def test_decision_weak_correlation_blocker():
    decision = stage65_decision(
        _metrics(
            maximum_clipping_defect_correlation=(
                STAGE65_MINIMUM_CLIPPING_DEFECT_CORRELATION - 1e-6
            )
        )
    )
    assert decision == "stage65_material_defect_weakly_correlated_with_clipping_blocker"


def test_decision_wall_localized_route():
    decision = stage65_decision(
        _metrics(maximum_wall_band_defect_share=STAGE65_WALL_LOCALIZATION_THRESHOLD)
    )
    assert "wall_localized" in decision
    assert "stage66" in decision


def test_decision_broad_route():
    decision = stage65_decision(
        _metrics(maximum_material_cell_fraction=STAGE65_BROAD_CELL_FRACTION_THRESHOLD)
    )
    assert "broad" in decision
    assert "stage66" in decision


def test_decision_mixed_route():
    decision = stage65_decision(_metrics())
    assert "mixed_localization" in decision
    assert "stage66" in decision
