import pytest

from vgdsmc import stage96_muscl_correction_growth_audit as stage96


def test_stage96_frozen_design_accepts_defaults():
    stage96.validate_stage96_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 1.0},
        {"grid": (32, 32)},
        {"source_relaxation": 0.5},
        {"radial_scale": 1.0},
        {"diagnostic_steps": 50},
        {"boundary_slope": "one_sided_first_difference_x_only"},
        {"material_correction_ratio": 0.2},
    ],
)
def test_stage96_frozen_design_rejects_retuning(override):
    with pytest.raises(ValueError):
        stage96.validate_stage96_design(**override)


def test_safe_ratio_is_finite_for_zero_denominator():
    assert stage96._safe_ratio(0.0, 0.0) == 0.0
    assert stage96._safe_ratio(1.0e-300, 0.0) == 1.0


def test_history_summary_reports_growth_without_retuning():
    summary = stage96._history_summary([0.25, 0.5, 0.75])
    assert summary["first"] == pytest.approx(0.25)
    assert summary["final"] == pytest.approx(0.75)
    assert summary["maximum"] == pytest.approx(0.75)
    assert summary["maximum_to_first_ratio"] == pytest.approx(3.0)


def test_decision_preserves_nonfinite_blocker():
    assert (
        stage96.stage96_decision(
            finite=False,
            maximum_correction_to_transport_ratio=1.0,
            maximum_growth_ratio=10.0,
        )
        == "stage96_nonfinite_fixed_window_blocker_without_retuning"
    )


def test_decision_routes_material_growing_signal_to_localization_only():
    assert (
        stage96.stage96_decision(
            finite=True,
            maximum_correction_to_transport_ratio=0.2,
            maximum_growth_ratio=2.5,
        )
        == "stage96_material_and_growing_muscl_correction_stage97_spatial_localization_audit"
    )


def test_decision_routes_material_persistent_signal_to_localization_only():
    assert (
        stage96.stage96_decision(
            finite=True,
            maximum_correction_to_transport_ratio=0.2,
            maximum_growth_ratio=1.5,
        )
        == "stage96_material_persistent_muscl_correction_stage97_spatial_localization_audit"
    )


def test_decision_preserves_submaterial_blocker_without_parameter_change():
    assert (
        stage96.stage96_decision(
            finite=True,
            maximum_correction_to_transport_ratio=0.05,
            maximum_growth_ratio=10.0,
        )
        == "stage96_muscl_correction_submaterial_in_fixed_window_blocker_without_retuning"
    )
