import pytest

from vgdsmc import stage91_fixed_parameter_nonconvergence_onset_audit as stage91


def test_stage91_frozen_design_accepts_defaults():
    stage91.validate_stage91_design()


@pytest.mark.parametrize(
    "override",
    [
        {"kn0": 1.0},
        {"grid": (32, 32)},
        {"source_relaxation": 0.5},
        {"radial_scale": 1.0},
        {"diagnostic_steps": 50},
    ],
)
def test_stage91_frozen_design_rejects_retuning(override):
    with pytest.raises(ValueError):
        stage91.validate_stage91_design(**override)


def test_first_positive_reports_one_based_activation_step():
    assert stage91._first_positive([0.0, 0.0, 0.125]) == 3
    assert stage91._first_positive([0.0, 0.0, 0.0]) is None


def _arm(*, finite=True, phi_step=None, psi_step=None):
    return {
        "finite": finite,
        "first_phi_floor_activation_step": phi_step,
        "first_psi_floor_activation_step": psi_step,
    }


def test_stage91_decision_preserves_both_arm_floor_activation_route():
    decision = stage91.stage91_decision(
        _arm(phi_step=1),
        _arm(psi_step=2),
    )
    assert decision.startswith("stage91_both_arms_activate_positivity_floor")


def test_stage91_decision_preserves_boundary_specific_route():
    decision = stage91.stage91_decision(
        _arm(),
        _arm(phi_step=2),
    )
    assert decision.startswith("stage91_boundary_specific_floor_onset_signal")


def test_stage91_decision_preserves_baseline_only_route():
    decision = stage91.stage91_decision(
        _arm(psi_step=3),
        _arm(),
    )
    assert decision.startswith("stage91_baseline_floor_onset_precedes")


def test_stage91_decision_preserves_nonfinite_blocker():
    decision = stage91.stage91_decision(
        _arm(),
        _arm(finite=False),
    )
    assert decision == "stage91_nonfinite_onset_blocker_without_retuning"


def test_stage91_decision_preserves_no_floor_blocker():
    decision = stage91.stage91_decision(
        _arm(),
        _arm(),
    )
    assert decision == "stage91_no_floor_onset_within_fixed_window_blocker_without_retuning"
