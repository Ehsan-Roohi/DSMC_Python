import math
import pytest

from vgdsmc.stage36_high_resolution_cross_kn import (
    STAGE35_BASELINES,
    STAGE36_CASES,
    case_key,
    stage36_decision,
    validate_stage36_design,
)


def make_row(case, q_error, velocity_error, sign=1.0, converged=True):
    return {
        "case": case,
        "converged": converged,
        "qav_relative_error": q_error,
        "velocity_metrics": {
            "relative_rms": velocity_error,
            "sign_agreement": sign,
        },
    }


def complete_rows(low_q=0.05, low_v=0.25, middle_q=0.03, middle_v=0.2,
                  high_q=0.03, high_v=0.2):
    return [
        make_row("kn0p1_24x24", 0.08, 0.4, 0.9),
        make_row("kn0p1_36x36", low_q, low_v, 0.9),
        make_row("kn1p0_24x24", middle_q, middle_v, 1.0),
        make_row("kn10p0_24x24", high_q, high_v, 1.0),
    ]


def test_stage36_design_accepts_only_preregistered_matrix():
    validate_stage36_design(STAGE36_CASES, 14000, 2e-5)


def test_stage36_design_rejects_case_retuning():
    with pytest.raises(ValueError):
        validate_stage36_design(STAGE36_CASES[:-1], 14000, 2e-5)
    changed = list(STAGE36_CASES)
    changed[0] = ("kn0p1_20x20", 0.1, (20, 20))
    with pytest.raises(ValueError):
        validate_stage36_design(tuple(changed), 14000, 2e-5)


def test_stage36_design_rejects_nonpositive_controls():
    with pytest.raises(ValueError):
        validate_stage36_design(STAGE36_CASES, 0, 2e-5)
    with pytest.raises(ValueError):
        validate_stage36_design(STAGE36_CASES, 14000, 0.0)


def test_stage35_baselines_are_exact_and_complete():
    assert set(STAGE35_BASELINES) == {0.1, 1.0, 10.0}
    assert STAGE35_BASELINES[0.1]["predicted_qav"] == pytest.approx(
        0.07954802746383147
    )
    assert STAGE35_BASELINES[1.0]["qav_relative_error"] == pytest.approx(
        0.03523097022297649
    )
    assert STAGE35_BASELINES[10.0]["wall_velocity_sign_agreement"] == 1.0
    for baseline in STAGE35_BASELINES.values():
        assert baseline["grid"] == [12, 12]
        assert baseline["best_wall_observable"] == "linear_extrapolated_wall"
        assert math.isfinite(baseline["wall_mass_balance_relative_error"])


def test_case_key_is_stable_and_unambiguous():
    assert case_key(0.1, (24, 24)) == "kn0p1_24x24"
    assert case_key(1.0, (24, 24)) == "kn1p0_24x24"
    assert case_key(10.0, (36, 36)) == "kn10p0_36x36"


def test_stage36_decision_advances_independent_validation_when_all_supported():
    decision = stage36_decision(complete_rows(), 0.10)
    assert decision == (
        "cross_kn_quantitative_support_advance_independent_reference_stage37"
    )


def test_stage36_decision_isolates_low_kn_failure_without_hiding_it():
    rows = complete_rows(low_q=0.14, low_v=0.7)
    decision = stage36_decision(rows, 0.25)
    assert decision == (
        "high_kn_supported_low_kn_requires_transport_or_collision_audit_stage37"
    )


def test_stage36_decision_preserves_nonconvergence_as_a_blocking_result():
    rows = complete_rows()
    rows[-1]["converged"] = False
    assert stage36_decision(rows, 0.10) == (
        "high_resolution_nonconvergence_requires_numerical_stability_audit_stage37"
    )


def test_stage36_decision_reports_general_discrepancy():
    rows = complete_rows(middle_q=0.2, middle_v=1.0, high_q=0.2, high_v=1.0)
    assert stage36_decision(rows, 0.10) == (
        "high_resolution_retains_cross_kn_discrepancy_stage37_model_observable_audit"
    )
