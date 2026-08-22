import numpy as np
import pytest

from vgdsmc import stage108_limiter_severity_correction_amplitude_coupling_audit as s


def test_stage108_design_is_frozen():
    s.validate_stage108_design()
    with pytest.raises(ValueError):
        s.validate_stage108_design(limiter="vanleer")
    with pytest.raises(ValueError):
        s.validate_stage108_design(rank_coupling_guard=0.1)
    with pytest.raises(ValueError):
        s.validate_stage108_design(stage107_run_id=-1)


def test_average_ranks_ties_are_average_ranked():
    x = np.array([1.0, 2.0, 2.0, 4.0])
    assert np.allclose(s._average_ranks(x), [1.0, 2.5, 2.5, 4.0])


def test_spearman_detects_monotone_relation():
    x = np.arange(12.0, dtype=float)
    assert np.isclose(s._spearman(x, x * x), 1.0)


def test_coupling_metrics_strong_stratification(monkeypatch):
    severity = np.arange(64.0, dtype=float).reshape(8, 8)
    amplitude = 1.0 + 3.0 * severity
    monkeypatch.setattr(s, "INTERIOR_EXTENT", 8)
    metrics, support = s._coupling_metrics(severity, amplitude)
    assert metrics["spearman"] > 0.99
    assert metrics["upper_to_lower_mean_amplitude_ratio"] > 1.5
    assert support["high_support"].shape == (8, 8)


def test_coupling_metrics_rejects_negative_amplitude():
    severity = np.ones((56, 56))
    amplitude = np.ones((56, 56))
    amplitude[0, 0] = -1.0
    with pytest.raises(ValueError):
        s._coupling_metrics(severity, amplitude)


def test_stage108_decision_strong_route():
    block = {"spearman": 0.6, "upper_to_lower_mean_amplitude_ratio": 2.0}
    metrics = {"phi": dict(block), "psi": dict(block), "joint": dict(block)}
    assert s.stage108_decision(metrics, True) == (
        "stage108_continuous_limiter_severity_coupling_stage109_limiter_intervention_mode_decomposition_audit"
    )


def test_stage108_decision_partial_route():
    block = {"spearman": 0.6, "upper_to_lower_mean_amplitude_ratio": 1.1}
    metrics = {"phi": dict(block), "psi": dict(block), "joint": dict(block)}
    assert s.stage108_decision(metrics, True) == (
        "stage108_partial_continuous_severity_coupling_stage109_spatial_monotonicity_audit"
    )


def test_stage108_decision_negative_route():
    block = {"spearman": 0.1, "upper_to_lower_mean_amplitude_ratio": 1.1}
    metrics = {"phi": dict(block), "psi": dict(block), "joint": dict(block)}
    assert s.stage108_decision(metrics, True) == (
        "stage108_no_continuous_severity_coupling_stage109_unlimited_gradient_smoothness_audit"
    )


def test_stage108_decision_nonfinite_blocker():
    assert s.stage108_decision({}, False) == (
        "stage108_nonfinite_severity_amplitude_coupling_blocker_without_retuning"
    )


def test_quantile_and_guard_contracts_remain_preregistered():
    assert s.LOWER_QUANTILE == 0.25
    assert s.UPPER_QUANTILE == 0.75
    assert s.RANK_COUPLING_GUARD == 0.40
    assert s.QUARTILE_AMPLITUDE_RATIO_GUARD == 1.50
    assert s.STAGE107_DECISION.endswith("stage108_limiter_severity_correction_amplitude_coupling_audit")
