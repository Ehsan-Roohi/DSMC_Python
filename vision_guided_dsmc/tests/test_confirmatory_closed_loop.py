import pytest

from vgdsmc.confirmatory_closed_loop import _condition_summaries


def test_condition_summaries_group_temperature_differences():
    rows = [
        {"delta_temperature": 20.0, "error_ratio": 0.90},
        {"delta_temperature": 20.0, "error_ratio": 1.00},
        {"delta_temperature": 40.0, "error_ratio": 1.10},
        {"delta_temperature": 40.0, "error_ratio": 0.90},
        {"delta_temperature": 60.0, "error_ratio": 0.80},
        {"delta_temperature": 60.0, "error_ratio": 0.90},
    ]
    summary = _condition_summaries(rows)
    assert set(summary) == {"deltaT_20", "deltaT_40", "deltaT_60"}
    assert summary["deltaT_20"]["mean"] == pytest.approx(0.95)
    assert summary["deltaT_20"]["improved_runs"] == 1
    assert summary["deltaT_40"]["mean"] == pytest.approx(1.0)
    assert summary["deltaT_60"]["mean"] == pytest.approx(0.85)
    for metrics in summary.values():
        assert metrics["run_count"] == 2
        assert metrics["ci95_low"] <= metrics["mean"] <= metrics["ci95_high"]
