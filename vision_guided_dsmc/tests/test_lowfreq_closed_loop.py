import numpy as np
import pytest

from vgdsmc.lowfreq_closed_loop import (
    _paired_statistics,
    exact_low_amplitude_ppc,
)


def test_constant_priority_returns_uniform_exact_budget():
    target = exact_low_amplitude_ppc(
        np.ones((6, 6)),
        base_ppc=20,
        amplitude=0.20,
    )
    assert target.shape == (6, 6)
    assert np.all(target == 20)
    assert int(target.sum()) == 6 * 6 * 20


@pytest.mark.parametrize("amplitude", [0.05, 0.10, 0.20])
def test_nonuniform_priority_respects_exact_budget_and_bounds(amplitude):
    priority = np.arange(36, dtype=float).reshape(6, 6)
    target = exact_low_amplitude_ppc(priority, base_ppc=20, amplitude=amplitude)
    minimum = max(2, int(np.floor(20 * (1.0 - amplitude))))
    maximum = int(np.ceil(20 * (1.0 + amplitude)))
    assert int(target.sum()) == 6 * 6 * 20
    assert int(target.min()) >= minimum
    assert int(target.max()) <= maximum
    assert np.std(target) > 0.0


def test_paired_statistics_are_finite_and_ordered():
    summary = _paired_statistics(np.array([0.95, 1.00, 1.05]))
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["ci95_low"] <= summary["mean"] <= summary["ci95_high"]
    assert summary["standard_error"] > 0.0
