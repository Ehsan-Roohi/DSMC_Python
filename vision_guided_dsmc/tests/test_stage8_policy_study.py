import numpy as np

from vgdsmc.physical_policy_study import (
    priority_candidates,
    summarize_policy,
)


def test_priority_candidates_are_finite_and_shape_preserving():
    fields = {
        "T": np.arange(64, dtype=float).reshape(8, 8) + 280.0,
        "rho": np.linspace(0.8, 1.2, 64).reshape(8, 8),
        "sigma_T": np.linspace(1.0, 5.0, 64).reshape(8, 8),
    }
    candidates = priority_candidates(fields)
    assert {
        "current",
        "grad_t",
        "grad_rho",
        "noise",
        "grad_t_noise",
        "snr_grad_t",
        "curvature_t",
        "sidewall",
        "sidewall_grad_t",
    } == set(candidates)
    for priority in candidates.values():
        assert priority.shape == (8, 8)
        assert np.isfinite(priority).all()
        assert priority.min() >= 0.0
        assert priority.max() <= 1.0


def test_policy_summary_uses_independent_seed_ratios():
    summary = summarize_policy(
        [0.90, 1.00, 1.10, 0.95, 1.05],
        bootstrap_seed=2,
    )
    assert np.isclose(summary["mean_ratio"], 1.0)
    assert summary["improved_seeds"] == 2
    assert summary["worst_ratio"] == 1.10
    assert len(summary["t_95_ci"]) == 2
    assert len(summary["bootstrap_95_ci"]) == 2
