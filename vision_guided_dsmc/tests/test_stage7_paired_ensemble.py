import numpy as np

from vgdsmc.physical_paired_ensemble import (
    _two_sided_sign_p,
    summarize_clusters,
)


def _cluster(seed: int, ratio: float, difference: float) -> dict:
    return {
        "base_seed": seed,
        "cluster_ratio": ratio,
        "cluster_difference": difference,
        "pairs": [{"repetition": 0}, {"repetition": 1}, {"repetition": 2}],
    }


def test_summary_uses_independent_warm_seed_as_statistical_unit():
    clusters = [
        _cluster(3, 0.90, -0.01),
        _cluster(1, 1.00, 0.00),
        _cluster(2, 1.10, 0.01),
    ]
    summary = summarize_clusters(
        clusters,
        bootstrap_samples=500,
        bootstrap_seed=4,
    )
    assert summary["independent_warm_seeds"] == 3
    assert summary["paired_continuations_per_seed"] == 3
    assert summary["paired_comparisons"] == 9
    assert summary["improved_warm_seeds"] == 1
    assert np.isclose(summary["mean_cluster_ratio"], 1.0)
    assert np.isclose(summary["mean_paired_error_difference"], 0.0)
    assert summary["adaptive_to_uniform_particle_ratio"] == 1.0


def test_two_sided_sign_test_is_exact_and_bounded():
    assert np.isclose(_two_sided_sign_p(0, 10), 2.0 / 1024.0)
    assert np.isclose(_two_sided_sign_p(10, 10), 2.0 / 1024.0)
    assert _two_sided_sign_p(5, 10) == 1.0
