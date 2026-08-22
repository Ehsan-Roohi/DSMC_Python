import numpy as np
import pytest

from vgdsmc.sampling_allocation import (
    estimate_from_sampling_counts,
    exact_sampling_counts,
    full_trajectory_mean,
    pilot_variance_priority,
)


def _synthetic_snapshots(time_steps=40, ny=4, nx=5):
    time = np.linspace(0.0, 2.0 * np.pi, time_steps, endpoint=False)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nx),
        indexing="ij",
    )
    amplitude = 0.05 + xx + 0.5 * yy
    temperature = 300.0 + amplitude[None] * np.sin(time[:, None, None])
    density = 1.0 + 0.02 * amplitude[None] * np.cos(time[:, None, None])
    u = amplitude[None] * np.sin(2.0 * time[:, None, None])
    v = 0.5 * amplitude[None] * np.cos(2.0 * time[:, None, None])
    w = np.zeros_like(u)
    return {"T": temperature, "rho": density, "u": u, "v": v, "w": w}


def test_pilot_variance_priority_tracks_spatial_variability():
    snapshots = _synthetic_snapshots()
    priority = pilot_variance_priority(snapshots)
    assert priority.shape == (4, 5)
    assert np.isfinite(priority).all()
    assert float(priority[-1, -1]) > float(priority[0, 0])
    assert 0.0 <= float(priority.min()) <= float(priority.max()) <= 1.0


@pytest.mark.parametrize(
    "minimum_samples,maximum_samples",
    [(15, 25), (10, 30)],
)
def test_exact_sampling_counts_preserve_total_budget(
    minimum_samples,
    maximum_samples,
):
    priority = np.arange(20, dtype=float).reshape(4, 5)
    counts = exact_sampling_counts(
        priority,
        base_samples_per_cell=20,
        minimum_samples=minimum_samples,
        maximum_samples=maximum_samples,
    )
    assert counts.shape == priority.shape
    assert int(counts.sum()) == 20 * 20
    assert int(counts.min()) >= minimum_samples
    assert int(counts.max()) <= maximum_samples
    assert np.std(counts) > 0.0


def test_nested_temporal_estimator_has_expected_shapes_and_full_mean():
    snapshots = _synthetic_snapshots(time_steps=40)
    counts = np.full((4, 5), 20, dtype=int)
    estimate = estimate_from_sampling_counts(snapshots, counts, seed=17)
    full = full_trajectory_mean(snapshots)
    assert set(estimate) == {"T", "rho", "u", "v", "w"}
    assert set(full) == set(estimate)
    for name in estimate:
        assert estimate[name].shape == (4, 5)
        assert np.isfinite(estimate[name]).all()
        assert full[name].shape == (4, 5)


def test_estimator_rejects_more_samples_than_available():
    snapshots = _synthetic_snapshots(time_steps=10)
    counts = np.full((4, 5), 11, dtype=int)
    with pytest.raises(ValueError, match="exceed"):
        estimate_from_sampling_counts(snapshots, counts, seed=1)
