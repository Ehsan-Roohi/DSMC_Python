import numpy as np

from vgdsmc.effective_sampling_allocation import (
    composite_variance_priority,
    lag1_autocorrelation,
)
from vgdsmc.vhs_model import PhysicalCavityConfig


def _ar1_series(rho, scale, time_steps=80, ny=3, nx=4, seed=5):
    rng = np.random.default_rng(seed)
    output = np.zeros((time_steps, ny, nx), dtype=float)
    noise_scale = scale * np.linspace(0.5, 1.5, nx)[None]
    for step in range(1, time_steps):
        output[step] = rho * output[step - 1] + rng.normal(
            0.0,
            noise_scale,
            size=(ny, nx),
        )
    return output


def test_lag1_autocorrelation_detects_persistent_series():
    values = _ar1_series(0.85, 1.0)
    correlation = lag1_autocorrelation(values)
    assert correlation.shape == (3, 4)
    assert np.isfinite(correlation).all()
    assert float(np.mean(correlation)) > 0.5
    assert float(correlation.min()) >= -0.5
    assert float(correlation.max()) <= 0.95


def test_effective_variance_priority_is_finite_and_nonuniform():
    cfg = PhysicalCavityConfig(nx=4, ny=3, particles_per_cell=5)
    temperature_noise = _ar1_series(0.8, 2.0, ny=3, nx=4)
    density_noise = _ar1_series(0.5, 0.01, ny=3, nx=4, seed=7)
    velocity_noise = _ar1_series(0.7, 5.0, ny=3, nx=4, seed=9)
    snapshots = {
        "T": 300.0 + temperature_noise,
        "rho": 1.0 + density_noise,
        "u": velocity_noise,
        "v": 0.5 * velocity_noise,
    }
    raw = composite_variance_priority(
        snapshots,
        cfg,
        autocorrelation_corrected=False,
    )
    effective = composite_variance_priority(
        snapshots,
        cfg,
        autocorrelation_corrected=True,
    )
    assert raw.shape == (3, 4)
    assert effective.shape == (3, 4)
    assert np.isfinite(raw).all()
    assert np.isfinite(effective).all()
    assert np.std(raw) > 0.0
    assert np.std(effective) > 0.0
    assert np.isclose(np.mean(raw), 1.0)
    assert np.isclose(np.mean(effective), 1.0)
    assert not np.allclose(raw, effective)
