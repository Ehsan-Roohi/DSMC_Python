import numpy as np
import pytest

from vgdsmc.dvm_convergence import (
    _is_monotone_nonincreasing,
    _observed_orders,
    field_difference_metrics,
    interpolate_cell_center_field,
)
from vgdsmc.dvm_shakhov import ShakhovReferenceConfig


def test_interpolation_preserves_constant_field_on_new_grid():
    field = np.full((4, 5), 3.25)
    interpolated = interpolate_cell_center_field(field, 9, 8)
    assert interpolated.shape == (9, 8)
    assert np.allclose(interpolated, 3.25)


def test_interpolation_identity_returns_copy():
    field = np.arange(20, dtype=float).reshape(4, 5)
    interpolated = interpolate_cell_center_field(field, 4, 5)
    assert np.array_equal(interpolated, field)
    assert interpolated is not field


def test_identical_fields_have_zero_convergence_error():
    ny, nx = 4, 5
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nx),
        indexing="ij",
    )
    fields = {
        "T": 300.0 + 10.0 * xx,
        "rho": 1.0 + 0.02 * yy,
        "u": 2.0 * xx,
        "v": -1.0 * yy,
        "qx": 100.0 + xx,
        "qy": -50.0 + yy,
    }
    cfg = ShakhovReferenceConfig(nx=nx, ny=ny, nv=8)
    metrics = field_difference_metrics(fields, fields, cfg)
    assert set(metrics) == {
        "temperature_relative_rms",
        "density_relative_rms",
        "velocity_thermal_rms",
        "heat_flux_relative_rms",
        "composite_error",
    }
    assert all(value == pytest.approx(0.0) for value in metrics.values())


def test_monotonicity_and_orders():
    assert _is_monotone_nonincreasing([0.3, 0.2, 0.1])
    assert not _is_monotone_nonincreasing([0.3, 0.35, 0.1])
    orders = _observed_orders((4, 8, 16), [0.25, 0.0625, 0.015625])
    assert orders == pytest.approx([2.0, 2.0])
