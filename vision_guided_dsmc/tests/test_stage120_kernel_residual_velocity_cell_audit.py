import numpy as np
import pytest

from vgdsmc import stage120_kernel_residual_velocity_cell_audit as s120


def test_stage120_design_is_frozen():
    s120.validate_stage120_design()
    with pytest.raises(ValueError):
        s120.validate_stage120_design(pair_sectors=(4, 5))
    with pytest.raises(ValueError):
        s120.validate_stage120_design(velocity_cell_shape=(5, 4))


def test_stage120_profile_metrics_identity():
    p = np.arange(1.0, 21.0).reshape(10, 2)
    m = s120._profile_metrics(p, p)
    assert m["profile_cosine"] == pytest.approx(1.0)
    assert m["overlap_coefficient"] == pytest.approx(1.0)
    assert m["total_variation_distance"] == pytest.approx(0.0)


def test_stage120_radial_decision_requires_tv_capture_closure():
    metrics = {
        band: {"radial_tv_capture_fraction": 1.0}
        for band in s120.BANDS
    }
    assert s120.stage120_decision(metrics, True, 1.0e-16) == s120.RADIAL
    mixed = {band: dict(v) for band, v in metrics.items()}
    mixed["inner_15_28"]["radial_tv_capture_fraction"] = 0.99
    assert s120.stage120_decision(mixed, True, 1.0e-16) == s120.MIXED


def test_stage120_decision_blockers_are_explicit():
    metrics = {band: {"radial_tv_capture_fraction": 1.0} for band in s120.BANDS}
    assert s120.stage120_decision(metrics, False, 0.0) == s120.NONFINITE
    assert s120.stage120_decision(metrics, True, 1.0e-8) == s120.CLOSURE_BLOCKER
