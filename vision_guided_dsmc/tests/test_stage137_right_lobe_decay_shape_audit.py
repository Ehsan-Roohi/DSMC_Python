import numpy as np
import pytest

from vgdsmc import stage137_right_lobe_decay_shape_audit as s137


def _smooth_block(rate=0.2):
    return {
        "positive_decay_rate": True,
        "decay_rate_per_cell": rate,
        "log_linear_r2": 0.99,
        "relative_l2_residual": 0.02,
        "nonincreasing_step_fraction": 1.0,
    }


def test_stage137_design_is_frozen():
    s137.validate_stage137_design()
    with pytest.raises(ValueError):
        s137.validate_stage137_design(kn0=9.0)
    with pytest.raises(ValueError):
        s137.validate_stage137_design(physical_parameter_retuning=True)


def test_decay_metrics_exact_exponential():
    x = np.arange(7.0)
    y = 2.5 * np.exp(-0.2 * x)
    metrics, fit = s137.decay_metrics(x, y)
    assert metrics["positive_decay_rate"] is True
    assert metrics["decay_rate_per_cell"] == pytest.approx(0.2, rel=1e-12)
    assert metrics["log_linear_r2"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["relative_l2_residual"] <= 1e-12
    assert metrics["nonincreasing_step_fraction"] == 1.0
    assert np.allclose(fit, y)


def test_stage137_classifies_smooth_common_rate():
    assert s137.classify_decay_shape(
        dominant=_smooth_block(0.20),
        parent=_smooth_block(0.22),
        common_rate_relative_difference=0.095,
    ) == s137.SMOOTH_COMMON_RATE


def test_stage137_classifies_smooth_rate_split():
    assert s137.classify_decay_shape(
        dominant=_smooth_block(0.14),
        parent=_smooth_block(0.28),
        common_rate_relative_difference=2.0 / 3.0,
    ) == s137.SMOOTH_RATE_SPLIT


def test_stage137_classifies_resolved_structure_and_blockers():
    rough = _smooth_block(0.2)
    rough["relative_l2_residual"] = 0.11
    assert s137.classify_decay_shape(
        dominant=rough,
        parent=_smooth_block(0.2),
        common_rate_relative_difference=0.0,
    ) == s137.RESOLVED_STRUCTURE
    assert s137.classify_decay_shape(
        dominant=_smooth_block(),
        parent=_smooth_block(),
        common_rate_relative_difference=0.0,
        parent_record_ok=False,
    ) == s137.PARENT_RECORD_BLOCKER
    assert s137.classify_decay_shape(
        dominant=_smooth_block(),
        parent=_smooth_block(),
        common_rate_relative_difference=0.0,
        parent_profile_closure=2.0e-12,
    ) == s137.PARENT_PROFILE_BLOCKER
