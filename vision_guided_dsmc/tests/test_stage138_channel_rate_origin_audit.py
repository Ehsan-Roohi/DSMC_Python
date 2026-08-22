import numpy as np
import pytest

from vgdsmc import stage138_channel_rate_origin_audit as s138


def test_stage138_design_is_frozen():
    s138.validate_stage138_design()
    with pytest.raises(ValueError):
        s138.validate_stage138_design(kn0=9.0)
    with pytest.raises(ValueError):
        s138.validate_stage138_design(physical_parameter_retuning=True)
    with pytest.raises(ValueError):
        s138.validate_stage138_design(cross_knudsen_extension_permitted=True)


def test_fitted_decay_rate_recovers_exact_exponential():
    x = np.arange(7.0)
    y = 3.0 * np.exp(-0.23 * x)
    assert s138.fitted_decay_rate(x, y) == pytest.approx(0.23, rel=1e-12)


def test_rate_identity_with_depth_varying_mixture():
    x = np.arange(7.0)
    dominant = -np.exp(-0.12 * x)
    mixing = 1.25 * np.exp(-0.08 * x)
    parent = dominant * mixing
    kd = s138.fitted_decay_rate(x, np.abs(dominant))
    kp = s138.fitted_decay_rate(x, np.abs(parent))
    kmix = s138.fitted_decay_rate(x, mixing)
    assert kp == pytest.approx(kd + kmix, abs=1e-12)


def test_first_sign_change_depth_is_linearly_interpolated():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-2.0, 1.0, 2.0])
    assert s138.first_sign_change_depth(x, y) == pytest.approx(2.0 / 3.0)


def test_pointwise_cancellation_fraction_distinguishes_reinforcement_and_opposition():
    d = np.array([-2.0, -2.0])
    c = np.array([-1.0, 1.0])
    out = s138.pointwise_cancellation_fraction(d, c)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(2.0 / 3.0)


def test_stage138_classifies_depth_varying_cancellation():
    assert s138.classify_rate_origin(
        sign_change_count=1,
        ratio_swing=0.6,
        endpoint_cancellation_fraction=0.6,
        rate_identity_closure=1e-16,
        rate_split_explained_fraction=1.0,
    ) == s138.DEPTH_VARYING_CANCELLATION


def test_stage138_classifies_same_sign_mixture_and_blockers():
    assert s138.classify_rate_origin(
        sign_change_count=0,
        ratio_swing=0.4,
        endpoint_cancellation_fraction=0.0,
        rate_identity_closure=1e-16,
        rate_split_explained_fraction=1.0,
    ) == s138.SAME_SIGN_MIXTURE
    assert s138.classify_rate_origin(
        sign_change_count=1,
        ratio_swing=0.6,
        endpoint_cancellation_fraction=0.6,
        rate_identity_closure=1e-16,
        rate_split_explained_fraction=1.0,
        parent_record_ok=False,
    ) == s138.PARENT_RECORD_BLOCKER
    assert s138.classify_rate_origin(
        sign_change_count=1,
        ratio_swing=0.6,
        endpoint_cancellation_fraction=0.6,
        rate_identity_closure=1e-16,
        rate_split_explained_fraction=1.0,
        parent_profile_closure=2e-12,
    ) == s138.PARENT_PROFILE_BLOCKER
