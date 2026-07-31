import math
import numpy as np

from vgdsmc.linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE3_Y,
    TABLE6_QAV_RATIO_0P1,
    local_relaxation_time,
    paper_relaxation_scale,
    sidewall_temperature_profile,
)


def test_sidewall_profile_is_linear_hot_to_cold():
    cfg = LinearSidewallConfig(ny=20)
    profile = sidewall_temperature_profile(cfg)
    assert profile.shape == (20,)
    assert np.all(np.diff(profile) < 0.0)
    expected = 1.0 - 0.9 * ((np.arange(20) + 0.5) / 20)
    assert np.allclose(profile, expected)


def test_relaxation_scale_matches_paper_equation():
    assert math.isclose(paper_relaxation_scale(1.0), 2.0 / math.sqrt(math.pi))
    cfg = LinearSidewallConfig(kn0=1.0)
    tau = local_relaxation_time(np.ones((2, 2)), np.ones((2, 2)), cfg)
    assert np.allclose(tau, 2.0 / math.sqrt(math.pi))


def test_literature_tables_are_complete():
    assert np.allclose(TABLE3_Y, np.arange(0.05, 1.0, 0.10))
    assert set(TABLE3_UY_RATIO_0P1) == {0.1, 1.0, 10.0}
    assert all(values.shape == (10,) for values in TABLE3_UY_RATIO_0P1.values())
    assert TABLE6_QAV_RATIO_0P1 == {0.1: 0.072, 1.0: 0.148, 10.0: 0.178}


def test_high_kn_reference_velocity_is_hot_to_cold():
    assert np.all(TABLE3_UY_RATIO_0P1[1.0] > 0.0)
    assert np.all(TABLE3_UY_RATIO_0P1[10.0] > 0.0)
