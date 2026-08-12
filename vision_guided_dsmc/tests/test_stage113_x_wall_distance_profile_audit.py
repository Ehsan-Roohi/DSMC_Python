from __future__ import annotations

import numpy as np
import pytest

from vgdsmc import stage113_x_wall_distance_profile_audit as s113


def test_frozen_design_accepts_exact_values():
    s113.validate_stage113_design(
        grid=(64, 64), kn0=10.0, cold_hot_ratio=0.1, rule=(40, 96),
        radial_scale=2.0, limiter="minmod", boundary_slope="zero",
        stage112_run_id=31618371834,
    )


def test_frozen_design_rejects_parameter_retuning():
    with pytest.raises(ValueError):
        s113.validate_stage113_design(kn0=0.1)


def test_frozen_design_rejects_velocity_quadrature_retuning():
    with pytest.raises(ValueError):
        s113.validate_stage113_design(rule=(32, 64))


def test_wall_distance_profile_uniform_map_is_uniform():
    a = np.ones((56, 56), dtype=np.float64)
    p = s113.wall_distance_profile(a)
    np.testing.assert_allclose(p, np.full(28, 1.0 / 28.0), rtol=0.0, atol=1e-15)


def test_wall_distance_profile_preserves_normalization():
    rng = np.random.default_rng(3)
    a = rng.random((56, 56))
    p = s113.wall_distance_profile(a)
    assert p.shape == (28,)
    assert np.isclose(np.sum(p), 1.0, rtol=0.0, atol=1e-15)
    assert np.all(p >= 0.0)


def test_wall_distance_profile_pairs_opposite_x_columns():
    a = np.zeros((56, 56))
    a[:, 0] = 1.0
    a[:, -1] = 3.0
    p = s113.wall_distance_profile(a)
    assert p[0] == pytest.approx(1.0)
    assert np.count_nonzero(p[1:]) == 0


def test_wall_distance_profile_rejects_wrong_shape():
    with pytest.raises(ValueError):
        s113.wall_distance_profile(np.ones((55, 56)))


def test_wall_distance_profile_rejects_negative_values():
    a = np.ones((56, 56))
    a[0, 0] = -1.0
    with pytest.raises(ValueError):
        s113.wall_distance_profile(a)


def test_profile_metrics_known_uniform_depths():
    m = s113.profile_metrics(np.ones(28))
    assert m["half_mass_depth_cells"] == 14
    assert m["three_quarter_mass_depth_cells"] == 21
    assert m["ninety_percent_mass_depth_cells"] == 26
    assert m["effective_profile_bin_count"] == pytest.approx(28.0)
    assert m["first_14_cumulative_share"] == pytest.approx(0.5)


def test_decision_routes_thin_layer_without_retuning():
    m = {
        "first_4_cumulative_share": 0.61,
        "first_14_cumulative_share": 0.92,
        "half_mass_depth_cells": 3,
        "effective_profile_bin_count": 8.0,
    }
    assert s113.stage113_decision(m) == "stage113_thin_x_wall_layer_stage114_near_wall_velocity_quadrature_audit"


def test_decision_routes_broad_profile_to_velocity_quadrature_audit():
    m = {
        "first_4_cumulative_share": 0.35,
        "first_14_cumulative_share": 0.82,
        "half_mass_depth_cells": 7,
        "effective_profile_bin_count": 17.0,
    }
    assert s113.stage113_decision(m) == "stage113_broad_x_wall_distance_profile_stage114_wall_distance_conditioned_velocity_quadrature_audit"


def test_nonfinite_decision_is_blocker_not_retuning():
    m = {
        "first_4_cumulative_share": 0.35,
        "first_14_cumulative_share": 0.82,
        "half_mass_depth_cells": 7,
        "effective_profile_bin_count": 17.0,
    }
    assert s113.stage113_decision(m, finite=False) == "stage113_nonfinite_wall_distance_profile_blocker_without_retuning"
