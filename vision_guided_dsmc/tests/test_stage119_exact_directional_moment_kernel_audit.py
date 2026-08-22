import numpy as np
import pytest

from vgdsmc import stage119_exact_directional_moment_kernel_audit as s119


def test_stage119_design_is_frozen():
    s119.validate_stage119_design()
    with pytest.raises(ValueError):
        s119.validate_stage119_design(common_cosine_min=0.94)
    with pytest.raises(ValueError):
        s119.validate_stage119_design(pair_sectors=(4, 5))


def test_stage119_radial_node_grouping_is_exact():
    radii = np.repeat(np.arange(1.0, 11.0), 96)
    theta = np.tile(np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False), 10)
    vx = radii * np.cos(theta)
    vy = radii * np.sin(theta)
    labels = s119.radial_node_indices_within_shell(vx, vy)
    assert labels.shape == (960,)
    assert [int(np.count_nonzero(labels == j)) for j in range(10)] == [96] * 10


def test_stage119_profile_metrics_identity():
    p = np.arange(1.0, 11.0)
    m = s119._profile_metrics(p, p)
    assert m["profile_cosine"] == pytest.approx(1.0)
    assert m["overlap_coefficient"] == pytest.approx(1.0)
    assert m["total_variation_distance"] == pytest.approx(0.0)
    assert m["transition_boundaries"] == []


def test_stage119_decision_guards():
    aligned = {
        band: {"profile_cosine": 0.96, "overlap_coefficient": 0.91}
        for band in s119.BANDS
    }
    assert s119.stage119_decision(aligned, True, 1.0e-16) == s119.ALIGNED
    incomplete = {band: dict(v) for band, v in aligned.items()}
    incomplete["inner_15_28"]["profile_cosine"] = 0.94
    assert s119.stage119_decision(incomplete, True, 1.0e-16) == s119.INCOMPLETE
    assert s119.stage119_decision(aligned, True, 1.0e-8) == s119.CLOSURE_BLOCKER
    assert s119.stage119_decision(aligned, False, 0.0) == s119.NONFINITE
