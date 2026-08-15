import numpy as np
import pytest

from vgdsmc import stage122_common_kernel_distribution_ratio_audit as s122


def test_stage122_design_is_frozen():
    s122.validate_stage122_design()
    with pytest.raises(ValueError):
        s122.validate_stage122_design(kn0=9.0)
    with pytest.raises(ValueError):
        s122.validate_stage122_design(limiter="none")
    with pytest.raises(ValueError):
        s122.validate_stage122_design(max_node_ratio_relative_range=0.25)


def test_centered_log_ratio_preserves_positive_ratio_shape():
    phi = np.arange(1.0, 11.0)
    psi = np.arange(10.0, 0.0, -1.0)
    ratio, centered = s122.centered_log_ratio(phi, psi)
    assert ratio.shape == (10,)
    assert centered.shape == (10,)
    assert np.all(ratio > 0.0)
    assert abs(float(centered.mean())) < 1.0e-14


def test_leave_one_out_template_excludes_held_out_band():
    ratios = np.vstack([
        np.ones(10),
        np.full(10, 4.0),
        np.full(10, 9.0),
    ])
    template = s122.leave_one_out_template(ratios, 2)
    assert np.allclose(template, 2.0)


def test_stage122_stable_decision_requires_all_preregistered_guards():
    decision = s122.stage122_decision(
        finite=True,
        parent_closure=0.0,
        min_centered_cosine=0.995,
        max_node_relative_range=0.10,
        identical_crossing=True,
        min_loo_cosine=0.998,
        max_loo_tv=0.02,
    )
    assert decision == s122.STABLE
    decision = s122.stage122_decision(
        finite=True,
        parent_closure=0.0,
        min_centered_cosine=0.98,
        max_node_relative_range=0.10,
        identical_crossing=True,
        min_loo_cosine=0.998,
        max_loo_tv=0.02,
    )
    assert decision == s122.UNSTABLE


def test_stage122_blockers_precede_structural_decision():
    common = dict(
        parent_closure=0.0,
        min_centered_cosine=1.0,
        max_node_relative_range=0.0,
        identical_crossing=True,
        min_loo_cosine=1.0,
        max_loo_tv=0.0,
    )
    assert s122.stage122_decision(finite=False, **common) == s122.NONFINITE
    assert s122.stage122_decision(
        finite=True,
        parent_closure=10.0 * s122.PARENT_CLOSURE_TOLERANCE,
        min_centered_cosine=1.0,
        max_node_relative_range=0.0,
        identical_crossing=True,
        min_loo_cosine=1.0,
        max_loo_tv=0.0,
    ) == s122.CLOSURE_BLOCKER
