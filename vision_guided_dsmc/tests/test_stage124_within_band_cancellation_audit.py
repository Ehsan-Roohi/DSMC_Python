import numpy as np
import pytest

from vgdsmc import stage124_within_band_cancellation_audit as s124


def test_stage124_design_is_frozen():
    s124.validate_stage124_design()
    with pytest.raises(ValueError):
        s124.validate_stage124_design(strong_cancellation_fraction_min=0.5)
    with pytest.raises(ValueError):
        s124.validate_stage124_design(stage123_run_id=1)


def test_amplitude_matched_residual_exact_template_closes():
    psi = np.vstack([np.arange(1.0, 11.0), np.arange(10.0, 0.0, -1.0)])
    template = np.linspace(0.5, 2.0, 10)
    phi = np.vstack([2.0 * psi[0] * template, 0.4 * psi[1] * template])
    residual, scale, mass = s124.amplitude_matched_residual(phi, psi, template)
    assert np.allclose(residual, 0.0, atol=1.0e-14)
    assert np.allclose(scale, [2.0, 0.4], rtol=1.0e-14, atol=1.0e-14)
    assert mass < 1.0e-14


def test_cancellation_metric_detects_exact_opposite_cells():
    r0 = np.array([1.0, -1.0] + [0.0] * 8)
    residual = np.vstack([r0, -r0])
    m = s124.cancellation_metrics(residual, np.array([True, False]))
    assert m["cancellation_fraction"] == pytest.approx(1.0)
    assert m["uncancelled_fraction"] == pytest.approx(0.0)


def test_cancellation_metric_detects_no_spatial_cancellation():
    r0 = np.array([1.0, -1.0] + [0.0] * 8)
    residual = np.vstack([r0, r0])
    m = s124.cancellation_metrics(residual, np.array([True, True]))
    assert m["cancellation_fraction"] == pytest.approx(0.0)
    assert m["maximum_node_uncancelled_fraction"] == pytest.approx(1.0)


def test_stage124_decision_routes_are_preregistered():
    common = dict(finite=True, parent_closure=0.0, cell_mass_closure=0.0)
    assert s124.stage124_decision(
        cancellation_fractions=[0.7, 0.6, 0.8],
        maximum_node_uncancelled_fraction=0.9,
        **common,
    ) == s124.STRONG_WITH_NODE_REMAINDER
    assert s124.stage124_decision(
        cancellation_fractions=[0.7, 0.6, 0.8],
        maximum_node_uncancelled_fraction=0.7,
        **common,
    ) == s124.STRONG_BROAD
    assert s124.stage124_decision(
        cancellation_fractions=[0.7, 0.59, 0.8],
        maximum_node_uncancelled_fraction=0.1,
        **common,
    ) == s124.WEAK


def test_stage124_blockers_precede_structural_routes():
    assert s124.stage124_decision(
        finite=False,
        parent_closure=0.0,
        cell_mass_closure=0.0,
        cancellation_fractions=[1.0, 1.0, 1.0],
        maximum_node_uncancelled_fraction=0.0,
    ) == s124.NONFINITE
    assert s124.stage124_decision(
        finite=True,
        parent_closure=10.0 * s124.PARENT_PROFILE_CLOSURE_TOLERANCE,
        cell_mass_closure=0.0,
        cancellation_fractions=[1.0, 1.0, 1.0],
        maximum_node_uncancelled_fraction=0.0,
    ) == s124.CLOSURE_BLOCKER
    assert s124.stage124_decision(
        finite=True,
        parent_closure=0.0,
        cell_mass_closure=10.0 * s124.CELL_MASS_CLOSURE_TOLERANCE,
        cancellation_fractions=[1.0, 1.0, 1.0],
        maximum_node_uncancelled_fraction=0.0,
    ) == s124.CLOSURE_BLOCKER
