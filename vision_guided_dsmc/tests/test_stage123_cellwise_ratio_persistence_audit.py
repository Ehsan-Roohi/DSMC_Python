import numpy as np
import pytest

from vgdsmc import stage123_cellwise_ratio_persistence_audit as s123


def test_stage123_design_is_frozen():
    s123.validate_stage123_design()
    with pytest.raises(ValueError):
        s123.validate_stage123_design(kn0=9.0)
    with pytest.raises(ValueError):
        s123.validate_stage123_design(limiter="none")
    with pytest.raises(ValueError):
        s123.validate_stage123_design(min_cell_pass_fraction=0.70)


def test_cell_prediction_metrics_recovers_exact_fixed_template():
    psi = np.vstack([
        np.arange(1.0, 11.0),
        np.arange(10.0, 0.0, -1.0),
    ])
    template = np.linspace(0.5, 2.0, 10)
    phi = psi * template[None, :]
    cosine, overlap, tv, valid = s123.cell_prediction_metrics(phi, psi, template)
    assert valid.tolist() == [True, True]
    assert np.allclose(cosine, 1.0, atol=1.0e-14)
    assert np.allclose(overlap, 1.0, atol=1.0e-14)
    assert np.allclose(tv, 0.0, atol=1.0e-14)


def test_cell_prediction_metrics_marks_zero_profile_invalid_without_floor_fitting():
    psi = np.ones((2, 10))
    phi = np.ones((2, 10))
    phi[1] = 0.0
    cosine, overlap, tv, valid = s123.cell_prediction_metrics(phi, psi, np.ones(10))
    assert bool(valid[0])
    assert not bool(valid[1])
    assert np.isfinite(cosine[0])
    assert np.isnan(cosine[1])
    assert np.isnan(overlap[1])
    assert np.isnan(tv[1])


def test_stage123_persistent_decision_requires_all_fixed_band_guards():
    decision = s123.stage123_decision(
        finite=True,
        parent_closure=0.0,
        valid_fractions=[0.99, 0.98, 1.0],
        pass_fractions=[0.80, 0.76, 0.90],
    )
    assert decision == s123.PERSISTENT
    decision = s123.stage123_decision(
        finite=True,
        parent_closure=0.0,
        valid_fractions=[0.99, 0.98, 1.0],
        pass_fractions=[0.80, 0.74, 0.90],
    )
    assert decision == s123.AGGREGATION_ONLY


def test_stage123_valid_fraction_is_not_silently_ignored():
    decision = s123.stage123_decision(
        finite=True,
        parent_closure=0.0,
        valid_fractions=[0.99, 0.94, 1.0],
        pass_fractions=[1.0, 1.0, 1.0],
    )
    assert decision == s123.AGGREGATION_ONLY


def test_stage123_blockers_precede_structural_decision():
    common = dict(valid_fractions=[1.0, 1.0, 1.0], pass_fractions=[1.0, 1.0, 1.0])
    assert s123.stage123_decision(finite=False, parent_closure=0.0, **common) == s123.NONFINITE
    assert s123.stage123_decision(
        finite=True,
        parent_closure=10.0 * s123.PARENT_CLOSURE_TOLERANCE,
        **common,
    ) == s123.CLOSURE_BLOCKER
