from __future__ import annotations

import numpy as np
import pytest

from vgdsmc.stage148_sample_scale_curvature_alternation_audit import (
    ALTERNATING_ENERGY_SHARE_MIN,
    COARSE_CENTER_RETENTION_MAX,
    IDENTITY_BLOCKER,
    MULTISCALE,
    PROVENANCE_BLOCKER,
    SAMPLE_SCALE,
    WEAK_ALTERNATION,
    classify_sample_scale_curvature,
    sample_scale_curvature_metrics,
    validate_stage148_design,
)


def _actual_stage147_arrays():
    depth = np.array([7.961196168207854, 8.961196168207854, 9.961196168207854, 10.961196168207854, 11.961196168207854])
    dominant = np.array([-0.06942764887925673, -0.06230659651118298, -0.04816489506330546, -0.04655430429054869, -0.035624223271658984])
    parent = np.array([-0.0657307037504089, -0.04500569892676798, -0.03899527101381384, -0.02863689878165021, -0.019052011035461724])
    complement = parent - dominant
    inherited_d = np.array([-0.0035103245399018823, 0.0062655553375603745, -0.004659745123066467])
    inherited_p = np.array([-0.0073572884553433915, 0.0021739721596047423, -0.00038674224298757])
    inherited_c = np.array([-0.010867612995245274, 0.008439527497165117, -0.005046487366054037])
    return depth, dominant, parent, complement, inherited_d, inherited_p, inherited_c


def _metrics():
    return sample_scale_curvature_metrics(*_actual_stage147_arrays())


def test_stage148_frozen_design_accepts_defaults() -> None:
    validate_stage148_design()


def test_stage148_rejects_retuning_or_threshold_change() -> None:
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage148_design(kn0=9.0)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage148_design(alternating_energy_share_min=0.70)
    with pytest.raises(ValueError, match="frozen-design violation"):
        validate_stage148_design(curvature_scale_used_for_solver=True)


def test_stage148_reproduces_parent_curvatures_exactly() -> None:
    m = _metrics()
    assert m["inherited_curvature_match_error"] < 1e-14
    assert m["fine_curvature_sign_sequence"] == [-1, 1, -1]
    assert m["maximum_identity_or_provenance_error"] < 1e-14


def test_stage148_actual_complement_is_strongly_alternating() -> None:
    m = _metrics()
    alt = m["complement_alternating_mode"]
    assert alt["energy_share"] == pytest.approx(0.9203997824377443)
    assert alt["energy_share"] > ALTERNATING_ENERGY_SHARE_MIN
    assert alt["sign_agreement_fraction"] == 1.0
    assert alt["relative_l2_residual"] == pytest.approx(0.2821351051575392)


def test_stage148_actual_two_cell_total_curvature_collapses() -> None:
    m = _metrics()
    assert m["coarse_center_complement_secant_deficit"] == pytest.approx(0.0009649546330309233)
    assert m["complement_coarse_to_fine_center_retention"] == pytest.approx(0.1143375187005501)
    assert m["complement_coarse_to_fine_center_retention"] < COARSE_CENTER_RETENTION_MAX


def test_stage148_channel_specific_coarse_behavior_is_not_hidden() -> None:
    m = _metrics()
    assert m["dominant_coarse_to_fine_center_retention"] == pytest.approx(0.6960342343493631)
    assert m["parent_coarse_to_fine_center_retention"] == pytest.approx(1.562157254000406)
    assert m["coarse_center_dominant_projected_curvature"] > 0.0
    assert m["coarse_center_parent_projected_curvature"] < 0.0


def test_stage148_classifies_actual_sample_scale_branch() -> None:
    assert classify_sample_scale_curvature(metrics=_metrics()) == SAMPLE_SCALE


def test_stage148_classifies_multiscale_when_coarse_retention_is_large() -> None:
    m = _metrics()
    m["complement_coarse_to_fine_center_retention"] = 0.8
    assert classify_sample_scale_curvature(metrics=m) == MULTISCALE


def test_stage148_classifies_weak_alternation_when_energy_share_is_low() -> None:
    m = _metrics()
    m["complement_alternating_mode"] = dict(m["complement_alternating_mode"])
    m["complement_alternating_mode"]["energy_share"] = 0.5
    assert classify_sample_scale_curvature(metrics=m) == WEAK_ALTERNATION


def test_stage148_blocks_parent_provenance_error() -> None:
    m = _metrics()
    m["inherited_curvature_match_error"] = 1e-6
    m["maximum_identity_or_provenance_error"] = 1e-6
    assert classify_sample_scale_curvature(metrics=m) == PROVENANCE_BLOCKER


def test_stage148_blocks_identity_error_after_provenance_passes() -> None:
    m = _metrics()
    m["maximum_identity_or_provenance_error"] = 1e-6
    assert classify_sample_scale_curvature(metrics=m) == IDENTITY_BLOCKER


def test_stage148_rejects_nonequal_depth_sampling() -> None:
    arrays = list(_actual_stage147_arrays())
    arrays[0] = arrays[0].copy()
    arrays[0][3] += 0.1
    with pytest.raises(ValueError, match="equal-depth sampling"):
        sample_scale_curvature_metrics(*arrays)
