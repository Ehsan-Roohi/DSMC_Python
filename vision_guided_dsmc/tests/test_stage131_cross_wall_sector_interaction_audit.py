import numpy as np
import pytest

from vgdsmc import stage131_cross_wall_sector_interaction_audit as s131


def test_fixed_design_accepts_exact_defaults():
    s131.validate_stage131_design()


def test_fixed_design_rejects_retuned_knudsen_number():
    with pytest.raises(ValueError):
        s131.validate_stage131_design(kn0=0.1)


def test_crossing_alignment_interpolates_second_wall_on_first_relative_depth_grid():
    depth = np.arange(1, 29, dtype=float)
    a = np.column_stack((depth, 2.0 * depth))
    b = np.column_stack((depth + 1.5, 2.0 * (depth + 1.5)))
    z, aa, bb = s131.crossing_aligned_profiles(depth, a, b, 10.0, 11.5)
    assert z.size >= 24
    assert aa.shape == bb.shape == (z.size, 2)
    assert np.isfinite(bb).all()


def test_mirror_mapping_classification_requires_profile_and_transition_coherence():
    result = s131.classify_mapping(
        direct_profile_cosine=0.2,
        mirrored_profile_cosine=0.98,
        mirrored_transition_cosine=0.94,
        mirrored_profile_relative_residual=0.08,
    )
    assert result == s131.MIRROR


def test_direct_mapping_classification_when_direct_is_best_and_coherent():
    result = s131.classify_mapping(
        direct_profile_cosine=0.98,
        mirrored_profile_cosine=0.3,
        mirrored_transition_cosine=0.2,
        mirrored_profile_relative_residual=0.9,
    )
    assert result == s131.DIRECT


def test_complex_mapping_classification_preserves_negative_result():
    result = s131.classify_mapping(
        direct_profile_cosine=0.7,
        mirrored_profile_cosine=0.89,
        mirrored_transition_cosine=0.95,
        mirrored_profile_relative_residual=0.10,
    )
    assert result == s131.COMPLEX


def test_closure_blocker_has_priority():
    result = s131.classify_mapping(
        direct_profile_cosine=0.2,
        mirrored_profile_cosine=0.99,
        mirrored_transition_cosine=0.99,
        mirrored_profile_relative_residual=0.01,
        closure=1.0e-6,
    )
    assert result == s131.CLOSURE_BLOCKER


def test_positive_scale_residual_is_exact_for_scaled_profile():
    reference = np.array([1.0, 2.0, -3.0])
    target = 2.5 * reference
    scale, residual = s131._positive_scale_and_residual(target, reference)
    assert scale == pytest.approx(2.5)
    assert residual < 1.0e-14


def test_sector_exchange_is_involutive():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(x[:, ::-1][:, ::-1], x)
