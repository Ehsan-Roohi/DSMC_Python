import numpy as np
import pytest

from vgdsmc import stage115_distribution_specific_velocity_sector_audit as s115


def _profile(pair=(5, 6), pair_share=0.60, skew=0.0):
    p = np.full(s115.ANGULAR_SECTORS, (1.0 - pair_share) / (s115.ANGULAR_SECTORS - 2), dtype=float)
    a, b = pair
    p[a] = pair_share / 2.0 + skew
    p[b] = pair_share / 2.0 - skew
    return p


def _block(phi, psi):
    out = s115._profile_metrics(phi, psi)
    out['phi_sector_share'] = np.asarray(phi, dtype=float).tolist()
    out['psi_sector_share'] = np.asarray(psi, dtype=float).tolist()
    out['joint_sector_share'] = (0.5 * (np.asarray(phi) + np.asarray(psi))).tolist()
    return out


def _metrics(near_phi=None, near_psi=None, mid_phi=None, mid_psi=None):
    near_phi = _profile() if near_phi is None else np.asarray(near_phi, dtype=float)
    near_psi = _profile(pair_share=0.56) if near_psi is None else np.asarray(near_psi, dtype=float)
    mid_phi = _profile(pair_share=0.55) if mid_phi is None else np.asarray(mid_phi, dtype=float)
    mid_psi = _profile(pair_share=0.52) if mid_psi is None else np.asarray(mid_psi, dtype=float)
    inner_phi = _profile(pair_share=0.48)
    inner_psi = _profile(pair_share=0.46)
    return {
        'near_1_4': _block(near_phi, near_psi),
        'mid_5_14': _block(mid_phi, mid_psi),
        'inner_15_28': _block(inner_phi, inner_psi),
    }


def test_stage115_design_is_frozen():
    s115.validate_stage115_design()
    with pytest.raises(ValueError):
        s115.validate_stage115_design(profile_cosine_common_min=0.90)
    with pytest.raises(ValueError):
        s115.validate_stage115_design(common_pair_share_min=0.49)
    with pytest.raises(ValueError):
        s115.validate_stage115_design(stage114_run_id=-1)


def test_profile_metrics_identical_profiles_have_unit_similarity():
    p = _profile(pair_share=0.62, skew=0.01)
    m = s115._profile_metrics(p, p)
    assert np.isclose(m['profile_cosine'], 1.0)
    assert np.isclose(m['total_variation_distance'], 0.0)
    assert np.isclose(m['overlap_coefficient'], 1.0)
    assert np.isclose(m['jensen_shannon_bits'], 0.0)
    assert m['phi_psi_top2_sets_match'] is True
    assert m['joint_top2_matches_both'] is True


def test_circular_adjacency_wraps_across_sector_zero():
    assert s115._circular_adjacent([5, 6])
    assert s115._circular_adjacent([7, 0])
    assert not s115._circular_adjacent([1, 3])


def test_band_profiles_are_normalized_for_each_band():
    maps = np.ones((8, 56, 56), dtype=float)
    growth = np.ones((56, 56), dtype=float)
    profiles = s115._band_profiles(maps, growth)
    assert profiles.shape == (3, 8)
    assert np.allclose(profiles, 1.0 / 8.0)
    assert np.allclose(np.sum(profiles, axis=1), 1.0)


def test_stage115_decision_blocks_nonfinite_or_reconstruction_failure():
    metrics = _metrics()
    assert s115.stage115_decision(metrics, False, 0.0) == 'stage115_nonfinite_distribution_profile_blocker_without_retuning'
    assert s115.stage115_decision(metrics, True, 1.1e-12) == 'stage115_stage114_profile_reconstruction_blocker_without_retuning'


def test_stage115_decision_common_adjacent_pair_support():
    metrics = _metrics()
    assert s115.stage115_decision(metrics, True, 1.0e-15) == 'stage115_common_adjacent_pair_support_stage116_pair_resolved_radial_node_audit'


def test_stage115_decision_distribution_specific_divergence():
    near_phi = _profile(pair=(5, 6), pair_share=0.70)
    near_psi = _profile(pair=(1, 2), pair_share=0.70)
    metrics = _metrics(near_phi=near_phi, near_psi=near_psi)
    assert s115.stage115_decision(metrics, True, 1.0e-15) == 'stage115_distribution_specific_angular_divergence_stage116_distribution_contrast_audit'


def test_stage115_decision_partial_common_support_when_pair_share_is_subthreshold():
    near_phi = _profile(pair=(5, 6), pair_share=0.48)
    near_psi = _profile(pair=(5, 6), pair_share=0.48)
    mid_phi = _profile(pair=(5, 6), pair_share=0.48)
    mid_psi = _profile(pair=(5, 6), pair_share=0.48)
    metrics = _metrics(near_phi=near_phi, near_psi=near_psi, mid_phi=mid_phi, mid_psi=mid_psi)
    assert s115.stage115_decision(metrics, True, 1.0e-15) == 'stage115_partial_common_angular_support_stage116_band_specific_pair_audit'


def test_stage115_decision_requires_same_pair_across_broad_wall_bands():
    near_phi = _profile(pair=(5, 6), pair_share=0.62)
    near_psi = _profile(pair=(5, 6), pair_share=0.58)
    mid_phi = _profile(pair=(4, 5), pair_share=0.62)
    mid_psi = _profile(pair=(4, 5), pair_share=0.58)
    metrics = _metrics(near_phi=near_phi, near_psi=near_psi, mid_phi=mid_phi, mid_psi=mid_psi)
    assert s115.stage115_decision(metrics, True, 1.0e-15) == 'stage115_partial_common_angular_support_stage116_band_specific_pair_audit'
