import numpy as np
import pytest

from vgdsmc import stage130_fixed_sector_continuity_audit as s130


def test_fixed_design_accepts_exact_defaults():
    s130.validate_stage130_design()


def test_fixed_design_rejects_retuned_parameter():
    with pytest.raises(ValueError):
        s130.validate_stage130_design(kn0=9.0)


def test_bracket_depths_are_fixed_integer_neighbors():
    assert s130.bracket_depths(14.0388) == (14, 15)
    assert s130.bracket_depths(15.5527) == (15, 16)


def test_single_sector_carrier_classification():
    deltas = np.array([[-3.0, -1.0], [-4.0, -1.0]])
    pair = deltas.sum(axis=1)
    assert s130.classify_sector_carriage(deltas, pair) == s130.SINGLE


def test_coherent_pair_classification():
    deltas = np.array([[-1.0, -1.0], [-2.0, -2.0]])
    pair = deltas.sum(axis=1)
    assert s130.classify_sector_carriage(deltas, pair) == s130.PAIR


def test_mixed_cross_wall_carriage_classification():
    deltas = np.array([[-1.0, -4.0], [-4.0, 0.05]])
    pair = deltas.sum(axis=1)
    assert s130.classify_sector_carriage(deltas, pair) == s130.MIXED


def test_closure_blocker_has_priority():
    deltas = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    pair = deltas.sum(axis=1)
    assert s130.classify_sector_carriage(deltas, pair, closure=1.0e-6) == s130.CLOSURE_BLOCKER


def test_parent_conditioned_sector_decomposition_is_additive():
    parent = np.zeros((56, 56))
    sectors = np.zeros((56, 56, 2))
    bands = np.zeros((56, 56), dtype=np.int8)
    node_net = np.ones((3, 10))
    for j in range(56):
        sectors[:28, j, 0] = 1.0
        sectors[:28, j, 1] = 2.0
        sectors[28:, j, 0] = 3.0
        sectors[28:, j, 1] = 4.0
    parent[:] = sectors.sum(axis=-1)
    p, c, support = s130.parent_conditioned_sector_profiles(parent, sectors, bands, node_net, 0)
    assert np.all(support > 0.0)
    assert np.max(np.abs(c.sum(axis=1) - p)) < 1.0e-14
