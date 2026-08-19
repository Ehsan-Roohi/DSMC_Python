import numpy as np
import pytest

from vgdsmc import stage139_complement_transition_geometry_audit as s139


def test_stage139_design_is_frozen():
    s139.validate_stage139_design()
    with pytest.raises(ValueError):
        s139.validate_stage139_design(kn0=9.0)
    with pytest.raises(ValueError):
        s139.validate_stage139_design(physical_parameter_retuning=True)
    with pytest.raises(ValueError):
        s139.validate_stage139_design(cross_knudsen_extension_permitted=True)
    with pytest.raises(ValueError):
        s139.validate_stage139_design(well_centered_edge_clearance_min_fraction=0.20)


def test_sign_change_bracket_and_linear_crossing():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-2.0, -1.0, 3.0])
    assert s139.sign_change_brackets(y) == [1]
    assert s139.interpolated_crossing(x, y, 1) == pytest.approx(1.25)


def test_piecewise_linear_signed_areas_split_at_crossing():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-1.0, 1.0, 1.0])
    negative, positive = s139.signed_piecewise_linear_abs_areas(x, y)
    assert negative == pytest.approx(0.25)
    assert positive == pytest.approx(1.25)


def test_stage139_classifies_node_proximate_crossing():
    assert s139.classify_transition_geometry(
        sign_change_count=1,
        left_sign_coherence=1.0,
        right_sign_coherence=1.0,
        bracket_width_cells=1.0,
        edge_clearance_fraction=0.13,
        parent_identity_closure=0.0,
    ) == s139.NODE_PROXIMATE_CROSSING


def test_stage139_classifies_well_localized_crossing():
    assert s139.classify_transition_geometry(
        sign_change_count=1,
        left_sign_coherence=1.0,
        right_sign_coherence=1.0,
        bracket_width_cells=1.0,
        edge_clearance_fraction=0.30,
        parent_identity_closure=0.0,
    ) == s139.WELL_LOCALIZED_CROSSING


def test_stage139_routes_underresolved_or_nonunique_geometry():
    assert s139.classify_transition_geometry(
        sign_change_count=1,
        left_sign_coherence=1.0,
        right_sign_coherence=1.0,
        bracket_width_cells=1.2,
        edge_clearance_fraction=0.30,
        parent_identity_closure=0.0,
    ) == s139.UNDERRESOLVED_CROSSING
    assert s139.classify_transition_geometry(
        sign_change_count=2,
        left_sign_coherence=1.0,
        right_sign_coherence=1.0,
        bracket_width_cells=1.0,
        edge_clearance_fraction=0.30,
        parent_identity_closure=0.0,
    ) == s139.UNRESOLVED_GEOMETRY


def test_stage139_provenance_and_closure_blockers_are_hard():
    base = dict(
        sign_change_count=1,
        left_sign_coherence=1.0,
        right_sign_coherence=1.0,
        bracket_width_cells=1.0,
        edge_clearance_fraction=0.30,
        parent_identity_closure=0.0,
    )
    assert s139.classify_transition_geometry(**base, parent_record_ok=False) == s139.PARENT_RECORD_BLOCKER
    broken = dict(base)
    broken["parent_identity_closure"] = 2.0e-12
    assert s139.classify_transition_geometry(**broken) == s139.PARENT_IDENTITY_BLOCKER
