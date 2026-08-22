import numpy as np
import pytest

from vgdsmc.stage157_fixed_ratio_curvature_audit import (
    MIXED_CURVATURE,
    PARENT_ROUTE_BLOCKER,
    STAGE156_RECORD_BLOCKER,
    classify_fixed_ratio_curvature,
    fixed_ratio_curvature_metrics,
    validate_stage157_design,
)


def _metrics():
    return fixed_ratio_curvature_metrics(
        support_depth=np.array([
            7.961196168207854,
            9.961196168207854,
            11.961196168207854,
        ]),
        dominant_raw_support=np.array([
            -0.06942764887925673,
            -0.04816489506330546,
            -0.035624223271658984,
        ]),
        parent_raw_support=np.array([
            -0.0657307037504089,
            -0.03899527101381384,
            -0.019052011035461724,
        ]),
        inherited_ratio=np.array([
            0.9467511115740463,
            0.8096201800618575,
            0.5348049525228145,
        ]),
    )


def test_stage157_frozen_design_rejects_retuning():
    validate_stage157_design()
    with pytest.raises(ValueError):
        validate_stage157_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage157_design(physical_parameter_retuning=True)
    with pytest.raises(ValueError):
        validate_stage157_design(single_channel_share_min=0.75)


def test_stage157_interval_decay_rates_and_identity_are_exact():
    m = _metrics()
    assert np.allclose(m["support_spacing"], [2.0, 2.0])
    assert np.allclose(
        m["dominant_decay_rate_per_cell"],
        [0.18282737474711785, 0.1508023010325017],
    )
    assert np.allclose(
        m["parent_decay_rate_per_cell"],
        [0.2610628830908732, 0.35813386027725835],
    )
    assert np.allclose(
        m["ratio_decay_rate_per_cell"],
        [0.07823550834375517, 0.20733155924475682],
    )
    assert m["maximum_identity_or_provenance_error"] <= 1.0e-12


def test_stage157_ratio_curvature_is_material_but_mixed_channel():
    m = _metrics()
    assert np.isclose(m["ratio_rate_acceleration_factor"], 2.6500953803964924)
    assert np.isclose(m["parent_curvature_share"], 0.7519283239796761)
    assert np.isclose(m["dominant_curvature_share"], 0.24807167602032107)
    assert m["ratio_rate_acceleration_factor"] >= 1.5
    assert m["parent_curvature_share"] < 0.80
    assert m["dominant_curvature_share"] < 0.80
    assert classify_fixed_ratio_curvature(metrics=m) == MIXED_CURVATURE


def test_stage157_curvature_decomposition_matches_log_second_difference():
    m = _metrics()
    assert np.isclose(m["log_second_difference_dominant"], 0.06405014742923232)
    assert np.isclose(m["log_second_difference_parent"], -0.19414195437277026)
    assert np.isclose(m["log_second_difference_ratio"], -0.2581921018020033)
    assert m["second_difference_identity_error"] <= 1.0e-12


def test_stage157_blocks_bad_parent_provenance_or_route():
    m = _metrics()
    assert (
        classify_fixed_ratio_curvature(metrics=m, stage156_record_ok=False)
        == STAGE156_RECORD_BLOCKER
    )
    assert (
        classify_fixed_ratio_curvature(metrics=m, parent_route_ok=False)
        == PARENT_ROUTE_BLOCKER
    )
