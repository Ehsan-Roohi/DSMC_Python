import numpy as np
import pytest

from vgdsmc.stage154_coarse_endpoint_support_audit import (
    BROAD_SUPPORT,
    NONFINITE,
    NONLOCAL_OPPOSITION,
    PROVENANCE_BLOCKER,
    SINGLE_SAMPLE_SUPPORT,
    STAGE147_RECORD_BLOCKER,
    STAGE153_RECORD_BLOCKER,
    PARENT_ROUTE_BLOCKER,
    classify_coarse_endpoint_support,
    coarse_endpoint_support_metrics,
    validate_stage154_design,
)


DEPTH = np.array(
    [7.961196168207854, 8.961196168207854, 9.961196168207854,
     10.961196168207854, 11.961196168207854]
)
DOMINANT = np.array(
    [-0.06942764887925673, -0.06230659651118298, -0.04816489506330546,
     -0.04655430429054869, -0.035624223271658984]
)
PARENT = np.array(
    [-0.0657307037504089, -0.04500569892676798, -0.03899527101381384,
     -0.02863689878165021, -0.019052011035461724]
)
COARSE = np.array([0.0043610410121524, -0.003396086379121477])


def observed_metrics():
    return coarse_endpoint_support_metrics(
        depth=DEPTH,
        dominant_signed=DOMINANT,
        parent_signed=PARENT,
        stage153_coarse_components=COARSE,
    )


def test_frozen_design_accepts_exact_contract():
    validate_stage154_design()


@pytest.mark.parametrize(
    "key,value",
    [
        ("kn0", 9.0),
        ("rule", (32, 96)),
        ("radial_scale", 1.5),
        ("limiter", "vanleer"),
        ("boundary_slope", "one-sided"),
        ("source_relaxation", 0.5),
        ("correction_floor", 0.01),
        ("support_metrics_used_for_solver", True),
        ("physical_parameter_retuning", True),
        ("velocity_grid_retuning", True),
    ],
)
def test_frozen_design_rejects_retuning(key, value):
    with pytest.raises(ValueError):
        validate_stage154_design(**{key: value})


def test_observed_support_reconstructs_stage153_coarse_endpoint():
    metrics = observed_metrics()
    np.testing.assert_allclose(
        metrics["reconstructed_coarse_components"], COARSE, rtol=0.0, atol=1e-14
    )
    assert metrics["maximum_identity_or_provenance_error"] <= 1e-14


def test_observed_support_is_pointwise_opposed_at_all_three_coarse_samples():
    metrics = observed_metrics()
    assert metrics["channel_sign_products"] == [-1, -1, -1]
    assert metrics["opposed_support_count"] == 3


def test_observed_cancellation_support_is_broad_not_single_sample():
    metrics = observed_metrics()
    assert metrics["maximum_single_sample_cancellation_support_share"] == pytest.approx(
        0.47913609123756346
    )
    assert metrics["maximum_support_sample_index"] == 2
    assert metrics["effective_cancellation_support_count"] == pytest.approx(
        2.460993991049381
    )
    assert sum(metrics["node_cancellation_support_share"]) == pytest.approx(1.0)


def test_observed_route_is_broad_support():
    metrics = observed_metrics()
    assert classify_coarse_endpoint_support(metrics=metrics) == BROAD_SUPPORT


def test_single_sample_route_uses_fixed_75_percent_guard():
    metrics = observed_metrics()
    metrics = dict(metrics)
    metrics["maximum_single_sample_cancellation_support_share"] = 0.80
    assert classify_coarse_endpoint_support(metrics=metrics) == SINGLE_SAMPLE_SUPPORT


def test_nonlocal_opposition_route_when_only_one_support_node_is_opposed():
    metrics = observed_metrics()
    metrics = dict(metrics)
    metrics["opposed_support_count"] = 1
    metrics["maximum_single_sample_cancellation_support_share"] = 0.60
    assert classify_coarse_endpoint_support(metrics=metrics) == NONLOCAL_OPPOSITION


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"finite": False}, NONFINITE),
        ({"stage153_record_ok": False}, STAGE153_RECORD_BLOCKER),
        ({"stage147_record_ok": False}, STAGE147_RECORD_BLOCKER),
        ({"parent_route_ok": False}, PARENT_ROUTE_BLOCKER),
    ],
)
def test_blocker_routes(kwargs, expected):
    assert classify_coarse_endpoint_support(metrics=observed_metrics(), **kwargs) == expected


def test_provenance_error_blocks_routing():
    metrics = observed_metrics()
    metrics = dict(metrics)
    metrics["maximum_identity_or_provenance_error"] = 1e-8
    assert classify_coarse_endpoint_support(metrics=metrics) == PROVENANCE_BLOCKER


@pytest.mark.parametrize(
    "bad_shape",
    [
        np.zeros(4),
        np.zeros(6),
    ],
)
def test_requires_five_point_profiles(bad_shape):
    with pytest.raises(ValueError):
        coarse_endpoint_support_metrics(
            depth=bad_shape,
            dominant_signed=DOMINANT,
            parent_signed=PARENT,
            stage153_coarse_components=COARSE,
        )
