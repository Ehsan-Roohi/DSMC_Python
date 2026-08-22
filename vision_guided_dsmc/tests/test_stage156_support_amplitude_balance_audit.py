import numpy as np
import pytest

from vgdsmc.stage156_support_amplitude_balance_audit import (
    CURVED_TREND,
    PARENT_ROUTE_BLOCKER,
    STAGE155_RECORD_BLOCKER,
    classify_support_amplitude_balance,
    support_amplitude_balance_metrics,
    validate_stage156_design,
)


def _metrics():
    return support_amplitude_balance_metrics(
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
        observed_cancellation_fraction=np.array([
            0.972647305498185,
            0.8947957024155009,
            0.6969028235721239,
        ]),
    )


def test_stage156_frozen_design_rejects_retuning():
    validate_stage156_design()
    with pytest.raises(ValueError):
        validate_stage156_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage156_design(physical_parameter_retuning=True)


def test_stage156_inherited_amplitude_ratio_is_material_and_monotone():
    m = _metrics()
    ratio = np.asarray(m["amplitude_ratio_parent_to_dominant"])
    assert np.allclose(
        ratio,
        [0.9467511115740463, 0.8096201800618575, 0.5348049525228145],
    )
    assert m["monotone_decrease"] is True
    assert np.isclose(m["endpoint_decline_fraction"], 0.43511558002434214)
    assert m["endpoint_decline_fraction"] >= 0.25
    assert m["cancellation_fraction_reconstruction_error"] <= 1.0e-15


def test_stage156_midpoint_rejects_preregistered_near_loglinear_guard():
    m = _metrics()
    assert np.isclose(m["midpoint_log_residual_abs"], 0.12909605090100162)
    assert m["midpoint_log_residual_abs"] > 0.10
    assert np.isclose(m["second_to_first_drop_ratio"], 2.0040353004866467)
    assert m["log_linear_r2"] > 0.90


def test_stage156_routes_to_fixed_ratio_curvature_audit():
    m = _metrics()
    assert classify_support_amplitude_balance(metrics=m) == CURVED_TREND
    assert (
        classify_support_amplitude_balance(metrics=m, stage155_record_ok=False)
        == STAGE155_RECORD_BLOCKER
    )
    assert (
        classify_support_amplitude_balance(metrics=m, parent_route_ok=False)
        == PARENT_ROUTE_BLOCKER
    )
