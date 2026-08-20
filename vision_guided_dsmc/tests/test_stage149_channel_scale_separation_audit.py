import numpy as np
import pytest

from vgdsmc.stage149_channel_scale_separation_audit import (
    COARSE_CANCELLATION,
    channel_scale_separation_metrics,
    classify_channel_scale_separation,
    validate_stage149_design,
)


def _metrics():
    return channel_scale_separation_metrics(
        np.array([-0.0035103245399018823, 0.0062655553375603745, -0.004659745123066467]),
        np.array([-0.0073572884553433915, 0.0021739721596047423, -0.00038674224298757]),
        np.array([-0.010867612995245274, 0.008439527497165117, -0.005046487366054037]),
        np.array([0.0043610410121524, -0.003396086379121477, 0.0009649546330309233]),
        np.array([0.6960342343493631, 1.562157254000406, 0.1143375187005501]),
    )


def test_stage149_frozen_design_accepts_only_registered_values():
    validate_stage149_design()
    with pytest.raises(ValueError):
        validate_stage149_design(kn0=9.0)
    with pytest.raises(ValueError):
        validate_stage149_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage149_design(channel_scale_metrics_used_for_solver=True)


def test_stage149_reproduces_stage148_retention_and_identity():
    m = _metrics()
    assert m["retention_match_error"] < 1e-15
    assert m["fine_identity_closure"] < 1e-15
    assert m["coarse_identity_closure"] < 1e-15
    assert np.allclose(m["recomputed_retention"], [0.6960342343493631, 1.562157254000406, 0.1143375187005501])


def test_stage149_detects_coarse_cross_channel_cancellation():
    m = _metrics()
    assert m["fine_channel_sign_product"] == 1
    assert m["coarse_channel_sign_product"] == -1
    assert m["fine_cancellation_fraction"] == pytest.approx(0.0, abs=1e-15)
    assert m["coarse_cancellation_fraction"] == pytest.approx(0.8756041271003984)
    assert m["minimum_component_retention"] == pytest.approx(0.6960342343493631)
    assert m["complement_retention"] == pytest.approx(0.1143375187005501)
    assert m["minimum_component_to_complement_retention_ratio"] == pytest.approx(6.087540137828916)
    assert classify_channel_scale_separation(metrics=m) == COARSE_CANCELLATION
