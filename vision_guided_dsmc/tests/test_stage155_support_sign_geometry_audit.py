import numpy as np
import pytest

from vgdsmc.stage155_support_sign_geometry_audit import (
    COEFFICIENT_IMPOSED,
    PARENT_ROUTE_BLOCKER,
    STAGE154_RECORD_BLOCKER,
    classify_support_sign_geometry,
    support_sign_geometry_metrics,
    validate_stage155_design,
)


def _metrics():
    return support_sign_geometry_metrics(
        support_depth=np.array([7.961196168207854, 9.961196168207854, 11.961196168207854]),
        dominant_support_contributions=np.array([
            0.034713824439628366,
            -0.04816489506330546,
            0.017812111635829492,
        ]),
        parent_support_contributions=np.array([
            -0.03286535187520445,
            0.03899527101381384,
            -0.009526005517730862,
        ]),
        stage154_cancellation_fraction=np.array([
            0.972647305498185,
            0.8947957024155009,
            0.6969028235721239,
        ]),
    )


def test_stage155_frozen_design_rejects_retuning():
    validate_stage155_design()
    with pytest.raises(ValueError):
        validate_stage155_design(limiter="vanleer")
    with pytest.raises(ValueError):
        validate_stage155_design(physical_parameter_retuning=True)


def test_stage155_recovers_same_sign_raw_profiles_and_coefficient_opposition():
    m = _metrics()
    assert m["raw_cross_channel_sign_products"] == [1, 1, 1]
    assert m["coefficient_cross_channel_sign_products"] == [-1, -1, -1]
    assert m["contribution_cross_channel_sign_products"] == [-1, -1, -1]
    assert m["raw_same_sign_fraction"] == 1.0
    assert m["coefficient_explained_opposition_fraction"] == 1.0
    assert m["dominant_raw_sign_reversal_count"] == 0
    assert m["parent_raw_sign_reversal_count"] == 0
    assert m["dominant_contribution_sign_reversal_count"] == 2
    assert m["parent_contribution_sign_reversal_count"] == 2
    assert m["cancellation_fraction_reconstruction_error"] <= 1.0e-15


def test_stage155_amplitude_balance_is_nontrivial_but_not_a_sign_mode():
    m = _metrics()
    ratios = np.asarray(m["amplitude_ratio_parent_to_dominant"])
    assert np.allclose(ratios, [0.9467511115740463, 0.8096201800618575, 0.5348049525228145])
    assert m["raw_magnitude_cosine"] > 0.95
    assert ratios[0] > ratios[1] > ratios[2]


def test_stage155_routes_to_fixed_amplitude_balance_audit():
    m = _metrics()
    assert classify_support_sign_geometry(metrics=m) == COEFFICIENT_IMPOSED
    assert classify_support_sign_geometry(metrics=m, stage154_record_ok=False) == STAGE154_RECORD_BLOCKER
    assert classify_support_sign_geometry(metrics=m, parent_route_ok=False) == PARENT_ROUTE_BLOCKER
