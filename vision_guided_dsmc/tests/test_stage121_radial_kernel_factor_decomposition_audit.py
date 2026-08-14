import numpy as np
import pytest

from vgdsmc import stage121_radial_kernel_factor_decomposition_audit as s121


def test_stage121_design_is_frozen():
    s121.validate_stage121_design()
    with pytest.raises(ValueError):
        s121.validate_stage121_design(kernel_material_tv_reduction_min=0.1)
    with pytest.raises(ValueError):
        s121.validate_stage121_design(pair_sectors=(4, 5))


def test_stage121_profile_metrics_identity():
    p = np.arange(1.0, 11.0)
    m = s121._profile_metrics(p, p)
    assert m["profile_cosine"] == pytest.approx(1.0)
    assert m["overlap_coefficient"] == pytest.approx(1.0)
    assert m["total_variation_distance"] == pytest.approx(0.0)


def test_stage121_exact_r2_counterfactual_closes_for_common_base():
    base = np.arange(1.0, 11.0)
    r2 = np.linspace(0.2, 2.0, 10)
    psi = s121._normalize10(base)
    phi = s121._normalize10(base * r2)
    pred = s121._normalize10(psi * r2)
    assert np.allclose(phi, pred, rtol=0, atol=1e-15)
    assert s121._profile_metrics(phi, pred)["total_variation_distance"] == pytest.approx(0.0, abs=1e-15)


def test_stage121_decision_routes_dominant_material_and_weak():
    def mm(r, o):
        return {
            band: {
                "kernel_counterfactual_tv_reduction_fraction": r,
                "kernel_counterfactual_overlap": o,
            }
            for band in s121.BANDS
        }

    assert s121.stage121_decision(mm(0.8, 0.97), True, 0.0) == s121.DOMINANT
    assert s121.stage121_decision(mm(0.3, 0.90), True, 0.0) == s121.MATERIAL
    assert s121.stage121_decision(mm(0.1, 0.99), True, 0.0) == s121.WEAK


def test_stage121_blockers_are_explicit():
    metrics = {
        band: {
            "kernel_counterfactual_tv_reduction_fraction": 0.3,
            "kernel_counterfactual_overlap": 0.9,
        }
        for band in s121.BANDS
    }
    assert s121.stage121_decision(metrics, False, 0.0) == s121.NONFINITE
    assert s121.stage121_decision(metrics, True, 1e-8) == s121.CLOSURE_BLOCKER
