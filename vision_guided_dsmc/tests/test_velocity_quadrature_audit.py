import json
import math
import numpy as np

from vgdsmc.velocity_quadrature_audit import (
    audit_condition,
    audit_tangential_response,
    build_stage29_quadratures,
    cartesian_midpoint,
    maxwellian,
    run_stage29,
    spherical_product,
    tensor_gauss_hermite,
)


def test_cartesian_midpoint_has_zero_and_exact_count():
    quadrature = cartesian_midpoint(17, 5.0)
    assert quadrature.point_count == 17**3
    assert np.any(quadrature.vx == 0.0)
    assert np.all(quadrature.weight > 0.0)


def test_gauss_hermite_is_reduced_and_positive():
    quadrature = tensor_gauss_hermite(12, 1.0)
    assert quadrature.point_count == 12**3
    assert quadrature.point_count < 17**3
    assert np.all(np.isfinite(quadrature.weight))
    assert np.all(quadrature.weight > 0.0)


def test_spherical_product_integrates_velocity_ball_volume():
    radius = 5.0
    quadrature = spherical_product(12, 10, 16, radius, "fixture")
    expected = 4.0 * math.pi * radius**3 / 3.0
    assert np.isclose(np.sum(quadrature.weight), expected, rtol=1.0e-12, atol=1.0e-12)
    assert np.all(quadrature.weight > 0.0)


def test_maxwellian_is_finite_and_positive():
    quadrature = spherical_product(8, 8, 12, 5.0, "fixture")
    values = maxwellian(quadrature, 0.1, 0.005)
    assert values.shape == quadrature.weight.shape
    assert np.isfinite(values).all()
    assert np.all(values > 0.0)


def test_spherical_rule_preserves_tangential_flux_sign():
    quadrature = spherical_product(16, 12, 24, 5.0, "fixture")
    for shift in (-0.005, 0.005):
        row = audit_tangential_response(quadrature, 1.0, shift)
        assert row["sign_correct"] is True
        assert row["relative_error"] < 0.05


def test_spherical_rule_resolves_reference_temperature_half_fluxes():
    quadrature = spherical_product(16, 12, 24, 5.0, "fixture")
    row = audit_condition(quadrature, 1.0)
    assert row["half_mass_flux_relative_error"] < 0.05
    assert row["half_energy_flux_relative_error"] < 0.05
    assert row["temperature_relative_error"] < 0.05


def test_stage29_design_contains_reduced_and_non_cartesian_rules():
    quadratures = build_stage29_quadratures()
    families = {quadrature.family for quadrature in quadratures}
    assert families == {
        "cartesian_midpoint",
        "tensor_gauss_hermite",
        "spherical_product",
    }
    assert any(quadrature.point_count < 17**3 for quadrature in quadratures)


def test_stage29_writes_reproducible_outputs(tmp_path):
    summary = run_stage29(tmp_path)
    assert summary["stage"] == 29
    assert summary["conditions"]["no_physical_parameter_retuning"] is True
    assert len(summary["schemes"]) == 5
    assert summary["decision"] in {
        "integrate_best_non_cartesian_rule_into_reduced_kinetic_solver",
        "audit_wall_observable_and_sign_convention_before_solver_integration",
    }
    stored = json.loads((tmp_path / "summary.json").read_text())
    assert stored["best_non_cartesian"] == summary["best_non_cartesian"]
    arrays = np.load(tmp_path / "quadrature_metrics.npz")
    assert arrays["point_count"].shape == (5,)
    assert np.isfinite(arrays["composite_max_error"]).all()
