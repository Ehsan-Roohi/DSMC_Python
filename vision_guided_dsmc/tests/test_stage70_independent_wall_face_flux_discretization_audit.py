import numpy as np
import pytest

from vgdsmc import stage70_independent_wall_face_flux_discretization_audit as s


def test_completed_stage69_endpoint_is_exact():
    endpoint = s.STAGE69_COMPLETED_ENDPOINT
    assert endpoint["workflow_run_id"] == 31027231271
    assert endpoint["workflow_job_id"] == 92378691275
    assert endpoint["tests_passed"] == 107
    assert endpoint["tests_failed"] == 0
    assert endpoint["artifact_id"] == 8942482076
    assert endpoint["artifact_sha256"] == (
        "8040126435a35726fe995360e5c5c3a807f6bfa6ed27d139405a53f1a5775e78"
    )
    assert endpoint["decision"].endswith(
        "stage70_independent_wall_face_flux_discretization_audit"
    )


def test_frozen_design_accepts_exact_configuration():
    s.validate_stage70_design()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grids": (8, 16, 32, 64)},
        {"fine_grid": 32},
        {"kn0": 1.0},
        {"cold_hot_ratio": 0.2},
        {"rule": (32, 96)},
        {"radial_scale": 1.0},
        {"limiter": "unlimited"},
        {"positivity": "clip"},
        {"restriction": "injection"},
        {"material_heat_flux_ratio": 0.05},
    ],
)
def test_frozen_design_rejects_retuning(kwargs):
    with pytest.raises(ValueError):
        s.validate_stage70_design(**kwargs)


def test_conservative_restriction_preserves_velocity_bin_means():
    rng = np.random.default_rng(70)
    fine = rng.random((8, 8, 9))
    for target in (1, 2, 4, 8):
        restricted = s.restrict_cell_average(fine, target)
        np.testing.assert_allclose(
            restricted.mean(axis=(0, 1)),
            fine.mean(axis=(0, 1)),
            rtol=0.0,
            atol=5.0e-16,
        )


def test_one_sided_wall_face_recovers_linear_extrapolation():
    yy, xx = np.indices((5, 6), dtype=float)
    field = (10.0 + 2.0 * xx + 3.0 * yy)[..., None]
    center, raw, bounded, theta = s.one_sided_wall_face(field, "left")
    np.testing.assert_allclose(raw, center - 1.0)
    np.testing.assert_allclose(bounded, raw)
    np.testing.assert_allclose(theta, 1.0)
    center, raw, bounded, theta = s.one_sided_wall_face(field, "right")
    np.testing.assert_allclose(raw, center + 1.0)
    np.testing.assert_allclose(bounded, raw)
    np.testing.assert_allclose(theta, 1.0)
    center, raw, bounded, theta = s.one_sided_wall_face(field, "bottom")
    np.testing.assert_allclose(raw, center - 1.5)
    np.testing.assert_allclose(bounded, raw)
    np.testing.assert_allclose(theta, 1.0)
    center, raw, bounded, theta = s.one_sided_wall_face(field, "top")
    np.testing.assert_allclose(raw, center + 1.5)
    np.testing.assert_allclose(bounded, raw)
    np.testing.assert_allclose(theta, 1.0)


def test_analytic_slope_rescaling_bounds_negative_extrapolation_without_clipping():
    field = np.ones((4, 4, 2), dtype=float)
    field[:, 0, 0] = 0.1
    field[:, 1, 0] = 1.1
    field[:, 2, 0] = 2.1
    center, raw, bounded, theta = s.one_sided_wall_face(field, "left")
    assert np.all(raw[:, 0] < 0.0)
    assert np.all(bounded[:, 0] >= -1.0e-15)
    assert np.all((theta[:, 0] >= 0.0) & (theta[:, 0] < 1.0))
    np.testing.assert_allclose(bounded[:, 1], raw[:, 1])


def test_diffuse_wall_face_recomputes_density_and_closes_mass_flux():
    vx = np.array([-1.0, 1.0, -1.0, 1.0])
    vy = np.array([-1.0, -1.0, 1.0, 1.0])
    weight = np.full(4, 0.25)
    normal, incoming, temperature = s.wall_geometry("bottom", 3, vx, vy)
    outgoing_phi = np.array(
        [[1.0, 1.2, 0.0, 0.0], [0.7, 1.4, 0.0, 0.0], [1.3, 0.9, 0.0, 0.0]]
    )
    outgoing_psi = 0.8 * outgoing_phi
    state = s.diffuse_wall_face_state(
        outgoing_phi,
        outgoing_psi,
        normal,
        incoming,
        temperature,
        vx,
        vy,
        weight,
    )
    assert np.max(np.abs(state["mass_flux"])) <= 2.0e-15
    assert np.all(state["scale"] > 0.0)
    assert np.all(np.isfinite(state["energy_flux"]))


def test_observed_order_and_monotonic_guards():
    assert s.observed_order(0.2, 0.1) == pytest.approx(1.0)
    assert s.monotonically_decreases_with_refinement([0.4, 0.2, 0.1]) is True
    assert s.monotonically_decreases_with_refinement([0.4, 0.4, 0.1]) is False


def test_decision_paths_preserve_scientific_guards():
    base = dict(
        finite=True,
        provenance_consistent=True,
        mass_flux_closed=True,
        qav_reproduced=True,
        qav_difference_monotonic=True,
        fine_qav_difference_ratio=0.0076,
        bounded_vs_raw_material=False,
    )
    assert s.stage70_decision(**base) == (
        "stage70_wall_face_heat_flux_difference_below_materiality_"
        "stage71_wall_layer_interior_face_transport_attribution_audit"
    )
    assert "linearized_wall_response" in s.stage70_decision(
        **{**base, "fine_qav_difference_ratio": 0.10}
    )
    assert s.stage70_decision(**{**base, "finite": False}) == (
        "stage70_nonfinite_wall_face_flux_blocker"
    )
    assert s.stage70_decision(**{**base, "provenance_consistent": False}) == (
        "stage70_completed_endpoint_reproduction_blocker"
    )
    assert s.stage70_decision(**{**base, "mass_flux_closed": False}) == (
        "stage70_diffuse_wall_mass_flux_blocker"
    )
    assert s.stage70_decision(**{**base, "qav_reproduced": False}) == (
        "stage70_retained_qav_reproduction_blocker"
    )
    assert s.stage70_decision(**{**base, "bounded_vs_raw_material": True}) == (
        "stage70_material_positivity_rescaling_blocker"
    )
    assert s.stage70_decision(**{**base, "qav_difference_monotonic": False}) == (
        "stage70_nonmonotone_wall_face_flux_blocker"
    )
