from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "vgdsmc"
    / "mohammadzadeh_mv9_heat_flux.py"
)
SPEC = importlib.util.spec_from_file_location("mv9_kinetic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mv9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mv9)


class VHS:
    mass = 5.0


class Config:
    nx = 2
    ny = 2
    cell_volume = 0.25
    number_density = 10.0
    t0 = 4.0
    lid_velocity_x = 3.0
    vhs = VHS()


def synthetic_payload(samples: int = 10):
    ncell = Config.nx * Config.ny
    m0 = np.full(ncell, 100.0)
    mean = np.tile(np.asarray([2.0, -1.0, 0.5]), (ncell, 1))
    covariance = np.asarray(
        [
            [4.0, 0.6, 0.0],
            [0.6, 3.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    second = covariance[None] + np.einsum("ni,nj->nij", mean, mean)
    mean_speed2 = np.trace(second, axis1=1, axis2=2)
    mean_velocity2 = np.sum(mean**2, axis=1)
    central_energy_velocity = np.tile(np.asarray([0.8, -0.4, 0.2]), (ncell, 1))
    energy_velocity = (
        central_energy_velocity
        + mean * mean_speed2[:, None]
        + 2.0 * np.einsum("nij,nj->ni", second, mean)
        - 2.0 * mean * mean_velocity2[:, None]
    )
    return {
        "samples": samples,
        "simulated_count": np.full(ncell, 50.0),
        "m0": m0,
        "m1": m0[:, None] * mean,
        "m2": m0[:, None, None] * second,
        "energy": m0 * mean_speed2,
        "energy_velocity": m0[:, None] * energy_velocity,
    }


def test_additive_moments_recover_stress_and_heat_flux():
    kb = 2.0
    payload = synthetic_payload()
    outputs, auxiliary, diagnostics = mv9.moment_fields(payload, Config(), kb)
    number_density = 100.0 / 10.0 / Config.cell_volume
    p_ref = Config.number_density * kb * Config.t0
    q_ref = p_ref * np.sqrt(kb * Config.t0 / Config.vhs.mass)
    expected_tau = Config.vhs.mass * number_density * 0.6 / p_ref
    expected_normal = Config.vhs.mass * number_density * (4.0 - 3.0) / p_ref
    expected_qx = 0.5 * Config.vhs.mass * number_density * 0.8 / q_ref
    expected_qy = 0.5 * Config.vhs.mass * number_density * -0.4 / q_ref
    assert outputs.shape == (4, 2, 2)
    assert auxiliary.shape == (4, 2, 2)
    assert np.allclose(outputs[0], expected_tau)
    assert np.allclose(outputs[1], expected_normal)
    assert np.allclose(outputs[2], expected_qx)
    assert np.allclose(outputs[3], expected_qy)
    assert diagnostics["minimum_covariance_eigenvalue_over_isotropic_scale"] > 0.0


def test_provenance_path_remains_float64_until_explicit_training_cast():
    payload = synthetic_payload()
    audit, audit_aux, _ = mv9.moment_fields(
        payload, Config(), 2.0, output_dtype=None
    )
    training, training_aux, _ = mv9.moment_fields(payload, Config(), 2.0)
    assert audit.dtype == np.float64
    assert audit_aux.dtype == np.float64
    assert training.dtype == np.float32
    assert training_aux.dtype == np.float32
    assert np.max(np.abs(audit - training.astype(np.float64))) > 0.0
    assert mv9._relative_array_difference(audit[2:], audit[2:].copy()) == 0.0


def test_merging_additive_blocks_recovers_full_payload():
    first = synthetic_payload(samples=5)
    second = synthetic_payload(samples=5)
    for name in ("simulated_count", "m0", "m1", "m2", "energy", "energy_velocity"):
        first[name] *= 0.5
        second[name] *= 0.5
    merged = mv9.merge_moment_payloads((first, second))
    full = synthetic_payload(samples=10)
    assert merged["samples"] == 10
    for name in ("simulated_count", "m0", "m1", "m2", "energy", "energy_velocity"):
        assert np.array_equal(merged[name], full[name])


def test_block_full_audit_is_fixed_scale_at_momentum_zero_crossings():
    full = synthetic_payload(samples=10)
    merged = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in full.items()
    }
    full["m1"][0, 2] = 0.0
    merged["m1"][0, 2] = 2.0e-10
    audit = mv9.additive_payload_agreement(merged, full)
    assert audit["sample_count_match"] is True
    assert audit["components"]["m1"]["absolute_linf"] == 2.0e-10
    assert audit["maximum_relative_linf"] < 1.0e-9

    structural = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in full.items()
    }
    structural["m1"] *= 1.001
    structural_audit = mv9.additive_payload_agreement(structural, full)
    assert structural_audit["maximum_relative_linf"] > 1.0e-4


def test_block_full_audit_reports_sample_count_mismatch_without_crashing():
    full = synthetic_payload(samples=10)
    merged = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in full.items()
    }
    merged["samples"] = 9
    audit = mv9.additive_payload_agreement(merged, full)
    assert audit["sample_count_match"] is False
    assert audit["maximum_relative_linf"] == 0.0


def test_componentwise_metric_is_scale_safe_at_zero_crossings():
    y = np.ones((3, 4, 5, 5), dtype=np.float32)
    y[:, 2:, :, 2] = 0.0
    prediction = y + 0.1
    metric = mv9.field_metrics(prediction, y)
    assert np.isfinite(metric["composite_nrmse"])
    assert np.isfinite(metric["heat_flux_composite_nrmse"])
    assert set(metric["per_field_nrmse"]) == set(mv9.OUTPUT_FIELDS)


def test_leave_one_out_targets_exclude_the_input_seed():
    values = {
        11: np.full((4, 2, 2), 1.0, dtype=np.float32),
        12: np.full((4, 2, 2), 3.0, dtype=np.float32),
        13: np.full((4, 2, 2), 8.0, dtype=np.float32),
    }
    targets = mv9._leave_one_out(values)
    assert np.array_equal(targets[11], np.full((4, 2, 2), 5.5, dtype=np.float32))
    assert np.array_equal(targets[12], np.full((4, 2, 2), 4.5, dtype=np.float32))
    assert np.array_equal(targets[13], np.full((4, 2, 2), 2.0, dtype=np.float32))


def test_physical_figure_has_vector_and_600_dpi_outputs(tmp_path):
    y, x = np.mgrid[-1:1:8j, -1:1:8j]
    reference = 0.05 * x * y
    methods = {
        "raw_b1": reference + 0.02 * np.sin(4 * x),
        "gaussian_b1": reference + 0.01,
        "tsvd_b1": reference - 0.008,
        "nafnet_small": reference + 0.004 * y,
        "mambairv2_tiny_adapted": reference - 0.003 * x,
        "raw_b10": reference + 0.006 * x * y,
    }
    record = mv9._physical_figure(tmp_path, 0, methods, reference, 20.0)
    assert (tmp_path / record["png"]).stat().st_size > 1000
    assert (tmp_path / record["pdf"]).stat().st_size > 1000


def test_protocol_is_explicitly_exploratory_and_forbids_local_percent_error():
    path = (
        Path(__file__).parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / mv9.PROTOCOL_FILE
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["stage"] == mv9.STAGE
    assert protocol["status"] == mv9.STATUS
    assert protocol["scientific_role"]["classification"].startswith("exploratory_")
    assert protocol["moment_contract"]["local_percent_error_forbidden"] is True
    assert (
        protocol["pre_model_feasibility_gates"][
            "block_full_additive_moment_fixed_scale_relative_linf_tolerance"
        ]
        == 1.0e-9
    )
    assert protocol["execution_matrix"]["model_tasks"] == 6
    assert protocol["target_contract"]["self_target_leakage"] is False
    assert (
        protocol["analysis_contract"][
            "maximum_heat_flux_composite_ratio_to_raw_B10"
        ]
        == 1.0
    )
    assert protocol["source_contract"]["mv8_protocol_sha256"]
