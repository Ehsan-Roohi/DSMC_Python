#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vgdsmc"
    / "mohammadzadeh_mv16b_jcp_evidence_audit.py"
)
SPEC = importlib.util.spec_from_file_location("mv16b_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mv16b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mv16b)


def test_contract_forbids_new_simulation_training_and_selection() -> None:
    value = mv16b.verify_contract()
    assert value["DSMC_rerun"] is False
    assert value["neural_training"] is False
    assert value["fresh_parameter_selection"] is False
    assert len(value["cylinder_seeds"]) == 4


def test_dct_round_trip_is_machine_precision() -> None:
    rng = np.random.default_rng(4)
    array = rng.normal(size=(3, 13, 17))
    restored = mv16b._idct2(mv16b._dct2(array))
    assert np.max(np.abs(array - restored)) < 1e-12


def test_continuous_wiener_is_bounded_and_preserves_dc() -> None:
    signal = np.ones((8, 9))
    noise = 3.0 * np.ones((8, 9))
    gain = mv16b.continuous_wiener_gain(signal, noise, 3)
    assert gain[0, 0] == 1.0
    assert np.all((gain >= 0.0) & (gain <= 1.0))
    raw = np.arange(72, dtype=float).reshape(1, 8, 9)
    result = mv16b.pure_wiener(raw, gain)
    assert abs(float(np.mean(raw) - np.mean(result))) < 1e-12


def test_data_consistency_preserves_each_observation_mean() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(4, 10, 12))
    prior = rng.normal(size=raw.shape)
    weight = np.zeros(raw.shape[-2:])
    weight[0, 0] = 1.0
    weight[1:3, 1:4] = 0.25
    result = mv16b.data_consistent_residual(raw, prior, weight)
    assert np.max(np.abs(np.mean(result, axis=(-2, -1)) - np.mean(raw, axis=(-2, -1)))) < 1e-12


def test_condition_prior_uses_only_development_conditions() -> None:
    conditions = np.asarray(
        ["kn0p05_u100", "kn0p05_u200", "kn0p1_u100", "kn0p1_u200"]
    )
    values = []
    for condition in conditions:
        log_kn, speed = mv16b.condition_coordinates(condition)
        qy = 2.0 + 3.0 * log_kn - 0.5 * speed + 0.25 * log_kn * speed
        channels = np.zeros((4, 5, 6))
        channels[mv16b.QY_INDEX] = qy
        values.append(channels)
    prior, audit = mv16b.development_condition_prior(
        np.asarray(values), conditions, ["kn0p08_u350", "kn0p1_u400"]
    )
    assert prior.shape == (2, 5, 6)
    assert audit["fresh_labels_used"] is False
    assert audit["condition_surface_RMSE"] < 1e-12


def test_leave_one_seed_out_never_uses_self() -> None:
    fields = np.asarray([0.0, 3.0, 6.0, 9.0])[:, None]
    target = mv16b.leave_one_seed_out(fields)
    assert target[:, 0].tolist() == [6.0, 5.0, 4.0, 3.0]
    conditioned = np.concatenate((fields, fields + 100.0), axis=0)
    labels = np.asarray(["a"] * 4 + ["b"] * 4)
    result = mv16b.leave_one_seed_out(conditioned, labels)
    assert result[0, 0] == 6.0 and result[4, 0] == 106.0


def test_reference_noise_correction_matches_closed_form() -> None:
    assert mv16b.reference_noise_corrected_ratio(0.5) == 0.0
    assert math.isclose(
        mv16b.reference_noise_corrected_ratio(1.0), 1.0, rel_tol=0.0, abs_tol=1e-15
    )


def test_paired_statistics_state_small_n_limit() -> None:
    result = mv16b.paired_log_ratio_statistics(
        [0.7, 0.8, 0.9, 0.6], [1.0, 1.0, 1.0, 1.0]
    )
    assert result["improved_pair_count"] == 4
    assert result["exact_sign_test_one_sided_p"] == 0.0625
    assert result["exact_sign_test_two_sided_p"] == 0.125
    assert result["geometric_mean_ratio"] < 1.0


def test_masked_basis_contains_no_solid_samples_and_has_full_rank() -> None:
    x = np.linspace(-0.2, 0.65, 25)
    y = np.linspace(0.0, 0.4, 18)
    xx, yy = np.meshgrid(x, y)
    keep = xx * xx + yy * yy >= 0.1524**2
    px, py = xx[keep], yy[keep]
    area = np.full(len(px), 1.0 / len(px))
    weight = np.zeros((20, 24))
    weight[0, 0] = 1.0
    weight[0, 1:5] = 0.2
    weight[1:4, 0:4] = 0.1
    matrix, gains, audit = mv16b.masked_mode_matrix(
        px, py, area, weight, (-0.2, 0.65, 0.0, 0.4)
    )
    assert matrix.shape == (len(px), np.count_nonzero(weight))
    assert gains[0] == 1.0
    assert audit["weighted_design_rank"] == matrix.shape[1]
    assert audit["solid_cells_in_operator"] == 0


def test_native_data_consistency_preserves_area_weighted_mean() -> None:
    x = np.linspace(0.0, 1.0, 80)
    matrix = np.column_stack((np.ones_like(x), np.cos(np.pi * x), np.cos(2 * np.pi * x)))
    gains = np.asarray([1.0, 0.25, 0.25])
    area = np.linspace(0.5, 1.5, len(x))
    raw = 1.5 + np.sin(2 * np.pi * x)
    prior = -0.3 + 0.5 * np.sin(2 * np.pi * x)
    result, audit = mv16b.native_data_consistent_residual(raw, prior, area, matrix, gains)
    assert result.shape == raw.shape
    assert audit["weighted_DC_absolute_error"] < 1e-12


def test_normal_heat_flux_projects_vector_on_radial_normal() -> None:
    x = np.asarray([1.0, 0.0, math.sqrt(0.5)])
    y = np.asarray([0.0, 1.0, math.sqrt(0.5)])
    qx = np.ones(3)
    qy = np.zeros(3)
    qn = mv16b._normal_heat_flux(qx, qy, x, y)
    assert np.allclose(qn, [1.0, 0.0, math.sqrt(0.5)])


def test_json_boundary_handles_numpy_scalars() -> None:
    encoded = mv16b._json_dumps({"ok": np.bool_(True), "value": np.float64(1.25)})
    assert json.loads(encoded) == {"ok": True, "value": 1.25}


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in sorted(tests, key=lambda function: function.__name__):
        test()
    print(f"MV16B_JCP_EVIDENCE_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
