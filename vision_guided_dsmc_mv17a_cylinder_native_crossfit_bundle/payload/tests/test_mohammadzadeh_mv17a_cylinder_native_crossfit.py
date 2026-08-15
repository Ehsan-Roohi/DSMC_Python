#!/usr/bin/env python3
"""Deterministic unit tests for MV17A."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import numpy as np

from vgdsmc import mohammadzadeh_mv17a_cylinder_native_crossfit as mv17a


def test_contract() -> None:
    value = mv17a.verify_contract()
    assert value["DSMC_rerun"] is False
    assert value["ordered_double_crossfit_fold_count"] == 12
    assert value["corrects_MV16B_near_wall_origin_error"] is True


def test_double_crossfit_roles_are_disjoint_and_complete() -> None:
    folds = mv17a.double_crossfit_roles()
    assert len(folds) == 12
    assert len(set(folds)) == 12
    for observation, reference, training in folds:
        assert observation != reference
        assert len(training) == 2
        assert set((observation, reference, *training)) == set(range(4))


def test_correct_cylinder_geometry() -> None:
    theta = np.linspace(0.05, math.pi - 0.05, 80)
    radius = mv17a.CYLINDER_RADIUS + np.linspace(0.001, 0.20, 80)
    x = mv17a.CYLINDER_CENTER[0] + radius * np.cos(theta)
    y = radius * np.sin(theta)
    geometry = mv17a.polar_geometry(x, y)
    assert float(geometry["minimum_radius_m"]) > mv17a.CYLINDER_RADIUS
    assert np.all(np.asarray(geometry["rho"]) >= 0.0)
    assert np.all(np.asarray(geometry["rho"]) <= 1.0)


def test_vector_rotation_round_trip() -> None:
    theta = np.linspace(0.0, math.pi, 100)
    cosine, sine = np.cos(theta), np.sin(theta)
    qx = np.sin(2.0 * theta) + 0.3
    qy = np.cos(3.0 * theta) - 0.2
    qn, qt = mv17a.cartesian_to_normal_tangential(qx, qy, cosine, sine)
    recovered_x, recovered_y = mv17a.normal_tangential_to_cartesian(qn, qt, cosine, sine)
    assert np.allclose(recovered_x, qx, atol=1.0e-13)
    assert np.allclose(recovered_y, qy, atol=1.0e-13)


def test_dct_round_trip() -> None:
    rng = np.random.default_rng(17)
    value = rng.normal(size=(2, 16, 12))
    assert np.allclose(mv17a._idct(mv17a._dct(value)), value, atol=1.0e-12)


def test_binned_transfer_recovers_known_map() -> None:
    rng = np.random.default_rng(18)
    source = rng.normal(size=(12, 2, 16, 12))
    target = 0.4 * source
    _, blocks, audit = mv17a.fit_binned_transfer(
        source,
        target,
        bins=(2, 2),
        ridge_fraction=0.0,
    )
    assert np.allclose(blocks[1, 1], 0.4 * np.eye(2), atol=2.0e-2)
    assert np.allclose(blocks[0, 0], np.eye(2), atol=0.0)
    assert audit["maximum_singular_value"] <= 1.0 + 1.0e-12


def test_apply_transfer_identity() -> None:
    rng = np.random.default_rng(19)
    prior = rng.normal(size=(2, 8, 6))
    raw = rng.normal(size=(2, 8, 6))
    identity = np.broadcast_to(np.eye(2), (8, 6, 2, 2)).copy()
    assert np.allclose(mv17a.apply_transfer(prior, raw, identity), raw)


def test_phase_control_preserves_dc_and_changes_non_dc() -> None:
    residual = np.ones((2, 16, 12))
    scrambled = mv17a.phase_scramble_residual(residual)
    assert np.array_equal(scrambled[:, 0, 0], residual[:, 0, 0])
    assert np.count_nonzero(scrambled != residual) > residual.size // 4


def test_cartesian_dc_preservation() -> None:
    area = np.linspace(1.0, 2.0, 50)
    raw_x = np.linspace(-2.0, 1.0, 50)
    raw_y = np.linspace(0.5, 3.0, 50)
    x, y, audit = mv17a.preserve_cartesian_dc(
        raw_x + 4.0,
        raw_y - 7.0,
        raw_x,
        raw_y,
        area,
    )
    assert audit["qx_DC_absolute_error"] < 1.0e-12
    assert audit["qy_DC_absolute_error"] < 1.0e-12
    assert abs(np.average(x, weights=area) - np.average(raw_x, weights=area)) < 1.0e-12
    assert abs(np.average(y, weights=area) - np.average(raw_y, weights=area)) < 1.0e-12


def test_area_weighted_nrmse() -> None:
    target = np.asarray([1.0, 2.0, 3.0])
    candidate = target + 0.5
    area = np.asarray([1.0, 2.0, 4.0])
    expected = math.sqrt(np.average((candidate - target) ** 2, weights=area)) / math.sqrt(
        np.average(target**2, weights=area)
    )
    assert abs(mv17a.area_weighted_nrmse(candidate, target, area) - expected) < 1.0e-14


def test_small_n_sign_limit() -> None:
    statistics = mv17a.paired_statistics([0.8, 0.8, 0.8, 0.8], [1.0, 1.0, 1.0, 1.0])
    assert statistics["improved_seed_count"] == 4
    assert statistics["exact_sign_test_one_sided_p"] == 0.0625
    assert statistics["minimum_attainable_one_sided_p_at_n"] == 0.0625


def test_numpy_json_types() -> None:
    value = mv17a._json_dumps({"flag": np.bool_(True), "number": np.float64(1.5)})
    assert json.loads(value) == {"flag": True, "number": 1.5}


def test_manifest_round_trip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "value.txt"
        path.write_text("locked\n", encoding="utf-8")
        mv17a._write_manifest(root, "manifest.json", [path])
        verified = mv17a._verify_manifest(root, "manifest.json")
        assert verified["files"][0]["sha256"] == mv17a._sha256(path)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"MV17A_CYLINDER_NATIVE_CROSSFIT_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
