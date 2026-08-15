#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vgdsmc"
    / "mohammadzadeh_mv16a_frozen_cylinder_transfer.py"
)
SPEC = importlib.util.spec_from_file_location("mv16a_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mv16a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mv16a)


def _raw_row(cell: int, x: float, y: float, ux: float, uy: float = 0.0) -> list[float]:
    m0 = 100.0
    variance = 4.0
    vvxx = ux * ux + variance
    vvyy = uy * uy + variance
    vvzz = variance
    speed2 = vvxx + vvyy + vvzz
    energy = 0.5 * mv16a.ARGON_MASS * speed2
    return [
        float(cell),
        1.0,
        x,
        y,
        0.01,
        m0,
        m0 * ux,
        m0 * uy,
        0.0,
        m0 * vvxx,
        m0 * vvyy,
        m0 * vvzz,
        m0 * ux * uy,
        0.0,
        0.0,
        m0 * energy,
        m0 * energy * ux,
        m0 * energy * uy,
    ]


def _write_moment(path: Path, nout: int, rows: list[list[float]]) -> None:
    lines = [
        "# MV11_ADDITIVE_KINETIC_MOMENTS_VERSION=1",
        f"# NOUT={nout} TIME= 1.0D-03 FNUM= 2.0 BLOCK_SAMPLES= 5",
        "# cell species x y area m0 m1x m1y m1z m2xx m2yy m2zz m2xy m2xz m2yz energy energy_vx energy_vy",
    ]
    lines.extend(" ".join(f"{value:.17e}" for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_contract_has_disjoint_complete_late_split() -> None:
    value = mv16a.verify_contract()
    assert value["DSMC_rerun"] is False
    assert value["neural_retraining"] is False
    assert set(mv16a.B3_NOUT).isdisjoint(mv16a.B10_NOUT)
    assert set(mv16a.B3_NOUT + mv16a.B10_NOUT + mv16a.QC_NOUT) == set(range(100, 117))


def test_parser_accepts_padded_fortran_metadata_and_exact_nout() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "MV11_MOMENTS_NOUT0100.DAT"
        _write_moment(path, 100, [_raw_row(1, -0.2, 0.2, 1.0)])
        metadata, data = mv16a.parse_moment_file(path)
        assert int(metadata["NOUT"]) == 100
        assert metadata["TIME"] == 1.0e-3
        assert data.shape == (1, 18)


def test_parser_rejects_filename_header_nout_mismatch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "MV11_MOMENTS_NOUT0100.DAT"
        _write_moment(path, 101, [_raw_row(1, -0.2, 0.2, 1.0)])
        try:
            mv16a.parse_moment_file(path)
        except ValueError as error:
            assert "mismatch" in str(error)
        else:
            raise AssertionError("expected NOUT mismatch failure")


def test_additive_sum_precedes_centralisation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "MV11_MOMENTS_NOUT0100.DAT"
        second = root / "MV11_MOMENTS_NOUT0101.DAT"
        _write_moment(first, 100, [_raw_row(1, -0.2, 0.2, 1.0)])
        _write_moment(second, 101, [_raw_row(1, -0.2, 0.2, -1.0)])
        m1, d1 = mv16a.aggregate_moment_files([first])
        m2, d2 = mv16a.aggregate_moment_files([second])
        ma, da = mv16a.aggregate_moment_files([first, second])
        f1, f2, fa = (
            mv16a.reconstruct_fields(m1, d1),
            mv16a.reconstruct_fields(m2, d2),
            mv16a.reconstruct_fields(ma, da),
        )
        # Mixture variance includes the between-block velocity fluctuation.
        individual_normal = 0.5 * (f1["outputs"][0, 1] + f2["outputs"][0, 1])
        assert fa["outputs"][0, 1] > individual_normal
        assert ma["BLOCK_SAMPLES"] == 10.0


def test_aggregation_rejects_geometry_change() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "MV11_MOMENTS_NOUT0100.DAT"
        second = root / "MV11_MOMENTS_NOUT0101.DAT"
        _write_moment(first, 100, [_raw_row(1, -0.2, 0.2, 1.0)])
        _write_moment(second, 101, [_raw_row(1, -0.19, 0.2, 1.0)])
        try:
            mv16a.aggregate_moment_files([first, second])
        except ValueError as error:
            assert "geometry changed" in str(error)
        else:
            raise AssertionError("expected fail-closed adaptive-geometry rejection")


def test_rasterisation_is_deterministic_and_masks_cylinder() -> None:
    x = np.linspace(mv16a.DOMAIN[0], mv16a.DOMAIN[1], 28)
    y = np.linspace(mv16a.DOMAIN[2], mv16a.DOMAIN[3], 18)
    xx, yy = np.meshgrid(x, y)
    keep = (xx * xx + yy * yy) >= mv16a.CYLINDER_RADIUS**2
    px, py = xx[keep], yy[keep]
    outputs = np.stack((px, py, px + py, px - py), axis=1)
    auxiliary = np.stack((np.ones_like(px), px, py, np.ones_like(px)), axis=1)
    fields = {"x_m": px, "y_m": py, "outputs": outputs, "auxiliary": auxiliary}
    first, mask1, audit1 = mv16a.rasterize_fields(fields, (32, 48))
    second, mask2, audit2 = mv16a.rasterize_fields(fields, (32, 48))
    assert first.shape == (10, 32, 48)
    assert np.array_equal(first, second)
    assert np.array_equal(mask1, mask2)
    assert np.all(first[:4, ~mask1] == 0.0)
    assert audit1 == audit2
    assert audit1["minimum_linear_fluid_coverage"] > 0.90
    assert audit1["condition_clipping_or_reinterpretation"] is False


def test_leave_one_seed_out_never_uses_own_value() -> None:
    values = np.asarray([0.0, 3.0, 6.0, 9.0])[:, None, None]
    target = mv16a.leave_one_seed_out(values)
    assert target[:, 0, 0].tolist() == [6.0, 5.0, 4.0, 3.0]


def test_masked_nrmse_ignores_solid_pixels() -> None:
    target = np.ones((2, 2))
    candidate = target.copy()
    candidate[0, 0] = 1000.0
    mask = np.asarray([[False, True], [True, True]])
    assert mv16a.masked_nrmse(candidate, target, mask) == 0.0


def test_metrics_keep_per_seed_identity() -> None:
    target = np.ones((4, 2, 2))
    masks = np.ones_like(target, dtype=bool)
    methods = {"perfect": target.copy(), "zero": np.zeros_like(target)}
    per_seed, means = mv16a._per_seed_metrics(methods, target, masks)
    assert tuple(int(seed) for seed in per_seed["perfect"]) == mv16a.SEEDS
    assert means["perfect"] == 0.0
    assert means["zero"] == 1.0


def test_six_panel_uses_locked_method_order_and_writes_both_formats() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        shape = (10, 12)
        mask = np.ones(shape, dtype=bool)
        reference = np.linspace(-1.0, 1.0, np.prod(shape)).reshape(shape)
        methods = {
            "raw_b3": reference + 0.2,
            "vision_b3": reference + 0.1,
            "selected_b3": reference + 0.05,
            "tsvd_b3": reference + 0.15,
            "raw_b10": reference + 0.02,
        }
        names = mv16a._plot_six_panel(
            output,
            mv16a.SEEDS[0],
            methods,
            reference,
            mask,
            100.0,
            150.0,
            25.0,
        )
        assert len(names) == 2
        assert all((output / name).stat().st_size > 1000 for name in names)


def test_json_serializer_handles_numpy_booleans() -> None:
    encoded = mv16a._json_dumps({"pass": np.bool_(True), "value": np.float64(1.0)})
    assert json.loads(encoded) == {"pass": True, "value": 1.0}


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in sorted(tests, key=lambda function: function.__name__):
        test()
    print(f"MV16A_FROZEN_CYLINDER_TRANSFER_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
