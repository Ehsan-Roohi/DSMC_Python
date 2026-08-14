from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np

from vgdsmc import mohammadzadeh_mv15a_spectral_information_audit as mv15a


def test_protocol_is_locked_and_stage_is_postprocessing_only() -> None:
    record = mv15a.verify_lock()
    assert record["stage"] == mv15a.STAGE
    assert record["DSMC_rerun"] is False
    assert record["neural_network_retraining"] is False
    assert record["legacy_targets_loaded_by_prediction_stage"] is False


def test_exact_affine_decomposition_closes_and_separates_modes() -> None:
    target = np.linspace(-2.0, 3.0, 100).reshape(10, 10)
    candidate = 0.8 * target + 0.35
    record = mv15a.exact_affine_error_decomposition(candidate, target)
    assert abs(record["slope"] - 0.8) < 1.0e-12
    assert abs(record["mean_offset"] - (np.mean(candidate) - np.mean(target))) < 1.0e-12
    assert record["orthogonal_residual_MSE"] < 1.0e-25
    assert record["closure_error"] < 1.0e-12
    assert record["oracle_diagnostic_only"] is True


def test_cross_seed_spectrum_detects_reliable_low_modes() -> None:
    rng = np.random.default_rng(20260814)
    ny, nx = 24, 28
    y = np.linspace(0.0, 1.0, ny)[:, None]
    x = np.linspace(0.0, 1.0, nx)[None, :]
    signal = 2.0 + 0.7 * np.cos(np.pi * x) * np.cos(np.pi * y)
    fields, conditions, identities = [], [], []
    for seed in (1001, 1002, 1003):
        for block in range(5):
            fields.append(signal + rng.normal(scale=1.5, size=(ny, nx)))
            conditions.append("kn0p05_u100")
            identities.append((seed, block, 1))
    audit = mv15a.cross_spectral_information(
        np.asarray(fields), np.asarray(conditions), np.asarray(identities), bins=8
    )
    reliability = audit["global_reliability_by_bin"]
    assert reliability[0] > reliability[-1]
    assert np.all((reliability >= 0.0) & (reliability <= 1.0))
    assert audit["records"][0]["pair_count"] == 15


def test_spectral_fusion_preserves_convex_endpoints() -> None:
    rng = np.random.default_rng(44)
    raw = rng.normal(size=(3, 9, 11))
    vision = rng.normal(size=raw.shape)
    np.testing.assert_allclose(
        mv15a.spectral_fuse(raw, vision, np.ones(raw.shape[-2:])), raw,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        mv15a.spectral_fuse(raw, vision, np.zeros(raw.shape[-2:])), vision,
        atol=2.0e-12,
    )


def test_numpy_dct_backend_is_orthonormal_and_matches_scipy_when_available() -> None:
    rng = np.random.default_rng(202615)
    value = rng.normal(size=(3, 9, 11))
    coefficients = mv15a._numpy_dct2(value)
    np.testing.assert_allclose(mv15a._numpy_idct2(coefficients), value, atol=2.0e-13)
    np.testing.assert_allclose(
        np.sum(coefficients**2, axis=(-2, -1)),
        np.sum(value**2, axis=(-2, -1)),
        atol=2.0e-12,
    )
    try:
        from scipy.fft import dctn
    except ModuleNotFoundError:
        return
    expected = dctn(value, axes=(-2, -1), norm="ortho")
    np.testing.assert_allclose(coefficients, expected, atol=2.0e-13)


def test_leave_condition_out_selection_returns_bounded_weights() -> None:
    rng = np.random.default_rng(987)
    shape = (12, 10)
    conditions = np.repeat(np.asarray(("a", "b", "c", "d")), 4)
    target = rng.normal(size=(len(conditions), *shape))
    raw = target + rng.normal(scale=0.25, size=target.shape)
    vision = 0.9 * target + 0.08 + rng.normal(scale=0.12, size=target.shape)
    mapping, _ = mv15a.radial_bin_map(shape, bins=6)
    selected, records = mv15a.select_spectral_fusion(
        raw, vision, target, conditions, mapping, np.linspace(0.9, 0.1, 6)
    )
    weights = np.asarray(selected["final_weight_by_bin"])
    assert len(records) == len(mv15a.WEIGHT_SHRINKAGES) * len(mv15a.RADIAL_SMOOTHING_PASSES)
    assert np.all((weights >= 0.0) & (weights <= 1.0))
    assert set(selected["leave_one_condition_out_qy_nrmse"]) == {"a", "b", "c", "d"}


def test_condition_only_control_never_uses_indexed_B1_fields() -> None:
    conditions = np.asarray(("a", "b", "c", "d"))
    images = np.zeros((4, 10, 3, 3), dtype=np.float64)
    images[:, -2] = np.asarray((-1.3, -1.3, -1.0, -1.0))[:, None, None]
    images[:, -1] = np.asarray((1.0, 4.0, 1.0, 4.0))[:, None, None]
    targets = (
        1.0
        + 2.0 * images[:, -2, 0, 0]
        - 0.4 * images[:, -1, 0, 0]
        + 0.2 * images[:, -2, 0, 0] * images[:, -1, 0, 0]
    )[:, None, None] * np.ones((4, 3, 3))
    first = mv15a.parametric_condition_only(images, targets, conditions, images)
    altered = images.copy()
    altered[:, :-2] = 1.0e6
    second = mv15a.parametric_condition_only(altered, targets, conditions, altered)
    np.testing.assert_array_equal(first, second)


def test_cross_condition_permutation_is_balanced_and_changes_condition() -> None:
    conditions = np.repeat(np.asarray(("a", "b", "c", "d")), 5)
    permutation = mv15a.cross_condition_permutation(conditions)
    assert sorted(permutation.tolist()) == list(range(len(conditions)))
    assert np.all(conditions[permutation] != conditions)


def test_prediction_stage_cannot_index_legacy_targets_and_scripts_submit_no_solver() -> None:
    tree = ast.parse(inspect.getsource(mv15a.run_prediction_stage))
    forbidden = {"test_y", "test_target10", "test_raw10", "test_scale", "test_gaussian", "test_tsvd"}
    indexed = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert not (indexed & forbidden)
    protocol = json.loads(mv15a.protocol_path().read_text(encoding="utf-8"))
    assert protocol["spectral_fusion_contract"]["legacy_target_used_for_weight_selection"] is False
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    submit = (scripts / "submit_mohammadzadeh_mv15a_spectral_audit_unity.sh").read_text(encoding="utf-8")
    assert "sbatch" in submit
    assert "dsmcFoam" not in submit
    assert "DS2V" not in submit


def main() -> None:
    test_protocol_is_locked_and_stage_is_postprocessing_only()
    test_exact_affine_decomposition_closes_and_separates_modes()
    test_cross_seed_spectrum_detects_reliable_low_modes()
    test_spectral_fusion_preserves_convex_endpoints()
    test_numpy_dct_backend_is_orthonormal_and_matches_scipy_when_available()
    test_leave_condition_out_selection_returns_bounded_weights()
    test_condition_only_control_never_uses_indexed_B1_fields()
    test_cross_condition_permutation_is_balanced_and_changes_condition()
    test_prediction_stage_cannot_index_legacy_targets_and_scripts_submit_no_solver()
    print("MV15A_SPECTRAL_INFORMATION_TESTS_PASS count=9")


if __name__ == "__main__":
    main()
