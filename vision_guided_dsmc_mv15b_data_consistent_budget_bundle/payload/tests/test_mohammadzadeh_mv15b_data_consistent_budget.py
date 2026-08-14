#!/usr/bin/env python3
"""Numerical and protocol tests for MV15B DCIR-QY."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from vgdsmc import mohammadzadeh_mv15b_data_consistent_budget as mv15b


def synthetic_blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images, conditions, identities, targets = [], [], [], []
    for condition_index, condition in enumerate(("c0", "c1")):
        for seed in (11, 13):
            target = np.full((4, 5, 6), condition_index + seed / 100.0, dtype=np.float32)
            for block in range(10):
                image = target.copy()
                image[3] += block
                image = np.concatenate(
                    (
                        image,
                        np.full((4, 5, 6), 0.25, dtype=np.float32),
                        np.full((1, 5, 6), condition_index, dtype=np.float32),
                        np.full((1, 5, 6), 2.0 * condition_index, dtype=np.float32),
                    )
                )
                images.append(image)
                conditions.append(condition)
                identities.append((seed, block, 1))
                targets.append(target)
    return (
        np.asarray(images),
        np.asarray(conditions, dtype="U32"),
        np.asarray(identities, dtype=np.int64),
        np.asarray(targets),
    )


def test_protocol_lock() -> None:
    result = mv15b.verify_lock()
    assert result["budgets"] == [1, 2, 3, 5]
    assert result["exact_DCT_DC_preservation"] is True


def test_cli_json_boundary_converts_numpy_scalars() -> None:
    encoded = mv15b._json_dumps(
        {
            "checks": {"close": np.bool_(True)},
            "count": np.int64(3),
        }
    )
    decoded = json.loads(encoded)
    assert decoded == {"checks": {"close": True}, "count": 3}


def test_disjoint_budget_groups() -> None:
    images, conditions, identities, targets = synthetic_blocks()
    result = mv15b.aggregate_disjoint(images, conditions, identities, 2, targets)
    assert len(result["images"]) == 2 * 2 * 5
    for condition in np.unique(result["conditions"]):
        for seed in np.unique(result["identities"][result["conditions"] == condition, 0]):
            mask = (result["conditions"] == condition) & (result["identities"][:, 0] == seed)
            used = result["members"][mask, :2].reshape(-1)
            assert np.array_equal(np.sort(used), np.arange(10))


def test_B3_explicitly_drops_only_block9() -> None:
    images, conditions, identities, targets = synthetic_blocks()
    result = mv15b.aggregate_disjoint(images, conditions, identities, 3, targets)
    for condition in np.unique(result["conditions"]):
        for seed in np.unique(result["identities"][result["conditions"] == condition, 0]):
            mask = (result["conditions"] == condition) & (result["identities"][:, 0] == seed)
            used = result["members"][mask, :3].reshape(-1)
            assert np.array_equal(np.sort(used), np.arange(9))


def test_budget_reliability_is_monotone() -> None:
    signal = np.asarray([[3.0, 0.3], [0.03, 0.0]])
    noise = np.ones_like(signal)
    previous = mv15b.budget_reliability(signal, noise, 1)
    for budget in (2, 3, 5):
        current = mv15b.budget_reliability(signal, noise, budget)
        assert np.all(current >= previous)
        previous = current


def test_trust_map_is_modewise_and_DC_exact() -> None:
    reliability = np.asarray(
        [[0.1, 0.95, 0.2], [0.99, 0.4, 0.91], [0.2, 0.2, 0.2]], dtype=np.float64
    )
    weight = mv15b.trust_weight_map(reliability, threshold=0.9, strength=0.5)
    assert weight[0, 0] == 1.0
    assert weight[0, 1] > 0.0 and weight[1, 0] > 0.0 and weight[1, 2] > 0.0
    assert weight[0, 2] == 0.0 and weight[2, 0] == 0.0


def test_data_consistency_preserves_each_raw_mean() -> None:
    rng = np.random.default_rng(20260814)
    raw = rng.normal(size=(5, 7, 9))
    vision = rng.normal(size=(5, 7, 9)) + 4.0
    reliability = rng.uniform(size=(7, 9))
    weight = mv15b.trust_weight_map(reliability, threshold=0.8, strength=0.75)
    prediction = mv15b.data_consistent_residual(raw, vision, weight)
    assert np.allclose(
        np.mean(prediction, axis=(-2, -1)),
        np.mean(raw, axis=(-2, -1)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_selection_repairs_a_DC_bias() -> None:
    rng = np.random.default_rng(31)
    target = rng.normal(size=(8, 8, 8))
    raw = target + rng.normal(scale=0.03, size=target.shape)
    vision = target + 0.75
    conditions = np.asarray(["a"] * 4 + ["b"] * 4)
    reliability = np.zeros((8, 8))
    selected, _ = mv15b.select_data_consistency(
        raw, vision, target, conditions, reliability
    )
    weight = mv15b.trust_weight_map(
        reliability,
        selected["mode_reliability_threshold"],
        selected["trusted_mode_strength"],
    )
    corrected = mv15b.data_consistent_residual(raw, vision, weight)
    assert mv15b._nrmse(corrected, target) < mv15b._nrmse(vision, target)
    assert selected["DC_weight"] == 1.0


def test_prediction_stage_has_no_legacy_target_key() -> None:
    source = Path(mv15b.__file__).read_text(encoding="utf-8")
    prediction_source = source[
        source.index("def run_prediction_stage") : source.index("def _per_seed_qy")
    ]
    assert 'data["test_y"]' not in prediction_source
    assert 'data["test_target10"]' not in prediction_source
    assert 'data["test_raw10"]' not in prediction_source


def test_compact_package_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "result"
        returned = Path(temporary) / "returned"
        root.mkdir()
        summary = {
            "decision": "synthetic",
            "diagnostic_best_budget": 2,
            "recommended_budget_for_separately_locked_fresh_confirmation": None,
        }
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        required = (
            "selection_summary.json",
            "mv15b_data_consistent_budget_protocol.json",
            "mv15b_development_selection.csv",
            "mv15b_legacy_budget_metrics.csv",
            "mv15b_qy_disjoint_budget_ladder.png",
            "mv15b_qy_disjoint_budget_ladder.pdf",
            "mv15b_qy_modewise_trust_B2.png",
            "mv15b_qy_modewise_trust_B2.pdf",
            "mv15b_qy_physical_contours_B2.png",
            "mv15b_qy_physical_contours_B2.pdf",
        )
        for name in required:
            (root / name).write_bytes(b"synthetic\n")
        locked = root / "locked_predictions.npz"
        np.savez_compressed(locked, value=np.arange(3))
        manifest = {
            "stage": mv15b.STAGE,
            "files": {
                locked.name: {
                    "sha256": mv15b._sha256(locked),
                    "size_bytes": locked.stat().st_size,
                }
            },
        }
        (root / "prediction_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = mv15b.package_results(root, returned)
        assert Path(result["archive"]).is_file()
        assert mv15b._sha256(Path(result["archive"])) == result["archive_sha256"]


def test_synthetic_prediction_to_post_lock() -> None:
    rng = np.random.default_rng(1515)
    development = (
        ("kn0p05_u100", np.log10(0.05), 1.0),
        ("kn0p05_u200", np.log10(0.05), 2.0),
        ("kn0p05_u400", np.log10(0.05), 4.0),
        ("kn0p1_u100", np.log10(0.1), 1.0),
    )
    legacy = (
        ("kn0p075_u150", np.log10(0.075), 1.5),
        ("kn0p075_u300", np.log10(0.075), 3.0),
        ("kn0p1_u200", np.log10(0.1), 2.0),
        ("kn0p1_u400", np.log10(0.1), 4.0),
    )

    def field(log_kn: float, speed: float) -> np.ndarray:
        y, x = np.mgrid[0:1:8j, 0:1:8j]
        qy = 0.08 * speed * np.cos(np.pi * x) - 0.04 * (log_kn + 1.5) + 0.02 * y
        return np.stack((0.01 * x, 0.01 * y, 0.02 * (x - y), qy)).astype(np.float32)

    def split(specs, seeds):
        images, targets, conditions, identities, scales = [], [], [], [], []
        full, full_targets, full_conditions, full_identities = [], [], [], []
        for condition, log_kn, speed in specs:
            target = field(log_kn, speed)
            for seed in seeds:
                blocks = []
                for block in range(10):
                    raw = target + rng.normal(scale=0.035, size=target.shape)
                    macros = np.stack(
                        (
                            np.ones((8, 8)),
                            np.full((8, 8), 0.1 * speed),
                            np.zeros((8, 8)),
                            np.ones((8, 8)),
                        )
                    )
                    metadata = np.stack(
                        (
                            np.full((8, 8), log_kn),
                            np.full((8, 8), speed),
                        )
                    )
                    image = np.concatenate((raw, macros, metadata)).astype(np.float32)
                    images.append(image)
                    targets.append(target)
                    conditions.append(condition)
                    identities.append((seed, block, 1))
                    scales.append(np.ones(4))
                    blocks.append(raw)
                full.append((np.mean(blocks, axis=0) + rng.normal(scale=0.004, size=target.shape)).astype(np.float32))
                full_targets.append(target)
                full_conditions.append(condition)
                full_identities.append((seed, 0, 10))
        return {
            "x": np.asarray(images),
            "y": np.asarray(targets),
            "condition": np.asarray(conditions, dtype="U32"),
            "identity": np.asarray(identities, dtype=np.int64),
            "scale": np.asarray(scales),
            "raw10": np.asarray(full),
            "target10": np.asarray(full_targets),
            "condition10": np.asarray(full_conditions, dtype="U32"),
            "identity10": np.asarray(full_identities, dtype=np.int64),
        }

    train = split(development, (101, 103, 107))
    validation = split(development, (109,))
    test = split(legacy, (94301, 94302, 94303, 94304))
    original_verify = mv15b.verify_data_contract
    original_mv14 = mv15b._mv14_module
    original_mv9 = mv15b._mv9_module

    class FakeMV14:
        @staticmethod
        def _predict_mamba_validation(_root, images, *, batch_size):
            del batch_size
            prediction = np.asarray(images[:, :4], dtype=np.float64).copy()
            prediction[:, 3] = 0.82 * prediction[:, 3] + 0.045
            return prediction.astype(np.float32)

    class FakeMV9:
        @staticmethod
        def _project_modules():
            return {"tsvd": lambda values, rank: np.asarray(values)}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        mv9_root = root / "mv9"
        mv15a_root = root / "mv15a"
        output = root / "mv15b"
        returned = root / "returned"
        mv9_root.mkdir()
        mv15a_root.mkdir()
        np.savez_compressed(
            mv9_root / "dataset.npz",
            train_x=train["x"],
            train_y=train["y"],
            train_condition=train["condition"],
            train_identity=train["identity"],
            validation_x=validation["x"],
            validation_y=validation["y"],
            validation_condition=validation["condition"],
            validation_identity=validation["identity"],
            test_x=test["x"],
            test_y=test["y"],
            test_condition=test["condition"],
            test_identity=test["identity"],
            test_scale=test["scale"],
            test_raw10=test["raw10"],
            test_target10=test["target10"],
            test_condition10=test["condition10"],
            test_identity10=test["identity10"],
        )
        (mv9_root / "assembly_summary.json").write_text(
            json.dumps({"classical_selection_development_only": {"tsvd_rank": 4}}),
            encoding="utf-8",
        )
        audit = mv15b._mv15a_module().cross_spectral_information(
            train["x"][:, mv15b.QY_INDEX], train["condition"], train["identity"]
        )
        np.savez_compressed(
            mv15a_root / "locked_predictions.npz",
            global_signal_mode=audit["global_signal_mode"],
            global_noise_mode=audit["global_noise_mode"],
        )
        try:
            mv15b.verify_data_contract = lambda *_args, **_kwargs: {"synthetic": True}
            mv15b._mv14_module = lambda: FakeMV14
            mv15b._mv9_module = lambda: FakeMV9
            prediction = mv15b.run_prediction_stage(
                mv9_root, mv15a_root, output, batch_size=4
            )
            assert prediction["legacy_test_targets_loaded"] is False
            summary = mv15b.run_legacy_post(output)
            assert set(summary["budget_results"]) == {"1", "2", "3", "5"}
            for budget in mv15b.BUDGETS:
                assert summary["budget_results"][str(budget)]["maximum_DC_absolute_error"] <= 1.0e-12
            result = mv15b.package_results(output, returned)
            assert Path(result["archive"]).is_file()
        finally:
            mv15b.verify_data_contract = original_verify
            mv15b._mv14_module = original_mv14
            mv15b._mv9_module = original_mv9


def main() -> None:
    tests = [
        test_protocol_lock,
        test_cli_json_boundary_converts_numpy_scalars,
        test_disjoint_budget_groups,
        test_B3_explicitly_drops_only_block9,
        test_budget_reliability_is_monotone,
        test_trust_map_is_modewise_and_DC_exact,
        test_data_consistency_preserves_each_raw_mean,
        test_selection_repairs_a_DC_bias,
        test_prediction_stage_has_no_legacy_target_key,
        test_compact_package_roundtrip,
        test_synthetic_prediction_to_post_lock,
    ]
    for test in tests:
        test()
    print(f"MV15B_DATA_CONSISTENT_BUDGET_TESTS_PASS count={len(tests)}")


if __name__ == "__main__":
    main()
