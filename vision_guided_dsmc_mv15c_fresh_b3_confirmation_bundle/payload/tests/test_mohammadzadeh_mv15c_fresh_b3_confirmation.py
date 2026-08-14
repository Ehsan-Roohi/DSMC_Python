#!/usr/bin/env python3
"""Protocol, leakage, numerical, and packaging tests for MV15C."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from vgdsmc import mohammadzadeh_mv15b_data_consistent_budget as mv15b
from vgdsmc import mohammadzadeh_mv15c_fresh_b3_confirmation as mv15c


def test_protocol_lock_and_fresh_matrix() -> None:
    original = mv15c.stage_configuration
    mv15c.stage_configuration = lambda stage, seed: (
        SimpleNamespace(nx=64, steps=250000),
        {},
        {"stage": stage, "seed": seed},
        {},
        {},
    )
    try:
        result = mv15c.verify_lock()
    finally:
        mv15c.stage_configuration = original
    assert result["input_budget"] == 3
    assert result["input_blocks"] == [0, 1, 2]
    assert result["parameter_tuning_on_fresh_data"] is False
    assert result["neural_network_retraining"] is False
    tasks = [(item["condition"], item["seed"]) for item in result["fresh_tasks"]]
    assert tasks == [
        ("kn0p1_u400", 151501),
        ("kn0p1_u400", 151502),
        ("kn0p1_u400", 151503),
        ("kn0p1_u400", 151504),
        ("kn0p08_u350", 151511),
        ("kn0p08_u350", 151512),
        ("kn0p08_u350", 151513),
        ("kn0p08_u350", 151514),
    ]


def test_task_index_is_strict() -> None:
    assert mv15c.task_from_index(0) == ("kn0p1_u400", 151501)
    assert mv15c.task_from_index(7) == ("kn0p08_u350", 151514)
    for value in (-1, 8):
        try:
            mv15c.task_from_index(value)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-contract task index was accepted")


def test_cli_json_boundary_converts_numpy_scalars() -> None:
    decoded = json.loads(
        mv15c._json_dumps({"gate": np.bool_(True), "count": np.int64(8)})
    )
    assert decoded == {"count": 8, "gate": True}


def test_B3_builder_uses_only_blocks_zero_one_two() -> None:
    blocks = np.stack(
        [np.full((4, 3, 5), index, dtype=np.float64) for index in range(10)]
    )
    auxiliary = np.stack(
        [np.full((2, 3, 5), 10 * index, dtype=np.float64) for index in range(10)]
    )
    captured: dict[str, np.ndarray] = {}

    class FakeMV9:
        @staticmethod
        def _conditioned_image(output, aux, condition):
            del condition
            captured["output"] = np.asarray(output)
            captured["auxiliary"] = np.asarray(aux)
            return np.concatenate((output, aux)).astype(np.float32)

    original = mv15c._mv9_module
    mv15c._mv9_module = lambda: FakeMV9
    try:
        result = mv15c.build_b3_image(
            {"blocks": blocks, "block_auxiliary": auxiliary},
            {"knudsen": 0.1, "lid_speed_m_per_s": 400.0},
        )
    finally:
        mv15c._mv9_module = original
    np.testing.assert_array_equal(captured["output"], np.ones((4, 3, 5)))
    np.testing.assert_array_equal(captured["auxiliary"], np.full((2, 3, 5), 10.0))
    assert result.shape == (6, 3, 5)


def test_leave_one_seed_out_target_excludes_indexed_seed() -> None:
    conditions = np.asarray(["a"] * 4 + ["b"] * 4)
    seeds = np.asarray((11, 12, 13, 14, 21, 22, 23, 24))
    raw_b10 = np.arange(8, dtype=np.float64)[:, None, None]
    targets = mv15c.leave_one_seed_out_targets(raw_b10, conditions, seeds)
    assert targets[0, 0, 0] == np.mean((1.0, 2.0, 3.0))
    assert targets[3, 0, 0] == np.mean((0.0, 1.0, 2.0))
    assert targets[4, 0, 0] == np.mean((5.0, 6.0, 7.0))
    perturbed = raw_b10.copy()
    perturbed[0] = 1.0e9
    changed = mv15c.leave_one_seed_out_targets(perturbed, conditions, seeds)
    assert changed[0, 0, 0] == targets[0, 0, 0]


def test_leave_one_seed_out_requires_exactly_three_peers() -> None:
    try:
        mv15c.leave_one_seed_out_targets(
            np.zeros((3, 2, 2)), np.asarray(["a"] * 3), np.arange(3)
        )
    except ValueError as error:
        assert "exactly three" in str(error)
    else:
        raise AssertionError("undersampled fresh target was accepted")


def test_frozen_data_consistency_preserves_each_B3_mean() -> None:
    rng = np.random.default_rng(151503)
    raw = rng.normal(size=(8, 9, 11))
    vision = rng.normal(size=raw.shape) + 7.0
    weight = np.zeros(raw.shape[-2:])
    weight[0, 0] = 1.0
    weight[1:3, 1:4] = 0.25
    selected = mv15b.data_consistent_residual(raw, vision, weight)
    np.testing.assert_allclose(
        np.mean(selected, axis=(-2, -1)),
        np.mean(raw, axis=(-2, -1)),
        rtol=0.0,
        atol=1.0e-12,
    )


def _synthetic_gate_inputs():
    conditions = np.asarray(
        [mv15c.PRIMARY_CONDITION] * 4 + [mv15c.NEW_CONDITION] * 4
    )
    seeds = np.asarray(
        mv15c.FRESH_SEEDS[mv15c.PRIMARY_CONDITION]
        + mv15c.FRESH_SEEDS[mv15c.NEW_CONDITION]
    )
    by_condition = {
        mv15c.PRIMARY_CONDITION: 1.0,
        mv15c.NEW_CONDITION: 1.0,
    }
    means = {
        "raw_b10": dict(by_condition),
        "raw_b3": {key: 2.0 for key in by_condition},
        "vision_b3": {key: 1.6 for key in by_condition},
        "dc_only_b3": {key: 1.0 for key in by_condition},
        "selected_b3": {key: 0.8 for key in by_condition},
        "tsvd_b3": {key: 1.3 for key in by_condition},
        "permuted_b3": {key: 0.9 for key in by_condition},
    }
    seed_ratios = {
        condition: {str(seed): 0.8 for seed in values}
        for condition, values in mv15c.FRESH_SEEDS.items()
    }
    contract = json.loads(mv15c.protocol_path().read_text(encoding="utf-8"))[
        "acceptance_gates"
    ]
    return means, seed_ratios, conditions, seeds, contract


def test_preregistered_gates_pass_and_fail_without_selection() -> None:
    means, seed_ratios, conditions, seeds, contract = _synthetic_gate_inputs()
    gates = mv15c.confirmation_gates(
        means=means,
        selected_seed_ratios=seed_ratios,
        dc_error=1.0e-13,
        conditions=conditions,
        seeds=seeds,
        contract=contract,
    )
    assert all(gates.values())
    means["selected_b3"][mv15c.PRIMARY_CONDITION] = 1.01
    failed = mv15c.confirmation_gates(
        means=means,
        selected_seed_ratios=seed_ratios,
        dc_error=1.0e-13,
        conditions=conditions,
        seeds=seeds,
        contract=contract,
    )
    assert failed["each_condition_mean_no_worse_than_Raw_B10"] is False
    assert failed["no_fresh_parameter_selection"] is True


def test_prediction_source_cannot_construct_fresh_target() -> None:
    prediction = inspect.getsource(mv15c.run_prediction_stage)
    post = inspect.getsource(mv15c.run_post)
    assert "leave_one_seed_out_targets" not in prediction
    assert "confirmation_gates" not in prediction
    assert "fresh_cross_seed_targets_constructed\": False" in prediction
    assert "leave_one_seed_out_targets" in post
    protocol = json.loads(mv15c.protocol_path().read_text(encoding="utf-8"))
    assert protocol["frozen_B3_contract"][
        "fresh_fields_forbidden_for_weight_threshold_strength_or_model_selection"
    ] is True


def test_compact_package_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "output"
        returned = Path(temporary) / "returned"
        output.mkdir()
        summary = {
            "decision": "synthetic_no_retuning",
            "all_gates_pass": False,
        }
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        required = [
            "submission_lock.json",
            "source_lock_manifest.json",
            "prediction_summary.json",
            "fresh_source_audit.csv",
            "mv15c_fresh_qy_metrics.csv",
            mv15c.PROTOCOL_FILE,
            "mv15c_fresh_qy_confirmation_ratios.png",
            "mv15c_fresh_qy_confirmation_ratios.pdf",
        ]
        for condition in (mv15c.PRIMARY_CONDITION, mv15c.NEW_CONDITION):
            seed = mv15c.FRESH_SEEDS[condition][0]
            required.extend(
                (
                    f"mv15c_fresh_qy_{condition}_seed_{seed}.png",
                    f"mv15c_fresh_qy_{condition}_seed_{seed}.pdf",
                )
            )
        for name in required:
            (output / name).write_bytes(b"synthetic\n")
        locked = output / "locked_fresh_predictions.npz"
        np.savez_compressed(locked, value=np.arange(3))
        prediction_manifest = {
            "stage": mv15c.STAGE,
            "files": {
                locked.name: {
                    "sha256": mv15c._sha256(locked),
                    "size_bytes": locked.stat().st_size,
                }
            },
        }
        (output / "prediction_manifest.json").write_text(
            json.dumps(prediction_manifest), encoding="utf-8"
        )
        result = mv15c.package_results(output, returned)
        archive = Path(result["archive"])
        assert archive.is_file()
        assert mv15c._sha256(archive) == result["archive_sha256"]
        pointer = returned / "LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
        assert pointer.is_file()


def main() -> None:
    test_protocol_lock_and_fresh_matrix()
    test_task_index_is_strict()
    test_cli_json_boundary_converts_numpy_scalars()
    test_B3_builder_uses_only_blocks_zero_one_two()
    test_leave_one_seed_out_target_excludes_indexed_seed()
    test_leave_one_seed_out_requires_exactly_three_peers()
    test_frozen_data_consistency_preserves_each_B3_mean()
    test_preregistered_gates_pass_and_fail_without_selection()
    test_prediction_source_cannot_construct_fresh_target()
    test_compact_package_roundtrip()
    print("MV15C_FRESH_B3_CONFIRMATION_TESTS_PASS count=10")


if __name__ == "__main__":
    main()
