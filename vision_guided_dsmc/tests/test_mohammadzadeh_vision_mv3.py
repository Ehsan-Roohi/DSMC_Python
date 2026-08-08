from __future__ import annotations

from pathlib import Path

import numpy as np

from vgdsmc.mohammadzadeh_mv3_reference import load_protocol
from vgdsmc.mohammadzadeh_vision_mv3 import (
    BUDGETS,
    MODEL_INPUT_FIELDS,
    REPAIR_CONDITION_ID,
    REPAIR_SEED,
    REPAIR_STATUS,
    _source_directory,
    _source_summary_passes,
    build_budget_arrays,
    evaluate_fields,
    fold_split,
    fold_targets,
    select_residual_gate,
    task_from_index,
    task_index,
)


def test_mv3_reuses_an_accepted_m3_seed_without_reinterpreting_stationarity() -> None:
    summary = {
        "status": "complete_M3_qy_precision_seed",
        "decision": "complete_M3_seed_awaiting_eight_seed_aggregation",
        "mechanical_checks": {
            "collision_probability_pass": True,
            "finite_fields_pass": True,
            "stationarity_pass": False,
        },
    }
    assert _source_summary_passes(summary, "complete_M3_qy_precision_seed")
    summary["mechanical_checks"]["finite_fields_pass"] = False
    assert not _source_summary_passes(summary, "complete_M3_qy_precision_seed")


def test_mv3_new_references_still_require_stationarity_and_accept_decision() -> None:
    summary = {
        "status": "complete_MV3_reference_seed",
        "decision": "accept_MV3_reference_seed",
        "scientific_scope": "T_and_u_reference_only_heat_flux_excluded",
        "mechanical_checks": {
            "finite_fields_pass": True,
            "stationarity_pass": True,
        },
        "stationarity": {
            "checks": {
                "macroscopic_lid_slip_center": True,
                "temperature_min_K": True,
            }
        },
    }
    assert _source_summary_passes(summary, "complete_MV3_reference_seed")
    summary["mechanical_checks"]["stationarity_pass"] = False
    summary["stationarity"]["checks"]["temperature_min_K"] = False
    assert not _source_summary_passes(summary, "complete_MV3_reference_seed")


def test_mv3_heat_flux_exclusion_prevents_qy_only_false_rejection() -> None:
    summary = {
        "status": "complete_MV3_reference_seed",
        "decision": "hold_MV3_reference_seed",
        "scientific_scope": "T_and_u_reference_only_heat_flux_excluded",
        "mechanical_checks": {
            "finite_fields_pass": True,
            "stationarity_pass": False,
        },
        "stationarity": {
            "checks": {
                "macroscopic_lid_slip_center": True,
                "microscopic_lid_slip_center": True,
                "temperature_max_K": True,
                "temperature_min_K": True,
                "qy_profile_max_normalized": False,
                "qy_profile_min_normalized": False,
            }
        },
    }
    assert _source_summary_passes(summary, "complete_MV3_reference_seed")


def test_mv3_repair_reuses_completed_T_u_data_when_only_qy_failed() -> None:
    summary = {
        "status": REPAIR_STATUS,
        "decision": "hold_MV3_reference_stability_repair_seed",
        "scientific_scope": "T_and_u_reference_only_heat_flux_excluded",
        "mechanical_checks": {
            "all_event_mechanics_gates_pass": True,
            "stationarity_pass": False,
        },
        "stationarity": {
            "checks": {
                "macroscopic_lid_slip_center": True,
                "microscopic_lid_slip_center": True,
                "temperature_max_K": True,
                "temperature_min_K": True,
                "qy_profile_max_normalized": True,
                "qy_profile_min_normalized": False,
            }
        },
    }
    assert _source_summary_passes(summary, REPAIR_STATUS)
    summary["stationarity"]["checks"]["temperature_min_K"] = False
    assert not _source_summary_passes(summary, REPAIR_STATUS)


def test_mv3_T_u_failure_is_not_hidden_by_heat_flux_exclusion() -> None:
    summary = {
        "status": "complete_MV3_reference_seed",
        "decision": "hold_MV3_reference_seed",
        "scientific_scope": "T_and_u_reference_only_heat_flux_excluded",
        "mechanical_checks": {
            "finite_fields_pass": True,
            "stationarity_pass": False,
        },
        "stationarity": {
            "checks": {
                "macroscopic_lid_slip_center": True,
                "temperature_max_K": True,
                "temperature_min_K": False,
                "qy_profile_max_normalized": True,
                "qy_profile_min_normalized": True,
            }
        },
    }
    assert not _source_summary_passes(summary, "complete_MV3_reference_seed")


def test_mv3_uses_the_locked_stability_repair_only_after_it_exists(
    tmp_path: Path,
) -> None:
    condition = {
        "id": REPAIR_CONDITION_ID,
        "source": "new_MV3_reference",
    }
    original = (
        tmp_path / "references" / REPAIR_CONDITION_ID / f"seed_{REPAIR_SEED}"
    )
    assert _source_directory(condition, REPAIR_SEED, Path(), tmp_path) == original
    repair = (
        tmp_path
        / "reference_stability_repair"
        / REPAIR_CONDITION_ID
        / f"seed_{REPAIR_SEED}"
    )
    repair.mkdir(parents=True)
    (repair / "summary.json").write_text("{}", encoding="utf-8")
    assert _source_directory(condition, REPAIR_SEED, Path(), tmp_path) == repair


def test_mv3_task_array_and_fold_conditions_are_disjoint() -> None:
    protocol = load_protocol()
    for index in range(16):
        fold, budget = task_from_index(index)
        assert task_index(fold, budget) == index
        assert budget in BUDGETS
        split = fold_split(fold, protocol)
        heldout = split["heldout_condition"]
        assert heldout not in split["train"]
        assert heldout not in split["validation"]
        assert set(split["train"]) == set(split["validation"])
        for condition_id in split["train"]:
            assert len(split["train"][condition_id]) == 3
            assert len(split["validation"][condition_id]) == 1
            assert not set(split["train"][condition_id]) & set(split["validation"][condition_id])


def test_mv3_targets_exclude_current_seed_and_test_condition_from_training() -> None:
    protocol = load_protocol()
    split = fold_split(0, protocol)
    full = {}
    for condition in protocol["conditions"]:
        condition_id = condition["id"]
        full[condition_id] = {
            int(seed): np.full((2, 2, 2), float(seed), dtype=np.float32)
            for seed in condition["evaluation_seeds"]
        }
    targets = fold_targets(full, split)
    for condition_id, train_seeds in split["train"].items():
        for seed in train_seeds:
            expected = np.mean([other for other in train_seeds if other != seed])
            assert np.allclose(targets[condition_id][seed], expected)
        validation_seed = split["validation"][condition_id][0]
        assert np.allclose(targets[condition_id][validation_seed], np.mean(train_seeds))
    heldout = split["heldout_condition"]
    test_seeds = split["test"][heldout]
    for seed in test_seeds:
        assert np.allclose(targets[heldout][seed], np.mean([other for other in test_seeds if other != seed]))


def test_mv3_condition_channels_are_appended_to_every_image() -> None:
    protocol = load_protocol()
    condition = protocol["conditions"][1]
    condition_id = condition["id"]
    seed = condition["development_seeds"][0]
    blocks = {condition_id: {seed: np.ones((10, 5, 3, 4), dtype=np.float32)}}
    targets = {condition_id: {seed: np.ones((2, 3, 4), dtype=np.float32)}}
    x, y, labels, identity = build_budget_arrays(
        blocks,
        targets,
        {condition_id: (seed,)},
        {condition_id: condition},
        2,
    )
    assert x.shape == (5, len(MODEL_INPUT_FIELDS), 3, 4)
    assert y.shape == (5, 2, 3, 4)
    assert np.allclose(x[:, -2], np.log10(condition["knudsen"]))
    assert np.allclose(x[:, -1], condition["lid_speed_m_per_s"] / 100.0)
    assert set(labels) == {condition_id}
    assert np.array_equal(identity[:, 1], np.arange(5))


def test_validation_residual_gate_can_disable_a_harmful_correction() -> None:
    target = np.zeros((3, 2, 3, 3), dtype=np.float32)
    raw = np.ones_like(target)
    harmful = np.full_like(target, 2.0)
    speeds = np.asarray([100.0, 200.0, 400.0])
    alpha, records = select_residual_gate(raw, harmful, target, speeds, (0.0, 0.5, 1.0))
    assert alpha == 0.0
    assert len(records) == 3


def test_field_evaluation_uses_the_condition_lid_speed_for_slip() -> None:
    target = np.zeros((1, 2, 2, 2), dtype=np.float32)
    target[:, 1, -1] = 100.0
    raw = target.copy()
    raw[:, 1, -1] = 80.0
    metrics_100 = evaluate_fields(raw, raw, target, 100.0)
    metrics_200 = evaluate_fields(raw, raw, target, 200.0)
    assert metrics_100["validated_profiles"]["macroscopic_lid_slip"]["raw_nrmse"] != metrics_200["validated_profiles"]["macroscopic_lid_slip"]["raw_nrmse"]
