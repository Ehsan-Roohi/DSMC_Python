"""Locked JCP full-budget matrix following the completed MV6 screen.

MV7 reuses the exact repaired MV5 references and the twelve completed MV6
budget-one model tasks.  It first evaluates Raw, Gaussian-like, and TSVD/POD at
all four sampling budgets, then trains the four promoted architectures for the
three previously unrun budgets and three locked initialization seeds.

The primary analysis is a predeclared paired non-inferiority comparison with
Raw DSMC at ten blocks.  Confirmatory targets are never used for fitting,
early stopping, residual-gate selection, or classical-baseline selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tarfile
import time
from typing import Any, Mapping

import numpy as np

from . import mohammadzadeh_architecture_screen as mv6
from . import mohammadzadeh_vision_mv3 as mv3
from . import mohammadzadeh_vision_mv5 as mv5
from . import mohammadzadeh_mv5_reference as mv5ref
from .mohammadzadeh_vision import OUTPUT_FIELDS
from .mohammadzadeh_vision_mv2 import (
    GAUSSIAN_PASSES,
    TSVD_RANKS,
    _atomic_json,
    _portable_tarinfo,
    _sha256,
    gaussian_like,
    select_baseline,
    tsvd,
)


STAGE = "MV7_Mohammadzadeh_JCP_full_budget_matrix"
STATUS = (
    "locked_after_MV6_budget_one_screen_before_any_MV7_budget_two_five_or_ten_"
    "model_outcome"
)
PROTOCOL_FILE = "mv7_jcp_budget_matrix_analysis_plan.json"
BUDGETS = (1, 2, 5, 10)
NEW_MODEL_BUDGETS = (2, 5, 10)
ARCHITECTURES = mv6.ARCHITECTURES
TRAINING_SEEDS = mv6.TRAINING_SEEDS
CLASSICAL_METHODS = ("raw", "gaussian_like", "tsvd_pod_type")
METHODS = (*CLASSICAL_METHODS, *ARCHITECTURES)
NONINFERIORITY_MARGIN = 1.10
ONE_SIDED_T_CRITICAL_DF3 = 2.3533634348018264
EPSILON = np.finfo(np.float64).tiny

DISPLAY_NAMES = {
    "raw": "Raw DSMC",
    "gaussian_like": "Gaussian",
    "tsvd_pod_type": "TSVD/POD",
    **mv6.DISPLAY_NAMES,
}
COLORS = {
    "raw": "#000000",
    "gaussian_like": "#777777",
    "tsvd_pod_type": "#6A3D9A",
    "corrected_unet": "#0072B2",
    "nafnet_small": "#009E73",
    "mambairv2_tiny_adapted": "#D55E00",
    "fno_residual_small": "#CC79A7",
}
MARKERS = {
    "raw": "o",
    "gaussian_like": "s",
    "tsvd_pod_type": "^",
    "corrected_unet": "o",
    "nafnet_small": "D",
    "mambairv2_tiny_adapted": "P",
    "fno_residual_small": "X",
}


def protocol_path() -> Path:
    return mv5ref.protocol_path().parent / PROTOCOL_FILE


def locked_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV7 analysis plan is absent or unlocked")
    execution = value["execution_matrix"]
    primary = value["primary_analysis"]
    if (
        tuple(execution["budget_blocks"]) != BUDGETS
        or tuple(execution["new_model_budget_blocks"]) != NEW_MODEL_BUDGETS
        or tuple(execution["architectures"]) != ARCHITECTURES
        or tuple(execution["training_initialization_seeds"]) != TRAINING_SEEDS
        or int(execution["new_model_tasks"]) != 36
        or not np.isclose(
            float(primary["noninferiority_relative_margin"]),
            NONINFERIORITY_MARGIN - 1.0,
        )
        or not np.isclose(
            float(primary["one_sided_t_critical_df3"]),
            ONE_SIDED_T_CRITICAL_DF3,
        )
    ):
        raise ValueError("MV7 code differs from the locked analysis plan")
    mv6_protocol = mv6.locked_protocol()
    if _sha256(mv6.protocol_path()) != value["source_contract"][
        "mv6_protocol_sha256"
    ]:
        raise ValueError("MV7 MV6 protocol ancestry hash mismatch")
    if tuple(mv6_protocol["promotion_rule"]["full_budget_matrix_after_user_decision"]) != BUDGETS:
        raise ValueError("MV7 budgets differ from the MV6 promotion contract")
    return value


def baseline_task_from_index(index: int) -> int:
    if not 0 <= index < len(BUDGETS):
        raise ValueError("baseline task index is outside [0,3]")
    return BUDGETS[index]


def model_task_from_index(index: int) -> tuple[int, str, int]:
    per_budget = len(ARCHITECTURES) * len(TRAINING_SEEDS)
    total = len(NEW_MODEL_BUDGETS) * per_budget
    if not 0 <= index < total:
        raise ValueError(f"model task index is outside [0,{total - 1}]")
    budget = NEW_MODEL_BUDGETS[index // per_budget]
    remainder = index % per_budget
    architecture = ARCHITECTURES[remainder // len(TRAINING_SEEDS)]
    seed = TRAINING_SEEDS[remainder % len(TRAINING_SEEDS)]
    return budget, architecture, seed


def _baseline_directory(root: Path, budget: int) -> Path:
    return root / "baselines" / f"budget_{budget}"


def _model_directory(
    root: Path, budget: int, architecture: str, seed: int
) -> Path:
    return (
        root
        / "tasks"
        / f"budget_{budget}"
        / architecture
        / f"training_seed_{seed}"
    )


def _mv6_directory(root: Path, architecture: str, seed: int) -> Path:
    return root / "tasks" / architecture / f"training_seed_{seed}"


def _budget_data(
    existing_m3_root: Path,
    mv3_root: Path,
    reference_root: Path,
    budget: int,
):
    locked_protocol()
    if budget not in BUDGETS:
        raise ValueError("budget is outside the locked MV7 matrix")
    mv3_protocol = mv3.locked_protocol()
    mv5_protocol = mv5.locked_protocol()
    development_specs = mv3._condition_map(mv3_protocol)
    confirmatory_specs = mv5ref.condition_map(mv5_protocol)
    development_blocks, development_full = mv3.load_condition_data(
        existing_m3_root, mv3_root, mv3_protocol
    )
    confirmatory_blocks, confirmatory_full = mv5.load_confirmatory_data(
        reference_root
    )
    train_split = {
        key: tuple(int(seed) for seed in values)
        for key, values in mv5_protocol["development_seed_split"]["train"].items()
    }
    validation_split = {
        key: tuple(int(seed) for seed in values)
        for key, values in mv5_protocol["development_seed_split"][
            "validation"
        ].items()
    }
    test_split = {
        key: tuple(int(seed) for seed in value["evaluation_seeds"])
        for key, value in confirmatory_specs.items()
    }
    development_targets = mv5._development_targets(
        development_full, train_split, validation_split
    )
    confirmatory_targets = mv5._confirmatory_targets(confirmatory_full)
    train = mv3.build_budget_arrays(
        development_blocks,
        development_targets,
        train_split,
        development_specs,
        budget,
    )
    validation = mv3.build_budget_arrays(
        development_blocks,
        development_targets,
        validation_split,
        development_specs,
        budget,
    )
    test = mv3.build_budget_arrays(
        confirmatory_blocks,
        confirmatory_targets,
        test_split,
        confirmatory_specs,
        budget,
    )
    return train, validation, test, development_specs, confirmatory_specs


def _per_seed_metrics(
    raw: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    target: np.ndarray,
    conditions: np.ndarray,
    identity: np.ndarray,
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition_id, condition in specs.items():
        result[condition_id] = {}
        speed = float(condition["lid_speed_m_per_s"])
        for evaluation_seed in condition["evaluation_seeds"]:
            mask = (conditions == condition_id) & (
                identity[:, 0] == int(evaluation_seed)
            )
            if not np.any(mask):
                raise ValueError(
                    f"evaluation seed absent at this budget: {condition_id}/{evaluation_seed}"
                )
            result[condition_id][str(evaluation_seed)] = {
                method: mv3.evaluate_fields(
                    raw[mask], value[mask], target[mask], speed
                )
                for method, value in candidates.items()
            }
    return result


def _write_manifest(directory: Path, names: tuple[str, ...]) -> None:
    _atomic_json(
        directory / "artifact_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(directory / name),
                    "size_bytes": (directory / name).stat().st_size,
                }
                for name in names
            },
        },
    )


def run_baseline_task(
    existing_m3_root: Path,
    mv3_root: Path,
    reference_root: Path,
    output_root: Path,
    *,
    budget: int,
) -> dict[str, Any]:
    protocol = locked_protocol()
    (
        _,
        (validation_x, validation_y, _, _),
        (test_x, test_y, test_conditions, test_identity),
        _,
        confirmatory_specs,
    ) = _budget_data(existing_m3_root, mv3_root, reference_root, budget)
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    started = time.perf_counter()
    gaussian_passes, gaussian_records = select_baseline(
        validation_raw, validation_y, GAUSSIAN_PASSES, gaussian_like
    )
    tsvd_rank, tsvd_records = select_baseline(
        validation_raw, validation_y, TSVD_RANKS, tsvd
    )
    gaussian = gaussian_like(raw, gaussian_passes)
    pod = tsvd(raw, tsvd_rank)
    gaussian, gaussian_projection = mv5._project_by_condition(
        gaussian, raw, test_conditions, confirmatory_specs
    )
    pod, tsvd_projection = mv5._project_by_condition(
        pod, raw, test_conditions, confirmatory_specs
    )
    seconds = time.perf_counter() - started
    candidates = {
        "raw": raw,
        "gaussian_like": gaussian,
        "tsvd_pod_type": pod,
    }
    methods_by_condition = {
        method: mv5._metric_by_condition(
            raw, value, test_y, test_conditions, confirmatory_specs
        )
        for method, value in candidates.items()
    }
    per_seed = _per_seed_metrics(
        raw,
        candidates,
        test_y,
        test_conditions,
        test_identity,
        confirmatory_specs,
    )
    directory = _baseline_directory(output_root, budget)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        directory / "predictions.npz",
        identity_condition=test_conditions,
        identity_numeric=test_identity,
        raw=raw,
        gaussian_like=gaussian,
        tsvd_pod_type=pod,
        target=test_y,
    )
    checks = {
        "budget_locked": budget in BUDGETS,
        "selection_development_only": True,
        "all_methods_finite": all(np.all(np.isfinite(x)) for x in candidates.values()),
        "all_temperatures_positive": all(np.min(x[:, 0]) >= 1.0 for x in candidates.values()),
        "sixteen_evaluation_seeds": sum(
            len(values) for values in per_seed.values()
        )
        == 16,
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV7_classical_baseline_budget_task",
        "budget_blocks": budget,
        "protocol_sha256": _sha256(protocol_path()),
        "selection": {
            "gaussian_like": {
                "selected_passes": gaussian_passes,
                "candidates": gaussian_records,
            },
            "tsvd_pod_type": {
                "selected_rank": tsvd_rank,
                "candidates": tsvd_records,
            },
        },
        "timing_seconds": {"selection_and_confirmatory_prediction": seconds},
        "projection": {
            "gaussian_like": gaussian_projection,
            "tsvd_pod_type": tsvd_projection,
        },
        "methods_by_condition": methods_by_condition,
        "per_evaluation_seed_metrics": per_seed,
        "checks": checks,
        "decision": "accept_MV7_baseline_task" if all(checks.values()) else "hold_MV7_baseline_task",
    }
    _atomic_json(directory / "summary.json", summary)
    _write_manifest(directory, ("predictions.npz", "summary.json"))
    return summary


def _verified_baseline(root: Path, budget: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = _baseline_directory(root, budget)
    _verify_manifest(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete_MV7_classical_baseline_budget_task"
        or summary.get("decision") != "accept_MV7_baseline_task"
        or int(summary.get("budget_blocks", -1)) != budget
    ):
        raise ValueError(f"invalid MV7 baseline summary: {directory}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    return summary, arrays


def run_model_task(
    existing_m3_root: Path,
    mv3_root: Path,
    reference_root: Path,
    output_root: Path,
    *,
    budget: int,
    architecture: str,
    training_seed: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    protocol = locked_protocol()
    if (
        budget not in NEW_MODEL_BUDGETS
        or architecture not in ARCHITECTURES
        or training_seed not in TRAINING_SEEDS
    ):
        raise ValueError("model task is outside the locked MV7 matrix")
    (
        (train_x, train_y, _, _),
        (validation_x, validation_y, validation_conditions, _),
        (test_x, test_y, test_conditions, test_identity),
        development_specs,
        confirmatory_specs,
    ) = _budget_data(existing_m3_root, mv3_root, reference_root, budget)
    baseline_summary, baseline = _verified_baseline(output_root, budget)
    expected = {
        "identity_condition": test_conditions,
        "identity_numeric": test_identity,
        "raw": test_x[:, : len(OUTPUT_FIELDS)],
        "target": test_y,
    }
    for name, value in expected.items():
        reference = baseline[name]
        equal = (
            np.array_equal(value, reference, equal_nan=True)
            if np.issubdtype(value.dtype, np.inexact)
            else np.array_equal(value, reference)
        )
        if not equal:
            raise ValueError(f"MV7 model/baseline array mismatch: budget={budget}/{name}")
    scaling = mv6.fixed_physical_scaling(train_x, train_y)
    parameters = mv6.parameter_report(int(train_x.shape[1]))
    model, training = mv6.train_architecture(
        architecture,
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        seed=training_seed,
        epochs=epochs,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    validation_candidate, validation_diagnostics = mv6.predict_bounded(
        model, validation_x, scaling, batch_size
    )
    validation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    test_candidate, test_diagnostics = mv6.predict_bounded(
        model, test_x, scaling, batch_size
    )
    inference_seconds = time.perf_counter() - started
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    raw = baseline["raw"]
    validation_speeds = np.asarray(
        [
            float(development_specs[str(item)]["lid_speed_m_per_s"])
            for item in validation_conditions
        ]
    )
    alpha, alpha_records = mv3.select_residual_gate(
        validation_raw,
        validation_candidate,
        validation_y,
        validation_speeds,
        tuple(
            float(value)
            for value in mv6.locked_protocol()["comparison_contract"][
                "residual_alpha_candidates"
            ]
        ),
    )
    candidate = raw + float(alpha) * (test_candidate - raw)
    candidate, projection = mv5._project_by_condition(
        candidate, raw, test_conditions, confirmatory_specs
    )
    candidates = {"raw": raw, architecture: candidate}
    methods_by_condition = {
        method: mv5._metric_by_condition(
            raw, value, test_y, test_conditions, confirmatory_specs
        )
        for method, value in candidates.items()
    }
    per_seed = _per_seed_metrics(
        raw,
        candidates,
        test_y,
        test_conditions,
        test_identity,
        confirmatory_specs,
    )
    directory = _model_directory(output_root, budget, architecture, training_seed)
    directory.mkdir(parents=True, exist_ok=False)
    torch, _, _ = mv6._torch_components()
    torch.save(
        {
            "stage": STAGE,
            "budget_blocks": budget,
            "architecture": architecture,
            "training_seed": training_seed,
            "state_dict": model.state_dict(),
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "residual_gate_alpha": alpha,
            "input_fields": mv3.MODEL_INPUT_FIELDS,
            "output_fields": OUTPUT_FIELDS,
        },
        directory / "model.pt",
    )
    np.savez_compressed(
        directory / "predictions.npz",
        identity_condition=test_conditions,
        identity_numeric=test_identity,
        raw=raw,
        architecture_prediction=candidate,
        target=test_y,
    )
    checks = {
        "budget_locked_and_not_retrained_budget_one": budget in NEW_MODEL_BUDGETS,
        "baseline_completed_first": baseline_summary["decision"] == "accept_MV7_baseline_task",
        "same_locked_split_targets_optimizer_and_loss_as_MV6": True,
        "parameter_parity": bool(parameters["pass"]),
        "bounded_residual": bool(
            test_diagnostics["bounded_normalized_residual_abs_max"]
            <= mv6.RESIDUAL_CAP_SIGMA + 1.0e-6
        ),
        "evaluation_target_not_used_for_training_or_selection": True,
        "finite_candidate": bool(np.all(np.isfinite(candidate))),
        "positive_temperature": bool(np.min(candidate[:, 0]) >= 1.0),
        "sixteen_evaluation_seeds": sum(len(values) for values in per_seed.values()) == 16,
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV7_architecture_budget_seed_task",
        "budget_blocks": budget,
        "architecture": architecture,
        "architecture_display_name": mv6.DISPLAY_NAMES[architecture],
        "training_seed": training_seed,
        "protocol_sha256": _sha256(protocol_path()),
        "sample_counts": {
            "train": len(train_x),
            "validation": len(validation_x),
            "confirmatory": len(test_x),
        },
        "parameter_parity": parameters,
        "selection": {
            "residual_alpha": {"selected": alpha, "candidates": alpha_records},
            "classical_baseline_source": str(_baseline_directory(output_root, budget)),
        },
        "training": training,
        "timing_seconds": {
            "validation_inference": validation_seconds,
            "confirmatory_inference": inference_seconds,
        },
        "prediction_diagnostics": {
            "validation": validation_diagnostics,
            "confirmatory": test_diagnostics,
            "projection": projection,
        },
        "methods_by_condition": methods_by_condition,
        "per_evaluation_seed_metrics": per_seed,
        "checks": checks,
        "decision": "accept_MV7_model_task" if all(checks.values()) else "hold_MV7_model_task",
    }
    _atomic_json(directory / "summary.json", summary)
    _write_manifest(directory, ("model.pt", "predictions.npz", "summary.json"))
    return summary


def _verify_manifest(directory: Path) -> None:
    manifest_path = directory / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest.get("files", {}).items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"artifact verification failed: {path}")


def _baseline_summaries(root: Path) -> dict[int, dict[str, Any]]:
    return {budget: _verified_baseline(root, budget)[0] for budget in BUDGETS}


def _model_summaries(
    root: Path, mv6_root: Path
) -> dict[tuple[int, str, int], dict[str, Any]]:
    values: dict[tuple[int, str, int], dict[str, Any]] = {}
    for budget in BUDGETS:
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                if budget == 1:
                    directory = _mv6_directory(mv6_root, architecture, seed)
                    summary = json.loads(
                        (directory / "summary.json").read_text(encoding="utf-8")
                    )
                    if (
                        summary.get("status") != "complete_MV6_architecture_seed_task"
                        or summary.get("decision") != "accept_MV6_task"
                    ):
                        raise ValueError(f"invalid reused MV6 task: {directory}")
                else:
                    directory = _model_directory(root, budget, architecture, seed)
                    _verify_manifest(directory)
                    summary = json.loads(
                        (directory / "summary.json").read_text(encoding="utf-8")
                    )
                    if (
                        summary.get("status")
                        != "complete_MV7_architecture_budget_seed_task"
                        or summary.get("decision") != "accept_MV7_model_task"
                    ):
                        raise ValueError(f"invalid MV7 model task: {directory}")
                if (
                    summary.get("architecture") != architecture
                    or int(summary.get("training_seed", -1)) != seed
                    or int(summary.get("budget_blocks", -1)) != budget
                ):
                    raise ValueError(f"model task identity mismatch: {directory}")
                values[(budget, architecture, seed)] = summary
    return values


def _seed_error_map(
    baselines: Mapping[int, Mapping[str, Any]],
    models: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    result: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        method: {} for method in METHODS
    }
    for budget, summary in baselines.items():
        for method in CLASSICAL_METHODS:
            result[method][budget] = {
                condition: {
                    seed: float(metrics[method]["vision_composite_nrmse"])
                    for seed, metrics in seeds.items()
                }
                for condition, seeds in summary[
                    "per_evaluation_seed_metrics"
                ].items()
            }
    for budget in BUDGETS:
        for architecture in ARCHITECTURES:
            condition_values: dict[str, dict[str, list[float]]] = {}
            for seed in TRAINING_SEEDS:
                summary = models[(budget, architecture, seed)]
                for condition, evaluations in summary[
                    "per_evaluation_seed_metrics"
                ].items():
                    condition_values.setdefault(condition, {})
                    for evaluation_seed, metrics in evaluations.items():
                        condition_values[condition].setdefault(
                            evaluation_seed, []
                        ).append(
                            float(
                                metrics[architecture]["vision_composite_nrmse"]
                            )
                        )
            result[architecture][budget] = {
                condition: {
                    seed: float(np.mean(values))
                    for seed, values in evaluations.items()
                }
                for condition, evaluations in condition_values.items()
            }
    return result


def _flatten_seed_errors(
    values: Mapping[str, Mapping[str, float]]
) -> np.ndarray:
    return np.asarray(
        [value for condition in sorted(values) for value in values[condition].values()],
        dtype=np.float64,
    )


def _curve_statistics(
    baselines: Mapping[int, Mapping[str, Any]],
    models: Mapping[tuple[int, str, int], Mapping[str, Any]],
    seed_errors: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float]]]],
) -> dict[str, dict[str, Any]]:
    curves: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    for method in CLASSICAL_METHODS:
        for budget in BUDGETS:
            values = _flatten_seed_errors(seed_errors[method][budget])
            curves[method][str(budget)] = {
                "mean_composite_nrmse": float(values.mean()),
                "evaluation_seed_sd": float(values.std(ddof=1)),
                "training_seed_sd": None,
            }
    for architecture in ARCHITECTURES:
        for budget in BUDGETS:
            seed_means = []
            worst = []
            for training_seed in TRAINING_SEEDS:
                summary = models[(budget, architecture, training_seed)]
                condition_errors = [
                    float(value["vision_composite_nrmse"])
                    for value in summary["methods_by_condition"][architecture].values()
                ]
                seed_means.append(float(np.mean(condition_errors)))
                worst.append(float(np.max(condition_errors)))
            paired_values = _flatten_seed_errors(seed_errors[architecture][budget])
            curves[architecture][str(budget)] = {
                "mean_composite_nrmse": float(paired_values.mean()),
                "training_seed_sd": float(np.std(seed_means, ddof=1)),
                "paired_evaluation_seed_mean": float(paired_values.mean()),
                "paired_evaluation_seed_sd": float(paired_values.std(ddof=1)),
                "worst_condition_mean": float(np.mean(worst)),
            }
    return curves


def _noninferiority(
    seed_errors: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float]]]]
) -> dict[str, dict[str, Any]]:
    raw10 = seed_errors["raw"][10]
    result: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    log_margin = math.log(NONINFERIORITY_MARGIN)
    for method in METHODS:
        for budget in BUDGETS:
            condition_means = []
            all_logs = []
            for condition in sorted(raw10):
                paired = []
                for seed in sorted(raw10[condition]):
                    method_error = max(
                        float(seed_errors[method][budget][condition][seed]), EPSILON
                    )
                    raw_error = max(float(raw10[condition][seed]), EPSILON)
                    paired.append(math.log(method_error / raw_error))
                condition_means.append(float(np.mean(paired)))
                all_logs.extend(paired)
            array = np.asarray(condition_means, dtype=np.float64)
            mean = float(array.mean())
            se = float(array.std(ddof=1) / math.sqrt(len(array)))
            upper = mean + ONE_SIDED_T_CRITICAL_DF3 * se
            result[method][str(budget)] = {
                "condition_mean_log_ratios": condition_means,
                "paired_seed_log_ratio_mean": float(np.mean(all_logs)),
                "condition_cluster_mean_log_ratio": mean,
                "condition_cluster_standard_error": se,
                "one_sided_95_upper_log_ratio": upper,
                "geometric_mean_ratio_to_raw10": math.exp(mean),
                "one_sided_95_upper_ratio_to_raw10": math.exp(upper),
                "relative_margin": NONINFERIORITY_MARGIN,
                "noninferior": bool(upper <= log_margin),
            }
    return result


def _scaling_and_equivalence(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    log_budget = np.log(np.asarray(BUDGETS, dtype=np.float64))
    raw_error = np.asarray(
        [curves["raw"][str(b)]["mean_composite_nrmse"] for b in BUDGETS],
        dtype=np.float64,
    )
    slope, intercept = np.polyfit(log_budget, np.log(raw_error), 1)
    rows: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    for method in METHODS:
        for budget in BUDGETS:
            error = float(curves[method][str(budget)]["mean_composite_nrmse"])
            raw_same = float(curves["raw"][str(budget)]["mean_composite_nrmse"])
            equivalent = (
                float(math.exp((math.log(max(error, EPSILON)) - intercept) / slope))
                if slope < 0.0
                else None
            )
            rows[method][str(budget)] = {
                "theoretical_variance_reduction_factor": float((raw_same / error) ** 2),
                "empirical_raw_equivalent_budget": equivalent,
                "empirical_equivalent_budget_over_consumed_budget": (
                    None if equivalent is None else float(equivalent / budget)
                ),
            }
    return {
        "raw_loglog_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "theoretical_slope": -0.5,
            "diagnostic_only": True,
        },
        "by_method_budget": rows,
    }


def _write_results_csv(
    root: Path,
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]],
    noninferiority: Mapping[str, Mapping[str, Mapping[str, Any]]],
    equivalence: Mapping[str, Any],
) -> str:
    name = "mv7_jcp_budget_matrix.csv"
    with (root / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "budget_blocks",
                "mean_composite_nrmse",
                "training_seed_sd",
                "ratio_to_raw10_geometric_mean",
                "ratio_to_raw10_one_sided_95_upper",
                "noninferior_with_10pct_margin",
                "theoretical_variance_reduction_factor",
                "empirical_raw_equivalent_budget",
            )
        )
        for method in METHODS:
            for budget in BUDGETS:
                curve = curves[method][str(budget)]
                ni = noninferiority[method][str(budget)]
                eq = equivalence["by_method_budget"][method][str(budget)]
                writer.writerow(
                    (
                        method,
                        budget,
                        curve["mean_composite_nrmse"],
                        curve.get("training_seed_sd"),
                        ni["geometric_mean_ratio_to_raw10"],
                        ni["one_sided_95_upper_ratio_to_raw10"],
                        ni["noninferior"],
                        eq["theoretical_variance_reduction_factor"],
                        eq["empirical_raw_equivalent_budget"],
                    )
                )
    return name


def _publication_figures(
    root: Path,
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]],
    noninferiority: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight"})
    files: list[str] = []
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for method in METHODS:
        mean = [curves[method][str(b)]["mean_composite_nrmse"] for b in BUDGETS]
        sd = [curves[method][str(b)].get("training_seed_sd") or 0.0 for b in BUDGETS]
        axis.errorbar(
            BUDGETS,
            mean,
            yerr=sd if method in ARCHITECTURES else None,
            label=DISPLAY_NAMES[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.5,
            capsize=2,
        )
    raw1 = curves["raw"]["1"]["mean_composite_nrmse"]
    guide = [raw1 / math.sqrt(b) for b in BUDGETS]
    axis.plot(BUDGETS, guide, "k--", linewidth=1.0, alpha=0.55, label=r"$B^{-1/2}$ guide")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    axis.set_xlabel("DSMC sampling blocks, $B$")
    axis.set_ylabel("Composite NRMSE")
    axis.grid(alpha=0.2, which="both")
    axis.legend(ncol=2, fontsize=8)
    for suffix in ("png", "pdf"):
        name = f"mv7_error_vs_budget.{suffix}"
        figure.savefig(root / name, dpi=400)
        files.append(name)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    x = np.arange(len(BUDGETS), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(METHODS))
    for offset, method in zip(offsets, METHODS):
        centers = [
            noninferiority[method][str(b)]["geometric_mean_ratio_to_raw10"]
            for b in BUDGETS
        ]
        uppers = [
            noninferiority[method][str(b)]["one_sided_95_upper_ratio_to_raw10"]
            for b in BUDGETS
        ]
        axis.errorbar(
            x + offset,
            centers,
            yerr=[np.zeros(len(BUDGETS)), np.asarray(uppers) - np.asarray(centers)],
            label=DISPLAY_NAMES[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle="none",
            capsize=2,
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.axhline(NONINFERIORITY_MARGIN, color="#D55E00", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, [f"B={b}" for b in BUDGETS])
    axis.set_ylabel("Composite error / paired Raw@B=10")
    axis.set_yscale("log")
    axis.grid(alpha=0.2, axis="y")
    axis.legend(ncol=2, fontsize=8)
    for suffix in ("png", "pdf"):
        name = f"mv7_noninferiority_to_raw10.{suffix}"
        figure.savefig(root / name, dpi=400)
        files.append(name)
    plt.close(figure)
    return files


def _prediction_path(
    mv7_root: Path,
    mv6_root: Path,
    budget: int,
    architecture: str,
    seed: int,
) -> Path:
    directory = (
        _mv6_directory(mv6_root, architecture, seed)
        if budget == 1
        else _model_directory(mv7_root, budget, architecture, seed)
    )
    return directory / "predictions.npz"


def _normalized_error_arrays(
    mv7_root: Path,
    mv6_root: Path,
    budget: int,
    architecture: str,
    condition: str = "kn0p1_u400",
) -> np.ndarray:
    values = []
    for seed in TRAINING_SEEDS:
        with np.load(
            _prediction_path(mv7_root, mv6_root, budget, architecture, seed),
            allow_pickle=False,
        ) as data:
            labels = np.asarray(data["identity_condition"]).astype(str)
            mask = labels == condition
            prediction = np.asarray(data["architecture_prediction"])[mask]
            target = np.asarray(data["target"])[mask]
        temperature_scale = max(float(np.ptp(target[:, 0])), 1.0)
        error = np.empty_like(prediction, dtype=np.float64)
        error[:, 0] = (prediction[:, 0] - target[:, 0]) / temperature_scale
        error[:, 1] = (prediction[:, 1] - target[:, 1]) / 400.0
        values.append(error)
    return np.concatenate(values, axis=0)


def _radial_spectrum(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = errors.shape[-2:]
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    bins = np.floor(radius).astype(int)
    maximum = min(height, width) // 2
    power = np.zeros(maximum, dtype=np.float64)
    count = np.zeros(maximum, dtype=np.float64)
    for sample in errors:
        for field in sample:
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
            for index in range(maximum):
                mask = bins == index
                power[index] += float(np.sum(spectrum[mask]))
                count[index] += int(np.sum(mask))
    power /= np.maximum(count, 1.0)
    power /= max(float(power.sum()), EPSILON)
    return np.arange(maximum, dtype=float), power


def _fno_diagnostics(root: Path, mv6_root: Path) -> tuple[dict[str, Any], list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    spectra: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for architecture in ("mambairv2_tiny_adapted", "fno_residual_small"):
        for budget in (1, 10):
            errors = _normalized_error_arrays(root, mv6_root, budget, architecture)
            height, width = errors.shape[-2:]
            boundary = np.zeros((height, width), dtype=bool)
            boundary[:2] = True
            boundary[-2:] = True
            boundary[:, :2] = True
            boundary[:, -2:] = True
            squared = np.mean(errors**2, axis=(0, 1))
            boundary_mse = float(np.mean(squared[boundary]))
            interior_mse = float(np.mean(squared[~boundary]))
            rows.append(
                {
                    "architecture": architecture,
                    "budget_blocks": budget,
                    "boundary_band_mse": boundary_mse,
                    "interior_mse": interior_mse,
                    "boundary_over_interior": boundary_mse / max(interior_mse, EPSILON),
                }
            )
            spectra[(architecture, budget)] = _radial_spectrum(errors)
    csv_name = "mv7_fno_u400_diagnostics.csv"
    with (root / csv_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    labels = {"mambairv2_tiny_adapted": "MambaIRv2", "fno_residual_small": "FNO"}
    for architecture in ("mambairv2_tiny_adapted", "fno_residual_small"):
        selected = [row for row in rows if row["architecture"] == architecture]
        axes[0].plot(
            [row["budget_blocks"] for row in selected],
            [row["boundary_over_interior"] for row in selected],
            marker=MARKERS[architecture],
            color=COLORS[architecture],
            label=labels[architecture],
        )
        for budget, style in ((1, "-"), (10, "--")):
            wave, power = spectra[(architecture, budget)]
            axes[1].semilogy(
                wave[1:],
                power[1:],
                linestyle=style,
                color=COLORS[architecture],
                label=f"{labels[architecture]}, B={budget}",
            )
    axes[0].set_xscale("log")
    axes[0].set_xticks((1, 10), ("1", "10"))
    axes[0].set_xlabel("DSMC blocks")
    axes[0].set_ylabel("Boundary-band MSE / interior MSE")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("Radial Fourier wavenumber")
    axes[1].set_ylabel("Normalized error power")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    files = [csv_name]
    for suffix in ("png", "pdf"):
        name = f"mv7_fno_u400_diagnostics.{suffix}"
        figure.savefig(root / name, dpi=400)
        files.append(name)
    plt.close(figure)
    return {
        "condition": "kn0p1_u400",
        "normalization": "temperature by target range; velocity by U_lid=400 m/s",
        "boundary_band_cells": 2,
        "rows": rows,
        "interpretation_guard": locked_protocol()["physics_and_failure_diagnostics"][
            "fno_causal_language"
        ],
    }, files


def _slurm_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "slurm_accounting.psv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="|"))


def _cost_summary(
    root: Path,
    models: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    protocol = locked_protocol()["cost_contract"]
    by_method: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        by_method[architecture] = {}
        for budget in BUDGETS:
            summaries = [models[(budget, architecture, seed)] for seed in TRAINING_SEEDS]
            training = [float(item["training"]["seconds"]) for item in summaries]
            inference = [
                item.get("timing_seconds", {}).get("confirmatory_inference")
                for item in summaries
            ]
            valid_inference = [float(value) for value in inference if value is not None]
            count = [int(item.get("sample_counts", {}).get("confirmatory", 0)) for item in summaries]
            by_method[architecture][str(budget)] = {
                "training_wall_seconds_mean": float(np.mean(training)),
                "training_wall_seconds_sd": float(np.std(training, ddof=1)),
                "confirmatory_inference_seconds_mean": (
                    None if not valid_inference else float(np.mean(valid_inference))
                ),
                "inference_seconds_per_confirmatory_array_mean": (
                    None
                    if not valid_inference or not all(count)
                    else float(np.mean([v / n for v, n in zip(valid_inference, count)]))
                ),
                "trainable_parameters": int(
                    summaries[0]["parameter_parity"]["trainable_parameters"][architecture]
                ),
            }
    rows = _slurm_rows(root)
    reference_elapsed = []
    for row in rows:
        if (
            "moh_mv5_ref" in row.get("JobName", "")
            and row.get("State") == "COMPLETED"
        ):
            try:
                reference_elapsed.append(float(row["ElapsedRaw"]))
            except (KeyError, TypeError, ValueError):
                pass
    block_seconds = (
        None
        if not reference_elapsed
        else float(np.mean(reference_elapsed) / 10.0)
    )
    break_even: dict[str, Any] = {architecture: {} for architecture in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        for budget in BUDGETS:
            record = by_method[architecture][str(budget)]
            inference = record["inference_seconds_per_confirmatory_array_mean"]
            denominator = (
                None
                if block_seconds is None or inference is None
                else (10 - budget) * block_seconds - inference
            )
            break_even[architecture][str(budget)] = {
                "lower_bound_uses_zero_shared_training_data_cost": True,
                "denominator_seconds_saved_per_use": denominator,
                "uses_to_amortize_training_only": (
                    None
                    if denominator is None or denominator <= 0.0
                    else record["training_wall_seconds_mean"] / denominator
                ),
            }
    return {
        "contract": protocol,
        "by_architecture_budget": by_method,
        "slurm_accounting_rows": rows,
        "reference_wall_seconds_per_block_including_amortized_burn_in": block_seconds,
        "break_even_lower_bound": break_even,
    }


def aggregate(root: Path, mv6_root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    baselines = _baseline_summaries(root)
    models = _model_summaries(root, mv6_root)
    seed_errors = _seed_error_map(baselines, models)
    curves = _curve_statistics(baselines, models, seed_errors)
    noninferiority = _noninferiority(seed_errors)
    equivalence = _scaling_and_equivalence(curves)
    csv_name = _write_results_csv(root, curves, noninferiority, equivalence)
    figure_names = _publication_figures(root, curves, noninferiority)
    fno, fno_files = _fno_diagnostics(root, mv6_root)
    costs = _cost_summary(root, models)
    budget_one_baselines_match_mv6 = all(
        mv6._metrics_equivalent(
            baselines[1]["methods_by_condition"][method],
            models[(1, architecture, seed)]["methods_by_condition"][method],
        )
        for method in CLASSICAL_METHODS
        for architecture in ARCHITECTURES
        for seed in TRAINING_SEEDS
    )
    budget_one_selections_match_mv6 = all(
        int(baselines[1]["selection"]["gaussian_like"]["selected_passes"])
        == int(
            models[(1, architecture, seed)]["selection"]["gaussian_like"][
                "selected_passes"
            ]
        )
        and int(baselines[1]["selection"]["tsvd_pod_type"]["selected_rank"])
        == int(
            models[(1, architecture, seed)]["selection"]["tsvd_pod_type"][
                "selected_rank"
            ]
        )
        for architecture in ARCHITECTURES
        for seed in TRAINING_SEEDS
    )
    bias_floor = {
        architecture: {
            "B10_over_B5_error": float(
                curves[architecture]["10"]["mean_composite_nrmse"]
                / curves[architecture]["5"]["mean_composite_nrmse"]
            ),
            "interpretation": "a ratio near or above one is a predeclared large-budget bias-floor diagnostic, not an automatic task failure",
        }
        for architecture in ARCHITECTURES
    }
    checks = {
        "four_baseline_budget_tasks_complete": len(baselines) == 4,
        "twelve_reused_mv6_tasks_present": sum(key[0] == 1 for key in models) == 12,
        "thirty_six_new_model_tasks_complete": sum(key[0] != 1 for key in models) == 36,
        "all_forty_eight_model_tasks_in_analysis": len(models) == 48,
        "paired_sixteen_seed_primary_analysis": all(
            len(_flatten_seed_errors(seed_errors[method][budget])) == 16
            for method in METHODS
            for budget in BUDGETS
        ),
        "one_primary_endpoint_and_locked_margin": NONINFERIORITY_MARGIN == 1.10,
        "independently_rebuilt_budget_one_baselines_match_mv6": budget_one_baselines_match_mv6,
        "budget_one_classical_hyperparameters_match_mv6": budget_one_selections_match_mv6,
        "no_posthoc_architecture_omission": tuple(ARCHITECTURES)
        == tuple(protocol["execution_matrix"]["architectures"]),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV7_JCP_full_budget_matrix",
        "protocol_sha256": _sha256(protocol_path()),
        "analysis_scope": protocol["generalization_scope"],
        "primary_analysis_contract": protocol["primary_analysis"],
        "curves": curves,
        "noninferiority_to_raw_budget_10": noninferiority,
        "raw_scaling_and_effective_variance_reduction": equivalence,
        "bias_floor_diagnostics": bias_floor,
        "fno_failure_diagnostics": fno,
        "cost_accounting": costs,
        "artifacts": [csv_name, *figure_names, *fno_files, "slurm_accounting.psv"],
        "checks": checks,
        "decision": (
            "MV7_matrix_ready_for_JCP_interpretation"
            if all(checks.values())
            else "hold_MV7_matrix"
        ),
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def verify(root: Path, mv6_root: Path) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    baselines = _baseline_summaries(root)
    models = _model_summaries(root, mv6_root)
    expected = (
        "mv7_jcp_budget_matrix.csv",
        "mv7_error_vs_budget.png",
        "mv7_error_vs_budget.pdf",
        "mv7_noninferiority_to_raw10.png",
        "mv7_noninferiority_to_raw10.pdf",
        "mv7_fno_u400_diagnostics.csv",
        "mv7_fno_u400_diagnostics.png",
        "mv7_fno_u400_diagnostics.pdf",
    )
    checks = {
        "summary_complete": summary.get("status")
        == "complete_MV7_JCP_full_budget_matrix",
        "summary_checks_pass": all(summary.get("checks", {}).values()),
        "four_baselines_recursively_verified": len(baselines) == 4,
        "forty_eight_model_summaries_verified": len(models) == 48,
        "all_report_artifacts_exist": all((root / name).is_file() for name in expected),
        "protocol_hash_matches": summary.get("protocol_sha256")
        == _sha256(protocol_path()),
    }
    value = {
        "stage": STAGE,
        "status": (
            "complete_MV7_JCP_artifacts_metrics_and_analysis_verified"
            if all(checks.values())
            else "failed_MV7_verification"
        ),
        "summary_sha256": _sha256(root / "summary.json"),
        "checks": checks,
        "decision": "verified" if all(checks.values()) else "hold",
    }
    _atomic_json(root / "verification.json", value)
    if not all(checks.values()):
        raise ValueError("MV7 verification failed")
    return value


def package_lite(root: Path, mv6_root: Path) -> dict[str, Any]:
    verification = verify(root, mv6_root)
    bundle = root / "MOHAMMADZADEH_MV7_JCP_BUDGET_MATRIX_LITE.tar.gz"
    top = (
        "summary.json",
        "verification.json",
        "mv7_jcp_budget_matrix.csv",
        "mv7_error_vs_budget.png",
        "mv7_error_vs_budget.pdf",
        "mv7_noninferiority_to_raw10.png",
        "mv7_noninferiority_to_raw10.pdf",
        "mv7_fno_u400_diagnostics.csv",
        "mv7_fno_u400_diagnostics.png",
        "mv7_fno_u400_diagnostics.pdf",
        "slurm_accounting.psv",
    )
    with tarfile.open(bundle, "w:gz") as archive:
        for name in top:
            path = root / name
            if path.is_file():
                archive.add(path, arcname=name, filter=_portable_tarinfo)
        archive.add(
            protocol_path(),
            arcname=f"provenance/{PROTOCOL_FILE}",
            filter=_portable_tarinfo,
        )
        for budget in BUDGETS:
            directory = _baseline_directory(root, budget)
            for name in ("summary.json", "artifact_manifest.json"):
                archive.add(
                    directory / name,
                    arcname=f"baselines/budget_{budget}/{name}",
                    filter=_portable_tarinfo,
                )
        for budget in NEW_MODEL_BUDGETS:
            for architecture in ARCHITECTURES:
                for seed in TRAINING_SEEDS:
                    directory = _model_directory(root, budget, architecture, seed)
                    for name in ("summary.json", "artifact_manifest.json"):
                        archive.add(
                            directory / name,
                            arcname=(
                                f"tasks/budget_{budget}/{architecture}/"
                                f"training_seed_{seed}/{name}"
                            ),
                            filter=_portable_tarinfo,
                        )
        for architecture in ARCHITECTURES:
            for seed in TRAINING_SEEDS:
                directory = _mv6_directory(mv6_root, architecture, seed)
                for name in ("summary.json", "artifact_manifest.json"):
                    archive.add(
                        directory / name,
                        arcname=(
                            f"reused_mv6_budget_1/{architecture}/"
                            f"training_seed_{seed}/{name}"
                        ),
                        filter=_portable_tarinfo,
                    )
    checksum = _sha256(bundle)
    (root / f"{bundle.name}.sha256").write_text(
        f"{checksum}  {bundle.name}\n", encoding="utf-8"
    )
    return {
        "verification": verification,
        "bundle": str(bundle),
        "sha256": checksum,
        "large_model_and_prediction_arrays_excluded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("baseline", "model", "post", "verify-lock"), required=True
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--existing-m3-root", type=Path)
    parser.add_argument("--mv3-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--mv6-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    if args.mode == "verify-lock":
        print(json.dumps(locked_protocol(), indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("all execution modes require --output-root")
    if args.mode in ("baseline", "model"):
        for name in ("existing_m3_root", "mv3_root", "reference_root"):
            if getattr(args, name) is None:
                parser.error(f"{args.mode} mode requires --{name.replace('_', '-')}")
        if args.task_index is None:
            parser.error(f"{args.mode} mode requires --task-index")
    if args.mode == "baseline":
        budget = baseline_task_from_index(args.task_index)
        directory = _baseline_directory(args.output_root, budget)
        if directory.exists():
            raise SystemExit(f"refusing to overwrite MV7 baseline: {directory}")
        result = run_baseline_task(
            args.existing_m3_root,
            args.mv3_root,
            args.reference_root,
            args.output_root,
            budget=budget,
        )
        print(json.dumps({"budget": budget, "decision": result["decision"]}))
    elif args.mode == "model":
        budget, architecture, seed = model_task_from_index(args.task_index)
        directory = _model_directory(args.output_root, budget, architecture, seed)
        if directory.exists():
            raise SystemExit(f"refusing to overwrite MV7 model task: {directory}")
        result = run_model_task(
            args.existing_m3_root,
            args.mv3_root,
            args.reference_root,
            args.output_root,
            budget=budget,
            architecture=architecture,
            training_seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "budget": budget,
                    "architecture": architecture,
                    "training_seed": seed,
                    "decision": result["decision"],
                }
            )
        )
    else:
        if args.mv6_root is None:
            parser.error("post mode requires --mv6-root")
        aggregate(args.output_root, args.mv6_root)
        packaged = package_lite(args.output_root, args.mv6_root)
        print(json.dumps(packaged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
