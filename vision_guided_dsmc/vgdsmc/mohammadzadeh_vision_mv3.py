"""Condition-held-out MV3 benchmark for the Mohammadzadeh cavity.

The test condition is absent from model fitting and hyperparameter selection.
Only T and u are reconstructed; heat flux is excluded from data and claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping

import numpy as np

from .mohammadzadeh_mv3_reference import (
    LOCK_FILE,
    LOCK_STATUS,
    PROTOCOL_FILE,
    SEED_FILE,
    expected_lock_hashes,
    load_protocol,
    new_reference_tasks,
)
from .mohammadzadeh_validation import reference_directory
from .mohammadzadeh_vision import INPUT_FIELDS, OUTPUT_FIELDS, fit_scaling, nrmse
from .mohammadzadeh_vision_mv2 import (
    BUDGETS,
    GAUSSIAN_PASSES,
    TSVD_RANKS,
    _atomic_json,
    _portable_tarinfo,
    _sha256,
    gaussian_like,
    group_blocks,
    predict,
    select_baseline,
    train_model,
    tsvd,
)


STAGE = "MV3_Mohammadzadeh_condition_heldout_benchmark"
METHODS = ("raw", "gaussian_like", "tsvd_pod_type", "vision")
MODEL_INPUT_FIELDS = INPUT_FIELDS + ("log10_Kn", "U_lid_over_100")


def protocol_path() -> Path:
    return reference_directory() / PROTOCOL_FILE


def lock_path() -> Path:
    return reference_directory() / LOCK_FILE


def locked_protocol() -> dict[str, Any]:
    protocol = load_protocol()
    if tuple(protocol["model_contract"]["budget_blocks"]) != BUDGETS:
        raise ValueError("MV3 budgets differ from the code contract")
    if tuple(protocol["model_contract"]["input_fields"]) != MODEL_INPUT_FIELDS:
        raise ValueError("MV3 input fields differ from the code contract")
    if tuple(protocol["model_contract"]["output_fields"]) != OUTPUT_FIELDS:
        raise ValueError("MV3 output fields differ from the code contract")
    lock = json.loads(lock_path().read_text(encoding="utf-8"))
    if lock.get("status") != LOCK_STATUS or lock.get("hashes") != expected_lock_hashes():
        raise ValueError("MV3 source lock is missing or stale")
    return protocol


def conditions(protocol: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    value = locked_protocol() if protocol is None else protocol
    return tuple(dict(item) for item in value["conditions"])


def task_index(fold_index: int, budget: int) -> int:
    if budget not in BUDGETS or not 0 <= fold_index < 4:
        raise ValueError("invalid MV3 fold/budget")
    return fold_index * len(BUDGETS) + BUDGETS.index(budget)


def task_from_index(index: int) -> tuple[int, int]:
    if not 0 <= index < 4 * len(BUDGETS):
        raise ValueError("MV3 task index is outside the locked array")
    return index // len(BUDGETS), BUDGETS[index % len(BUDGETS)]


def fold_split(
    fold_index: int, protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    specs = conditions(locked_protocol() if protocol is None else protocol)
    if not 0 <= fold_index < len(specs):
        raise ValueError("invalid MV3 fold")
    heldout = specs[fold_index]
    training: dict[str, tuple[int, ...]] = {}
    validation: dict[str, tuple[int, ...]] = {}
    for condition_index, condition in enumerate(specs):
        if condition_index == fold_index:
            continue
        seeds = tuple(int(seed) for seed in condition["development_seeds"])
        validation_seed = seeds[(fold_index + condition_index) % len(seeds)]
        validation[condition["id"]] = (validation_seed,)
        training[condition["id"]] = tuple(seed for seed in seeds if seed != validation_seed)
    return {
        "heldout_condition": str(heldout["id"]),
        "test": {str(heldout["id"]): tuple(int(seed) for seed in heldout["evaluation_seeds"])},
        "train": training,
        "validation": validation,
    }


def _source_directory(
    condition: Mapping[str, Any], seed: int, existing_m3_root: Path, reference_root: Path
) -> Path:
    if condition["source"] == "existing_M3_QY100":
        return existing_m3_root / f"seed_{seed}"
    return reference_root / "references" / str(condition["id"]) / f"seed_{seed}"


def _source_summary_passes(
    summary: Mapping[str, Any], expected_status: str
) -> bool:
    """Apply the acceptance contract of the source stage that made the data.

    M3 intentionally deferred its per-seed stationarity decision to the locked
    eight-seed aggregation.  Its producer therefore excludes
    ``stationarity_pass`` from the per-seed mechanical decision.  MV3 reference
    seeds, in contrast, require every mechanical/stationarity check.  Keeping
    these contracts separate prevents a valid, immutable M3 source from being
    reinterpreted by the newer MV3 gate.
    """
    if summary.get("status") != expected_status:
        return False
    checks = summary.get("mechanical_checks")
    if not isinstance(checks, Mapping) or not checks:
        return False
    if expected_status == "complete_M3_qy_precision_seed":
        relevant = [value for key, value in checks.items() if key != "stationarity_pass"]
        return (
            bool(relevant)
            and all(bool(value) for value in relevant)
            and summary.get("decision")
            == "complete_M3_seed_awaiting_eight_seed_aggregation"
        )
    if expected_status == "complete_MV3_reference_seed":
        return all(bool(value) for value in checks.values()) and summary.get(
            "decision"
        ) == "accept_MV3_reference_seed"
    return False


def _verify_source(
    directory: Path, expected_status: str, *, all_manifest_files: bool = False
) -> None:
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = (
        tuple(manifest.get("files", {}))
        if all_manifest_files
        else ("fields.npz", "block_fields.npz", "summary.json")
    )
    for name in names:
        record = manifest.get("files", {}).get(name)
        path = directory / name
        if (
            not record
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV3 source artifact verification failed: {path}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not _source_summary_passes(summary, expected_status):
        raise ValueError(f"MV3 source did not pass its locked source-stage gate: {directory}")


def load_condition_data(
    existing_m3_root: Path, reference_root: Path, protocol: Mapping[str, Any] | None = None
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    specs = conditions(locked_protocol() if protocol is None else protocol)
    blocks: dict[str, dict[int, np.ndarray]] = {}
    full: dict[str, dict[int, np.ndarray]] = {}
    shape: tuple[int, int] | None = None
    for condition in specs:
        condition_id = str(condition["id"])
        blocks[condition_id], full[condition_id] = {}, {}
        for seed in condition["evaluation_seeds"]:
            seed = int(seed)
            directory = _source_directory(condition, seed, existing_m3_root, reference_root)
            expected = (
                "complete_M3_qy_precision_seed"
                if condition["source"] == "existing_M3_QY100"
                else "complete_MV3_reference_seed"
            )
            _verify_source(directory, expected)
            with np.load(directory / "block_fields.npz", allow_pickle=False) as data:
                missing = set(INPUT_FIELDS) - set(data.files)
                if missing:
                    raise ValueError(f"MV3 blocks missing {sorted(missing)}: {directory}")
                image = np.stack(
                    [np.asarray(data[name], dtype=np.float32) for name in INPUT_FIELDS], axis=1
                )
            with np.load(directory / "fields.npz", allow_pickle=False) as data:
                missing = set(OUTPUT_FIELDS) - set(data.files)
                if missing:
                    raise ValueError(f"MV3 fields missing {sorted(missing)}: {directory}")
                target = np.stack(
                    [np.asarray(data[name], dtype=np.float32) for name in OUTPUT_FIELDS], axis=0
                )
            if image.ndim != 4 or target.ndim != 3 or image.shape[-2:] != target.shape[-2:]:
                raise ValueError(f"invalid MV3 source array contract: {directory}")
            if shape is None:
                shape = target.shape[-2:]
            if target.shape[-2:] != shape or not np.all(np.isfinite(image)) or not np.all(np.isfinite(target)):
                raise ValueError(f"inconsistent/non-finite MV3 source: {directory}")
            blocks[condition_id][seed] = image
            full[condition_id][seed] = target
    return blocks, full


def fold_targets(
    full: Mapping[str, Mapping[int, np.ndarray]], split: Mapping[str, Any]
) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    for condition_id, train_seeds in split["train"].items():
        result[condition_id] = {}
        training_mean = np.mean(
            [np.asarray(full[condition_id][seed], dtype=np.float64) for seed in train_seeds],
            axis=0,
        ).astype(np.float32)
        for seed in train_seeds:
            result[condition_id][seed] = np.mean(
                [
                    np.asarray(full[condition_id][other], dtype=np.float64)
                    for other in train_seeds
                    if other != seed
                ],
                axis=0,
            ).astype(np.float32)
        for seed in split["validation"][condition_id]:
            result[condition_id][seed] = training_mean.copy()
    heldout = split["heldout_condition"]
    test_seeds = split["test"][heldout]
    result[heldout] = {}
    for seed in test_seeds:
        result[heldout][seed] = np.mean(
            [
                np.asarray(full[heldout][other], dtype=np.float64)
                for other in test_seeds
                if other != seed
            ],
            axis=0,
        ).astype(np.float32)
    return result


def _condition_map(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in protocol["conditions"]}


def build_budget_arrays(
    blocks: Mapping[str, Mapping[int, np.ndarray]],
    targets: Mapping[str, Mapping[int, np.ndarray]],
    selection: Mapping[str, tuple[int, ...]],
    specs: Mapping[str, Mapping[str, Any]],
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, condition_identity, numeric_identity = [], [], [], []
    for condition_id, seeds in selection.items():
        condition = specs[condition_id]
        for seed in seeds:
            grouped = group_blocks(blocks[condition_id][seed], budget)
            for group_index, image in enumerate(grouped):
                ny, nx = image.shape[-2:]
                conditioned = np.concatenate(
                    (
                        image,
                        np.full((1, ny, nx), np.log10(float(condition["knudsen"])), dtype=np.float32),
                        np.full((1, ny, nx), float(condition["lid_speed_m_per_s"]) / 100.0, dtype=np.float32),
                    ),
                    axis=0,
                )
                x.append(conditioned)
                y.append(targets[condition_id][seed])
                condition_identity.append(condition_id)
                numeric_identity.append((seed, group_index, budget))
    return (
        np.stack(x),
        np.stack(y),
        np.asarray(condition_identity, dtype="U32"),
        np.asarray(numeric_identity, dtype=np.int64),
    )


def evaluate_fields(raw: np.ndarray, candidate: np.ndarray, target: np.ndarray, lid_speed: float) -> dict[str, Any]:
    per_field: dict[str, Any] = {}
    for index, name in enumerate(OUTPUT_FIELDS):
        raw_error = nrmse(raw[:, index], target[:, index])
        candidate_error = nrmse(candidate[:, index], target[:, index])
        per_field[name] = {
            "raw_nrmse": raw_error,
            "vision_nrmse": candidate_error,
            "ratio": candidate_error / max(raw_error, 1.0e-12),
        }
    raw_composite = float(np.mean([value["raw_nrmse"] for value in per_field.values()]))
    candidate_composite = float(np.mean([value["vision_nrmse"] for value in per_field.values()]))
    ix = int(round(0.8 * raw.shape[-1] - 0.5))
    profiles = {
        "macroscopic_lid_temperature": (raw[:, 0, -1], candidate[:, 0, -1], target[:, 0, -1]),
        "vertical_temperature_x08": (raw[:, 0, :, ix], candidate[:, 0, :, ix], target[:, 0, :, ix]),
        "macroscopic_lid_slip": (
            1.0 - raw[:, 1, -1] / lid_speed,
            1.0 - candidate[:, 1, -1] / lid_speed,
            1.0 - target[:, 1, -1] / lid_speed,
        ),
    }
    profile_metrics = {}
    for name, (baseline, corrected, reference) in profiles.items():
        raw_error = nrmse(baseline, reference)
        candidate_error = nrmse(corrected, reference)
        profile_metrics[name] = {
            "raw_nrmse": raw_error,
            "vision_nrmse": candidate_error,
            "ratio": candidate_error / max(raw_error, 1.0e-12),
        }
    return {
        "per_field": per_field,
        "validated_profiles": profile_metrics,
        "raw_composite_nrmse": raw_composite,
        "vision_composite_nrmse": candidate_composite,
        "vision_over_raw_composite": candidate_composite / max(raw_composite, 1.0e-12),
    }


def select_residual_gate(
    raw: np.ndarray,
    ungated: np.ndarray,
    target: np.ndarray,
    lid_speeds: np.ndarray,
    candidates: tuple[float, ...],
) -> tuple[float, list[dict[str, float]]]:
    records = []
    for alpha in candidates:
        corrected = raw + float(alpha) * (ungated - raw)
        scores = []
        for speed in np.unique(lid_speeds):
            mask = lid_speeds == speed
            scores.append(evaluate_fields(raw[mask], corrected[mask], target[mask], float(speed))["vision_composite_nrmse"])
        records.append({"alpha": float(alpha), "validation_composite_nrmse": float(np.mean(scores))})
    best = min(records, key=lambda item: (item["validation_composite_nrmse"], item["alpha"]))
    return float(best["alpha"]), records


def _article_rows(filename: str, knudsen: float, value: str) -> tuple[np.ndarray, np.ndarray]:
    with (reference_directory() / filename).open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get("sampling", "macroscopic") == "macroscopic"
            and ("kn" not in row or np.isclose(float(row["kn"]), knudsen, atol=1.0e-12, rtol=0.0))
        ]
    coordinate = "x_over_l" if rows and "x_over_l" in rows[0] else "y_over_l"
    return np.asarray([float(row[coordinate]) for row in rows]), np.asarray([float(row[value]) for row in rows])


def article_profile_metrics(
    condition: Mapping[str, Any], raw: np.ndarray, vision: np.ndarray
) -> dict[str, Any]:
    if not np.isclose(float(condition["lid_speed_m_per_s"]), 100.0):
        return {"status": "not_available_for_this_lid_speed", "metrics": {}}
    knudsen = float(condition["knudsen"])
    x = (np.arange(raw.shape[-1]) + 0.5) / raw.shape[-1]
    metrics: dict[str, Any] = {}
    for name, filename, value_name, raw_profile, vision_profile in (
        ("lid_slip", "fig4_wall_slip_profiles.csv", "u_slip_over_uwall", 1.0 - raw[:, 1, -1] / 100.0, 1.0 - vision[:, 1, -1] / 100.0),
        ("lid_temperature", "fig5_wall_temperature_profiles.csv", "temperature_K", raw[:, 0, -1], vision[:, 0, -1]),
    ):
        coordinate, reference = _article_rows(filename, knudsen, value_name)
        if len(reference):
            raw_interp = np.stack([np.interp(coordinate, x, item) for item in raw_profile])
            vision_interp = np.stack([np.interp(coordinate, x, item) for item in vision_profile])
            metrics[name] = {
                "raw_nrmse": nrmse(raw_interp, np.broadcast_to(reference, raw_interp.shape)),
                "vision_nrmse": nrmse(vision_interp, np.broadcast_to(reference, vision_interp.shape)),
            }
    if condition["id"] == "kn0p05_u100":
        y = (np.arange(raw.shape[-2]) + 0.5) / raw.shape[-2]
        coordinate, reference = _article_rows("fig9b_dsmc_temperature_profile_x08.csv", knudsen, "temperature_K")
        ix = int(round(0.8 * raw.shape[-1] - 0.5))
        raw_interp = np.stack([np.interp(coordinate, y, item[0, :, ix]) for item in raw])
        vision_interp = np.stack([np.interp(coordinate, y, item[0, :, ix]) for item in vision])
        metrics["vertical_temperature_x08"] = {
            "raw_nrmse": nrmse(raw_interp, np.broadcast_to(reference, raw_interp.shape)),
            "vision_nrmse": nrmse(vision_interp, np.broadcast_to(reference, vision_interp.shape)),
        }
    return {"status": "reported_not_used_for_tuning", "metrics": metrics}


def _task_directory(root: Path, fold: int, budget: int) -> Path:
    return root / "tasks" / f"fold_{fold}" / f"budget_{budget}"


def run_task(
    existing_m3_root: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    fold_index: int,
    budget: int,
    epochs: int,
    batch_size: int,
    training_seed: int,
) -> dict[str, Any]:
    protocol = locked_protocol()
    specs = _condition_map(protocol)
    split = fold_split(fold_index, protocol)
    blocks, full = load_condition_data(existing_m3_root, reference_root, protocol)
    targets = fold_targets(full, split)
    train_x, train_y, train_conditions, train_identity = build_budget_arrays(blocks, targets, split["train"], specs, budget)
    validation_x, validation_y, validation_conditions, validation_identity = build_budget_arrays(blocks, targets, split["validation"], specs, budget)
    test_x, test_y, test_conditions, test_identity = build_budget_arrays(blocks, targets, split["test"], specs, budget)
    scaling = fit_scaling(train_x, train_y)
    model, training = train_model(
        train_x, train_y, validation_x, validation_y, scaling,
        epochs=epochs, batch_size=batch_size,
        seed=training_seed + task_index(fold_index, budget),
    )
    validation_ungated, validation_seconds = predict(model, validation_x, scaling, batch_size)
    test_ungated, inference_seconds = predict(model, test_x, scaling, batch_size)
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    validation_speeds = np.asarray([float(specs[item]["lid_speed_m_per_s"]) for item in validation_conditions])
    alpha, alpha_selection = select_residual_gate(
        validation_raw, validation_ungated, validation_y, validation_speeds,
        tuple(float(item) for item in protocol["model_contract"]["residual_gate_candidates"]),
    )
    vision = raw + alpha * (test_ungated - raw)
    gaussian_passes, gaussian_selection = select_baseline(validation_raw, validation_y, GAUSSIAN_PASSES, gaussian_like)
    tsvd_rank, tsvd_selection = select_baseline(validation_raw, validation_y, TSVD_RANKS, tsvd)
    gaussian = gaussian_like(raw, gaussian_passes)
    pod_type = tsvd(raw, tsvd_rank)
    heldout = specs[split["heldout_condition"]]
    lid_speed = float(heldout["lid_speed_m_per_s"])
    candidates = {"raw": raw, "gaussian_like": gaussian, "tsvd_pod_type": pod_type, "vision": vision}
    methods = {name: evaluate_fields(raw, value, test_y, lid_speed) for name, value in candidates.items()}
    per_seed = {}
    for seed in split["test"][split["heldout_condition"]]:
        mask = test_identity[:, 0] == seed
        per_seed[str(seed)] = {
            name: evaluate_fields(raw[mask], value[mask], test_y[mask], lid_speed)
            for name, value in candidates.items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(
        {
            "stage": STAGE,
            "state_dict": model.state_dict(),
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "fold_index": fold_index,
            "budget": budget,
            "split": split,
            "residual_gate_alpha": alpha,
            "input_fields": MODEL_INPUT_FIELDS,
            "output_fields": OUTPUT_FIELDS,
        },
        output_dir / "model.pt",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        identity_condition=test_conditions,
        identity_numeric=test_identity,
        raw=raw,
        gaussian_like=gaussian,
        tsvd_pod_type=pod_type,
        vision=vision,
        vision_ungated=test_ungated,
        target=test_y,
    )
    checks = {
        "condition_disjoint": split["heldout_condition"] not in split["train"] and split["heldout_condition"] not in split["validation"],
        "seed_disjoint_within_training_conditions": all(not (set(split["train"][key]) & set(split["validation"][key])) for key in split["train"]),
        "heat_flux_excluded": not any(name.lower().startswith("q") for name in MODEL_INPUT_FIELDS + OUTPUT_FIELDS),
        "finite_metrics": all(np.isfinite(value["vision_composite_nrmse"]) for value in methods.values()),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV3_task",
        "protocol_sha256": _sha256(protocol_path()),
        "lock_sha256": _sha256(lock_path()),
        "fold_index": fold_index,
        "heldout_condition": split["heldout_condition"],
        "budget_blocks": budget,
        "effective_DSMC_samples": int(budget * protocol["model_contract"]["samples_per_block"]),
        "split": {section: {key: list(value) for key, value in split[section].items()} for section in ("train", "validation", "test")},
        "sample_counts": {"train": len(train_x), "validation": len(validation_x), "test": len(test_x)},
        "target_contract": "same-condition cross-fit; test condition absent from training and tuning; every test seed excluded from its own target",
        "baseline_selection": {
            "gaussian_like": {"selected_passes": gaussian_passes, "candidates": gaussian_selection},
            "tsvd_pod_type": {"selected_rank": tsvd_rank, "candidates": tsvd_selection},
            "residual_gate": {"selected_alpha": alpha, "candidates": alpha_selection},
        },
        "training": training,
        "validation_inference_seconds": validation_seconds,
        "test_inference_seconds": inference_seconds,
        "methods": methods,
        "per_seed_metrics": per_seed,
        "article_profiles": article_profile_metrics(heldout, raw, vision),
        "checks": checks,
        "decision": "accept_MV3_task" if all(checks.values()) else "hold_MV3_task",
    }
    _atomic_json(output_dir / "summary.json", summary)
    manifest = {
        "stage": STAGE,
        "files": {
            name: {"sha256": _sha256(output_dir / name), "size_bytes": (output_dir / name).stat().st_size}
            for name in ("model.pt", "predictions.npz", "summary.json")
        },
    }
    _atomic_json(output_dir / "artifact_manifest.json", manifest)
    return summary


def _records(root: Path) -> list[dict[str, Any]]:
    result = []
    for fold in range(4):
        for budget in BUDGETS:
            path = _task_directory(root, fold, budget) / "summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "complete_MV3_task" or value.get("fold_index") != fold or value.get("budget_blocks") != budget:
                raise ValueError(f"invalid MV3 task summary: {path}")
            result.append(value)
    return result


def _curve_figure(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    colors = {"raw": "0.35", "gaussian_like": "#2ca02c", "tsvd_pod_type": "#ff7f0e", "vision": "#1f77b4"}
    labels = {"raw": "Raw", "gaussian_like": "Spatial filter", "tsvd_pod_type": "TSVD/POD-type", "vision": "Conditioned U-Net"}
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), constrained_layout=True)
    for fold, axis in enumerate(axes.ravel()):
        selected = [item for item in records if item["fold_index"] == fold]
        for method in METHODS:
            axis.plot(BUDGETS, [next(item for item in selected if item["budget_blocks"] == budget)["methods"][method]["vision_composite_nrmse"] for budget in BUDGETS], marker="o", linewidth=2, color=colors[method], label=labels[method])
        axis.set(title=selected[0]["heldout_condition"], xlabel="Temporal blocks averaged", ylabel="Held-out composite NRMSE", xscale="log", yscale="log")
        axis.set_xticks(BUDGETS, labels=[str(item) for item in BUDGETS])
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("MV3 condition-held-out sampling-budget comparison")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _contour_error_figure(root: Path, field: int, label: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 5, figsize=(14.2, 11.0), constrained_layout=True)
    for fold in range(4):
        with np.load(_task_directory(root, fold, 1) / "predictions.npz", allow_pickle=False) as data:
            raw, vision, target = (np.asarray(data[name])[0, field] for name in ("raw", "vision", "target"))
        low, high = min(raw.min(), vision.min(), target.min()), max(raw.max(), vision.max(), target.max())
        levels = np.linspace(low, high, 25)
        error_high = max(np.abs(raw - target).max(), np.abs(vision - target).max(), 1.0e-12)
        for column, value in enumerate((raw, vision, target)):
            plot = axes[fold, column].contourf(value, levels=levels, cmap="coolwarm", extend="both")
        for column, value in enumerate((np.abs(raw - target), np.abs(vision - target)), start=3):
            error_plot = axes[fold, column].contourf(value, levels=np.linspace(0, error_high, 25), cmap="magma", extend="max")
        axes[fold, 0].set_ylabel(f"fold {fold}")
        figure.colorbar(plot, ax=axes[fold, :3], label=label, shrink=0.7)
        figure.colorbar(error_plot, ax=axes[fold, 3:], label=f"absolute error ({label})", shrink=0.7)
    for axis, title in zip(axes[0], ("Raw", "Vision", "Cross-fit target", "Raw error", "Vision error")):
        axis.set_title(title)
    for axis in axes.ravel():
        axis.set_aspect("equal"); axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(f"MV3 unsmoothed one-block {label} contours")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _profile_figure(root: Path, protocol: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    specs = _condition_map(protocol)
    figure, axes = plt.subplots(4, 3, figsize=(12.0, 13.0), constrained_layout=True)
    for fold in range(4):
        record = json.loads((_task_directory(root, fold, 1) / "summary.json").read_text(encoding="utf-8"))
        condition = specs[record["heldout_condition"]]
        speed = float(condition["lid_speed_m_per_s"])
        with np.load(_task_directory(root, fold, 1) / "predictions.npz", allow_pickle=False) as data:
            values = {name: np.asarray(data[name])[0] for name in ("raw", "vision", "target")}
        ny, nx = values["raw"].shape[-2:]
        x, y = (np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny
        ix = int(round(0.8 * nx - 0.5))
        for name, style, color in (("raw", ":", "0.4"), ("vision", "-", "#1f77b4"), ("target", "--", "black")):
            axes[fold, 0].plot(x, 1.0 - values[name][1, -1] / speed, style, color=color, label=name)
            axes[fold, 1].plot(x, values[name][0, -1], style, color=color)
            axes[fold, 2].plot(y, values[name][0, :, ix], style, color=color)
        if np.isclose(speed, 100.0):
            for column, filename, value_name in ((0, "fig4_wall_slip_profiles.csv", "u_slip_over_uwall"), (1, "fig5_wall_temperature_profiles.csv", "temperature_K")):
                coordinate, reference = _article_rows(filename, float(condition["knudsen"]), value_name)
                if len(reference): axes[fold, column].plot(coordinate, reference, "o", ms=2.5, color="#d62728", label="PRE digitization")
            if condition["id"] == "kn0p05_u100":
                coordinate, reference = _article_rows("fig9b_dsmc_temperature_profile_x08.csv", .05, "temperature_K")
                axes[fold, 2].plot(coordinate, reference, "o", ms=2.5, color="#d62728")
        axes[fold, 0].set_ylabel(str(condition["id"]))
    for column, title in enumerate(("Lid slip", "Lid temperature", "T at x/L=0.8")):
        axes[0, column].set_title(title)
    for axis in axes.ravel(): axis.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle("MV3 one-block physical profiles (article points where available)")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _aggregate_values(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_condition: dict[str, Any] = {}
    for record in records:
        condition = record["heldout_condition"]
        budget = str(record["budget_blocks"])
        by_condition.setdefault(condition, {})[budget] = {
            method: record["methods"][method] for method in METHODS
        }
    by_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        selected = [record for record in records if record["budget_blocks"] == budget]
        by_budget[str(budget)] = {}
        for method in METHODS:
            values = np.asarray([record["methods"][method]["vision_composite_nrmse"] for record in selected])
            raw = np.asarray([record["methods"]["raw"]["vision_composite_nrmse"] for record in selected])
            by_budget[str(budget)][method] = {
                "mean_composite_nrmse": float(np.mean(values)),
                "median_composite_nrmse": float(np.median(values)),
                "mean_over_raw": float(np.mean(values / raw)),
                "heldout_condition_count": len(values),
            }
    return by_condition, by_budget


def aggregate(root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    records = _records(root)
    by_condition, by_budget = _aggregate_values(records)
    budget_one = [item for item in records if item["budget_blocks"] == 1]
    high_budget = [item for item in records if item["budget_blocks"] == 10]
    gates = protocol["gates"]
    checks = {
        "all_16_model_tasks_complete": len(records) == 16,
        "all_12_new_reference_tasks_mechanical_and_stationary": all(
            json.loads((root / "references" / condition / f"seed_{seed}" / "summary.json").read_text(encoding="utf-8")).get("decision") == "accept_MV3_reference_seed"
            for condition, seed in new_reference_tasks()
        ),
        "heat_flux_excluded": not any(name.lower().startswith("q") for name in MODEL_INPUT_FIELDS + OUTPUT_FIELDS),
        "condition_disjoint_every_fold": all(item["checks"]["condition_disjoint"] for item in records),
        "vision_beats_raw_at_budget_1_all_conditions": all(item["methods"]["vision"]["vision_composite_nrmse"] < item["methods"]["raw"]["vision_composite_nrmse"] for item in budget_one),
        "vision_beats_both_selected_baselines_at_budget_1_min_conditions": sum(item["methods"]["vision"]["vision_composite_nrmse"] < min(item["methods"]["gaussian_like"]["vision_composite_nrmse"], item["methods"]["tsvd_pod_type"]["vision_composite_nrmse"]) for item in budget_one) >= int(gates["vision_beats_both_selected_baselines_at_budget_1_min_conditions"]),
        "maximum_high_budget_field_degradation_over_raw": all(item["methods"]["vision"]["per_field"][field]["vision_nrmse"] <= (1.0 + float(gates["maximum_high_budget_field_degradation_over_raw"])) * item["methods"]["raw"]["per_field"][field]["vision_nrmse"] for item in high_budget for field in OUTPUT_FIELDS),
    }
    root.mkdir(parents=True, exist_ok=True)
    _curve_figure(records, root / "sampling_budget_curves_by_condition.png")
    _contour_error_figure(root, 0, "T (K)", root / "heldout_temperature_contours_and_errors.png")
    _contour_error_figure(root, 1, "u (m/s)", root / "heldout_velocity_contours_and_errors.png")
    _profile_figure(root, protocol, root / "heldout_physical_profiles.png")
    summary = {
        "stage": STAGE,
        "status": "complete_MV3_aggregate",
        "scope": "four-condition leave-one-condition-out reconstruction benchmark for T and u; heat flux excluded",
        "protocol_sha256": _sha256(protocol_path()),
        "lock_sha256": _sha256(lock_path()),
        "task_count": len(records),
        "new_reference_count": len(new_reference_tasks()),
        "by_condition": by_condition,
        "by_budget": by_budget,
        "checks": checks,
        "decision": "MV3_cross_condition_evidence_pass" if all(checks.values()) else "hold_for_MV3_diagnosis",
    }
    _atomic_json(root / "summary.json", summary)
    names = ("summary.json", "sampling_budget_curves_by_condition.png", "heldout_temperature_contours_and_errors.png", "heldout_velocity_contours_and_errors.png", "heldout_physical_profiles.png")
    _atomic_json(root / "artifact_manifest.json", {"stage": STAGE, "files": {name: {"sha256": _sha256(root / name), "size_bytes": (root / name).stat().st_size} for name in names}})
    return summary


def verify(root: Path) -> dict[str, Any]:
    top_manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in top_manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"MV3 top artifact verification failed: {path}")
    for condition_id, seed in new_reference_tasks():
        _verify_source(
            root / "references" / condition_id / f"seed_{seed}",
            "complete_MV3_reference_seed",
            all_manifest_files=True,
        )
    records = []
    maximum_difference = 0.0
    verified_task_files = 0
    protocol = locked_protocol()
    specs = _condition_map(protocol)
    for fold in range(4):
        for budget in BUDGETS:
            directory = _task_directory(root, fold, budget)
            manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
            for name, record in manifest["files"].items():
                path = directory / name
                if not path.is_file() or path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
                    raise ValueError(f"MV3 task artifact verification failed: {path}")
                verified_task_files += 1
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            expected_split = fold_split(fold, protocol)
            if summary["heldout_condition"] != expected_split["heldout_condition"] or summary["budget_blocks"] != budget or not summary["checks"]["condition_disjoint"]:
                raise ValueError(f"MV3 split/task identity mismatch: {directory}")
            with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                raw, target = np.asarray(data["raw"]), np.asarray(data["target"])
                speed = float(specs[summary["heldout_condition"]]["lid_speed_m_per_s"])
                for method in METHODS:
                    rebuilt = evaluate_fields(raw, np.asarray(data[method]), target, speed)
                    recorded = summary["methods"][method]
                    maximum_difference = max(maximum_difference, abs(rebuilt["vision_composite_nrmse"] - recorded["vision_composite_nrmse"]))
                    for field in OUTPUT_FIELDS:
                        maximum_difference = max(maximum_difference, abs(rebuilt["per_field"][field]["vision_nrmse"] - recorded["per_field"][field]["vision_nrmse"]))
            records.append(summary)
    if maximum_difference > 2.0e-6:
        raise ValueError(f"MV3 metric reconstruction mismatch: {maximum_difference}")
    by_condition, by_budget = _aggregate_values(records)
    saved = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if by_condition != saved["by_condition"] or by_budget != saved["by_budget"]:
        raise ValueError("MV3 aggregate reconstruction mismatch")
    return {
        "status": "complete_MV3_artifacts_splits_and_metrics_verified",
        "decision": saved["decision"],
        "task_count": len(records),
        "new_reference_count": len(new_reference_tasks()),
        "verified_task_files": verified_task_files,
        "maximum_metric_reconstruction_difference": maximum_difference,
        "summary_sha256": _sha256(root / "summary.json"),
    }


def package(root: Path) -> dict[str, Any]:
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "complete_MV3_artifacts_splits_and_metrics_verified":
        raise ValueError("MV3 must pass the recursive verifier before packaging")
    bundle = root / "MOHAMMADZADEH_MV3_JCP_RETURN_BUNDLE.tar.gz"
    top_names = ("summary.json", "sampling_budget_curves_by_condition.png", "heldout_temperature_contours_and_errors.png", "heldout_velocity_contours_and_errors.png", "heldout_physical_profiles.png", "artifact_manifest.json", "verification.json")
    with tarfile.open(bundle, "w:gz") as archive:
        for name in top_names:
            archive.add(root / name, arcname=name, filter=_portable_tarinfo)
        for condition_id, seed in new_reference_tasks():
            directory = root / "references" / condition_id / f"seed_{seed}"
            for name in ("summary.json", "artifact_manifest.json"):
                archive.add(directory / name, arcname=f"references/{condition_id}/seed_{seed}/{name}", filter=_portable_tarinfo)
        for fold in range(4):
            for budget in BUDGETS:
                directory = _task_directory(root, fold, budget)
                for name in ("model.pt", "summary.json", "predictions.npz", "artifact_manifest.json"):
                    archive.add(directory / name, arcname=f"tasks/fold_{fold}/budget_{budget}/{name}", filter=_portable_tarinfo)
    checksum = _sha256(bundle)
    (root / f"{bundle.name}.sha256").write_text(f"{checksum}  {bundle.name}\n", encoding="utf-8")
    return {"status": "complete_MV3_portable_bundle", "bundle": str(bundle), "bundle_sha256": checksum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-m3-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--training-seed", type=int, default=20260808)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output_dir)
    elif args.package:
        result = package(args.output_dir)
    elif args.aggregate:
        result = aggregate(args.output_dir)
    else:
        if args.task_index is not None:
            fold, budget = task_from_index(args.task_index)
        elif args.fold_index is not None and args.budget is not None:
            fold, budget = args.fold_index, args.budget
        else:
            parser.error("task mode requires --task-index or --fold-index and --budget")
        if args.existing_m3_root is None or args.reference_root is None:
            parser.error("task mode requires --existing-m3-root and --reference-root")
        result = run_task(
            args.existing_m3_root, args.reference_root,
            _task_directory(args.output_dir, fold, budget),
            fold_index=fold, budget=budget, epochs=args.epochs,
            batch_size=args.batch_size, training_seed=args.training_seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
