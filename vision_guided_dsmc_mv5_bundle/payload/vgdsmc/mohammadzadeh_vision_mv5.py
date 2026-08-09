"""MV5 preregistered confirmatory selector for Mohammadzadeh field recovery.

MV5 is evaluated only on four new condition combinations.  The development
model uses the locked MV3 conditions and never sees an MV5 target.  A true
two-dimensional convex-hull gate permits bounded vision only for budget one
inside development support.  All other cases use a classical method selected
by condition-balanced validation risk with a maximum-degradation constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping

import numpy as np

from . import mohammadzadeh_vision_mv3 as mv3
from . import mohammadzadeh_vision_mv4 as mv4
from . import mohammadzadeh_mv5_reference as mv5ref
from .mohammadzadeh_vision import OUTPUT_FIELDS, fit_scaling
from .mohammadzadeh_vision_mv2 import (
    BUDGETS,
    GAUSSIAN_PASSES,
    TSVD_RANKS,
    _atomic_json,
    _portable_tarinfo,
    _sha256,
    gaussian_like,
    select_baseline,
    tsvd,
)


STAGE = "MV5_Mohammadzadeh_confirmatory_selector_benchmark"
METHODS = (
    "raw",
    "gaussian_like",
    "tsvd_pod_type",
    "vision_bounded",
    "mv5_selected",
)
CLASSICAL_METHODS = ("raw", "gaussian_like", "tsvd_pod_type")
VALIDATION_MAX_RATIO = 1.05
PROTOCOL_FILE = "mv5_confirmatory_protocol.json"


def protocol_path() -> Path:
    return mv5ref.protocol_path()


def locked_protocol() -> dict[str, Any]:
    value = mv5ref.locked_protocol()
    contract = value["selector_contract"]
    if (
        contract["support_rule"]
        != "two_dimensional_convex_hull_in_log10_Kn_and_U_lid_over_100_no_test_target"
        or float(
            value["model_contract"]["physical_envelope"][
                "absolute_u_max_over_lid_speed"
            ]
        )
        != mv4.PHYSICAL_U_LID_MULTIPLIER
        or float(value["model_contract"]["residual_cap_sigma"])
        != mv4.RESIDUAL_CAP_SIGMA
        or float(
            value["acceptance_gates"]["selected_maximum_over_raw"]
        )
        != VALIDATION_MAX_RATIO
    ):
        raise ValueError("MV5 protocol differs from implementation constants")
    if tuple(value["source_contract"]["budget_blocks"]) != BUDGETS:
        raise ValueError("MV5 budgets differ from code")
    return value


def _task_directory(root: Path, budget: int) -> Path:
    return root / "tasks" / f"budget_{budget}"


def task_from_index(index: int) -> int:
    if not 0 <= index < len(BUDGETS):
        raise ValueError("MV5 task index is outside the locked budget array")
    return BUDGETS[index]


def _descriptor(condition: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            np.log10(float(condition["knudsen"])),
            float(condition["lid_speed_m_per_s"]) / 100.0,
        ],
        dtype=np.float64,
    )


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise two-dimensional monotone-chain hull."""
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) < 3:
        raise ValueError("a two-dimensional support hull needs three unique points")
    ordered = sorted((float(x), float(y)) for x, y in unique)
    lower: list[np.ndarray] = []
    for point in map(np.asarray, ordered):
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in map(np.asarray, reversed(ordered)):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.stack(lower[:-1] + upper[:-1]).astype(np.float64)


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    denominator = float(np.dot(delta, delta))
    if denominator <= 1.0e-30:
        return float(np.linalg.norm(point - a))
    fraction = np.clip(float(np.dot(point - a, delta)) / denominator, 0.0, 1.0)
    return float(np.linalg.norm(point - (a + fraction * delta)))


def convex_hull_support_report(
    heldout: Mapping[str, Any], development: tuple[Mapping[str, Any], ...]
) -> dict[str, Any]:
    """Classify support in joint condition space without using a target."""
    cloud = np.stack([_descriptor(item) for item in development])
    point = _descriptor(heldout)
    lower, upper = cloud.min(axis=0), cloud.max(axis=0)
    scale = np.maximum(upper - lower, 1.0e-12)
    normalized_cloud = (cloud - lower) / scale
    normalized_point = (point - lower) / scale
    hull = convex_hull(normalized_cloud)
    tolerance = 1.0e-9
    signs = np.asarray(
        [
            _cross(hull[index], hull[(index + 1) % len(hull)], normalized_point)
            for index in range(len(hull))
        ]
    )
    inside = bool(np.all(signs >= -tolerance))
    distance = 0.0 if inside else min(
        _point_segment_distance(
            normalized_point, hull[index], hull[(index + 1) % len(hull)]
        )
        for index in range(len(hull))
    )
    return {
        "rule": "two_dimensional_convex_hull_in_log10_Kn_and_U_lid_over_100_no_test_target",
        "inside_development_hull": inside,
        "normalized_distance_to_hull": float(distance),
        "heldout_descriptor": point.astype(float).tolist(),
        "development_descriptors": cloud.astype(float).tolist(),
        "normalized_hull_vertices": hull.astype(float).tolist(),
    }


def _verify_reference(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name in ("fields.npz", "block_fields.npz", "summary.json"):
        record = manifest.get("files", {}).get(name)
        path = directory / name
        if (
            not record
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV5 reference artifact verification failed: {path}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != mv5ref.COMPLETE_STATUS:
        raise ValueError(f"MV5 reference is incomplete: {directory}")
    mechanics = summary.get("mechanical_checks", {})
    stationarity = summary.get("stationarity", {}).get("checks", {})
    relevant_stationarity = [
        bool(value)
        for key, value in stationarity.items()
        if not mv3._is_heat_flux_stationarity_key(str(key))
    ]
    mechanical_values = [
        bool(value) for key, value in mechanics.items() if key != "stationarity_pass"
    ]
    if (
        not mechanical_values
        or not all(mechanical_values)
        or not relevant_stationarity
        or not all(relevant_stationarity)
    ):
        raise ValueError(f"MV5 reference mechanical/T-u stationarity gate failed: {directory}")
    return summary


def load_confirmatory_data(
    reference_root: Path,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    specs = mv5ref.condition_map()
    blocks: dict[str, dict[int, np.ndarray]] = {}
    full: dict[str, dict[int, np.ndarray]] = {}
    shape: tuple[int, int] | None = None
    for condition_id, condition in specs.items():
        blocks[condition_id], full[condition_id] = {}, {}
        for seed_value in condition["evaluation_seeds"]:
            seed = int(seed_value)
            directory = reference_root / "references" / condition_id / f"seed_{seed}"
            _verify_reference(directory)
            with np.load(directory / "block_fields.npz", allow_pickle=False) as data:
                missing = set(mv3.INPUT_FIELDS) - set(data.files)
                if missing:
                    raise ValueError(f"MV5 blocks missing {sorted(missing)}: {directory}")
                image = np.stack(
                    [np.asarray(data[name], dtype=np.float32) for name in mv3.INPUT_FIELDS],
                    axis=1,
                )
            with np.load(directory / "fields.npz", allow_pickle=False) as data:
                missing = set(OUTPUT_FIELDS) - set(data.files)
                if missing:
                    raise ValueError(f"MV5 fields missing {sorted(missing)}: {directory}")
                target = np.stack(
                    [np.asarray(data[name], dtype=np.float32) for name in OUTPUT_FIELDS],
                    axis=0,
                )
            if image.ndim != 4 or target.ndim != 3 or image.shape[-2:] != target.shape[-2:]:
                raise ValueError(f"invalid MV5 source array contract: {directory}")
            if shape is None:
                shape = target.shape[-2:]
            if (
                target.shape[-2:] != shape
                or not np.all(np.isfinite(image))
                or not np.all(np.isfinite(target))
            ):
                raise ValueError(f"inconsistent/non-finite MV5 source: {directory}")
            blocks[condition_id][seed] = image
            full[condition_id][seed] = target
    return blocks, full


def _development_targets(
    full: Mapping[str, Mapping[int, np.ndarray]],
    train: Mapping[str, tuple[int, ...]],
    validation: Mapping[str, tuple[int, ...]],
) -> dict[str, dict[int, np.ndarray]]:
    targets: dict[str, dict[int, np.ndarray]] = {}
    for condition_id, train_seeds in train.items():
        if len(train_seeds) < 3:
            raise ValueError("MV5 requires three development training seeds per condition")
        values = full[condition_id]
        mean = np.mean([values[seed] for seed in train_seeds], axis=0).astype(np.float32)
        targets[condition_id] = {}
        for seed in train_seeds:
            targets[condition_id][seed] = np.mean(
                [values[other] for other in train_seeds if other != seed], axis=0
            ).astype(np.float32)
        for seed in validation[condition_id]:
            targets[condition_id][seed] = mean.copy()
    return targets


def _confirmatory_targets(
    full: Mapping[str, Mapping[int, np.ndarray]]
) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    for condition_id, values in full.items():
        seeds = tuple(values)
        if len(seeds) < 3:
            raise ValueError("MV5 confirmatory target requires at least three seeds")
        result[condition_id] = {
            seed: np.mean(
                [values[other] for other in seeds if other != seed], axis=0
            ).astype(np.float32)
            for seed in seeds
        }
    return result


def _metric_by_condition(
    raw: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(condition_id): mv3.evaluate_fields(
            raw[labels == condition_id],
            candidate[labels == condition_id],
            target[labels == condition_id],
            float(specs[str(condition_id)]["lid_speed_m_per_s"]),
        )
        for condition_id in np.unique(labels)
    }


def select_risk_controlled_classical(
    raw: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    specs: Mapping[str, Mapping[str, Any]],
    gaussian_passes: int,
    tsvd_rank: int,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = {
        "raw": raw,
        "gaussian_like": gaussian_like(raw, gaussian_passes),
        "tsvd_pod_type": tsvd(raw, tsvd_rank),
    }
    records = []
    for method, value in candidates.items():
        metrics = _metric_by_condition(raw, value, target, labels, specs)
        ratios = [item["vision_over_raw_composite"] for item in metrics.values()]
        nrmse = [item["vision_composite_nrmse"] for item in metrics.values()]
        records.append(
            {
                "method": method,
                "validation_condition_mean_composite_nrmse": float(np.mean(nrmse)),
                "maximum_validation_condition_over_raw": float(np.max(ratios)),
                "admissible": bool(np.max(ratios) <= VALIDATION_MAX_RATIO),
                "per_condition": metrics,
            }
        )
    admissible = [item for item in records if item["admissible"]]
    if not admissible:
        raise RuntimeError("raw identity must always make the MV5 classical set feasible")
    best = min(
        admissible,
        key=lambda item: (
            item["validation_condition_mean_composite_nrmse"],
            CLASSICAL_METHODS.index(item["method"]),
        ),
    )
    return str(best["method"]), records


def _project_by_condition(
    candidate: np.ndarray,
    raw: np.ndarray,
    labels: np.ndarray,
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    projected = np.asarray(candidate, dtype=np.float32).copy()
    diagnostics: dict[str, Any] = {}
    for condition_id in np.unique(labels):
        mask = labels == condition_id
        projected[mask], diagnostics[str(condition_id)] = mv4.project_physical_fields(
            projected[mask],
            raw[mask],
            float(specs[str(condition_id)]["lid_speed_m_per_s"]),
        )
    return projected, diagnostics


def run_task(
    existing_m3_root: Path,
    mv3_root: Path,
    mv5_reference_root: Path,
    output_dir: Path,
    *,
    budget: int,
    epochs: int,
    batch_size: int,
    training_seed: int,
) -> dict[str, Any]:
    protocol = locked_protocol()
    mv3_protocol = mv3.locked_protocol()
    development_specs = mv3._condition_map(mv3_protocol)
    confirmatory_specs = mv5ref.condition_map(protocol)
    development_blocks, development_full = mv3.load_condition_data(
        existing_m3_root, mv3_root, mv3_protocol
    )
    confirmatory_blocks, confirmatory_full = load_confirmatory_data(mv5_reference_root)

    train = {
        key: tuple(int(seed) for seed in value)
        for key, value in protocol["development_seed_split"]["train"].items()
    }
    validation = {
        key: tuple(int(seed) for seed in value)
        for key, value in protocol["development_seed_split"]["validation"].items()
    }
    development_targets = _development_targets(
        development_full, train, validation
    )
    confirmatory_targets = _confirmatory_targets(confirmatory_full)
    test = {
        key: tuple(int(seed) for seed in value["evaluation_seeds"])
        for key, value in confirmatory_specs.items()
    }
    train_x, train_y, train_conditions, train_identity = mv3.build_budget_arrays(
        development_blocks, development_targets, train, development_specs, budget
    )
    validation_x, validation_y, validation_conditions, validation_identity = (
        mv3.build_budget_arrays(
            development_blocks,
            development_targets,
            validation,
            development_specs,
            budget,
        )
    )
    test_x, test_y, test_conditions, test_identity = mv3.build_budget_arrays(
        confirmatory_blocks,
        confirmatory_targets,
        test,
        confirmatory_specs,
        budget,
    )

    scaling = fit_scaling(train_x, train_y)
    model, training = mv4.train_bounded_model(
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        epochs=epochs,
        batch_size=batch_size,
        seed=training_seed + BUDGETS.index(budget),
    )
    validation_bounded, validation_seconds, validation_diagnostics = (
        mv4.predict_bounded(model, validation_x, scaling, batch_size)
    )
    test_bounded, inference_seconds, test_diagnostics = mv4.predict_bounded(
        model, test_x, scaling, batch_size
    )
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    validation_speeds = np.asarray(
        [
            float(development_specs[str(item)]["lid_speed_m_per_s"])
            for item in validation_conditions
        ]
    )
    alpha, alpha_records = mv3.select_residual_gate(
        validation_raw,
        validation_bounded,
        validation_y,
        validation_speeds,
        tuple(
            float(item)
            for item in protocol["model_contract"]["residual_alpha_candidates"]
        ),
    )
    validation_vision = validation_raw + float(alpha) * (
        validation_bounded - validation_raw
    )
    vision = raw + float(alpha) * (test_bounded - raw)

    gaussian_passes, gaussian_records = select_baseline(
        validation_raw, validation_y, GAUSSIAN_PASSES, gaussian_like
    )
    tsvd_rank, tsvd_records = select_baseline(
        validation_raw, validation_y, TSVD_RANKS, tsvd
    )
    classical_name, classical_records = select_risk_controlled_classical(
        validation_raw,
        validation_y,
        validation_conditions,
        development_specs,
        gaussian_passes,
        tsvd_rank,
    )
    gaussian = gaussian_like(raw, gaussian_passes)
    pod_type = tsvd(raw, tsvd_rank)
    gaussian, gaussian_projection = _project_by_condition(
        gaussian, raw, test_conditions, confirmatory_specs
    )
    pod_type, tsvd_projection = _project_by_condition(
        pod_type, raw, test_conditions, confirmatory_specs
    )
    vision, vision_projection = _project_by_condition(
        vision, raw, test_conditions, confirmatory_specs
    )
    candidate_arrays = {
        "raw": raw,
        "gaussian_like": gaussian,
        "tsvd_pod_type": pod_type,
        "vision_bounded": vision,
    }

    development_conditions = tuple(development_specs.values())
    supports = {
        condition_id: convex_hull_support_report(condition, development_conditions)
        for condition_id, condition in confirmatory_specs.items()
    }
    actions: dict[str, dict[str, Any]] = {}
    selected = np.empty_like(raw)
    for condition_id in confirmatory_specs:
        mask = test_conditions == condition_id
        use_vision = bool(
            budget == 1 and supports[condition_id]["inside_development_hull"]
        )
        method = "vision_bounded" if use_vision else classical_name
        selected[mask] = candidate_arrays[method][mask]
        actions[condition_id] = {
            "selected_method": method,
            "reason": "inside_hull_budget_1"
            if use_vision
            else "validation_risk_controlled_classical",
            "support": supports[condition_id],
        }
    selected, selected_projection = _project_by_condition(
        selected, raw, test_conditions, confirmatory_specs
    )
    candidates = {**candidate_arrays, "mv5_selected": selected}
    methods_by_condition = {
        method: _metric_by_condition(
            raw, value, test_y, test_conditions, confirmatory_specs
        )
        for method, value in candidates.items()
    }
    aggregate_methods = {
        method: {
            "mean_over_raw": float(
                np.mean(
                    [
                        metric["vision_over_raw_composite"]
                        for metric in by_condition.values()
                    ]
                )
            ),
            "maximum_over_raw": float(
                np.max(
                    [
                        metric["vision_over_raw_composite"]
                        for metric in by_condition.values()
                    ]
                )
            ),
            "mean_composite_nrmse": float(
                np.mean(
                    [
                        metric["vision_composite_nrmse"]
                        for metric in by_condition.values()
                    ]
                )
            ),
        }
        for method, by_condition in methods_by_condition.items()
    }

    per_seed: dict[str, Any] = {}
    for condition_id, condition in confirmatory_specs.items():
        per_seed[condition_id] = {}
        speed = float(condition["lid_speed_m_per_s"])
        for seed in test[condition_id]:
            mask = (test_conditions == condition_id) & (test_identity[:, 0] == seed)
            per_seed[condition_id][str(seed)] = {
                method: mv3.evaluate_fields(
                    raw[mask], value[mask], test_y[mask], speed
                )
                for method, value in candidates.items()
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "stage": STAGE,
            "state_dict": model.state_dict(),
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "budget": budget,
            "residual_gate_alpha": alpha,
            "classical_fallback": classical_name,
            "selector_actions": actions,
            "input_fields": mv3.MODEL_INPUT_FIELDS,
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
        vision_bounded=vision,
        mv5_selected=selected,
        target=test_y,
    )
    checks = {
        "development_and_confirmatory_conditions_disjoint": not bool(
            set(development_specs) & set(confirmatory_specs)
        ),
        "heat_flux_excluded": not any(
            str(name).lower().startswith("q")
            for name in mv3.MODEL_INPUT_FIELDS + OUTPUT_FIELDS
        ),
        "target_free_selector": all(
            action["support"]["rule"].endswith("no_test_target")
            for action in actions.values()
        ),
        "bounded_residual_head": bool(
            test_diagnostics["bounded_normalized_residual_abs_max"]
            <= mv4.RESIDUAL_CAP_SIGMA + 1.0e-6
        ),
        "classical_fallback_validation_admissible": next(
            item for item in classical_records if item["method"] == classical_name
        )["admissible"],
        "finite_selected_fields": bool(np.all(np.isfinite(selected))),
        "positive_selected_temperature": bool(np.min(selected[:, 0]) >= 1.0),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV5_task",
        "budget_blocks": budget,
        "effective_DSMC_samples": int(
            budget * mv3_protocol["model_contract"]["samples_per_block"]
        ),
        "mv5_protocol_sha256": _sha256(protocol_path()),
        "source_contract": protocol["source_contract"],
        "sample_counts": {
            "train": len(train_x),
            "validation": len(validation_x),
            "confirmatory": len(test_x),
        },
        "development_seed_split": protocol["development_seed_split"],
        "confirmatory_seed_split": {key: list(value) for key, value in test.items()},
        "selection_contract": "predeclared convex-hull/budget rule and validation-only classical risk; confirmatory targets evaluated only after selection",
        "selection": {
            "residual_alpha": {
                "selected": alpha,
                "candidates": alpha_records,
            },
            "gaussian_like": {
                "selected_passes": gaussian_passes,
                "candidates": gaussian_records,
            },
            "tsvd_pod_type": {
                "selected_rank": tsvd_rank,
                "candidates": tsvd_records,
            },
            "risk_controlled_classical": {
                "selected_method": classical_name,
                "maximum_allowed_validation_ratio": VALIDATION_MAX_RATIO,
                "candidates": classical_records,
            },
            "confirmatory_actions": actions,
        },
        "training": training,
        "timing_seconds": {
            "validation_inference": validation_seconds,
            "confirmatory_inference": inference_seconds,
        },
        "diagnostics": {
            "validation_prediction": validation_diagnostics,
            "confirmatory_prediction": test_diagnostics,
            "validation_vision_mean_nrmse": float(
                np.mean(
                    [
                        item["vision_composite_nrmse"]
                        for item in _metric_by_condition(
                            validation_raw,
                            validation_vision,
                            validation_y,
                            validation_conditions,
                            development_specs,
                        ).values()
                    ]
                )
            ),
            "projection": {
                "gaussian_like": gaussian_projection,
                "tsvd_pod_type": tsvd_projection,
                "vision_bounded": vision_projection,
                "mv5_selected": selected_projection,
            },
        },
        "methods_by_condition": methods_by_condition,
        "aggregate_methods": aggregate_methods,
        "per_seed_metrics": per_seed,
        "checks": checks,
        "decision": "accept_MV5_task" if all(checks.values()) else "hold_MV5_task",
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "artifact_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output_dir / name),
                    "size_bytes": (output_dir / name).stat().st_size,
                }
                for name in ("model.pt", "predictions.npz", "summary.json")
            },
        },
    )
    return summary


def _records(root: Path) -> list[dict[str, Any]]:
    records = []
    for budget in BUDGETS:
        value = json.loads(
            (_task_directory(root, budget) / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            value.get("status") != "complete_MV5_task"
            or int(value.get("budget_blocks", -1)) != budget
        ):
            raise ValueError(f"invalid MV5 task summary for budget {budget}")
        records.append(value)
    return records


def _physical_figure(
    root: Path,
    *,
    condition_id: str,
    budget: int,
    methods: tuple[tuple[str, str], ...],
    filename: str,
) -> None:
    import matplotlib.pyplot as plt

    directory = _task_directory(root, budget)
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        labels = np.asarray(data["identity_condition"])
        identities = np.asarray(data["identity_numeric"])
        indices = np.flatnonzero(labels == condition_id)
        index = int(indices[0])
        values = {name: np.asarray(data[name])[index] for name, _ in methods}
        target = np.asarray(data["target"])[index]
    columns = (*methods, ("target", "Reference"))
    values["target"] = target
    figure, axes = plt.subplots(
        2,
        len(columns),
        figsize=(2.15 * len(columns) + 0.5, 4.75),
        constrained_layout=True,
    )
    labels_fields = (("T", "K"), ("u", "m s$^{-1}$"))
    panel = ord("a")
    for field, (symbol, unit) in enumerate(labels_fields):
        low = min(float(values[name][field].min()) for name, _ in columns if name != "vision_bounded")
        high = max(float(values[name][field].max()) for name, _ in columns if name != "vision_bounded")
        for column, (name, title) in enumerate(columns):
            image = axes[field, column].imshow(
                values[name][field],
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="coolwarm",
                vmin=low,
                vmax=high,
                interpolation="nearest",
            )
            axes[field, column].set_aspect("equal")
            axes[field, column].set_xlabel(r"$x/L$")
            if column == 0:
                axes[field, column].set_ylabel(r"$y/L$")
            else:
                axes[field, column].set_yticklabels([])
            if field == 0:
                axes[field, column].set_title(title, fontsize=9)
            axes[field, column].text(
                0.02,
                0.98,
                f"({chr(panel)})",
                transform=axes[field, column].transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="black",
            )
            panel += 1
        figure.colorbar(image, ax=axes[field, :], shrink=0.78, label=f"${symbol}$ [{unit}]")
    seed, block, _ = identities[index]
    condition = mv5ref.condition_map()[condition_id]
    figure.suptitle(
        rf"$Kn={condition['knudsen']:.3g}$, $U_w={condition['lid_speed_m_per_s']:.0f}$ m s$^{{-1}}$; "
        rf"budget={budget}, fixed seed={seed}, block={block}",
        fontsize=10,
    )
    figure.savefig(root / f"{filename}.png", dpi=400)
    figure.savefig(root / f"{filename}.pdf")
    plt.close(figure)


def _selector_map(root: Path) -> None:
    import matplotlib.pyplot as plt

    development = mv3._condition_map(mv3.locked_protocol())
    confirmation = mv5ref.condition_map()
    cloud = np.stack([_descriptor(item) for item in development.values()])
    hull = convex_hull(cloud)
    closed = np.vstack((hull, hull[0]))
    figure, axis = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    axis.fill(closed[:, 1] * 100.0, 10.0 ** closed[:, 0], color="#dbeafe", alpha=0.75)
    axis.plot(closed[:, 1] * 100.0, 10.0 ** closed[:, 0], color="#1f77b4", lw=1.8)
    for item in development.values():
        axis.scatter(item["lid_speed_m_per_s"], item["knudsen"], marker="o", color="black")
    for item in confirmation.values():
        support = convex_hull_support_report(item, tuple(development.values()))
        axis.scatter(
            item["lid_speed_m_per_s"],
            item["knudsen"],
            marker="*",
            s=120,
            color="#2ca02c" if support["inside_development_hull"] else "#d62728",
        )
        axis.annotate(item["id"], (item["lid_speed_m_per_s"], item["knudsen"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set(xlabel=r"Lid speed $U_w$ [m s$^{-1}$]", ylabel=r"Knudsen number $Kn$", yscale="log")
    axis.grid(alpha=0.2)
    figure.savefig(root / "mv5_condition_support_map.png", dpi=400)
    figure.savefig(root / "mv5_condition_support_map.pdf")
    plt.close(figure)


def aggregate(root: Path, reference_root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    records = _records(root)
    by_budget = {
        str(record["budget_blocks"]): record["aggregate_methods"]
        for record in records
    }
    selected_ratios = [
        metrics["vision_over_raw_composite"]
        for record in records
        for metrics in record["methods_by_condition"]["mv5_selected"].values()
    ]
    budget_one = next(item for item in records if item["budget_blocks"] == 1)
    inside = [
        condition_id
        for condition_id, action in budget_one["selection"]["confirmatory_actions"].items()
        if action["support"]["inside_development_hull"]
    ]
    gates = {
        "all_16_confirmatory_references_mechanical_and_T_u_stationary": all(
            _verify_reference(
                reference_root / "references" / condition / f"seed_{seed}"
            )
            for condition, seed in mv5ref.reference_tasks()
        ),
        "all_4_budget_tasks_complete": len(records) == len(BUDGETS),
        "selector_is_target_free": all(
            item["checks"]["target_free_selector"] for item in records
        ),
        "selected_maximum_over_raw": bool(
            max(selected_ratios)
            <= float(protocol["acceptance_gates"]["selected_maximum_over_raw"])
        ),
        "inside_hull_budget_1_improves_raw": bool(inside)
        and all(
            budget_one["methods_by_condition"]["mv5_selected"][condition][
                "vision_over_raw_composite"
            ]
            < 1.0
            for condition in inside
        ),
        "inside_hull_budget_1_beats_both_selected_classical_baselines": bool(inside)
        and all(
            budget_one["methods_by_condition"]["mv5_selected"][condition][
                "vision_composite_nrmse"
            ]
            < min(
                budget_one["methods_by_condition"]["gaussian_like"][condition][
                    "vision_composite_nrmse"
                ],
                budget_one["methods_by_condition"]["tsvd_pod_type"][condition][
                    "vision_composite_nrmse"
                ],
            )
            for condition in inside
        ),
    }
    _physical_figure(
        root,
        condition_id="kn0p075_u150",
        budget=1,
        methods=(
            ("raw", "Raw DSMC"),
            ("gaussian_like", "Gaussian-like"),
            ("tsvd_pod_type", "TSVD/POD"),
            ("vision_bounded", "Bounded vision"),
        ),
        filename="mv5_inside_hull_physical_fields",
    )
    _physical_figure(
        root,
        condition_id="kn0p1_u400",
        budget=1,
        methods=(
            ("raw", "Raw DSMC"),
            ("vision_bounded", "Bounded vision"),
            ("mv5_selected", "MV5 selected"),
        ),
        filename="mv5_ood_safety_physical_fields",
    )
    _selector_map(root)
    summary = {
        "stage": STAGE,
        "status": "complete_MV5_confirmatory_aggregate",
        "scope": "four new condition combinations; T and u only; predeclared convex-hull/validation-risk selector",
        "task_count": len(records),
        "new_reference_count": len(mv5ref.reference_tasks()),
        "mv5_protocol_sha256": _sha256(protocol_path()),
        "by_budget": by_budget,
        "checks": gates,
        "decision": "MV5_confirmatory_evidence_pass"
        if all(gates.values())
        else "hold_for_MV5_diagnosis",
    }
    _atomic_json(root / "summary.json", summary)
    top_names = (
        "summary.json",
        "mv5_inside_hull_physical_fields.png",
        "mv5_inside_hull_physical_fields.pdf",
        "mv5_ood_safety_physical_fields.png",
        "mv5_ood_safety_physical_fields.pdf",
        "mv5_condition_support_map.png",
        "mv5_condition_support_map.pdf",
    )
    _atomic_json(
        root / "artifact_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(root / name),
                    "size_bytes": (root / name).stat().st_size,
                }
                for name in top_names
            },
        },
    )
    return summary


def verify(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV5 top artifact verification failed: {path}")
    maximum_difference = 0.0
    verified_task_files = 0
    for budget in BUDGETS:
        directory = _task_directory(root, budget)
        task_manifest = json.loads(
            (directory / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        for name, record in task_manifest["files"].items():
            path = directory / name
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or _sha256(path) != record["sha256"]
            ):
                raise ValueError(f"MV5 task artifact verification failed: {path}")
            verified_task_files += 1
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        with np.load(directory / "predictions.npz", allow_pickle=False) as data:
            raw = np.asarray(data["raw"])
            target = np.asarray(data["target"])
            labels = np.asarray(data["identity_condition"])
            for method in METHODS:
                rebuilt = _metric_by_condition(
                    raw,
                    np.asarray(data[method]),
                    target,
                    labels,
                    mv5ref.condition_map(),
                )
                recorded = summary["methods_by_condition"][method]
                for condition_id in rebuilt:
                    maximum_difference = max(
                        maximum_difference,
                        abs(
                            rebuilt[condition_id]["vision_composite_nrmse"]
                            - recorded[condition_id]["vision_composite_nrmse"]
                        ),
                    )
    if maximum_difference > 2.5e-6:
        raise ValueError(f"MV5 metric reconstruction mismatch: {maximum_difference}")
    saved = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return {
        "status": "complete_MV5_artifacts_metrics_and_selector_verified",
        "decision": saved["decision"],
        "task_count": len(BUDGETS),
        "new_reference_count": len(mv5ref.reference_tasks()),
        "verified_task_files": verified_task_files,
        "maximum_metric_reconstruction_difference": maximum_difference,
        "summary_sha256": _sha256(root / "summary.json"),
    }


def package(root: Path) -> dict[str, Any]:
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "complete_MV5_artifacts_metrics_and_selector_verified":
        raise ValueError("MV5 must pass its verifier before packaging")
    bundle = root / "MOHAMMADZADEH_MV5_JCP_RETURN_BUNDLE.tar.gz"
    top_names = (
        "summary.json",
        "artifact_manifest.json",
        "verification.json",
        "mv5_inside_hull_physical_fields.png",
        "mv5_inside_hull_physical_fields.pdf",
        "mv5_ood_safety_physical_fields.png",
        "mv5_ood_safety_physical_fields.pdf",
        "mv5_condition_support_map.png",
        "mv5_condition_support_map.pdf",
    )
    with tarfile.open(bundle, "w:gz") as archive:
        for name in top_names:
            archive.add(root / name, arcname=name, filter=_portable_tarinfo)
        archive.add(
            protocol_path(),
            arcname=f"provenance/{PROTOCOL_FILE}",
            filter=_portable_tarinfo,
        )
        for budget in BUDGETS:
            directory = _task_directory(root, budget)
            for name in ("model.pt", "summary.json", "predictions.npz", "artifact_manifest.json"):
                archive.add(
                    directory / name,
                    arcname=f"tasks/budget_{budget}/{name}",
                    filter=_portable_tarinfo,
                )
    checksum = _sha256(bundle)
    (root / f"{bundle.name}.sha256").write_text(
        f"{checksum}  {bundle.name}\n", encoding="utf-8"
    )
    return {
        "status": "complete_MV5_portable_bundle",
        "bundle": str(bundle),
        "bundle_sha256": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-m3-root", type=Path)
    parser.add_argument("--mv3-root", type=Path)
    parser.add_argument("--mv5-reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--training-seed", type=int, default=20260809)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output_dir)
    elif args.package:
        result = package(args.output_dir)
    elif args.aggregate:
        if args.mv5_reference_root is None:
            parser.error("aggregate mode requires --mv5-reference-root")
        result = aggregate(args.output_dir, args.mv5_reference_root)
    else:
        budget = task_from_index(args.task_index) if args.task_index is not None else args.budget
        if budget is None:
            parser.error("task mode requires --task-index or --budget")
        if args.existing_m3_root is None or args.mv3_root is None or args.mv5_reference_root is None:
            parser.error("task mode requires all three source roots")
        result = run_task(
            args.existing_m3_root,
            args.mv3_root,
            args.mv5_reference_root,
            _task_directory(args.output_dir, budget),
            budget=budget,
            epochs=args.epochs,
            batch_size=args.batch_size,
            training_seed=args.training_seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
