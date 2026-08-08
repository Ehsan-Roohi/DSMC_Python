"""Seed-cross-validated, sampling-budget benchmark for Mohammadzadeh vision.

MV2 reuses the completed M3 trajectories.  It compares a residual U-Net with
unprocessed temporal averaging, a validation-selected separable spatial
filter, and a validation-selected truncated-SVD (POD-type) baseline.  Every
M3 seed is held out once; no heat-flux quantity enters the stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .mohammadzadeh_vision import (
    INPUT_FIELDS,
    OUTPUT_FIELDS,
    SEEDS,
    _loss,
    build_model,
    evaluate,
    fit_scaling,
    load_m3_images,
)

STAGE = "MV2_Mohammadzadeh_seed_crossvalidated_budget_benchmark"
BUDGETS = (1, 2, 5, 10)
GAUSSIAN_PASSES = (1, 2, 4)
TSVD_RANKS = (1, 2, 4, 8, 16, 32)
PROTOCOL_FILE = "mv2_jcp_benchmark_protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / PROTOCOL_FILE
    )


def load_protocol() -> dict[str, Any]:
    path = protocol_path()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != "locked_before_MV2_outcomes":
        raise ValueError("MV2 protocol is not locked")
    if tuple(value["data_contract"]["budget_blocks"]) != BUDGETS:
        raise ValueError("MV2 budget list differs from the code contract")
    if tuple(value["fold_contract"]["seeds"]) != SEEDS:
        raise ValueError("MV2 seed list differs from the code contract")
    return value


def fold_split(fold_index: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not 0 <= fold_index < len(SEEDS):
        raise ValueError(f"fold_index must be in [0,{len(SEEDS) - 1}]")
    test = (SEEDS[fold_index],)
    validation = (SEEDS[(fold_index - 1) % len(SEEDS)],)
    train = tuple(seed for seed in SEEDS if seed not in test + validation)
    if set(train) & set(validation + test) or set(validation) & set(test):
        raise AssertionError("MV2 fold construction leaked a seed")
    return train, validation, test


def group_blocks(values: np.ndarray, budget: int) -> np.ndarray:
    """Average consecutive, nonoverlapping temporal blocks."""
    array = np.asarray(values, dtype=np.float32)
    if budget not in BUDGETS or array.ndim < 1 or len(array) % budget:
        raise ValueError("block array is incompatible with the locked budget")
    groups = len(array) // budget
    return array.reshape(groups, budget, *array.shape[1:]).mean(axis=1, dtype=np.float64).astype(np.float32)


def fold_contained_targets(
    full: Mapping[int, np.ndarray],
    train_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Build targets without using validation or test converged fields.

    A training realization is excluded from its own target. Validation and
    test targets are the mean of the six training realizations. Consequently,
    neither held-out converged field can influence training, hyperparameter
    selection, or its own reference.
    """
    if len(train_seeds) < 2 or set(train_seeds) & set(validation_seeds + test_seeds):
        raise ValueError("invalid fold for cross-fitted targets")
    training_reference = np.mean(
        [np.asarray(full[seed], dtype=np.float64) for seed in train_seeds], axis=0
    ).astype(np.float32)
    targets: dict[int, np.ndarray] = {}
    for seed in train_seeds:
        targets[seed] = np.mean(
            [np.asarray(full[other], dtype=np.float64) for other in train_seeds if other != seed],
            axis=0,
        ).astype(np.float32)
    for seed in validation_seeds + test_seeds:
        targets[seed] = training_reference.copy()
    return targets


def build_budget_arrays(
    blocks: Mapping[int, np.ndarray],
    targets: Mapping[int, np.ndarray],
    seeds: tuple[int, ...],
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, identity = [], [], []
    for seed in seeds:
        grouped = group_blocks(blocks[seed], budget)
        for group_index, image in enumerate(grouped):
            x.append(image)
            y.append(targets[seed])
            identity.append((seed, group_index, budget))
    return np.stack(x), np.stack(y), np.asarray(identity, dtype=np.int64)


def _smooth_axis(values: np.ndarray, axis: int) -> np.ndarray:
    padded = np.pad(values, [(0, 0)] * (values.ndim - 2) + [(1, 1), (1, 1)], mode="reflect")
    if axis == -1:
        return 0.25 * padded[..., 1:-1, :-2] + 0.5 * padded[..., 1:-1, 1:-1] + 0.25 * padded[..., 1:-1, 2:]
    return 0.25 * padded[..., :-2, 1:-1] + 0.5 * padded[..., 1:-1, 1:-1] + 0.25 * padded[..., 2:, 1:-1]


def gaussian_like(values: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if passes < 1:
        raise ValueError("passes must be positive")
    for _ in range(passes):
        result = _smooth_axis(_smooth_axis(result, -1), -2).astype(np.float32)
    return result


def tsvd(values: np.ndarray, rank: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError("TSVD expects [sample, field, y, x]")
    result = np.empty_like(array)
    for sample in range(array.shape[0]):
        for field in range(array.shape[1]):
            u, singular, vt = np.linalg.svd(array[sample, field], full_matrices=False)
            retained = min(rank, len(singular))
            result[sample, field] = (u[:, :retained] * singular[:retained]) @ vt[:retained]
    return result


def _composite(candidate: np.ndarray, target: np.ndarray) -> float:
    raw = np.asarray(candidate)
    return float(evaluate(raw, raw, target)["raw_composite_nrmse"])


def select_baseline(
    validation_raw: np.ndarray,
    validation_target: np.ndarray,
    candidates: tuple[int, ...],
    transform: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[int, list[dict[str, float]]]:
    records = []
    for parameter in candidates:
        score = _composite(transform(validation_raw, parameter), validation_target)
        records.append({"parameter": int(parameter), "validation_composite_nrmse": score})
    best = min(records, key=lambda item: (item["validation_composite_nrmse"], item["parameter"]))
    return int(best["parameter"]), records


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensors(x: np.ndarray, y: np.ndarray):
        normalized = (x - scaling["input_mean"]) / scaling["input_std"]
        residual = (y - x[:, : len(OUTPUT_FIELDS)]) / scaling["residual_std"]
        return torch.from_numpy(normalized), torch.from_numpy(residual)

    tx, ty = tensors(train_x, train_y)
    vx, vy = tensors(validation_x, validation_y)
    loader = DataLoader(
        TensorDataset(tx, ty),
        batch_size=min(batch_size, len(tx)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = build_model(in_channels=int(train_x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    best_state, best_value, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        model.eval()
        with torch.no_grad():
            validation = float(_loss(model(vx.to(device)), vy.to(device)).cpu())
        history.append({"epoch": epoch, "train_loss": total / len(tx), "validation_loss": validation})
        if validation < best_value - 1.0e-7:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_value, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is None:
        raise RuntimeError("MV2 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, {
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "history": history,
    }


def predict(model: Any, x: np.ndarray, scaling: Mapping[str, np.ndarray], batch_size: int) -> tuple[np.ndarray, float]:
    import torch

    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    values = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            residual = model(torch.from_numpy(normalized[start : start + batch_size])).numpy()
            values.append(x[start : start + batch_size, : len(OUTPUT_FIELDS)] + residual * scaling["residual_std"])
    return np.concatenate(values), time.perf_counter() - started


def task_index(fold_index: int, budget: int) -> int:
    if budget not in BUDGETS:
        raise ValueError("unknown MV2 budget")
    return fold_index * len(BUDGETS) + BUDGETS.index(budget)


def task_from_index(index: int) -> tuple[int, int]:
    if not 0 <= index < len(SEEDS) * len(BUDGETS):
        raise ValueError("MV2 task index is outside the locked array")
    return index // len(BUDGETS), BUDGETS[index % len(BUDGETS)]


def run_task(
    m3_root: Path,
    output_dir: Path,
    *,
    fold_index: int,
    budget: int,
    epochs: int,
    batch_size: int,
    training_seed: int,
) -> dict[str, Any]:
    protocol = load_protocol()
    train_seeds, validation_seeds, test_seeds = fold_split(fold_index)
    blocks, full = load_m3_images(m3_root)
    targets = fold_contained_targets(
        full, train_seeds, validation_seeds, test_seeds
    )
    train_x, train_y, train_id = build_budget_arrays(blocks, targets, train_seeds, budget)
    validation_x, validation_y, validation_id = build_budget_arrays(blocks, targets, validation_seeds, budget)
    test_x, test_y, test_id = build_budget_arrays(blocks, targets, test_seeds, budget)
    scaling = fit_scaling(train_x, train_y)
    model, training = train_model(
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        epochs=epochs,
        batch_size=batch_size,
        seed=training_seed + task_index(fold_index, budget),
    )
    vision, inference_seconds = predict(model, test_x, scaling, batch_size)
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    gaussian_passes, gaussian_selection = select_baseline(
        validation_raw, validation_y, GAUSSIAN_PASSES, gaussian_like
    )
    tsvd_rank, tsvd_selection = select_baseline(
        validation_raw, validation_y, TSVD_RANKS, tsvd
    )
    gaussian = gaussian_like(raw, gaussian_passes)
    pod_type = tsvd(raw, tsvd_rank)
    methods = {
        "raw": evaluate(raw, raw, test_y),
        "gaussian_like": evaluate(raw, gaussian, test_y),
        "tsvd_pod_type": evaluate(raw, pod_type, test_y),
        "vision": evaluate(raw, vision, test_y),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": STAGE,
        "state_dict": model.state_dict(),
        "scaling": {key: np.asarray(value) for key, value in scaling.items()},
        "fold_index": fold_index,
        "budget": budget,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
    }
    import torch

    torch.save(checkpoint, output_dir / "model.pt")
    np.savez_compressed(
        output_dir / "predictions.npz",
        identity=test_id,
        raw=raw,
        gaussian_like=gaussian,
        tsvd_pod_type=pod_type,
        vision=vision,
        target=test_y,
    )
    checks = {
        "seed_disjoint": not bool(set(train_seeds) & set(validation_seeds + test_seeds))
        and not bool(set(validation_seeds) & set(test_seeds)),
        "heat_flux_excluded": not any(name.lower().startswith("q") for name in INPUT_FIELDS + OUTPUT_FIELDS),
        "finite_metrics": all(
            np.isfinite(result["vision_composite_nrmse"])
            for result in methods.values()
        ),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV2_task",
        "protocol_sha256": _sha256(protocol_path()),
        "fold_index": fold_index,
        "budget_blocks": budget,
        "effective_DSMC_samples": int(
            budget * protocol["data_contract"]["samples_per_block"]
        ),
        "split": {
            "train": list(train_seeds),
            "validation": list(validation_seeds),
            "test": list(test_seeds),
        },
        "sample_counts": {
            "train": len(train_x),
            "validation": len(validation_x),
            "test": len(test_x),
        },
        "target_contract": "strict fold-contained cross-fit; validation/test converged fields excluded from training and tuning",
        "baseline_selection": {
            "gaussian_like": {"selected_passes": gaussian_passes, "candidates": gaussian_selection},
            "tsvd_pod_type": {"selected_rank": tsvd_rank, "candidates": tsvd_selection},
        },
        "training": training,
        "inference_seconds": inference_seconds,
        "methods": methods,
        "checks": checks,
        "decision": "accept_MV2_task" if all(checks.values()) else "hold_MV2_task",
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


def _task_directory(root: Path, fold: int, budget: int) -> Path:
    return root / "tasks" / f"fold_{fold}" / f"budget_{budget}"


def _curve_figure(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    methods = ("raw", "gaussian_like", "tsvd_pod_type", "vision")
    labels = {"raw": "Raw average", "gaussian_like": "Spatial filter", "tsvd_pod_type": "TSVD/POD-type", "vision": "Residual U-Net"}
    colors = {"raw": "0.35", "gaussian_like": "#2ca02c", "tsvd_pod_type": "#ff7f0e", "vision": "#1f77b4"}
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for method in methods:
        matrix = np.asarray(
            [[next(r for r in records if r["fold_index"] == fold and r["budget_blocks"] == budget)["methods"][method]["vision_composite_nrmse"] for budget in BUDGETS] for fold in range(len(SEEDS))]
        )
        for row in matrix:
            axes[0].plot(BUDGETS, row, color=colors[method], alpha=0.13, linewidth=0.8)
        axes[0].plot(BUDGETS, np.median(matrix, axis=0), marker="o", color=colors[method], linewidth=2.2, label=labels[method])
        ratio = matrix / np.asarray(
            [[next(r for r in records if r["fold_index"] == fold and r["budget_blocks"] == budget)["methods"]["raw"]["vision_composite_nrmse"] for budget in BUDGETS] for fold in range(len(SEEDS))]
        )
        axes[1].plot(BUDGETS, np.median(ratio, axis=0), marker="o", color=colors[method], linewidth=2.2, label=labels[method])
    axes[0].set(xlabel="Temporal blocks averaged", ylabel="Held-out composite NRMSE", xscale="log", yscale="log")
    axes[1].set(xlabel="Temporal blocks averaged", ylabel="Error / raw error", xscale="log")
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    for axis in axes:
        axis.set_xticks(BUDGETS, labels=[str(value) for value in BUDGETS])
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("MV2 seed-held-out sampling-budget benchmark")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _contour_figure(task_dir: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    with np.load(task_dir / "predictions.npz", allow_pickle=False) as data:
        values = {name: np.asarray(data[name])[0] for name in ("raw", "gaussian_like", "tsvd_pod_type", "vision", "target")}
    methods = ("raw", "gaussian_like", "tsvd_pod_type", "vision", "target")
    titles = ("Raw block", "Spatial filter", "TSVD/POD-type", "Residual U-Net", "LOSO target")
    figure, axes = plt.subplots(2, 5, figsize=(15.2, 6.0), constrained_layout=True)
    for row, (field, label) in enumerate(((0, "T (K)"), (1, "u (m/s)"))):
        low = min(values[name][field].min() for name in methods)
        high = max(values[name][field].max() for name in methods)
        levels = np.linspace(low, high, 24)
        for column, (name, title) in enumerate(zip(methods, titles)):
            plot = axes[row, column].contourf(values[name][field], levels=levels, cmap="coolwarm", extend="both")
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        figure.colorbar(plot, ax=axes[row, :], label=label, shrink=0.78)
    figure.suptitle("MV2 unsmoothed held-out contours, one-block budget")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _profile_figure(task_dir: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    with np.load(task_dir / "predictions.npz", allow_pickle=False) as data:
        values = {
            name: np.asarray(data[name])[0]
            for name in ("raw", "gaussian_like", "tsvd_pod_type", "vision", "target")
        }
    methods = ("raw", "gaussian_like", "tsvd_pod_type", "vision", "target")
    labels = ("Raw block", "Spatial filter", "TSVD/POD-type", "Residual U-Net", "LOSO target")
    styles = (":", "-.", "--", "-", "-")
    colors = ("0.45", "#2ca02c", "#ff7f0e", "#1f77b4", "black")
    ny, nx = values["target"].shape[-2:]
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    vertical_index = int(round(0.8 * nx - 0.5))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for name, label, style, color in zip(methods, labels, styles, colors):
        value = values[name]
        axes[0].plot(x, 1.0 - value[1, -1] / 100.0, linestyle=style, color=color, linewidth=2.0, label=label)
        axes[1].plot(x, value[0, -1], linestyle=style, color=color, linewidth=2.0)
        axes[2].plot(y, value[0, :, vertical_index], linestyle=style, color=color, linewidth=2.0)
    axes[0].set(xlabel="x/L", ylabel=r"$(U_{lid}-u)/U_{lid}$", title="Macroscopic lid slip")
    axes[1].set(xlabel="x/L", ylabel="T (K)", title="Lid temperature")
    axes[2].set(xlabel="y/L", ylabel="T (K)", title="Vertical temperature at x/L=0.8")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("MV2 unsmoothed held-out physical profiles, one-block budget")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def aggregate(root: Path) -> dict[str, Any]:
    protocol = load_protocol()
    records = []
    for fold in range(len(SEEDS)):
        for budget in BUDGETS:
            path = _task_directory(root, fold, budget) / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing MV2 task summary: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "complete_MV2_task" or value.get("protocol_sha256") != _sha256(protocol_path()):
                raise ValueError(f"invalid MV2 task summary: {path}")
            records.append(value)
    methods = ("raw", "gaussian_like", "tsvd_pod_type", "vision")
    by_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        selected = [record for record in records if record["budget_blocks"] == budget]
        by_budget[str(budget)] = {}
        for method in methods:
            values = np.asarray([record["methods"][method]["vision_composite_nrmse"] for record in selected])
            raw = np.asarray([record["methods"]["raw"]["vision_composite_nrmse"] for record in selected])
            by_budget[str(budget)][method] = {
                "median_composite_nrmse": float(np.median(values)),
                "mean_composite_nrmse": float(np.mean(values)),
                "min_composite_nrmse": float(np.min(values)),
                "max_composite_nrmse": float(np.max(values)),
                "median_over_raw": float(np.median(values / raw)),
                "heldout_seed_count": len(values),
            }
    one = by_budget["1"]
    checks = {
        "all_32_tasks_complete": len(records) == len(SEEDS) * len(BUDGETS),
        "seed_disjoint_every_fold": all(all(record["checks"].values()) for record in records),
        "heat_flux_excluded": not any(name.lower().startswith("q") for name in INPUT_FIELDS + OUTPUT_FIELDS),
        "median_vision_over_raw_composite_max": one["vision"]["median_over_raw"] <= float(protocol["gates"]["median_vision_over_raw_composite_max"]),
        "vision_beats_selected_baselines_at_budget_1": one["vision"]["median_composite_nrmse"] < min(one["gaussian_like"]["median_composite_nrmse"], one["tsvd_pod_type"]["median_composite_nrmse"]),
    }
    root.mkdir(parents=True, exist_ok=True)
    _curve_figure(records, root / "sampling_budget_curves.png")
    _contour_figure(_task_directory(root, 0, 1), root / "heldout_method_contours.png")
    _profile_figure(_task_directory(root, 0, 1), root / "heldout_method_profiles.png")
    summary = {
        "stage": STAGE,
        "status": "complete_MV2_aggregate",
        "scope": "single physical condition; eight-seed cross-validation and four sampling budgets",
        "protocol_sha256": _sha256(protocol_path()),
        "task_count": len(records),
        "by_budget": by_budget,
        "checks": checks,
        "decision": "advance_to_cross_condition_MV3" if all(checks.values()) else "hold_for_MV2_diagnosis",
    }
    _atomic_json(root / "summary.json", summary)
    artifact_names = (
        "summary.json",
        "sampling_budget_curves.png",
        "heldout_method_contours.png",
        "heldout_method_profiles.png",
    )
    manifest = {
        "stage": STAGE,
        "files": {name: {"sha256": _sha256(root / name), "size_bytes": (root / name).stat().st_size} for name in artifact_names},
    }
    _atomic_json(root / "artifact_manifest.json", manifest)
    return summary


def verify(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"MV2 artifact verification failed for {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    records = []
    maximum_metric_difference = 0.0
    verified_task_files = 0
    for fold in range(len(SEEDS)):
        for budget in BUDGETS:
            directory = _task_directory(root, fold, budget)
            task_manifest = json.loads(
                (directory / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            for name, record in task_manifest.get("files", {}).items():
                path = directory / name
                if (
                    not path.is_file()
                    or path.stat().st_size != record["size_bytes"]
                    or _sha256(path) != record["sha256"]
                ):
                    raise ValueError(f"MV2 task artifact verification failed: {path}")
                verified_task_files += 1
            task_summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            if (
                task_summary.get("fold_index") != fold
                or task_summary.get("budget_blocks") != budget
                or task_summary.get("status") != "complete_MV2_task"
            ):
                raise ValueError(f"MV2 task identity/status mismatch: {directory}")
            with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                raw = np.asarray(data["raw"])
                target = np.asarray(data["target"])
                for method in ("raw", "gaussian_like", "tsvd_pod_type", "vision"):
                    rebuilt = evaluate(raw, np.asarray(data[method]), target)
                    recorded = task_summary["methods"][method]
                    for field in OUTPUT_FIELDS:
                        maximum_metric_difference = max(
                            maximum_metric_difference,
                            abs(
                                rebuilt["per_field"][field]["vision_nrmse"]
                                - recorded["per_field"][field]["vision_nrmse"]
                            ),
                        )
                    maximum_metric_difference = max(
                        maximum_metric_difference,
                        abs(
                            rebuilt["vision_composite_nrmse"]
                            - recorded["vision_composite_nrmse"]
                        ),
                    )
            records.append(task_summary)
    if maximum_metric_difference > 2.0e-6:
        raise ValueError(
            f"MV2 metric reconstruction mismatch: {maximum_metric_difference}"
        )
    for budget in BUDGETS:
        selected = [record for record in records if record["budget_blocks"] == budget]
        for method in ("raw", "gaussian_like", "tsvd_pod_type", "vision"):
            values = np.asarray(
                [record["methods"][method]["vision_composite_nrmse"] for record in selected]
            )
            raw_values = np.asarray(
                [record["methods"]["raw"]["vision_composite_nrmse"] for record in selected]
            )
            rebuilt = {
                "mean_composite_nrmse": float(np.mean(values)),
                "median_composite_nrmse": float(np.median(values)),
                "min_composite_nrmse": float(np.min(values)),
                "max_composite_nrmse": float(np.max(values)),
                "median_over_raw": float(np.median(values / raw_values)),
            }
            for key, value in rebuilt.items():
                if abs(value - summary["by_budget"][str(budget)][method][key]) > 1.0e-12:
                    raise ValueError(f"MV2 aggregate reconstruction mismatch: {budget}/{method}/{key}")
    return {
        "status": "complete_MV2_artifacts_and_metrics_verified",
        "decision": summary["decision"],
        "summary_sha256": _sha256(root / "summary.json"),
        "task_count": len(records),
        "verified_task_files": verified_task_files,
        "maximum_metric_reconstruction_difference": maximum_metric_difference,
    }


def _portable_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def package(root: Path) -> dict[str, Any]:
    verification_path = root / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "complete_MV2_artifacts_and_metrics_verified":
        raise ValueError("MV2 must pass the recursive verifier before packaging")
    bundle = root / "MOHAMMADZADEH_MV2_JCP_RETURN_BUNDLE.tar.gz"
    top_names = (
        "summary.json",
        "sampling_budget_curves.png",
        "heldout_method_contours.png",
        "heldout_method_profiles.png",
        "artifact_manifest.json",
        "verification.json",
    )
    with tarfile.open(bundle, "w:gz") as archive:
        for name in top_names:
            archive.add(root / name, arcname=name, filter=_portable_tarinfo)
        for fold in range(len(SEEDS)):
            for budget in BUDGETS:
                directory = _task_directory(root, fold, budget)
                for name in (
                    "model.pt",
                    "summary.json",
                    "predictions.npz",
                    "artifact_manifest.json",
                ):
                    archive.add(
                        directory / name,
                        arcname=f"tasks/fold_{fold}/budget_{budget}/{name}",
                        filter=_portable_tarinfo,
                    )
    checksum = _sha256(bundle)
    (root / f"{bundle.name}.sha256").write_text(
        f"{checksum}  {bundle.name}\n", encoding="utf-8"
    )
    return {
        "status": "complete_MV2_portable_bundle",
        "bundle": str(bundle),
        "bundle_sha256": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--training-seed", type=int, default=20260807)
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
            parser.error("task mode requires --task-index or both --fold-index and --budget")
        m3_root = args.m3_root or Path("results/mohammadzadeh_2012/m3_qy_precision")
        result = run_task(
            m3_root,
            _task_directory(args.output_dir, fold, budget),
            fold_index=fold,
            budget=budget,
            epochs=args.epochs,
            batch_size=args.batch_size,
            training_seed=args.training_seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
