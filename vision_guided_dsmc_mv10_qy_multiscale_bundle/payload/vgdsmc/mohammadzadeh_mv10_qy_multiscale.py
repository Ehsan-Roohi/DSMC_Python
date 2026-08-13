"""MV10 multiscale repair of the coherent low-frequency DSMC q_y bias.

MV10 is a method-development stage locked after the MV9 outcomes were seen.
It reuses only MV9 development data for training and selection.  The old MV9
evaluation seeds are evaluated only in postprocessing as a disclosed legacy
diagnostic; they cannot provide a confirmatory publication claim.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV10_Mohammadzadeh_qy_multiscale_bias_repair"
STATUS = "locked_after_observing_MV9_before_any_MV10_model_outcome"
PROTOCOL_FILE = "mv10_qy_multiscale_bias_repair_protocol.json"
MODEL_NAME = "qy_local_coarse_global_residual"
TRAINING_SEEDS = (2608101, 2608102, 2608103)
QY_INDEX = 3
RESIDUAL_CAP_SIGMA = 5.0
LOW_PASS_FACTOR = 8
LOW_PASS_LOSS_WEIGHT = 2.0
GLOBAL_MEAN_LOSS_WEIGHT = 2.0
GRADIENT_LOSS_WEIGHT = 0.05
BOUNDARY_LOSS_STRENGTH = 1.0
BOUNDARY_DECAY_FRACTION = 0.08
EXPECTED_LEGACY_SEEDS = (94301, 94302, 94303, 94304)


def _mv9_module():
    from . import mohammadzadeh_mv9_heat_flux as mv9

    return mv9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def protocol_path() -> Path:
    return _mv9_module().protocol_path().parent / PROTOCOL_FILE


def locked_protocol() -> dict[str, Any]:
    mv9 = _mv9_module()
    path = protocol_path()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV10 protocol is absent or unlocked")
    execution = value["execution_matrix"]
    training = value["training_contract"]
    disclosure = value["scientific_role"]
    if (
        execution["model_name"] != MODEL_NAME
        or tuple(execution["training_initialization_seeds"]) != TRAINING_SEEDS
        or int(execution["model_tasks"]) != len(TRAINING_SEEDS)
        or int(training["low_pass_factor"]) != LOW_PASS_FACTOR
        or float(training["low_pass_loss_weight"]) != LOW_PASS_LOSS_WEIGHT
        or float(training["global_mean_loss_weight"]) != GLOBAL_MEAN_LOSS_WEIGHT
        or float(training["gradient_loss_weight"]) != GRADIENT_LOSS_WEIGHT
        or not bool(disclosure["MV9_outcomes_observed_before_lock"])
        or not bool(disclosure["old_evaluation_seeds_forbidden_as_confirmation"])
        or tuple(disclosure["old_evaluation_seeds"]) != EXPECTED_LEGACY_SEEDS
        or execution["primary_condition"] not in execution["secondary_conditions"]
    ):
        raise ValueError("MV10 source differs from the locked protocol")
    source = value["source_contract"]
    if _sha256(Path(mv9.__file__)) != source["mv9_module_sha256"]:
        raise ValueError("MV10 MV9 module ancestry hash mismatch")
    if _sha256(mv9.protocol_path()) != source["mv9_protocol_sha256"]:
        raise ValueError("MV10 MV9 protocol ancestry hash mismatch")
    return value


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV10_lock_verified_without_reading_any_MV10_outcome",
        "protocol_sha256": _sha256(protocol_path()),
        "model_tasks": len(TRAINING_SEEDS),
        "model_name": MODEL_NAME,
        "scientific_classification": protocol["scientific_role"]["classification"],
        "confirmation_requires_fresh_seeds": True,
    }


def task_from_index(index: int) -> int:
    if not 0 <= index < len(TRAINING_SEEDS):
        raise ValueError(f"MV10 task index must be in [0,{len(TRAINING_SEEDS) - 1}]")
    return TRAINING_SEEDS[index]


def component_nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidate.shape != target.shape:
        raise ValueError("MV10 component metric arrays must have matching shapes")
    error = np.sqrt(np.mean((candidate - target) ** 2))
    scale = np.sqrt(np.mean(target**2))
    return float(error / max(scale, 1.0e-12))


def numpy_block_average(field: np.ndarray, factor: int = LOW_PASS_FACTOR) -> np.ndarray:
    """Return non-overlapping block means, including partial boundary blocks."""

    field = np.asarray(field, dtype=np.float64)
    if field.ndim < 2 or factor < 1:
        raise ValueError("MV10 block averaging requires a spatial array and positive factor")
    ny, nx = field.shape[-2:]
    rows = []
    for y0 in range(0, ny, factor):
        columns = []
        for x0 in range(0, nx, factor):
            columns.append(field[..., y0 : y0 + factor, x0 : x0 + factor].mean(axis=(-2, -1)))
        rows.append(np.stack(columns, axis=-1))
    return np.stack(rows, axis=-2)


def _verify_manifest(root: Path, manifest_name: str) -> dict[str, Any]:
    manifest_path = Path(root) / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = Path(root) / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV10 source artifact verification failed: {path}")
    return manifest


def _task_directory(root: Path, seed: int) -> Path:
    return Path(root) / "tasks" / MODEL_NAME / f"training_seed_{seed}"


def seed_identity_by_condition(
    conditions: np.ndarray, identities: np.ndarray
) -> dict[str, tuple[int, ...]]:
    """Return the unique evaluation seeds for each condition.

    MV9 stores all four evaluation conditions in one test tensor.  Seed IDs are
    condition-specific, so the MV10 primary legacy gate must compare the
    primary-condition slice rather than the union across the complete tensor.
    """

    conditions = np.asarray(conditions)
    identities = np.asarray(identities)
    if conditions.ndim != 1:
        raise ValueError("MV10 evaluation conditions must be one-dimensional")
    if (
        identities.ndim != 2
        or identities.shape[0] != conditions.shape[0]
        or identities.shape[1] < 1
    ):
        raise ValueError("MV10 evaluation identity/condition shapes do not match")
    result: dict[str, set[int]] = {}
    for condition, identity in zip(conditions, identities):
        result.setdefault(str(condition), set()).add(int(identity[0]))
    return {
        condition: tuple(sorted(seeds))
        for condition, seeds in sorted(result.items())
    }


def run_assembly(mv9_output_root: Path, output_root: Path) -> dict[str, Any]:
    """Verify and stage the completed MV9 dataset without rerunning DSMC."""

    mv9 = _mv9_module()
    protocol = locked_protocol()
    mv9_output_root = Path(mv9_output_root).resolve()
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite MV10 output: {output_root}")

    mv9_summary = json.loads((mv9_output_root / "summary.json").read_text(encoding="utf-8"))
    mv9_verification = json.loads(
        (mv9_output_root / "verification.json").read_text(encoding="utf-8")
    )
    mv9_assembly = json.loads(
        (mv9_output_root / "assembly_summary.json").read_text(encoding="utf-8")
    )
    if (
        mv9_summary.get("decision")
        != "MV9_feasibility_does_not_support_one_block_kinetic_moment_claim"
        or mv9_verification.get("decision") != "verified"
        or mv9_assembly.get("decision") != "proceed_to_MV9_B1_kinetic_models"
    ):
        raise ValueError("MV10 requires the verified, completed MV9 failure outcome")
    _verify_manifest(mv9_output_root, "assembly_manifest.json")
    _verify_manifest(mv9_output_root, "artifact_manifest.json")

    mv9_predictions: dict[str, list[np.ndarray]] = {
        architecture: [] for architecture in mv9.ARCHITECTURES
    }
    verified_tasks = 0
    for architecture in mv9.ARCHITECTURES:
        for seed in mv9.TRAINING_SEEDS:
            _, prediction = mv9._verify_task(
                mv9._task_directory(mv9_output_root, architecture, seed), architecture, seed
            )
            mv9_predictions[architecture].append(prediction)
            verified_tasks += 1
    ensembles = {
        architecture: np.mean(values, axis=0).astype(np.float32)
        for architecture, values in mv9_predictions.items()
    }

    with np.load(mv9_output_root / "dataset.npz", allow_pickle=False) as source:
        required = (
            "train_x",
            "train_y",
            "train_condition",
            "train_identity",
            "validation_x",
            "validation_y",
            "validation_condition",
            "validation_identity",
            "validation_raw10",
            "validation_target10",
            "test_x",
            "test_y",
            "test_condition",
            "test_identity",
            "test_scale",
            "test_gaussian",
            "test_tsvd",
            "test_raw10",
            "test_target10",
            "test_condition10",
            "test_identity10",
            "test_scale10",
        )
        arrays = {name: np.asarray(source[name]).copy() for name in required}

    observed_seeds_by_condition = seed_identity_by_condition(
        arrays["test_condition"], arrays["test_identity"]
    )
    observed_seeds10_by_condition = seed_identity_by_condition(
        arrays["test_condition10"], arrays["test_identity10"]
    )
    primary_condition = str(protocol["execution_matrix"]["primary_condition"])
    expected_conditions = tuple(
        sorted(str(item) for item in protocol["execution_matrix"]["secondary_conditions"])
    )
    observed_seeds = observed_seeds_by_condition.get(primary_condition, ())
    observed_seeds10 = observed_seeds10_by_condition.get(primary_condition, ())
    checks = {
        "MV9_recursive_return_verified": True,
        "MV9_information_gate_passed": True,
        "MV9_failure_outcome_explicitly_required": True,
        "all_six_MV9_model_tasks_recursively_verified": verified_tasks == 6,
        "legacy_condition_set_matches_locked_matrix": (
            tuple(observed_seeds_by_condition) == expected_conditions
            and tuple(observed_seeds10_by_condition) == expected_conditions
        ),
        "legacy_B1_and_B10_identity_maps_match": (
            observed_seeds_by_condition == observed_seeds10_by_condition
        ),
        "legacy_seed_identity_matches_disclosed_observed_set": (
            observed_seeds == EXPECTED_LEGACY_SEEDS
            and observed_seeds10 == EXPECTED_LEGACY_SEEDS
        ),
        "training_validation_and_legacy_arrays_finite": all(
            np.all(np.isfinite(value))
            for name, value in arrays.items()
            if value.dtype.kind in "fiu"
        ),
        "legacy_labels_not_used_to_train_or_select_MV10": True,
    }
    if not all(checks.values()):
        raise ValueError(f"MV10 assembly contract failed: {checks}")
    if ensembles["mambairv2_tiny_adapted"].shape != arrays["test_y"].shape:
        raise ValueError("MV9 ensemble and legacy target shapes differ")

    output_root.mkdir(parents=True)
    np.savez_compressed(
        output_root / "dataset.npz",
        **arrays,
        mv9_nafnet_ensemble=ensembles["nafnet_small"],
        mv9_mamba_ensemble=ensembles["mambairv2_tiny_adapted"],
    )
    summary = {
        "stage": STAGE,
        "status": "complete_MV10_verified_MV9_reuse_assembly",
        "protocol_sha256": _sha256(protocol_path()),
        "mv9_output_root": str(mv9_output_root),
        "mv9_summary_sha256": _sha256(mv9_output_root / "summary.json"),
        "mv9_dataset_sha256": _sha256(mv9_output_root / "dataset.npz"),
        "mv9_decision": mv9_summary["decision"],
        "verified_MV9_model_tasks": verified_tasks,
        "primary_legacy_condition": primary_condition,
        "observed_legacy_evaluation_seeds": list(observed_seeds),
        "observed_evaluation_seeds_by_condition": {
            condition: list(seeds)
            for condition, seeds in observed_seeds_by_condition.items()
        },
        "sample_counts": {
            "development_train_B1": len(arrays["train_x"]),
            "development_validation_B1": len(arrays["validation_x"]),
            "observed_legacy_B1": len(arrays["test_x"]),
            "observed_legacy_B10": len(arrays["test_raw10"]),
        },
        "checks": checks,
        "decision": "proceed_to_MV10_development_only_qy_models",
    }
    _atomic_json(output_root / "assembly_summary.json", summary)
    _atomic_json(
        output_root / "assembly_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(output_root / name),
                    "size_bytes": (output_root / name).stat().st_size,
                }
                for name in ("dataset.npz", "assembly_summary.json")
            },
        },
    )
    return summary


def _fit_scaling(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray]:
    input_mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    input_std = np.maximum(train_x.std(axis=(0, 2, 3), keepdims=True), 1.0e-6)
    # Preserve the physically declared condition scaling outside the sampled range.
    for offset, (center, scale) in enumerate(
        zip((-1.1505149978319906, 2.5), (0.3010299956639812, 3.0))
    ):
        index = train_x.shape[1] - 2 + offset
        input_mean[0, index, 0, 0] = center
        input_std[0, index, 0, 0] = scale
    residual = train_y[:, QY_INDEX : QY_INDEX + 1] - train_x[:, QY_INDEX : QY_INDEX + 1]
    residual_std = np.maximum(residual.std(axis=(0, 2, 3), keepdims=True), 1.0e-4)
    return {
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "residual_std": residual_std.astype(np.float32),
    }


def _build_model(input_channels: int):
    import torch
    from torch import nn
    import torch.nn.functional as functional

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
            self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

        def forward(self, value):
            return value + self.scale * self.body(value)

    class QYLocalCoarseGlobal(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 32
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, padding=1), nn.SiLU()
            )
            self.local = nn.Sequential(
                ResidualBlock(channels, 1),
                ResidualBlock(channels, 2),
                ResidualBlock(channels, 4),
                ResidualBlock(channels, 2),
            )
            self.local_head = nn.Conv2d(channels, 1, 3, padding=1)
            self.coarse_head = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels, 1, 3, padding=1),
            )
            self.global_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, 16, 1),
                nn.SiLU(),
                nn.Conv2d(16, 1, 1),
            )

        def forward(self, value):
            features = self.local(self.stem(value))
            factor = max(1, min(LOW_PASS_FACTOR, features.shape[-2], features.shape[-1]))
            coarse = functional.avg_pool2d(features, kernel_size=factor, stride=factor)
            coarse = self.coarse_head(coarse)
            coarse = functional.interpolate(
                coarse, size=features.shape[-2:], mode="bilinear", align_corners=False
            )
            return self.local_head(features) + coarse + self.global_head(features)

    return QYLocalCoarseGlobal()


def _train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cpu")
    tx = torch.from_numpy(
        ((train_x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    )
    ty = torch.from_numpy(
        (
            (train_y[:, QY_INDEX : QY_INDEX + 1] - train_x[:, QY_INDEX : QY_INDEX + 1])
            / scaling["residual_std"]
        ).astype(np.float32)
    )
    vx = torch.from_numpy(
        ((validation_x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    )
    vy = torch.from_numpy(
        (
            (
                validation_y[:, QY_INDEX : QY_INDEX + 1]
                - validation_x[:, QY_INDEX : QY_INDEX + 1]
            )
            / scaling["residual_std"]
        ).astype(np.float32)
    )
    loader = DataLoader(
        TensorDataset(tx, ty),
        batch_size=min(batch_size, len(tx)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    model = _build_model(int(train_x.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-5)

    ny, nx = int(train_y.shape[-2]), int(train_y.shape[-1])
    ycoord = torch.linspace(0.0, 1.0, ny).view(ny, 1)
    xcoord = torch.linspace(0.0, 1.0, nx).view(1, nx)
    distance = torch.minimum(
        torch.minimum(xcoord.expand(ny, nx), 1.0 - xcoord.expand(ny, nx)),
        torch.minimum(ycoord.expand(ny, nx), 1.0 - ycoord.expand(ny, nx)),
    )
    spatial = 1.0 + BOUNDARY_LOSS_STRENGTH * torch.exp(
        -distance / BOUNDARY_DECAY_FRACTION
    )
    spatial = (spatial / spatial.mean()).view(1, 1, ny, nx)

    def loss_function(prediction, target):
        pixel = torch.mean(spatial * (prediction - target) ** 2)
        factor = max(1, min(LOW_PASS_FACTOR, prediction.shape[-2], prediction.shape[-1]))
        coarse_prediction = functional.avg_pool2d(prediction, factor, factor)
        coarse_target = functional.avg_pool2d(target, factor, factor)
        coarse = torch.mean((coarse_prediction - coarse_target) ** 2)
        global_mean = torch.mean(
            (prediction.mean(dim=(-2, -1)) - target.mean(dim=(-2, -1))) ** 2
        )
        grad_x = torch.mean(
            (
                (prediction[..., 1:] - prediction[..., :-1])
                - (target[..., 1:] - target[..., :-1])
            )
            ** 2
        )
        grad_y = torch.mean(
            (
                (prediction[..., 1:, :] - prediction[..., :-1, :])
                - (target[..., 1:, :] - target[..., :-1, :])
            )
            ** 2
        )
        return (
            pixel
            + LOW_PASS_LOSS_WEIGHT * coarse
            + GLOBAL_MEAN_LOSS_WEIGHT * global_mean
            + GRADIENT_LOSS_WEIGHT * (grad_x + grad_y)
        )

    best_state: dict[str, Any] | None = None
    best_value, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            latent = model(xb)
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(latent / RESIDUAL_CAP_SIGMA)
            loss = loss_function(bounded, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            latent = model(vx)
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(latent / RESIDUAL_CAP_SIGMA)
            validation = float(loss_function(bounded, vy))
        history.append(
            {"epoch": epoch, "train_loss": running / len(tx), "validation_loss": validation}
        )
        if validation < best_value - 1.0e-7:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_value, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if stale >= 35:
            break
    if best_state is None:
        raise RuntimeError("MV10 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "training_seed": seed,
        "device": str(device),
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "loss_contract": {
            "pixel_MSE": 1.0,
            "low_pass_MSE": LOW_PASS_LOSS_WEIGHT,
            "global_mean_MSE": GLOBAL_MEAN_LOSS_WEIGHT,
            "gradient_MSE": GRADIENT_LOSS_WEIGHT,
            "low_pass_factor": LOW_PASS_FACTOR,
            "adversarial_perceptual_or_diffusion_loss": False,
        },
        "history": history,
    }


def _predict_qy(
    model: Any,
    x: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, float]:
    import torch

    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(np.float32)
    values, maximum = [], 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            latent = model(torch.from_numpy(normalized[start : start + batch_size]))
            bounded = RESIDUAL_CAP_SIGMA * torch.tanh(latent / RESIDUAL_CAP_SIGMA)
            maximum = max(maximum, float(torch.max(torch.abs(bounded))))
            raw_qy = x[start : start + batch_size, QY_INDEX : QY_INDEX + 1]
            values.append(raw_qy + bounded.numpy() * scaling["residual_std"])
    return np.concatenate(values, axis=0)[:, 0].astype(np.float32), maximum


def run_model_task(
    output_root: Path, *, training_seed: int, epochs: int, batch_size: int
) -> dict[str, Any]:
    protocol = locked_protocol()
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("MV10 training seed is outside the locked matrix")
    output_root = Path(output_root)
    assembly = json.loads((output_root / "assembly_summary.json").read_text(encoding="utf-8"))
    if assembly.get("decision") != "proceed_to_MV10_development_only_qy_models":
        raise ValueError("MV10 assembly did not authorize model training")
    directory = _task_directory(output_root, training_seed)
    directory.mkdir(parents=True, exist_ok=False)

    # Deliberately do not load test_y or test_target10 in a model task.
    with np.load(output_root / "dataset.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["train_x"])
        train_y = np.asarray(data["train_y"])
        validation_x = np.asarray(data["validation_x"])
        validation_y = np.asarray(data["validation_y"])
        test_x = np.asarray(data["test_x"])
        test_condition = np.asarray(data["test_condition"])
        test_identity = np.asarray(data["test_identity"])
    scaling = _fit_scaling(train_x, train_y)
    model, training = _train_model(
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        seed=training_seed,
        epochs=epochs,
        batch_size=batch_size,
    )
    validation_ungated, validation_bound = _predict_qy(
        model, validation_x, scaling, batch_size
    )
    legacy_ungated, legacy_bound = _predict_qy(model, test_x, scaling, batch_size)
    validation_raw = validation_x[:, QY_INDEX]
    validation_target = validation_y[:, QY_INDEX]
    alpha_records = []
    for alpha in protocol["selection_contract"]["residual_alpha_candidates"]:
        candidate = validation_raw + float(alpha) * (validation_ungated - validation_raw)
        alpha_records.append(
            {"alpha": float(alpha), "validation_qy_nrmse": component_nrmse(candidate, validation_target)}
        )
    selected = min(alpha_records, key=lambda item: (item["validation_qy_nrmse"], item["alpha"]))
    alpha = float(selected["alpha"])
    legacy_prediction = test_x[:, QY_INDEX] + alpha * (
        legacy_ungated - test_x[:, QY_INDEX]
    )

    import torch

    torch.save(
        {
            "stage": STAGE,
            "model_name": MODEL_NAME,
            "training_seed": training_seed,
            "state_dict": model.state_dict(),
            "scaling": {key: np.asarray(value) for key, value in scaling.items()},
            "residual_alpha": alpha,
            "input_fields": (
                "tau_xy_over_p_ref",
                "normal_stress_difference_over_p_ref",
                "qx_over_q_ref",
                "qy_over_q_ref",
                "rho_over_rho_ref",
                "u_over_U_lid",
                "v_over_U_lid",
                "T_over_T0",
                "log10_Kn",
                "U_lid_over_100",
            ),
            "output_field": "qy_over_q_ref",
        },
        directory / "model.pt",
    )
    np.savez_compressed(
        directory / "predictions.npz",
        identity_condition=test_condition,
        identity_numeric=test_identity,
        raw_qy=test_x[:, QY_INDEX],
        architecture_prediction_qy=legacy_prediction.astype(np.float32),
    )
    checks = {
        "development_only_training_and_selection": True,
        "legacy_targets_not_loaded_by_model_task": True,
        "finite_prediction": bool(np.all(np.isfinite(legacy_prediction))),
        "bounded_normalized_residual": max(validation_bound, legacy_bound)
        <= RESIDUAL_CAP_SIGMA + 1.0e-6,
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV10_development_only_qy_model_task",
        "model_name": MODEL_NAME,
        "training_seed": training_seed,
        "protocol_sha256": _sha256(protocol_path()),
        "training": training,
        "residual_alpha_selection_development_only": {
            "selected": alpha,
            "candidates": alpha_records,
        },
        "legacy_evaluation_metrics_intentionally_absent": True,
        "checks": checks,
        "decision": "accept_MV10_model_task" if all(checks.values()) else "hold_MV10_model_task",
    }
    _atomic_json(directory / "summary.json", summary)
    _atomic_json(
        directory / "artifact_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {
                    "sha256": _sha256(directory / name),
                    "size_bytes": (directory / name).stat().st_size,
                }
                for name in ("model.pt", "predictions.npz", "summary.json")
            },
        },
    )
    return summary


def _verify_task(directory: Path, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    _verify_manifest(directory, "artifact_manifest.json")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete_MV10_development_only_qy_model_task"
        or summary.get("decision") != "accept_MV10_model_task"
        or summary.get("model_name") != MODEL_NAME
        or int(summary.get("training_seed", -1)) != seed
        or not summary.get("legacy_evaluation_metrics_intentionally_absent")
    ):
        raise ValueError(f"MV10 task summary contract failed: {directory}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        prediction = np.asarray(data["architecture_prediction_qy"]).copy()
    return summary, prediction


def _aggregate_per_seed(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = list(records.values())
    fields = values[0]["per_field_nrmse"]
    return {
        "mean_composite_nrmse": float(np.mean([item["composite_nrmse"] for item in values])),
        "mean_heat_flux_composite_nrmse": float(
            np.mean([item["heat_flux_composite_nrmse"] for item in values])
        ),
        "mean_per_field_nrmse": {
            field: float(np.mean([item["per_field_nrmse"][field] for item in values]))
            for field in fields
        },
    }


def _qy_figure(
    output_root: Path,
    methods: Mapping[str, np.ndarray],
    reference: np.ndarray,
    q_scale: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    columns = (
        "reference",
        "raw_b1",
        "gaussian_b1",
        "tsvd_b1",
        "mv9_nafnet",
        "mv9_mamba",
        "mv10_hybrid",
        "raw_b10",
    )
    titles = (
        "Reference",
        "Raw DSMC\nB=1",
        "Gaussian\nB=1",
        "TSVD/POD\nB=1",
        "MV9 NAFNet\nB=1",
        "MV9 Mamba\nB=1",
        "MV10 hybrid\nB=1",
        "Raw DSMC\nB=10",
    )
    normalized = {"reference": reference, **methods}
    physical = {name: normalized[name] * q_scale for name in columns}
    errors = {
        name: 100.0 * (normalized[name] - reference)
        for name in columns
        if name != "reference"
    }
    physical_limit = max(
        float(np.quantile(np.concatenate([np.abs(value).ravel() for value in physical.values()]), 0.995)),
        1.0e-12,
    )
    error_limit = max(
        float(np.quantile(np.concatenate([np.abs(value).ravel() for value in errors.values()]), 0.995)),
        1.0e-4,
    )
    fig, axes = plt.subplots(2, len(columns), figsize=(17.2, 6.0), constrained_layout=True)
    physical_norm = TwoSlopeNorm(vmin=-physical_limit, vcenter=0.0, vmax=physical_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    levels = np.linspace(-physical_limit, physical_limit, 41)
    error_levels = np.linspace(-error_limit, error_limit, 41)
    physical_artist = error_artist = None
    for column, (name, title) in enumerate(zip(columns, titles)):
        physical_artist = axes[0, column].contourf(
            physical[name], levels=levels, cmap="RdBu_r", norm=physical_norm, extend="both"
        )
        axes[0, column].set_title(title, pad=7, fontsize=9.5)
        error = np.zeros_like(reference) if name == "reference" else errors[name]
        error_artist = axes[1, column].contourf(
            error, levels=error_levels, cmap="RdBu_r", norm=error_norm, extend="both"
        )
        for row in range(2):
            axis = axes[row, column]
            axis.set_aspect("equal")
            axis.set_xlim(0, reference.shape[1] - 1)
            axis.set_ylim(0, reference.shape[0] - 1)
            axis.set_xticks([0, (reference.shape[1] - 1) / 2, reference.shape[1] - 1])
            axis.set_yticks([0, (reference.shape[0] - 1) / 2, reference.shape[0] - 1])
            axis.set_xticklabels(["0", "0.5", "1"] if row == 1 else [])
            axis.set_yticklabels(["0", "0.5", "1"] if column == 0 else [])
            if row == 1:
                axis.set_xlabel(r"$x/L$")
            if column == 0:
                axis.set_ylabel(r"$y/L$")
    assert physical_artist is not None and error_artist is not None
    first = fig.colorbar(physical_artist, ax=axes[0, :], shrink=0.88, pad=0.012)
    first.set_label(r"$q_y$ [W m$^{-2}$]")
    second = fig.colorbar(error_artist, ax=axes[1, :], shrink=0.88, pad=0.012)
    second.set_label(r"$100\,\Delta q_y/q_{ref}$ [%]")
    png = Path(output_root) / "mv10_qy_B1_vs_B10_physical_contours.png"
    pdf = Path(output_root) / "mv10_qy_B1_vs_B10_physical_contours.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "png": png.name,
        "pdf": pdf.name,
        "physical_limit": physical_limit,
        "error_percent_limit": error_limit,
    }


def _create_archive(output_root: Path, names: Sequence[Path], return_directory: Path) -> Path:
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return_directory = Path(return_directory)
    return_directory.mkdir(parents=True, exist_ok=True)
    archive = return_directory / f"MV10_QY_ANALYSIS_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV10 archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in names:
            stream.write(path, arcname=str(path.relative_to(output_root)))
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV10 return archive exceeds the 450 MiB upload limit")
    return archive


def run_post(output_root: Path, return_directory: Path) -> dict[str, Any]:
    mv9 = _mv9_module()
    protocol = locked_protocol()
    output_root = Path(output_root).resolve()
    _verify_manifest(output_root, "assembly_manifest.json")
    task_summaries, qy_predictions = [], []
    for seed in TRAINING_SEEDS:
        summary, prediction = _verify_task(_task_directory(output_root, seed), seed)
        task_summaries.append(summary)
        qy_predictions.append(prediction)
    mv10_qy = np.mean(qy_predictions, axis=0).astype(np.float32)

    with np.load(output_root / "dataset.npz", allow_pickle=False) as data:
        test_x = np.asarray(data["test_x"])
        test_y = np.asarray(data["test_y"])
        conditions = np.asarray(data["test_condition"])
        identities = np.asarray(data["test_identity"])
        scales = np.asarray(data["test_scale"])
        gaussian = np.asarray(data["test_gaussian"])
        tsvd_value = np.asarray(data["test_tsvd"])
        raw10 = np.asarray(data["test_raw10"])
        target10 = np.asarray(data["test_target10"])
        conditions10 = np.asarray(data["test_condition10"])
        identities10 = np.asarray(data["test_identity10"])
        scales10 = np.asarray(data["test_scale10"])
        nafnet = np.asarray(data["mv9_nafnet_ensemble"])
        mamba = np.asarray(data["mv9_mamba_ensemble"])
    raw = test_x[:, : len(mv9.OUTPUT_FIELDS)]
    if mv10_qy.shape != raw[:, QY_INDEX].shape:
        raise ValueError("MV10 qy ensemble shape differs from legacy dataset")
    hybrid = mamba.copy()
    hybrid[:, QY_INDEX] = mv10_qy
    if not np.array_equal(hybrid[:, :QY_INDEX], mamba[:, :QY_INDEX]):
        raise ValueError("MV10 hybrid changed a preserved Mamba channel")

    methods_b1 = {
        "raw_b1": raw,
        "gaussian_b1": gaussian,
        "tsvd_b1": tsvd_value,
        "mv9_nafnet": nafnet,
        "mv9_mamba": mamba,
        "mv10_hybrid": hybrid,
    }
    per_seed = {
        method: mv9._per_seed_metrics(value, test_y, conditions, identities)
        for method, value in methods_b1.items()
    }
    per_seed["raw_b10"] = mv9._per_seed_metrics(
        raw10, target10, conditions10, identities10
    )
    aggregates: dict[str, dict[str, Any]] = {}
    for method, condition_records in per_seed.items():
        aggregates[method] = {
            condition: _aggregate_per_seed(seed_records)
            for condition, seed_records in condition_records.items()
        }

    primary = str(protocol["execution_matrix"]["primary_condition"])
    raw10_primary = aggregates["raw_b10"][primary]
    hybrid_primary = aggregates["mv10_hybrid"][primary]
    primary_qy_ratio = hybrid_primary["mean_per_field_nrmse"][mv9.OUTPUT_FIELDS[QY_INDEX]] / max(
        raw10_primary["mean_per_field_nrmse"][mv9.OUTPUT_FIELDS[QY_INDEX]], 1.0e-12
    )
    primary_composite_ratio = hybrid_primary["mean_composite_nrmse"] / max(
        raw10_primary["mean_composite_nrmse"], 1.0e-12
    )
    primary_seed_ratios = {}
    for seed, record in per_seed["mv10_hybrid"][primary].items():
        denominator = per_seed["raw_b10"][primary][seed]["per_field_nrmse"][mv9.OUTPUT_FIELDS[QY_INDEX]]
        primary_seed_ratios[seed] = record["per_field_nrmse"][mv9.OUTPUT_FIELDS[QY_INDEX]] / max(
            denominator, 1.0e-12
        )
    all_condition_qy_ratios = {}
    for condition in aggregates["mv10_hybrid"]:
        all_condition_qy_ratios[condition] = (
            aggregates["mv10_hybrid"][condition]["mean_per_field_nrmse"][mv9.OUTPUT_FIELDS[QY_INDEX]]
            / max(
                aggregates["raw_b10"][condition]["mean_per_field_nrmse"][mv9.OUTPUT_FIELDS[QY_INDEX]],
                1.0e-12,
            )
        )
    contract = protocol["analysis_contract"]
    gates = {
        "primary_mean_qy_no_worse_than_Raw_B10": primary_qy_ratio
        <= float(contract["maximum_primary_mean_qy_ratio_to_raw_B10"]),
        "primary_hybrid_all_moment_composite_within_cap": primary_composite_ratio
        <= float(contract["maximum_primary_composite_ratio_to_raw_B10"]),
        "every_primary_seed_qy_within_cap": max(primary_seed_ratios.values())
        <= float(contract["maximum_individual_seed_qy_ratio_to_raw_B10"]),
        "no_condition_mean_qy_worse_than_Raw_B10": max(all_condition_qy_ratios.values())
        <= float(contract["maximum_each_condition_mean_qy_ratio_to_raw_B10"]),
        "non_qy_channels_bitwise_preserved_from_MV9_Mamba": True,
        "legacy_diagnostic_not_reclassified_as_confirmation": True,
    }
    supports_fresh_confirmation = all(gates.values())

    metrics_path = output_root / "mv10_legacy_diagnostic_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "condition",
                "method",
                "mean_composite_nrmse",
                "mean_heat_flux_composite_nrmse",
                *mv9.OUTPUT_FIELDS,
            )
        )
        for method in sorted(aggregates):
            for condition in sorted(aggregates[method]):
                record = aggregates[method][condition]
                writer.writerow(
                    (
                        condition,
                        method,
                        record["mean_composite_nrmse"],
                        record["mean_heat_flux_composite_nrmse"],
                        *(record["mean_per_field_nrmse"][field] for field in mv9.OUTPUT_FIELDS),
                    )
                )

    representative_seed = int(protocol["execution_matrix"]["representative_contour_seed"])
    representative_block = int(protocol["execution_matrix"]["representative_contour_block"])
    mask = (
        (conditions == primary)
        & (identities[:, 0] == representative_seed)
        & (identities[:, 1] == representative_block)
    )
    mask10 = (conditions10 == primary) & (identities10[:, 0] == representative_seed)
    if np.count_nonzero(mask) != 1 or np.count_nonzero(mask10) != 1:
        raise ValueError("MV10 representative contour identity is absent")
    index, index10 = int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask10)[0])
    if not np.array_equal(test_y[index], target10[index10]):
        raise ValueError("MV10 representative B1/B10 cross-fit targets differ")
    if not np.allclose(scales[index], scales10[index10], rtol=1.0e-12, atol=0.0):
        raise ValueError("MV10 representative B1/B10 physical scales differ")
    figure = _qy_figure(
        output_root,
        {
            "raw_b1": raw[index, QY_INDEX],
            "gaussian_b1": gaussian[index, QY_INDEX],
            "tsvd_b1": tsvd_value[index, QY_INDEX],
            "mv9_nafnet": nafnet[index, QY_INDEX],
            "mv9_mamba": mamba[index, QY_INDEX],
            "mv10_hybrid": hybrid[index, QY_INDEX],
            "raw_b10": raw10[index10, QY_INDEX],
        },
        test_y[index, QY_INDEX],
        float(scales[index, QY_INDEX]),
    )

    compact_tasks = [
        {
            "training_seed": item["training_seed"],
            "training": item["training"],
            "residual_alpha_selection_development_only": item[
                "residual_alpha_selection_development_only"
            ],
            "checks": item["checks"],
        }
        for item in task_summaries
    ]
    _atomic_json(output_root / "model_task_summaries.json", {"tasks": compact_tasks})
    summary = {
        "stage": STAGE,
        "status": "complete_MV10_post_MV9_method_development_diagnostic",
        "protocol_sha256": _sha256(protocol_path()),
        "scientific_classification": protocol["scientific_role"]["classification"],
        "old_evaluation_seeds_are_confirmation": False,
        "fresh_unobserved_seed_confirmation_still_required": True,
        "primary_condition": primary,
        "primary_qy_ratio_to_raw_B10": primary_qy_ratio,
        "primary_hybrid_composite_ratio_to_raw_B10": primary_composite_ratio,
        "primary_per_seed_qy_ratios_to_raw_B10": primary_seed_ratios,
        "all_condition_mean_qy_ratios_to_raw_B10": all_condition_qy_ratios,
        "legacy_diagnostic_aggregates": aggregates,
        "gates_for_authorizing_fresh_seed_confirmation": gates,
        "figure_record": figure,
        "decision": (
            "MV10_legacy_diagnostic_supports_separately_locked_fresh_seed_confirmation"
            if supports_fresh_confirmation
            else "MV10_legacy_diagnostic_does_not_support_fresh_seed_confirmation"
        ),
    }
    _atomic_json(output_root / "summary.json", summary)
    shutil.copy2(protocol_path(), output_root / PROTOCOL_FILE)
    generated = [
        output_root / "assembly_summary.json",
        output_root / "summary.json",
        output_root / PROTOCOL_FILE,
        output_root / "model_task_summaries.json",
        metrics_path,
        output_root / figure["png"],
        output_root / figure["pdf"],
    ]
    accounting = output_root / "slurm_accounting.psv"
    if accounting.is_file():
        generated.append(accounting)
    manifest = {
        "stage": STAGE,
        "status": "complete_MV10_return_artifact_manifest",
        "files": {
            str(path.relative_to(output_root)): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(set(generated))
        },
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    generated.append(output_root / "artifact_manifest.json")
    _verify_manifest(output_root, "artifact_manifest.json")
    verification = {
        "stage": STAGE,
        "status": "complete_MV10_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "artifact_manifest.json"),
    }
    _atomic_json(output_root / "verification.json", verification)
    generated.append(output_root / "verification.json")
    archive = _create_archive(output_root, generated, Path(return_directory))
    archive_sha256 = _sha256(archive)
    result_env = Path(return_directory) / "LAST_MOHAMMADZADEH_MV10_QY_RESULT.env"
    result_env.write_text(
        f"MV10_RESULT_ARCHIVE={archive}\nMV10_RESULT_ARCHIVE_SHA256={archive_sha256}\n"
        f"MV10_RESULT_OUTPUT_ROOT={output_root}\n",
        encoding="utf-8",
    )
    print(f"MV10_OUTPUT_ROOT={output_root}")
    print(f"ARCHIVE={archive}")
    print(f"ARCHIVE_SIZE_MIB={archive.stat().st_size / 1024**2:.2f}")
    print(f"ARCHIVE_SHA256={archive_sha256}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify-lock", "assemble", "model", "post"), required=True)
    parser.add_argument("--mv9-output-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--return-directory", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "verify-lock":
        result = verify_lock()
    elif args.mode == "assemble":
        if args.mv9_output_root is None or args.output_root is None:
            parser.error("assemble requires --mv9-output-root and --output-root")
        result = run_assembly(args.mv9_output_root, args.output_root)
    elif args.mode == "model":
        if args.output_root is None or args.task_index is None:
            parser.error("model requires --output-root and --task-index")
        result = run_model_task(
            args.output_root,
            training_seed=task_from_index(args.task_index),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        if args.output_root is None or args.return_directory is None:
            parser.error("post requires --output-root and --return-directory")
        result = run_post(args.output_root, args.return_directory)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
