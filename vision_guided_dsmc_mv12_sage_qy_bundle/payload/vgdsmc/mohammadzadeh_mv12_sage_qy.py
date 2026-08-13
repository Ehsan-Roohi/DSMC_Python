"""MV12 SAGE-QY: conservative q_y stacking with uncertainty abstention.

The prediction stage deliberately does not index or materialize legacy
evaluation labels.  It locks a recursively verified prediction artifact using
development-validation data only.  A separate post stage opens the observed
MV9 legacy labels after the prediction hash is fixed.  Legacy results remain
exploratory and cannot be used as confirmation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV12_Mohammadzadeh_SAGE_qy"
STATUS = "amended_after_pre_outcome_split_failure_before_any_MV12_SAGE_output"
PROTOCOL_FILE = "mv12_sage_qy_protocol.json"
METHOD_NAME = "SAGE-QY"
QY_INDEX = 3
DEVELOPMENT_CONDITIONS = (
    "kn0p05_u100",
    "kn0p05_u200",
    "kn0p05_u400",
    "kn0p1_u100",
)
EVALUATION_CONDITIONS = (
    "kn0p075_u150",
    "kn0p075_u300",
    "kn0p1_u200",
    "kn0p1_u400",
)
EXPERT_NAMES = (
    "raw_b1",
    "gaussian_b1",
    "tsvd_b1",
    "mv9_nafnet",
    "mv9_mamba",
    "mv10_multiscale",
)


def _upstream_modules():
    from . import mohammadzadeh_mv9_heat_flux as mv9
    from . import mohammadzadeh_mv10_qy_multiscale as mv10

    return mv9, mv10


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
    _, mv10 = _upstream_modules()
    return mv10.protocol_path().parent / PROTOCOL_FILE


def locked_protocol() -> dict[str, Any]:
    mv9, mv10 = _upstream_modules()
    path = protocol_path()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != STATUS:
        raise ValueError("MV12 protocol is absent or unlocked")
    execution = value["execution_contract"]
    stacking = value["stacking_contract"]
    abstention = value["abstention_contract"]
    if (
        value.get("method_name") != METHOD_NAME
        or tuple(execution["experts"]) != EXPERT_NAMES
        or tuple(execution["development_validation_conditions"])
        != DEVELOPMENT_CONDITIONS
        or tuple(execution["conditions"]) != EVALUATION_CONDITIONS
        or execution["output_replaced"] != "qy_over_q_ref"
        or stacking["weight_scope"] != "one global vector across development conditions"
        or stacking["regularization_selection"]
        != "leave-one-development-condition-out"
        or float(stacking["minimum_relative_validation_gain_for_blend"]) != 0.005
        or float(abstention["minimum_threshold"]) != 1.0e-6
    ):
        raise ValueError("MV12 source differs from the locked method contract")
    source = value["source_contract"]
    if _sha256(Path(mv9.__file__)) != source["mv9_module_sha256"]:
        raise ValueError("MV12 MV9 ancestry hash mismatch")
    if _sha256(Path(mv10.__file__)) != source["mv10_module_sha256"]:
        raise ValueError("MV12 MV10 ancestry hash mismatch")
    return value


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV12_lock_verified",
        "protocol_status": protocol["status"],
        "protocol_amendment_count": len(protocol.get("amendment_history", [])),
        "protocol_sha256": _sha256(protocol_path()),
        "experts": list(EXPERT_NAMES),
        "selection_scope": protocol["stacking_contract"]["weight_scope"],
        "legacy_labels_forbidden_during_prediction": bool(
            protocol["scientific_role"][
                "legacy_evaluation_labels_forbidden_during_model_stage"
            ]
        ),
        "fresh_confirmation_required": True,
    }


def _condition_values(values: np.ndarray) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in np.unique(np.asarray(values).astype("U32"))))


def verify_data_contract(mv10_output_root: Path) -> dict[str, Any]:
    """Verify disjoint condition identities without loading any target array."""

    protocol = locked_protocol()
    dataset = Path(mv10_output_root).resolve() / "dataset.npz"
    with np.load(dataset, allow_pickle=False) as data:
        validation_conditions = np.asarray(data["validation_condition"])
        evaluation_conditions = np.asarray(data["test_condition"])
    observed_development = _condition_values(validation_conditions)
    observed_evaluation = _condition_values(evaluation_conditions)
    expected_development = tuple(
        sorted(protocol["execution_contract"]["development_validation_conditions"])
    )
    expected_evaluation = tuple(sorted(protocol["execution_contract"]["conditions"]))
    checks = {
        "development_condition_matrix_matches_protocol": observed_development
        == expected_development,
        "evaluation_condition_matrix_matches_protocol": observed_evaluation
        == expected_evaluation,
        "development_and_evaluation_conditions_disjoint": set(
            observed_development
        ).isdisjoint(observed_evaluation),
        "targets_not_loaded": True,
    }
    if not all(checks.values()):
        raise ValueError(
            "MV12 dataset condition contract failed: "
            f"development={observed_development}, evaluation={observed_evaluation}, "
            f"checks={checks}"
        )
    return {
        "stage": STAGE,
        "status": "MV12_condition_identity_contract_verified_without_targets",
        "development_conditions": list(observed_development),
        "evaluation_conditions": list(observed_evaluation),
        "development_sample_count": int(validation_conditions.size),
        "evaluation_sample_count": int(evaluation_conditions.size),
        "checks": checks,
    }


def _verify_manifest(root: Path, manifest_name: str) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV12 ancestry artifact verification failed: {path}")
    return manifest


def project_simplex(value: np.ndarray) -> np.ndarray:
    """Euclidean projection of a vector onto the probability simplex."""

    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("simplex projection requires a finite nonempty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=np.float64)
    positive = ordered - cumulative / indices > 0.0
    if not np.any(positive):
        raise RuntimeError("simplex projection failed to find an active set")
    rho = int(np.flatnonzero(positive)[-1])
    theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    projected /= projected.sum()
    return projected


def component_nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidate.shape != target.shape:
        raise ValueError("component metric arrays must have matching shapes")
    numerator = np.sqrt(np.mean((candidate - target) ** 2))
    denominator = np.sqrt(np.mean(target**2))
    return float(numerator / max(float(denominator), 1.0e-12))


def fit_convex_weights(
    experts: np.ndarray,
    target: np.ndarray,
    anchor_index: int,
    normalized_ridge_strength: float,
    *,
    maximum_iterations: int = 3000,
    tolerance: float = 1.0e-11,
) -> np.ndarray:
    """Fit deterministic nonnegative, sum-one scalar expert weights."""

    experts = np.asarray(experts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if experts.ndim < 3 or experts.shape[0] != len(EXPERT_NAMES):
        raise ValueError("expert tensor must have leading expert dimension")
    if experts.shape[1:] != target.shape:
        raise ValueError("expert and target shapes differ")
    if not 0 <= anchor_index < experts.shape[0]:
        raise ValueError("anchor expert index is invalid")
    design = np.moveaxis(experts, 0, -1).reshape(-1, experts.shape[0])
    response = target.reshape(-1)
    gram = design.T @ design / max(len(response), 1)
    cross = design.T @ response / max(len(response), 1)
    scale = max(float(np.trace(gram) / experts.shape[0]), 1.0e-12)
    ridge = float(normalized_ridge_strength) * scale
    anchor = np.zeros(experts.shape[0], dtype=np.float64)
    anchor[anchor_index] = 1.0
    hessian = gram + ridge * np.eye(experts.shape[0])
    right = cross + ridge * anchor
    lipschitz = max(float(2.0 * np.linalg.eigvalsh(hessian)[-1]), 1.0e-12)
    weights = anchor.copy()
    for _ in range(int(maximum_iterations)):
        gradient = 2.0 * (hessian @ weights - right)
        updated = project_simplex(weights - gradient / lipschitz)
        if float(np.max(np.abs(updated - weights))) <= tolerance:
            weights = updated
            break
        weights = updated
    if np.min(weights) < -1.0e-12 or abs(float(weights.sum()) - 1.0) > 1.0e-10:
        raise RuntimeError("convex stacking weights violate the simplex")
    return weights


def _weighted_prediction(experts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    experts = np.asarray(experts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return np.tensordot(weights, experts, axes=(0, 0)).astype(np.float32)


def disagreement_scores(experts: np.ndarray) -> np.ndarray:
    """Return target-free samplewise relative RMS expert disagreement."""

    experts = np.asarray(experts, dtype=np.float64)
    if experts.ndim != 4:
        raise ValueError("disagreement expects [expert,sample,y,x]")
    mean = np.mean(experts, axis=0)
    spread = np.sqrt(np.mean((experts - mean[None]) ** 2, axis=(0, 2, 3)))
    scale = np.sqrt(np.mean(mean**2, axis=(1, 2)))
    return spread / np.maximum(scale, 1.0e-12)


def _anchor_index(experts: np.ndarray, target: np.ndarray) -> tuple[int, list[float]]:
    errors = [component_nrmse(experts[index], target) for index in range(experts.shape[0])]
    return int(min(range(len(errors)), key=lambda index: (errors[index], index))), errors


def fit_global_gate(
    validation_experts: np.ndarray,
    validation_target: np.ndarray,
    validation_conditions: np.ndarray,
    test_experts: np.ndarray,
    test_conditions: np.ndarray,
    protocol: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one transferable gate and apply target-free test abstention."""

    validation_experts = np.asarray(validation_experts, dtype=np.float64)
    validation_target = np.asarray(validation_target, dtype=np.float64)
    validation_conditions = np.asarray(validation_conditions).astype("U32")
    test_experts = np.asarray(test_experts, dtype=np.float64)
    test_conditions = np.asarray(test_conditions).astype("U32")
    if validation_experts.ndim != 4 or test_experts.ndim != 4:
        raise ValueError("global gate experts must be [expert,sample,y,x]")
    if validation_experts.shape[0] != test_experts.shape[0]:
        raise ValueError("validation and test expert counts differ")
    if validation_experts.shape[1:] != validation_target.shape:
        raise ValueError("validation expert and target shapes differ")
    if validation_conditions.shape != (validation_target.shape[0],):
        raise ValueError("validation condition identity shape differs")
    if test_conditions.shape != (test_experts.shape[1],):
        raise ValueError("test condition identity shape differs")
    validation_groups = tuple(sorted(str(item) for item in np.unique(validation_conditions)))
    evaluation_groups = tuple(sorted(str(item) for item in np.unique(test_conditions)))
    if len(validation_groups) < 2 or not evaluation_groups:
        raise ValueError("global transfer gate requires multiple development conditions")
    if set(validation_groups) & set(evaluation_groups):
        raise ValueError("development and evaluation condition sets must be disjoint")

    stacking = protocol["stacking_contract"]
    anchor, expert_errors = _anchor_index(validation_experts, validation_target)
    ridge_candidates = [
        float(item) for item in stacking["normalized_ridge_strength_candidates"]
    ]
    loo_records = []
    for strength in ridge_candidates:
        squared_error = 0.0
        squared_target = 0.0
        condition_errors: dict[str, float] = {}
        for held_out in validation_groups:
            keep = validation_conditions != held_out
            held = ~keep
            fold_anchor, _ = _anchor_index(
                validation_experts[:, keep], validation_target[keep]
            )
            fold_weights = fit_convex_weights(
                validation_experts[:, keep],
                validation_target[keep],
                fold_anchor,
                strength,
                maximum_iterations=int(stacking["maximum_optimizer_iterations"]),
                tolerance=float(stacking["optimizer_tolerance"]),
            )
            held_prediction = _weighted_prediction(
                validation_experts[:, held], fold_weights
            )
            held_target = validation_target[held]
            condition_squared_error = float(
                np.sum((held_prediction - held_target) ** 2)
            )
            condition_squared_target = float(np.sum(held_target**2))
            squared_error += condition_squared_error
            squared_target += condition_squared_target
            condition_errors[held_out] = float(
                np.sqrt(
                    condition_squared_error / max(condition_squared_target, 1.0e-24)
                )
            )
        loo_records.append(
            {
                "normalized_ridge_strength": strength,
                "leave_one_condition_out_qy_nrmse": float(
                    np.sqrt(squared_error / max(squared_target, 1.0e-24))
                ),
                "held_out_condition_qy_nrmse": condition_errors,
            }
        )
    selected = min(
        loo_records,
        key=lambda item: (
            item["leave_one_condition_out_qy_nrmse"],
            item["normalized_ridge_strength"],
        ),
    )
    weights = fit_convex_weights(
        validation_experts,
        validation_target,
        anchor,
        float(selected["normalized_ridge_strength"]),
        maximum_iterations=int(stacking["maximum_optimizer_iterations"]),
        tolerance=float(stacking["optimizer_tolerance"]),
    )
    validation_blend = _weighted_prediction(validation_experts, weights)
    blend_error = component_nrmse(validation_blend, validation_target)
    anchor_error = expert_errors[anchor]
    minimum_gain = float(stacking["minimum_relative_validation_gain_for_blend"])
    fallback_to_anchor = blend_error > anchor_error * (1.0 - minimum_gain)
    if fallback_to_anchor:
        weights = np.zeros(len(EXPERT_NAMES), dtype=np.float64)
        weights[anchor] = 1.0
        validation_blend = validation_experts[anchor].astype(np.float32)
        blend_error = anchor_error

    validation_scores = disagreement_scores(validation_experts)
    test_scores = disagreement_scores(test_experts)
    abstention = protocol["abstention_contract"]
    threshold = max(
        float(abstention["minimum_threshold"]),
        float(np.quantile(validation_scores, 0.95)) * 1.5,
    )
    test_prediction = _weighted_prediction(test_experts, weights)
    abstained = test_scores > threshold
    if np.any(abstained):
        test_prediction[abstained] = test_experts[anchor, abstained]

    record = {
        "weight_scope": "global_across_disjoint_development_and_evaluation_conditions",
        "development_validation_conditions": list(validation_groups),
        "evaluation_conditions": list(evaluation_groups),
        "development_validation_sample_count": int(validation_target.shape[0]),
        "anchor_expert": EXPERT_NAMES[anchor],
        "anchor_validation_qy_nrmse": anchor_error,
        "expert_validation_qy_nrmse": {
            name: expert_errors[index] for index, name in enumerate(EXPERT_NAMES)
        },
        "selected_normalized_ridge_strength": float(
            selected["normalized_ridge_strength"]
        ),
        "leave_one_development_condition_out_selection": loo_records,
        "final_weights": {
            name: float(weights[index]) for index, name in enumerate(EXPERT_NAMES)
        },
        "validation_blend_qy_nrmse": blend_error,
        "fallback_to_anchor_for_insufficient_validation_gain": fallback_to_anchor,
        "validation_disagreement_95pct": float(np.quantile(validation_scores, 0.95)),
        "abstention_threshold": threshold,
        "test_sample_count": int(test_experts.shape[1]),
        "test_abstention_count": int(np.count_nonzero(abstained)),
        "test_abstention_indices": [int(item) for item in np.flatnonzero(abstained)],
        "test_abstention_count_by_condition": {
            condition: int(np.count_nonzero(abstained[test_conditions == condition]))
            for condition in evaluation_groups
        },
        "simplex_verified": bool(
            np.min(weights) >= -1.0e-12 and abs(float(weights.sum()) - 1.0) <= 1.0e-10
        ),
    }
    return test_prediction.astype(np.float32), record


def _load_mv9_architecture_predictions(
    mv9_root: Path,
    architecture: str,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mv9, _ = _upstream_modules()
    import torch

    validation_members, test_members, member_records = [], [], []
    for seed in mv9.TRAINING_SEEDS:
        directory = mv9._task_directory(mv9_root, architecture, seed)
        _verify_manifest(directory, "artifact_manifest.json")
        checkpoint = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("stage") != mv9.STAGE
            or checkpoint.get("architecture") != architecture
            or int(checkpoint.get("training_seed", -1)) != seed
        ):
            raise ValueError(f"MV12 MV9 checkpoint identity mismatch: {directory}")
        model = mv9._project_modules()["mv6"].build_architecture(
            architecture,
            int(validation_x.shape[1]),
            out_channels=len(mv9.OUTPUT_FIELDS),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scaling = {key: np.asarray(value) for key, value in checkpoint["scaling"].items()}
        validation_ungated, _ = mv9._predict(
            model, validation_x, scaling, batch_size
        )
        test_ungated, _ = mv9._predict(model, test_x, scaling, batch_size)
        alpha = float(checkpoint["residual_alpha"])
        validation_raw = validation_x[:, : len(mv9.OUTPUT_FIELDS)]
        test_raw = test_x[:, : len(mv9.OUTPUT_FIELDS)]
        validation_prediction = validation_raw + alpha * (
            validation_ungated - validation_raw
        )
        test_prediction = test_raw + alpha * (test_ungated - test_raw)
        validation_members.append(validation_prediction.astype(np.float32))
        test_members.append(test_prediction.astype(np.float32))
        member_records.append(
            {
                "training_seed": int(seed),
                "model_sha256": _sha256(directory / "model.pt"),
                "stored_prediction_sha256": _sha256(directory / "predictions.npz"),
                "residual_alpha": alpha,
                "task_artifact_manifest_verified": True,
            }
        )
    return (
        np.mean(validation_members, axis=0).astype(np.float32),
        np.mean(test_members, axis=0).astype(np.float32),
        {"architecture": architecture, "members": member_records},
    )


def _load_mv10_predictions(
    mv10_root: Path,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _, mv10 = _upstream_modules()
    import torch

    validation_members, test_members, member_records = [], [], []
    for seed in mv10.TRAINING_SEEDS:
        directory = mv10._task_directory(mv10_root, seed)
        _verify_manifest(directory, "artifact_manifest.json")
        checkpoint = torch.load(
            directory / "model.pt", map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("stage") != mv10.STAGE
            or checkpoint.get("model_name") != mv10.MODEL_NAME
            or int(checkpoint.get("training_seed", -1)) != seed
        ):
            raise ValueError(f"MV12 MV10 checkpoint identity mismatch: {directory}")
        model = mv10._build_model(int(validation_x.shape[1]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        scaling = {key: np.asarray(value) for key, value in checkpoint["scaling"].items()}
        validation_ungated, _ = mv10._predict_qy(
            model, validation_x, scaling, batch_size
        )
        test_ungated, _ = mv10._predict_qy(model, test_x, scaling, batch_size)
        alpha = float(checkpoint["residual_alpha"])
        validation_raw = validation_x[:, QY_INDEX]
        test_raw = test_x[:, QY_INDEX]
        validation_prediction = validation_raw + alpha * (
            validation_ungated - validation_raw
        )
        test_prediction = test_raw + alpha * (test_ungated - test_raw)
        with np.load(directory / "predictions.npz", allow_pickle=False) as data:
            stored_test = np.asarray(data["architecture_prediction_qy"])
        if not np.allclose(test_prediction, stored_test, rtol=2.0e-6, atol=2.0e-7):
            raise ValueError(f"MV12 reload differs from stored MV10 prediction: {directory}")
        validation_members.append(validation_prediction.astype(np.float32))
        test_members.append(test_prediction.astype(np.float32))
        member_records.append(
            {
                "training_seed": int(seed),
                "model_sha256": _sha256(directory / "model.pt"),
                "stored_prediction_sha256": _sha256(directory / "predictions.npz"),
                "residual_alpha": alpha,
                "task_artifact_manifest_verified": True,
            }
        )
    return (
        np.mean(validation_members, axis=0).astype(np.float32),
        np.mean(test_members, axis=0).astype(np.float32),
        {"architecture": "mv10_multiscale", "members": member_records},
    )


def run_prediction_stage(
    mv10_output_root: Path,
    output_root: Path,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Lock SAGE predictions without loading legacy test targets."""

    mv9, mv10 = _upstream_modules()
    protocol = locked_protocol()
    mv10_output_root = Path(mv10_output_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite MV12 output: {output_root}")
    data_contract = verify_data_contract(mv10_output_root)
    mv10_summary = json.loads(
        (mv10_output_root / "summary.json").read_text(encoding="utf-8")
    )
    mv10_verification = json.loads(
        (mv10_output_root / "verification.json").read_text(encoding="utf-8")
    )
    mv10_assembly = json.loads(
        (mv10_output_root / "assembly_summary.json").read_text(encoding="utf-8")
    )
    required_mv10 = protocol["source_contract"]["required_mv10_decision"]
    if (
        mv10_summary.get("decision") != required_mv10
        or mv10_verification.get("decision") != "verified"
    ):
        raise ValueError("MV12 requires the verified completed MV10 failure result")
    _verify_manifest(mv10_output_root, "assembly_manifest.json")
    _verify_manifest(mv10_output_root, "artifact_manifest.json")

    mv9_root = Path(mv10_assembly["mv9_output_root"]).resolve()
    mv9_summary = json.loads((mv9_root / "summary.json").read_text(encoding="utf-8"))
    if mv9_summary.get("decision") != protocol["source_contract"]["required_mv9_decision"]:
        raise ValueError("MV12 MV9 ancestry is not the locked failure result")
    _verify_manifest(mv9_root, "assembly_manifest.json")
    _verify_manifest(mv9_root, "artifact_manifest.json")

    # Deliberately omit test_y, test_target10, and all legacy metrics here.
    with np.load(mv10_output_root / "dataset.npz", allow_pickle=False) as data:
        validation_x = np.asarray(data["validation_x"])
        validation_y = np.asarray(data["validation_y"])
        validation_conditions = np.asarray(data["validation_condition"])
        test_x = np.asarray(data["test_x"])
        test_conditions = np.asarray(data["test_condition"])
        test_identity = np.asarray(data["test_identity"])
        stored_gaussian = np.asarray(data["test_gaussian"])
        stored_tsvd = np.asarray(data["test_tsvd"])
        stored_nafnet = np.asarray(data["mv9_nafnet_ensemble"])
        stored_mamba = np.asarray(data["mv9_mamba_ensemble"])

    mv9_assembly = json.loads(
        (mv9_root / "assembly_summary.json").read_text(encoding="utf-8")
    )
    classical = mv9_assembly["classical_selection_development_only"]
    modules = mv9._project_modules()
    validation_raw = validation_x[:, : len(mv9.OUTPUT_FIELDS)]
    test_raw = test_x[:, : len(mv9.OUTPUT_FIELDS)]
    validation_gaussian = modules["gaussian_like"](
        validation_raw, int(classical["gaussian_passes"])
    ).astype(np.float32)
    validation_tsvd = modules["tsvd"](
        validation_raw, int(classical["tsvd_rank"])
    ).astype(np.float32)
    test_gaussian = modules["gaussian_like"](
        test_raw, int(classical["gaussian_passes"])
    ).astype(np.float32)
    test_tsvd = modules["tsvd"](
        test_raw, int(classical["tsvd_rank"])
    ).astype(np.float32)
    if not np.allclose(test_gaussian, stored_gaussian, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("MV12 Gaussian reconstruction differs from MV9 dataset")
    if not np.allclose(test_tsvd, stored_tsvd, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("MV12 TSVD reconstruction differs from MV9 dataset")

    validation_nafnet, test_nafnet, nafnet_record = _load_mv9_architecture_predictions(
        mv9_root,
        "nafnet_small",
        validation_x,
        test_x,
        batch_size=batch_size,
    )
    validation_mamba, test_mamba, mamba_record = _load_mv9_architecture_predictions(
        mv9_root,
        "mambairv2_tiny_adapted",
        validation_x,
        test_x,
        batch_size=batch_size,
    )
    if not np.allclose(test_nafnet, stored_nafnet, rtol=2.0e-6, atol=2.0e-7):
        raise ValueError("MV12 NAFNet ensemble differs from staged MV9 ensemble")
    if not np.allclose(test_mamba, stored_mamba, rtol=2.0e-6, atol=2.0e-7):
        raise ValueError("MV12 Mamba ensemble differs from staged MV9 ensemble")
    validation_mv10, test_mv10, mv10_record = _load_mv10_predictions(
        mv10_output_root,
        validation_x,
        test_x,
        batch_size=batch_size,
    )

    validation_experts = np.stack(
        (
            validation_raw[:, QY_INDEX],
            validation_gaussian[:, QY_INDEX],
            validation_tsvd[:, QY_INDEX],
            validation_nafnet[:, QY_INDEX],
            validation_mamba[:, QY_INDEX],
            validation_mv10,
        )
    )
    test_experts = np.stack(
        (
            test_raw[:, QY_INDEX],
            test_gaussian[:, QY_INDEX],
            test_tsvd[:, QY_INDEX],
            test_nafnet[:, QY_INDEX],
            test_mamba[:, QY_INDEX],
            test_mv10,
        )
    )
    observed_development = _condition_values(validation_conditions)
    observed_evaluation = _condition_values(test_conditions)
    expected_development = tuple(
        sorted(protocol["execution_contract"]["development_validation_conditions"])
    )
    expected_evaluation = tuple(sorted(protocol["execution_contract"]["conditions"]))
    if (
        observed_development != expected_development
        or observed_evaluation != expected_evaluation
    ):
        raise ValueError(
            "MV12 development/evaluation condition matrix mismatch: "
            f"development={observed_development}, evaluation={observed_evaluation}"
        )
    sage_qy, global_gate = fit_global_gate(
        validation_experts,
        validation_y[:, QY_INDEX],
        validation_conditions,
        test_experts,
        test_conditions,
        protocol,
    )

    if not np.all(np.isfinite(sage_qy)):
        raise ValueError("MV12 produced a nonfinite SAGE prediction")
    output_root.mkdir(parents=True)
    np.savez_compressed(
        output_root / "locked_predictions.npz",
        sage_qy=sage_qy,
        mv10_qy=test_mv10,
        identity_condition=test_conditions,
        identity_numeric=test_identity,
    )
    shutil.copy2(protocol_path(), output_root / PROTOCOL_FILE)
    model_summary = {
        "stage": STAGE,
        "status": "complete_MV12_prediction_lock_before_legacy_label_access",
        "protocol_sha256": _sha256(protocol_path()),
        "mv10_output_root": str(mv10_output_root),
        "mv9_output_root": str(mv9_root),
        "legacy_test_targets_loaded": False,
        "condition_identity_contract": data_contract,
        "expert_ancestry": [nafnet_record, mamba_record, mv10_record],
        "classical_selection_inherited_from_MV9_development": {
            "gaussian_passes": int(classical["gaussian_passes"]),
            "tsvd_rank": int(classical["tsvd_rank"]),
        },
        "global_transfer_gate": global_gate,
        "checks": {
            "all_upstream_manifests_recursively_verified": True,
            "stored_expert_predictions_reproduced": True,
            "global_weights_on_simplex": bool(global_gate["simplex_verified"]),
            "development_and_evaluation_condition_matrices_verified_disjoint": bool(
                set(observed_development).isdisjoint(observed_evaluation)
            ),
            "legacy_test_targets_not_loaded": True,
            "finite_prediction": True,
        },
        "decision": "lock_MV12_SAGE_predictions_for_separate_legacy_diagnostic",
    }
    _atomic_json(output_root / "model_summary.json", model_summary)
    prediction_manifest = {
        "stage": STAGE,
        "status": "complete_MV12_prediction_manifest",
        "files": {
            name: {
                "sha256": _sha256(output_root / name),
                "size_bytes": (output_root / name).stat().st_size,
            }
            for name in (
                "locked_predictions.npz",
                "model_summary.json",
                PROTOCOL_FILE,
            )
        },
    }
    _atomic_json(output_root / "prediction_manifest.json", prediction_manifest)
    _verify_manifest(output_root, "prediction_manifest.json")
    return model_summary


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
        {"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    columns = (
        "reference",
        "raw_b1",
        "gaussian_b1",
        "tsvd_b1",
        "mv9_nafnet",
        "mv9_mamba",
        "mv10_hybrid",
        "mv12_sage",
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
        "MV12 SAGE-QY\nB=1",
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
        float(
            np.quantile(
                np.concatenate([np.abs(value).ravel() for value in physical.values()]),
                0.995,
            )
        ),
        1.0e-12,
    )
    error_limit = max(
        float(
            np.quantile(
                np.concatenate([np.abs(value).ravel() for value in errors.values()]),
                0.995,
            )
        ),
        1.0e-4,
    )
    fig, axes = plt.subplots(2, len(columns), figsize=(19.2, 6.0), constrained_layout=True)
    physical_norm = TwoSlopeNorm(vmin=-physical_limit, vcenter=0.0, vmax=physical_limit)
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    levels = np.linspace(-physical_limit, physical_limit, 41)
    error_levels = np.linspace(-error_limit, error_limit, 41)
    physical_artist = error_artist = None
    for column, (name, title) in enumerate(zip(columns, titles)):
        physical_artist = axes[0, column].contourf(
            physical[name], levels=levels, cmap="RdBu_r", norm=physical_norm, extend="both"
        )
        axes[0, column].set_title(title, pad=7, fontsize=9.3)
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
    png = Path(output_root) / "mv12_sage_qy_B1_vs_B10_physical_contours.png"
    pdf = Path(output_root) / "mv12_sage_qy_B1_vs_B10_physical_contours.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "png": png.name,
        "pdf": pdf.name,
        "physical_limit": physical_limit,
        "error_percent_limit": error_limit,
    }


def run_legacy_post(output_root: Path) -> dict[str, Any]:
    """Evaluate the already locked prediction against observed legacy labels."""

    mv9, _ = _upstream_modules()
    protocol = locked_protocol()
    output_root = Path(output_root).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    model_summary = json.loads(
        (output_root / "model_summary.json").read_text(encoding="utf-8")
    )
    if model_summary.get("legacy_test_targets_loaded") is not False:
        raise ValueError("MV12 prediction stage did not preserve label separation")
    mv10_root = Path(model_summary["mv10_output_root"]).resolve()
    with np.load(output_root / "locked_predictions.npz", allow_pickle=False) as data:
        sage_qy = np.asarray(data["sage_qy"])
        mv10_qy = np.asarray(data["mv10_qy"])
        locked_conditions = np.asarray(data["identity_condition"])
        locked_identities = np.asarray(data["identity_numeric"])
    with np.load(mv10_root / "dataset.npz", allow_pickle=False) as data:
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
    if not np.array_equal(conditions, locked_conditions) or not np.array_equal(
        identities, locked_identities
    ):
        raise ValueError("MV12 locked prediction identities differ from legacy dataset")
    raw = test_x[:, : len(mv9.OUTPUT_FIELDS)]
    mv10_hybrid = mamba.copy()
    mv10_hybrid[:, QY_INDEX] = mv10_qy
    sage_hybrid = mamba.copy()
    sage_hybrid[:, QY_INDEX] = sage_qy
    if not np.array_equal(sage_hybrid[:, :QY_INDEX], mamba[:, :QY_INDEX]):
        raise ValueError("MV12 changed a protected non-qy channel")

    methods_b1 = {
        "raw_b1": raw,
        "gaussian_b1": gaussian,
        "tsvd_b1": tsvd_value,
        "mv9_nafnet": nafnet,
        "mv9_mamba": mamba,
        "mv10_hybrid": mv10_hybrid,
        "mv12_sage": sage_hybrid,
    }
    per_seed = {
        method: mv9._per_seed_metrics(value, test_y, conditions, identities)
        for method, value in methods_b1.items()
    }
    per_seed["raw_b10"] = mv9._per_seed_metrics(
        raw10, target10, conditions10, identities10
    )
    aggregates = {
        method: {
            condition: _aggregate_per_seed(seed_records)
            for condition, seed_records in condition_records.items()
        }
        for method, condition_records in per_seed.items()
    }
    primary = str(protocol["execution_contract"]["primary_condition"])
    qy_name = mv9.OUTPUT_FIELDS[QY_INDEX]
    raw10_primary = aggregates["raw_b10"][primary]
    sage_primary = aggregates["mv12_sage"][primary]
    primary_qy_ratio = sage_primary["mean_per_field_nrmse"][qy_name] / max(
        raw10_primary["mean_per_field_nrmse"][qy_name], 1.0e-12
    )
    primary_composite_ratio = sage_primary["mean_composite_nrmse"] / max(
        raw10_primary["mean_composite_nrmse"], 1.0e-12
    )
    primary_seed_ratios = {
        seed: record["per_field_nrmse"][qy_name]
        / max(per_seed["raw_b10"][primary][seed]["per_field_nrmse"][qy_name], 1.0e-12)
        for seed, record in per_seed["mv12_sage"][primary].items()
    }
    condition_ratios = {
        condition: aggregates["mv12_sage"][condition]["mean_per_field_nrmse"][qy_name]
        / max(aggregates["raw_b10"][condition]["mean_per_field_nrmse"][qy_name], 1.0e-12)
        for condition in aggregates["mv12_sage"]
    }
    analysis = protocol["analysis_contract"]
    gates = {
        "primary_mean_qy_no_worse_than_Raw_B10": primary_qy_ratio
        <= float(analysis["maximum_primary_mean_qy_ratio_to_raw_B10"]),
        "primary_all_moment_composite_within_cap": primary_composite_ratio
        <= float(analysis["maximum_primary_composite_ratio_to_raw_B10"]),
        "every_primary_seed_qy_within_cap": max(primary_seed_ratios.values())
        <= float(analysis["maximum_individual_seed_qy_ratio_to_raw_B10"]),
        "no_condition_mean_qy_worse_than_Raw_B10": max(condition_ratios.values())
        <= float(analysis["maximum_each_condition_mean_qy_ratio_to_raw_B10"]),
        "non_qy_channels_bitwise_preserved_from_MV9_Mamba": True,
        "prediction_hash_locked_before_legacy_label_access": True,
        "legacy_diagnostic_not_reclassified_as_confirmation": True,
    }
    supports_fresh_confirmation = all(gates.values())

    metrics_path = output_root / "mv12_legacy_diagnostic_metrics.csv"
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

    representative_seed = int(protocol["execution_contract"]["representative_seed"])
    representative_block = int(protocol["execution_contract"]["representative_block"])
    mask = (
        (conditions == primary)
        & (identities[:, 0] == representative_seed)
        & (identities[:, 1] == representative_block)
    )
    mask10 = (conditions10 == primary) & (identities10[:, 0] == representative_seed)
    if np.count_nonzero(mask) != 1 or np.count_nonzero(mask10) != 1:
        raise ValueError("MV12 representative identity is absent")
    index, index10 = int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask10)[0])
    if not np.array_equal(test_y[index], target10[index10]):
        raise ValueError("MV12 representative B1/B10 targets differ")
    if not np.allclose(scales[index], scales10[index10], rtol=1.0e-12, atol=0.0):
        raise ValueError("MV12 representative B1/B10 scales differ")
    figure = _qy_figure(
        output_root,
        {
            "raw_b1": raw[index, QY_INDEX],
            "gaussian_b1": gaussian[index, QY_INDEX],
            "tsvd_b1": tsvd_value[index, QY_INDEX],
            "mv9_nafnet": nafnet[index, QY_INDEX],
            "mv9_mamba": mamba[index, QY_INDEX],
            "mv10_hybrid": mv10_hybrid[index, QY_INDEX],
            "mv12_sage": sage_hybrid[index, QY_INDEX],
            "raw_b10": raw10[index10, QY_INDEX],
        },
        test_y[index, QY_INDEX],
        float(scales[index, QY_INDEX]),
    )
    summary = {
        "stage": STAGE,
        "status": "complete_MV12_post_lock_legacy_diagnostic",
        "protocol_sha256": _sha256(protocol_path()),
        "scientific_classification": protocol["scientific_role"]["classification"],
        "old_evaluation_seeds_are_confirmation": False,
        "fresh_unobserved_seed_confirmation_still_required": True,
        "primary_condition": primary,
        "primary_qy_ratio_to_raw_B10": primary_qy_ratio,
        "primary_composite_ratio_to_raw_B10": primary_composite_ratio,
        "primary_per_seed_qy_ratios_to_raw_B10": primary_seed_ratios,
        "all_condition_mean_qy_ratios_to_raw_B10": condition_ratios,
        "legacy_diagnostic_aggregates": aggregates,
        "gates_for_authorizing_fresh_seed_confirmation": gates,
        "figure_record": figure,
        "decision": (
            "MV12_SAGE_legacy_diagnostic_supports_separately_locked_fresh_seed_confirmation"
            if supports_fresh_confirmation
            else "MV12_SAGE_legacy_diagnostic_does_not_support_fresh_seed_confirmation"
        ),
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    return_directory = Path(return_directory).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    generated = [
        output_root / "model_summary.json",
        output_root / "prediction_manifest.json",
        output_root / "summary.json",
        output_root / PROTOCOL_FILE,
        output_root / "mv12_legacy_diagnostic_metrics.csv",
        output_root / "mv12_sage_qy_B1_vs_B10_physical_contours.png",
        output_root / "mv12_sage_qy_B1_vs_B10_physical_contours.pdf",
    ]
    accounting = output_root / "slurm_accounting.psv"
    if accounting.is_file():
        generated.append(accounting)
    manifest = {
        "stage": STAGE,
        "status": "complete_MV12_return_artifact_manifest",
        "files": {
            str(path.relative_to(output_root)): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in generated
        },
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    _verify_manifest(output_root, "artifact_manifest.json")
    verification = {
        "stage": STAGE,
        "status": "complete_MV12_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "artifact_manifest.json"),
    }
    _atomic_json(output_root / "verification.json", verification)
    generated.extend((output_root / "artifact_manifest.json", output_root / "verification.json"))
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return_directory.mkdir(parents=True, exist_ok=True)
    archive = return_directory / f"MV12_SAGE_QY_ANALYSIS_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV12 archive: {archive}")
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as stream:
        for path in generated:
            stream.write(path, arcname=str(path.relative_to(output_root)))
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV12 return archive exceeds the 450 MiB upload limit")
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "decision": summary["decision"],
        "primary_qy_ratio_to_raw_B10": summary["primary_qy_ratio_to_raw_B10"],
    }
    _atomic_json(output_root / "return.json", result)
    pointer = return_directory / "LAST_MOHAMMADZADEH_MV12_SAGE_QY_RESULT.env"
    pointer.write_text(
        "\n".join(
            (
                f"MV12_OUTPUT_ROOT={output_root}",
                f"MV12_RESULT_ARCHIVE={archive}",
                f"MV12_RESULT_ARCHIVE_SHA256={result['archive_sha256']}",
                f"MV12_DECISION={result['decision']}",
                f"MV12_PRIMARY_QY_RATIO_TO_RAW_B10={result['primary_qy_ratio_to_raw_B10']}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-lock")
    verify.set_defaults(action="verify")
    verify_data = subparsers.add_parser("verify-data")
    verify_data.add_argument("--mv10-output-root", required=True, type=Path)
    verify_data.set_defaults(action="verify_data")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--mv10-output-root", required=True, type=Path)
    predict.add_argument("--output-root", required=True, type=Path)
    predict.add_argument("--batch-size", type=int, default=8)
    predict.set_defaults(action="predict")
    post = subparsers.add_parser("post")
    post.add_argument("--output-root", required=True, type=Path)
    post.set_defaults(action="post")
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", required=True, type=Path)
    package.add_argument("--return-directory", required=True, type=Path)
    package.set_defaults(action="package")
    args = parser.parse_args()
    if args.action == "verify":
        value = verify_lock()
    elif args.action == "verify_data":
        value = verify_data_contract(args.mv10_output_root)
    elif args.action == "predict":
        value = run_prediction_stage(
            args.mv10_output_root,
            args.output_root,
            batch_size=args.batch_size,
        )
    elif args.action == "post":
        value = run_legacy_post(args.output_root)
    else:
        value = package_results(args.output_root, args.return_directory)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
