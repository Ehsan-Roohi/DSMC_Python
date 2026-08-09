"""MV4 bounded and condition-aware repair of the Mohammadzadeh benchmark.

MV4 keeps the locked MV3 folds, targets, fields, and classical baselines.  It
changes only the learned reconstruction contract:

* the neural residual is bounded in units of the training residual standard
  deviation;
* the residual amplitude is selected on validation conditions only;
* a coordinate-support gate detects extrapolation without using test targets;
* extrapolation falls back exactly to the raw identity reconstruction;
* a fixed, broad physical envelope prevents non-finite or impossible fields.

Heat flux remains excluded.  MV4 is a diagnostic/repair benchmark and does not
rewrite or reinterpret any MV3 artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tarfile
import time
from typing import Any, Mapping

import numpy as np

from .mohammadzadeh_vision import OUTPUT_FIELDS, _loss, build_model, fit_scaling
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
from . import mohammadzadeh_vision_mv3 as mv3


STAGE = "MV4_Mohammadzadeh_bounded_condition_aware_reconstruction"
METHODS = (
    "raw",
    "gaussian_like",
    "tsvd_pod_type",
    "vision_bounded",
    "vision_safe",
)
FALLBACK_METHODS = ("raw", "gaussian_like", "tsvd_pod_type")
RESIDUAL_CAP_SIGMA = 4.0
PHYSICAL_T_MIN_K = 1.0
PHYSICAL_T_MAX_K = 2000.0
PHYSICAL_U_LID_MULTIPLIER = 2.0
SUPPORT_TOLERANCE = 1.0e-7
PROTOCOL_FILE = "mv4_stability_repair_protocol.json"


def protocol_path() -> Path:
    return mv3.reference_directory() / PROTOCOL_FILE


def locked_mv4_protocol() -> dict[str, Any]:
    value = json.loads(protocol_path().read_text(encoding="utf-8"))
    if value.get("stage") != STAGE or value.get("status") != "locked_before_MV4_outcomes":
        raise ValueError("MV4 protocol is missing or not locked before outcomes")
    source = value["source_contract"]
    bounded = value["bounded_vision_contract"]
    support = value["support_contract"]
    envelope = value["physical_envelope"]
    if tuple(source["budget_blocks"]) != BUDGETS or tuple(source["output_fields"]) != OUTPUT_FIELDS:
        raise ValueError("MV4 source contract differs from code")
    if float(bounded["residual_cap_sigma"]) != RESIDUAL_CAP_SIGMA:
        raise ValueError("MV4 residual cap differs from locked protocol")
    if support["rule"] != "coordinate_wise_training_range_no_test_target" or float(support["tolerance"]) != SUPPORT_TOLERANCE:
        raise ValueError("MV4 support gate differs from locked protocol")
    if (
        float(envelope["temperature_min_K"]) != PHYSICAL_T_MIN_K
        or float(envelope["temperature_max_K"]) != PHYSICAL_T_MAX_K
        or float(envelope["absolute_u_max_over_lid_speed"])
        != PHYSICAL_U_LID_MULTIPLIER
    ):
        raise ValueError("MV4 physical envelope differs from locked protocol")
    return value


def _task_directory(root: Path, fold: int, budget: int) -> Path:
    return root / "tasks" / f"fold_{fold}" / f"budget_{budget}"


def bounded_residual_candidate(
    raw: np.ndarray,
    unbounded_candidate: np.ndarray,
    residual_std: np.ndarray,
    cap_sigma: float = RESIDUAL_CAP_SIGMA,
) -> np.ndarray:
    """Bound an existing residual without changing its small-signal limit."""
    if cap_sigma <= 0.0:
        raise ValueError("cap_sigma must be positive")
    raw_array = np.asarray(raw, dtype=np.float32)
    candidate = np.asarray(unbounded_candidate, dtype=np.float32)
    scale = np.maximum(np.asarray(residual_std, dtype=np.float32), 1.0e-12)
    normalized = (candidate - raw_array) / scale
    bounded = cap_sigma * np.tanh(normalized / cap_sigma)
    return (raw_array + scale * bounded).astype(np.float32)


def project_physical_fields(
    candidate: np.ndarray,
    raw: np.ndarray,
    lid_speed: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply fixed, target-free sanity bounds to T and streamwise velocity."""
    value = np.asarray(candidate, dtype=np.float32).copy()
    raw_array = np.asarray(raw, dtype=np.float32)
    if value.shape != raw_array.shape or value.ndim != 4 or value.shape[1] != 2:
        raise ValueError("physical projection expects [sample,2,y,x] arrays")
    nonfinite = ~np.isfinite(value)
    value[nonfinite] = raw_array[nonfinite]
    before = value.copy()
    value[:, 0] = np.clip(value[:, 0], PHYSICAL_T_MIN_K, PHYSICAL_T_MAX_K)
    speed_limit = PHYSICAL_U_LID_MULTIPLIER * max(abs(float(lid_speed)), 1.0)
    value[:, 1] = np.clip(value[:, 1], -speed_limit, speed_limit)
    changed = value != before
    return value, {
        "nonfinite_replaced_count": int(np.count_nonzero(nonfinite)),
        "projected_value_count": int(np.count_nonzero(changed)),
        "projected_fraction": float(np.mean(changed)),
        "temperature_bounds_K": [PHYSICAL_T_MIN_K, PHYSICAL_T_MAX_K],
        "velocity_bounds_m_per_s": [-speed_limit, speed_limit],
    }


def _descriptor(condition: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            np.log10(float(condition["knudsen"])),
            float(condition["lid_speed_m_per_s"]) / 100.0,
        ],
        dtype=np.float64,
    )


def coordinate_support_report(
    heldout: Mapping[str, Any],
    training: tuple[Mapping[str, Any], ...],
    tolerance: float = SUPPORT_TOLERANCE,
) -> dict[str, Any]:
    """Report target-free coordinate-wise interpolation/extrapolation status.

    This is deliberately more conservative than neural confidence.  A held-out
    condition is trusted only when every conditioning coordinate lies inside
    the range represented by training conditions.  Degenerate coordinates are
    trusted only at the observed value.
    """
    if not training:
        raise ValueError("support gate requires at least one training condition")
    names = ("log10_Kn", "U_lid_over_100")
    point = _descriptor(heldout)
    cloud = np.stack([_descriptor(item) for item in training])
    lower, upper = cloud.min(axis=0), cloud.max(axis=0)
    records: dict[str, Any] = {}
    trusted = True
    for index, name in enumerate(names):
        width = upper[index] - lower[index]
        scale = max(abs(lower[index]), abs(upper[index]), 1.0)
        tol = tolerance * scale
        inside = bool(lower[index] - tol <= point[index] <= upper[index] + tol)
        if point[index] < lower[index]:
            distance = (lower[index] - point[index]) / max(width, tol, 1.0e-12)
        elif point[index] > upper[index]:
            distance = (point[index] - upper[index]) / max(width, tol, 1.0e-12)
        else:
            distance = 0.0
        records[name] = {
            "heldout": float(point[index]),
            "training_min": float(lower[index]),
            "training_max": float(upper[index]),
            "inside": inside,
            "normalized_extrapolation_distance": float(distance),
        }
        trusted = trusted and inside
    return {
        "rule": "coordinate_wise_training_range_no_test_target",
        "trusted_interpolation": bool(trusted),
        "action": "bounded_vision" if trusted else "raw_identity_fallback",
        "coordinates": records,
    }


def _condition_mean_score(
    raw: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    condition_labels: np.ndarray,
    specs: Mapping[str, Mapping[str, Any]],
) -> float:
    scores = []
    for condition_id in np.unique(condition_labels):
        mask = condition_labels == condition_id
        speed = float(specs[str(condition_id)]["lid_speed_m_per_s"])
        scores.append(
            mv3.evaluate_fields(raw[mask], candidate[mask], target[mask], speed)[
                "vision_composite_nrmse"
            ]
        )
    return float(np.mean(scores))


def select_fallback(
    validation_raw: np.ndarray,
    validation_target: np.ndarray,
    validation_conditions: np.ndarray,
    specs: Mapping[str, Mapping[str, Any]],
    gaussian_passes: int,
    tsvd_rank: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Rank classical alternatives on validation data for diagnostics."""
    candidates = {
        "raw": validation_raw,
        "gaussian_like": gaussian_like(validation_raw, gaussian_passes),
        "tsvd_pod_type": tsvd(validation_raw, tsvd_rank),
    }
    records = [
        {
            "method": name,
            "validation_condition_mean_composite_nrmse": _condition_mean_score(
                validation_raw,
                value,
                validation_target,
                validation_conditions,
                specs,
            ),
        }
        for name, value in candidates.items()
    ]
    best = min(
        records,
        key=lambda item: (
            item["validation_condition_mean_composite_nrmse"],
            FALLBACK_METHODS.index(item["method"]),
        ),
    )
    return str(best["method"]), records


def train_bounded_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    cap_sigma: float = RESIDUAL_CAP_SIGMA,
) -> tuple[Any, dict[str, Any]]:
    """Train a U-Net whose normalized residual is bounded by construction."""
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
        normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(
            np.float32
        )
        residual = (
            (y - x[:, : len(OUTPUT_FIELDS)]) / scaling["residual_std"]
        ).astype(np.float32)
        return torch.from_numpy(normalized), torch.from_numpy(residual)

    def bounded(latent):
        return float(cap_sigma) * torch.tanh(latent / float(cap_sigma))

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
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(bounded(model(xb)), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(xb)
        model.eval()
        with torch.no_grad():
            validation = float(_loss(bounded(model(vx.to(device))), vy.to(device)).cpu())
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(tx),
                "validation_loss": validation,
            }
        )
        if validation < best_value - 1.0e-7:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_value, best_epoch, stale = validation, epoch, 0
        else:
            stale += 1
        if stale >= 25:
            break
    if best_state is None:
        raise RuntimeError("MV4 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, {
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "residual_head": "cap_sigma*tanh(latent/cap_sigma)",
        "residual_cap_sigma": float(cap_sigma),
        "history": history,
    }


def predict_bounded(
    model: Any,
    x: np.ndarray,
    scaling: Mapping[str, np.ndarray],
    batch_size: int,
    cap_sigma: float = RESIDUAL_CAP_SIGMA,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    import torch

    normalized = ((x - scaling["input_mean"]) / scaling["input_std"]).astype(
        np.float32
    )
    values, latent_abs_max, bounded_abs_max = [], 0.0, 0.0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            latent = model(torch.from_numpy(normalized[start : start + batch_size]))
            bounded = float(cap_sigma) * torch.tanh(latent / float(cap_sigma))
            latent_abs_max = max(latent_abs_max, float(torch.max(torch.abs(latent))))
            bounded_abs_max = max(bounded_abs_max, float(torch.max(torch.abs(bounded))))
            values.append(
                x[start : start + batch_size, : len(OUTPUT_FIELDS)]
                + bounded.numpy() * scaling["residual_std"]
            )
    return (
        np.concatenate(values).astype(np.float32),
        time.perf_counter() - started,
        {
            "latent_normalized_residual_abs_max": latent_abs_max,
            "bounded_normalized_residual_abs_max": bounded_abs_max,
            "cap_sigma": float(cap_sigma),
        },
    )


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
    mv4_protocol = locked_mv4_protocol()
    protocol = mv3.locked_protocol()
    specs = mv3._condition_map(protocol)
    split = mv3.fold_split(fold_index, protocol)
    blocks, full = mv3.load_condition_data(existing_m3_root, reference_root, protocol)
    targets = mv3.fold_targets(full, split)
    train_x, train_y, train_conditions, train_identity = mv3.build_budget_arrays(
        blocks, targets, split["train"], specs, budget
    )
    validation_x, validation_y, validation_conditions, validation_identity = (
        mv3.build_budget_arrays(blocks, targets, split["validation"], specs, budget)
    )
    test_x, test_y, test_conditions, test_identity = mv3.build_budget_arrays(
        blocks, targets, split["test"], specs, budget
    )
    scaling = fit_scaling(train_x, train_y)
    model, training = train_bounded_model(
        train_x,
        train_y,
        validation_x,
        validation_y,
        scaling,
        epochs=epochs,
        batch_size=batch_size,
        seed=training_seed + mv3.task_index(fold_index, budget),
    )
    validation_bounded, validation_seconds, validation_diagnostics = predict_bounded(
        model, validation_x, scaling, batch_size
    )
    test_bounded, inference_seconds, test_diagnostics = predict_bounded(
        model, test_x, scaling, batch_size
    )
    validation_raw = validation_x[:, : len(OUTPUT_FIELDS)]
    raw = test_x[:, : len(OUTPUT_FIELDS)]
    validation_speeds = np.asarray(
        [float(specs[str(item)]["lid_speed_m_per_s"]) for item in validation_conditions]
    )
    alpha, alpha_selection = mv3.select_residual_gate(
        validation_raw,
        validation_bounded,
        validation_y,
        validation_speeds,
        tuple(
            float(item)
            for item in protocol["model_contract"]["residual_gate_candidates"]
        ),
    )
    vision_bounded = raw + float(alpha) * (test_bounded - raw)

    gaussian_passes, gaussian_selection = select_baseline(
        validation_raw, validation_y, GAUSSIAN_PASSES, gaussian_like
    )
    tsvd_rank, tsvd_selection = select_baseline(
        validation_raw, validation_y, TSVD_RANKS, tsvd
    )
    gaussian = gaussian_like(raw, gaussian_passes)
    pod_type = tsvd(raw, tsvd_rank)
    diagnostic_classical_name, fallback_selection = select_fallback(
        validation_raw,
        validation_y,
        validation_conditions,
        specs,
        gaussian_passes,
        tsvd_rank,
    )
    # A validation-selected smoother can still fail on a genuinely new
    # extrapolation.  Identity is the only target-free fallback that guarantees
    # zero degradation with respect to the reported Raw comparator.
    fallback_name = "raw"
    fallback_values = raw
    heldout = specs[split["heldout_condition"]]
    support = coordinate_support_report(
        heldout, tuple(specs[item] for item in split["train"])
    )
    selected = vision_bounded if support["trusted_interpolation"] else fallback_values
    lid_speed = float(heldout["lid_speed_m_per_s"])
    vision_bounded, bounded_projection = project_physical_fields(
        vision_bounded, raw, lid_speed
    )
    vision_safe, safe_projection = project_physical_fields(selected, raw, lid_speed)

    candidates = {
        "raw": raw,
        "gaussian_like": gaussian,
        "tsvd_pod_type": pod_type,
        "vision_bounded": vision_bounded,
        "vision_safe": vision_safe,
    }
    methods = {
        name: mv3.evaluate_fields(raw, value, test_y, lid_speed)
        for name, value in candidates.items()
    }
    per_seed: dict[str, Any] = {}
    for seed in split["test"][split["heldout_condition"]]:
        mask = test_identity[:, 0] == seed
        per_seed[str(seed)] = {
            name: mv3.evaluate_fields(raw[mask], value[mask], test_y[mask], lid_speed)
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
            "residual_cap_sigma": RESIDUAL_CAP_SIGMA,
            "residual_gate_alpha": alpha,
            "support_gate": support,
            "fallback_method": fallback_name,
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
        vision_bounded=vision_bounded,
        vision_safe=vision_safe,
        target=test_y,
    )
    checks = {
        "condition_disjoint": split["heldout_condition"] not in split["train"]
        and split["heldout_condition"] not in split["validation"],
        "heat_flux_excluded": not any(
            name.lower().startswith("q")
            for name in mv3.MODEL_INPUT_FIELDS + OUTPUT_FIELDS
        ),
        "bounded_residual_head": bool(
            test_diagnostics["bounded_normalized_residual_abs_max"]
            <= RESIDUAL_CAP_SIGMA + 1.0e-6
        ),
        "finite_safe_fields": bool(np.all(np.isfinite(vision_safe))),
        "positive_safe_temperature": bool(np.min(vision_safe[:, 0]) >= PHYSICAL_T_MIN_K),
        "safe_velocity_envelope": bool(
            np.max(np.abs(vision_safe[:, 1]))
            <= PHYSICAL_U_LID_MULTIPLIER * max(abs(lid_speed), 1.0) + 1.0e-5
        ),
        "target_free_support_gate": support["rule"]
        == "coordinate_wise_training_range_no_test_target",
        "extrapolation_uses_raw_identity_fallback": bool(
            support["trusted_interpolation"]
            or support["action"] == "raw_identity_fallback"
        ),
    }
    summary = {
        "stage": STAGE,
        "status": "complete_MV4_task",
        "protocol_sha256": _sha256(mv3.protocol_path()),
        "lock_sha256": _sha256(mv3.lock_path()),
        "mv4_protocol_sha256": _sha256(protocol_path()),
        "mv3_source_summary_sha256": mv4_protocol["source_contract"]["mv3_summary_sha256"],
        "fold_index": fold_index,
        "heldout_condition": split["heldout_condition"],
        "budget_blocks": budget,
        "effective_DSMC_samples": int(
            budget * protocol["model_contract"]["samples_per_block"]
        ),
        "split": {
            section: {key: list(value) for key, value in split[section].items()}
            for section in ("train", "validation", "test")
        },
        "sample_counts": {
            "train": len(train_x),
            "validation": len(validation_x),
            "test": len(test_x),
        },
        "target_contract": "unchanged MV3 same-condition cross-fit; held-out condition absent from training and tuning",
        "selection_contract": "alpha and baseline parameters selected on validation conditions only; support gate uses condition coordinates only; extrapolation returns Raw identity",
        "baseline_selection": {
            "gaussian_like": {
                "selected_passes": gaussian_passes,
                "candidates": gaussian_selection,
            },
            "tsvd_pod_type": {
                "selected_rank": tsvd_rank,
                "candidates": tsvd_selection,
            },
            "bounded_residual_gate": {
                "selected_alpha": alpha,
                "candidates": alpha_selection,
                "cap_sigma": RESIDUAL_CAP_SIGMA,
            },
            "safe_fallback": {
                "selected_method": fallback_name,
                "policy": "raw identity outside coordinate support; no test target",
                "validation_classical_ranking_not_used_for_safety": fallback_selection,
                "best_validation_classical_method": diagnostic_classical_name,
            },
        },
        "support_gate": support,
        "training": training,
        "validation_inference_seconds": validation_seconds,
        "test_inference_seconds": inference_seconds,
        "diagnostics": {
            "scaling": {
                key: np.asarray(value).reshape(-1).astype(float).tolist()
                for key, value in scaling.items()
            },
            "validation_prediction": validation_diagnostics,
            "test_prediction": test_diagnostics,
            "bounded_projection": bounded_projection,
            "safe_projection": safe_projection,
            "test_ranges": {
                name: {
                    "T_min_K": float(np.min(value[:, 0])),
                    "T_max_K": float(np.max(value[:, 0])),
                    "u_min_m_per_s": float(np.min(value[:, 1])),
                    "u_max_m_per_s": float(np.max(value[:, 1])),
                }
                for name, value in candidates.items()
            },
        },
        "methods": methods,
        "per_seed_metrics": per_seed,
        "article_profiles": mv3.article_profile_metrics(heldout, raw, vision_safe),
        "checks": checks,
        "decision": "accept_MV4_task" if all(checks.values()) else "hold_MV4_task",
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
    for fold in range(4):
        for budget in BUDGETS:
            path = _task_directory(root, fold, budget) / "summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("status") != "complete_MV4_task"
                or value.get("fold_index") != fold
                or value.get("budget_blocks") != budget
            ):
                raise ValueError(f"invalid MV4 task summary: {path}")
            records.append(value)
    return records


def _aggregate_values(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_condition: dict[str, Any] = {}
    for record in records:
        by_condition.setdefault(record["heldout_condition"], {})[
            str(record["budget_blocks"])
        ] = {method: record["methods"][method] for method in METHODS}
    by_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        selected = [item for item in records if item["budget_blocks"] == budget]
        by_budget[str(budget)] = {}
        for method in METHODS:
            ratios = np.asarray(
                [
                    item["methods"][method]["vision_over_raw_composite"]
                    for item in selected
                ]
            )
            by_budget[str(budget)][method] = {
                "mean_over_raw": float(np.mean(ratios)),
                "median_over_raw": float(np.median(ratios)),
                "maximum_over_raw": float(np.max(ratios)),
                "heldout_condition_count": len(ratios),
            }
    return by_condition, by_budget


def _curve_figure(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "raw": "0.35",
        "gaussian_like": "#2ca02c",
        "tsvd_pod_type": "#ff7f0e",
        "vision_bounded": "#9467bd",
        "vision_safe": "#1f77b4",
    }
    labels = {
        "raw": "Raw",
        "gaussian_like": "Gaussian-like",
        "tsvd_pod_type": "TSVD/POD-type",
        "vision_bounded": "Bounded vision",
        "vision_safe": "Safe hybrid",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for fold, axis in enumerate(axes.ravel()):
        selected = [item for item in records if item["fold_index"] == fold]
        for method in METHODS:
            axis.plot(
                BUDGETS,
                [
                    next(
                        item for item in selected if item["budget_blocks"] == budget
                    )["methods"][method]["vision_composite_nrmse"]
                    for budget in BUDGETS
                ],
                marker="o",
                linewidth=2,
                color=colors[method],
                label=labels[method],
            )
        trusted = selected[0]["support_gate"]["trusted_interpolation"]
        axis.set(
            title=f"{selected[0]['heldout_condition']} ({'interpolation' if trusted else 'fallback'})",
            xlabel="Temporal blocks averaged",
            ylabel="Held-out composite NRMSE",
            xscale="log",
            yscale="log",
        )
        axis.set_xticks(BUDGETS, labels=[str(item) for item in BUDGETS])
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    figure.suptitle("MV4 bounded condition-aware reconstruction")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _physical_figures(root: Path, records: list[dict[str, Any]]) -> list[str]:
    import matplotlib.pyplot as plt

    output_names = []
    columns = (
        ("raw", "Raw DSMC"),
        ("gaussian_like", "Gaussian-like"),
        ("tsvd_pod_type", "TSVD/POD-type"),
        ("vision_bounded", "Bounded vision"),
        ("vision_safe", "Safe hybrid"),
        ("target", "Reference"),
    )
    for fold in range(4):
        record = next(
            item
            for item in records
            if item["fold_index"] == fold and item["budget_blocks"] == 1
        )
        directory = _task_directory(root, fold, 1)
        with np.load(directory / "predictions.npz", allow_pickle=False) as data:
            values = {name: np.asarray(data[name])[0] for name, _ in columns}
        figure, axes = plt.subplots(2, len(columns), figsize=(15.0, 5.6), constrained_layout=True)
        for field, (label, unit) in enumerate((("T", "K"), ("u", "m/s"))):
            low = min(float(values[name][field].min()) for name, _ in columns if name != "vision_bounded")
            high = max(float(values[name][field].max()) for name, _ in columns if name != "vision_bounded")
            for column, (name, title) in enumerate(columns):
                plot = axes[field, column].imshow(
                    values[name][field],
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    cmap="coolwarm",
                    vmin=low,
                    vmax=high,
                    interpolation="nearest",
                )
                axes[field, column].set_aspect("equal")
                axes[field, column].set_xlabel("x/L")
                if column == 0:
                    axes[field, column].set_ylabel(f"y/L\n{label} [{unit}]")
                if field == 0:
                    axes[field, column].set_title(title)
            figure.colorbar(plot, ax=axes[field, :], shrink=0.78, label=f"{label} [{unit}]")
        action = record["support_gate"]["action"]
        figure.suptitle(
            f"{record['heldout_condition']}, budget=1; safe action: {action.replace('_', ' ')}"
        )
        name = f"physical_fields_fold_{fold}_budget_1.png"
        figure.savefig(root / name, dpi=220)
        plt.close(figure)
        output_names.append(name)
    return output_names


def aggregate(root: Path) -> dict[str, Any]:
    mv4_protocol = locked_mv4_protocol()
    records = _records(root)
    by_condition, by_budget = _aggregate_values(records)
    trusted = [item for item in records if item["support_gate"]["trusted_interpolation"]]
    extrapolation = [item for item in records if not item["support_gate"]["trusted_interpolation"]]
    budget_one_trusted = [item for item in trusted if item["budget_blocks"] == 1]
    checks = {
        "all_16_tasks_complete": len(records) == 16,
        "all_task_safety_checks_pass": all(all(item["checks"].values()) for item in records),
        "all_extrapolation_tasks_use_raw_identity_fallback": all(
            item["support_gate"]["action"]
            == "raw_identity_fallback"
            for item in extrapolation
        ),
        "safe_method_never_exceeds_1p05_raw": all(
            item["methods"]["vision_safe"]["vision_over_raw_composite"] <= 1.05
            for item in records
        ),
        "trusted_budget_one_improves_raw": bool(budget_one_trusted)
        and all(
            item["methods"]["vision_safe"]["vision_over_raw_composite"] < 1.0
            for item in budget_one_trusted
        ),
        "trusted_budget_one_beats_both_baselines_at_least_once": any(
            item["methods"]["vision_safe"]["vision_composite_nrmse"]
            < min(
                item["methods"]["gaussian_like"]["vision_composite_nrmse"],
                item["methods"]["tsvd_pod_type"]["vision_composite_nrmse"],
            )
            for item in budget_one_trusted
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    curve_name = "mv4_sampling_budget_curves.png"
    _curve_figure(records, root / curve_name)
    physical_names = _physical_figures(root, records)
    summary = {
        "stage": STAGE,
        "status": "complete_MV4_aggregate",
        "scope": "locked MV3 T/u benchmark with bounded vision and target-free coordinate support/raw-identity fallback",
        "task_count": len(records),
        "mv4_protocol_sha256": _sha256(protocol_path()),
        "mv3_source_summary_sha256": mv4_protocol["source_contract"]["mv3_summary_sha256"],
        "trusted_task_count": len(trusted),
        "extrapolation_task_count": len(extrapolation),
        "by_condition": by_condition,
        "by_budget": by_budget,
        "checks": checks,
        "decision": "MV4_safe_cross_condition_evidence_pass"
        if all(checks.values())
        else "hold_for_MV4_diagnosis",
    }
    _atomic_json(root / "summary.json", summary)
    top_names = ("summary.json", curve_name, *physical_names)
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
    top = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, record in top["files"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV4 top artifact verification failed: {path}")
    specs = mv3._condition_map(mv3.locked_protocol())
    maximum_difference, verified_task_files = 0.0, 0
    records = []
    for fold in range(4):
        for budget in BUDGETS:
            directory = _task_directory(root, fold, budget)
            manifest = json.loads(
                (directory / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            for name, record in manifest["files"].items():
                path = directory / name
                if (
                    not path.is_file()
                    or path.stat().st_size != record["size_bytes"]
                    or _sha256(path) != record["sha256"]
                ):
                    raise ValueError(f"MV4 task artifact verification failed: {path}")
                verified_task_files += 1
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            expected = mv3.fold_split(fold)
            if (
                summary["heldout_condition"] != expected["heldout_condition"]
                or summary["budget_blocks"] != budget
                or not all(summary["checks"].values())
            ):
                raise ValueError(f"MV4 task identity/safety mismatch: {directory}")
            speed = float(specs[summary["heldout_condition"]]["lid_speed_m_per_s"])
            with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                raw, target = np.asarray(data["raw"]), np.asarray(data["target"])
                for method in METHODS:
                    rebuilt = mv3.evaluate_fields(raw, np.asarray(data[method]), target, speed)
                    recorded = summary["methods"][method]
                    maximum_difference = max(
                        maximum_difference,
                        abs(
                            rebuilt["vision_composite_nrmse"]
                            - recorded["vision_composite_nrmse"]
                        ),
                    )
                    for field in OUTPUT_FIELDS:
                        maximum_difference = max(
                            maximum_difference,
                            abs(
                                rebuilt["per_field"][field]["vision_nrmse"]
                                - recorded["per_field"][field]["vision_nrmse"]
                            ),
                        )
            records.append(summary)
    if maximum_difference > 2.0e-6:
        raise ValueError(f"MV4 metric reconstruction mismatch: {maximum_difference}")
    by_condition, by_budget = _aggregate_values(records)
    saved = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if by_condition != saved["by_condition"] or by_budget != saved["by_budget"]:
        raise ValueError("MV4 aggregate reconstruction mismatch")
    return {
        "status": "complete_MV4_artifacts_metrics_and_safety_verified",
        "decision": saved["decision"],
        "task_count": len(records),
        "verified_task_files": verified_task_files,
        "maximum_metric_reconstruction_difference": maximum_difference,
        "summary_sha256": _sha256(root / "summary.json"),
    }


def package(root: Path) -> dict[str, Any]:
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "complete_MV4_artifacts_metrics_and_safety_verified":
        raise ValueError("MV4 must pass its verifier before packaging")
    bundle = root / "MOHAMMADZADEH_MV4_JCP_RETURN_BUNDLE.tar.gz"
    top_names = tuple(
        name
        for name in (
            "summary.json",
            "artifact_manifest.json",
            "verification.json",
            "mv4_sampling_budget_curves.png",
            *(f"physical_fields_fold_{fold}_budget_1.png" for fold in range(4)),
        )
        if (root / name).is_file()
    )
    with tarfile.open(bundle, "w:gz") as archive:
        for name in top_names:
            archive.add(root / name, arcname=name, filter=_portable_tarinfo)
        archive.add(
            protocol_path(),
            arcname=f"provenance/{PROTOCOL_FILE}",
            filter=_portable_tarinfo,
        )
        for fold in range(4):
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
        "status": "complete_MV4_portable_bundle",
        "bundle": str(bundle),
        "bundle_sha256": checksum,
    }


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
        result = aggregate(args.output_dir)
    else:
        if args.task_index is not None:
            fold, budget = mv3.task_from_index(args.task_index)
        elif args.fold_index is not None and args.budget is not None:
            fold, budget = args.fold_index, args.budget
        else:
            parser.error("task mode requires --task-index or --fold-index and --budget")
        if args.existing_m3_root is None or args.reference_root is None:
            parser.error("task mode requires --existing-m3-root and --reference-root")
        result = run_task(
            args.existing_m3_root,
            args.reference_root,
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
