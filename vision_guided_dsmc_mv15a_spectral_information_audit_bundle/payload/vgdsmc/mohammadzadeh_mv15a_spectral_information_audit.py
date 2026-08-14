"""MV15A spectral information audit and trust-region reconstruction of q_y.

MV15A is a post-processing stage.  It launches no DSMC trajectory and trains
no new neural network.  Independent development-seed blocks estimate the
mode-wise signal and sampling-noise spectra.  Development labels then select a
convex spectral trust region between the indexed Raw-B1 observation and the
already locked MV9 Mamba prediction.  Legacy labels are inaccessible until the
prediction archive has been recursively hash locked.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


STAGE = "MV15A_Mohammadzadeh_spectral_information_audit"
STATUS = "locked_after_MV14_failure_before_any_MV15A_legacy_outcome"
PROTOCOL_FILE = "mv15a_spectral_information_audit_protocol.json"
QY_INDEX = 3
RADIAL_BINS = 12
WEIGHT_SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
RADIAL_SMOOTHING_PASSES = (0, 1, 2)
CONDITION_RIDGE = 0.01
AUTOCORRELATION_LAGS = (1, 2, 3, 4, 5)
EPS = 1.0e-12


def _mv9_module():
    from . import mohammadzadeh_mv9_heat_flux as mv9

    return mv9


def _mv14_module():
    from . import mohammadzadeh_mv14_kinetic_conservation_cavity as mv14

    return mv14


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


def _ancestry_protocol_path(module: Any, filename: str) -> Path:
    path = (
        Path(module.__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / filename
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def protocol_path() -> Path:
    path = (
        Path(__file__).resolve().parents[1]
        / "reference_data"
        / "mohammadzadeh_2012"
        / PROTOCOL_FILE
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def locked_protocol() -> dict[str, Any]:
    protocol = json.loads(protocol_path().read_text(encoding="utf-8"))
    if protocol.get("stage") != STAGE or protocol.get("status") != STATUS:
        raise ValueError("MV15A protocol is absent or unlocked")
    information = protocol["information_audit_contract"]
    fusion = protocol["spectral_fusion_contract"]
    if (
        int(information["radial_frequency_bins"]) != RADIAL_BINS
        or tuple(int(x) for x in information["block_autocorrelation_lags"])
        != AUTOCORRELATION_LAGS
        or tuple(float(x) for x in fusion["candidate_weight_shrinkage_to_development_optimum"])
        != WEIGHT_SHRINKAGES
        or tuple(int(x) for x in fusion["candidate_radial_smoothing_passes"])
        != RADIAL_SMOOTHING_PASSES
    ):
        raise ValueError("MV15A implementation differs from its locked matrix")
    mv9 = _mv9_module()
    mv14 = _mv14_module()
    sources = {
        "mv9_module_sha256": Path(mv9.__file__),
        "mv9_protocol_sha256": _ancestry_protocol_path(
            mv9, "mv9_heat_flux_noise2noise_protocol.json"
        ),
        "mv14_module_sha256": Path(mv14.__file__),
        "mv14_protocol_sha256": _ancestry_protocol_path(
            mv14, "mv14_kinetic_conservation_cavity_protocol.json"
        ),
    }
    for key, path in sources.items():
        if _sha256(path) != protocol["source_contract"][key]:
            raise ValueError(f"MV15A immutable ancestry mismatch: {key}")
    return protocol


def verify_lock() -> dict[str, Any]:
    protocol = locked_protocol()
    return {
        "stage": STAGE,
        "status": "MV15A_lock_verified_without_any_MV15A_legacy_outcome",
        "protocol_sha256": _sha256(protocol_path()),
        "method": protocol["method_name"],
        "DSMC_rerun": False,
        "neural_network_retraining": False,
        "legacy_targets_loaded_by_prediction_stage": False,
        "transform": "orthonormal_DCT_II",
        "radial_bins": RADIAL_BINS,
    }


def _verify_manifest(root: Path, name: str) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads((root / name).read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise ValueError(f"MV15A recursive artifact verification failed: {path}")
    return manifest


def verify_data_contract(mv9_output_root: Path, mv14_output_root: Path) -> dict[str, Any]:
    protocol = locked_protocol()
    mv9_root = Path(mv9_output_root).resolve()
    mv14_root = Path(mv14_output_root).resolve()
    _verify_manifest(mv9_root, "assembly_manifest.json")
    _verify_manifest(mv9_root, "artifact_manifest.json")
    _verify_manifest(mv14_root, "prediction_manifest.json")
    _verify_manifest(mv14_root, "artifact_manifest.json")
    mv9_summary = json.loads((mv9_root / "summary.json").read_text(encoding="utf-8"))
    mv14_summary = json.loads((mv14_root / "summary.json").read_text(encoding="utf-8"))
    mv14_selection = json.loads(
        (mv14_root / "selection_summary.json").read_text(encoding="utf-8")
    )
    checks = {
        "MV9_failure_outcome_explicitly_required": mv9_summary.get("decision")
        == protocol["source_contract"]["required_MV9_decision"],
        "MV14_failure_outcome_explicitly_required": mv14_summary.get("decision")
        == protocol["source_contract"]["required_MV14_decision"],
        "MV9_dataset_present": (mv9_root / "dataset.npz").is_file(),
        "MV14_predictions_present": (mv14_root / "locked_predictions.npz").is_file(),
        "MV14_uses_same_MV9_root": Path(mv14_selection["mv9_output_root"]).resolve()
        == mv9_root,
        "legacy_outcomes_not_reclassified_as_confirmation": True,
    }
    if not all(checks.values()):
        raise ValueError(f"MV15A data contract failed: {checks}")
    return {
        "stage": STAGE,
        "status": "MV15A_data_contract_verified",
        "mv9_output_root": str(mv9_root),
        "mv14_output_root": str(mv14_root),
        "checks": checks,
    }


@lru_cache(maxsize=16)
def _orthonormal_dct_matrix(size: int) -> np.ndarray:
    """Return the exact orthonormal DCT-II matrix for a nonperiodic axis."""

    size = int(size)
    if size < 1:
        raise ValueError("DCT axis must be nonempty")
    modes = np.arange(size, dtype=np.float64)[:, None]
    points = np.arange(size, dtype=np.float64)[None, :]
    matrix = np.sqrt(2.0 / size) * np.cos(
        np.pi * (points + 0.5) * modes / size
    )
    matrix[0] /= math.sqrt(2.0)
    matrix.setflags(write=False)
    return matrix


def _numpy_dct2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    cy = _orthonormal_dct_matrix(value.shape[-2])
    cx = _orthonormal_dct_matrix(value.shape[-1])
    return np.matmul(np.matmul(cy, value), cx.T)


def _numpy_idct2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    cy = _orthonormal_dct_matrix(value.shape[-2])
    cx = _orthonormal_dct_matrix(value.shape[-1])
    return np.matmul(np.matmul(cy.T, value), cx)


def _dct2(value: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import dctn
    except ModuleNotFoundError:
        return _numpy_dct2(value)
    return dctn(np.asarray(value, dtype=np.float64), axes=(-2, -1), norm="ortho")


def _idct2(value: np.ndarray) -> np.ndarray:
    try:
        from scipy.fft import idctn
    except ModuleNotFoundError:
        return _numpy_idct2(value)
    return idctn(np.asarray(value, dtype=np.float64), axes=(-2, -1), norm="ortho")


def radial_bin_map(shape: Sequence[int], bins: int = RADIAL_BINS) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = int(shape[-2]), int(shape[-1])
    ky = np.arange(ny, dtype=np.float64) / max(ny - 1, 1)
    kx = np.arange(nx, dtype=np.float64) / max(nx - 1, 1)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) / math.sqrt(2.0)
    mapping = np.minimum((radius * bins).astype(np.int64), bins - 1)
    centers = (np.arange(bins, dtype=np.float64) + 0.5) / bins
    return mapping, centers


def _bin_mean(value: np.ndarray, mapping: np.ndarray, bins: int) -> np.ndarray:
    result = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        mask = mapping == index
        result[index] = float(np.mean(np.asarray(value)[mask])) if np.any(mask) else 0.0
    return result


def _matching_cross_seed_pairs(
    condition_mask: np.ndarray, identities: np.ndarray
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seeds = sorted(int(x) for x in np.unique(identities[condition_mask, 0]))
    for left_position, left_seed in enumerate(seeds):
        for right_seed in seeds[left_position + 1 :]:
            left = {
                int(identities[index, 1]): int(index)
                for index in np.flatnonzero(condition_mask & (identities[:, 0] == left_seed))
            }
            right = {
                int(identities[index, 1]): int(index)
                for index in np.flatnonzero(condition_mask & (identities[:, 0] == right_seed))
            }
            pairs.extend((left[block], right[block]) for block in sorted(left.keys() & right.keys()))
    return pairs


def cross_spectral_information(
    raw_qy: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
    *,
    bins: int = RADIAL_BINS,
) -> dict[str, Any]:
    raw_qy = np.asarray(raw_qy, dtype=np.float64)
    conditions = np.asarray(conditions)
    identities = np.asarray(identities)
    if raw_qy.ndim != 3 or len(raw_qy) != len(conditions) or identities.shape[0] != len(raw_qy):
        raise ValueError("MV15A spectral inputs have incompatible shapes")
    mapping, centers = radial_bin_map(raw_qy.shape[-2:], bins)
    transformed = _dct2(raw_qy)
    records: list[dict[str, Any]] = []
    all_products: list[np.ndarray] = []
    all_differences: list[np.ndarray] = []
    for condition in np.unique(conditions):
        mask = conditions == condition
        pairs = _matching_cross_seed_pairs(mask, identities)
        if not pairs:
            raise ValueError(f"MV15A needs independent development seeds: {condition}")
        products, differences = [], []
        for left, right in pairs:
            first, second = transformed[left], transformed[right]
            products.append(first * second)
            differences.append(0.5 * (first - second) ** 2)
        product = np.mean(products, axis=0)
        difference = np.mean(differences, axis=0)
        signal = np.maximum(product, 0.0)
        noise = np.maximum(difference, 0.0)
        signal_bins = _bin_mean(signal, mapping, bins)
        noise_bins = _bin_mean(noise, mapping, bins)
        reliability = signal_bins / np.maximum(signal_bins + noise_bins, EPS)
        snr = signal_bins / np.maximum(noise_bins, EPS)
        mmse = float(np.sum(signal * noise / np.maximum(signal + noise, EPS)))
        raw_b10_floor = float(np.sum(noise / 10.0))
        records.append(
            {
                "condition": str(condition),
                "pair_count": len(pairs),
                "signal_power_by_bin": signal_bins.tolist(),
                "noise_power_by_bin": noise_bins.tolist(),
                "reliability_by_bin": reliability.tolist(),
                "snr_by_bin": snr.tolist(),
                "linear_MMSE_RMSE_ratio_to_ideal_independent_Raw_B10": math.sqrt(
                    mmse / max(raw_b10_floor, EPS)
                ),
            }
        )
        all_products.extend(products)
        all_differences.extend(differences)
    signal_mode = np.maximum(np.mean(all_products, axis=0), 0.0)
    noise_mode = np.maximum(np.mean(all_differences, axis=0), 0.0)
    signal_bins = _bin_mean(signal_mode, mapping, bins)
    noise_bins = _bin_mean(noise_mode, mapping, bins)
    reliability_bins = signal_bins / np.maximum(signal_bins + noise_bins, EPS)
    return {
        "records": records,
        "mapping": mapping,
        "centers": centers,
        "global_signal_mode": signal_mode,
        "global_noise_mode": noise_mode,
        "global_signal_by_bin": signal_bins,
        "global_noise_by_bin": noise_bins,
        "global_reliability_by_bin": reliability_bins,
        "global_reliability_mode": signal_mode / np.maximum(signal_mode + noise_mode, EPS),
    }


def block_autocorrelation(
    raw_qy: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
    lags: Sequence[int] = AUTOCORRELATION_LAGS,
) -> dict[str, float]:
    raw_qy = np.asarray(raw_qy, dtype=np.float64)
    values: dict[int, list[np.ndarray]] = {int(lag): [] for lag in lags}
    denominators: dict[int, list[float]] = {int(lag): [] for lag in lags}
    for condition in np.unique(conditions):
        mask_condition = conditions == condition
        for seed in np.unique(identities[mask_condition, 0]):
            indices = np.flatnonzero(mask_condition & (identities[:, 0] == seed))
            indices = indices[np.argsort(identities[indices, 1])]
            sequence = raw_qy[indices]
            residual = sequence - np.mean(sequence, axis=0, keepdims=True)
            variance = float(np.mean(residual**2))
            for lag in lags:
                lag = int(lag)
                if lag < len(residual) and variance > EPS:
                    values[lag].append(residual[:-lag] * residual[lag:])
                    denominators[lag].append(variance)
    result = {}
    for lag in lags:
        lag = int(lag)
        numerator = float(np.mean(np.concatenate([x.reshape(-1) for x in values[lag]]))) if values[lag] else 0.0
        denominator = float(np.mean(denominators[lag])) if denominators[lag] else 1.0
        result[str(lag)] = numerator / max(denominator, EPS)
    return result


def _smooth_bins(values: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for _ in range(int(passes)):
        padded = np.pad(result, (1, 1), mode="edge")
        result = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    return result


def optimal_bin_weights(
    raw_qy: np.ndarray,
    vision_qy: np.ndarray,
    target_qy: np.ndarray,
    mapping: np.ndarray,
    reliability_bins: np.ndarray,
    *,
    shrinkage: float,
    smoothing_passes: int,
) -> np.ndarray:
    raw_error = _dct2(np.asarray(raw_qy) - np.asarray(target_qy))
    vision_error = _dct2(np.asarray(vision_qy) - np.asarray(target_qy))
    return _optimal_bin_weights_from_errors(
        raw_error,
        vision_error,
        mapping,
        reliability_bins,
        shrinkage=shrinkage,
        smoothing_passes=smoothing_passes,
    )


def _optimal_bin_weights_from_errors(
    raw_error: np.ndarray,
    vision_error: np.ndarray,
    mapping: np.ndarray,
    reliability_bins: np.ndarray,
    *,
    shrinkage: float,
    smoothing_passes: int,
) -> np.ndarray:
    bins = len(reliability_bins)
    optimum = np.empty(bins, dtype=np.float64)
    for index in range(bins):
        mask = mapping == index
        er = raw_error[:, mask]
        ev = vision_error[:, mask]
        raw_variance = float(np.mean(er**2))
        vision_variance = float(np.mean(ev**2))
        covariance = float(np.mean(er * ev))
        denominator = raw_variance + vision_variance - 2.0 * covariance
        if denominator <= EPS:
            optimum[index] = float(reliability_bins[index])
        else:
            optimum[index] = (vision_variance - covariance) / denominator
    optimum = np.clip(optimum, 0.0, 1.0)
    weights = (1.0 - float(shrinkage)) * np.asarray(reliability_bins) + float(shrinkage) * optimum
    return np.clip(_smooth_bins(weights, smoothing_passes), 0.0, 1.0)


def expand_bin_weights(weights: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if np.min(mapping) < 0 or np.max(mapping) >= len(weights):
        raise ValueError("MV15A radial map is outside the weight vector")
    return weights[mapping]


def spectral_fuse(raw_qy: np.ndarray, vision_qy: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    raw_qy = np.asarray(raw_qy, dtype=np.float64)
    vision_qy = np.asarray(vision_qy, dtype=np.float64)
    weight_map = np.asarray(weight_map, dtype=np.float64)
    if raw_qy.shape != vision_qy.shape or raw_qy.shape[-2:] != weight_map.shape:
        raise ValueError("MV15A fusion shapes are incompatible")
    fused = weight_map * _dct2(raw_qy) + (1.0 - weight_map) * _dct2(vision_qy)
    return _idct2(fused).astype(np.float32)


def _nrmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((candidate - target) ** 2)) / max(np.sqrt(np.mean(target**2)), EPS))


def select_spectral_fusion(
    raw_qy: np.ndarray,
    vision_qy: np.ndarray,
    target_qy: np.ndarray,
    conditions: np.ndarray,
    mapping: np.ndarray,
    reliability_bins: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    unique = sorted(str(x) for x in np.unique(conditions))
    raw_coefficients = _dct2(raw_qy)
    vision_coefficients = _dct2(vision_qy)
    target_coefficients = _dct2(target_qy)
    raw_error = raw_coefficients - target_coefficients
    vision_error = vision_coefficients - target_coefficients
    for shrinkage in WEIGHT_SHRINKAGES:
        for smoothing in RADIAL_SMOOTHING_PASSES:
            held_scores: dict[str, float] = {}
            for held in unique:
                fit = conditions != held
                evaluate = conditions == held
                weights = _optimal_bin_weights_from_errors(
                    raw_error[fit], vision_error[fit], mapping,
                    reliability_bins, shrinkage=shrinkage,
                    smoothing_passes=smoothing,
                )
                weight_map = expand_bin_weights(weights, mapping)
                prediction = _idct2(
                    weight_map * raw_coefficients[evaluate]
                    + (1.0 - weight_map) * vision_coefficients[evaluate]
                )
                held_scores[held] = _nrmse(prediction, target_qy[evaluate])
            records.append(
                {
                    "shrinkage_to_development_optimum": float(shrinkage),
                    "radial_smoothing_passes": int(smoothing),
                    "leave_one_condition_out_qy_nrmse": held_scores,
                    "mean_leave_one_condition_out_qy_nrmse": float(np.mean(list(held_scores.values()))),
                }
            )
    selected = min(
        records,
        key=lambda row: (
            row["mean_leave_one_condition_out_qy_nrmse"],
            row["shrinkage_to_development_optimum"],
            row["radial_smoothing_passes"],
        ),
    )
    final_weights = _optimal_bin_weights_from_errors(
        raw_error, vision_error, mapping, reliability_bins,
        shrinkage=float(selected["shrinkage_to_development_optimum"]),
        smoothing_passes=int(selected["radial_smoothing_passes"]),
    )
    selected = dict(selected)
    selected["final_weight_by_bin"] = final_weights.tolist()
    selected["minimum_weight"] = float(np.min(final_weights))
    selected["maximum_weight"] = float(np.max(final_weights))
    return selected, records


def _condition_features(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 4 or images.shape[1] < 2:
        raise ValueError("MV15A conditioned images are malformed")
    log_kn = images[:, -2, 0, 0]
    speed = images[:, -1, 0, 0]
    return np.stack((np.ones(len(images)), log_kn, speed, log_kn * speed), axis=1)


def parametric_condition_only(
    development_images: np.ndarray,
    development_targets: np.ndarray,
    development_conditions: np.ndarray,
    query_images: np.ndarray,
    *,
    ridge: float = CONDITION_RIDGE,
) -> np.ndarray:
    condition_features, condition_targets = [], []
    for condition in sorted(str(x) for x in np.unique(development_conditions)):
        mask = development_conditions == condition
        condition_features.append(np.mean(_condition_features(development_images[mask]), axis=0))
        condition_targets.append(np.mean(development_targets[mask], axis=0))
    design = np.asarray(condition_features, dtype=np.float64)
    targets = np.asarray(condition_targets, dtype=np.float64)
    penalty = np.diag((0.0, ridge, ridge, ridge))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets.reshape(len(targets), -1))
    query = _condition_features(query_images) @ coefficients
    return query.reshape((len(query_images), *targets.shape[1:])).astype(np.float32)


def cross_condition_permutation(conditions: np.ndarray) -> np.ndarray:
    conditions = np.asarray(conditions)
    unique = sorted(str(x) for x in np.unique(conditions))
    if len(unique) < 2:
        raise ValueError("MV15A permutation requires at least two conditions")
    groups = [np.flatnonzero(conditions == condition) for condition in unique]
    if len({len(group) for group in groups}) != 1:
        raise ValueError("MV15A cross-condition permutation requires balanced groups")
    permutation = np.empty(len(conditions), dtype=np.int64)
    for position, destination in enumerate(groups):
        source = groups[(position + 1) % len(groups)]
        permutation[destination] = source
    if np.any(conditions[permutation] == conditions):
        raise AssertionError("MV15A permutation did not change condition")
    return permutation


def exact_affine_error_decomposition(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    target_centered = target - np.mean(target)
    candidate_centered = candidate - np.mean(candidate)
    variance = float(np.dot(target_centered, target_centered))
    slope = float(np.dot(candidate_centered, target_centered) / max(variance, EPS))
    residual = candidate_centered - slope * target_centered
    amplitude = (slope - 1.0) * target_centered
    offset = np.full_like(target, np.mean(candidate) - np.mean(target))
    total_error = candidate - target
    amplitude_mse = float(np.mean(amplitude**2))
    offset_mse = float(np.mean(offset**2))
    residual_mse = float(np.mean(residual**2))
    total_mse = float(np.mean(total_error**2))
    closure = abs(total_mse - (amplitude_mse + offset_mse + residual_mse))
    return {
        "slope": slope,
        "mean_offset": float(np.mean(offset)),
        "amplitude_MSE": amplitude_mse,
        "offset_MSE": offset_mse,
        "orthogonal_residual_MSE": residual_mse,
        "total_MSE": total_mse,
        "closure_error": closure,
        "target_RMS": float(np.sqrt(np.mean(target**2))),
        "amplitude_fraction": amplitude_mse / max(total_mse, EPS),
        "offset_fraction": offset_mse / max(total_mse, EPS),
        "orthogonal_residual_fraction": residual_mse / max(total_mse, EPS),
        "oracle_diagnostic_only": True,
    }


def _write_spectral_csv(path: Path, audit: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition", "radial_bin", "normalized_frequency", "signal_power", "noise_power", "snr", "reliability"))
        for record in audit["records"]:
            for index, center in enumerate(audit["centers"]):
                writer.writerow((
                    record["condition"], index, float(center),
                    record["signal_power_by_bin"][index],
                    record["noise_power_by_bin"][index],
                    record["snr_by_bin"][index],
                    record["reliability_by_bin"][index],
                ))


def _write_selection_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    conditions = sorted(next(iter(records))["leave_one_condition_out_qy_nrmse"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shrinkage", "smoothing_passes", "mean_LOCO_qy_nrmse", *conditions))
        for row in records:
            writer.writerow((
                row["shrinkage_to_development_optimum"],
                row["radial_smoothing_passes"],
                row["mean_leave_one_condition_out_qy_nrmse"],
                *(row["leave_one_condition_out_qy_nrmse"][condition] for condition in conditions),
            ))


def run_prediction_stage(
    mv9_output_root: Path,
    mv14_output_root: Path,
    output_root: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Select on development data and lock legacy predictions without labels."""

    data_contract = verify_data_contract(mv9_output_root, mv14_output_root)
    mv9_root = Path(mv9_output_root).resolve()
    mv14_root = Path(mv14_output_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite MV15A output: {output_root}")
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["train_x"])
        train_conditions = np.asarray(data["train_condition"])
        train_identities = np.asarray(data["train_identity"])
        validation_x = np.asarray(data["validation_x"])
        validation_y = np.asarray(data["validation_y"])
        validation_conditions = np.asarray(data["validation_condition"])
        test_x = np.asarray(data["test_x"])
        test_conditions = np.asarray(data["test_condition"])
        test_identities = np.asarray(data["test_identity"])
    with np.load(mv14_root / "locked_predictions.npz", allow_pickle=False) as locked:
        locked_conditions = np.asarray(locked["identity_condition"])
        locked_identities = np.asarray(locked["identity_numeric"])
        test_vision_fields = np.asarray(locked["vision_only_fields"])
    if not np.array_equal(test_conditions, locked_conditions) or not np.array_equal(test_identities, locked_identities):
        raise ValueError("MV15A/MV14 legacy identities differ")
    mv14 = _mv14_module()
    validation_vision_fields = mv14._predict_mamba_validation(
        mv9_root, validation_x, batch_size=batch_size
    )
    train_raw_qy = train_x[:, QY_INDEX]
    validation_raw_qy = validation_x[:, QY_INDEX]
    validation_target_qy = validation_y[:, QY_INDEX]
    validation_vision_qy = validation_vision_fields[:, QY_INDEX]
    test_raw_qy = test_x[:, QY_INDEX]
    test_vision_qy = test_vision_fields[:, QY_INDEX]
    audit = cross_spectral_information(
        train_raw_qy, train_conditions, train_identities, bins=RADIAL_BINS
    )
    autocorrelation = block_autocorrelation(
        train_raw_qy, train_conditions, train_identities
    )
    selected, selection_records = select_spectral_fusion(
        validation_raw_qy,
        validation_vision_qy,
        validation_target_qy,
        validation_conditions,
        audit["mapping"],
        audit["global_reliability_by_bin"],
    )
    weights = np.asarray(selected["final_weight_by_bin"], dtype=np.float64)
    weight_map = expand_bin_weights(weights, audit["mapping"])
    fusion_qy = spectral_fuse(test_raw_qy, test_vision_qy, weight_map)
    condition_only_qy = parametric_condition_only(
        validation_x,
        validation_target_qy,
        validation_conditions,
        test_x,
    )
    permutation = cross_condition_permutation(test_conditions)
    permuted_raw_fusion_qy = spectral_fuse(
        test_raw_qy[permutation], test_vision_qy, weight_map
    )
    output_root.mkdir(parents=True)
    np.savez_compressed(
        output_root / "locked_predictions.npz",
        identity_condition=test_conditions,
        identity_numeric=test_identities,
        raw_b1_qy=test_raw_qy,
        vision_only_qy=test_vision_qy,
        spectral_fusion_qy=fusion_qy,
        condition_only_qy=condition_only_qy,
        cross_condition_permuted_raw_fusion_qy=permuted_raw_fusion_qy,
        cross_condition_permutation=permutation,
        radial_bin_map=audit["mapping"],
        radial_bin_centers=audit["centers"],
        global_signal_mode=audit["global_signal_mode"],
        global_noise_mode=audit["global_noise_mode"],
        global_reliability_mode=audit["global_reliability_mode"],
        global_reliability_by_bin=audit["global_reliability_by_bin"],
        selected_raw_weight_by_bin=weights,
        selected_raw_weight_map=weight_map,
    )
    _write_spectral_csv(output_root / "mv15a_spectral_information_audit.csv", audit)
    _write_selection_csv(output_root / "mv15a_development_LOCO_selection.csv", selection_records)
    compact_records = [
        {
            "condition": row["condition"],
            "pair_count": row["pair_count"],
            "linear_MMSE_RMSE_ratio_to_ideal_independent_Raw_B10": row[
                "linear_MMSE_RMSE_ratio_to_ideal_independent_Raw_B10"
            ],
        }
        for row in audit["records"]
    ]
    selection_summary = {
        "stage": STAGE,
        "status": "complete_MV15A_development_selection_and_legacy_prediction",
        "protocol_sha256": _sha256(protocol_path()),
        "mv9_output_root": str(mv9_root),
        "mv14_output_root": str(mv14_root),
        "data_contract": data_contract,
        "selected_spectral_fusion": selected,
        "spectral_information_summary": compact_records,
        "block_autocorrelation": autocorrelation,
        "development_conditions": sorted(str(x) for x in np.unique(validation_conditions)),
        "legacy_conditions_predicted_without_labels": sorted(str(x) for x in np.unique(test_conditions)),
        "legacy_test_targets_loaded": False,
        "legacy_prediction_count": int(len(test_x)),
        "control_arms_locked": [
            "development_parametric_condition_only",
            "cross_condition_permuted_Raw_B1_fusion",
        ],
        "decision": "lock_MV15A_predictions_before_legacy_diagnostic",
    }
    _atomic_json(output_root / "selection_summary.json", selection_summary)
    (output_root / PROTOCOL_FILE).write_bytes(protocol_path().read_bytes())
    locked_files = (
        "locked_predictions.npz",
        "selection_summary.json",
        "mv15a_spectral_information_audit.csv",
        "mv15a_development_LOCO_selection.csv",
        PROTOCOL_FILE,
    )
    _atomic_json(
        output_root / "prediction_manifest.json",
        {
            "stage": STAGE,
            "files": {
                name: {"sha256": _sha256(output_root / name), "size_bytes": (output_root / name).stat().st_size}
                for name in locked_files
            },
        },
    )
    return selection_summary


def _per_seed_qy(
    candidate: np.ndarray,
    target: np.ndarray,
    conditions: np.ndarray,
    identities: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for condition in np.unique(conditions):
        condition_key = str(condition)
        result[condition_key] = {}
        mask_condition = conditions == condition
        for seed in np.unique(identities[mask_condition, 0]):
            mask = mask_condition & (identities[:, 0] == seed)
            result[condition_key][str(int(seed))] = _nrmse(candidate[mask], target[mask])
    return result


def _mean_seed_metric(records: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        condition: float(np.mean(list(per_seed.values())))
        for condition, per_seed in records.items()
    }


def _reference_noise_secondary(
    methods: Mapping[str, np.ndarray],
    target: np.ndarray,
    raw10: np.ndarray,
    target10: np.ndarray,
    conditions: np.ndarray,
    conditions10: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in np.unique(conditions):
        mask = conditions == condition
        mask10 = conditions10 == condition
        samples = np.asarray(raw10[mask10], dtype=np.float64)
        pair_variances = [
            0.5 * np.mean((samples[left] - samples[right]) ** 2)
            for left in range(len(samples))
            for right in range(left + 1, len(samples))
        ]
        single_b10_variance = float(np.mean(pair_variances))
        reference_variance = single_b10_variance / max(len(samples) - 1, 1)
        comparator = max(float(np.mean((raw10[mask10] - target10[mask10]) ** 2)) - reference_variance, 0.0)
        result[str(condition)] = {
            "estimated_single_Raw_B10_noise_MSE": single_b10_variance,
            "estimated_leave_one_seed_out_reference_noise_MSE": reference_variance,
            "corrected_ratio_to_Raw_B10": {
                name: math.sqrt(
                    max(float(np.mean((value[mask] - target[mask]) ** 2)) - reference_variance, 0.0)
                    / max(comparator, EPS)
                )
                for name, value in methods.items()
            },
            "secondary_not_a_gate": True,
        }
    return result


def _write_decomposition_csv(
    path: Path,
    methods: Mapping[str, np.ndarray],
    target: np.ndarray,
    conditions: np.ndarray,
    raw10: np.ndarray,
    target10: np.ndarray,
    conditions10: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    records: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        keys = (
            "slope", "mean_offset", "amplitude_MSE", "offset_MSE",
            "orthogonal_residual_MSE", "total_MSE", "closure_error",
            "target_RMS", "amplitude_fraction", "offset_fraction",
            "orthogonal_residual_fraction",
        )
        writer.writerow(("condition", "method", *keys, "oracle_diagnostic_only"))
        extended = dict(methods)
        for condition in np.unique(conditions):
            condition_key = str(condition)
            records[condition_key] = {}
            mask = conditions == condition
            for name, value in extended.items():
                row = exact_affine_error_decomposition(value[mask], target[mask])
                records[condition_key][name] = row
                writer.writerow((condition_key, name, *(row[key] for key in keys), True))
            mask10 = conditions10 == condition
            row = exact_affine_error_decomposition(raw10[mask10], target10[mask10])
            records[condition_key]["raw_b10"] = row
            writer.writerow((condition_key, "raw_b10", *(row[key] for key in keys), True))
    return records


def _plot_spectral_audit(output_root: Path, selection: Mapping[str, Any], locked: Mapping[str, np.ndarray]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.asarray(locked["radial_bin_centers"])
    reliability = np.asarray(locked["global_reliability_by_bin"])
    weights = np.asarray(locked["selected_raw_weight_by_bin"])
    signal = np.asarray(locked["global_signal_mode"])
    noise = np.asarray(locked["global_noise_mode"])
    autocorrelation = selection["block_autocorrelation"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0, 0].semilogy(centers, np.maximum(reliability / np.maximum(1.0 - reliability, EPS), EPS), "o-")
    axes[0, 0].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="normalized DCT frequency", ylabel="cross-seed SNR", title="B1 spectral information")
    axes[0, 1].plot(centers, reliability, "o-", label="target-free reliability")
    axes[0, 1].plot(centers, weights, "s-", label="selected Raw-B1 weight")
    axes[0, 1].set(xlabel="normalized DCT frequency", ylabel="weight", ylim=(-0.03, 1.03), title="Spectral trust region")
    axes[0, 1].legend(frameon=False)
    image = axes[1, 0].imshow(np.log10((signal + EPS) / (noise + EPS)), origin="lower", cmap="coolwarm", aspect="auto")
    axes[1, 0].set(title="mode-wise log10(S/N)", xlabel="DCT x mode", ylabel="DCT y mode")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.85)
    lags = np.asarray([int(x) for x in autocorrelation])
    values = np.asarray([autocorrelation[str(x)] for x in lags])
    axes[1, 1].bar(lags, values, color="#4472c4")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(xlabel="block lag", ylabel="correlation", title="Within-run B1 block correlation")
    paths = []
    for suffix in ("png", "pdf"):
        path = output_root / f"mv15a_spectral_information_and_trust_region.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_contours(
    output_root: Path,
    fields: Mapping[str, np.ndarray],
    target: np.ndarray,
    scale: float,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("raw_b1", "vision_only", "spectral_fusion", "condition_only", "raw_b10")
    display = ("Raw B1", "MV9 Mamba", "MV15A SITR-QY", "Condition only", "Raw B10")
    physical = [np.asarray(fields[name]) * scale for name in names]
    physical_target = np.asarray(target) * scale
    limit = max(float(np.max(np.abs(value))) for value in (*physical, physical_target))
    levels = np.linspace(-limit, limit, 41)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    arrays = (*physical, physical_target)
    titles = (*display, "Cross-seed reference")
    contour = None
    for axis, value, title in zip(axes.flat, arrays, titles):
        contour = axis.contourf(value, levels=levels, cmap="coolwarm", extend="both")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    assert contour is not None
    fig.colorbar(contour, ax=axes, shrink=0.82, label=r"$q_y$ (W m$^{-2}$)")
    paths = []
    for suffix in ("png", "pdf"):
        path = output_root / f"mv15a_qy_spectral_fusion_physical_contours.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_ratios(output_root: Path, ratios: Mapping[str, Mapping[str, float]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("raw_b1", "vision_only", "spectral_fusion", "condition_only", "tsvd_b1")
    conditions = sorted(ratios["spectral_fusion"])
    x = np.arange(len(conditions), dtype=np.float64)
    width = 0.15
    fig, axis = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    for offset, method in enumerate(methods):
        axis.bar(x + (offset - 2) * width, [ratios[method][condition] for condition in conditions], width=width, label=method)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Raw B10")
    axis.set_xticks(x, conditions, rotation=15)
    axis.set_ylabel(r"mean seed $q_y$ NRMSE / Raw B10")
    axis.set_title("Locked legacy diagnostic by condition")
    axis.legend(ncol=3, frameon=False)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_root / f"mv15a_qy_condition_ratios.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def run_legacy_post(output_root: Path) -> dict[str, Any]:
    """Evaluate recursively locked predictions on legacy labels."""

    protocol = locked_protocol()
    output_root = Path(output_root).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    selection = json.loads((output_root / "selection_summary.json").read_text(encoding="utf-8"))
    if selection.get("legacy_test_targets_loaded") is not False:
        raise ValueError("MV15A prediction/label separation failed")
    mv9_root = Path(selection["mv9_output_root"]).resolve()
    with np.load(output_root / "locked_predictions.npz", allow_pickle=False) as data:
        locked = {key: np.asarray(data[key]) for key in data.files}
    with np.load(mv9_root / "dataset.npz", allow_pickle=False) as data:
        target = np.asarray(data["test_y"])[:, QY_INDEX]
        conditions = np.asarray(data["test_condition"])
        identities = np.asarray(data["test_identity"])
        scales = np.asarray(data["test_scale"])
        gaussian = np.asarray(data["test_gaussian"])[:, QY_INDEX]
        tsvd = np.asarray(data["test_tsvd"])[:, QY_INDEX]
        raw10 = np.asarray(data["test_raw10"])[:, QY_INDEX]
        target10 = np.asarray(data["test_target10"])[:, QY_INDEX]
        conditions10 = np.asarray(data["test_condition10"])
        identities10 = np.asarray(data["test_identity10"])
        scales10 = np.asarray(data["test_scale10"])
    if not np.array_equal(conditions, locked["identity_condition"]) or not np.array_equal(identities, locked["identity_numeric"]):
        raise ValueError("MV15A locked identities differ from legacy labels")
    methods = {
        "raw_b1": locked["raw_b1_qy"],
        "vision_only": locked["vision_only_qy"],
        "spectral_fusion": locked["spectral_fusion_qy"],
        "condition_only": locked["condition_only_qy"],
        "permuted_raw_fusion": locked["cross_condition_permuted_raw_fusion_qy"],
        "gaussian_b1": gaussian,
        "tsvd_b1": tsvd,
    }
    per_seed = {name: _per_seed_qy(value, target, conditions, identities) for name, value in methods.items()}
    per_seed["raw_b10"] = _per_seed_qy(raw10, target10, conditions10, identities10)
    aggregate = {name: _mean_seed_metric(records) for name, records in per_seed.items()}
    ratios = {
        name: {
            condition: value / max(aggregate["raw_b10"][condition], EPS)
            for condition, value in records.items()
        }
        for name, records in aggregate.items()
        if name != "raw_b10"
    }
    primary = str(protocol["analysis_contract"]["primary_condition"])
    primary_seed_ratios = {
        seed: value / max(per_seed["raw_b10"][primary][seed], EPS)
        for seed, value in per_seed["spectral_fusion"][primary].items()
    }
    contract = protocol["analysis_contract"]
    gates = {
        "primary_qy_no_worse_than_Raw_B10": ratios["spectral_fusion"][primary] <= float(contract["maximum_primary_mean_qy_ratio_to_Raw_B10"]),
        "every_primary_seed_within_cap": max(primary_seed_ratios.values()) <= float(contract["maximum_individual_primary_seed_qy_ratio_to_Raw_B10"]),
        "no_condition_mean_worse_than_Raw_B10": max(ratios["spectral_fusion"].values()) <= float(contract["maximum_each_condition_mean_qy_ratio_to_Raw_B10"]),
        "fusion_beats_vision_only_primary": aggregate["spectral_fusion"][primary] < aggregate["vision_only"][primary],
        "fusion_beats_TSVD_B1_primary": aggregate["spectral_fusion"][primary] < aggregate["tsvd_b1"][primary],
        "fusion_beats_condition_only_primary": aggregate["spectral_fusion"][primary] < aggregate["condition_only"][primary],
        "cross_condition_permutation_degrades_by_preregistered_margin": aggregate["permuted_raw_fusion"][primary] >= (1.0 + float(contract["permuted_control_minimum_degradation_fraction"])) * aggregate["spectral_fusion"][primary],
        "prediction_hash_locked_before_legacy_label_access": True,
        "legacy_diagnostic_not_reclassified_as_confirmation": True,
    }
    with (output_root / "mv15a_legacy_qy_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("condition", "method", "mean_seed_qy_nrmse", "ratio_to_Raw_B10"))
        for method in sorted(aggregate):
            for condition in sorted(aggregate[method]):
                ratio = 1.0 if method == "raw_b10" else ratios[method][condition]
                writer.writerow((condition, method, aggregate[method][condition], ratio))
    decomposition = _write_decomposition_csv(
        output_root / "mv15a_exact_error_decomposition.csv",
        methods, target, conditions, raw10, target10, conditions10,
    )
    corrected = _reference_noise_secondary(
        methods, target, raw10, target10, conditions, conditions10
    )
    representative_seed = int(contract["representative_seed"])
    representative_block = int(contract["representative_block"])
    mask = (conditions == primary) & (identities[:, 0] == representative_seed) & (identities[:, 1] == representative_block)
    mask10 = (conditions10 == primary) & (identities10[:, 0] == representative_seed)
    if np.count_nonzero(mask) != 1 or np.count_nonzero(mask10) != 1:
        raise ValueError("MV15A representative identity is absent")
    index, index10 = int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask10)[0])
    if not np.array_equal(target[index], target10[index10]):
        raise ValueError("MV15A representative references differ")
    if not np.isclose(scales[index, QY_INDEX], scales10[index10, QY_INDEX], rtol=1.0e-12):
        raise ValueError("MV15A representative physical scales differ")
    figures = []
    figures.extend(_plot_spectral_audit(output_root, selection, locked))
    figures.extend(_plot_contours(
        output_root,
        {
            "raw_b1": methods["raw_b1"][index],
            "vision_only": methods["vision_only"][index],
            "spectral_fusion": methods["spectral_fusion"][index],
            "condition_only": methods["condition_only"][index],
            "raw_b10": raw10[index10],
        },
        target[index],
        float(scales[index, QY_INDEX]),
    ))
    figures.extend(_plot_ratios(output_root, ratios))
    supports_fresh = all(gates.values())
    summary = {
        "stage": STAGE,
        "status": "complete_MV15A_post_lock_legacy_diagnostic",
        "protocol_sha256": _sha256(protocol_path()),
        "scientific_classification": protocol["scientific_role"]["classification"],
        "primary_condition": primary,
        "primary_qy_ratios_to_Raw_B10": {name: values[primary] for name, values in ratios.items()},
        "all_condition_qy_ratios_to_Raw_B10": ratios,
        "primary_spectral_fusion_per_seed_ratios_to_Raw_B10": primary_seed_ratios,
        "gates_for_authorizing_fresh_DSMC": gates,
        "exact_error_decomposition": decomposition,
        "reference_noise_corrected_secondary_analysis": corrected,
        "spectral_information_summary": selection["spectral_information_summary"],
        "selected_spectral_fusion": selection["selected_spectral_fusion"],
        "figures": figures,
        "old_evaluation_seeds_are_confirmation": False,
        "fresh_seed_and_fresh_condition_confirmation_still_required": True,
        "decision": (
            "MV15A_legacy_diagnostic_authorizes_separately_locked_fresh_DSMC"
            if supports_fresh
            else "MV15A_legacy_diagnostic_does_not_authorize_fresh_DSMC"
        ),
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


def package_results(output_root: Path, return_directory: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    return_directory = Path(return_directory).resolve()
    _verify_manifest(output_root, "prediction_manifest.json")
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    names = [
        "selection_summary.json",
        "prediction_manifest.json",
        "summary.json",
        PROTOCOL_FILE,
        "mv15a_spectral_information_audit.csv",
        "mv15a_development_LOCO_selection.csv",
        "mv15a_legacy_qy_metrics.csv",
        "mv15a_exact_error_decomposition.csv",
        "mv15a_spectral_information_and_trust_region.png",
        "mv15a_spectral_information_and_trust_region.pdf",
        "mv15a_qy_spectral_fusion_physical_contours.png",
        "mv15a_qy_spectral_fusion_physical_contours.pdf",
        "mv15a_qy_condition_ratios.png",
        "mv15a_qy_condition_ratios.pdf",
    ]
    accounting = output_root / "slurm_accounting.psv"
    if accounting.is_file():
        names.append(accounting.name)
    generated = [output_root / name for name in names]
    for path in generated:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "stage": STAGE,
        "status": "complete_MV15A_return_artifact_manifest",
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in generated
        },
    }
    _atomic_json(output_root / "artifact_manifest.json", manifest)
    _verify_manifest(output_root, "artifact_manifest.json")
    verification = {
        "stage": STAGE,
        "status": "complete_MV15A_recursive_return_verification",
        "decision": "verified",
        "verified_file_count": len(manifest["files"]),
        "manifest_sha256": _sha256(output_root / "artifact_manifest.json"),
    }
    _atomic_json(output_root / "verification.json", verification)
    generated.extend((output_root / "artifact_manifest.json", output_root / "verification.json"))
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return_directory.mkdir(parents=True, exist_ok=True)
    archive = return_directory / f"MV15A_SPECTRAL_INFORMATION_AUDIT_BUNDLE_{tag}.zip"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite MV15A archive: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in generated:
            stream.write(path, arcname=path.name)
    if archive.stat().st_size > 450 * 1024 * 1024:
        raise RuntimeError("MV15A return archive exceeds 450 MiB")
    result = {
        "stage": STAGE,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "decision": summary["decision"],
        "primary_spectral_fusion_qy_ratio_to_Raw_B10": summary["primary_qy_ratios_to_Raw_B10"]["spectral_fusion"],
    }
    _atomic_json(output_root / "return.json", result)
    pointer = return_directory / "LAST_MOHAMMADZADEH_MV15A_SPECTRAL_AUDIT_RESULT.env"
    pointer.write_text(
        "\n".join((
            f"MV15A_OUTPUT_ROOT={output_root}",
            f"MV15A_RESULT_ARCHIVE={archive}",
            f"MV15A_RESULT_ARCHIVE_SHA256={result['archive_sha256']}",
            f"MV15A_DECISION={result['decision']}",
            f"MV15A_PRIMARY_QY_RATIO_TO_RAW_B10={result['primary_spectral_fusion_qy_ratio_to_Raw_B10']}",
            "",
        )),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-lock")
    verify = subparsers.add_parser("verify-data")
    verify.add_argument("--mv9-output-root", type=Path, required=True)
    verify.add_argument("--mv14-output-root", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--mv9-output-root", type=Path, required=True)
    predict.add_argument("--mv14-output-root", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--batch-size", type=int, default=8)
    post = subparsers.add_parser("post")
    post.add_argument("--output-root", type=Path, required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--output-root", type=Path, required=True)
    package.add_argument("--return-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-lock":
        result = verify_lock()
    elif args.command == "verify-data":
        result = verify_data_contract(args.mv9_output_root, args.mv14_output_root)
    elif args.command == "predict":
        result = run_prediction_stage(
            args.mv9_output_root, args.mv14_output_root, args.output_root,
            batch_size=args.batch_size,
        )
    elif args.command == "post":
        result = run_legacy_post(args.output_root)
    else:
        result = package_results(args.output_root, args.return_directory)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
