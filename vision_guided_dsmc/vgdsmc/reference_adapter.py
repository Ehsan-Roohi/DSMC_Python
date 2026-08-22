from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .vhs_model import KB, MASS_AR

REQUIRED_REFERENCE_FIELDS = ("T", "rho", "u", "v")


def load_reference_npz(path: str | Path, expected_shape: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [name for name in REQUIRED_REFERENCE_FIELDS if name not in data]
        if missing:
            raise ValueError(f"Reference is missing fields: {missing}")
        fields = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
    shape = fields["T"].shape
    if len(shape) != 2:
        raise ValueError("Reference fields must be two-dimensional cell maps")
    for name in REQUIRED_REFERENCE_FIELDS:
        if fields[name].shape != shape or not np.isfinite(fields[name]).all():
            raise ValueError(f"Invalid reference field {name!r}")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Reference shape {shape} does not match expected {expected_shape}")
    if np.any(fields["T"] <= 0.0) or np.any(fields["rho"] <= 0.0):
        raise ValueError("Reference temperature and density must be positive")
    return fields


def deterministic_error_map(coarse, reference, temperature_weight=0.45, velocity_weight=0.25, density_weight=0.30):
    shape = reference["T"].shape
    for name in REQUIRED_REFERENCE_FIELDS:
        if name not in coarse or np.asarray(coarse[name]).shape != shape:
            raise ValueError(f"Coarse field {name!r} is missing or not aligned")
    weights = np.array([temperature_weight, velocity_weight, density_weight], dtype=float)
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Error weights must be nonnegative and sum to one")
    e_t = np.abs(coarse["T"] - reference["T"]) / np.maximum(reference["T"], 1.0e-30)
    coarse_speed = np.hypot(coarse["u"], coarse["v"])
    reference_speed = np.hypot(reference["u"], reference["v"])
    thermal_speed = np.sqrt(2.0 * KB * float(np.mean(reference["T"])) / MASS_AR)
    e_u = np.abs(coarse_speed - reference_speed) / max(thermal_speed, 1.0e-30)
    e_rho = np.abs(coarse["rho"] - reference["rho"]) / np.maximum(reference["rho"], 1.0e-30)
    return weights[0] * e_t + weights[1] * e_u + weights[2] * e_rho


def quantile_labels(score, lower_quantile=1.0 / 3.0, upper_quantile=2.0 / 3.0):
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Require 0 < lower_quantile < upper_quantile < 1")
    flat = np.asarray(score, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="mergesort")
    first = int(np.clip(round(lower_quantile * len(flat)), 1, len(flat) - 2))
    second = int(np.clip(round(upper_quantile * len(flat)), first + 1, len(flat) - 1))
    flat_label = np.empty(len(flat), dtype=np.int64)
    flat_label[order[:first]] = 0
    flat_label[order[first:second]] = 1
    flat_label[order[second:]] = 2
    return flat_label.reshape(score.shape), (float(flat[order[first]]), float(flat[order[second]]))


def build_supervised_reference_case(coarse_case_path, reference_path, output_path):
    coarse_case_path = Path(coarse_case_path)
    with np.load(coarse_case_path) as data:
        if "x" not in data:
            raise ValueError("Coarse case must contain the flow-channel array 'x'")
        x = np.asarray(data["x"], dtype=np.float32)
        context = np.asarray(data["context"], dtype=np.float32) if "context" in data else None
        case_seed = np.int64(data["case_seed"]) if "case_seed" in data else None
        coarse = {}
        for name in REQUIRED_REFERENCE_FIELDS:
            key = f"coarse_{name}"
            if key not in data:
                raise ValueError(f"Coarse case is missing {key!r}")
            coarse[name] = np.asarray(data[key], dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 4:
        raise ValueError("Coarse input 'x' must have shape (channels, ny, nx)")
    if context is not None and (context.ndim != 1 or not np.isfinite(context).all()):
        raise ValueError("Optional physical context must be a finite one-dimensional array")
    reference = load_reference_npz(reference_path, expected_shape=coarse["T"].shape)
    score = deterministic_error_map(coarse, reference)
    label, thresholds = quantile_labels(score)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x": x,
        "score": score.astype(np.float32),
        "label": label,
        "threshold_low": np.float64(thresholds[0]),
        "threshold_high": np.float64(thresholds[1]),
        **{f"coarse_{name}": value for name, value in coarse.items()},
        **{f"reference_{name}": reference[name] for name in reference},
    }
    if context is not None:
        arrays["context"] = context
    if case_seed is not None:
        arrays["case_seed"] = case_seed
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "coarse_case": str(coarse_case_path),
        "reference": str(reference_path),
        "required_reference_fields": list(REQUIRED_REFERENCE_FIELDS),
        "shape": list(score.shape),
        "class_counts": np.bincount(label.ravel(), minlength=3).tolist(),
        "thresholds": {"low": thresholds[0], "high": thresholds[1]},
        "context": context.tolist() if context is not None else None,
        "case_seed": int(case_seed) if case_seed is not None else None,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path
