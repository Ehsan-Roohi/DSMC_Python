from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .vhs_model import KB, MASS_AR

REQUIRED_REFERENCE_FIELDS = ("T", "rho", "u", "v")


def load_reference_npz(
    path: str | Path,
    expected_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Load a deterministic kinetic reference with a strict field contract."""
    with np.load(path) as data:
        missing = [name for name in REQUIRED_REFERENCE_FIELDS if name not in data]
        if missing:
            raise ValueError(f"Reference is missing fields: {missing}")
        fields = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
    shape = fields["T"].shape
    if len(shape) != 2:
        raise ValueError("Reference fields must be two-dimensional cell maps")
    for name in REQUIRED_REFERENCE_FIELDS:
        if fields[name].shape != shape:
            raise ValueError(f"Field {name!r} has shape {fields[name].shape}, expected {shape}")
        if not np.isfinite(fields[name]).all():
            raise ValueError(f"Field {name!r} contains non-finite values")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Reference shape {shape} does not match expected {expected_shape}")
    if np.any(fields["T"] <= 0.0) or np.any(fields["rho"] <= 0.0):
        raise ValueError("Reference temperature and density must be positive")
    return fields


def deterministic_error_map(
    coarse: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    temperature_weight: float = 0.45,
    velocity_weight: float = 0.25,
    density_weight: float = 0.30,
) -> np.ndarray:
    """Return a dimensionless local error target against a clean reference."""
    shape = reference["T"].shape
    for name in REQUIRED_REFERENCE_FIELDS:
        if name not in coarse:
            raise ValueError(f"Coarse DSMC fields are missing {name!r}")
        if np.asarray(coarse[name]).shape != shape:
            raise ValueError(f"Coarse field {name!r} is not aligned with the reference")
    weights = np.array([temperature_weight, velocity_weight, density_weight], dtype=float)
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Error weights must be nonnegative and sum to one")

    e_temperature = np.abs(coarse["T"] - reference["T"]) / np.maximum(reference["T"], 1.0e-30)
    coarse_speed = np.hypot(coarse["u"], coarse["v"])
    reference_speed = np.hypot(reference["u"], reference["v"])
    thermal_speed = np.sqrt(2.0 * KB * float(np.mean(reference["T"])) / MASS_AR)
    e_velocity = np.abs(coarse_speed - reference_speed) / max(thermal_speed, 1.0e-30)
    e_density = np.abs(coarse["rho"] - reference["rho"]) / np.maximum(reference["rho"], 1.0e-30)
    return weights[0] * e_temperature + weights[1] * e_velocity + weights[2] * e_density


def quantile_labels(
    score: np.ndarray,
    lower_quantile: float = 1.0 / 3.0,
    upper_quantile: float = 2.0 / 3.0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Convert a continuous reference target to rank-balanced classes."""
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


def build_supervised_reference_case(
    coarse_case_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Combine a coarse DSMC case and deterministic reference into ML-ready data."""
    coarse_case_path = Path(coarse_case_path)
    with np.load(coarse_case_path) as data:
        if "x" not in data:
            raise ValueError("Coarse case must contain the flow-channel array 'x'")
        x = np.asarray(data["x"], dtype=np.float32)
        context = np.asarray(data["context"], dtype=np.float32) if "context" in data else None
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
    np.savez_compressed(output_path, **arrays)
    metadata = {
        "coarse_case": str(coarse_case_path),
        "reference": str(reference_path),
        "required_reference_fields": list(REQUIRED_REFERENCE_FIELDS),
        "shape": list(score.shape),
        "class_counts": np.bincount(label.ravel(), minlength=3).tolist(),
        "thresholds": {"low": thresholds[0], "high": thresholds[1]},
        "context": context.tolist() if context is not None else None,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path
