#!/usr/bin/env python3
"""Normalize, align, validate, and ensemble SPARTA normal-shock profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import tempfile


K_B = 1.380649e-23

RAW_NAMES = (
    "number_density", "u", "v", "w", "temperature", "pressure",
    "Pxx", "Pyy", "Pzz", "qx",
)
NORMALIZED_NAMES = (
    "x_over_lambda_1", "number_density_over_n1", "u_over_u1",
    "temperature_over_T1", "Tx_over_T1", "Tperp_over_T1",
    "Pxx_over_p1", "qx_over_n1kT1u1", "mass_flux_over_upstream",
    "momentum_flux_over_upstream", "energy_flux_over_upstream",
)

FAR_FIELD_MAX_RELATIVE_ERROR_LIMIT = 0.03
FLUX_MAXIMUM_RELATIVE_ERROR_FROM_UPSTREAM_LIMIT = 0.005
PLATEAU_MAX_RELATIVE_SLOPE_PER_LAMBDA_LIMIT = 0.005
CHECKPOINT_SHOCK_CENTER_CHANGE_LIMIT = 0.5
CHECKPOINT_THICKNESS_RELATIVE_CHANGE_LIMIT = 0.03
CHECKPOINT_FAR_FIELD_RELATIVE_CHANGE_LIMIT = 0.01
CHECKPOINT_PROFILE_RMS_RELATIVE_CHANGE_LIMIT = 0.02
CHECKPOINT_PROFILE_MAX_RELATIVE_CHANGE_LIMIT = 0.05
CHECKPOINT_PROFILE_WINDOW = (-10.0, 10.0)

# Two-sided 95% Student-t critical values.  Production uses three independent
# seeds (df=2), for which 1.96 would materially understate uncertainty.
STUDENT_T_975 = {
    1: 12.7062047364,
    2: 4.30265272991,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183564,
    6: 2.44691184879,
    7: 2.36462425101,
    8: 2.30600413503,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282967,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.0738730679,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.0484071418,
    29: 2.04522964213,
    30: 2.0422724563,
}


def read_last_grid_snapshot(path: Path) -> list[dict[str, float]]:
    """Read the last ITEM: CELLS block from a SPARTA grid dump."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header_indexes = [i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS")]
    if not header_indexes:
        raise ValueError(f"No ITEM: CELLS block in {path}")
    start = header_indexes[-1]
    columns = lines[start].split()[2:]
    expected_width = 2 + len(RAW_NAMES)
    if len(columns) != expected_width:
        raise ValueError(
            f"Expected {expected_width} dump columns (id, xc, 10 fields), got {len(columns)}"
        )
    rows: list[dict[str, float]] = []
    for line in lines[start + 1 :]:
        if line.startswith("ITEM:"):
            break
        fields = line.split()
        if not fields:
            continue
        if len(fields) != expected_width:
            raise ValueError(f"Unexpected dump row width {len(fields)}: {line}")
        values = [float(value) for value in fields]
        row = {"id": values[0], "x": values[1]}
        row.update(dict(zip(RAW_NAMES, values[2:])))
        rows.append(row)
    if not rows:
        raise ValueError(f"No grid rows in {path}")
    return sorted(rows, key=lambda row: row["x"])


def crossing_x(
    x: list[float], y: list[float], target: float, prefer: float = 0.0,
    require_crossing: bool = False,
) -> float:
    """Linearly locate a target crossing, choosing the one nearest *prefer*."""
    candidates: list[float] = []
    for index in range(len(x) - 1):
        a = y[index] - target
        b = y[index + 1] - target
        if a == 0.0:
            candidates.append(x[index])
        elif a * b < 0.0:
            fraction = -a / (b - a)
            candidates.append(x[index] + fraction * (x[index + 1] - x[index]))
    if candidates:
        return min(candidates, key=lambda value: abs(value - prefer))
    if require_crossing:
        raise ValueError(f"Profile does not cross required target {target}")
    return x[min(range(len(x)), key=lambda index: abs(y[index] - target))]


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-300)


def mean_rows(rows: list[dict[str, float]], indexes: list[int], key: str) -> float:
    if not indexes:
        raise ValueError(f"Empty averaging window for {key}")
    return fmean(rows[index][key] for index in indexes)


def indexes_in_window(x: list[float], bounds: tuple[float, float]) -> list[int]:
    lower, upper = bounds
    indexes = [index for index, value in enumerate(x) if lower <= value <= upper]
    if len(indexes) < 2:
        raise ValueError(f"Window {bounds} contains fewer than two grid cells")
    return indexes


def validation_windows(meta: dict[str, object]) -> dict[str, tuple[float, float]]:
    """Return physical (not shock-aligned) x/lambda_1 validation windows."""
    half_span = float(meta["half_span_lambda"])
    if half_span >= 30.0:
        return {
            "upstream": (-28.0, -24.0),
            "downstream": (24.0, 28.0),
            "interior": (-28.0, 28.0),
        }
    # Smoke/pilot and synthetic unit-test cases retain scale-aware windows.
    return {
        "upstream": (-0.90 * half_span, -0.70 * half_span),
        "downstream": (0.70 * half_span, 0.90 * half_span),
        "interior": (-0.90 * half_span, 0.90 * half_span),
    }


def relative_linear_slope(
    rows: list[dict[str, float]], indexes: list[int], x: list[float], key: str
) -> float:
    """Return |d(field)/d(x/lambda)| normalized by the window mean."""
    x_mean = fmean(x[index] for index in indexes)
    y_mean = mean_rows(rows, indexes, key)
    numerator = sum(
        (x[index] - x_mean) * (rows[index][key] - y_mean) for index in indexes
    )
    denominator = sum((x[index] - x_mean) ** 2 for index in indexes)
    if denominator <= 0.0:
        raise ValueError(f"Degenerate x coordinates in plateau window for {key}")
    return abs(numerator / denominator) / max(abs(y_mean), 1.0e-300)


def student_t_critical_95(realization_count: int) -> float:
    if realization_count < 2:
        raise ValueError("Student-t confidence intervals require at least two realizations")
    degrees_of_freedom = realization_count - 1
    return STUDENT_T_975.get(degrees_of_freedom, 1.95996398454)


def normalize_rows(rows: list[dict[str, float]], meta: dict[str, object]) -> tuple[list[dict[str, float]], dict[str, object]]:
    n1 = float(meta["number_density_1"])
    t1 = float(meta["temperature_1"])
    u1 = float(meta["velocity_1"])
    mass = float(meta["argon_mass_kg"])
    lambda1 = float(meta["mean_free_path_1"])
    ratio = float(meta["density_ratio"])
    p1 = n1 * K_B * t1
    rho1 = mass * n1
    mass_flux_1 = rho1 * u1
    momentum_flux_1 = rho1 * u1**2 + p1
    energy_flux_1 = u1 * (0.5 * rho1 * u1**2 + 2.5 * p1)

    x = [row["x"] for row in rows]
    density_ratio = [row["number_density"] / n1 for row in rows]
    midpoint = 0.5 * (1.0 + ratio)
    shock_center = crossing_x(x, density_ratio, midpoint, require_crossing=True)
    aligned: list[dict[str, float]] = []
    for row in rows:
        n = row["number_density"]
        rho = mass * n
        u = row["u"]
        pxx, pyy, pzz, qx = row["Pxx"], row["Pyy"], row["Pzz"], row["qx"]
        tx = pxx / (max(n, 1.0e-300) * K_B)
        tperp = 0.5 * (pyy + pzz) / (max(n, 1.0e-300) * K_B)
        mass_flux = rho * u
        momentum_flux = rho * u**2 + pxx
        energy_flux = u * (
            0.5 * rho * u**2 + 0.5 * (pxx + pyy + pzz) + pxx
        ) + qx
        aligned.append({
            "x_over_lambda_1": (row["x"] - shock_center) / lambda1,
            "number_density_over_n1": n / n1,
            "u_over_u1": u / u1,
            "temperature_over_T1": row["temperature"] / t1,
            "Tx_over_T1": tx / t1,
            "Tperp_over_T1": tperp / t1,
            "Pxx_over_p1": pxx / p1,
            "qx_over_n1kT1u1": qx / (p1 * u1),
            "mass_flux_over_upstream": mass_flux / mass_flux_1,
            "momentum_flux_over_upstream": momentum_flux / momentum_flux_1,
            "energy_flux_over_upstream": energy_flux / energy_flux_1,
        })

    xa = [row["x_over_lambda_1"] for row in aligned]
    nd = [row["number_density_over_n1"] for row in aligned]
    x10 = crossing_x(
        xa, nd, 1.0 + 0.1 * (ratio - 1.0), prefer=-1.0,
        require_crossing=True,
    )
    x90 = crossing_x(
        xa, nd, 1.0 + 0.9 * (ratio - 1.0), prefer=1.0,
        require_crossing=True,
    )
    if x90 <= x10:
        raise ValueError(
            f"Invalid density-thickness crossings: x10={x10}, x90={x90}"
        )
    # Validate in physical reservoir windows.  Alignment is only for comparing
    # profile shape and must not move the regions used for boundary checks.
    physical_x_over_lambda = [row["x"] / lambda1 for row in rows]
    windows = validation_windows(meta)
    upstream = indexes_in_window(physical_x_over_lambda, windows["upstream"])
    downstream = indexes_in_window(physical_x_over_lambda, windows["downstream"])
    interior = indexes_in_window(physical_x_over_lambda, windows["interior"])
    expected = {
        "number_density_over_n1": (1.0, ratio),
        "u_over_u1": (1.0, 1.0 / ratio),
        "temperature_over_T1": (1.0, float(meta["temperature_ratio"])),
    }
    far_field: dict[str, object] = {}
    far_errors: list[float] = []
    for key, (left_expected, right_expected) in expected.items():
        left = mean_rows(aligned, upstream, key)
        right = mean_rows(aligned, downstream, key)
        left_error = relative_error(left, left_expected)
        right_error = relative_error(right, right_expected)
        far_errors.extend((left_error, right_error))
        far_field[key] = {
            "upstream_mean": left,
            "upstream_expected": left_expected,
            "upstream_relative_error": left_error,
            "downstream_mean": right,
            "downstream_expected": right_expected,
            "downstream_relative_error": right_error,
        }
    conservation: dict[str, object] = {}
    flux_errors: list[float] = []
    for key in (
        "mass_flux_over_upstream",
        "momentum_flux_over_upstream",
        "energy_flux_over_upstream",
    ):
        values = [aligned[index][key] for index in interior]
        mean_value = fmean(values)
        max_deviation = max(abs(value - mean_value) for value in values) / max(abs(mean_value), 1.0e-300)
        mean_error = abs(mean_value - 1.0)
        maximum_error_from_upstream = max(abs(value - 1.0) for value in values)
        flux_errors.append(maximum_error_from_upstream)
        conservation[key] = {
            "mean": mean_value,
            "mean_relative_error_from_upstream": mean_error,
            "maximum_relative_deviation_from_mean": max_deviation,
            "maximum_relative_error_from_upstream": maximum_error_from_upstream,
        }
    plateau: dict[str, object] = {}
    plateau_slopes: list[float] = []
    for side, indexes in (("upstream", upstream), ("downstream", downstream)):
        side_metrics: dict[str, float] = {}
        for key in expected:
            slope = relative_linear_slope(aligned, indexes, physical_x_over_lambda, key)
            plateau_slopes.append(slope)
            side_metrics[key] = slope
        plateau[side] = side_metrics
    metrics: dict[str, object] = {
        "alignment_method": "unsmoothed density-midpoint translation",
        "smoothing": "none",
        "shock_center_m": shock_center,
        "shock_center_over_lambda_1_before_alignment": shock_center / lambda1,
        "density_10_90_thickness_over_lambda_1": x90 - x10,
        "validation_windows_x_over_lambda_1": {
            key: list(bounds) for key, bounds in windows.items()
        },
        "far_field": far_field,
        "conservation": conservation,
        "plateau_relative_slope_per_lambda_1": plateau,
        "gates": {
            "far_field_max_relative_error_limit": FAR_FIELD_MAX_RELATIVE_ERROR_LIMIT,
            "flux_maximum_relative_error_from_upstream_limit": FLUX_MAXIMUM_RELATIVE_ERROR_FROM_UPSTREAM_LIMIT,
            "plateau_max_relative_slope_per_lambda_1_limit": PLATEAU_MAX_RELATIVE_SLOPE_PER_LAMBDA_LIMIT,
            "far_field_max_relative_error": max(far_errors),
            "flux_maximum_relative_error_from_upstream": max(flux_errors),
            "plateau_max_relative_slope_per_lambda_1": max(plateau_slopes),
        },
    }
    metrics["physics_gate_pass"] = (
        max(far_errors) <= FAR_FIELD_MAX_RELATIVE_ERROR_LIMIT
        and max(flux_errors) <= FLUX_MAXIMUM_RELATIVE_ERROR_FROM_UPSTREAM_LIMIT
        and max(plateau_slopes) <= PLATEAU_MAX_RELATIVE_SLOPE_PER_LAMBDA_LIMIT
    )
    return aligned, metrics


def write_csv(path: Path, rows: list[dict[str, float]], names: tuple[str, ...] = NORMALIZED_NAMES) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(names))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_single(path: Path, rows: list[dict[str, float]], mach: float) -> None:
    cache = Path(tempfile.gettempdir()) / "sparta-normal-shock-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = [row["x_over_lambda_1"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    series = (
        ("number_density_over_n1", r"$n/n_1$", "#0b4f9c"),
        ("u_over_u1", r"$u/u_1$", "#1d8a99"),
        ("temperature_over_T1", r"$T/T_1$", "#d73027"),
        ("qx_over_n1kT1u1", r"$q_x/(n_1kT_1u_1)$", "#7b3294"),
    )
    for axis, (key, label, color) in zip(axes.flat, series):
        axis.plot(x, [row[key] for row in rows], color=color, linewidth=1.3)
        axis.set(xlabel=r"$(x-x_s)/\lambda_1$", ylabel=label)
        axis.grid(alpha=0.2)
    figure.suptitle(f"SPARTA steady normal shock, M1={mach:g} (no smoothing)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def checkpoint_stability_metrics(
    current: dict[str, object], previous: dict[str, object],
    current_rows: list[dict[str, float]], previous_rows: list[dict[str, float]],
) -> dict[str, object]:
    """Compare nested final and preceding cumulative-average checkpoints."""
    center_change = abs(
        float(current["shock_center_over_lambda_1_before_alignment"])
        - float(previous["shock_center_over_lambda_1_before_alignment"])
    )
    thickness_change = relative_error(
        float(current["density_10_90_thickness_over_lambda_1"]),
        float(previous["density_10_90_thickness_over_lambda_1"]),
    )
    far_field_changes: dict[str, object] = {}
    change_values: list[float] = []
    current_far = current["far_field"]
    previous_far = previous["far_field"]
    assert isinstance(current_far, dict) and isinstance(previous_far, dict)
    for key in (
        "number_density_over_n1", "u_over_u1", "temperature_over_T1",
    ):
        key_changes: dict[str, float] = {}
        current_key = current_far[key]
        previous_key = previous_far[key]
        assert isinstance(current_key, dict) and isinstance(previous_key, dict)
        for side in ("upstream", "downstream"):
            field = f"{side}_mean"
            change = relative_error(float(current_key[field]), float(previous_key[field]))
            key_changes[side] = change
            change_values.append(change)
        far_field_changes[key] = key_changes
    maximum_far_field_change = max(change_values)
    lower = max(
        current_rows[0]["x_over_lambda_1"],
        previous_rows[0]["x_over_lambda_1"],
        CHECKPOINT_PROFILE_WINDOW[0],
    )
    upper = min(
        current_rows[-1]["x_over_lambda_1"],
        previous_rows[-1]["x_over_lambda_1"],
        CHECKPOINT_PROFILE_WINDOW[1],
    )
    profile_grid = [
        row["x_over_lambda_1"] for row in current_rows
        if lower <= row["x_over_lambda_1"] <= upper
    ]
    if len(profile_grid) < 2:
        raise ValueError("No usable common support for checkpoint profile comparison")
    profile_changes: dict[str, object] = {}
    rms_changes: list[float] = []
    maximum_changes: list[float] = []
    for key in NORMALIZED_NAMES[1:]:
        current_values = [interpolate(current_rows, xnew, key) for xnew in profile_grid]
        previous_values = [interpolate(previous_rows, xnew, key) for xnew in profile_grid]
        rms_difference = math.sqrt(fmean(
            (current_value - previous_value) ** 2
            for current_value, previous_value in zip(current_values, previous_values)
        ))
        rms_scale = math.sqrt(fmean(value**2 for value in current_values))
        maximum_difference = max(
            abs(current_value - previous_value)
            for current_value, previous_value in zip(current_values, previous_values)
        )
        maximum_scale = max(abs(value) for value in current_values)
        rms_change = rms_difference / max(rms_scale, 1.0e-300)
        maximum_change = maximum_difference / max(maximum_scale, 1.0e-300)
        rms_changes.append(rms_change)
        maximum_changes.append(maximum_change)
        profile_changes[key] = {
            "rms_relative_change": rms_change,
            "maximum_relative_change": maximum_change,
        }
    maximum_profile_rms_change = max(rms_changes)
    maximum_profile_change = max(maximum_changes)
    passed = (
        center_change <= CHECKPOINT_SHOCK_CENTER_CHANGE_LIMIT
        and thickness_change <= CHECKPOINT_THICKNESS_RELATIVE_CHANGE_LIMIT
        and maximum_far_field_change <= CHECKPOINT_FAR_FIELD_RELATIVE_CHANGE_LIMIT
        and maximum_profile_rms_change <= CHECKPOINT_PROFILE_RMS_RELATIVE_CHANGE_LIMIT
        and maximum_profile_change <= CHECKPOINT_PROFILE_MAX_RELATIVE_CHANGE_LIMIT
    )
    return {
        "required": True,
        "available": True,
        "comparison": "final cumulative average versus preceding cumulative checkpoint",
        "comparison_samples_are_nested": True,
        "interpretation": "checkpoint stability diagnostic, not an independent-sample hypothesis test",
        "shock_center_change_over_lambda_1": center_change,
        "density_10_90_thickness_relative_change": thickness_change,
        "far_field_relative_changes": far_field_changes,
        "far_field_max_relative_change": maximum_far_field_change,
        "aligned_profile_window_x_over_lambda_1": list(CHECKPOINT_PROFILE_WINDOW),
        "aligned_profile_changes": profile_changes,
        "aligned_profile_max_rms_relative_change": maximum_profile_rms_change,
        "aligned_profile_maximum_relative_change": maximum_profile_change,
        "limits": {
            "shock_center_change_over_lambda_1": CHECKPOINT_SHOCK_CENTER_CHANGE_LIMIT,
            "density_10_90_thickness_relative_change": CHECKPOINT_THICKNESS_RELATIVE_CHANGE_LIMIT,
            "far_field_max_relative_change": CHECKPOINT_FAR_FIELD_RELATIVE_CHANGE_LIMIT,
            "aligned_profile_max_rms_relative_change": CHECKPOINT_PROFILE_RMS_RELATIVE_CHANGE_LIMIT,
            "aligned_profile_maximum_relative_change": CHECKPOINT_PROFILE_MAX_RELATIVE_CHANGE_LIMIT,
        },
        "pass": passed,
    }


def dump_step(path: Path) -> int:
    try:
        return int(path.name.rsplit(".", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Dump filename has no numeric timestep suffix: {path}") from error


def process_single(run_dir: Path, dump: Path | None = None) -> dict[str, object]:
    metadata_path = run_dir / "case_metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates = sorted(run_dir.glob("profile.final.*"), key=dump_step)
    if dump is None:
        if not candidates:
            raise FileNotFoundError(f"No profile.final.* dump in {run_dir}")
        dump = candidates[-1]
    else:
        dump = dump.resolve()
    rows = read_last_grid_snapshot(dump)
    normalized, metrics = normalize_rows(rows, meta)
    spatial_pass = bool(metrics["physics_gate_pass"])
    production = meta.get("level") == "production"
    previous_candidates = [path for path in candidates if dump_step(path) < dump_step(dump)]
    if production and previous_candidates:
        previous_dump = previous_candidates[-1]
        previous_normalized, previous_metrics = normalize_rows(
            read_last_grid_snapshot(previous_dump), meta
        )
        checkpoint = checkpoint_stability_metrics(
            metrics, previous_metrics, normalized, previous_normalized
        )
        checkpoint["previous_dump"] = previous_dump.name
        metrics["physics_gate_pass"] = spatial_pass and bool(checkpoint["pass"])
    elif production:
        checkpoint = {
            "required": True,
            "available": False,
            "pass": False,
            "reason": "No preceding cumulative checkpoint dump was found",
        }
        metrics["physics_gate_pass"] = False
    else:
        checkpoint = {
            "required": False,
            "available": False,
            "pass": True,
            "reason": "Checkpoint stability is enforced for production cases only",
        }
    metrics["spatial_physics_gate_pass"] = spatial_pass
    metrics["checkpoint_stability"] = checkpoint
    write_csv(run_dir / "profile_normalized.csv", normalized)
    metrics.update({
        "source_dump": dump.name,
        "mach_1": meta["mach_1"],
        "seed": meta["seed"],
        "grid_cell_count": len(rows),
    })
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_single(run_dir / "shock_profile.png", normalized, float(meta["mach_1"]))
    return metrics


def interpolate(rows: list[dict[str, float]], xnew: float, key: str) -> float:
    x = [row["x_over_lambda_1"] for row in rows]
    if xnew < x[0] or xnew > x[-1]:
        raise ValueError("Interpolation point outside profile overlap")
    lo, hi = 0, len(x) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= xnew:
            lo = mid
        else:
            hi = mid
    if x[hi] == x[lo]:
        return rows[lo][key]
    weight = (xnew - x[lo]) / (x[hi] - x[lo])
    return rows[lo][key] * (1.0 - weight) + rows[hi][key] * weight


def plot_ensemble(path: Path, rows: list[dict[str, float]], mach: float) -> None:
    cache = Path(tempfile.gettempdir()) / "sparta-normal-shock-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = [row["x_over_lambda_1"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    series = (
        ("number_density_over_n1", r"$n/n_1$", "#0b4f9c"),
        ("u_over_u1", r"$u/u_1$", "#1d8a99"),
        ("temperature_over_T1", r"$T/T_1$", "#d73027"),
        ("qx_over_n1kT1u1", r"$q_x/(n_1kT_1u_1)$", "#7b3294"),
    )
    for axis, (key, label, color) in zip(axes.flat, series):
        mean = [row[f"{key}_mean"] for row in rows]
        half_width = [row[f"{key}_ci95"] for row in rows]
        axis.plot(x, mean, color=color, linewidth=1.5)
        axis.fill_between(
            x,
            [value - width for value, width in zip(mean, half_width)],
            [value + width for value, width in zip(mean, half_width)],
            color=color,
            alpha=0.20,
            linewidth=0.0,
        )
        axis.set(xlabel=r"$(x-x_s)/\lambda_1$", ylabel=label)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"SPARTA steady normal shock, M1={mach:g} "
        "(3-seed mean and pointwise 95% Student-t CI)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def process_ensemble(output: Path, run_dirs: list[Path]) -> dict[str, object]:
    if len(run_dirs) != 3:
        raise ValueError("The production ensemble requires exactly three realizations")
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError("Ensemble run directories must be distinct")
    profiles = [read_csv(path / "profile_normalized.csv") for path in run_dirs]
    metadata = [json.loads((path / "case_metadata.json").read_text(encoding="utf-8")) for path in run_dirs]
    validation = [
        json.loads((path / "validation_metrics.json").read_text(encoding="utf-8"))
        for path in run_dirs
    ]
    rejected = [str(path) for path, result in zip(run_dirs, validation) if not result.get("physics_gate_pass")]
    if rejected:
        raise ValueError(f"Ensemble contains realizations that failed validation: {rejected}")
    machs = {round(float(meta["mach_1"]), 12) for meta in metadata}
    if len(machs) != 1:
        raise ValueError(f"Mixed Mach numbers in one ensemble: {sorted(machs)}")
    seeds = [int(meta["seed"]) for meta in metadata]
    if len(set(seeds)) != 3:
        raise ValueError(f"Production seeds must be distinct: {seeds}")
    lower = max(profile[0]["x_over_lambda_1"] for profile in profiles)
    upper = min(profile[-1]["x_over_lambda_1"] for profile in profiles)
    spacing = max(float(meta["dx_over_lambda_1"]) for meta in metadata)
    count = math.floor((upper - lower) / spacing) + 1
    if count < 2:
        raise ValueError("Aligned profiles have no usable common support")
    grid = [lower + index * spacing for index in range(count)]
    critical_value = student_t_critical_95(len(profiles))
    value_keys = NORMALIZED_NAMES[1:]
    names = ["x_over_lambda_1"]
    for key in value_keys:
        names.extend((f"{key}_mean", f"{key}_stderr", f"{key}_ci95"))
    rows: list[dict[str, float]] = []
    for xnew in grid:
        row = {"x_over_lambda_1": xnew}
        for key in value_keys:
            values = [interpolate(profile, xnew, key) for profile in profiles]
            standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{key}_mean"] = fmean(values)
            row[f"{key}_stderr"] = standard_error
            row[f"{key}_ci95"] = critical_value * standard_error
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ensemble_profile.csv", rows, tuple(names))
    summary: dict[str, object] = {
        "schema_version": 2,
        "mach_1": next(iter(machs)),
        "realization_count": len(profiles),
        "seeds": seeds,
        "alignment": "each realization translated to its unsmoothed density midpoint",
        "spatial_smoothing": "none",
        "common_grid_spacing_over_lambda_1": spacing,
        "common_grid_point_count": count,
        "uncertainty": {
            "confidence_level": 0.95,
            "interval_type": "pointwise Student-t confidence interval, not a simultaneous band",
            "distribution_assumption": "independent approximately normal seed realizations",
            "degrees_of_freedom": len(profiles) - 1,
            "critical_value": critical_value,
            "csv_ci95_columns_are_half_widths": True,
            "evaluated_after_density_midpoint_alignment": True,
        },
        "run_directories": [str(path) for path in run_dirs],
    }
    (output / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_ensemble(output / "ensemble_profile.png", rows, float(next(iter(machs))))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("run_dir", type=Path)
    single.add_argument("--dump", type=Path)
    ensemble = subparsers.add_parser("ensemble")
    ensemble.add_argument("--output", type=Path, required=True)
    ensemble.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "single":
        result = process_single(args.run_dir.resolve(), args.dump)
    else:
        result = process_ensemble(args.output.resolve(), [path.resolve() for path in args.run_dirs])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
