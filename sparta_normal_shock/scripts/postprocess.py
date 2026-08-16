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


def crossing_x(x: list[float], y: list[float], target: float, prefer: float = 0.0) -> float:
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
    return x[min(range(len(x)), key=lambda index: abs(y[index] - target))]


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-300)


def mean_rows(rows: list[dict[str, float]], indexes: list[int], key: str) -> float:
    if not indexes:
        raise ValueError(f"Empty averaging window for {key}")
    return fmean(rows[index][key] for index in indexes)


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
    shock_center = crossing_x(x, density_ratio, midpoint)
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
    x10 = crossing_x(xa, nd, 1.0 + 0.1 * (ratio - 1.0), prefer=-1.0)
    x90 = crossing_x(xa, nd, 1.0 + 0.9 * (ratio - 1.0), prefer=1.0)
    # Far-field windows are tied to the physical box, not to the aligned
    # coordinate.  This remains well-defined even for a deliberately short
    # smoke run whose shock has not settled near the box center.
    count = len(aligned)
    upstream = list(range(max(0, count // 20), max(1, count // 5)))
    downstream = list(range(max(0, 4 * count // 5), max(1, 19 * count // 20)))
    interior = list(range(max(0, count // 20), max(1, 19 * count // 20)))
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
        flux_errors.append(max_deviation)
        conservation[key] = {"mean": mean_value, "maximum_relative_deviation": max_deviation}
    metrics: dict[str, object] = {
        "alignment_method": "unsmoothed density-midpoint translation",
        "smoothing": "none",
        "shock_center_m": shock_center,
        "shock_center_over_lambda_1_before_alignment": shock_center / lambda1,
        "density_10_90_thickness_over_lambda_1": abs(x90 - x10),
        "far_field": far_field,
        "conservation": conservation,
        "gates": {
            "far_field_max_relative_error_limit": 0.10,
            "flux_maximum_relative_deviation_limit": 0.20,
            "far_field_max_relative_error": max(far_errors),
            "flux_maximum_relative_deviation": max(flux_errors),
        },
    }
    metrics["physics_gate_pass"] = max(far_errors) <= 0.10 and max(flux_errors) <= 0.20
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


def process_single(run_dir: Path, dump: Path | None = None) -> dict[str, object]:
    metadata_path = run_dir / "case_metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if dump is None:
        candidates = sorted(run_dir.glob("profile.final.*"))
        if not candidates:
            raise FileNotFoundError(f"No profile.final.* dump in {run_dir}")
        dump = candidates[-1]
    rows = read_last_grid_snapshot(dump)
    normalized, metrics = normalize_rows(rows, meta)
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


def process_ensemble(output: Path, run_dirs: list[Path]) -> dict[str, object]:
    if len(run_dirs) < 2:
        raise ValueError("An ensemble requires at least two completed realizations")
    profiles = [read_csv(path / "profile_normalized.csv") for path in run_dirs]
    metadata = [json.loads((path / "case_metadata.json").read_text(encoding="utf-8")) for path in run_dirs]
    machs = {round(float(meta["mach_1"]), 12) for meta in metadata}
    if len(machs) != 1:
        raise ValueError(f"Mixed Mach numbers in one ensemble: {sorted(machs)}")
    lower = max(profile[0]["x_over_lambda_1"] for profile in profiles)
    upper = min(profile[-1]["x_over_lambda_1"] for profile in profiles)
    count = 481
    grid = [lower + index * (upper - lower) / (count - 1) for index in range(count)]
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
            row[f"{key}_ci95"] = 1.96 * standard_error
        rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ensemble_profile.csv", rows, tuple(names))
    summary: dict[str, object] = {
        "schema_version": 1,
        "mach_1": next(iter(machs)),
        "realization_count": len(profiles),
        "seeds": [meta["seed"] for meta in metadata],
        "alignment": "each realization translated to its unsmoothed density midpoint",
        "spatial_smoothing": "none",
        "uncertainty": "pointwise 95% confidence interval from independent seeds",
        "run_directories": [str(path) for path in run_dirs],
    }
    (output / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
