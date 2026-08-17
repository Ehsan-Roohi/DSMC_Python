#!/usr/bin/env python3
"""Build the Kn=0.20 SPARTA ensemble mean, 95% CI, and quick-look figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


FIELDS = ("nrho", "u", "v", "w", "T", "qx", "qy")
FIX_COLUMNS = tuple(f"f_fieldavg[{index}]" for index in range(1, 8))
T95 = {
    2: 12.706204736,
    3: 4.302652730,
    4: 3.182446305,
    5: 2.776445105,
    6: 2.570581836,
    7: 2.446911851,
    8: 2.364624252,
}


def latest_dump(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("grid.final.*"))
    if not candidates:
        raise FileNotFoundError(f"No grid.final.* dump in {run_dir}")

    def step(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def read_last_snapshot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    positions = [i for i, line in enumerate(lines) if line.startswith("ITEM: CELLS")]
    if not positions:
        raise ValueError(f"No ITEM: CELLS block in {path}")
    start = positions[-1]
    columns = lines[start].split()[2:]
    rows: list[list[float]] = []
    for line in lines[start + 1 :]:
        if line.startswith("ITEM:"):
            break
        if line.strip():
            rows.append([float(value) for value in line.split()])
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"Malformed grid block in {path}")
    return columns, data


def load_member(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "case_metadata.json").read_text(encoding="utf-8"))
    columns, data = read_last_snapshot(latest_dump(run_dir))
    index = {name: i for i, name in enumerate(columns)}
    required = ("id", "xc", "yc", *FIX_COLUMNS)
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Missing dump columns {missing} in {run_dir}; got {columns}")
    order = np.argsort(data[:, index["id"]])
    data = data[order]
    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "id": data[:, index["id"]].astype(np.int64),
        "x": data[:, index["xc"]],
        "y": data[:, index["yc"]],
        "fields": np.column_stack([data[:, index[name]] for name in FIX_COLUMNS]),
    }


def ci95(samples: np.ndarray) -> np.ndarray:
    count = samples.shape[0]
    if count < 2:
        return np.full(samples.shape[1:], np.nan)
    if count not in T95:
        raise ValueError(f"No audited Student-t factor for {count} seeds")
    return T95[count] * np.std(samples, axis=0, ddof=1) / math.sqrt(count)


def write_field_csv(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    length: float,
    mean: np.ndarray,
    ci: np.ndarray,
) -> None:
    header = ["x_over_L", "y_over_L"]
    for field in FIELDS:
        header.extend((f"{field}_mean", f"{field}_ci95"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for cell in range(mean.shape[0]):
            row: list[float] = [x[cell] / length, y[cell] / length]
            for column in range(mean.shape[1]):
                row.extend((mean[cell, column], ci[cell, column]))
            writer.writerow(row)


def write_centerlines(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    length: float,
    mean: np.ndarray,
    ci: np.ndarray,
) -> None:
    x_nd = x / length
    y_nd = y / length
    x_mid = np.unique(x_nd)[np.argmin(np.abs(np.unique(x_nd) - 0.5))]
    y_mid = np.unique(y_nd)[np.argmin(np.abs(np.unique(y_nd) - 0.5))]
    line_specs = (
        ("horizontal_y_mid", np.isclose(y_nd, y_mid), x_nd),
        ("vertical_x_mid", np.isclose(x_nd, x_mid), y_nd),
    )
    header = ["line", "s_over_L", "x_over_L", "y_over_L"]
    for field in FIELDS:
        header.extend((f"{field}_mean", f"{field}_ci95"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for label, mask, coordinate in line_specs:
            indices = np.flatnonzero(mask)
            indices = indices[np.argsort(coordinate[indices])]
            for cell in indices:
                row: list[Any] = [label, coordinate[cell], x_nd[cell], y_nd[cell]]
                for column in range(mean.shape[1]):
                    row.extend((mean[cell, column], ci[cell, column]))
                writer.writerow(row)


def make_quicklook(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    length: float,
    mean: np.ndarray,
    ci: np.ndarray,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; CSV and NPZ products were still created"

    x_nd = x / length
    y_nd = y / length
    xs = np.unique(x_nd)
    ys = np.unique(y_nd)
    order = np.lexsort((x_nd, y_nd))
    shape = (len(ys), len(xs))
    grids = [mean[order, i].reshape(shape) for i in range(len(FIELDS))]
    ci_grids = [ci[order, i].reshape(shape) for i in range(len(FIELDS))]
    _, u, v, _, temperature, qx, qy = grids

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), constrained_layout=True)
    panel = axes[0, 0].pcolormesh(xs, ys, temperature - 300.0, shading="auto", cmap="coolwarm")
    axes[0, 0].streamplot(xs, ys, u, v, color="k", density=1.05, linewidth=0.55)
    fig.colorbar(panel, ax=axes[0, 0], label=r"$T-300$ [K]")
    axes[0, 0].set_title("Mean temperature and velocity topology")

    qmag = np.hypot(qx, qy)
    panel = axes[0, 1].pcolormesh(xs, ys, qmag, shading="auto", cmap="viridis")
    skip = (slice(None, None, 8), slice(None, None, 8))
    axes[0, 1].quiver(
        xs[::8], ys[::8], qx[skip], qy[skip], color="white", pivot="mid", scale=None
    )
    fig.colorbar(panel, ax=axes[0, 1], label=r"$|q|$ [SPARTA SI]")
    axes[0, 1].set_title("Mean DSMC heat-flux field")

    x_index = int(np.argmin(np.abs(xs - 0.5)))
    y_index = int(np.argmin(np.abs(ys - 0.5)))
    axes[1, 0].plot(ys, qx[:, x_index], label=r"$q_x$")
    axes[1, 0].fill_between(
        ys,
        qx[:, x_index] - ci_grids[5][:, x_index],
        qx[:, x_index] + ci_grids[5][:, x_index],
        alpha=0.22,
    )
    axes[1, 0].plot(ys, qy[:, x_index], label=r"$q_y$")
    axes[1, 0].fill_between(
        ys,
        qy[:, x_index] - ci_grids[6][:, x_index],
        qy[:, x_index] + ci_grids[6][:, x_index],
        alpha=0.22,
    )
    axes[1, 0].set_title(r"Vertical centerline, $x/L\approx0.5$")
    axes[1, 0].set_xlabel(r"$y/L$")
    axes[1, 0].set_ylabel("Heat-flux density [SPARTA SI]")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(xs, qx[y_index, :], label=r"$q_x$")
    axes[1, 1].fill_between(
        xs,
        qx[y_index, :] - ci_grids[5][y_index, :],
        qx[y_index, :] + ci_grids[5][y_index, :],
        alpha=0.22,
    )
    axes[1, 1].plot(xs, qy[y_index, :], label=r"$q_y$")
    axes[1, 1].fill_between(
        xs,
        qy[y_index, :] - ci_grids[6][y_index, :],
        qy[y_index, :] + ci_grids[6][y_index, :],
        alpha=0.22,
    )
    axes[1, 1].set_title(r"Horizontal centerline, $y/L\approx0.5$")
    axes[1, 1].set_xlabel(r"$x/L$")
    axes[1, 1].legend(frameon=False)

    for panel_index, axis in enumerate(axes.flat):
        axis.set_xlim(0.0, 1.0)
        if panel_index < 2:
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel(r"$x/L$")
            axis.set_ylabel(r"$y/L$")
        axis.grid(alpha=0.15)

    fig.suptitle("SPARTA DSMC, Kn=0.20: 8-seed ensemble mean and 95% CI")
    fig.savefig(output / "kn020_dsmc_quicklook.png", dpi=240)
    fig.savefig(output / "kn020_dsmc_quicklook.pdf")
    plt.close(fig)
    return "created kn020_dsmc_quicklook.png and .pdf"


def process(results_root: Path, output: Path, expected_seeds: list[int]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    members: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for seed in expected_seeds:
        run_dir = results_root / f"seed_{seed}"
        try:
            if not (run_dir / "unity_run_metadata.txt").is_file():
                raise FileNotFoundError("unity_run_metadata.txt missing")
            status = (run_dir / "unity_run_metadata.txt").read_text(encoding="utf-8")
            if "status=complete" not in status.splitlines():
                raise ValueError("run is not marked complete")
            members.append(load_member(run_dir))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            errors[str(seed)] = str(exc)

    if not members:
        raise RuntimeError("No complete SPARTA Kn=0.20 member could be read")

    contract_keys = (
        "kn",
        "kn_convention",
        "nx",
        "ny",
        "particles_per_cell",
        "sample_steps",
        "sample_stride",
        "accumulated_samples_per_cell",
        "argon_mass_kg",
    )
    reference = members[0]
    for member in members[1:]:
        for key in contract_keys:
            if member["metadata"].get(key) != reference["metadata"].get(key):
                raise ValueError(f"Ensemble contract mismatch for {key}")
        if not np.array_equal(member["id"], reference["id"]):
            raise ValueError("Grid cell IDs differ across ensemble members")
        if not np.allclose(member["x"], reference["x"], rtol=0.0, atol=1.0e-18):
            raise ValueError("Grid x coordinates differ across ensemble members")
        if not np.allclose(member["y"], reference["y"], rtol=0.0, atol=1.0e-18):
            raise ValueError("Grid y coordinates differ across ensemble members")

    samples = np.stack([member["fields"] for member in members], axis=0)
    mean = np.mean(samples, axis=0)
    interval = ci95(samples)
    x = reference["x"]
    y = reference["y"]
    metadata = reference["metadata"]
    length = float(metadata["length_m"])
    finite = bool(np.isfinite(samples).all())
    positive_density = bool(np.min(samples[:, :, 0]) > 0.0)
    positive_temperature = bool(np.min(samples[:, :, 4]) > 0.0)

    write_field_csv(output / "ensemble_mean_fields.csv", x, y, length, mean, interval)
    write_centerlines(output / "ensemble_centerlines.csv", x, y, length, mean, interval)
    payload: dict[str, Any] = {
        "seed": np.asarray([member["metadata"]["seed"] for member in members], dtype=np.int64),
        "cell_id": reference["id"],
        "x_over_L": x / length,
        "y_over_L": y / length,
    }
    for column, field in enumerate(FIELDS):
        payload[f"{field}_samples"] = samples[:, :, column]
        payload[f"{field}_mean"] = mean[:, column]
        payload[f"{field}_ci95"] = interval[:, column]
    np.savez_compressed(output / "ensemble_fields.npz", **payload)

    plot_note = make_quicklook(output, x, y, length, mean, interval)
    complete = len(members) == len(expected_seeds)
    summary: dict[str, Any] = {
        "status": "complete" if complete and finite and positive_density and positive_temperature else "incomplete",
        "expected_seeds": expected_seeds,
        "complete_seeds": [int(member["metadata"]["seed"]) for member in members],
        "member_errors": errors,
        "ensemble_size": len(members),
        "confidence_interval": "two-sided 95% Student-t interval across independent seeds",
        "kn": metadata["kn"],
        "kn_convention": metadata["kn_convention"],
        "grid": [metadata["nx"], metadata["ny"]],
        "particles_per_cell": metadata["particles_per_cell"],
        "accumulated_samples_per_cell": metadata["accumulated_samples_per_cell"],
        "dump_columns": list(FIELDS),
        "sanity": {
            "all_values_finite": finite,
            "density_positive": positive_density,
            "temperature_positive": positive_temperature,
            "rho_min": float(np.min(samples[:, :, 0])),
            "temperature_min_K": float(np.min(samples[:, :, 4])),
        },
        "products": {
            "fields_npz": "ensemble_fields.npz",
            "mean_and_ci_csv": "ensemble_mean_fields.csv",
            "centerlines_csv": "ensemble_centerlines.csv",
            "quicklook": plot_note,
        },
        "evidence_boundary": (
            "Raw DSMC ensemble product; R13/R26 comparison, common-mask analysis, "
            "and final article styling must be performed in the reviewer pipeline."
        ),
    }
    (output / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    summary = process(args.results_root.resolve(), args.output.resolve(), args.seeds)
    return 0 if summary["status"] == "complete" else 6


if __name__ == "__main__":
    raise SystemExit(main())
