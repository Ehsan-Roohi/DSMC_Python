#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ("ux", "uy", "T", "rho")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    parser.add_argument("--route", required=True)
    return parser.parse_args()


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "model", "kn", "rt", "seed", "figure"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Unexpected case-table columns in {path}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case_id in case table")
    return rows


def output_stem(row: dict[str, str]) -> str:
    kn = f"{float(row['kn']):g}"
    rt = f"{float(row['rt']):g}".replace(".", "p")
    seed = int(row["seed"])
    if row["model"] == "HS":
        return f"ThermalCavity_HS_DSMC_Kn{kn}_RT{rt}_quarter_seed{seed}"
    return f"ThermalCavity_{row['model']}_Kn{kn}_RT{rt}_quarter_seed{seed}"


def weighted_stats(stack: np.ndarray, weights: np.ndarray):
    weights = np.asarray(weights, dtype=np.float64)
    normalized = weights / weights.sum()
    mean = np.tensordot(normalized, stack, axes=(0, 0))
    sum_w = weights.sum()
    denominator = sum_w - np.dot(weights, weights) / sum_w
    if denominator <= 0.0:
        raise ValueError("At least two positive realization weights are required")
    variance = np.tensordot(
        weights, (stack - mean) ** 2, axes=(0, 0)
    ) / denominator
    n_effective = sum_w * sum_w / np.dot(weights, weights)
    sd = np.sqrt(np.maximum(variance, 0.0))
    se = sd / math.sqrt(n_effective)
    return mean, sd, se, float(n_effective)


def interpolate_row(y: np.ndarray, field: np.ndarray, target=0.25) -> np.ndarray:
    return np.array([np.interp(target, y, field[:, i]) for i in range(field.shape[1])])


def write_dat(path: Path, x, y, means, sds, ses) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write('TITLE="JFM high-statistics raw ensemble mean"\n')
        handle.write(
            'VARIABLES="x","y","u_x","u_y","T","rho","Umag",'
            '"ux_sd","uy_sd","ux_se","uy_se"\n'
        )
        handle.write(f"ZONE I={len(x)}, J={len(y)}, F=POINT\n")
        for j, y_value in enumerate(y):
            for i, x_value in enumerate(x):
                ux = means["ux"][j, i]
                uy = means["uy"][j, i]
                values = (
                    x_value, y_value, ux, uy, means["T"][j, i],
                    means["rho"][j, i], math.hypot(ux, uy),
                    sds["ux"][j, i], sds["uy"][j, i],
                    ses["ux"][j, i], ses["uy"][j, i],
                )
                handle.write(" ".join(f"{value:.10e}" for value in values) + "\n")


def write_profile(path: Path, x, y, means, sds, ses) -> None:
    rows = {}
    for field in ("ux", "uy", "T"):
        rows[f"{field}_mean"] = interpolate_row(y, means[field])
        rows[f"{field}_sd"] = interpolate_row(y, sds[field])
        rows[f"{field}_se"] = interpolate_row(y, ses[field])
    names = ["x_over_L", *rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for i, x_value in enumerate(x):
            record = {"x_over_L": x_value}
            record.update({name: values[i] for name, values in rows.items()})
            writer.writerow(record)


def diagnostic_plot(path: Path, x, y, means, ses, title: str) -> None:
    xx, yy = np.meshgrid(x, y)
    ux_row = interpolate_row(y, means["ux"])
    uy_row = interpolate_row(y, means["uy"])
    ux_se = interpolate_row(y, ses["ux"])
    uy_se = interpolate_row(y, ses["uy"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.1), constrained_layout=True)
    axes[0].plot(x, ux_row, color="tab:blue")
    axes[0].fill_between(x, ux_row - 2 * ux_se, ux_row + 2 * ux_se,
                         color="tab:blue", alpha=0.2)
    axes[0].set_title(r"$u_x$ at $y/L=0.25$")
    axes[1].plot(x, uy_row, color="tab:red")
    axes[1].fill_between(x, uy_row - 2 * uy_se, uy_row + 2 * uy_se,
                         color="tab:red", alpha=0.2)
    axes[1].set_title(r"$u_y$ at $y/L=0.25$")
    contour = axes[2].contourf(xx, yy, means["T"], levels=30, cmap="viridis")
    axes[2].streamplot(x, y, means["ux"], means["uy"], color="k",
                       density=1.15, linewidth=0.7, arrowsize=0.7)
    axes[2].set_title("Raw mean field and streamlines")
    fig.colorbar(contour, ax=axes[2], label=r"$T/T_h$")
    for axis in axes:
        axis.set_xlabel(r"$x/L$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("nondimensional velocity")
    axes[2].set_ylabel(r"$y/L$")
    axes[2].set_aspect("equal")
    fig.suptitle(title + "\nRaw ensemble mean; bands are +/-2 realization SE")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_group(key, rows, input_dir: Path, output_dir: Path, route: str):
    model, kn, rt, figure = key
    entries = []
    for row in rows:
        stem = output_stem(row)
        npz_path = input_dir / f"{stem}_raw.npz"
        metrics_path = input_dir / f"{stem}_metrics.json"
        if not npz_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Missing outputs for {stem}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as data:
            entry = {
                "seed": int(row["seed"]),
                "x": np.asarray(data["x"], dtype=np.float64),
                "y": np.asarray(data["y"], dtype=np.float64),
                **{field: np.asarray(data[field], dtype=np.float64) for field in FIELDS},
                "metrics": metrics,
            }
        entries.append(entry)

    x = entries[0]["x"]
    y = entries[0]["y"]
    for entry in entries:
        if not np.array_equal(entry["x"], x) or not np.array_equal(entry["y"], y):
            raise ValueError(f"Grid mismatch in {key}")
    weights = np.array([
        entry["metrics"]["particles"] * entry["metrics"]["profile_samples"]
        for entry in entries
    ], dtype=np.float64)
    stacks = {field: np.stack([entry[field] for entry in entries]) for field in FIELDS}
    if any(not np.all(np.isfinite(stack)) for stack in stacks.values()):
        raise ValueError(f"Non-finite field in {key}")

    means, sds, ses = {}, {}, {}
    n_effective = None
    for field, stack in stacks.items():
        means[field], sds[field], ses[field], n_eff = weighted_stats(stack, weights)
        n_effective = n_eff if n_effective is None else n_effective

    tag = f"{figure}_{model}_Kn{kn:g}_RT{str(rt).replace('.', 'p')}_{route}"
    np.savez_compressed(
        output_dir / f"{tag}_RAW_UNFILTERED.npz",
        x=x, y=y, seeds=np.array([entry["seed"] for entry in entries]),
        particle_time_weights=weights,
        **{f"{field}_runs": stack for field, stack in stacks.items()},
        **{f"{field}_mean": means[field] for field in FIELDS},
        **{f"{field}_sample_sd": sds[field] for field in FIELDS},
        **{f"{field}_standard_error": ses[field] for field in FIELDS},
    )
    write_dat(output_dir / f"{tag}_RAW_UNFILTERED.dat", x, y, means, sds, ses)
    write_profile(output_dir / f"{tag}_profile_y0p25.csv", x, y, means, sds, ses)
    diagnostic_plot(output_dir / f"{tag}_diagnostic.png", x, y, means, ses,
                    f"{figure}: {model}, Kn={kn:g}, RT={rt:g}, route={route}")

    velocity_rms = float(np.sqrt(np.mean(means["ux"] ** 2 + means["uy"] ** 2)))
    velocity_se_rms = float(np.sqrt(np.mean(ses["ux"] ** 2 + ses["uy"] ** 2)))
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    ek_runs = np.sum(
        stacks["rho"] * (stacks["ux"] ** 2 + stacks["uy"] ** 2), axis=(1, 2)
    ) * dx * dy
    summary = {
        "route": route,
        "figure": figure,
        "model": model,
        "Kn_paper": kn,
        "RT": rt,
        "seeds": [entry["seed"] for entry in entries],
        "number_of_independent_runs": len(entries),
        "particles_per_run": [entry["metrics"]["particles"] for entry in entries],
        "steps_per_run": [entry["metrics"]["steps"] for entry in entries],
        "samples_per_run": [entry["metrics"]["profile_samples"] for entry in entries],
        "effective_realization_count": n_effective,
        "velocity_mean_rms": velocity_rms,
        "velocity_standard_error_rms": velocity_se_rms,
        "velocity_signal_to_standard_error_ratio": (
            velocity_rms / velocity_se_rms if velocity_se_rms else None
        ),
        "kinetic_energy_per_run": ek_runs.tolist(),
        "last_block_velocity_rmse_vs_all_samples": [
            entry["metrics"]["last_block_velocity_rmse_vs_all_samples"]
            for entry in entries
        ],
        "quantitative_fields_are_unfiltered": True,
        "spatial_smoothing_applied": False,
        "velocity_projection_applied": False,
    }
    (output_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_cases(args.case_table)
    groups = defaultdict(list)
    for row in rows:
        key = (row["model"], float(row["kn"]), float(row["rt"]), row["figure"])
        groups[key].append(row)
    if len(groups) != 5:
        raise ValueError(f"Expected five physical cases, found {len(groups)}")

    summaries = []
    for key in sorted(groups, key=lambda value: (value[3], value[0], value[1], value[2])):
        summaries.append(summarize_group(
            key, groups[key], args.input, args.output, args.route
        ))
        print(f"[OK] summarized {key}")

    (args.output / "ALL_HIGHSTAT_ENSEMBLES.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "route", "figure", "model", "Kn_paper", "RT",
        "number_of_independent_runs", "effective_realization_count",
        "velocity_mean_rms", "velocity_standard_error_rms",
        "velocity_signal_to_standard_error_ratio",
    )
    with (args.output / "ALL_HIGHSTAT_ENSEMBLES.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})
    print(f"[OK] wrote five high-statistics ensembles to {args.output}")


if __name__ == "__main__":
    main()
