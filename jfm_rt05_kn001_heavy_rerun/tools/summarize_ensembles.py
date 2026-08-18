#!/usr/bin/env python3
"""
Create raw three-seed ensemble fields, uncertainty diagnostics, profiles and Ek.

No spatial smoothing, interpolation of two-dimensional fields, or velocity
projection is performed. The only interpolation is the explicitly reported
one-dimensional extraction at y/L=0.25.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


T95_DF2 = 4.302652729911275
NAME_RE = re.compile(
    r"ThermalCavity_(HS_DSMC|BGK|SHAKHOV)_Kn([^_]+)_RT([^_]+)"
    r"_quarter_seed(\d+)_raw\.npz$"
)
EXPECTED_SEEDS = (42, 271828, 314159)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--case-table", action="append", default=[], type=Path,
        help="CSV table to validate; may be supplied more than once",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def model_name(token: str) -> str:
    return "HS" if token == "HS_DSMC" else token


def load_expected(paths: list[Path]) -> set[tuple[str, float, float, int]]:
    expected: set[tuple[str, float, float, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                expected.add(
                    (
                        row["model"],
                        float(row["kn"]),
                        float(row["rt"]),
                        int(row["seed"]),
                    )
                )
    return expected


def find_runs(root: Path) -> dict[tuple[str, float, float], list[dict]]:
    groups: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("*_raw.npz")):
        if "smoke" in path.parts:
            continue
        match = NAME_RE.match(path.name)
        if not match:
            continue
        token, kn_text, rt_tag, seed_text = match.groups()
        model = model_name(token)
        kn = float(kn_text)
        rt = float(rt_tag.replace("p", "."))
        seed = int(seed_text)
        metrics_path = Path(str(path).replace("_raw.npz", "_metrics.json"))
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics for {path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        groups[(model, kn, rt)].append(
            {
                "path": path,
                "metrics_path": metrics_path,
                "metrics": metrics,
                "seed": seed,
            }
        )
    return groups


def write_dat(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    mean: dict[str, np.ndarray],
    sd: dict[str, np.ndarray],
    se: dict[str, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write('TITLE="Raw three-seed mean; no spatial filtering"\n')
        handle.write(
            'VARIABLES="x","y","ux_mean","uy_mean","T_mean","rho_mean",'
            '"Umag_mean","ux_sd","uy_sd","T_sd","rho_sd","ux_se","uy_se"\n'
        )
        handle.write(f"ZONE I={len(x)}, J={len(y)}, F=POINT\n")
        for j, y_value in enumerate(y):
            for i, x_value in enumerate(x):
                speed = math.hypot(mean["ux"][j, i], mean["uy"][j, i])
                values = (
                    x_value,
                    y_value,
                    mean["ux"][j, i],
                    mean["uy"][j, i],
                    mean["T"][j, i],
                    mean["rho"][j, i],
                    speed,
                    sd["ux"][j, i],
                    sd["uy"][j, i],
                    sd["T"][j, i],
                    sd["rho"][j, i],
                    se["ux"][j, i],
                    se["uy"][j, i],
                )
                handle.write(" ".join(f"{v:.10e}" for v in values) + "\n")


def parity_diagnostics(ux: np.ndarray, uy: np.ndarray, scalar: np.ndarray) -> dict:
    return {
        "scalar_x_even_max_abs": float(np.max(np.abs(scalar - scalar[:, ::-1]))),
        "scalar_y_even_max_abs": float(np.max(np.abs(scalar - scalar[::-1, :]))),
        "ux_x_odd_max_abs": float(np.max(np.abs(ux + ux[:, ::-1]))),
        "ux_y_even_max_abs": float(np.max(np.abs(ux - ux[::-1, :]))),
        "uy_x_even_max_abs": float(np.max(np.abs(uy - uy[:, ::-1]))),
        "uy_y_odd_max_abs": float(np.max(np.abs(uy + uy[::-1, :]))),
    }


def interpolate_row(
    y: np.ndarray, field: np.ndarray, target: float
) -> np.ndarray:
    return np.array(
        [np.interp(target, y, field[:, i]) for i in range(field.shape[1])]
    )


def write_profile(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    arrays: dict[str, np.ndarray],
    seeds: list[int],
) -> None:
    ux = arrays["ux"]
    uy = arrays["uy"]
    temperature = arrays["T"]
    target = 0.25
    ux_rows = np.stack([interpolate_row(y, item, target) for item in ux])
    uy_rows = np.stack([interpolate_row(y, item, target) for item in uy])
    t_rows = np.stack(
        [interpolate_row(y, item, target) for item in temperature]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["x_over_L"]
        for seed in seeds:
            fields.extend((f"ux_seed{seed}", f"uy_seed{seed}", f"T_seed{seed}"))
        fields.extend(
            (
                "ux_mean", "ux_sample_sd", "ux_standard_error",
                "uy_mean", "uy_sample_sd", "uy_standard_error",
                "T_mean", "T_sample_sd", "T_standard_error",
            )
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i, x_value in enumerate(x):
            row: dict[str, float] = {"x_over_L": x_value}
            for k, seed in enumerate(seeds):
                row[f"ux_seed{seed}"] = ux_rows[k, i]
                row[f"uy_seed{seed}"] = uy_rows[k, i]
                row[f"T_seed{seed}"] = t_rows[k, i]
            for label, values in (
                ("ux", ux_rows[:, i]),
                ("uy", uy_rows[:, i]),
                ("T", t_rows[:, i]),
            ):
                row[f"{label}_mean"] = float(np.mean(values))
                row[f"{label}_sample_sd"] = float(np.std(values, ddof=1))
                row[f"{label}_standard_error"] = float(
                    np.std(values, ddof=1) / math.sqrt(len(values))
                )
            writer.writerow(row)


def plot_fields(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    mean: dict[str, np.ndarray],
    title: str,
) -> None:
    xx, yy = np.meshgrid(x, y)
    speed = np.hypot(mean["ux"], mean["uy"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), constrained_layout=True)
    levels_t = np.linspace(float(mean["T"].min()), float(mean["T"].max()), 30)
    cf0 = axes[0].contourf(xx, yy, mean["T"], levels=levels_t, cmap="coolwarm")
    axes[0].streamplot(
        x, y, mean["ux"], mean["uy"], color="k", density=1.15,
        linewidth=0.8, arrowsize=0.8,
    )
    axes[0].set_title("Raw ensemble mean streamlines")
    fig.colorbar(cf0, ax=axes[0], label=r"$T/T_h$")

    speed_max = max(float(speed.max()), np.finfo(float).eps)
    cf1 = axes[1].contourf(
        xx, yy, speed, levels=np.linspace(0.0, speed_max, 30), cmap="viridis"
    )
    stride = 8
    axes[1].quiver(
        xx[::stride, ::stride], yy[::stride, ::stride],
        mean["ux"][::stride, ::stride], mean["uy"][::stride, ::stride],
        color="white", angles="xy", scale_units="xy",
    )
    axes[1].set_title("Raw ensemble mean vectors")
    fig.colorbar(cf1, ax=axes[1], label=r"$|\mathbf{u}|/U_{mp,h}$")
    for axis in axes:
        axis.set_xlabel(r"$x/L$")
        axis.set_ylabel(r"$y/L$")
        axis.set_aspect("equal")
    fig.suptitle(title + "\nNo spatial filtering or velocity projection")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_group(
    key: tuple[str, float, float],
    runs: list[dict],
    output: Path,
    make_plots: bool,
) -> dict:
    model, kn, rt = key
    runs = sorted(runs, key=lambda entry: entry["seed"])
    seeds = [entry["seed"] for entry in runs]
    if tuple(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"{key}: expected seeds {EXPECTED_SEEDS}, found {seeds}")

    loaded = [np.load(entry["path"], allow_pickle=False) for entry in runs]
    try:
        x = np.asarray(loaded[0]["x"], dtype=np.float64)
        y = np.asarray(loaded[0]["y"], dtype=np.float64)
        arrays = {
            field: np.stack(
                [np.asarray(item[field], dtype=np.float64) for item in loaded]
            )
            for field in ("ux", "uy", "T", "rho")
        }
    finally:
        for item in loaded:
            item.close()

    for field, stack in arrays.items():
        if not np.all(np.isfinite(stack)):
            raise ValueError(f"{key}: non-finite values in {field}")
        if stack.shape != (3, len(y), len(x)):
            raise ValueError(f"{key}: unexpected shape {stack.shape}")

    mean = {field: values.mean(axis=0) for field, values in arrays.items()}
    sd = {field: values.std(axis=0, ddof=1) for field, values in arrays.items()}
    se = {field: values.std(axis=0, ddof=1) / math.sqrt(3) for field, values in arrays.items()}
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    darea = dx * dy

    ek_seed = np.sum(
        arrays["rho"] * (arrays["ux"] ** 2 + arrays["uy"] ** 2),
        axis=(1, 2),
    ) * darea
    ek_from_mean = float(
        np.sum(mean["rho"] * (mean["ux"] ** 2 + mean["uy"] ** 2)) * darea
    )
    # Fully independent cross-seed estimator.  Each velocity product is
    # weighted by the density from the third seed, so the three factors in
    # every term have independent sampling noise.
    pair_indices = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    ek_cross_pairs = np.asarray(
        [
            np.sum(
                arrays["rho"][k]
                * (
                    arrays["ux"][i] * arrays["ux"][j]
                    + arrays["uy"][i] * arrays["uy"][j]
                )
            )
            * darea
            for i, j, k in pair_indices
        ],
        dtype=np.float64,
    )
    ek_cross_seed = float(np.mean(ek_cross_pairs))
    ek_pair_min = float(np.min(ek_cross_pairs))
    ek_pair_max = float(np.max(ek_cross_pairs))
    ek_pair_range_relative = float(
        (ek_pair_max - ek_pair_min) / abs(ek_cross_seed)
        if ek_cross_seed
        else math.inf
    )

    speed_se_rms = float(
        np.sqrt(np.mean(se["ux"] ** 2 + se["uy"] ** 2))
    )
    velocity_mean_rms = float(
        np.sqrt(np.mean(mean["ux"] ** 2 + mean["uy"] ** 2))
    )
    ek_sd = float(np.std(ek_seed, ddof=1))
    ek_se = ek_sd / math.sqrt(3)

    tag = f"{model}_Kn{kn:g}_RT{str(rt).replace('.', 'p')}"
    np.savez_compressed(
        output / f"{tag}_RAW_UNFILTERED_three_seed_ensemble.npz",
        x=x,
        y=y,
        seeds=np.asarray(seeds, dtype=np.int64),
        ux_seeds=arrays["ux"],
        uy_seeds=arrays["uy"],
        T_seeds=arrays["T"],
        rho_seeds=arrays["rho"],
        ux_mean=mean["ux"],
        uy_mean=mean["uy"],
        T_mean=mean["T"],
        rho_mean=mean["rho"],
        ux_sample_sd=sd["ux"],
        uy_sample_sd=sd["uy"],
        T_sample_sd=sd["T"],
        rho_sample_sd=sd["rho"],
        ux_standard_error=se["ux"],
        uy_standard_error=se["uy"],
        T_standard_error=se["T"],
        rho_standard_error=se["rho"],
        kinetic_energy_per_seed=ek_seed,
        kinetic_energy_from_raw_mean_velocity=np.float64(ek_from_mean),
        kinetic_energy_cross_seed_noise_unbiased=np.float64(ek_cross_seed),
        kinetic_energy_cross_seed_pair_values=ek_cross_pairs,
        kinetic_energy_cross_seed_pair_min=np.float64(ek_pair_min),
        kinetic_energy_cross_seed_pair_max=np.float64(ek_pair_max),
        kinetic_energy_cross_seed_pair_range_relative=np.float64(
            ek_pair_range_relative
        ),
        Kn_paper=np.float64(kn),
        RT=np.float64(rt),
    )
    write_dat(
        output / f"{tag}_RAW_UNFILTERED_three_seed_mean.dat",
        x, y, mean, sd, se,
    )
    write_profile(
        output / f"{tag}_profile_y0p25.csv", x, y, arrays, seeds
    )
    if make_plots:
        plot_fields(
            output / f"{tag}_RAW_UNFILTERED_diagnostic.png",
            x, y, mean, f"{model}, Kn={kn:g}, Tc/Th={rt:g}",
        )

    summary = {
        "model": model,
        "Kn_paper": kn,
        "RT": rt,
        "seeds": seeds,
        "number_of_independent_runs": 3,
        "particles_per_run": runs[0]["metrics"]["particles"],
        "steps_per_run": runs[0]["metrics"]["steps"],
        "samples_per_run": runs[0]["metrics"]["profile_samples"],
        "kinetic_energy_per_seed": ek_seed.tolist(),
        "kinetic_energy_seed_mean": float(np.mean(ek_seed)),
        "kinetic_energy_seed_sample_sd": ek_sd,
        "kinetic_energy_seed_standard_error": ek_se,
        "kinetic_energy_seed_mean_95pct_t_CI": [
            float(np.mean(ek_seed) - T95_DF2 * ek_se),
            float(np.mean(ek_seed) + T95_DF2 * ek_se),
        ],
        "kinetic_energy_from_raw_mean_velocity": ek_from_mean,
        "kinetic_energy_cross_seed_noise_unbiased": ek_cross_seed,
        "kinetic_energy_cross_seed_pair_values": ek_cross_pairs.tolist(),
        "kinetic_energy_cross_seed_pair_min": ek_pair_min,
        "kinetic_energy_cross_seed_pair_max": ek_pair_max,
        "kinetic_energy_cross_seed_pair_range_relative": (
            ek_pair_range_relative
        ),
        "kinetic_energy_cross_seed_estimator": (
            "mean[rho_3*u_1.u_2, rho_2*u_1.u_3, rho_1*u_2.u_3]"
        ),
        "max_speed_raw_mean": float(
            np.hypot(mean["ux"], mean["uy"]).max()
        ),
        "velocity_mean_rms": velocity_mean_rms,
        "velocity_standard_error_rms": speed_se_rms,
        "velocity_signal_to_standard_error_ratio": (
            velocity_mean_rms / speed_se_rms if speed_se_rms else None
        ),
        "kinetic_energy_quality_status": (
            "NOISE_LIMITED"
            if ek_cross_seed <= 0.0 or ek_pair_min <= 0.0
            else "GOOD"
            if ek_pair_range_relative <= 0.10
            else "USABLE_WITH_UNCERTAINTY"
            if ek_pair_range_relative <= 0.25
            else "PROVISIONAL"
        ),
        "temperature_mean_min": float(mean["T"].min()),
        "temperature_mean_max": float(mean["T"].max()),
        "density_mean_min": float(mean["rho"].min()),
        "density_mean_max": float(mean["rho"].max()),
        "parity_diagnostics": parity_diagnostics(
            mean["ux"], mean["uy"], mean["T"]
        ),
        "solver_versions": sorted(
            {entry["metrics"]["solver_version"] for entry in runs}
        ),
        "last_block_velocity_rmse_vs_all_samples": [
            entry["metrics"]["last_block_velocity_rmse_vs_all_samples"]
            for entry in runs
        ],
        "quantitative_fields_are_unfiltered": True,
        "spatial_smoothing_applied": False,
        "velocity_projection_applied": False,
    }
    (output / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_master(output: Path, summaries: list[dict]) -> None:
    summaries.sort(key=lambda item: (item["RT"], item["model"], item["Kn_paper"]))
    (output / "ALL_ENSEMBLES_manifest.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "model", "Kn_paper", "RT", "number_of_independent_runs",
        "particles_per_run", "steps_per_run",
        "kinetic_energy_seed_mean", "kinetic_energy_seed_sample_sd",
        "kinetic_energy_seed_standard_error",
        "kinetic_energy_from_raw_mean_velocity",
        "kinetic_energy_cross_seed_noise_unbiased",
        "kinetic_energy_cross_seed_pair_min",
        "kinetic_energy_cross_seed_pair_max",
        "kinetic_energy_cross_seed_pair_range_relative",
        "kinetic_energy_quality_status",
        "max_speed_raw_mean", "velocity_mean_rms",
        "velocity_standard_error_rms",
        "velocity_signal_to_standard_error_ratio",
    )
    with (output / "ALL_ENSEMBLES_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})

    rt05 = [item for item in summaries if item["RT"] == 0.5]
    if {item["model"] for item in rt05} == {"HS", "BGK", "SHAKHOV"}:
        fig, axis = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
        for model, marker in (("HS", "o"), ("BGK", "s"), ("SHAKHOV", "^")):
            rows = sorted(
                (item for item in rt05 if item["model"] == model),
                key=lambda item: item["Kn_paper"],
            )
            x = np.array([item["Kn_paper"] for item in rows])
            y = np.array(
                [
                    item["kinetic_energy_cross_seed_noise_unbiased"]
                    for item in rows
                ]
            )
            axis.loglog(x, y, marker=marker, label=model)
        axis.set_xlabel("Kn (paper definition)")
        axis.set_ylabel(r"$E_k$ from fully independent cross-seed estimator")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        fig.savefig(output / "Ek_RT0p5_RAW_UNFILTERED_three_seed.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = find_runs(args.input)
    actual = {
        (model, kn, rt, entry["seed"])
        for (model, kn, rt), runs in groups.items()
        for entry in runs
    }
    expected = load_expected(args.case_table)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected) if expected else []
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Missing {len(missing)} expected runs; first: {missing[:6]}")
    if unexpected:
        print(f"NOTE: {len(unexpected)} extra runs will also be summarized")

    summaries: list[dict] = []
    for key in sorted(groups, key=lambda item: (item[2], item[0], item[1])):
        try:
            summaries.append(
                summarize_group(
                    key, groups[key], args.output, not args.no_plots
                )
            )
            print(f"[OK] summarized {key}")
        except ValueError as error:
            if args.allow_incomplete:
                print(f"[SKIP] {error}")
            else:
                raise
    write_master(args.output, summaries)
    print(f"[OK] {len(summaries)} three-seed ensembles in {args.output}")


if __name__ == "__main__":
    main()
